import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48, aux_max_len: int = 200, state_as_loc_tokens: bool = False):
        self._max_len = max_len
        self._aux_max_len = aux_max_len
        self._state_as_loc_tokens = state_as_loc_tokens

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            discretized_state = np.clip(discretized_state, 0, 255)
            if self._state_as_loc_tokens:
                state_str = " ".join(f"<loc{value:04d}>" for value in discretized_state)
            else:
                state_str = " ".join(map(str, discretized_state))
            # full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompt = f"Task: {cleaned_text}\nState: {state_str}\n"
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            # full_prompt = cleaned_text
            full_prompt = f"Task: {cleaned_text}\n"
            tokens = self._tokenizer.encode(full_prompt, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)

    def tokenize_auxiliary(
        self, auxiliary_targets: dict[str, object], *, fast_action_tokens: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        lines = []
        for key, title in (("grounding", "Grounding"), ("subtask", "Subtask"), ("focus", "Focus"), ("phase", "Phase")):
            value = auxiliary_targets.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                value = value.item() if np.asarray(value).ndim == 0 else str(value)
            value = str(value).strip()
            if not value:
                continue

            lines.append(f"{title}: {value}")

        tokens = self._tokenizer.encode("\n".join(lines))
        if fast_action_tokens is not None:
            separator = "\n" if tokens else ""
            tokens.extend(self._tokenizer.encode(f"{separator}Action: "))
            tokens.extend(np.asarray(fast_action_tokens).tolist())
        if tokens:
            tokens.extend(self._tokenizer.encode("|", add_eos=True))
        token_mask = [True] * len(tokens)
        if len(tokens) < self._aux_max_len:
            padding = [False] * (self._aux_max_len - len(tokens))
            tokens.extend([0] * len(padding))
            token_mask.extend(padding)
        else:
            if len(tokens) > self._aux_max_len:
                logging.warning(
                    f"Auxiliary token length ({len(tokens)}) exceeds max length ({self._aux_max_len}), truncating. "
                    "Consider increasing `aux_max_len` in `PaligemmaTokenizer` if this happens frequently."
                )
            tokens = tokens[: self._aux_max_len]
            token_mask = token_mask[: self._aux_max_len]

        return np.asarray(tokens), np.asarray(token_mask)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens_in_pg = self.tokenize_actions(actions)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def tokenize_actions(self, actions: np.ndarray) -> np.ndarray:
        action_tokens = self._fast_tokenizer(actions[None])[0]
        return self._act_tokens_to_paligemma_tokens(action_tokens)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FASTThinkingTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = "physical-intelligence/fast",
        state_as_loc_tokens: bool = False,
    ):
        self._max_len = max_len
        self._state_as_loc_tokens = state_as_loc_tokens

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer. FAST action ids are mapped below the image/location tokens.
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_token_start = self._paligemma_tokenizer.piece_to_id("<start_of_image>") - 1

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None,
        *,
        grounding: str | None = None,
        subtask: str | None = None,
        focus: str | None = None,
        phase: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = self._as_text(prompt).lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1]).
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        discretized_state = np.clip(discretized_state, 0, 255)
        if self._state_as_loc_tokens:
            state_str = " ".join(f"<loc{value:04d}>" for value in discretized_state)
        else:
            state_str = " ".join(map(str, discretized_state))

        prefix = f"Task: {cleaned_text}\nState: {state_str}\n<unused0>"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            action_tokens_in_pg = self.tokenize_actions(actions)
            grounding_text = self._as_text(grounding)
            subtask_text = self._as_text(subtask)
            focus_text = self._as_text(focus)
            phase_text = self._as_text(phase)
            thinking_text = (
                f"Grounding: {grounding_text}\n"
                f"Subtask: {subtask_text}\n"
                f"Focus: {focus_text}\n"
                f"Phase: {phase_text}\n"
                "Action: <unused1>"
            )
            thinking_tokens = self._paligemma_tokenizer.encode(thinking_text)
            postfix_tokens = thinking_tokens + action_tokens_in_pg.tolist() + self._paligemma_tokenizer.encode("|", add_eos=True)
        else:
            postfix_tokens = []

        # Prefix is bidirectional context. Postfix is teacher-forced autoregressive target and receives loss.
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        if len(tokens) < self._max_len:
            padding = [False] * (self._max_len - len(tokens))
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def tokenize_actions(self, actions: np.ndarray) -> np.ndarray:
        action_tokens = self._fast_tokenizer(actions[None])[0]
        return self._act_tokens_to_paligemma_tokens(action_tokens)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        start_pattern = self._paligemma_tokenizer.encode("<unused1>")
        end_pattern = self._paligemma_tokenizer.encode("|", add_eos=False)
        action_start = self._find_subsequence(tokens, start_pattern)
        action_end = self._find_subsequence(tokens, end_pattern)
        if action_start < 0 or action_end < 0 or action_end <= action_start:
            raise ValueError("Could not find a valid <unused1> ... | action span in generated tokens.")

        action_start += len(start_pattern)
        action_tokens = self._paligemma_tokens_to_act_tokens(tokens[action_start:action_end])
        try:
            return self._fast_tokenizer.decode(
                [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
            )[0]
        except Exception:
            logging.exception("Failed to decode FAST thinking action tokens.")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

    def extract_thinking(self, tokens: np.ndarray) -> str:
        tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        start_pattern = self._paligemma_tokenizer.encode("<unused0>")
        end_pattern = self._paligemma_tokenizer.encode("<unused1>")
        thinking_start = self._find_subsequence(tokens, start_pattern)
        thinking_end = self._find_subsequence(tokens, end_pattern)
        if thinking_start < 0 or thinking_end < 0 or thinking_end <= thinking_start:
            raise ValueError("Could not find a valid <unused0> ... <unused1> thinking span in generated tokens.")

        thinking_start += len(start_pattern)
        thinking_tokens = tokens[thinking_start:thinking_end]
        thinking_tokens = thinking_tokens[thinking_tokens != self._paligemma_tokenizer.pad_id()]
        if thinking_tokens.size == 0:
            return ""
        return self._paligemma_tokenizer.decode(thinking_tokens.tolist()).strip()

    @staticmethod
    def _find_subsequence(tokens: np.ndarray, pattern: list[int]) -> int:
        if not pattern or len(tokens) < len(pattern):
            return -1
        pattern_array = np.asarray(pattern, dtype=tokens.dtype)
        for idx in range(len(tokens) - len(pattern_array) + 1):
            if np.array_equal(tokens[idx : idx + len(pattern_array)], pattern_array):
                return idx
        return -1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._fast_token_start - tokens

    def _paligemma_tokens_to_act_tokens(self, tokens: np.ndarray) -> np.ndarray:
        vocab_size = self._get_fast_vocab_size()
        min_token = self._fast_token_start - vocab_size + 1
        action_tokens_in_pg = tokens[(tokens >= min_token) & (tokens <= self._fast_token_start)]
        return self._fast_token_start - action_tokens_in_pg

    def _get_fast_vocab_size(self) -> int:
        vocab_size = getattr(self._fast_tokenizer, "vocab_size", None)
        if callable(vocab_size):
            return int(vocab_size())
        if vocab_size is not None:
            return int(vocab_size)
        raise AttributeError("FAST tokenizer does not expose vocab_size")

    @staticmethod
    def _as_text(value: object | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        if hasattr(value, "item"):
            return FASTThinkingTokenizer._as_text(value.item())
        return str(value)


class ARThinkingTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        state_as_loc_tokens: bool = False,
        num_action_bins: int = 256,
    ):
        self._max_len = max_len
        self._state_as_loc_tokens = state_as_loc_tokens

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # AR action bins are mapped below the image/location tokens.
        self._num_action_bins = num_action_bins
        self._action_token_start = self._paligemma_tokenizer.piece_to_id("<start_of_image>") - 1

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None,
        *,
        grounding: str | None = None,
        subtask: str | None = None,
        focus: str | None = None,
        phase: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = self._as_text(prompt).lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1]).
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        discretized_state = np.clip(discretized_state, 0, 255)
        if self._state_as_loc_tokens:
            state_str = " ".join(f"<loc{value:04d}>" for value in discretized_state)
        else:
            state_str = " ".join(map(str, discretized_state))

        prefix = f"Task: {cleaned_text}\nState: {state_str}\n<unused0>"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            action_tokens_in_pg = self.tokenize_actions(actions)
            grounding_text = self._as_text(grounding)
            subtask_text = self._as_text(subtask)
            focus_text = self._as_text(focus)
            phase_text = self._as_text(phase)
            thinking_text = (
                f"Grounding: {grounding_text}\n"
                f"Subtask: {subtask_text}\n"
                f"Focus: {focus_text}\n"
                f"Phase: {phase_text}\n"
                "Action: <unused1>"
            )
            thinking_tokens = self._paligemma_tokenizer.encode(thinking_text)
            postfix_tokens = thinking_tokens + action_tokens_in_pg.tolist() + self._paligemma_tokenizer.encode("|", add_eos=True)
        else:
            postfix_tokens = []

        # Prefix is bidirectional context. Postfix is teacher-forced autoregressive target and receives loss.
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        if len(tokens) < self._max_len:
            padding = [False] * (self._max_len - len(tokens))
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def tokenize_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, -1.0, 1.0)
        action_bins = np.rint((actions + 1.0) * 0.5 * (self._num_action_bins - 1)).astype(np.int32)
        action_bins = np.clip(action_bins, 0, self._num_action_bins - 1).reshape(-1)
        return self._act_tokens_to_paligemma_tokens(action_bins)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        start_pattern = self._paligemma_tokenizer.encode("<unused1>")
        end_pattern = self._paligemma_tokenizer.encode("|", add_eos=False)
        action_start = self._find_subsequence(tokens, start_pattern)
        action_end = self._find_subsequence(tokens, end_pattern)
        if action_start < 0 or action_end < 0 or action_end <= action_start:
            raise ValueError("Could not find a valid <unused1> ... | action span in generated tokens.")

        action_start += len(start_pattern)
        action_bins = self._paligemma_tokens_to_act_tokens(tokens[action_start:action_end])
        expected_tokens = action_horizon * action_dim
        if len(action_bins) != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} AR action tokens, got {len(action_bins)}.")
        actions = action_bins.astype(np.float32) / (self._num_action_bins - 1) * 2.0 - 1.0
        return actions.reshape(action_horizon, action_dim)

    def extract_thinking(self, tokens: np.ndarray) -> str:
        tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        start_pattern = self._paligemma_tokenizer.encode("<unused0>")
        end_pattern = self._paligemma_tokenizer.encode("<unused1>")
        thinking_start = self._find_subsequence(tokens, start_pattern)
        thinking_end = self._find_subsequence(tokens, end_pattern)
        if thinking_start < 0 or thinking_end < 0 or thinking_end <= thinking_start:
            raise ValueError("Could not find a valid <unused0> ... <unused1> thinking span in generated tokens.")

        thinking_start += len(start_pattern)
        thinking_tokens = tokens[thinking_start:thinking_end]
        thinking_tokens = thinking_tokens[thinking_tokens != self._paligemma_tokenizer.pad_id()]
        if thinking_tokens.size == 0:
            return ""
        return self._paligemma_tokenizer.decode(thinking_tokens.tolist()).strip()

    @staticmethod
    def _find_subsequence(tokens: np.ndarray, pattern: list[int]) -> int:
        if not pattern or len(tokens) < len(pattern):
            return -1
        pattern_array = np.asarray(pattern, dtype=tokens.dtype)
        for idx in range(len(tokens) - len(pattern_array) + 1):
            if np.array_equal(tokens[idx : idx + len(pattern_array)], pattern_array):
                return idx
        return -1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._action_token_start - tokens

    def _paligemma_tokens_to_act_tokens(self, tokens: np.ndarray) -> np.ndarray:
        min_token = self._action_token_start - self._num_action_bins + 1
        action_tokens_in_pg = tokens[(tokens >= min_token) & (tokens <= self._action_token_start)]
        return self._action_token_start - action_tokens_in_pg

    @staticmethod
    def _as_text(value: object | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        if hasattr(value, "item"):
            return ARThinkingTokenizer._as_text(value.item())
        return str(value)


class OFTThinkingTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        state_as_loc_tokens: bool = False,
        action_query_token_id: int = 244502,  # 🔍
        action_query_token: str | None = None,
    ):
        self._max_len = max_len
        self._state_as_loc_tokens = state_as_loc_tokens
        self.action_query_token_id = action_query_token_id

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        if action_query_token is not None:
            action_query_token_ids = self._paligemma_tokenizer.encode(action_query_token)
            if len(action_query_token_ids) != 1:
                raise ValueError(f"OFT action query token must encode to one token, got {action_query_token_ids}.")
            if action_query_token_ids[0] != action_query_token_id:
                raise ValueError(
                    f"OFT action query token id mismatch: {action_query_token!r} encodes to "
                    f"{action_query_token_ids[0]}, but action_query_token_id={action_query_token_id}."
                )

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None,
        *,
        grounding: str | None = None,
        subtask: str | None = None,
        focus: str | None = None,
        phase: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = self._as_text(prompt).lower().strip().replace("_", " ")

        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        discretized_state = np.clip(discretized_state, 0, 255)
        if self._state_as_loc_tokens:
            state_str = " ".join(f"<loc{value:04d}>" for value in discretized_state)
        else:
            state_str = " ".join(map(str, discretized_state))

        prefix = f"Task: {cleaned_text}\nState: {state_str}\n<unused0>"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            action_horizon = int(np.asarray(actions).shape[0])
            grounding_text = self._as_text(grounding)
            subtask_text = self._as_text(subtask)
            focus_text = self._as_text(focus)
            phase_text = self._as_text(phase)
            thinking_text = (
                f"Grounding: {grounding_text}\n"
                f"Subtask: {subtask_text}\n"
                f"Focus: {focus_text}\n"
                f"Phase: {phase_text}\n"
                "Action: <unused1>"
            )
            thinking_tokens = self._paligemma_tokenizer.encode(thinking_text)
            action_query_tokens = [self.action_query_token_id] * action_horizon
            end_tokens = self._paligemma_tokenizer.encode("|", add_eos=True)
            postfix_tokens = thinking_tokens + action_query_tokens + end_tokens
            postfix_loss_mask = [True] * len(thinking_tokens) + [False] * (len(action_query_tokens) + len(end_tokens))
        else:
            postfix_tokens = []
            postfix_loss_mask = []

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + postfix_loss_mask

        if len(tokens) < self._max_len:
            padding = [False] * (self._max_len - len(tokens))
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_thinking(self, tokens: np.ndarray) -> str:
        tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        start_pattern = self._paligemma_tokenizer.encode("<unused0>")
        end_pattern = self._paligemma_tokenizer.encode("<unused1>")
        thinking_start = self._find_subsequence(tokens, start_pattern)
        thinking_end = self._find_subsequence(tokens, end_pattern)
        if thinking_start < 0 or thinking_end < 0 or thinking_end <= thinking_start:
            raise ValueError("Could not find a valid <unused0> ... <unused1> thinking span in generated tokens.")

        thinking_start += len(start_pattern)
        thinking_tokens = tokens[thinking_start:thinking_end]
        thinking_tokens = thinking_tokens[thinking_tokens != self._paligemma_tokenizer.pad_id()]
        if thinking_tokens.size == 0:
            return ""
        return self._paligemma_tokenizer.decode(thinking_tokens.tolist()).strip()

    @staticmethod
    def _find_subsequence(tokens: np.ndarray, pattern: list[int]) -> int:
        if not pattern or len(tokens) < len(pattern):
            return -1
        pattern_array = np.asarray(pattern, dtype=tokens.dtype)
        for idx in range(len(tokens) - len(pattern_array) + 1):
            if np.array_equal(tokens[idx : idx + len(pattern_array)], pattern_array):
                return idx
        return -1

    @staticmethod
    def _as_text(value: object | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        if hasattr(value, "item"):
            return OFTThinkingTokenizer._as_text(value.item())
        return str(value)


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

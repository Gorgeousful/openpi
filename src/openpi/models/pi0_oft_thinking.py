import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
import openpi.models.tokenizer as _tokenizer
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1
PALIGEMMA_UNUSED1_TOKEN = 8
PALIGEMMA_DEFAULT_ACTION_QUERY_TOKEN = 244502  # 🔍
PALIGEMMA_ACTION_END_TOKEN = 235371


def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0OFTThinkingConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250
    state_as_loc_tokens: bool = False
    oft_thinking_loss_weight: float = 1.0
    action_mlp_hidden_dim: int | None = None
    action_mlp_num_blocks: int = 2
    action_query_token_id: int = PALIGEMMA_DEFAULT_ACTION_QUERY_TOKEN

    oft_model_tokenizer: Any | None = _tokenizer.OFTThinkingTokenizer
    oft_model_tokenizer_kwargs: dict[str, Any] | None = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_OFT_THINKING

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0OFTThinking":
        return Pi0OFTThinking(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class MLPResNetBlock(nnx.Module):
    def __init__(self, dim: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.linear = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b d"]:
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = jax.nn.relu(x)
        return x + residual


class MLPResNetActionHead(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int, rngs: nnx.Rngs):
        self.num_blocks = num_blocks
        self.norm_in = nnx.LayerNorm(input_dim, rngs=rngs)
        self.fc_in = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        for i in range(num_blocks):
            setattr(self, f"block_{i}", MLPResNetBlock(hidden_dim, rngs=rngs))
        self.norm_out = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.fc_out = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b ad"]:
        x = self.norm_in(x)
        x = self.fc_in(x)
        x = jax.nn.relu(x)
        for i in range(self.num_blocks):
            x = getattr(self, f"block_{i}")(x)
        x = self.norm_out(x)
        return self.fc_out(x)


class Pi0OFTThinking(_model.BaseModel):
    def __init__(self, config: Pi0OFTThinkingConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.oft_thinking_loss_weight = config.oft_thinking_loss_weight
        self.action_query_token = config.action_query_token_id

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        hidden_dim = paligemma_config.width
        mlp_hidden_dim = config.action_mlp_hidden_dim or hidden_dim * 2
        self.action_head = MLPResNetActionHead(
            input_dim=hidden_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=config.action_dim,
            num_blocks=config.action_mlp_num_blocks,
            rngs=rngs,
        )

    @at.typecheck
    def embed_inputs(
        self,
        obs: _model.Observation,
        tokens: at.Int[at.Array, "b token_s"] | None = None,
        token_mask: at.Bool[at.Array, "b token_s"] | None = None,
        token_ar_mask: at.Int[at.Array, "b token_s"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b total_s emb"],
        at.Bool[at.Array, "b total_s"],
        at.Int[at.Array, "b total_s"],
    ]:
        input_mask = []
        ar_mask = []
        token_embeddings = []

        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)
            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            ar_mask.append(0 * input_mask[-1])

        if tokens is None:
            assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
            assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
            assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
            tokens = obs.tokenized_prompt
            token_mask = obs.tokenized_prompt_mask
            token_ar_mask = obs.token_ar_mask
        assert token_mask is not None, "Token mask is required"
        assert token_ar_mask is not None, "Token auto-regressive mask is required"

        tokenized_inputs_embeddings = self.PaliGemma.llm(tokens, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(token_mask)
        ar_mask.append(token_ar_mask)

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    def _action_head(self, action_query_hidden: at.Float[at.Array, "b ah emb"]) -> at.Float[at.Array, "b ah ad"]:
        return self.action_head(action_query_hidden)

    def _gather_action_query_hidden(
        self,
        token_hidden: at.Float[at.Array, "b s emb"],
        token_ids: at.Int[at.Array, "b s"],
    ) -> at.Float[at.Array, "b ah emb"]:
        batch_size, seq_len, hidden_dim = token_hidden.shape
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        query_positions = jnp.where(token_ids == self.action_query_token, positions, -1)
        selected_positions, _ = jax.lax.top_k(query_positions, self.action_horizon)
        selected_positions = jnp.sort(selected_positions, axis=-1)
        selected_positions = jnp.maximum(selected_positions, 0)
        gather_index = selected_positions[..., None]
        gather_index = jnp.broadcast_to(gather_index, (batch_size, self.action_horizon, hidden_dim))
        return jnp.take_along_axis(token_hidden, gather_index, axis=1)

    def _predict_actions_from_tokens(
        self,
        observation: _model.Observation,
        tokens: at.Int[at.Array, "b s"],
        token_mask: at.Bool[at.Array, "b s"],
        token_ar_mask: at.Int[at.Array, "b s"],
    ) -> at.Float[at.Array, "b ah ad"]:
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(
            observation, tokens=tokens, token_mask=token_mask, token_ar_mask=token_ar_mask
        )
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=-1) - 1
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings,
            mask=attn_mask,
            positions=positions,
            return_prelogits=True,
        )
        token_hidden = pre_logits[:, -tokens.shape[1] :]
        action_query_hidden = self._gather_action_query_hidden(token_hidden, tokens)
        return self._action_head(action_query_hidden)

    def _compute_losses(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "b"], at.Float[at.Array, "b"], at.Float[at.Array, "b"]]:
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        assert observation.tokenized_prompt is not None, "Tokenized prompt is required"
        assert observation.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert observation.token_ar_mask is not None, "Token auto-regressive mask is required"
        assert observation.token_loss_mask is not None, "Token loss mask is required"

        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=-1) - 1

        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )

        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            positions=positions[:, :-1],
            return_prelogits=True,
        )
        logits, _ = self.PaliGemma.llm(pre_logits=pre_logits[:, -targets.shape[1] :])
        logp = jax.nn.log_softmax(logits, axis=-1)

        loss_mask = observation.token_loss_mask[:, 1:].astype(jnp.float32)
        token_nll = -jnp.sum(targets * logp, axis=-1)
        thinking_loss = jnp.sum(token_nll * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

        token_hidden = pre_logits[:, -(observation.tokenized_prompt.shape[1] - 1) :]
        token_ids_for_hidden = observation.tokenized_prompt[:, :-1]
        action_query_hidden = self._gather_action_query_hidden(token_hidden, token_ids_for_hidden)
        pred_actions = self._action_head(action_query_hidden)
        oft_action_loss = jnp.mean(jnp.abs(pred_actions - actions), axis=(-1, -2))
        total_loss = oft_action_loss + self.oft_thinking_loss_weight * thinking_loss
        return total_loss, thinking_loss, oft_action_loss

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "b"]:
        total_loss, _, _ = self._compute_losses(rng, observation, actions, train=train)
        return total_loss

    @override
    def compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        total_loss, thinking_loss, oft_action_loss = self._compute_losses(rng, observation, actions, train=train)
        return total_loss, {
            "action_loss": jnp.mean(oft_action_loss),
            "auxiliary_loss": jnp.mean(thinking_loss),
            "weighted_auxiliary_loss": self.oft_thinking_loss_weight * jnp.mean(thinking_loss),
        }

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> dict[str, at.Array]:
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)
        output_mask = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.bool_)

        def step(carry):
            rng, last_logit, output_tokens, output_mask, cache, done, step = carry
            rng, rng_step = jax.random.split(rng)
            sampled_token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            token = jnp.where(done[:, None], PALIGEMMA_EOS_TOKEN, sampled_token)
            valid_token = ~done
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)
            output_mask = put_along_last_axis(
                output_mask, jnp.broadcast_to(step, (token.shape[0], 1)), valid_token[:, None]
            )

            has_stop = jnp.any((token == PALIGEMMA_EOS_TOKEN) | (token == PALIGEMMA_UNUSED1_TOKEN), axis=-1)
            done = done | has_stop

            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return rng, last_logit, output_tokens, output_mask, kv_cache, done, step + 1

        def cond(carry):
            _, _, _, _, _, done, step = carry
            return (~jnp.all(done)) & (step < max_decoding_steps)

        batch_size = last_logit.shape[0]
        _, _, output_tokens, output_mask, _, _, _ = jax.lax.while_loop(
            cond,
            step,
            (
                rng,
                last_logit,
                output_tokens,
                output_mask,
                kv_cache,
                jnp.zeros(batch_size, dtype=jnp.bool_),
                0,
            ),
        )

        query_tokens = jnp.full((batch_size, self.action_horizon), self.action_query_token, dtype=jnp.int32)
        end_tokens = jnp.full((batch_size, 1), PALIGEMMA_ACTION_END_TOKEN, dtype=jnp.int32)
        query_mask = jnp.ones((batch_size, self.action_horizon + 1), dtype=jnp.bool_)
        query_ar_mask = jnp.ones((batch_size, self.action_horizon + 1), dtype=jnp.int32)
        full_tokens = jnp.concatenate([observation.tokenized_prompt, output_tokens, query_tokens, end_tokens], axis=1)
        full_mask = jnp.concatenate([observation.tokenized_prompt_mask, output_mask, query_mask], axis=1)
        full_ar_mask = jnp.concatenate([observation.token_ar_mask, output_mask.astype(jnp.int32), query_ar_mask], axis=1)

        valid_output_tokens = jnp.where(output_mask, output_tokens, -1)
        generated_unused1 = jnp.any(valid_output_tokens == PALIGEMMA_UNUSED1_TOKEN, axis=-1)
        generated_eos = jnp.any(valid_output_tokens == PALIGEMMA_EOS_TOKEN, axis=-1)
        stopped_by_eos_without_unused1 = (~generated_unused1) & generated_eos
        reached_max_without_unused1 = (~generated_unused1) & (~generated_eos)

        actions = self._predict_actions_from_tokens(observation, full_tokens, full_mask, full_ar_mask)
        return {
            "actions": actions,
            "tokens": jnp.concatenate([observation.tokenized_prompt, output_tokens], axis=1),
            "oft_stopped_by_eos_without_unused1": stopped_by_eos_without_unused1,
            "oft_reached_max_without_unused1": reached_max_without_unused1,
        }

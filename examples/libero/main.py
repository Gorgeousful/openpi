import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import rich
from rich.console import Console
cs = Console()

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    normalize_gripper: bool = False
    invert_gripper: bool = False

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    result_out_path: str = "data/libero/results.json"  # Path to save evaluation metrics

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    cs.print(args, markup=False)
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.result_out_path).parent.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_completion = 0.0
    task_results = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_completion = 0.0
        total_goals = len(_get_goal_states(env))
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    action = _prepare_gripper_action(
                        action,
                        normalize_gripper=args.normalize_gripper,
                        invert_gripper=args.invert_gripper,
                    )

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            completed_goals, total_goals = _count_completed_goals(env)
            completion = float(completed_goals) / float(total_goals) if total_goals else 0.0
            task_completion += completion
            total_completion += completion

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"task_{task_id:03d}_ep_{episode_idx:03d}_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            current_task_sr = float(task_successes) / float(task_episodes)
            current_task_cr = float(task_completion) / float(task_episodes)
            current_total_sr = float(total_successes) / float(total_episodes)
            current_total_cr = float(total_completion) / float(total_episodes)
            logging.info(f"Success: {done}")
            logging.info(f"Completed goals: {completed_goals}/{total_goals}")
            logging.info(f"Current Task SR: {current_task_sr * 100:.1f}%")
            logging.info(f"Current Task CR: {current_task_cr * 100:.1f}%")
            logging.info(f"Total SR: {current_total_sr * 100:.1f}%")
            logging.info(f"Total CR: {current_total_cr * 100:.1f}%")

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes)
        task_completion_rate = float(task_completion) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        total_completion_rate = float(total_completion) / float(total_episodes)
        task_results.append(
            {
                "task_id": task_id,
                "task_desc": task_description,
                "total_goals": total_goals,
                "success_rate": task_success_rate,
                "completion_rate": task_completion_rate,
                "num_episodes": task_episodes,
            }
        )
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current task completion rate: {task_completion_rate}")
        logging.info(f"Current total success rate: {total_success_rate}")
        logging.info(f"Current total completion rate: {total_completion_rate}")

    success_rate = float(total_successes) / float(total_episodes)
    completion_rate = float(total_completion) / float(total_episodes)
    result = {
        "task_suite": args.task_suite_name,
        "success_rate": success_rate,
        "completion_rate": completion_rate,
        "total_episodes": total_episodes,
        "tasks": task_results,
    }
    with open(args.result_out_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Total success rate: {success_rate}")
    logging.info(f"Total completion rate: {completion_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Saved results to: {args.result_out_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _get_core_env(env):
    return getattr(env, "env", env)


def _get_goal_states(env):
    return _get_core_env(env).parsed_problem["goal_state"]


def _count_completed_goals(env) -> tuple[int, int]:
    core_env = _get_core_env(env)
    goal_state = _get_goal_states(env)
    completed_goals = sum(bool(core_env._eval_predicate(state)) for state in goal_state)
    return completed_goals, len(goal_state)


def _prepare_gripper_action(action, normalize_gripper: bool, invert_gripper: bool) -> np.ndarray:
    action = np.asarray(action).copy()

    if normalize_gripper:
        action[6] = action[6] * 2.0 - 1.0

    if invert_gripper:
        action[6] = -action[6]

    action[6] = np.clip(action[6], -1.0, 1.0)
    return action


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)

"""Script to play a checkpoint if an RL agent from RSL-RL with joystick control."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and joystick override.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch
from rsl_rl.runners import OnPolicyRunner
from omegaconf import OmegaConf
import isaaclab.utils.string as string_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
import berkeley_humanoid_lite.tasks  # noqa: F401
#from isaaclab.markers import VisualizationMarkers

def get_joystick_cmd():
    try:
        with open('/tmp/joystick_cmd.txt', 'r') as file:
            parts = file.readline().strip().split()
            if len(parts) >= 3:
                x, y, z = map(float, parts[:3])
                return [x, y, z]
    except Exception as e:
        print(f"⚠️ Could not read joystick command: {e}")
    return [0.0, 0.0, 0.0]


def main():
    """Play with RSL-RL agent and joystick control."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)



       
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    #print(f"✅ Initial obs['velocity_commands']: type={type(obs['velocity_commands'])}, shape={obs['velocity_commands'].shape}")


    while simulation_app.is_running():
        # Read joystick
        joystick_cmd = get_joystick_cmd()
        print(f"🎮 Joystick Command: {joystick_cmd}")

        # Convert to tensor
        joystick_tensor = torch.tensor(joystick_cmd, device=env.unwrapped.device).unsqueeze(0)

        # Clone obs to make it writable
        obs = obs.clone()

        # Apply override to first 3 columns (velocity_commands)
        print(f"🔍 Before override: obs[:, :3] = {obs[:, :3]}")
        obs[:, :3] = joystick_tensor.repeat(env.num_envs, 1)
        print(f"✅ After override: obs[:, :3] = {obs[:, :3]}")

        # Step
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

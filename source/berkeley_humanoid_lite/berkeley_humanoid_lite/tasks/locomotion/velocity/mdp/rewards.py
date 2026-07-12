
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)



def _moving_command_mask(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.12) -> torch.Tensor:
    """Mask for commands large enough that gait/translation rewards should apply."""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :2], dim=1) > threshold


def moving_velocity_along_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward base velocity projected onto the commanded horizontal direction.

    This is a direct anti-stall term: when the command is clearly nonzero, the
    policy receives reward for actually translating in the commanded direction.
    It does not split commands into bins and works for forward, reverse, lateral,
    and diagonal commands.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.vector_norm(command, dim=1)
    command_dir = command / torch.clamp(command_norm.unsqueeze(-1), min=1.0e-6)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])[:, :2]
    projected_speed = torch.sum(vel_yaw * command_dir, dim=1)
    return torch.clamp(projected_speed, min=0.0) * (command_norm > command_threshold)


def moving_no_progress_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.12,
    min_fraction: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize bracing in place under clear movement commands.

    The penalty is active only when horizontal command speed is above
    ``command_threshold``. It penalizes the deficit between the commanded speed
    fraction and actual velocity along the command direction.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.vector_norm(command, dim=1)
    command_dir = command / torch.clamp(command_norm.unsqueeze(-1), min=1.0e-6)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])[:, :2]
    projected_speed = torch.sum(vel_yaw * command_dir, dim=1)
    required = min_fraction * command_norm
    deficit = torch.relu(required - projected_speed)
    return deficit * (command_norm > command_threshold)


def moving_single_support_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.12,
    max_reward_time_s: float = 0.20,
) -> torch.Tensor:
    """Reward single-support phases under nonzero motion commands.

    This directly counters the observed double-support bracing behavior while
    keeping the reward capped so the policy is not encouraged to balance on one
    foot forever.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_support = torch.sum(in_contact.int(), dim=1) == 1
    # Use the shorter of stance contact time and swing air time as a robust phase-time proxy.
    stance_time = torch.max(torch.where(in_contact, contact_time, torch.zeros_like(contact_time)), dim=1)[0]
    swing_time = torch.max(torch.where(~in_contact, air_time, torch.zeros_like(air_time)), dim=1)[0]
    phase_time = torch.minimum(stance_time, swing_time)
    reward = torch.clamp(phase_time, max=max_reward_time_s)
    return reward * single_support * _moving_command_mask(env, command_name, command_threshold)


def moving_double_support_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.12,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize persistent both-feet bracing under clear motion commands."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    double_support = torch.all(in_contact, dim=1)
    return double_support * _moving_command_mask(env, command_name, command_threshold)


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.12,
    clearance_target_m: float = 0.025,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """Reward lifting an airborne foot above the stance-foot height.

    The reward is active only under clear horizontal motion commands and uses a
    relative height measure so it is less sensitive to absolute terrain/body-frame
    offsets.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_support = torch.sum(in_contact.int(), dim=1) == 1

    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    stance_z = torch.sum(foot_z * in_contact.float(), dim=1) / torch.clamp(
        torch.sum(in_contact.float(), dim=1), min=1.0
    )
    swing_clearance = torch.relu(foot_z - stance_z.unsqueeze(-1))
    swing_clearance = torch.where(~in_contact, swing_clearance, torch.zeros_like(swing_clearance))
    reward = torch.clamp(torch.sum(swing_clearance, dim=1) / clearance_target_m, max=max_reward)
    return reward * single_support * _moving_command_mask(env, command_name, command_threshold)


def swing_foot_velocity_along_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.12,
) -> torch.Tensor:
    """Reward airborne foot motion aligned with the commanded horizontal direction.

    This is a global anti-shuffle term rather than a command-bin curriculum. It
    supports forward, reverse, lateral, and diagonal commands using the same logic.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    airborne = contact_time <= 0.0

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.vector_norm(command, dim=1)
    command_dir = command / torch.clamp(command_norm.unsqueeze(-1), min=1.0e-6)
    foot_vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1),
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :3],
    )[:, :, :2]
    projected = torch.sum(foot_vel_yaw * command_dir.unsqueeze(1), dim=-1)
    reward = torch.sum(torch.clamp(projected, min=0.0) * airborne.float(), dim=1)
    return reward * (command_norm > command_threshold)


#==========================#============================
#
#               ADDITIONAL REWARDS TEST   
#
#==========================#============================


# Reward robot for lifting the swing leg during step 
def swing_leg_lift(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward lifting the thigh on the non-contact leg using current_air_time.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # [N, 2]
    asset = env.scene[asset_cfg.name]
    joint_names = asset.joint_names
    joint_pos = asset.data.joint_pos  # [N, num_joints]
    left_index = joint_names.index("leg_left_hip_pitch_joint")
    right_index = joint_names.index("leg_right_hip_pitch_joint")
    left_thigh_pitch = joint_pos[:, left_index]
    right_thigh_pitch = joint_pos[:, right_index]
    left_air = (air_time[:, 0] > 0.0).float()
    right_air = (air_time[:, 1] > 0.0).float()
    lift_score = (left_thigh_pitch * left_air) + (right_thigh_pitch * right_air)
    return torch.clamp(lift_score, min=0.0)


# Reward robot for lifting the swing foot a certain height during step
def swing_foot_plant(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_hold_threshold: float = 0.1,  # seconds foot should stay planted
    alignment_tolerance: float = 0.1,     # radians for foot placement direction
    max_reward: float = 0.2,
    cmd_scaling_factor: float = 2.0
) -> torch.Tensor:
    """
    Reward the swing foot for planting in the commanded velocity direction and maintaining contact.

    Parameters:
    - contact_hold_threshold: Minimum duration foot must stay planted (s)
    - alignment_tolerance: Tolerance (rad) for foot landing direction alignment
    - max_reward: Cap per foot plant
    - cmd_scaling_factor: Sensitivity to commanded velocity magnitude
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Get air time and contact time for both feet
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]    # [N, 2]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # [N, 2]

    # Mask: foot just transitioned from air to contact and holds for threshold
    planted_mask = (contact_time > contact_hold_threshold).float() * (air_time < 0.05).float()

    # Get foot horizontal landing velocities
    foot_velocities = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]  # [N, 2, xy]

    # Get commanded horizontal velocity direction
    cmd_vel = env.command_manager.get_command("base_velocity")[:, :2]  # [N, 2]
    cmd_vel_mag = torch.norm(cmd_vel, dim=1, keepdim=True)  # [N, 1]
    cmd_dir = cmd_vel / (cmd_vel_mag + 1e-6)  # Normalize to unit vector [N, 2]

    # Foot horizontal velocity direction
    foot_dir = foot_velocities / (foot_velocities.norm(dim=-1, keepdim=True) + 1e-6)  # [N, 2, xy]

    # Dot product to find alignment (-1 = opposite, 1 = aligned)
    alignment = torch.sum(foot_dir * cmd_dir.unsqueeze(1), dim=-1)  # [N, 2]
    alignment_reward = torch.clamp(alignment, min=0.0)  # Reward only positive alignment

    # Combine planted mask and alignment
    plant_reward_per_foot = alignment_reward * planted_mask

    # Command velocity scaling
    cmd_scaling = torch.tanh(cmd_scaling_factor * cmd_vel_mag.squeeze(-1))  # [N]

    # Total reward: sum feet, scale, clamp
    reward = (plant_reward_per_foot.sum(dim=1) * cmd_scaling).clamp(max=max_reward)

    return reward


# Reward robot for for leaning over stance leg 
def lean_over_stance_leg(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward the agent for keeping its pelvis over the stance foot (when in single support phase).
    Uses `current_air_time` to infer contact states.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # [num_envs, 2]
    # A foot is in contact if air_time is near zero
    contact_states = air_time < 0.05  # small threshold in seconds
    # Retrieve pelvis position and foot positions
    base_pos_w = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    foot_positions = env.scene[asset_cfg.name].data.body_pos_w[:, sensor_cfg.body_ids, :2]
    # Compute stance foot midpoint (or average if both in contact)
    contact_mask = contact_states
    mask_sum = torch.sum(contact_mask, dim=1, keepdim=True).clamp(min=1.0)
    weighted_avg = torch.sum(foot_positions * contact_mask.unsqueeze(-1), dim=1) / mask_sum
    # Reward pelvis being near the stance foot (leaning over support leg)
    dist = torch.norm(base_pos_w - weighted_avg, dim=1)
    #sharp curve
    #reward = torch.exp(-5.0 * dist**2)
    #standard weighted average
    reward = 1.0 / (1.0 + dist**2)
    return reward


# Reward robot for maintaining a base height
def maintain_base_height(
    env: ManagerBasedRLEnv,
    desired_height: float = 0.40,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.05
) -> torch.Tensor:
    """
    Reward keeping the robot's pelvis at a desired height.
    Uses a Gaussian kernel centered at `desired_height`.
    """
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]  # height of base in world frame
    #reward = torch.exp(-((current_height - desired_height) ** 2) / (2 * std**2))
    reward = 0.1 + 0.8 * torch.exp(-((current_height - desired_height) ** 2) / (2 * std**2))
    return reward

#=============================
# === Penalty Definitions  ===
#=============================
# predefined l1- penalize ground contact of key bodies
# predefined l1- penalize deviation from default of the joints that are not essential for locomotion
# predefined l1- penalize deviation of ankle roll joints
# predefined l1- penalize moving into a splits position
#==========================
# ===  Reward / Penalty ===
#==========================



# Reward robot for stepping with one foot lifting off (discourage hopping)
def encourage_stepping_not_hopping_refined(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lift_threshold: float = 0.5,
    base_single_leg_reward: float = 0.1,
    hop_penalty_scale: float = -1.0,
    scaling_factor: float = 2.0,
) -> torch.Tensor:
    """
    Encourage stepping with one foot lifting off while the other remains grounded.
    Discourage hopping (both feet airborne), but tolerate it at low commanded velocities.

    Parameters:
    - lift_threshold: Base threshold for upward foot velocity (m/s), scaled by env.step_dt
    - base_single_leg_reward: Reward for single-leg stance without active lift
    - hop_penalty_scale: Base penalty applied when both feet are airborne
    - scaling_factor: Multiplier for command velocity magnitude scaling
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    contact_magnitude = net_forces.norm(dim=-1)

    # Flags: 1 if foot is airborne (low contact force), 0 if in contact
    airborne_flags = (contact_magnitude < 1.0).float()
    num_airborne = airborne_flags.sum(dim=1)  # [num_envs]

    # Get vertical velocity of feet
    asset = env.scene[asset_cfg.name]
    vertical_velocities = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # [num_envs, feet]

    # Dynamically scale upward lift threshold by step_dt
    dynamic_lift_threshold = lift_threshold * env.step_dt
    upward_lift = (vertical_velocities > dynamic_lift_threshold).float()

    # Reward if exactly one foot airborne and moving upward
    lift_reward = (num_airborne == 1).float() * upward_lift.sum(dim=1)

    # Base reward for single-leg stance (even without lift)
    single_leg_reward = (num_airborne == 1).float() * base_single_leg_reward

    # Penalty for hopping (both feet airborne), scaled by commanded velocity
    cmd_vel = env.command_manager.get_command("base_velocity")[:, :2]  # [num_envs, 2]
    cmd_vel_mag = torch.norm(cmd_vel, dim=1)  # [num_envs]
    scaling = torch.tanh(scaling_factor * cmd_vel_mag)  # scaling ∈ [0,1]
    hop_penalty = (num_airborne == 2).float() * (hop_penalty_scale * scaling)

    # Combine rewards and penalties
    total_reward = scaling * (lift_reward + single_leg_reward) + hop_penalty

    return total_reward

def encourage_foot_alternation(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    history_len: int = 5,
    penalty_scale: float = 0.05,
    ignore_double_contact: bool = True,
    scaling_factor: float = 8.0,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """
    Encourage alternating foot contacts over time with rolling history.
    Penalize stalling. Scales reward with commanded velocity magnitude.

    Parameters:
    - history_len: Number of timesteps to consider for alternation
    - penalty_scale: Scale of penalty for no alternation
    - ignore_double_contact: Whether to ignore double-stance transitions
    - scaling_factor: Sensitivity to commanded velocity magnitude
    - max_reward: Cap on the total alternation reward
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_history = (
        contact_sensor.data.net_forces_w_history[:, -history_len:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    ).float()  # [N, T, 2]

    # Encode: 0 = none, 1 = left, 2 = right, 3 = both
    contact_codes = contact_history[..., 0] * 1 + contact_history[..., 1] * 2  # [N, T]
    contact_diff = contact_codes[:, 1:] - contact_codes[:, :-1]  # [N, T-1]
    changed = (contact_diff != 0).float()

    if ignore_double_contact:
        valid_mask = ((contact_codes[:, :-1] < 3) & (contact_codes[:, 1:] < 3)).float()
        changed *= valid_mask

    alternation_score = torch.sum(changed, dim=1) / (history_len - 1)
    stall_penalty = torch.sum((contact_diff == 0).float(), dim=1) * penalty_scale

    # Scale reward based on commanded velocity magnitude
    cmd_vel = env.command_manager.get_command("base_velocity")[:, :2]
    cmd_vel_mag = torch.norm(cmd_vel, dim=1)
    cmd_scaling = torch.tanh(cmd_vel_mag * scaling_factor)

    total_reward = (alternation_score - stall_penalty) * cmd_scaling
    return total_reward.clamp(max=max_reward)
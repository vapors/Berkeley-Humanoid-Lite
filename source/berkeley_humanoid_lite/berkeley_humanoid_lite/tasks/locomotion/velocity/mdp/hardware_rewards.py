"""Standing-aware and bounded-action reward terms for v1.2.3 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _standing_mask(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :3], dim=1) < threshold


def bounded_last_action(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the current bounded normalized action for the policy observation."""
    term = env.action_manager.get_term(action_name)
    return term.bounded_actions


def bounded_action_rate_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Squared change in bounded normalized action."""
    term = env.action_manager.get_term(action_name)
    delta = term.bounded_actions - term.previous_bounded_actions
    return torch.sum(torch.square(delta), dim=1)


def bounded_action_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Squared bounded normalized action magnitude."""
    action = env.action_manager.get_term(action_name).bounded_actions
    return torch.sum(torch.square(action), dim=1)


def raw_action_excess_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Penalize only raw policy output magnitude beyond the valid [-1, 1] contract."""
    raw = env.action_manager.get_term(action_name).raw_actions
    excess = torch.relu(torch.abs(raw) - 1.0)
    return torch.sum(torch.square(excess), dim=1)


def standing_base_xy_speed_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal base speed only for exact/near standing commands."""
    asset = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    value = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return value * mask


def standing_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw rotation only for standing commands."""
    asset = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask


def standing_default_joint_pose_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from q_default when commanded to stand."""
    asset: Articulation = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(error), dim=1) * mask


def standing_base_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    desired_height: float,
    std: float = 0.05,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward nominal root-body COM height under standing commands.

    The Lilgreen asset's root/link frame is near ground level, so ``root_pos_w[:, 2]``
    is not a useful pelvis/base-height measure.  The physical root-body center of mass
    is represented by ``root_com_pos_w`` and is the quantity shaped here.
    """
    asset = env.scene[asset_cfg.name]
    mask = _standing_mask(env, command_name, command_threshold)
    current_height = asset.data.root_com_pos_w[:, 2]
    error = current_height - desired_height
    reward = torch.exp(-0.5 * torch.square(error / std))
    return reward * mask


def soft_torque_utilization_l2(
    env: ManagerBasedRLEnv,
    torque_limit_nm: float,
    soft_ratio: float = 0.70,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only torque utilization above a soft fraction of peak torque.

    This complements the small global torque L2 term. Moderate control effort remains
    available for balance, while sustained operation near the ST3215 peak/stall limit
    becomes increasingly expensive.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")

    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    utilization = torque / torque_limit_nm
    excess = torch.relu(utilization - soft_ratio)
    return torch.sum(torch.square(excess), dim=1)


def standing_both_feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward both feet being in contact during standing."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > force_threshold
    both_contact = torch.all(in_contact, dim=1)
    return both_contact * _standing_mask(env, command_name, command_threshold)


def standing_feet_slide(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize planted-foot horizontal slip during standing."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > force_threshold
    )
    asset = env.scene[asset_cfg.name]
    body_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(torch.linalg.vector_norm(body_vel_xy, dim=-1) * contacts, dim=1)
    return slip * _standing_mask(env, command_name, command_threshold)



def standing_soft_torque_utilization_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    torque_limit_nm: float,
    soft_ratio: float = 0.70,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Standing-only soft torque utilization penalty.

    This is useful for Stand checkpoints where we want bent knees but do not want
    a static crouch that sits continuously near the ST3215 torque envelope.
    """
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be positive")
    if not 0.0 <= soft_ratio < 1.0:
        raise ValueError("soft_ratio must be in [0, 1)")
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    utilization = torque / torque_limit_nm
    excess = torch.relu(utilization - soft_ratio)
    return torch.sum(torch.square(excess), dim=1) * _standing_mask(env, command_name, command_threshold)


def standing_contact_force_balance_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    force_threshold: float = 1.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Penalize left/right support-force imbalance under standing commands.

    This is a stand-only anti-lean term, not a knee-symmetry term. It discourages
    the first v1.4.5 failure mode where the policy stood by loading one leg hard.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    if forces.shape[1] != 2:
        # Keep the term safe if a future foot pattern matches more bodies.
        left = torch.sum(forces[:, 0::2], dim=1)
        right = torch.sum(forces[:, 1::2], dim=1)
    else:
        left, right = forces[:, 0], forces[:, 1]
    total = left + right
    both_loaded = (left > force_threshold) & (right > force_threshold)
    normalized_diff = (left - right) / (total + eps)
    return torch.square(normalized_diff) * both_loaded * _standing_mask(env, command_name, command_threshold)


def standing_com_over_feet_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize standing COM displacement from the midpoint of the two feet.

    This keeps the Stand seed from leaning onto one leg without teaching gait-time
    knee symmetry. Use only on standing tasks.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    midpoint_xy = torch.mean(foot_xy, dim=1)
    com_xy = asset.data.root_com_pos_w[:, :2]
    return torch.sum(torch.square(com_xy - midpoint_xy), dim=1) * _standing_mask(env, command_name, command_threshold)

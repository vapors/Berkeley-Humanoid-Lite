"""Velocity-Lilgreen-*-ST3215-Loaded-v5: v1.4.5 athletic vector-residual profile.

This branch is intentionally deployment-impacting. It keeps the policy I/O shape
(45-D observation, 12-D action) and the residual target equation, but changes the
q_default profile and uses a per-joint vector residual scale:

* a lower athletic q_default with more knee bend;
* more residual authority for hip/knee/ankle pitch joints;
* lower stand and moving COM-height targets;
* v1.4.4 alternating-step rewards made more grounded with no-flight, clearance-window,
  COM-over-stance-foot, and knee-flexion shaping.

Older v1.4.0-v1.4.4 tasks are left untouched for reproducibility.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.mdp.hardware_contract import (
    ACTIONABLE_JOINTS_V1_2_3,
    HARDWARE_LOWER_LIMIT_RAD,
    HARDWARE_UPPER_LIMIT_RAD,
    RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC,
    RESIDUAL_ACTION_SCALE_RAD_V1_4_5_STABILIZED,
    ST3215_NO_LOAD_SPEED_RAD_S,
    ST3215_PEAK_TORQUE_NM,
    TRAINING_DEFAULT_RAD_V1_4_5_ATHLETIC,
    TRAINING_DEFAULT_RAD_V1_4_5_STABILIZED,
    V1_4_5_ATHLETIC_MOVING_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_MOVING_BASE_COM_HEIGHT_M,
    V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
    V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
    athletic_default_joint_pos_dict,
    athletic_stabilized_default_joint_pos_dict,
)
from berkeley_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_actuator_model import (
    DATASET_NAME,
    ST3215_BUS_PHASE_WAIT_RANGE_S,
    ST3215_CENTER_HYSTERESIS_SPAN_RAD,
    ST3215_FIRST_ENCODER_DELAY_MEDIAN_S,
    ST3215_FIRST_ENCODER_DELAY_RANGE_S,
    ST3215_PEAK_VELOCITY_CURVES_RAD_S,
    ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
    ST3215_STATIC_GAIN_MEDIAN,
    ST3215_STEP_AMPLITUDE_KNOTS_RAD,
    ST3215_TAU_MEDIAN_S,
    ST3215_TAU_P10_S,
    ST3215_TAU_P90_S,
)
from berkeley_humanoid_lite.tasks.locomotion.velocity.mdp.st3215_loaded_actuator_model import (
    LOADED_CROUCH_DIRECTION_SIGN,
    LOADED_CROUCH_LOW_DEMAND_GAIN,
    LOADED_CROUCH_VMAX_RAD_S,
    LOADED_DATASET_NAME,
    LOADED_DIRECTION_CONDITIONING_WEIGHT,
    LOADED_ENVELOPE_COMBINATION_RULE,
    LOADED_RETURN_LOW_DEMAND_GAIN,
    LOADED_RETURN_TAU_SCALE,
    LOADED_RETURN_VMAX_RAD_S,
)

from .env_cfg_hardware import HardwareRewardsCfg
from .env_cfg_stand import StandRewardsCfg
from .v1_2_3_common import (
    FEET_BODY_PATTERN,
    HardwareCommandsCfg,
    HardwareObservationsCfg,
    StandCommandsCfg,
    StandObservationsCfg,
    V123TerminationsCfg,
)
from .v1_3_0_st3215_common import ST3215HardwareEventsCfg, ST3215StandEventsCfg
from .v1_4_0_st3215_loaded_common import V140ST3215LoadedHardwareAlignedEnvCfg


def _scaled(values: list[float], scale: float) -> list[float]:
    return [float(v) * scale for v in values]


def _scaled_curves(values: list[list[float]], scale: float) -> list[list[float]]:
    return [[float(v) * scale for v in row] for row in values]


def _apply_athletic_default(scene_robot_cfg) -> None:
    """Patch the robot init-state q_default for the v1.4.5 athletic profile."""
    scene_robot_cfg.init_state.joint_pos.update(athletic_default_joint_pos_dict(include_toes=True))


def _apply_stabilized_athletic_default(scene_robot_cfg) -> None:
    """Patch q_default for the v1.4.5 Stand-stabilized athletic profile."""
    scene_robot_cfg.init_state.joint_pos.update(athletic_stabilized_default_joint_pos_dict(include_toes=True))


@configclass
class V145ST3215LoadedAthleticActionsCfg:
    """Loaded ST3215 action model with v1.4.5 athletic q_default/vector residual.

    This keeps the residual target equation but changes the action profile:
    q_target = clip(q_default_athletic + clip(a_raw,-1,1) * residual_scale_vector, limits).
    Exported policies should be treated as action_contract_version 4 / athletic
    vector residual profile by deployment code.
    """

    joint_pos = mdp.ST3215MeasuredResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS_V1_2_3,
        lower_limits=HARDWARE_LOWER_LIMIT_RAD,
        upper_limits=HARDWARE_UPPER_LIMIT_RAD,
        residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_4_5_ATHLETIC,
        preserve_order=True,
        actuator_model_name=(
            f"{DATASET_NAME}:stage_a_athletic_quick+{LOADED_DATASET_NAME}:stage_b_loaded_athletic_quick"
        ),
        actuator_model_stage="stage_b_loaded_v145_athletic_vector_residual",
        velocity_amplitude_knots_rad=ST3215_STEP_AMPLITUDE_KNOTS_RAD,
        velocity_curves_rad_s=_scaled_curves(ST3215_PEAK_VELOCITY_CURVES_RAD_S, 1.08),
        tau_median_s=_scaled(ST3215_TAU_MEDIAN_S, 0.84),
        tau_p10_s=_scaled(ST3215_TAU_P10_S, 0.84),
        tau_p90_s=_scaled(ST3215_TAU_P90_S, 0.84),
        static_gain=ST3215_STATIC_GAIN_MEDIAN,
        small_signal_error_floor_rad=ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
        center_hysteresis_span_rad=ST3215_CENTER_HYSTERESIS_SPAN_RAD,
        bus_phase_delay_s_range=ST3215_BUS_PHASE_WAIT_RANGE_S,
        response_delay_s_range=ST3215_FIRST_ENCODER_DELAY_RANGE_S,
        response_delay_s_nominal=ST3215_FIRST_ENCODER_DELAY_MEDIAN_S,
        response_delay_scale=0.68,
        velocity_scale_range=(1.02, 1.14),
        randomize_tau=True,
        randomize_velocity_scale=True,
        randomize_response_delay=True,
        randomize_bus_phase=True,
        loaded_envelope_enabled=True,
        loaded_dataset_name=LOADED_DATASET_NAME,
        loaded_envelope_combination_rule=LOADED_ENVELOPE_COMBINATION_RULE,
        loaded_crouch_direction_sign=LOADED_CROUCH_DIRECTION_SIGN,
        loaded_crouch_low_demand_gain=LOADED_CROUCH_LOW_DEMAND_GAIN,
        loaded_crouch_vmax_rad_s=LOADED_CROUCH_VMAX_RAD_S,
        loaded_return_low_demand_gain=LOADED_RETURN_LOW_DEMAND_GAIN,
        loaded_return_vmax_rad_s=LOADED_RETURN_VMAX_RAD_S,
        loaded_direction_conditioning_weight=LOADED_DIRECTION_CONDITIONING_WEIGHT,
        loaded_return_tau_scale=LOADED_RETURN_TAU_SCALE,
        loaded_velocity_scale_range=(1.02, 1.12),
        randomize_loaded_velocity_scale=True,
    )


@configclass
class HardwareV145AthleticStandRewardsCfg(StandRewardsCfg):
    """Standing rewards for rebuilding a lower athletic Stand checkpoint."""

    # The new default pose itself is knee-bent; keep posture but don't force the old tall COM.
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
            "std": 0.090,
        },
        weight=1.05,
    )
    stand_default_pose = RewTerm(
        func=mdp.standing_default_joint_pose_l2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.70,
    )
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.120)
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.72,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.018,
    )


@configclass
class HardwareV145GroundedStepRewardsCfg(HardwareRewardsCfg):
    """v1.4.5 Hardware rewards: lower, grounded, alternating command-aligned steps."""

    # Tracking and progress: keep the v1.4.4 movement pressure, but reward the body
    # moving in the command direction more than swing motion alone.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.28},
        weight=3.7,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.40},
        weight=1.6,
    )
    moving_velocity_along_command = RewTerm(
        func=mdp.moving_velocity_along_command,
        params={"command_name": "base_velocity", "command_threshold": 0.12},
        weight=1.8,
    )
    moving_no_progress_l1 = RewTerm(
        func=mdp.moving_no_progress_l1,
        params={"command_name": "base_velocity", "command_threshold": 0.12, "min_fraction": 0.50},
        weight=-1.45,
    )

    # Grounded alternating support. v1.4.4 alternated but hopped; v1.4.5 makes
    # single support valuable only when at least one foot remains grounded.
    moving_single_support_time = RewTerm(
        func=mdp.moving_single_support_time,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_reward_time_s": 0.14,
        },
        weight=0.20,
    )
    moving_alternating_single_support = RewTerm(
        func=mdp.moving_alternating_single_support_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=0.90,
    )
    moving_contact_switch = RewTerm(
        func=mdp.moving_contact_switch_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=0.10,
    )
    moving_no_support = RewTerm(
        func=mdp.moving_no_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=-1.80,
    )
    moving_double_support_penalty = RewTerm(
        func=mdp.moving_double_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=-0.10,
    )
    moving_foot_air_balance = RewTerm(
        func=mdp.moving_foot_air_time_balance_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_unpenalized_s": 0.10,
        },
        weight=-0.75,
    )
    moving_long_single_support = RewTerm(
        func=mdp.moving_long_single_support_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_air_time_s": 0.22,
        },
        weight=-1.25,
    )
    moving_stance_foot_slide = RewTerm(
        func=mdp.moving_stance_foot_slide_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
        },
        weight=-1.05,
    )
    moving_com_over_stance_foot = RewTerm(
        func=mdp.moving_com_over_stance_foot_reward,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "force_threshold": 1.0,
            "std": 0.14,
        },
        weight=0.45,
    )

    # Lower athletic height and knee flexion permission.
    moving_relaxed_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_ATHLETIC_MOVING_BASE_COM_HEIGHT_M,
            "std": 0.105,
            "command_threshold": 0.12,
        },
        weight=0.50,
    )
    moving_knee_flexion_band = RewTerm(
        func=mdp.moving_knee_flexion_band_reward,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"], preserve_order=True),
            "command_threshold": 0.12,
            "lower_rad": 0.55,
            "upper_rad": 1.22,
            "std": 0.22,
        },
        weight=0.35,
    )

    # Step clearance: allow a small foot lift, discourage the v1.4.4 hopping solution.
    swing_foot_clearance = RewTerm(
        func=mdp.swing_foot_clearance,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "clearance_target_m": 0.028,
            "max_reward": 1.0,
        },
        weight=0.25,
    )
    moving_swing_clearance_window = RewTerm(
        func=mdp.moving_swing_clearance_window_penalty,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
            "max_clearance_m": 0.060,
        },
        weight=-4.0,
    )
    swing_foot_velocity_along_command = RewTerm(
        func=mdp.swing_foot_velocity_along_command,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.12,
        },
        weight=0.35,
    )

    # Keep safety, but do not over-penalize knees now that we need them to work.
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.140)
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.74,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.012,
    )
    knee_soft_torque_utilization = RewTerm(
        func=mdp.moving_soft_torque_utilization_l2,
        params={
            "command_name": "base_velocity",
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.78,
            "command_threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"], preserve_order=True),
        },
        weight=-0.006,
    )
    ankle_pitch_soft_torque_utilization = RewTerm(
        func=mdp.moving_soft_torque_utilization_l2,
        params={
            "command_name": "base_velocity",
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.72,
            "command_threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"], preserve_order=True),
        },
        weight=-0.020,
    )

    # Standing is still preserved, but around the lower athletic q_default.
    stand_base_xy_speed = RewTerm(
        func=mdp.standing_base_xy_speed_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.90,
    )
    stand_yaw_rate = RewTerm(
        func=mdp.standing_yaw_rate_l2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
        weight=-0.25,
    )
    stand_default_pose = RewTerm(
        func=mdp.standing_default_joint_pose_l2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.35,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
            "std": 0.095,
        },
        weight=0.60,
    )
    stand_both_feet_contact = RewTerm(
        func=mdp.standing_both_feet_contact,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "force_threshold": 1.0,
        },
        weight=0.60,
    )
    stand_feet_slide = RewTerm(
        func=mdp.standing_feet_slide,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
        },
        weight=-0.75,
    )


@configclass
class ST3215LoadedV145HardwareCurriculumsCfg:
    hardware_stage = CurrTerm(func=mdp.st3215_loaded_v145_hardware_stage_curriculum)
    policy_diagnostics = CurrTerm(
        func=mdp.PolicyDiagnostics,
        params={
            "command_name": "base_velocity",
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "joint_velocity_limit_rad_s": ST3215_NO_LOAD_SPEED_RAD_S,
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "torque_soft_ratio": 0.74,
            "desired_base_com_height_m": V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class ST3215LoadedV145StandCurriculumsCfg:
    policy_diagnostics = CurrTerm(
        func=mdp.PolicyDiagnostics,
        params={
            "command_name": "base_velocity",
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "joint_velocity_limit_rad_s": ST3215_NO_LOAD_SPEED_RAD_S,
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "torque_soft_ratio": 0.72,
            "desired_base_com_height_m": V1_4_5_ATHLETIC_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenStandST3215LoadedV145EnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """v1.4.5 athletic standing task. Train this before Hardware-v5."""

    commands: StandCommandsCfg = StandCommandsCfg()
    observations: StandObservationsCfg = StandObservationsCfg()
    actions: V145ST3215LoadedAthleticActionsCfg = V145ST3215LoadedAthleticActionsCfg()
    rewards: HardwareV145AthleticStandRewardsCfg = HardwareV145AthleticStandRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215StandEventsCfg = ST3215StandEventsCfg()
    curriculum: ST3215LoadedV145StandCurriculumsCfg = ST3215LoadedV145StandCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_athletic_default(self.scene.robot)
        self.events.actuator_gains = None


@configclass
class LilgreenHardwareST3215LoadedV145EnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """v1.4.5 athletic Hardware task for lower, grounded, alternating stepping."""

    commands: HardwareCommandsCfg = HardwareCommandsCfg()
    observations: HardwareObservationsCfg = HardwareObservationsCfg()
    actions: V145ST3215LoadedAthleticActionsCfg = V145ST3215LoadedAthleticActionsCfg()
    rewards: HardwareV145GroundedStepRewardsCfg = HardwareV145GroundedStepRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215HardwareEventsCfg = ST3215HardwareEventsCfg()
    curriculum: ST3215LoadedV145HardwareCurriculumsCfg = ST3215LoadedV145HardwareCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_athletic_default(self.scene.robot)
        self.events.actuator_gains = None


@configclass
class V145ST3215LoadedAthleticStabilizedActionsCfg(V145ST3215LoadedAthleticActionsCfg):
    """v1.4.5 stabilized athletic action profile.

    Same contract v4/vector residual idea as v5, but with a moderated q_default
    applied by the environment. The residual vector is kept large for later gait.
    """

    joint_pos = mdp.ST3215MeasuredResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS_V1_2_3,
        lower_limits=HARDWARE_LOWER_LIMIT_RAD,
        upper_limits=HARDWARE_UPPER_LIMIT_RAD,
        residual_scale_rad=RESIDUAL_ACTION_SCALE_RAD_V1_4_5_STABILIZED,
        preserve_order=True,
        actuator_model_name=(
            f"{DATASET_NAME}:stage_a_athletic_stabilized+{LOADED_DATASET_NAME}:stage_b_loaded_athletic_stabilized"
        ),
        actuator_model_stage="stage_b_loaded_v145_stabilized_vector_residual",
        velocity_amplitude_knots_rad=ST3215_STEP_AMPLITUDE_KNOTS_RAD,
        velocity_curves_rad_s=_scaled_curves(ST3215_PEAK_VELOCITY_CURVES_RAD_S, 1.06),
        tau_median_s=_scaled(ST3215_TAU_MEDIAN_S, 0.88),
        tau_p10_s=_scaled(ST3215_TAU_P10_S, 0.88),
        tau_p90_s=_scaled(ST3215_TAU_P90_S, 0.88),
        static_gain=ST3215_STATIC_GAIN_MEDIAN,
        small_signal_error_floor_rad=ST3215_SMALL_SIGNAL_ERROR_FLOOR_RAD,
        center_hysteresis_span_rad=ST3215_CENTER_HYSTERESIS_SPAN_RAD,
        bus_phase_delay_s_range=ST3215_BUS_PHASE_WAIT_RANGE_S,
        response_delay_s_range=ST3215_FIRST_ENCODER_DELAY_RANGE_S,
        response_delay_s_nominal=ST3215_FIRST_ENCODER_DELAY_MEDIAN_S,
        response_delay_scale=0.72,
        velocity_scale_range=(1.00, 1.10),
        randomize_tau=True,
        randomize_velocity_scale=True,
        randomize_response_delay=True,
        randomize_bus_phase=True,
        loaded_envelope_enabled=True,
        loaded_dataset_name=LOADED_DATASET_NAME,
        loaded_envelope_combination_rule=LOADED_ENVELOPE_COMBINATION_RULE,
        loaded_crouch_direction_sign=LOADED_CROUCH_DIRECTION_SIGN,
        loaded_crouch_low_demand_gain=LOADED_CROUCH_LOW_DEMAND_GAIN,
        loaded_crouch_vmax_rad_s=LOADED_CROUCH_VMAX_RAD_S,
        loaded_return_low_demand_gain=LOADED_RETURN_LOW_DEMAND_GAIN,
        loaded_return_vmax_rad_s=LOADED_RETURN_VMAX_RAD_S,
        loaded_direction_conditioning_weight=LOADED_DIRECTION_CONDITIONING_WEIGHT,
        loaded_return_tau_scale=LOADED_RETURN_TAU_SCALE,
        loaded_velocity_scale_range=(1.00, 1.10),
        randomize_loaded_velocity_scale=True,
    )


@configclass
class ST3215LoadedV145StabilizedStandEventsCfg(ST3215StandEventsCfg):
    """Stand-v5 stabilization events with gentle external balance perturbations.

    The interval push is deliberately small. It encourages both legs to participate
    in recovery without becoming a Hardware locomotion disturbance curriculum.
    """

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        params={"velocity_range": {"x": (-0.055, 0.055), "y": (-0.055, 0.055)}},
        mode="interval",
        interval_range_s=(5.0, 8.0),
    )


@configclass
class HardwareV145StabilizedStandRewardsCfg(HardwareV145AthleticStandRewardsCfg):
    """Stabilized Stand-v5 rewards.

    Targets a moderate 0.43-0.44 m athletic stance, reduces the incentive for a
    deep right-leg brace, and avoids knee-symmetry shaping that might be harmful
    for later gait transfer.
    """

    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
            "std": 0.075,
        },
        weight=0.95,
    )
    stand_default_pose = RewTerm(
        func=mdp.standing_default_joint_pose_l2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.90,
    )
    raw_action_excess_l2 = RewTerm(func=mdp.raw_action_excess_l2, params={"action_name": "joint_pos"}, weight=-0.140)
    soft_torque_utilization = RewTerm(
        func=mdp.soft_torque_utilization_l2,
        params={
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.68,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
        },
        weight=-0.026,
    )
    stand_sagittal_soft_torque = RewTerm(
        func=mdp.standing_soft_torque_utilization_l2,
        params={
            "command_name": "base_velocity",
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "soft_ratio": 0.62,
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint", ".*_knee_pitch_joint", ".*_ankle_pitch_joint"],
                preserve_order=True,
            ),
        },
        weight=-0.045,
    )
    stand_contact_force_balance = RewTerm(
        func=mdp.standing_contact_force_balance_l2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.05,
            "force_threshold": 1.0,
        },
        weight=-0.35,
    )
    stand_com_over_feet = RewTerm(
        func=mdp.standing_com_over_feet_l2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "command_threshold": 0.05,
        },
        weight=-1.10,
    )
    stand_both_feet_contact = RewTerm(
        func=mdp.standing_both_feet_contact,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "force_threshold": 1.0,
        },
        weight=0.80,
    )


@configclass
class HardwareV145StabilizedGroundedStepRewardsCfg(HardwareV145GroundedStepRewardsCfg):
    """Hardware-v5s rewards matching the stabilized standing profile."""

    moving_relaxed_base_height = RewTerm(
        func=mdp.moving_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_MOVING_BASE_COM_HEIGHT_M,
            "std": 0.100,
            "command_threshold": 0.12,
        },
        weight=0.45,
    )
    stand_base_height = RewTerm(
        func=mdp.standing_base_height_exp,
        params={
            "command_name": "base_velocity",
            "desired_height": V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
            "std": 0.080,
        },
        weight=0.60,
    )


@configclass
class ST3215LoadedV145StabilizedStandCurriculumsCfg(ST3215LoadedV145StandCurriculumsCfg):
    policy_diagnostics = CurrTerm(
        func=mdp.PolicyDiagnostics,
        params={
            "command_name": "base_velocity",
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "joint_velocity_limit_rad_s": ST3215_NO_LOAD_SPEED_RAD_S,
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "torque_soft_ratio": 0.68,
            "desired_base_com_height_m": V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class ST3215LoadedV145StabilizedHardwareCurriculumsCfg(ST3215LoadedV145HardwareCurriculumsCfg):
    policy_diagnostics = CurrTerm(
        func=mdp.PolicyDiagnostics,
        params={
            "command_name": "base_velocity",
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTIONABLE_JOINTS_V1_2_3, preserve_order=True
            ),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_PATTERN),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_PATTERN),
            "joint_velocity_limit_rad_s": ST3215_NO_LOAD_SPEED_RAD_S,
            "torque_limit_nm": ST3215_PEAK_TORQUE_NM,
            "torque_soft_ratio": 0.70,
            "desired_base_com_height_m": V1_4_5_STABILIZED_STAND_BASE_COM_HEIGHT_M,
            "update_interval_steps": 25,
            "standing_command_threshold": 0.05,
        },
    )


@configclass
class LilgreenStandST3215LoadedV145StabilizedEnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """Stabilized v1.4.5 Stand task: moderate athletic stance plus anti-lean shaping."""

    commands: StandCommandsCfg = StandCommandsCfg()
    observations: StandObservationsCfg = StandObservationsCfg()
    actions: V145ST3215LoadedAthleticStabilizedActionsCfg = V145ST3215LoadedAthleticStabilizedActionsCfg()
    rewards: HardwareV145StabilizedStandRewardsCfg = HardwareV145StabilizedStandRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215LoadedV145StabilizedStandEventsCfg = ST3215LoadedV145StabilizedStandEventsCfg()
    curriculum: ST3215LoadedV145StabilizedStandCurriculumsCfg = ST3215LoadedV145StabilizedStandCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_stabilized_athletic_default(self.scene.robot)
        self.events.actuator_gains = None


@configclass
class LilgreenHardwareST3215LoadedV145StabilizedEnvCfg(V140ST3215LoadedHardwareAlignedEnvCfg):
    """Hardware task matching the stabilized v1.4.5 default/profile."""

    commands: HardwareCommandsCfg = HardwareCommandsCfg()
    observations: HardwareObservationsCfg = HardwareObservationsCfg()
    actions: V145ST3215LoadedAthleticStabilizedActionsCfg = V145ST3215LoadedAthleticStabilizedActionsCfg()
    rewards: HardwareV145StabilizedGroundedStepRewardsCfg = HardwareV145StabilizedGroundedStepRewardsCfg()
    terminations: V123TerminationsCfg = V123TerminationsCfg()
    events: ST3215HardwareEventsCfg = ST3215HardwareEventsCfg()
    curriculum: ST3215LoadedV145StabilizedHardwareCurriculumsCfg = ST3215LoadedV145StabilizedHardwareCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_stabilized_athletic_default(self.scene.robot)
        self.events.actuator_gains = None

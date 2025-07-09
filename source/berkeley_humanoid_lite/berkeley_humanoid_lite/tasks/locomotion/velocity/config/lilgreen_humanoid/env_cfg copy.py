import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from berkeley_humanoid_lite_assets.robots.lilgreen_humanoid import LILGREEN_CFG, LILGREEN_JOINTS, ACTIONABLE_JOINTS


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LILGREEN_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-2.0, n_max=2.0),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LILGREEN_JOINTS, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        #resampling_time_range=(10.0, 10.0),
        resampling_time_range=(2.0, 4.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            #lin_vel_x=(-1.2, 1.0),lin_vel_y=(-0.5, 0.5),ang_vel_z=(-1.0, 1.0),
            #lin_vel_x=(-1.0, 1.0),lin_vel_y=(-0.75, 0.75),ang_vel_z=(-1.5, 1.5),# success
            lin_vel_x=(-1.2, 1.0),lin_vel_y=(-0.75, 0.75),ang_vel_z=(-1.5, 1.5),
            #lin_vel_x=(-1.0, 1.0),lin_vel_y=(-0.5, 0.5),ang_vel_z=(-1.5, 1.5),
            #lin_vel_x=(-1.0, 1.0),lin_vel_y=(-0.5, 0.5),ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi, math.pi),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIONABLE_JOINTS,
        preserve_order=True,
        scale=0.35,
        use_default_offset=True,
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # === Reward for basic survival ===
    # termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )

    # motion smoothness
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        #weight=-0.1,
        weight=-0.02,
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        #weight=-0.05,
        weight=-0.005,
    )
    # ensure the robot is standing upright
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        #weight=-1.0,
        weight=-0.05,
    )

    # joint efforts
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        #weight=-2.0e-5,
        weight=-6.0e-6,
        
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LILGREEN_JOINTS)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        #weight=-1.0e-7,
        weight=-0.01e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LILGREEN_JOINTS)},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        #weight=-1.0,
        weight=-0.1,
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        #weight=-0.001,
        weight=-0.0005,
    )


    # General tracking (all XY + yaw commands)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,  # Slightly reduced to avoid overshadowing the others
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Specific forward walking
    track_forward_velocity_exp = RewTerm(
        func=mdp.track_forward_velocity_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Specific backward walking
    track_backward_velocity_exp = RewTerm(
        func=mdp.track_backward_velocity_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Angular Z (yaw) rotation tracking
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )



    """
    # === Reward for task-space performance ===
    # command tracking performance
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        #weight=2.5,
        #params={"command_name": "base_velocity", "std": 0.5},
        #weight=0.5,# success        
        weight=1.5,# success
        params={"command_name": "base_velocity", "std": 0.3},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        #weight=1.0,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    """
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            #"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*", ".*_shoulder_.*", ".*_elbow_.*"]),"threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_yaw", ".*_knee_pitch"]),"threshold": 1.0,
        },
    )

    # encourage robot to take steps
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        #weight=1.0,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "threshold": 0.25,
        },
    )
    # penalize feet sliding on the ground to exploit physics sim inaccuracies
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
    )

    # penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        #weight=-0.6,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )

    # penalize deviation of ankle roll joints
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        #weight=-0.4,
        #weight=-0.02,#success
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
    )

    #**vapors***
    # ❌ Penalty to prevent doing the splits (encourages efficient stance)
    joint_deviation_splits= RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    # Encourage stepping with one foot lifting off (discourage hopping)
    step_no_hop = RewTerm(
        func=mdp.encourage_stepping_not_hopping_refined,
        weight=0.005,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        }
    )
    
    foot_alternation = RewTerm(
        func=mdp.encourage_foot_alternation,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
        }
    )

    swing_leg_lift = RewTerm(
        func=mdp.swing_leg_lift,
        #weight=0.5,  # Try 0.2–0.4; tune based on observed gait  #success
        weight=0.8,  # Try 0.2–0.4; tune based on observed gait  #success
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    lean_over_stance_leg = RewTerm(
        func=mdp.lean_over_stance_leg,
        weight=0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )

    discourage_double_air = RewTerm(
        func=mdp.discourage_double_air_time,
        #weight=-0.5,  # soft penalty #success
        weight=0.7,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll")}
    )


    '''
    # penalize deviation of shoulder joints
    joint_deviation_shoulder = RewTerm(
        func=mdp.joint_deviation_l1,
        
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"])},
    )

    # Penalize deviation of elbow joints
    joint_deviation_elbow = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_elbow_pitch_joint", ".*_elbow_roll_joint"])},
    )
    '''

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.85, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )

    scale_all_actuator_torque_constant = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-2.0, 2.0),
            "torque_range": (-2.0, 2.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class LilgreenHumanoidEnvCfg(LocomotionVelocityEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Physics settings
        # 25 Hz override
        self.decimation = 8
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005

        # Scene
        self.scene.robot = LILGREEN_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None

        self.events.push_robot = None


from berkeley_humanoid_lite.tasks.locomotion.velocity.config.lilgreen_humanoid.env_cfg_hardware_st3215_loaded_v144 import (
    HardwareV144AlternatingStepRewardsCfg,
    LilgreenHardwareST3215LoadedV144EnvCfg,
    V144ST3215LoadedQuickActionsCfg,
)
from berkeley_humanoid_lite.tasks.locomotion.velocity.mdp.curriculums import (
    st3215_loaded_v144_hardware_stage_curriculum,
)


def test_v144_task_contract_is_unchanged():
    cfg = LilgreenHardwareST3215LoadedV144EnvCfg()
    assert cfg.actions.joint_pos.residual_scale_rad == 0.20
    assert len(cfg.actions.joint_pos.joint_names) == 12
    assert cfg.actions.joint_pos.preserve_order is True
    assert cfg.actions.joint_pos.actuator_model_stage == "stage_b_loaded_v144_quick_train"


def test_v144_keeps_continuous_command_curriculum():
    # The function defaults are the experiment contract. This confirms we did not
    # introduce command bins while adding alternating-step rewards.
    defaults = st3215_loaded_v144_hardware_stage_curriculum.__defaults__
    assert defaults[1] == (32000, 128000, 256000)
    assert defaults[2] == (0.40, 0.30, 0.25, 0.20)
    assert defaults[3][0] == (-0.28, 0.28)
    assert defaults[4][1] == (-0.16, 0.16)


def test_v144_reward_terms_exist_and_relax_moving_height():
    rewards = HardwareV144AlternatingStepRewardsCfg()
    assert rewards.moving_alternating_single_support.weight > 0.0
    assert rewards.moving_foot_air_balance.weight < 0.0
    assert rewards.moving_long_single_support.weight < 0.0
    assert rewards.moving_stance_foot_slide.weight < 0.0
    assert rewards.moving_relaxed_base_height.weight > 0.0
    assert rewards.stand_base_height.params["std"] >= 0.075

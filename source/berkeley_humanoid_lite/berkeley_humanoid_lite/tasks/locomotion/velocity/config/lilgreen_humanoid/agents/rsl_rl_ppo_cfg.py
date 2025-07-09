from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class LilgreenHumanoidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "humanoid"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims = [256, 128, 128],
        critic_hidden_dims = [256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class LilgreenHumanoidPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48  # slightly longer rollouts for stability
    max_iterations = 6000
    save_interval = 100
    experiment_name = "humanoid"
    empirical_normalization = True  # normalize observations/rewards
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,  # slightly higher exploration for smoke test
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,  # conservative for smoke test
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.015,  # slightly faster learning
        max_grad_norm=1.0,
    )

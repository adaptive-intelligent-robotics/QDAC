from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

class DotDict(dict):

    class _Void:
        ...

    """
    A dictionary supporting dot notation.
    From https://gist.github.com/miku/dc6d06ed894bc23dfd5a364b7def5ed8#file-23689767-py
    """
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = DotDict(v)

    def __getattr__(self, item):
        attribute = self.get(item, self._Void())
        if isinstance(attribute, self._Void):
            raise AttributeError(f"Attribute {item} not found")
        return attribute

@dataclass
class PPGAConfig:
    # TODO: add feature space somewhere

    # others, for retrocompatibality only:
    name: str

    # Env
    episode_length: int
    
    env_name: Optional[str]
    grid_size: int  # Required, marked for completion
    num_dims: int  # Required, marked for completion
    popsize: int  # Required, marked for completion
    seed: int  # Required, marked for completion
    backend: str  # Required, marked for completion
    torch_deterministic: bool = False
    use_wandb: bool = False
    wandb_run_name: str = 'ppo_ant'
    wandb_group: Optional[str] = None
    wandb_project: str = 'PPGA'
    env_batch_size: int = 1
    report_interval: int = 5
    rollout_length: int = 2048
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    clip_value_coef: float = 0.2
    entropy_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    normalize_obs: bool = False
    normalize_returns: bool = False
    value_bootstrap: bool = False
    weight_decay: Optional[float] = None
    clip_obs_rew: bool = False
    num_emitters: int = 1
    log_arch_freq: int = 10
    save_scheduler: bool = True
    load_scheduler_from_cp: Optional[str] = None
    load_archive_from_cp: Optional[str] = None
    total_iterations: int = 100
    dqd_algorithm: Optional[str] = None  # Consider defining a default or making it a required argument
    expdir: Optional[str] = None
    save_heatmaps: bool = True
    use_surrogate_archive: bool = False
    sigma0: float = 1.0
    restart_rule: Optional[str] = None  # Consider defining a default or making it a required argument
    calc_gradient_iters: Optional[int] = None  # Consider defining a default or making it a required argument
    move_mean_iters: Optional[int] = None  # Consider defining a default or making it a required argument
    archive_lr: Optional[float] = None  # Consider defining a default or making it a required argument
    threshold_min: float = 0.0
    take_archive_snapshots: bool = False
    adaptive_stddev: bool = True

    @classmethod
    def get_grid_size_num_dims(cls, brax_env_name, feature_name):
        if brax_env_name == "humanoid":
            if feature_name == "jump":
                grid_size = 50
                num_dims = 1
            elif feature_name == "feet_contact":
                grid_size = 50
                num_dims = 2
            elif feature_name in ("angle", "angle_notrigo"):
                grid_size = 50
                num_dims = 1
            else:
                raise NotImplementedError(f"feature_name={feature_name} not implemented for brax_env_name={brax_env_name}")
        elif brax_env_name == "ant":
            if feature_name == "feet_contact":
                grid_size = 5
                num_dims = 4
            elif feature_name == "velocity":
                grid_size = 50
                num_dims = 2
            else:
                raise NotImplementedError(f"feature_name={feature_name} not implemented for brax_env_name={brax_env_name}")
        elif brax_env_name == "walker2d":
            if feature_name == "feet_contact":
                grid_size = 50
                num_dims = 2
            else:
                raise NotImplementedError(f"feature_name={feature_name} not implemented for brax_env_name={brax_env_name}")
        else:
            raise NotImplementedError(f"brax_env_name={brax_env_name} not implemented")

        return grid_size, num_dims

    @classmethod
    def override_config(cls, brax_env_name):
        if brax_env_name == "humanoid":
            override_config = dict()  # The default config is fine
        elif brax_env_name == "ant":
            override_config = {
                "anneal_lr": True,
                "clip_obs_rew": False,
                "adaptive_stddev": True,
                "learning_rate": 0.001,
                "target_kl": None,
                "sigma0": 3,
                "threshold_min": -500,
            }
        elif brax_env_name == "walker2d":
            override_config = {
                "clip_obs_rew": False,
                "learning_rate": 0.001,
                "sigma0": 1,
                "archive_lr": 0.5,
                "threshold_min": -10,
            }
        else:
            raise NotImplementedError(f"brax_env_name={brax_env_name} not implemented")

        return override_config

    @classmethod
    def create(cls, hydra_config,) -> PPGAConfig:
        assert "expdir" not in hydra_config.algo, "expdir should not be set in the config file"
        assert "env_name" not in hydra_config.algo, "env_name should not be set in the config file"
        assert "wandb_run_name" not in hydra_config.algo, "wandb_run_name should not be set in the config file"
        assert "seed" not in hydra_config.algo, "seed should not be set in the config file"
        assert "run_name" not in hydra_config.algo, "run_name should not be set in the config file"
        assert "grid_size" not in hydra_config.algo, "grid_size should not be set in the config file"
        assert "num_dims" not in hydra_config.algo, "num_dims should not be set in the config file"

        brax_env_name = hydra_config.task
        feature_name = hydra_config.feat

        grid_size, num_dims = cls.get_grid_size_num_dims(brax_env_name, feature_name)

        env_name = f"{brax_env_name}_{feature_name}"

        expdir = f"./experiments/paper_ppga_{env_name}"
        seed = hydra_config.seed

        run_name = f"paper_ppga_{env_name}_seed_{seed}"

        wandb_run_name = run_name

        dict_config = dict(
            **hydra_config.algo,
            expdir=expdir,
            seed=seed,
            env_name=env_name,
            wandb_run_name=wandb_run_name,
            grid_size=grid_size,
            num_dims=num_dims,
        )

        dict_config.update(
            cls.override_config(brax_env_name)
        )

        return cls(**dict_config)

    def as_dot_dict(self):
        return DotDict(asdict(self))

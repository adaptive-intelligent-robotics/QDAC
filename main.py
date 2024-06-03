from utils.env_utils import Config

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@hydra.main(version_base="1.2", config_path="configs/", config_name="config")
def main(config: Config) -> None:
    # QDAC-MB
    if config.algo.name == "qdac_mb":
        import main_qdac_mb as main

    # QDAC
    elif config.algo.name == "qdac":
        import main_qdac as main

    # Ablations
    ##  Ours with feat-based/timestep-based skill (separable skill)
    elif config.algo.name == "qdac_mb_no_sf":
        import main_qdac_mb as main
    ##  Ours with fixed lambda (still using main script with different config)
    elif config.algo.name == "qdac_mb_fixed_lambda":
        import main_qdac_mb as main
    ##  UVFA (still using main script with different config)
    elif config.algo.name == "uvfa":
        import main_qdac_mb as main

    # QD
    elif config.algo.name == "qd_pg":
        import main_qd_pg as main
    elif config.algo.name == "dcg_me":
        import main_dcg_me as main
    elif config.algo.name == "ppga":
        import main_ppga as main

    # URL
    elif config.algo.name == "domino":
        import main_domino as main
    elif config.algo.name == "smerl":
        import main_smerl as main
    elif config.algo.name == "smerl_reverse":
        import main_smerl_reverse as main
    else:
        raise NotImplementedError

    main.main(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()

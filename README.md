# Quality-Diversity Actor-Critic

This repository contains the code for "Quality-Diversity Actor-Critic: Learning High-Performing and Diverse Behaviors via Value and Successor Features Critics". Quality-Diversity Actor-Critic (QDAC) is a quality-diversity reinforcement learning algorithm that discovers high-performing and diverse skills.

## Installation

This code is supported on Python 3.10 and dependencies can be installed using the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you want to run PPGA as well, you will need to install pyribs as well:
```bash
pip install baselines/PPGA/pyribs
```

The experiments were run using one NVIDIA Quadro RTX 6000 with CUDA 11.

## Launch Experiments

### Learning Diverse High-Performing Skills

To launch an experiment, you can run the following command:
```bash
python main.py seed=$RANDOM algo=<algo> task=<task> feat=<feat>
```
where:
- `<algo>` can be any of the following algorithms:
  - `qdac`: QDAC
  - `qdac_mb`: QDAC-MB
  - `ppga`: PPGA
  - `dcg_me`: DCG-ME
  - `qd_pg`: QD-PG
  - `domino`: DOMiNO
  - `smerl`: SMERL
  - `smerl_reverse`: Reverse SMERL
  - `qdac_mb_fixed_lambda`: QDAC-MB with fixed lambda, it requires an extra parameter `+goal.fixed_lagrangian_coeff=<value>` where `<value>` is the value of the fixed lambda (between 0 and 1)
  - `qdac_mb_no_sf`: No-SF
  - `uvfa`: UVFA, it requires an extra parameter `+goal.fixed_lagrangian_coeff=<value>` where `<value>` is the value of the fixed lambda (between 0 and 1)
- `<task>` and `<feat>` can be any of the following combinations:
  - `task=humanoid` and `feat=feet_contact`
  - `task=ant` and `feat=feet_contact`
  - `task=walker2d` and `feat=feet_contact`
  - `task=ant` and `feat=velocity`
  - `task=humanoid` and `feat=jump`
  - `task=humanoid` and `feat=angle`

The configurations are located in the `configs` folder. The results are all saved in the `output/` folder. We use [WandB](https://wandb.ai/site) for logging.

For `qdac_mb`, `qdac_mb_fixed_lambda`, `qdac_mb_no_sf`, and `uvfa`, you can specify the Brax backend to use by adding the following parameter:`+backend=<backend>` where `<backend>` can be any Brax backend (e.g. `spring` and `generalized`). For other algorithms, the backend can be specified by adding the following parameter: `algo.backend=<backend>`. The `spring` backend is used by default.

### Harnessing Skills for Few-Shot Adaptation and Hierarchical Learning

To launch a few-shot adaptation experiment, you can run the following command:
```bash
python main_adaptation_<type>.py --algo=<algo> --path=<results_path> --seed=$RANDOM
```
where:
- `<type>` can be any of the following types:
  - `failure`: Only works with `task=humanoid` and `feat=feet_contact`
  - `friction`: Only works with `task=walker2d` and `feat=feet_contact`
  - `gravity`: Only works with `task=ant` and `feat=feet_contact`
  - `hurdle`: Only works with `task=humanoid` and `feat=jump`
- `<algo>` can be any of the above algorithms, except for `qdac_mb_no_sf` and `qdac_mb_fixed_lambda`
- `<results_path>` is the path to the results of the quality-diversity experiment

To launch a hierarchical learning experiment, you can run the following command:
```bash
python main_adaptation_wall.py algo_name=<algo> path=<results_path> seed=$RANDOM
```
where:
- `<algo>` can be any of the above algorithms, except for `qdac_mb_no_sf` and `qdac_mb_fixed_lambda`
- `<results_path>` is the path to the results of the quality-diversity experiment (only works with `task=ant` and `feat=velocity`)

The results take the form of a csv file in the quality-diversity experiment folder.

## BibTeX

```
@inproceedings{airl2024qdac,
	title={Quality-Diversity Actor-Critic: Learning High-Performing and Diverse Behaviors via Value and Successor Features Critics},
	author={Grillotti, Luca and Faldor, Maxence and González León, Borja and Cully, Antoine},
	booktitle={International Conference on Machine Learning},
	year={2024},
	organization={PMLR},
}
```

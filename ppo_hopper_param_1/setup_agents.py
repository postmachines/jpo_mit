import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("../MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["HopperPyBulletEnv-v0"],
    "mode": ["ppo"],
    "out_dir": ["ppo_hopper_param_1/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [False],
    "ppo_lr_adam": [3e-6] * 10,
    "advanced_logging": [True],
    "val_lr": [2e-5],
    "cpu": [True],
    "gamma": [0.80],
    "ppo_epochs": [4],
    "lambda": [0.80],
}


all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)

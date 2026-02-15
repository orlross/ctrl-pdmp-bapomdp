#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------

import argparse
import importlib
import sys
from typing import Optional
import ray
from ray import air, tune
import yaml
import gymnasium
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from gymnasium.envs.registration import register
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO
import os
from env.full_pdmp import Patient
from env.wrappers import (
    ActionMaskingWrapper,
    PartiallyObservableWrapper,
    BayesAdaptiveWrapper,
)

from action_mask_model import (
    DQNTorchActionMaskModel,
    R2D2LSTMTorchActionMaskModel,
    TorchActionMaskModel,
    LSTMTorchActionMaskModel,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-file",
    type=str,
    help="Path to the python file containing the configuration to run",
    required=True,
)

parser.add_argument(
    "--model",
    type=str,
    choices=["pomdp", "bapomdp"],
    required=True,
    help="Choose one of the models: pomdp, bapomdp.",
)
parser.add_argument("--stop-iters", type=int, help="Number of iterations to train.")
parser.add_argument("--stop-timesteps", type=int, help="Number of timesteps to train.")
parser.add_argument("--stop-reward", type=float, help="Reward at which we stop .")
parser.add_argument(
    "--num-samples",
    type=int,
    default=3,
    help="Number of times to sample from the hyperparameter space.If this is -1, (virtually) infinite samples are generated until a stopping condition is met.",
)
parser.add_argument(
    "--output-file",
    type=str,
    default="tuned-hyperparams.yaml",
    help="Name of the file to save with tuned hyperparameters",
)

args = parser.parse_args()
module_name = os.path.basename(args.config_file).replace(".py", "")
print("module_name", module_name)
spec = importlib.util.spec_from_file_location(module_name, args.config_file)
print("spec ", spec)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
print("end loaded module")


# Register your environment
register_env("pdmp/Patient-v0", Patient)


def create_wrapped_env(env_config):
    env = PartiallyObservableWrapper(Patient())
    env.reset(options=env_config)
    return env


def create_wrapped_ba_env(env_config):
    env = BayesAdaptiveWrapper(PartiallyObservableWrapper(Patient()))
    env.reset(options=env_config)
    return env


def create_action_mask_wrapped_env(env_config):
    env = ActionMaskingWrapper(PartiallyObservableWrapper(Patient()))
    env.reset(options=env_config)
    return env


def create_action_mask_wrapped_ba_env(env_config):
    env = BayesAdaptiveWrapper(
        ActionMaskingWrapper(PartiallyObservableWrapper(Patient()))
    )
    env.reset(options=env_config)
    return env


if not hasattr(module, "param_space"):
    raise ValueError(
        "Your Python file must contain a 'param_space' variable "
        "that is an AlgorithmConfig object."
    )

env_param_space = getattr(module, "param_space")
stop = {}

if args.stop_iters is not None:
    # Add the parameter to the dictionary
    stop["training_iteration"] = args.stop_iters

if args.stop_timesteps is not None:
    # Add the parameter to the dictionary
    stop["timesteps_total"] = args.stop_timesteps

if args.stop_reward is not None:
    # Add the parameter to the dictionary
    stop["episode_reward_mean"] = args.stop_reward


param_space = dict(env_param_space["config"])
masking = False
if "model" in param_space:
    if "custom_model" in param_space["model"]:
        if (
            param_space["model"]["custom_model"].is_valid("LSTMTorchActionMaskModel")
            or param_space["model"]["custom_model"].is_valid("TorchActionMaskModel")
            or param_space["model"]["custom_model"].is_valid("DQNTorchActionMaskModel")
            or param_space["model"]["custom_model"].is_valid(
                "R2D2LSTMTorchActionMaskModel"
            )
        ):
            register_env("bapomdp_env_v0", create_action_mask_wrapped_ba_env)
            register_env("pomdp_env_v0", create_action_mask_wrapped_env)
            # Register also the possible action masking models in the catalog
            ModelCatalog.register_custom_model(
                "TorchActionMaskModel", TorchActionMaskModel
            )
            ModelCatalog.register_custom_model(
                "LSTMTorchActionMaskModel", LSTMTorchActionMaskModel
            )
            ModelCatalog.register_custom_model(
                "DQNTorchActionMaskModel", DQNTorchActionMaskModel
            )
            ModelCatalog.register_custom_model(
                "R2D2LSTMTorchActionMaskModel", R2D2LSTMTorchActionMaskModel
            )
            print("Masking activated")
            masking = True
if not (masking):
    print("Masking non activated")
    register_env("pomdp_env_v0", create_wrapped_env)
    register_env("bapomdp_env_v0", create_wrapped_ba_env)


param_space["framework"] = env_param_space["framework"]
if args.model == "pomdp":
    print("Tunning the model pomdp")
    param_space["env"] = "pomdp_env_v0"
elif args.model == "bapomdp":
    print("Tunning the model bapomdp")
    param_space["env"] = "bapomdp_env_v0"


# Create Tuner
ray.init()

tuner = tune.Tuner(
    env_param_space["run"],
    # Add some parameters to tune
    param_space=param_space,
    # Specify tuning behavior
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean", mode="max", num_samples=args.num_samples
    ),
    run_config=air.RunConfig(stop=stop),
)

result_grid = tuner.fit()
print("type(result_grid)", type(result_grid))
print("result_grid", result_grid)
print(
    "result_grid._experiment_analysis.trials", result_grid._experiment_analysis.trials
)
print("Trial evaluated params")

for trial in result_grid._experiment_analysis.trials:
    print(trial.evaluated_params)

num_results = len(result_grid)

# Check if there have been errors
if result_grid.errors:
    print("At least one trial failed.")

# get best trial
best_trial = result_grid._experiment_analysis.get_best_trial()
# Get the best result
best_result = result_grid.get_best_result()
print("best_result", best_result)
print("type(best_result)", type(best_result))
# Note use best_trial.config not best_trial.evaluated_params
# because otherwise parametres specified using sample_from are filtered
# Print best trial config
print("best_trial.config", best_trial.config)
best_config = best_trial.config
# Print best hyperparameters
print("best_trial.evaluated_params", best_trial.evaluated_params)
# best_config = best_trial.evaluated_params
best_config["framework"] = "torch"

experiment_name = str(env_param_space["env"]) + "_" + env_param_space["run"]
# Create experiment from configuration with best hyperparameters
experiments = {
    experiment_name: {
        "env": env_param_space["env"],
        "run": env_param_space["run"],
        # "local_dir": local_dir,
        # "config": unflatten_dict(best_config),
        "config": best_config,
    }
}

print(experiments)

# Store experiment with best hyperparameters
output_file_name = args.output_file
with open(output_file_name, "w") as file:
    yaml.dump(experiments, file)

ray.shutdown()

# Delete all logs resulting from this tunning session
# for result in result_grid:
#    dirpath = Path(result.log_dir)
#    if dirpath.exists() and dirpath.is_dir():
#        shutil.rmtree(dirpath)

# Print best result log_dir
# print("best_result.log_dir: ", best_result.log_dir)
# And the best checkpoint
# best_checkpoint = best_result.checkpoint
# And the best metrics
# best_metric = best_result.metrics
# print('best_result.metrics', best_result.metrics)
# Get the dataframe for further analysis
# results_db = result_grid.get_dataframe()
# print('results.head()', results_db.head())
# print('results.describe()',results_db.describe())
# print('results.columns', results_db.columns)
# Loading experiment results from a directory
# We can retrieve the ResultGrid from a restored Tuner, passing in the experiment directory,
# experiment_path = f"{local_dir}/{exp_name}"
# print(f"Loading results from {experiment_path}...")
# restored_tuner = tune.Tuner.restore(experiment_path)
# result_grid = restored_tuner.get_results()

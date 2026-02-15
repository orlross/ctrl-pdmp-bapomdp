#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : Meritxell Vinyals

#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : Meritxell Vinyals

import argparse
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
from training.action_mask_model import (
    DQNTorchActionMaskModel,
    R2D2LSTMTorchActionMaskModel,
    TorchActionMaskModel,
    LSTMTorchActionMaskModel,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-file",
    type=str,
    help="Path to the yaml file containing the configuration of evaluate",
    required=True,
)

parser.add_argument("--stop-iters", type=int, help="Number of iterations to train.")
parser.add_argument("--stop-timesteps", type=int, help="Number of timesteps to train.")
parser.add_argument("--stop-reward", type=float, help="Reward at which we stop.")
parser.add_argument("--stop-samples", type=int, help="Number of samples used for training at which we stop.")
parser.add_argument(
    "--num-samples",
    type=int,
    default=3,
    help="Number of times that we run with the same hyperparameter with different seeds.",
)
parser.add_argument(
    "--evaluation-interval",
    type=int,
    default=1,
    help="Interval of training iterations for which we will run evaluation. E.g. we will run the evaluation iteration every [evaluation_interval] training training_iterations",
)
parser.add_argument(
    "--output-folder",
    type=str,
    default="./env/results/",
    help="Name/path of the folder to save results",
)

args = parser.parse_args()
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

if args.stop_samples is not None:
    # Add the parameter to the dictionary
    stop["num_env_steps_sampled"] = args.stop_samples

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


# Load the experiment to run with the tunned parameters
# Function copied from
# Inspired from already implemented rllib train --file .yaml functionality
def load_experiments_from_file(
    config_file: str, checkpoint_config: Optional[dict] = None
) -> dict:
    """Load experiments from a YAML file
    Args:
        config_file: The yaml file to be used as experiment definition.
            Must only contain exactly one experiment.
        checkpoint_config: An optional checkpoint config to add to the returned
            experiments dict.
    Returns:
        The experiments dict ready to be passed into `tune.run_experiments()`.
    """

    # Yaml file.
    with open(config_file) as f:
        experiments = yaml.safe_load(f)

    for key, val in experiments.items():
        experiments[key]["checkpoint_config"] = checkpoint_config or {}

    return experiments


experiments = load_experiments_from_file(args.config_file)
print("experiments", experiments)
exp_name = list(experiments.keys())[0]
print("exp_name", exp_name)
algo = experiments[exp_name]["run"]
print("algo", algo)
print("experiments[exp_name][config]", experiments[exp_name]["config"])

# config = AlgorithmConfig().from_dict(experiments[exp_name]["config"]).environment(experiments[exp_name]["env"])
config = dict(experiments[exp_name]["config"])
masking = False
if "model" in config:
    if "custom_model" in config["model"]:
        if (
            (config["model"]["custom_model"] == "LSTMTorchActionMaskModel")
            or (config["model"]["custom_model"] == "TorchActionMaskModel")
            or (config["model"]["custom_model"] == "DQNTorchActionMaskModel")
            or (config["model"]["custom_model"] == "R2D2LSTMTorchActionMaskModel")
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


config["env"] = experiments[exp_name]["env"]
if "env_config" in experiments[exp_name]:
    config["env_config"] = experiments[exp_name]["env_config"]
config["evaluation_interval"] = args.evaluation_interval
config["always_attach_evaluation_results"] = True
config["metrics_num_episodes_for_smoothing"] = 1
config["evaluation_num_workers"] = 1
print("config", config)


# Create Tuner
ray.init()
results_folder = os.path.abspath(args.output_folder)
tuner = tune.Tuner(
    algo,
    # Add some parameters to tune
    param_space=config,
    # Specify tuning behavior
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean", mode="max", num_samples=args.num_samples
    ),
    run_config=air.RunConfig(stop=stop, local_dir=results_folder, name=exp_name),
)


# Run tuning job
# Tuner.fit() generates a ResultGrid object. This object contains metrics, results, and checkpoints of each trial.
result_grid = tuner.fit()

num_results = len(result_grid)

# Check if there have been errors
if result_grid.errors:
    print("At least one trial failed.")

# Get the best result
best_result = result_grid.get_best_result()
# Print best hyperparameters
print("best hyperparameters: ", result_grid.get_best_result().config)
# Print best result log_dir
# print("best_result.log_dir: ", best_result.log_dir)
# And the best checkpoint
best_checkpoint = best_result.checkpoint
# And the best metrics
best_metric = best_result.metrics
# Get the dataframe for further analysis
results_db = result_grid.get_dataframe()
print("results.head()", results_db.head())
print("results.describe()", results_db.describe())
print("results.columns", results_db.columns)

ray.shutdown()
# Loading experiment results from a directory
# We can retrieve the ResultGrid from a restored Tuner, passing in the experiment directory,
# experiment_path = f"{local_dir}/{exp_name}"
# print(f"Loading results from {experiment_path}...")
# restored_tuner = tune.Tuner.restore(experiment_path)
# result_grid = restored_tuner.get_results()


#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : Orlane Rossini

#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
# ----------------------------------------------------------------------------
# Created By  : Orlane Rossini

import argparse
import gymnasium
import numpy as np
import os
import random as rd
from gymnasium.envs.registration import register
from scipy.stats import norm
from env.full_pdmp import Patient
from training.action_mask_model import (
    TorchActionMaskModel,
    LSTMTorchActionMaskModel,
)
from env.wrappers import (
    ActionMaskingWrapper,
    PartiallyObservableWrapper,
    BayesAdaptiveWrapper,
)

from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from typing import Optional
import ray
from ray import air, tune
import yaml
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


# POLICIES DEFINITION
# The policy function to use in simulate_data
def hasard(obs: np.ndarray) -> int:
    """Generate a random action according to the obs
    Returns:
        The decision related to the observation.
    """
    m, k, zeta, u, t, tau = obs
    if zeta >= 5:
        if t + 60 <= 2400:
            return (
                rd.choice([3, 4, 5]) if 0 < tau < 45 else rd.choice([0, 1, 2, 3, 4, 5])
            )
        elif t + 30 <= 2400:
            return rd.choice([3, 4]) if 0 < tau < 45 else rd.choice([0, 1, 3, 4])
        else:
            return rd.choice([3]) if 0 < tau < 45 else rd.choice([0, 3])
    else:
        if tau == 0 or tau > 45:
            if t + 60 <= 2400:
                return rd.choice([0, 1, 2])
            return rd.choice([0, 1]) if t + 30 <= 2400 else rd.choice([0])
        else:
            if t + 60 <= 2400:
                return rd.choice([3, 4, 5])
            return rd.choice([3, 4]) if t + 30 <= 2400 else rd.choice([3])


def thresh(obs: np.ndarray) -> int:
    """Generate an action according to the obs
    Returns:
        The decision related to the observation.
    """
    print("on observe obs")
    tau, k, t, y, z = obs

    if y <= 5 and tau == 0:
        if t + 60 <= 2400:
            return 2
        else:
            return 1 if t + 30 <= 2400 else 0

    elif 5 < y < 30:
        if tau == 0:
            return 4 if t + 30 <= 2400 else 3
        else:
            return 3

    elif y >= 30:
        if tau <= 45:
            return 3
        elif t + 60 <= 2400:
            return 2
        else:
            return 1 if t + 30 <= 2400 else 0
    else:
        if tau <= 60:
            return 3
        else:
            return 0

    if 5 <= zeta_hat <= 25 and (tau == 0):
        return 4 if t + 30 <= 2400 else 3

    if 5 <= zeta_hat <= 25 and (tau > 0):
        return 3

    if zeta_hat >= 25 and (0 < tau < 45):
        return 3

    if zeta_hat >= 25 and (tau == 0 or tau > 45):
        if t + 60 <= 2400:
            return 2
        else:
            return 1 if t + 30 <= 2400 else 0

    return 0


def memory(obs: np.ndarray, old_obs: np.ndarray) -> int:
    """Generate an action according to the obs
    Returns:
        The decision related to the observation.
    """
    old_tau, old_t, old_y, old_z = old_obs
    tau, t, y, z = obs
    eps = norm.rvs(size=1, loc=0, scale=1)[0]

    zeta_hat = y - eps
    old_zeta_hat = old_y - eps

    if zeta_hat < 5 and (tau == 0 or tau > 45):
        if t + 60 <= 2400:
            return 2
        else:
            return 1 if t + 30 <= 2400 else 0

    if 5 < zeta_hat < 10 and (tau == 0):
        return 0

    if 5 < zeta_hat < 10 and (tau > 0):
        return 3

    if 10 <= zeta_hat <= 30 and (tau == 0):
        return 4 if t + 30 <= 2400 else 3

    if 10 <= zeta_hat <= 30 and (tau > 0):
        return 3

    if zeta_hat >= 30 and (0 < tau < 45):
        return 3

    if zeta_hat >= 30 and (tau == 0 or tau > 45):
        if t + 60 <= 2400:
            return 2
        else:
            return 1 if t + 30 <= 2400 else 0

    return 0


def inactive(obs: np.ndarray) -> int:
    """Generate the action that do not treat and planning the next visit in 60days
    Returns:
        The decision related to the observation.
    """
    m, k, zeta, u, t, tau = obs
    if t + 60 <= 2400:
        return 2
    else:
        return 1 if t + 30 <= 2400 else 0

def abusive() -> int:
    """Generate the action treat and schedule a visit evry 15 days
    Returns:
        The decision related to the observation.
    """
    m,k,zeta,u,t,tau = obs 
    if t + 60 <= 2400: 
        return 5
    if t + 30 <= 2400: 
        return 4 
    return 3


def perfect(obs: np.ndarray) -> int:
    """Generate the perfect action according to the obs
    Args:
        obs: The current observation.
    Returns:
        The decision related to the observation.
    """
    m, k, zeta, u, t, tau = obs

    if m == 0:
        if 0 < tau < 45:
            return 3
        else:
            if t + 60 <= 2400:
                return 2
            else:
                return 1 if t + 30 <= 2400 else 0

    if m == 1:
        if zeta < 5 and tau == 0:
            return 0
        elif zeta >= 5 and t + 30 <= 2400:
            return 5 if tau == 0 and t + 60 <= 2400 else 4
        else:
            return 3

    if m == 2:
        if tau == 0 or tau > 45:
            if t + 60 <= 2400:
                return 2
            return 1 if t + 30 <= 2400 else 0
        else:
            return 3


# Register your environment
register(
    id="env/Patient",
    entry_point="env.full_pdmp:Patient",
)

# Argument to parse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-file", type=str, help="Path to the csv file to store the simulated data"
)
parser.add_argument(
    "--model",
    type=str,
    choices=["pdmp", "pomdp", "bapomdp"],
    help="Environment model. Options: pdmp pomdp bapomdp",
)
parser.add_argument(
    "--policy",
    type=str,
    choices=["perfect", "alea", "inactive", "thresh", "memory", "abusive"],
    help="Name of the fixed policy type. Options: perfect alea inactive thresh memory",
)
parser.add_argument("--policy-path", type=str, help="Path to the policy checkpoint")
parser.add_argument(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of times that we run the environment.",
)

args = parser.parse_args()

if args.policy is None and args.policy_path is None:
    parser.error("You should specify either a --policy or a --policy-path.")

masking = False
recurrent = False
if args.policy_path is not None:
    ModelCatalog.register_custom_model("TorchActionMaskModel", TorchActionMaskModel)
    ModelCatalog.register_custom_model(
        "LSTMTorchActionMaskModel", LSTMTorchActionMaskModel
    )
    policy_loaded = Policy.from_checkpoint(os.path.abspath(args.policy_path))

    if policy_loaded.observation_space.shape[0] == 5:
        masking = False
    elif policy_loaded.observation_space.shape[0] == 11:
        masking = True
    else:
        raise Exception("Observation space shape not supported")

    print("policy_loaded.is_recurrent()", policy_loaded.is_recurrent())
    recurrent = policy_loaded.is_recurrent()
    if recurrent:
        init_state = state = policy_loaded.model.get_initial_state()
        print("recurrent init_state", init_state)


env = gymnasium.make("env/Patient")
if args.model == "pdmp":
    print("Load PDMP env")
    env = gymnasium.make("env/Patient")
elif args.model == "pomdp":
    if masking:
        print("Load POMDP-MASKING env")
        env = ActionMaskingWrapper(
            PartiallyObservableWrapper(gymnasium.make("env/Patient"))
        )
    else:
        print("Load POMDP env")
        env = PartiallyObservableWrapper(gymnasium.make("env/Patient"))
elif args.model == "bapomdp":
    if masking:
        print("Load BAPOMDP-MASKING env")
        env = BayesAdaptiveWrapper(
            ActionMaskingWrapper(
                PartiallyObservableWrapper(gymnasium.make("env/Patient"))
            )
        )
    else:
        print("Load BAPOMDP env")
        env = BayesAdaptiveWrapper(
            PartiallyObservableWrapper(gymnasium.make("env/Patient"))
        )
else:
    raise Exception("Model not supported")


all_action = {
    0: {"ell": 0, "r": 15},
    1: {"ell": 0, "r": 30},
    2: {"ell": 0, "r": 60},
    3: {"ell": 1, "r": 15},
    4: {"ell": 1, "r": 30},
    5: {"ell": 1, "r": 60},
}


#obs, info = env.reset(options={"pomdp": "pomdp_fixed_v", "v1": -3.5})
obs, info = env.reset()
old_obs = obs
id_traj = 0
traj_info = np.zeros((1, 20))

for i in range(args.num_samples):
    if i % 1000 == 0:
        print(i)
    for n in range(10**6):
        # Get the action
        if args.policy == "perfect":
            action = perfect(obs)
        elif args.policy == "alea":
            action = hasard(obs)
        elif args.policy == "abusive":
            action = abusive()
        elif args.policy == "inactive":
            action = inactive(obs)
        elif args.policy == "thresh":
            action = thresh(obs)
        elif args.policy == "memory":
            action = memory(obs, old_obs)
        elif args.policy_path is not None:
            if recurrent:
                action, state, _ = policy_loaded.compute_single_action(obs, state)
            else:
                action, _, _ = policy_loaded.compute_single_action(obs)
        else:
            raise Exception("Policy " + args.policy + "  not supported yet")

        # Move forward in the environment according to the chosen action
        old_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)
        only_obs = obs["obs"] if masking else obs

        if args.model == "pdmp":
            nb_ttmt = only_obs[1]
            tps = only_obs[4]
            m = only_obs[0]
            zeta = only_obs[2]
            noise = 0
            death_ind = 0
            tau = only_obs[5]
        else:
            nb_ttmt = info["real_state"][1]
            tps = only_obs[2]
            m = info["real_state"][0]
            zeta = info["real_state"][2]
            noise = only_obs[3]
            death_ind = only_obs[4]
            tau = only_obs[0]

        # if model == "bapomdp-masking" or model == "bapomdp":
        #    alpha = only_obs[5]
        #    beta = only_obs[6]
        #    kappa = only_obs[7]
        #    nu = only_obs[8]
        # else:
        #   alpha = kappa = beta = nu = 0
        alpha = kappa = beta = nu = 0
        # forbidden = info['costs'] if model == 'pomdp' else 0

        #print("action ", action, " next r = ", all_action[action]["r"], " next ell = ", all_action[action]["ell"])

        traj_info = np.vstack(
            [
                traj_info,
                [
                    -reward,
                    info["costs"]["visit_cost"],
                    info["costs"]["chemotherapy_cost"],
                    info["costs"]["death"],
                    0,
                    nb_ttmt,
                    all_action[action]["r"],
                    all_action[action]["ell"],
                    tps,
                    m,
                    zeta,
                    noise,
                    death_ind,
                    tau,
                    alpha,
                    kappa,
                    beta,
                    nu,
                    id_traj,
                    n,
                ],
            ]
        )

        #print(traj_info[])
        # Is the episode `done`? -> Reset.
        if terminated or truncated:  # If episode is done => reset
            id_traj += 1
            obs, info = env.reset()
            if recurrent:
                state = init_state
            break
        # prev_a = init_prev_a
        # prev_r = init_prev_r
        # Episode is still ongoing.
        # else:
        # if init_prev_a is not None:
        # prev_a = a
        # if init_prev_r is not None:
        # prev_r = reward

        # End the loop if the episode if over
        # if terminated or truncated:
        # id_traj += 1
        # obs, info = env.reset()
        # state = init_state
        # break

if args.output_file is not None:
    save_path_traj = args.output_file
else:
    if args.policy is not None:
        # The name of the csv
        filename_info = args.model + "_" + args.policy + "_v0.csv"
    else:
        filename_info = args.model + "_rllib_v0.csv"
    # Relative path to save as CSV in dash_app/data
    print("Save the final dataset")
    save_path_traj = os.path.join("./simulations/data", filename_info)

# Delete the first line of traj_info
traj_info = traj_info[1:]

# Save in CSV
np.savetxt(save_path_traj, traj_info, delimiter=",", comments="", fmt="%1.2f")


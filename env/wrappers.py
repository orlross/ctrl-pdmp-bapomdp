# This module contains three wrappers for the Patient class:
# - ActionMaskingWrapper
# - PartiallyObservableWrapper
# - BayesAdaptiveWrapper
# They should be applicable in combination abd in any order.

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Discrete

# from scipy.stats import norm
# from gymnasium.spaces import Discrete
# import math


class ActionMaskingWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        """
        Wrapper to add action masking to a (potentially wrapped) Patient
        environment.
        This wrapper modifies the methods reset() and step() of a Patient class
        environment instance in such a way that they return observations
        produced by the unwrapped environment, plus a set of valid actions
        for this observation.

        :param gymnasium.Env env: The gymnasium environment to wrap.
        """
        super().__init__(env)
        self._skip_env_checking = True
        self.env = env

        self.observation_space = spaces.Dict(
            {
                "obs": self.observation_space,
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(self.env.action_space.n,), dtype=np.int8
                ),
            }
        )
        self.action_space = (
            self.env.action_space
        )  # Use the action space from the inner environment
        # Masking only works for Discrete actions.
        assert isinstance(self.action_space, Discrete)

    def reset(self, seed=None, options=None):
        # Uses the environment's reset function and returns the velid actions.
        obs, info = self.env.reset(options=options)
        obs = self._fix_action_mask(obs)
        info["action_mask"] = obs["action_mask"]
        return obs, info

    def step(self, act):
        """
        Run one timestep of the environment's dynamics using the agent actions.

        The action mask attached to the returned observation is computed and
        returned as well.

        :param act: An action provided by the agent to update environment state.
        :return: action_mask (including observation), reward, terminated,
        truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(act)
        action_mask = self._fix_action_mask(obs)
        info["action_mask"] = action_mask["action_mask"]
        return action_mask, reward, terminated, truncated, info

    def _fix_action_mask(self, obs):
        """
        :param obs: An observation.
        :return: A dictionary including a field "action_mask" specifying the
        valis actions for the current observation.
        """
        # There are two cases, depending on whether the environment is or not
        # observable:
        if len(obs) == 6:  # Fully observed case
            _, _, _, _, t, tau = obs
        else:  # Partial observation: tau and t are not in the same place.
            tau, _, t, _, _ = obs

        # Valid actions should treat for at least 45 days and not after until
        # after the horizon. Furthermore, they should not treat at t=0.
        # print(f"tau={tau} and t={t}")
        self.valid_actions = np.ones((6,), dtype=int)
        actions_dic = self.env.all_action
        for a in range(6):
            if actions_dic[a]["ell"] == 1:
                # No treatment at t=0 or that exceeds the horizon.
                if t == 0 or t + actions_dic[a]["r"] > self.env.horizon:
                    self.valid_actions[a] = 0
            else:  # Actions should last for at least 45 days.
                if tau < 45 and t > 0:
                    self.valid_actions[a] = 0

        return {"obs": obs, "action_mask": self.valid_actions}

class PartiallyObservableWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        """
        Wrapper to add partial observability to a (potentially wrapped) Patient
        environment.
        This wrapper modifies the methods reset() and step() of a Patient class
        environment instance in such a way that they return modifications of the
        observations produced by the unwrapped environment.

        :param gymnasium.Env env: The gymnasium environment to wrap.
        """
        super().__init__(env)
        self._skip_env_checking = True
        self.env = env

        # Construct observations (tau, k, t, y, z) from full state
        # (m, k, zeta, u, t, tau). Note that z is the observation of death.
        low = np.array([0, 0, 0, -5, 0])
        t_hor = self.env.horizon + 500
        high = np.array([t_hor, t_hor, t_hor, 45, 1])
        self.observation_space = spaces.Box(low, high, shape=(5,), dtype=np.float32)
        self.action_space = self.env.action_space
        # Use the action space from the inner environment

        # Random number generator
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial internal state, returning an
        initial observation and information.

        :param optional int seed: The seed that is used to initialize the
        environment's PRNG.
        :param optional dict options: Additional information to specify how the
        environment is reset.

        :return: observation, info
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if options is None:
            options = {}

        else:
            options["pomdp"] = options.get("pomdp", "pomdp")
            if options["pomdp"] == "pomdp_prior":
                options["alpha"] = options.get("alpha", -3.51)
                options["beta"] = options.get("beta", 36.001)
                options["kappa"] = options.get("kappa", 19.01)
                options["nu"] = options.get("nu", 1.80)

                # Initialise the current belief of a patient
                self._alpha = options["alpha"]
                self._beta = options["beta"]
                self._kappa = options["kappa"]
                self._nu = options["nu"]

                # Sample sigma and mu for the trajectory
                # Careful, here tau is log-normal precision not tau parameter
                # it should be changed in future !
                tau = self._rng.gamma(self._kappa, self._nu)
                options["sigma"] = 1 / tau
                options["mu"] = self._rng.normal(self._alpha, np.sqrt(1 / (self._beta * tau)))

            elif options["pomdp"] == "pomdp_fixed_v":
                options["sigma"] = 0
                options["mu"] = options.get("v1", -3.5)

        obs, info = self.env.reset(options=options)
        if isinstance(self.env, ActionMaskingWrapper):
            obs["obs"] = self._partial_obs(obs["obs"])
            return obs, info
        else:
            return self._partial_obs(obs), info

    def step(self, act):
        """
        Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached, it is necessary to call :meth:`
        reset` to reset this environment's state for the next episode.

        :param act: An action provided by the agent to update environment state.
        :return: observation, reward, terminated, truncated, info

        """
        new_obs, reward, terminated, truncated, new_info = self.env.step(act)
        if isinstance(self.env, ActionMaskingWrapper):
            partial_obs = dict()
            nobs = new_obs["obs"]
            partial_obs["action_mask"] = new_obs["action_mask"]
            partial_obs["obs"] = self._partial_obs(nobs)
            new_info["noise"] = partial_obs["obs"][3] - nobs[2]  # eps
        else:
            nobs = new_obs
            partial_obs = self._partial_obs(nobs)
            new_info["noise"] = partial_obs[3] - nobs[2]  # eps
        new_info["real_state"] = (nobs[0], nobs[1], nobs[2], nobs[3])
        return partial_obs, reward, terminated, truncated, new_info

    def _partial_obs(self, obs):
        """
        :param obs: perfect observation (m, k, zeta, u, t, tau)
        :return: partial observation (tau, k, t, y, z)
        """
        # Marker observation noise
        eps = self._rng.normal(size=None, loc=0, scale=1)

        self._y = obs[2] + eps
        self._t = obs[4]
        self._tau = obs[5]
        self._k = obs[1]
        self._z = 1 if obs[0] == 3 else 0
        return np.array([self._tau, self._k, self._t, self._y, self._z])

    def set_parameters(self, sigma, mu):
        self.env.set_parameters(sigma=sigma, mu=mu)

    # self.env.set_parameters


class BayesAdaptiveWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        """
        Wrapper to add bayes adaptive behaviour to a (partial observation
        wrapped) Patient environment.
        This wrapper modifies the methods reset() and step() of a Patient class
        environment instance in such a way that they handle dynamics parameter
        updating in a partially observed environment.
        Note that the environment to wrapp maybe wrapped with action masking in
        addition to partial observation (and in any order). The only requirement
        is that it is partially observed.

        :param gymnasium.Env env: The gymnasium environment to wrap.
        """
        super().__init__(env)
        self._skip_env_checking = True
        if isinstance(env, ActionMaskingWrapper):
            assert isinstance(
                env.env, PartiallyObservableWrapper
            ), "Only applies to a partially observed environment"
        else:
            assert isinstance(
                env, PartiallyObservableWrapper
            ), "Only applies to a partially observed environment"
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # Random number generator
        self._rng = np.random.default_rng(self.seed)
        self.old_info = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial internal state, returning an initial observation and information.

        :param optional int seed: The seed that is used to initialize the environment's PRNG.
        :param optional dict options: Additional information to specify how the environment is reset.

        :return: observation, info

        """
        # We need the following line to seed self._rng
        super().reset(seed=seed)
        # Below is hard coding the options values.
        # This should be modified, but I do not know why, options['m'] and co
        # are not known.
        if options is None:
            options = {}
        options["m"] = options.get("m", 0)
        options["k"] = options.get("k", 1)
        options["zeta"] = options.get("zeta", 1)
        options["u"] = options.get("u", 0)
        options["t"] = options.get("t", 0)
        options["tau"] = options.get("tau", 0)
        options["alpha"] = options.get("alpha", -3.51)
        options["beta"] = options.get("beta", 36.001)
        options["kappa"] = options.get("kappa", 19.01)
        options["nu"] = options.get("nu", 1.80)

        # Initialise the current belief of a patient
        self._alpha = options["alpha"]
        self._beta = options["beta"]
        self._kappa = options["kappa"]
        self._nu = options["nu"]

        # Sample sigma and mu for the trajectory
        # Careful, here tau is log-normal precision not tau parameter
        # it should be changed in future !
        tau = self._rng.gamma(self._kappa, self._nu)
        options["sigma"] = 1 / tau
        options["mu"] = self._rng.normal(self._alpha, np.sqrt(1 / (self._beta * tau)))

        # Call the right pomdp wrapper
        # options["pomdp"] = "bapomdp"

        obs, info = self.env.reset(options=options)
        self.old_state = options["m"], options["k"], options["zeta"], options["u"]

        return obs, info

    def step(self, act):
        """
        Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached, it is necessary to call :meth:`reset` to reset this environment's state
        for the next episode.

        :param int action: An action provided by the agent to update environment state.
        :return: observation, reward, terminated, truncated, info

        """
        new_obs, reward, terminated, truncated, new_info = self.env.step(
            act
        )  # Call with one argument
        # Update hyperparameters for the next trajectory
        update = self._update_prior(new_info["real_state"],new_info["slope"], act)
        self.old_state = new_info["real_state"]
        if update:
            tau = self._rng.gamma(self._kappa, self._nu)
            sigma = 1 / tau
            mu = self._rng.normal(self._alpha, np.sqrt(1 / (self._beta * tau)))
            self.env.set_parameters(sigma=sigma, mu=mu)
        new_info["theta"] = {
            "alpha": self._alpha,
            "beta": self._beta,
            "kappa": self._kappa,
            "nu": self._nu,
        }
        return new_obs, reward, terminated, truncated, new_info

    def _update_prior(self, new_state,v2, act):
        """
        Update the hyperparameters from (s,a,s') transition

        :param int new_info: A current state.
        :param int act: An action provided by the agent to update environment state.

        """
        v1_hat = self._get_v1_hat(new_state,v2, act)
        # If no v1_hat estimator available or hyperparameters are already been updated
        if v1_hat is None:
            return False
        else:
            print("v1_hat: ", v1_hat)
        self._alpha = (np.log(v1_hat) / self._beta) + (
            (self._beta - 1) / self._beta
        ) * self._alpha
        self._beta = self._beta + 1
        self._kappa = self._kappa + 1 / 2
        self._nu = self._nu + (self._beta * (np.log(v1_hat) - self._alpha) ** 2) / (
            2 * (self._beta + 1)
        )
        return True

    def _get_v1_hat(self, new_state,v2, act):
        """
        Get an estimator of v1 from (s,a,s') transition

        :param int new_info: A current state.
        :param int act: An action provided by the agent to update environment state.

        """
        ell = self.env.all_action[act]["ell"]
        r = self.env.all_action[act]["r"]
        m, k, zeta, _ = self.old_state
        new_m, _, new_zeta, new_u = new_state

        if ell == 0:
            if m == 0:
                if new_m == 1:
                    return (1 / new_u) * np.log(new_zeta / zeta)
            elif m == 1:
                if new_m == 1:
                    return (1 / r) * np.log(new_zeta / zeta)
                elif new_m == 2:
                    return (1 / (r - new_u)) * (np.log(new_zeta / zeta) - v2 * new_u)
        else:
            if m == 1:
                if new_m == 0:
                    return (k / (r - new_u)) * np.log(zeta / new_zeta)
                elif new_m == 1:
                    return (k / r) * np.log(zeta / new_zeta)
                elif new_m == 2:
                    return (k / (r - new_u)) * (np.log(zeta / new_zeta) - v2 * new_u)
        return None

    #def set_parameters(self, sigma, mu):
    #    self.sigma = sigma
    #    self.mu = mu
    #    self._v1 = self._rng.lognormal(mean=mu, sigma=np.sqrt(sigma))

        # self.env.set_parameters

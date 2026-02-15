#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

import gymnasium
from gymnasium import spaces
import numpy as np
# from scipy.stats import uniform
# import math


class Patient(gymnasium.Env):
    """
    ## Description
    This environment corresponds to a medical treatment optimization problem.

    The stochastic control problem is modelled by a controlled MDP (Markov
    Decision Process).
    This MDP is derived from a PDMP (Piecewise-Deterministic Markov Process).

    ## Action Spaces
    There 2 discrete actions available:
    - ell: the choice of treatment (0: nothing, 1: chemotherapy)
    - r: the date of the next visit (1: 15 days, 2: 30 days, 3: 60 days)

    ## Observation Space
    The state is a 5-dimensional vector: the mode, the number of treatment,
    the biomarker, the time since last jump and
    the time since the beginning, the since a treatment is applied.

    ## Rewards
    After every step a cost is granted (the cost is transformed into a reward
    at the end) to match with RLLIB algorithms.
    """

    def __init__(self, config=None):
        """
        Initializes a controlled PDMP.
        """
        # Builds or complete the config Dictionary
        if config is None:
            config = dict()
        ######################################################################
        # Starts with the general parameters of the simulation
        ######################################################################
        if "z0" not in config:
            config["z0"] = 1
        if "D" not in config:
            config["D"] = 40
        if "horizon" not in config:
            config["horizon"] = 2400
        if "seed" not in config:
            config["seed"] = None
        if "low" not in config:
            config["low"] = np.array([0, 1, config["z0"], 0, 0, 0])
        if "high" not in config:
            h = config["horizon"]
            config["high"] = np.array([50, 100, config["D"], h, h+500, h+500])
        ######################################################################
        # Continues with the cost parameters
        ######################################################################
        if "cv" not in config:
            config["cv"] = 1
        if "ckappa" not in config:
            config["ckappa"] = 0.1
        if "cd" not in config:
            config["cd"] = 330
        if "beyhorcost" not in config:
            config["beyhorcost"] = 10**6
        ######################################################################
        # Then, the default initial state of a trajectory
        ######################################################################
        if "m" not in config:
            config["m"] = 0
        if "k" not in config:
            config["k"] = 1
        if "zeta" not in config:
            config["zeta"] = 1
        if "u" not in config:
            config["u"] = 0
        if "t" not in config:
            config["t"] = 0
        if "tau" not in config:
            config["tau"] = 0
        if "mu" not in config:
            config["mu"] = -3.5
        if "sigma" not in config:
            config["sigma"] = 0.3
        ######################################################################
        # The model dynamics parameters
        # I do not understand the -next_jump function, especially the
        # probability thresholds 0.4 and 0.8 for epsilon, so I cannot write
        # the config elements (Régis, May, 23).
        # But this will have to be done :-)
        ######################################################################

        self.config = config

        # Initialise global model parameters (common to all patients)
        self.z0 = self.config.get("z0")
        self.D = self.config.get("D")
        self.horizon = self.config.get("horizon")
        self.seed = self.config.get("seed")
        self.low = self.config.get("low")
        self.high = self.config.get("high")

        # Actions are encoded into 1 value, define a dictionary to interpret
        # each value
        self.all_action = {
            0: {"ell": 0, "r": 15},
            1: {"ell": 0, "r": 30},
            2: {"ell": 0, "r": 60},
            3: {"ell": 1, "r": 15},
            4: {"ell": 1, "r": 30},
            5: {"ell": 1, "r": 60},
        }
        # Get the state space (E = E_0 U ... U E_3)
        self.observation_space = spaces.Box(self.low, self.high, shape=(6,),
                                            dtype=np.float32)
        # D = ell x r
        self.action_space = spaces.Discrete(6)
        # Random number generator
        self._rng = np.random.default_rng(seed=self.seed)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached, it is necessary to call :meth:
        `reset` to reset this environment's state for the next episode.

        :param action: The action selected by the agent.
        :return: observation, reward, terminated, truncated, info.

        """
        # Avoid error 
        if self._v1 > 0.0615:
            self._v1 = 0.0615

        # Interpret the action and get a dictionary with (ell, r)
        act = self._convert_act(action)
        old_tau = self._tau
        # From current state and chosen action get the next state
        jump_dict = self._next_step(act)
        # Compute the cost and turn in into a reward
        cost = self._cost(action, old_tau)

        visit_cost, chemo_cost, death_cost, constraint_cost = cost
        reward = -sum(cost)

        # Check if the process is over (patient is dead or horizon is reach)
        terminated = (self._m == 3) or (self._t >= self.horizon)
        truncated = False

        # Save all model information: jump times, costs, ...
        info = {
            "jumps": jump_dict,
            "costs": {
                "visit_cost": visit_cost,
                "chemotherapy_cost": chemo_cost,
                "death": death_cost,
                "constraint": constraint_cost,
            },
            "slope": self._v2,
        }

        obs = np.array([self._m, self._k, self._zeta, self._u, self._t,
                        self._tau])
       
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial internal state, returning an
        initial observation and information.

        :param optional int seed: The seed that is used to initialize the
        environment's PRNG.
        :param optional dict options: Information to specify how the
        environment is reset. This information overrides the config dict.

        :return: observation, info

        """
        if options is None:
            options = {}
        # Initialise the current state of a patient
        self._m = options.get("m", self.config["m"])
        self._k = options.get("k", self.config["k"])
        self._zeta = options.get("zeta", self.config["zeta"])
        self._u = options.get("u", self.config["u"])
        self._t = options.get("t", self.config["t"])
        self._tau = options.get("tau", self.config["tau"])
        self._mu = options.get("mu", self.config["mu"])
        self._sigma = options.get("sigma", self.config["sigma"])

        # Simulates a slope value that will be fixed for the entire trajectory
        self._v1 = self._rng.lognormal(mean=self._mu,
                                       sigma=np.sqrt(self._sigma))
        self._v2 = self._rng.uniform(0.0001, 0.06)

        obs = np.array(
            [
                self._m,
                self._k,
                self._zeta,
                self._u,
                self._t,
                self._tau,
            ]
        )

        return obs, {}

    def _convert_act(self, act):
        """
        Return the current action and compute the days before the next visit

        :param act: The action selected by the agent.

        :return: the actions (treatment and days left before the next visit)

        """
        return self.all_action[act]

    def _cost(self, action, old_tau, params=None):
        """
        Compute the cost according the current state and the choosen action.

        :param int action: An action provided by the agent to update
        environment state.
        :param optional dict params: Additional information to specify
        different cost values.

        :return: the cost
        """
        # Interpret the action and get a dictionary with (ell, r)
        act = self._convert_act(action)
        if params is None:
            params = (self.config["cv"], self.config["ckappa"],
                      self.config["cd"])
        cv, kappa, cd = params
        death_indicator = 1 if self._m == 3 else 0
        chemo_indicator = 1 if act["ell"] == 1 else 0
        beyond_horizon = 1 if self._t > self.horizon else 0
        ttmt_continuity = 1 if act["ell"] == 1 and self._tau > 0 else 0
        change_treatment = 1 if act["ell"] == 0 and 0 < old_tau + act["r"] < 45 else 0
        visit_cost = cv
        death_cost = death_indicator * cd  # * ((self.horizon + 100) - self._t)
        chemo_cost = kappa * act["r"] * chemo_indicator # - 0.5*kappa*ttmt_continuity*act["r"]/2
        constraint = 10**6 * beyond_horizon + 10**6 * change_treatment

        return visit_cost, chemo_cost, death_cost, constraint

    def _next_step(self, action):
        """
        Perform the next step in the environment based on the chosen action.

        :param int action: An action provided by the agent to update
        environment state.

        :return: a dictionary that contains all jump that occurs in a time
        step.
        """

        # If the patient is dead, the time is advanced until the horizon.
        if self._m == 3:
            self._m = 3
            self._k = self._k
            self._zeta = self.D
            self._u = self._u + (self.horizon - self._t)
            self._t = self.horizon
            self._tau = 0
            return {}
        # We test if there is at least one jump between t and t+r
        else:
            r_left, jump_hist = self._all_jumps(
                action, t_max=self._t + action["r"], r_left=action["r"], h=[]
            )
            self._m = self._m
            self._k = self._k + 1 if action["ell"] == 1 and self._tau == 0 \
                else self._k
            self._zeta = self._update_biomarker(
                ell=action["ell"], new_u=self._u + r_left
            )
            self._u = self._u + r_left
            self._t = self._t + r_left
            self._tau = 0 if action["ell"] == 0 else self._tau + r_left

            jump_dict = {}
            for i in range(0, len(jump_hist)):
                jump_dict[i] = jump_hist[i]

            return jump_dict

    def _all_jumps(self, action, t_max, r_left, h=[]):
        """
        Perform all jumps that occurs between 2 visit dates.

        :param int action: An action provided by the agent to update
        environment state.
        :param float t_max: the time which the next visit occurs
        :param float r_left: the time until the next visit
        :param list h: the list of all jumps between two visit dates

        :return: The left time until next visit and the list of all jumps
        between two visit dates.
        """

        # Simulate the next jump and get the mode
        tj, mj = self._next_jump(action)

        # If the next jump occurs before next visit
        if tj < t_max:
            self._zeta = self._update_biomarker(ell=action["ell"], new_u=tj)
            self._tau = 0 if action["ell"] == 0 else self._tau + (tj - self._t)
            self._t = tj
            self._k = self._k
            self._m = mj
            self._u = 0
            self._zeta = self._update_biomarker(ell=action["ell"], new_u=0)

            r_left = t_max - self._t

            h.append((tj, mj))

            # Just a security to not have more than 3 jumps between 2 visits -
            # will be set with jump parameters
            if len(h) >= 3:
                return r_left, h
            return self._all_jumps(action, t_max, r_left, h)

        # If it happens after next visit
        else:
            return r_left, h

    def _next_jump(self, action):
        """
        Simulate a jump time

        :param action: the current action
        :return: the time of the next jump and the new mode
        """
        #
        eps = self._rng.uniform(0, 1)
        if self._m == 0:
            t1 = self._jump(action, new_m=1)
            t2 = self._jump(action, new_m=2)

            # Choose the jump that comes first
            # If ell = 1 (chimio) only jump -> m=2 can occurs.
            tj = min(t1, t2) if action["ell"] != 1 else t2
            mj = np.argmin([t1, t2]) + 1 if action["ell"] != 1 else 2

            if eps < 0.4 or (eps < 0.8 and mj == 2):
                tj = 10**5

            return tj, mj
        if self._m == 1:
            t0 = (self._k / self._v1) * np.log(self._zeta / self.z0) + self._t
            t2 = self._jump(action, new_m=2)
            t3 = (1 / self._v1) * np.log(self.D / self._zeta) + self._t

            # Choose the jump that comes first
            tj = min(t0, t2) if action["ell"] == 1 else min(t2, t3)
            mj = (
                np.argmin([t0, t2]) * 2
                if action["ell"] == 1
                else np.argmin([t2, t3]) + 2
            )

            if eps < 0.5 and mj == 2:
                tj = 10**5

            return tj, mj

        if self._m == 2:
            t3 = (1 / self._v2) * np.log(self.D / self._zeta) + self._t
            return t3, 3

        if self._m == 3:
            return self.horizon, 3

    def _jump(self, action, new_m):
        """
        Simulate a jump time

        :param action: the current action
        :param new_m: the possible next mode

        :return: The time until the next jump into new_m
        """
        y = self._rng.uniform(0, 1)
        if self._m == 0:
            beta = 3.5
            alpha_scale = 0.5
            alpha = alpha_scale * (0.0002 / self._k if new_m == 1 else 0.00001)
            t_jump = (
                (self._u + self._t) ** (beta + 1)
                - (beta + 1) / (alpha**beta) * np.log(y)
            ) ** (1 / (beta + 1)) - self._u
            return t_jump

        elif action["ell"] == 1:
            alphas = [1.116e11, 1.116e10, 1.116e9]
            if self._k < 4:
                alpha = alphas[self._k - 1]
            else:
                alpha = 1.116e7
            beta = -0.8
            t_jump = (
                -self._k
                / (beta * self._v1)
                * np.log(
                    np.exp(-beta * self._t * self._v1 / self._k)
                    + (beta * self._v1)
                    / (self._k * (alpha * self._zeta) ** beta)
                    * np.log(y)
                )
            )
            return t_jump
        else:
            alpha = 1.116e-20 * (self._k / 4)
            beta = 4.5
            t_jump = (
                1
                / (beta * self._v1)
                * np.log(
                    np.exp(beta * self._t * self._v1)
                    - (beta * self._v1) / (alpha * self._zeta) ** beta *
                    np.log(y)
                )
            )

            return t_jump

    def _update_biomarker(self, ell, new_u):
        """
        Update the biomarker based on the chosen treatment and current disease
        regime

        :param int ell: The chosen treatment.
        :param float new_u: Date of biomarker calculus.

        :return: The updated biomarker value.

        """
        if self._m == 0:
            return self.z0
        elif self._m == 3:
            return self.D
        else:
            v = self._v1 if self._m == 1 else self._v2
            if ell == 1 and self._m == 1:
                zeta = self._zeta * np.exp(-(v / self._k) * (new_u - self._u))
                if zeta <= 1:
                    self._m = 0
                    return self.z0
                return zeta
            return self._zeta * np.exp(v * (new_u - self._u))

    def set_parameters(self, sigma, mu):
        self.sigma = sigma
        self.mu = mu
        self._v1 = self._rng.lognormal(mean=mu, sigma=np.sqrt(sigma))

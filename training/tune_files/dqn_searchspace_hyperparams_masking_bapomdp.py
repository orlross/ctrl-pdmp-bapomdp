from ray import tune

param_space = {
    "run": "DQN",
    "env": "bapomdp_env_v0",
    "framework": "torch",
    "config": {
        "_enable_rl_module_api": tune.choice([False]),
        "_enable_learner_api": tune.choice([False]),
        "td_error_loss": tune.choice(["huber", "mse"]),
        "gamma": tune.choice([0.95, 0.97, 0.99, 0.999]),
        "n_step": tune.choice([1, 2, 3]),  # N-step for q-learning
        "double_q": tune.choice(
            [False, True]
        ),  # Number of SGD iterations in each outer loop (n_update_epochs)
        "dueling": tune.choice([False]),
        "replay_buffer_config": {
            "type": tune.choice(
                ["MultiAgentPrioritizedReplayBuffer"]
            ),  # Specify prioritized replay by supplying a buffer type that supports prioritization, for example: MultiAgentPrioritizedReplayBuffer.
            "capacity": tune.choice([50000, 100000, 500000, 1000000]),
            "prioritized_replay_alpha": tune.choice(
                [0.5, 0.6]
            ),  # Alpha parameter controls the degree of
            # prioritization in the buffer. In other words, when a buffer sample has
            # a higher temporal-difference error, with how much more probability
            # should it drawn to use to update the parametrized Q-network. 0.0
            # corresponds to uniform probability. Setting much above 1.0 may quickly
            # result as the sampling distribution could become heavily “pointy” with
            # low entropy
            "prioritized_replay_beta": tune.choice(
                [0.4, 0.5]
            ),  # Beta parameter controls the degree of
            # importance sampling which suppresses the influence of gradient updates
            # from samples that have higher probability of being sampled via alpha
            # parameter and the temporal-difference error.
            "prioritized_replay_eps": tune.choice(
                [1e-6, 3e-6]
            ),  # Epsilon parameter sets the baseline probability
            # for sampling so that when the temporal-difference error of a sample is
            # zero, there is still a chance of drawing the sample.
            "replay_sequence_length": tune.choice(
                [1]
            ),  # The number of continuous environment steps to replay at once. This may be set to greater than 1 to support recurrent models
            "worker_side_prioritization": False,  # Whether to compute priorities on workers
        },
        "exploration_config": {
            # "type": tune.choice(["EpsilonGreedy", "SoftQ"]),# The Exploration class to use. In the simplest case, this is the name
            "type": tune.choice(["EpsilonGreedy"]),
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            "epsilon_timesteps": tune.choice([2, 10000, 50000, 100000, 200000]),
            "final_epsilon": tune.choice([0.0, 0.01, 0.02]),
            "initial_epsilon": tune.choice([0.9, 1.0, 1.5]),
            # "temperature": tune.choice([0.5, 1.0]),
        },
        "num_atoms": tune.choice(
            [1]
        ),  # Number of atoms for representing the distribution of return. When this is greater than 1, distributional Q-learning is used.
        "v_min": tune.choice([-60000, -50000, -40000]),  # Minimum value estimation
        "v_max": tune.choice([-500, -10, 0, 10]),  # Maximum value estimation
        "train_batch_size": tune.choice([32, 65, 256, 512, 1024, 2048, 4096]),
        "lr": tune.choice([3e-05, 0.0001, 0.0003, 0.001]),
        "model": {
            "custom_model": tune.choice(["TorchActionMaskModel"]),
            "fcnet_hiddens": tune.choice([[32], [32, 32], [64], [64, 64]]),
            "fcnet_activation": tune.choice(["linear", "relu", "tanh"]),
        },
        "observation_filter": tune.choice(["NoFilter"]),
        "noisy": tune.choice([False]),
        "sigma0": tune.choice(
            [0.5]
        ),  # Control the initial parameter noise for noisy nets
        "batch_mode": tune.choice(["truncate_episodes", "complete_episodes"]),
        "hiddens": tune.choice(
            [[]]
        ),  # Dense-layer setup for each the advantage branch and the value branch
        "training_intensity": tune.choice(
            [1, 4, 16, 32]
        ),  # Number of iteration to run during the buffer phase
        "target_network_update_freq": tune.choice([500, 1000, 5000, 10000, 20000]),
        "num_steps_sampled_before_learning_starts": tune.choice(
            [1000, 10000, 20000]
        ),  # To initialise the replay buffer
    },
}

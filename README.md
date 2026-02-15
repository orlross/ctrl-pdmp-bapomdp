# Deep Reinforcement Learning for Controlled Piecewise Deterministic Markov Process in Cancer Treatment Follow-up

## Install 
Run following lines to download the git repository 

```
git clone git@forge.inrae.fr:orlane.le-quellennec/controlled_pdmp_po.git
cd controlled_pdmp_po
pip install -r requirements.txt
pip install -e .
```

## Environments

The folder `env` contains all environment used in the paper.
The `full_pdmp.py` corresponds to a piecewise deterministic Markov process (PDMP) simulator.
It simulates patient's trajectories. 
Those trajectories are fully observable. 

To create a PDMP trajectory instance : 
```
import gymnasium 
from gymnasium.envs.registration import register

# Import your environment
from env.full_pdmp import Patient

# Register your environment
register(
    id="env/Patient",
    entry_point="env.full_pdmp:Patient",
)

# Load an instance of Patient PDMP model
env = gymnasium.make('env/Patient', render_mode="human")
```

The class PartiallyObservableWrapper in `wrappers.py` transforms the PDMP into Partially Observable Markov Decision Process (POMDP) model detailed in the paper. It also includes a version of the wrapper with action masking ('POWrapperWithActionMask').
To create a partially observable patient trajectory instance : 
```
import gymnasium 
from gymnasium.envs.registration import register

# Import your environment
from env.full_pdmp import Patient
from env.wrappers import PartiallyObservableWrapper, ActionMaskWrapper

# Register your environment
register(
    id="env/Patient",
    entry_point="env.full_pdmp:Patient",
)

# Load an instance of partially observable patient (POMDP model)
env = gymnasium.make('env/Patient', render_mode="human")
env_po = PartiallyObservableWrapper(env)
env_po_masking = ActionMaskWrapper(env)
```

The BayesianWrapper in `wrappers.py` transforms the Partially Observable Markov Decision Process (POMDP) model into a Bayes-Adaptive Partially Observable Markov Decision Process (BAPOMDP) detailed in the paper.
To create a bayes-adaptive partially observable patient trajectory instance : 

```
import gymnasium 
from gymnasium.envs.registration import register

# Import your environment
from env.full_pdmp import Patient
from env.wrappers import BayesianWrapper
from env.wrappers import PartiallyObservableWrapper, ActionMaskWrapper

# Register your environment
register(
    id="env/Patient",
    entry_point="env.full_pdmp:Patient",
)

# Load an instance of partially observable patient (POMDP model)
env = gymnasium.make('env/Patient', render_mode="human")
env_bapo = BayesianWrapper(PartiallyObservableWrapper(env))
env_bapo_masking = BayesianWrapper(ActionMaskWrapper(PartiallyObservableWrapper(env)) 
```

## Tests 

The folder `tests` contains some functions test for each environment and wrappers.
 

## Training

The `training` folder contains all necessary scripts to train, run, and exploit neural networks. This folder includes:

### Hyperparameter Tuning

This script performs hyperparameter tuning using Rllib Tuner to find optimal hyperparameters for the algorithm. 

```
python ./training/tune.py [-h] --config-file CONFIG_FILE --model {pomdp,bapomdp} [--stop-iters STOP_ITERS] [--stop-timesteps STOP_TIMESTEPS] [--stop-reward STOP_REWARD] [--num-samples NUM_SAMPLES]
                   [--output-file OUTPUT_FILE]
```
It outputs a YAML file containing the tested hyperparameter combinations and the best configuration found in the specified `OUTPUT_FILE`.

Notice that there are only two required arguments:  `--config-file`, which specifies the path to the file that defines the hyperparameter spaces and  `--model`, which specifies the model to execute (pomdp or bapomdp).
Example configuration files are available in `./training/tune_files`:

#### Without Action Masking

- `dqn_searchspace_hyperparams.py`: Specifies hyperparameters spaces for DQN algorithm.
- `r2d2_searchspace_hyperparams.py`: Specifies hyperparameters spaces for R2D2 algorithm.
- `ppo_searchspace_hyperparams.py`: Specifies hyperparameters spaces for PPO algorithm.
- `ppo_searchspace_hyperparams_lstm.py`: Specifies hyperparameters spaces for PPO algorithm with LSTM.

#### With Action Masking 
(Note R2D2 with action masking is not available)

- `ppo_searchspace_hyperparams_masking.py`: Specifies hyperparameters spaces for PPO algorithm with masking
- `ppo_searchspace_hyperparams_lstm_masking.py`: Specifies hyperparameters spaces for PPO algorithm with LSTM and masking
- `dqn_searchspace_hyperparams_masking.py`: Specifies hyperparameters spaces for DQN algorithm with masking
### Example Usage

To tune the hyperparameters of the DQN algorithm:

```
python ./training/tune.py --model pomdp --config-file ./training/tune_files/dqn_searchspace_hyperparams.py \
    --stop-timesteps 100000  --num-samples 1000 --stop-iters 1000 \
    --output-file ./training/tune_files/tuned_hyperparams_dqn_v2.yaml
```

### Neural Network Training

This script performs multiple training and evaluation cycles using the tuned hyperparameters.

```
python ./training/evaluate.py [-h] --config-file CONFIG_FILE  [--stop-iters STOP_ITERS] \
    [--stop-timesteps STOP_TIMESTEPS] [--stop-reward STOP_REWARD] [--num-samples NUM_SAMPLES] \
    [--evaluation-interval EVALUATION_INTERVAL] [--output-folder OUTPUT_FOLDER]
```
After execution, the training results and the trained neural network model will be available in the specified `OUTPUT_FOLDER`.


Notice that the only required argument is `--config-file`, which specifies the path to the configuration file. Example configuration files are available in `./training/tune_files`:

#### Without Action Masking
- `tuned_ppo_pomdp.yaml`: Runs PPO algorithm on the POMDP model.
- `tuned_ppo_bapomdp.yaml`: Runs PPO algorithm on the BAPOMDP model.
- `tuned_ppo_lstm_pomdp.yaml`: Runs PPO algorithm with LSTM on the POMDP model.
- `tuned_ppo_lstm_bapomdp.yaml`: Runs PPO algorithm with LSTM on the BAPOMDP model.

#### With Action Masking

- `tuned_ppo_action_masking_pomdp.yaml`: Runs PPO algorithm on the POMDP model with action masking.
- `tuned_ppo_action_masking_bapomdp.yaml`: Runs PPO algorithm on the BAPOMDP model with action masking.
- `tuned_ppo_action_masking_lstm_pomdp.yaml`: Runs PPO algorithm with LSTM on the POMDP model with action masking.
- `tuned_ppo_action_masking_lstm_bapomdp.yaml`: Runs PPO algorithm with LSTM on the BAPOMDP model with action masking.

### Example Usage

To train a neural network using the PPO algorithm on the BAPOMDP model with action masking:

```
python ./training/evaluate.py --config-file ./training/tune_files/tuned_ppo_action_masking_bapomdp.yaml \
    --num-samples 1 --stop-iters 1000 --stop-timesteps 100000 --stop-reward 10 \
    --evaluation-interval 5 --output-folder ./training/results/bapomdp_xp_ppo_masking
```

## Simulations 

This folder contains scripts to simulate trajectories based on a specific model and chosen policy. 
All trajectory costs are stored in the `data` folder.

### Usage

```
python ./simulations/generate_data.py [-h] [--output-file OUTPUT_FILE] [--model MODEL] [--policy POLICY] \
    [--policy-path POLICY_PATH] [--num-samples NUM_SAMPLES]

```

### Options

- `-h, --help` : Show this help message and exit.
- `--output-file OUTPUT_FILE` : Path to the CSV file to store the simulated data.
- `--model MODEL` : Environment model. Options: `pdmp` `pomdp` `bapomdp`
- `--policy POLICY` : Fixed policy type. Options: `perfect` `alea inactive` `thresh` `memory`
- `--policy-path POLICY_PATH` : Path to the policy checkpoint in case of rllib deep policy
- `--num-samples NUM_SAMPLES` : Number of times the environment is run.

This is an example of command line to run to execute the code. 

```
python ./simulations/generate_data.py --model pdmp --policy alea --num-samples 100000
python ./simulations/generate_data.py --model pomdp --num-samples 100000 --policy-path POLICY_PATH
```

To compare all policy costs run `compare_cost.py` script.

```
cd simulations
python compare_cost.py --logdir ./data/pdmp_alea.csv ./data/pomdp_thresh.csv ./data/pdmp_inactive.csv ./data/pomdp_dqn.csv
```


## Authors and acknowledgment
Alice Cleynen, Benoite de Saporta, Orlane Rossini, Régis Sabbadin and Meritxell Vinyals

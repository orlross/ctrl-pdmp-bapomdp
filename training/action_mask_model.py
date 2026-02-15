from gymnasium.spaces import Dict, Box
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.misc import SlimFC
from ray.util.debug import log_once
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FCNet
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        print("orig_space", orig_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "obs" in orig_space.spaces
        )
        # super is used to call all parent class
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        print('orig_space["obs"]', orig_space["obs"])
        self.internal_model = FullyConnectedNetwork(
            orig_space["obs"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        print(
            "self.internal_model({obs: input_dict[obs][obs]})",
            self.internal_model({"obs": input_dict["obs"]["obs"]}),
        )
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["obs"]})
        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "obs" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["obs"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["obs"]})
        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-:wqtype mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class LSTMTorchActionMaskWrapper(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "obs" in orig_space.spaces
        )

        nn.Module.__init__(self)
        # super(LSTMTorchActionMaskWrapper, self).__init__(
        #    orig_space["obs"], action_space, None, model_config, name
        # )
        action_mask_length = orig_space["action_mask"].shape[0]
        obs_space_fcnet_shape = obs_space.shape
        obs_space_fcnet_shape = (obs_space.shape[0] - action_mask_length,)
        obs_space_fcnet = Box(
            obs_space.low[action_mask_length:],
            obs_space.high[action_mask_length:],
            obs_space_fcnet_shape,
            obs_space.dtype,
        )
        # old
        # super(LSTMTorchActionMaskWrapper, self).__init__(
        #    obs_space, action_space, None, model_config, name
        # )
        # new
        super(LSTMTorchActionMaskWrapper, self).__init__(
            obs_space_fcnet, action_space, None, model_config, name
        )
        # At this point, self.num_outputs is the number coming from the wrapped underlying model
        # new
        # Get the obs_space back to the original one (with 10 observations) to the original one
        self.obs_space = obs_space
        # new
        # Also set to the original obs space the requirements set on ModelV2
        self.view_requirements = {
            SampleBatch.OBS: ViewRequirement(shift=0, space=self.obs_space),
        }
        # Get input modifiers from model_config
        self.model_config = model_config
        self.cell_size = model_config["lstm_cell_size"]
        print("self.cell_size", self.cell_size)
        # print("self.cell_size", self.cell_size)
        self.use_prev_action = model_config.get("lstm_use_prev_action", False)
        # print("self.use_prev_action ", self.use_prev_action)
        self.use_prev_reward = model_config.get("lstm_use_prev_reward", False)
        # print("self.use_prev_reward ", self.use_prev_reward)
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        # self.fcnet = nn.Sequential(*layers)
        # self.lstm = nn.LSTM(hiddens[-1], self.cell_size, batch_first=True)
        # TODO: verify they are equal
        # TODO: Change
        # Define actual LSTM layer (with num_outputs being the nodes coming
        # from the wrapped (underlying) layer).
        self.lstm = nn.LSTM(
            self.num_outputs, self.cell_size, batch_first=not self.time_major
        )
        # Set self.num_outputs to the number of output nodes desired by the
        # caller of this constructor.
        self.num_outputs = num_outputs
        # changed
        # self.obs_space = obs_space
        # Postprocess LSTM output with another hidden layer and compute values.
        self._logits_branch = SlimFC(
            in_size=self.cell_size,
            out_size=self.num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self._value_branch = SlimFC(
            in_size=self.cell_size,
            out_size=1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        linear = next(self._logits_branch._model.children())
        h = [
            linear.weight.new(1, self.cell_size).zero_().squeeze(0),
            linear.weight.new(1, self.cell_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._features), [-1])

    @override(RecurrentNetwork)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):
        # print(
        #    "Entered FORWARD with input_dict ",
        #    input_dict,
        #    "state ",
        #    state,
        #    "seq_lens ",
        #    seq_lens,
        # )

        # if isinstance(input_dict["obs_flat"], dict):
        #    print(input_dict["obs_flat"], " is a dictionary")
        #    input_dict["obs_flat"] = input_dict["obs_flat"]["obs"]

        action_mask = input_dict["obs"]["action_mask"]
        input_dict["obs"] = input_dict["obs"]["obs"]
        input_dict["obs_flat"] = flatten_inputs_to_1d_tensor(input_dict["obs"])

        # print(
        #    "FORWARD calling FCNet forward with input_dict ",
        #    input_dict,
        #    "input_dict[obs]",
        #    input_dict["obs"],
        #    "input_dict[obs_flat]",
        #    input_dict["obs_flat"],
        # )

        # Get from the input the observation and the action_mask
        # Forward through the FCNet first
        wrapped_out, _ = self._wrapped_forward(input_dict, [], None)
        # print("output from FCNet", wrapped_out)
        # Push everything through our LSTM
        input_dict["obs_flat"] = wrapped_out
        # This calls the forward of RecurrentNetwork with the output of FCNet
        logits, new_state = super().forward(input_dict, state, seq_lens)
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        return masked_logits, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._features in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        self._features, [h, c] = self.lstm(
            inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        model_out = self._logits_branch(self._features)
        return model_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


LSTMTorchActionMaskModel = ModelCatalog._wrap_if_needed(
    FCNet, LSTMTorchActionMaskWrapper
)
LSTMTorchActionMaskModel._wrapped_forward = FCNet.forward
# print("LSTMTorchActionMaskModel.__mro__", LSTMTorchActionMaskModel.__mro__)


class DQNActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        print("orig_space", orig_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "obs" in orig_space.spaces
        )
        # super is used to call all parent class
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        print('orig_space["obs"]', orig_space["obs"])
        self.internal_model = FullyConnectedNetwork(
            orig_space["obs"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        print(
            "self.internal_model({obs: input_dict[obs][obs]})",
            self.internal_model({"obs": input_dict["obs"]["obs"]}),
        )
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["obs"]})
        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class DQNTorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        print("orig_space", orig_space)
        print("obs_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "obs" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["obs"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["obs"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-:wqtype mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class R2D2LSTMTorchActionMaskWrapper(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "obs" in orig_space.spaces
        )
        nn.Module.__init__(self)
        # super(LSTMTorchActionMaskWrapper, self).__init__(
        #    orig_space["obs"], action_space, None, model_config, name
        # )
        action_mask_length = orig_space["action_mask"].shape[0]
        obs_space_fcnet_shape = obs_space.shape
        obs_space_fcnet_shape = (obs_space.shape[0] - action_mask_length,)
        obs_space_fcnet = Box(
            obs_space.low[action_mask_length:],
            obs_space.high[action_mask_length:],
            obs_space_fcnet_shape,
            obs_space.dtype,
        )
        # old
        # super(LSTMTorchActionMaskWrapper, self).__init__(
        #    obs_space, action_space, None, model_config, name
        # )
        # new
        super(R2D2LSTMTorchActionMaskWrapper, self).__init__(
            obs_space_fcnet, action_space, None, model_config, name
        )
        # At this point, self.num_outputs is the number coming from the wrapped underlying model
        # new
        # Get the obs_space back to the original one (with 10 observations) to the original one
        self.obs_space = obs_space
        # new
        # Also set to the original obs space the requirements set on ModelV2
        self.view_requirements = {
            SampleBatch.OBS: ViewRequirement(shift=0, space=self.obs_space),
        }

        # Get input modifiers from model_config
        self.model_config = model_config
        self.cell_size = model_config["lstm_cell_size"]
        # print("self.cell_size", self.cell_size)
        self.use_prev_action = model_config.get("lstm_use_prev_action", False)
        # print("self.use_prev_action ", self.use_prev_action)
        self.use_prev_reward = model_config.get("lstm_use_prev_reward", False)
        # print("self.use_prev_reward ", self.use_prev_reward)
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        # self.fcnet = nn.Sequential(*layers)
        # self.lstm = nn.LSTM(hiddens[-1], self.cell_size, batch_first=True)
        # TODO: verify they are equal
        # TODO: Change
        # Define actual LSTM layer (with num_outputs being the nodes coming
        # from the wrapped (underlying) layer).
        self.lstm = nn.LSTM(
            self.num_outputs, self.cell_size, batch_first=not self.time_major
        )
        # Set self.num_outputs to the number of output nodes desired by the
        # caller of this constructor.
        self.num_outputs = num_outputs
        # changed
        # self.obs_space = obs_space
        # Postprocess LSTM output with another hidden layer and compute values.
        self._logits_branch = SlimFC(
            in_size=self.cell_size,
            out_size=self.num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self._value_branch = SlimFC(
            in_size=self.cell_size,
            out_size=1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        linear = next(self._logits_branch._model.children())
        h = [
            linear.weight.new(1, self.cell_size).zero_().squeeze(0),
            linear.weight.new(1, self.cell_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._features), [-1])

    @override(RecurrentNetwork)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):
        # print(
        #    "Entered FORWARD with input_dict ",
        #    input_dict,
        #    "state ",
        #    state,
        #    "seq_lens ",
        #    seq_lens,
        # )

        # if isinstance(input_dict["obs_flat"], dict):
        #    print(input_dict["obs_flat"], " is a dictionary")
        #    input_dict["obs_flat"] = input_dict["obs_flat"]["obs"]

        action_mask = input_dict["obs"]["action_mask"]
        input_dict["obs"] = input_dict["obs"]["obs"]
        input_dict["obs_flat"] = flatten_inputs_to_1d_tensor(input_dict["obs"])

        # print(
        #    "FORWARD calling FCNet forward with input_dict ",
        #    input_dict,
        #    "input_dict[obs]",
        #    input_dict["obs"],
        #    "input_dict[obs_flat]",
        #    input_dict["obs_flat"],
        # )

        # Get from the input the observation and the action_mask
        # Forward through the FCNet first
        wrapped_out, _ = self._wrapped_forward(input_dict, [], None)
        # print("output from FCNet", wrapped_out)
        # Push everything through our LSTM
        input_dict["obs_flat"] = wrapped_out
        # This calls the forward of RecurrentNetwork with the output of FCNet
        logits, new_state = super().forward(input_dict, state, seq_lens)
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        return masked_logits, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._features in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        self._features, [h, c] = self.lstm(
            inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        model_out = self._logits_branch(self._features)
        return model_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


R2D2LSTMTorchActionMaskModel = ModelCatalog._wrap_if_needed(
    FCNet, R2D2LSTMTorchActionMaskWrapper
)
R2D2LSTMTorchActionMaskModel._wrapped_forward = FCNet.forward
# print("R2D2LSTMTorchActionMaskModel.__mro__", R2D2LSTMTorchActionMaskModel.__mro__)

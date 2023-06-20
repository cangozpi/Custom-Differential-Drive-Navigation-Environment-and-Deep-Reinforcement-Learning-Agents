import torch
from torch import nn
from copy import deepcopy
from rl.DDPG.utils import log_gradients_in_model, log_training_losses
from VPG_Buffer import VPGBuffer


class VPG_Agent(): #TODO: make this extend a baseclass (ABC) of Agent and call its super().__init__()
    """
    Refer to 
    https://spinningup.openai.com/en/latest/algorithms/vpg.html, 
    https://towardsdatascience.com/policy-gradient-reinforce-algorithm-with-baseline-e95ace11c1c4 and, 
    https://towardsdatascience.com/generalized-advantage-estimation-in-reinforcement-learning-bf4a957f7975 
    for implementation details.
    """
    def __init__(self, obs_dim, action_dim, policy_hidden_dims, valueBaseline_hidden_dims, policy_lr, valueBaseline_lr, train_v_iters, gamma, max_action, logger=None, log_full_detail=False):
        self.logger = logger
        self.log_full_detail = log_full_detail

        self.mode = "train"
        self.train_v_iters = train_v_iters
        self.gamma = gamma
        self.max_action = max_action # actions returned is in range [-max_action, max_action]

        self.policy = Policy(obs_dim, action_dim, policy_hidden_dims)
        self.valueBaseline = ValueBaseline(obs_dim, valueBaseline_hidden_dims)

        self.optim_policy = torch.optim.Adam(self.policy.parameters(), policy_lr)
        self.optim_valueBaseline = torch.optim.Adam(self.valueBaseline.parameters(), valueBaseline_lr)

        self.valueBaseline_loss_func = torch.nn.MSELoss()

        # self.vpg_buffer = VPGBuffer(obs_dim, action_dim, buffer_size)

        self.train_mode()
        self._n_updates = 0

    def choose_action(self, obs):
        """
        Returns actions that are normalized in the [-1, 1] range. Don't forget to scale them up for your need.
        Note that obs and goal_state are torch.Tensor with no batch dimension.
        """
        obs = torch.unsqueeze(obs, dim=0) # batch dimension of 1
        action, action_log_prob = self.policy(obs)
        # Clip action into valid range 
        action = torch.clip(action, -self.max_action, self.max_action)

        return action, action_log_prob

    
    def update(self, states, actions, returns, advantages):
        """
        Given a batch of data(i.e. (s,a,R,A)) performs training/model update on the VPG agent's DNNs.
        states (torch.tensor) --> [(num_episodes_before_each_update * max_episode_length), state_dim]
        actions (torch.tensor) --> [(num_episodes_before_each_update * max_episode_length), action_dim]
        returns (torch.tensor) --> [(num_episodes_before_each_update * max_episode_length), 1]
        advantages (torch.tensor) --> [(num_episodes_before_each_update * max_episode_length), 1]
        """
        self.train_mode()

        # Update Policy model (Loss = - (1/N) * (log(pi(a|s)) * R))
        log_probs = self.policy._distribution(states).log_prob(actions) # --> [(num_episodes_before_each_update * max_episode_length), action_dim]
        policy_loss = - torch.mean(log_probs * torch.unsqueeze(advantages, dim=1))
        # policy_loss = - torch.mean(log_probs * returns)
        # assert log_probs.shape == returns.shape
        # print(states, states.shape, "states")
        # print(actions, actions.shape, "actions")
        # print(returns, returns.shape, "returns")
        # print(log_probs, log_probs.shape, "log_probs")
        # print(policy_loss, policy_loss.shape, "policy_loss")
        # print(advantages, advantages.shape, "advantages")

        self.optim_policy.zero_grad()
        policy_loss.backward()
        self.optim_policy.step()

        # Log Policy's stats to tensorboard
        log_gradients_in_model(self.policy, self.logger, self._n_updates, "Policy", self.log_full_detail)
        log_training_losses(policy_loss.cpu().detach(), self.logger, self._n_updates, "Policy")

        # ----------------
        # Update ValueBaseline model (Loss = (1/N) * (V- R)**2)
        for i in range(self.train_v_iters):
            value_preds = self.valueBaseline(states) # --> [(num_episodes_before_each_update * max_episode_length), 1]
            valueBaseline_loss = self.valueBaseline_loss_func(value_preds, returns)

            self.optim_valueBaseline.zero_grad()
            valueBaseline_loss.backward()
            self.optim_valueBaseline.step()

            # Log ValueBaseline's stats to tensorboard
            log_gradients_in_model(self.valueBaseline, self.logger, self._n_updates, "ValueBaseline", self.log_full_detail)
            log_training_losses(valueBaseline_loss.cpu().detach(), self.logger, self._n_updates, "ValueBaseline")


        self._n_updates += 1


    def train_mode(self):
        """
        Sets actor and target networks to model.train() mode of torch.
        Also makes choose_action return actions with added noise for training (exploration reasons).
        """
        self.mode = "train"
        self.policy.train()
        self.valueBaseline.train()

    def eval_mode(self):
        """
        Sets actor and target networks to model.train() mode of torch.
        Also makes choose_action return actions with no noise added for testing.
        """
        self.mode = "eval"
        self.policy.eval()
        self.valueBaseline.eval()
    
    def save_model(self):
        """
        Saves the current state of the neural network models of the actor and the critic of the VPG agent.
        """
        torch.save(
            self.policy.state_dict(),
            'VPG_policy.pkl'
        )
        torch.save(
            self.valueBaseline.state_dict(),
            'VPG_valueBaseline.pkl'
        )

    def load_model(self):
        """
        Loads the previously saved states of the actor and critic models to the current VPG agent.
        """
        self.policy.load_state_dict(
            torch.load('VPG_policy.pkl')
        )
        self.valueBaseline.load_state_dict(
            torch.load('VPG_valueBaseline.pkl')
        )


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims:list):
        """
        Inputs:
            obs_dim (tuple): dimension of the observations. (e.g. (C, H, W), for and RGB image observation).
            action_dim (tuple): dimension of the action space.
            hidden_dims (list): holds dimensions of the hidden layers excluding the input layer 
                and the input and the output dimensions.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        log_std = -0.5 * torch.ones(self.action_dim, dtype=torch.float32) # TODO: bunu anlamadım niye initial değeri bu ve neural net mu gibi estimate etmiyor!
        self.log_std = torch.nn.Parameter(log_std) # note that this is a nn.Parameter not a regular Tensor

        layers = []
        prev_dim = self.obs_dim  # shape of the flattened input to the network
        for i, hidden_dim in enumerate(hidden_dims):
            # if i == 0:
            #     layers.append(torch.nn.Flatten())
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        # layers.append(torch.nn.LayerNorm(prev_dim)) # Add batchNorm to mitigate tanh saturation problem
        layers.append(torch.nn.Linear(prev_dim, action_dim, bias=False))
                
        self.model_layers = torch.nn.ModuleList(layers)
    
    def _distribution(self, obs_batch):
        """
        Inputs:
            obs_batch (torch.Tensor): a batch of states.
        """
        # pass input through the layers of the model to find mu (mean) of the Normal distribution which we'll use as the Gaussian Policy
        batch_dim = obs_batch.shape[0]
        mu = obs_batch.reshape(batch_dim, -1) # --> [B, obs_dim]

        for layer in self.model_layers:
            mu = layer(mu)
        
        # std = torch.exp(self.log_std)
        std = 0.01
        return torch.distributions.Normal(mu, std)
        
    def _log_prob_from_distribution(self, act_distribution, act):
        return act_distribution.log_prob(act)
    
    def forward(self, obs_batch):
        """
        Returns an action sampled from the generated distribution, and the sampled actions log probability.
        Inputs:
            obs_batch (torch.Tensor): a batch of states.
        Returns:
            act: action sampled from the distribution
            act_log_prob: log probability of the sampled action
        """
        act_distribution = self._distribution(obs_batch)
        act = act_distribution.sample()
        act_log_prob = self._log_prob_from_distribution(act_distribution, act)
        return act, act_log_prob


class ValueBaseline(nn.Module):
    def __init__(self, obs_dim, hidden_dims:list):
        """
        Inputs:
            obs_dim (tuple): dimension of the observations. (e.g. (C, H, W), for and RGB image observation).
            action_dim (tuple): dimension of the actions.
            hidden_dims (list): holds dimensions of the hidden layers excluding the input layer 
                and the input and the output dimensions.
        """
        super().__init__()
        self.obs_dim = obs_dim 
        self.output_dim = 1 # V(s,a) is of shape (1,)

        layers = []
        prev_dim = self.obs_dim # shape of the flattened input to the network
        for i, hidden_dim in enumerate(hidden_dims):
            # if i == 0:
            #     layers.append(torch.nn.Flatten())
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, self.output_dim))
                
        self.model_layers = torch.nn.ModuleList(layers)
    
    
    def forward(self, state_batch):
        """
        Inputs:
            state_batch (torch.Tensor): a batch of states.
        """
        # pass input through the layers of the model
        batch_dim = state_batch.shape[0]
        out = state_batch.reshape(batch_dim, -1) # --> [B, self.obs_dim]

        for layer in self.model_layers:
            out = layer(out)
        
        return out
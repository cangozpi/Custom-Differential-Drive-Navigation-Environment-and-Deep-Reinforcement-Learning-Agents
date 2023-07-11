import numpy as np
import torch
from torch import nn
from rl.diff_drive_GymEnv import DiffDrive_env
from rl.DDPG.Replay_Buffer import ReplayBuffer
from rl.DDPG.utils import log_gradients_in_model, log_training_losses
from copy import deepcopy


class DQN_Agent(nn.Module):
    """
    Agent that support both DQN and DDQN.
    """
    def __init__(self, obs_dim, action_dim, hidden_dims, lr, initial_epsilon, epsilon_decay, min_epsilon, gamma, tau=0.995, target_update_frequency=2, use_target_network = True, logger=None, log_full_detail=False):
        super().__init__()
        self.logger = logger
        self.log_full_detail = log_full_detail

        self.obs_dim = obs_dim
        self.num_hidden_dims = len(hidden_dims) * 2 # note that *2 is due to activation functions that follow
        self.action_dim = action_dim
        self.num_action_heads = len(action_dim)
        self.mode = "train"
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma

        modules = []
        prev_h_dim = obs_dim
        for h in hidden_dims:
            modules.append(torch.nn.Linear(prev_h_dim, h))
            modules.append(torch.nn.ReLU())
            prev_h_dim = h
        for action_head in action_dim:
            modules.append(torch.nn.Linear(prev_h_dim, action_head))
        self.q_network = torch.nn.ModuleList(modules)
        self.target_q_network = deepcopy(self.q_network)
        for name, l in self.target_q_network.named_parameters(): # Freeze target network
            l.requires_grad = False

        # self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr)
        # self.loss_func = torch.nn.MSELoss()
        self.loss_func = torch.nn.SmoothL1Loss() # Huber Loss

        self._n_updates = 0
        self.use_target_network = bool(use_target_network)
        self.target_update_frequency = int(target_update_frequency)
        self.tau = float(tau)

        self.train_mode()
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, obs):
        """
        Inputs:
            obs: [B, obs_dim]
        Returns:
            preds: [B, num_action_heads, action_dim], logits/unnormalized probs over possible actions
        """
        # use GPU if available
        obs = obs.to(self.device)
        B = obs.shape[0]
        assert len(obs.shape) > 1 and obs.shape == (B, self.obs_dim)
        out = obs
        # Pass through base
        for l in range(self.num_hidden_dims):
            out = self.q_network[l](out) # [B, hidden_dim]

        # Pass through action_heads
        action_head_logits = []
        for h in range(self.num_action_heads):
            h = self.num_hidden_dims + h
            cur_action = self.q_network[h](out) # [B, action_dim]
            action_head_logits.append(cur_action)

        out = torch.stack(action_head_logits, dim=1) # [B, num_action_heads, action_dim]

        return out

    def target_network_forward(self, obs):
        """
        Inputs:
            obs: [B, obs_dim]
        Returns:
            preds: [B, num_action_heads, action_dim], logits/unnormalized probs over possible actions
        """
        # use GPU if available
        obs = obs.to(self.device)
        B = obs.shape[0]
        assert len(obs.shape) > 1 and obs.shape == (B, self.obs_dim)
        out = obs
        # Pass through base
        for l in range(self.num_hidden_dims):
            out = self.target_q_network[l](out) # [B, hidden_dim]

        # Pass through action_heads
        action_head_logits = []
        for h in range(self.num_action_heads):
            h = self.num_hidden_dims + h
            cur_action = self.target_q_network[h](out) # [B, action_dim]
            action_head_logits.append(cur_action)

        out = torch.stack(action_head_logits, dim=1) # [B, num_action_heads, action_dim]

        return out
    
    def update_target_network(self):
        for target_p, p in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_p.data.copy_(
                (target_p.data * self.tau) + (p.data * (1.0 - self.tau)) 
            )

    def choose_action(self, obs):
        """
        sample action from an e-greedy policy.
        Inputs:
            obs: [B, obs_dim]
            action: [B, action_dim]
        Returns:
            action_preds: [B], integer id of the predicted action
        """
        # use GPU if available
        obs = obs.to(self.device)

        B = obs.shape[0]
        assert len(obs.shape) > 1 and obs.shape == (B, self.obs_dim)
        if self.mode == "train":
            # e-greey policy
            p = np.random.random(1)
            if self.epsilon >= p: # random action
                B = obs.shape[0] # batch dimension
                random_actions = []
                for action_dim in self.action_dim:
                    cur_random_action = np.random.choice([i for i in range(action_dim)], size=B, replace=True) # [B]
                    random_actions.append(torch.tensor(cur_random_action))

                chosen_actions = torch.stack(random_actions, dim=1) # [B, num_action_heads]
            else: # policy action
                with torch.no_grad():
                    preds = self.forward(obs) # [B, num_action_heads, action_dim]
                    q_value_preds, action_preds = torch.max(preds, dim=-1) # [B, num_action_heads]
                    chosen_actions = action_preds

            # Decay epsilon
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon

            return chosen_actions.cpu()

        else: # at test time do not take random actions
            with torch.no_grad():
                preds = self.forward(obs) # [B, num_action_heds, action_dim]
                q_value_preds, action_preds = torch.max(preds, dim=-1) # [B, num_action_heads]
            return action_preds.cpu()
    
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        """
        Given a batch of data(i.e. (s,a,r,s',d)) performs training/model update on the DQN agent.
        """
        B = state_batch.shape[0]
        assert state_batch.shape == (B, self.obs_dim) and \
            action_batch.shape == (B, self.num_action_heads) and \
                reward_batch.shape == (B, 1) and \
                    next_state_batch.shape == (B, self.obs_dim) and \
                        terminal_batch.shape == (B, 1)

        # use GPU if available
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        terminal_batch = terminal_batch.to(self.device)

        self.train_mode()

        # TD Target
        with torch.no_grad():
            if self.use_target_network:
                preds = self.target_network_forward(next_state_batch) # [B, num_action_heads, action_dim]
            else:
                preds = self.forward(next_state_batch) # [B, num_action_heads, action_dim]
            max_q_value_preds, _ = torch.max(preds, dim=-1, keepdim=True) # [B, 1]
            assert max_q_value_preds.shape == (B, self.num_action_heads, 1)

            TD_target = reward_batch.unsqueeze(1) + ((1 - terminal_batch.unsqueeze(1).int()) * self.gamma * max_q_value_preds) # [B, num_action_heads, 1]


        preds = self.forward(state_batch) # [B, num_action_heads, action_dim]
        q_value_preds = torch.gather(preds, dim=2, index=action_batch.unsqueeze(-1).long()) # [B, num_action_heads, 1]
        assert q_value_preds.shape == (B, self.num_action_heads, 1)

        loss = self.loss_func(input=q_value_preds, target=TD_target)

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        if self.logger is not None:
            # Log Critic's stats to tensorboard
            log_gradients_in_model(self.q_network, self.logger, self._n_updates, "Q_Network", self.log_full_detail)
            log_training_losses(loss.cpu().detach(), self.logger, self._n_updates, "DQN")

        self._n_updates += 1

        if (self._n_updates % self.target_update_frequency == 0) and self.use_target_network: # update target network
            self.update_target_network()


    def train_mode(self):
        """
        Sets actor and target networks to model.train() mode of torch.
        Also makes choose_action return actions with added noise for training (exploration reasons).
        """
        self.mode = "train"
        self.train()
        self.q_network.train()

    def eval_mode(self):
        """
        Sets actor and target networks to model.train() mode of torch.
        Also makes choose_action return actions with no noise added for testing.
        """
        self.mode = "eval"
        self.eval()
        self.q_network.eval()

    
    def save_model(self):
        """
        Saves the current state of the neural network models of the actor and the critic of the DDPG agent.
        """
        torch.save(
            self.state_dict(),
            'DQN_Agent.pkl'
        )

    def load_model(self):
        """
        Loads the previously saved states of the actor and critic models to the current DDPG agent.
        """
        self.load_state_dict(
            torch.load('DQN_Agent.pkl')
        )


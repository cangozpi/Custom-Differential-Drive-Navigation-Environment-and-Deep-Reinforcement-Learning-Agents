import numpy as np
import torch
from torch import nn
from rl.diff_drive_GymEnv import DiffDrive_env
from rl.DDPG.Replay_Buffer import ReplayBuffer
from rl.DDPG.utils import log_gradients_in_model, log_training_losses


class DQN_Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, lr, initial_epsilon, epsilon_decay, min_epsilon, gamma, logger=None, log_full_detail=False):
        super().__init__()
        self.logger = logger
        self.log_full_detail = log_full_detail

        self.obs_dim = obs_dim
        self.action_dim = action_dim
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
        modules.append(torch.nn.Linear(prev_h_dim, action_dim))
        self.q_network = torch.nn.ModuleList(modules)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr)
        # self.loss_func = torch.nn.MSELoss()
        self.loss_func = torch.nn.SmoothL1Loss() # Huber Loss

        self.train_mode()

        self._n_updates = 0

    def forward(self, obs):
        """
        Inputs:
            obs: [B, obs_dim]
        Returns:
            preds: [B, action_dim], logits/unnormalized probs over possible actions
        """
        assert len(obs.shape) > 1
        out = obs
        for l in self.q_network:
            out = l(out)
        return out

    def choose_action(self, obs):
        """
        sample action from an e-greedy policy.
        Inputs:
            obs: [B, obs_dim]
            action: [B, action_dim]
        Returns:
            action_preds: [B], integer id of the predicted action
        """
        assert len(obs.shape) > 1
        if self.mode == "train":
            # e-greey policy
            p = np.random.randint(0, 1)
            if self.epsilon >= p: # random action
                B = obs.shape[0] # batch dimension
                random_actions = np.random.choice([i for i in range(self.action_dim)], size=B, replace=True)
                chosen_actions = torch.tensor(random_actions)
            else: # policy action
                with torch.no_grad():
                    preds = self.forward(obs) # [B, action_dim]
                    q_value_preds, action_preds = torch.max(preds, dim=-1) # [B]
                    assert len(q_value_preds.shape) == 1 and len(action_preds.shape) == 1
                    chosen_actions = action_preds

            # Decay epsilon
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon

            return chosen_actions

        else: # at test time do not take random actions
            with torch.no_grad():
                preds = self.forward(obs) # [B, action_dim]
                q_value_preds, action_preds = torch.max(preds, dim=-1) # [B]
                assert len(q_value_preds.shape) == 1 and len(action_preds.shape) == 1
            return action_preds
    
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        """
        Given a batch of data(i.e. (s,a,r,s',d)) performs training/model update on the DQN agent.
        """
        self.train_mode()

        # TD Target
        with torch.no_grad():
            B = state_batch.shape[0]
            preds = self.forward(next_state_batch) # [B, action_dim]
            max_q_value_preds, _ = torch.max(preds, dim=-1, keepdim=True) # [B, 1]
            assert max_q_value_preds.shape == (B, 1)
            TD_target = reward_batch + ((1 - terminal_batch.int()) * self.gamma * max_q_value_preds) # [B, 1]

        preds = self.forward(state_batch) # [B, action_dim]
        q_value_preds = torch.gather(preds, dim=1, index=action_batch.long()) # [B, 1]
        assert q_value_preds.shape == (B, 1)

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




if __name__ == "__main__":
    # Read in parameters from config.yaml
    config_path = 'rl/config/config_DQN_DiffDriveEnv.yaml'
    env = DiffDrive_Env(config_path)


    agent = DQN_Agent()

    observation = env.reset()
    env.render()
    while 1:
        if env.config['random_agent']: # use random agent
            action = env.action_space.sample() # [linear_vel, angular_vel]
        else: # use hard coded controller
            action = agent.choose_action(observation, env)

        next_observation, reward, done, info = env.step(action)

        observation = next_observation # [agent_x, agent_y, agent_theta]

        print(f'action: {action}, done: {done}, iteration: {info["iteration"]}, theta: {env.theta}')

        if done:
            time.sleep(2)

            # set target location randomly
            target_range = 10
            random_target_state = (np.random.rand(2) * 2 * target_range) - target_range
            target_x = np.clip(random_target_state[0], -target_range, target_range)
            target_y = np.clip(random_target_state[1], -target_range, target_range)
    
            observation = env.reset(target_x=target_x, target_y=target_y)

        env.render()
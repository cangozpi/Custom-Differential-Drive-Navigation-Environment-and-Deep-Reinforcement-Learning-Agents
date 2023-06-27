import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Debug_logger():
    def __init__(self, agent, tb_summaryWriter):
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        self.agent = agent # DDPG model with self.critic network

        self.tb_summaryWriter = tb_summaryWriter
        self.entry_count_dict = defaultdict(lambda : 0)

    
    def tb_log_scalar_entry(self, entry_name, entry_value):
        entry_index = self.entry_count_dict[entry_name]
        self.tb_summaryWriter.add_scalar(entry_name, entry_value, entry_index)
        self.entry_count_dict[entry_name] = entry_index + 1


    def record_entry(self, reward, state, action):
        self.episode_rewards.append(np.copy(reward))
        self.episode_states.append(np.copy(state))
        self.episode_actions.append(np.copy(action))

    def clear_recorded_rewards(self):
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
    
    def plot_predQ_vs_return(self, plt_gt_line=False):
        # Get predicted Q values
        with torch.no_grad():
            state_batch = torch.tensor(self.episode_states)
            action_batch = torch.tensor(self.episode_actions)
            pred_Q_values = self.agent.critic(state_batch=state_batch, action_batch=action_batch).numpy()
            pred_Q_values = pred_Q_values.reshape(-1)

        # Calculate Returns
        returns = [0]
        for r in reversed(self.episode_rewards):
            returns.insert(0, (self.agent.gamma * returns[0]) + r)
        returns = returns[:-1]

        # Plot
        if plt_gt_line:
            all_coords = np.concatenate([pred_Q_values, returns])
            min_coord = np.min(all_coords)
            max_coord = np.max(all_coords)
            plt.plot([min_coord, max_coord], [min_coord, max_coord], 'r-')

        plt.scatter(returns, pred_Q_values)
        plt.xlabel("Return")
        plt.ylabel("Predicted Q Value")
        plt.show()

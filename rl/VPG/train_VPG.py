import gym
import time
from VPG_Agent import VPG_Agent
from rl.DDPG.Replay_Buffer import ReplayBuffer 
from rl.DDPG.utils import *

from rl.env_utils import read_yaml_config

from torch.utils.tensorboard import SummaryWriter
import datetime

def main():
    # Read in parameters from config.yaml
    config_path = 'rl/config/config_VPG_DiffDrive_env.yaml'

    # Start Env
    # Initialize Diff_drive environment
    # env = DiffDrive_Env(config_path)
    # ---
    env = gym.make('Pendulum-v1', g=9.81)
    # env = gym.make('MountainCarContinuous-v0')
    config = read_yaml_config(config_path)
    env.config = config
    env.max_episode_length = 200
    global verbose
    verbose = config['verbose']
    # ---

    mode = env.config["mode"]
    if mode == "train": # train agent
        train_agent(env)
    elif mode == "test": # test pre-trained agent
        test_agent(env)
    else:
        raise Exception('\'mode\' must be either [\'train\', \'test\']')


def train_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "pendulum my_VPG agent training_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    # seed_everything(env.config["seed"]) # set seed # TODO: uncomment it
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cur_iteration = 0
    cum_episode_rewards = 0
    from collections import deque
    rewards = deque(maxlen=100)
    cur_num_updates = 0 # total number of updates including all the episodes

    concatenated_obs_dim = sum(env.observation_space.shape)
    concatenated_action_dim = sum(env.action_space.shape)

    agent = VPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
        env.config["policy_hidden_dims"], env.config["valueBaseline_hidden_dims"], float(env.config["policy_lr"]), float(env.config["valueBaseline_lr"]), \
            env.config['train_v_iters'], env.config["gamma"], max_action=torch.tensor(env.action_space.high).float(), \
                logger=tb_summaryWriter, log_full_detail=env.config['log_full_detail'])
    agent.train_mode() # TODO: handle .eval() case for testing the model too.

    obs = env.reset()
    obs = torch.tensor(obs).float()

    while cur_episode < env.config["total_episodes"]: 
        # Collect episodes by running the policy
        paths = []
        state_buffer = []
        action_buffer = []
        reward_buffer = []
        num_collected_episodes = 0
        while num_collected_episodes < env.config['num_episodes_before_each_update']:
            cur_iteration += 1
            with torch.no_grad():
                action, action_log_prob = agent.choose_action(obs.detach().clone()) # predict action in the range of [-1,1]
                action = torch.squeeze(action, dim=0).numpy()

            # Take action
            next_obs, reward, done, info = env.step(np.copy(action))
            next_obs = torch.tensor(next_obs).float()

            cum_episode_rewards += reward

            # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
            if done and cur_iteration < env.max_episode_length:#TODO: done and term is not being used in the return and advantage calculations!
                term = True
            else:
                term = False
        
            # Store (s,a,r) needed for update
            state_buffer.append(obs.numpy().copy())
            action_buffer.append(action.copy())
            reward_buffer.append(reward.copy())

            # Update current state
            obs = next_obs

            if verbose:
                print(f'episode: {cur_episode}, iteration: {cur_iteration}, action: {action}, reward: {reward}')

            if done:
                mean_ep_reward = cum_episode_rewards/cur_iteration
                rewards.append(cum_episode_rewards)
                # Log to Tensorboard
                tb_summaryWriter.add_scalar("Training Reward/[per episode]", mean_ep_reward, cur_episode)
                tb_summaryWriter.add_scalar("Training Reward/[ep_rew_mean]", np.mean(rewards), cur_episode)

                # Reset env
                obs = env.reset()
                # Concatenate observation with goal_state for regular DDPG agent
                obs = torch.tensor(obs).float()

                # Record current episode info into paths which is needed for agent update
                from copy import deepcopy
                paths.append({
                    'state_buffer': deepcopy(state_buffer),
                    'action_buffer': deepcopy(action_buffer),
                    'reward_buffer': deepcopy(reward_buffer)
                })
                # clean episode buffers for the next episode
                state_buffer = []
                action_buffer = []
                reward_buffer = []

                # time.sleep(3)
                # Reset episode parameters for a new episode
                cur_episode += 1
                cum_episode_rewards = 0
                cur_iteration = 0
                num_collected_episodes += 1
            

        # Update model 
        print(f"Updating the agent.")
        agent.train_mode()
        # Concatenate and flatten stored episode information
        states = torch.tensor(np.concatenate([path['state_buffer'] for path in paths], axis=0), dtype=torch.float32) # --> [(num_episodes_before_each_update * max_episode_length), state_dim]
        actions = torch.tensor(np.concatenate([path['action_buffer'] for path in paths], axis=0), dtype=torch.float32) # --> [(num_episodes_before_each_update * max_episode_length), action_dim]
        # Calculate Returns (G)
        all_returns = []
        for path in paths:
            all_rewards = path['reward_buffer']
            returns = []
            g_t = 0
            for r in reversed(all_rewards):
                g_t = r + (env.config['gamma'] * g_t)
                returns.insert(0, g_t)
            all_returns.append(returns)
        returns = torch.unsqueeze(torch.tensor(np.concatenate(all_returns, axis=0), dtype=torch.float32), dim=1) # --> [(num_episodes_before_each_update * max_episode_length), 1]
        # Calculate Advantages (A)
        with torch.no_grad():
            values = agent.valueBaseline(states) # --> [(num_episodes_before_each_update * max_episode_length), 1]
            advantages = returns - values # --> [(num_episodes_before_each_update * max_episode_length), 1]
            advantages = (advantages - torch.mean(advantages)) / torch.sqrt(torch.sum(advantages**2)) # normalize for stability, --> [(num_episodes_before_each_update * max_episode_length), 1]

        agent.update(states, actions, returns, advantages)

        # Save the model
        if ((cur_num_updates % env.config["save_every"] == 0) and (cur_num_updates > 0 )) or \
            (cur_episode + 1 == env.config["total_episodes"]) :
            print("Saving the model ...")
            agent.save_model()

        cur_num_updates += 1
        num_collected_episodes = 0



def test_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "pendulum my_VPG agent testing_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    # seed_everything(env.config["seed"]) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cum_episode_rewards = 0
    cur_iteration = 0

    concatenated_obs_dim = sum(env.observation_space.shape)
    concatenated_action_dim = sum(env.action_space.shape)

    agent = VPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
        env.config["policy_hidden_dims"], env.config["valueBaseline_hidden_dims"], float(env.config["policy_lr"]), float(env.config["valueBaseline_lr"]), \
            env.config["gamma"], max_action=torch.tensor(env.action_space.high).float(), logger=tb_summaryWriter, log_full_detail=False)
    agent.load_model()
    if env.config["verbose"]:
        print("Loaded a pre-trained agent...")
    agent.eval_mode()

    obs = env.reset()
    env.render()
    obs = torch.tensor(obs).float()

    while True:
        cur_iteration += 1
        with torch.no_grad():
            action, action_log_prob = agent.choose_action(obs.detach().clone()) # predict action in the range of [-1,1]
            action = torch.squeeze(action, dim=0).numpy()

        # Take action
        next_obs, reward, done, info = env.step(np.copy(action))
        next_obs = torch.tensor(next_obs).float()

        cum_episode_rewards += reward

        # Update current state
        obs = next_obs
        env.render()

        if verbose:
            print(f'episode: {cur_episode}, iteration: {cur_iteration}, action: {action}, reward: {reward}')


        if done:
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Testing Reward", cum_episode_rewards/cur_iteration, cur_episode)

            # Reset env
            obs = env.reset()
            env.render()
            # Concatenate observation with goal_state for regular DDPG agent
            obs = torch.tensor(obs).float()

            # time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0
            cur_iteration = 0
    

if __name__ == "__main__":
    main()
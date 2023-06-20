import gym
import time
from TD3_Agent import TD3_Agent
from rl.DDPG.Replay_Buffer import ReplayBuffer
from rl.DDPG.utils import *

from rl.env_utils import read_yaml_config

from torch.utils.tensorboard import SummaryWriter
import datetime
from rl.diff_drive_GymEnv.DiffDrive_env import DiffDrive_Env

def main():
    # Read in parameters from config.yaml
    config_path = 'rl/config/config_TD3_DiffDriveEnv.yaml'

    # Start Env
    # Initialize Diff_drive environment
    # env = DiffDrive_Env(config_path)
    # ---
    # env = gym.make('Pendulum-v1', g=9.81)
    # env = gym.make('MountainCarContinuous-v0')
    env = DiffDrive_Env(config_path)
    config = read_yaml_config(config_path)
    env.config = config
    env.max_episode_length = config['max_episode_length']
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
    log_dir, run_name = "logs_tensorboard/", "diff_drive my_TD3 agent training_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    # seed_everything(env.config["seed"]) # set seed # TODO: uncomment it
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cur_iteration = 0
    cur_iterations_btw_updates = 0
    cum_episode_rewards = 0
    from collections import deque
    rewards = deque(maxlen=100)
    cur_num_updates = 0 # total number of updates including all the episodes

    concatenated_obs_dim = sum(env.observation_space.shape)
    concatenated_action_dim = sum(env.action_space.shape)

    agent = TD3_Agent(concatenated_obs_dim, concatenated_action_dim, \
        env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["critic_lr"]), \
            env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config['act_noise'], env.config['target_noise'], env.config['clip_noise_range'], env.config["gamma"], env.config["tau"], \
                max_action=torch.tensor(env.action_space.high).float(), policy_update_delay=env.config['policy_update_delay'], logger=tb_summaryWriter, log_full_detail=env.config['log_full_detail'])
    agent.train_mode() # TODO: handle .eval() case for testing the model too.
    replay_buffer = ReplayBuffer(env.config["replay_buffer_size"], concatenated_obs_dim, concatenated_action_dim, env.config["batch_size"])

    obs = env.reset()
    env.render()
    obs = torch.tensor(obs).float()

    replay_buffer.clear_staged_for_append()
    num_warmup_steps_taken = 0
    while cur_episode < env.config["total_episodes"]: 
        cur_iteration += 1
        cur_iterations_btw_updates += 1
        # For warmup_steps many iterations take random actions to explore better
        if num_warmup_steps_taken < env.config["warmup_steps"]: # take random actions
            action = env.action_space.sample()
            num_warmup_steps_taken += 1
        else: # agent's policy takes action
            with torch.no_grad():
                action = agent.choose_action(obs.detach().clone()) # predict action in the range of [-1,1]
                action = torch.squeeze(action, dim=0).numpy()

        # Take action
        next_obs, reward, done, info = env.step(np.copy(action))
        next_obs = torch.tensor(next_obs).float()

        cum_episode_rewards += reward

        # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
        if done and cur_iteration < env.max_episode_length:
            term = True
        else:
            term = False

        # Stage current (s,a,s') to replay buffer as to be appended at the end of the current episode
        replay_buffer.stage_for_append(obs, torch.tensor(action), torch.tensor(reward), next_obs, torch.tensor(term))

        # Update current state
        obs = next_obs
        env.render()

        if verbose:
            print(f'episode: {cur_episode}, iteration: {cur_iteration}, action: {action}, reward: {reward}')


        if done:
            mean_ep_reward = cum_episode_rewards/cur_iteration
            rewards.append(cum_episode_rewards)
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Training Reward/[per episode]", mean_ep_reward, cur_episode)
            tb_summaryWriter.add_scalar("Training Reward/[ep_rew_mean]", np.mean(rewards), cur_episode)
            tb_summaryWriter.add_scalar("Training epsilon", agent.epsilon, cur_episode)

            # Commit HER experiences to replay_buffer
            replay_buffer.commit_append()
            replay_buffer.clear_staged_for_append()

            # Reset env
            obs = env.reset()
            # Concatenate observation with goal_state for regular DDPG agent
            obs = torch.tensor(obs).float()

            # time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0
            cur_iteration = 0


        # Update model if its time
        if (cur_iterations_btw_updates % env.config["update_every"]== 0) and (num_warmup_steps_taken >= env.config["warmup_steps"]) and replay_buffer.can_sample_a_batch():
            print(f"epoch: {cur_episode}, updating the agent with {env.config['num_updates']} sampled batches.")
            for _ in range(env.config["num_updates"]):
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_buffer.sample_batch() 
                agent.update(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
                cur_num_updates += 1

                # Save the model
                if ((cur_num_updates % env.config["save_every"] == 0) and (cur_num_updates > 0 )) or \
                    (cur_episode == env.config["total_episodes"]) :
                    print("Saving the model ...")
                    agent.save_model()

            cur_iterations_btw_updates = 0


def test_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "diff_drive my_TD3 agent testing_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    # seed_everything(env.config["seed"]) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cum_episode_rewards = 0
    cur_iteration = 0

    concatenated_obs_dim = sum(env.observation_space.shape)
    concatenated_action_dim = sum(env.action_space.shape)

    agent = TD3_Agent(concatenated_obs_dim, concatenated_action_dim, \
        env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["actor_lr"]), \
            env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config['act_noise'], env.config['target_noise'], env.config['clip_noise_range'], env.config["gamma"], env.config["tau"], \
                max_action=torch.tensor(env.action_space.high).float())
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
            action = agent.choose_action(obs.detach().clone()) # predict action in the range of [-1,1]
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
import gym
import time
from DDPG_Agent import DDPG_Agent
from DDPG_Agent_actorWithRotationDependentTranslation import DDPG_Agent_actorWithRotationDependentTranslation
from Replay_Buffer import ReplayBuffer 
from utils import *

from rl.env_utils import read_yaml_config

from torch.utils.tensorboard import SummaryWriter
import datetime
from rl.diff_drive_GymEnv.DiffDrive_env import DiffDrive_Env

from rl.debug_utils import *

def main():
    # Read in parameters from config.yaml
    config_path = 'rl/config/config_DDPG_DiffDriveEnv.yaml'

    # Start Env
    # Initialize Diff_drive environment
    # env = DiffDrive_Env(config_path)
    # ---
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('Pendulum-v1', g=9.81)
    env = DiffDrive_Env(config_path)
    env.seed(42)
    config = read_yaml_config(config_path)
    env.config = config
    env.max_episode_length = config['max_episode_length']
    global plot_predQ_vs_return
    plot_predQ_vs_return = False
    global verbose
    verbose = config['verbose']
    # ---

    mode = env.config["mode"]
    if mode == "train": # train agent
        train_agent(env)
    elif mode == "test": # test pre-trained agent
        test_agent(env)
    elif mode == "hyperparameter_search": # perform hyperparameter search
        hidden_dims = [[8, 16], [400, 300]]
        learning_rates = [1e-3, 5e-5]
        act_noises = [0.0, 0.3, 0.5]
        for h_dim in hidden_dims:
            for lr in learning_rates:
                for act_noise in act_noises:
                    env.config['actor_hidden_dims'] = h_dim
                    env.config['critic_hidden_dims'] = h_dim
                    env.config['actor_lr'] = lr
                    env.config['critic_lr'] = lr
                    env.config['act_noise'] = act_noise
                    train_agent(env)
    else:
        raise Exception('\'mode\' must be either [\'train\', \'test\']')


def train_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "diff_drive my_DDPG agent training_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    seed_everything(env.config["seed"]) # set seed # TODO: uncomment it
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cur_iteration = 0
    cur_iteration_btw_updates = 0
    cum_episode_rewards = 0
    from collections import deque
    rewards = deque(maxlen=100)
    cur_num_updates = 0 # total number of updates including all the episodes

    concatenated_obs_dim = sum(env.observation_space.shape)
    concatenated_action_dim = sum(env.action_space.shape)

    if True:
        agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
            env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["critic_lr"]), \
                env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config['act_noise'], env.config['target_noise'], env.config['clip_noise_range'], env.config["gamma"], env.config["tau"], \
                    max_action=torch.tensor(env.action_space.high).float(), policy_update_delay=env.config['policy_update_delay'], logger=tb_summaryWriter, log_full_detail=env.config['log_full_detail'])
    else:
        agent = DDPG_Agent_actorWithRotationDependentTranslation(concatenated_obs_dim, concatenated_action_dim, \
            env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["critic_lr"]), \
                env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config['act_noise'], env.config['target_noise'], env.config['clip_noise_range'], env.config["gamma"], env.config["tau"], \
                    max_action=torch.tensor(env.action_space.high).float(), policy_update_delay=env.config['policy_update_delay'], logger=tb_summaryWriter, log_full_detail=env.config['log_full_detail'])
    agent.train_mode() # TODO: handle .eval() case for testing the model too.
    replay_buffer = ReplayBuffer(env.config["replay_buffer_size"], concatenated_obs_dim, concatenated_action_dim, env.config["batch_size"])

    debug_logger = Debug_logger(agent, tb_summaryWriter)

    def plot_activation_grads(name, model, debug_logger):
        for n, c in model.named_children():
            def get_activation_grad_plot_hook(name):
                def hook_fn(module, grad_input, grad_output):
                    # Calculate grad norm
                    grad = torch.sum(grad_input[0], dim=0)
                    grad_norm = torch.norm(grad, 2) # L2 norm
                    debug_logger.tb_log_scalar_entry(name, grad_norm)
                return hook_fn

            if isinstance(c, torch.nn.ModuleList):
                plot_activation_grads(name, c, debug_logger)

            if isinstance(c, torch.nn.ReLU):
                c.register_full_backward_hook(get_activation_grad_plot_hook(name=name + f'/ReLU:{n} grad norm'))
            if isinstance(c, torch.nn.Tanh):
                c.register_full_backward_hook(get_activation_grad_plot_hook(name=name + f'/Tanh:{n} grad norm'))

    # plot_activation_grads(agent.critic)
    plot_activation_grads("Actor", agent.actor, debug_logger)
    plot_activation_grads("Critic", agent.actor, debug_logger)

    obs = env.reset()
    env.render()
    obs = torch.tensor(obs).float()

    replay_buffer.clear_staged_for_append()
    num_warmup_steps_taken = 0
    while cur_episode < env.config["total_episodes"]: 
        cur_iteration += 1
        cur_iteration_btw_updates += 1
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
        debug_logger.record_entry(reward, obs, action)

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
            print(f'episode: {cur_episode}, iteration: {cur_iteration}, action: {action}, reward: {reward:.2f}')


        if done:
            mean_ep_reward = cum_episode_rewards/cur_iteration
            rewards.append(cum_episode_rewards)
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Training Reward/[average per episode]", mean_ep_reward, cur_episode)
            tb_summaryWriter.add_scalar("Training Reward/[total per episode]", cum_episode_rewards, cur_episode)
            tb_summaryWriter.add_scalar("Training Reward/[running mean of episodes]", np.mean(rewards), cur_episode)
            tb_summaryWriter.add_scalar("Training epsilon", agent.epsilon, cur_episode)

            # Commit experiences to replay_buffer
            replay_buffer.commit_append()
            replay_buffer.clear_staged_for_append()

            if plot_predQ_vs_return:
                debug_logger.plot_predQ_vs_return() # plot 
            debug_logger.clear_recorded_rewards() # clear recorded values for a new episode

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
        if (cur_iteration_btw_updates % env.config["update_every"]== 0) and (num_warmup_steps_taken >= env.config["warmup_steps"]) and replay_buffer.can_sample_a_batch():
            print(f"Updating the agent with {env.config['num_updates']} sampled batches.")
            for _ in range(env.config["num_updates"]):
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_buffer.sample_batch() 
                agent.update(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
                cur_num_updates += 1

                # Save the model
                if ((cur_num_updates % env.config["save_every"] == 0) and (cur_num_updates > 0 )) or \
                    (cur_episode == env.config["total_episodes"]) :
                    print("Saving the model ...")
                    agent.save_model()
                
            cur_iteration_btw_updates = 0


def test_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "diff_drive my_DDPG agent testing_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    # seed_everything(env.config["seed"]) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cum_episode_rewards = 0
    cur_iteration = 0

    concatenated_obs_dim = sum(env.observation_space.shape)
    concatenated_action_dim = sum(env.action_space.shape)

    if True:
        agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
            env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["actor_lr"]), \
                env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config['act_noise'], env.config['target_noise'], env.config['clip_noise_range'], env.config["gamma"], env.config["tau"], \
                    max_action=torch.tensor(env.action_space.high).float())
    else:
        agent = DDPG_Agent_actorWithRotationDependentTranslation(concatenated_obs_dim, concatenated_action_dim, \
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
            print(f'episode: {cur_episode}, iteration: {cur_iteration}, action: {action}, reward: {reward:.2f}')


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
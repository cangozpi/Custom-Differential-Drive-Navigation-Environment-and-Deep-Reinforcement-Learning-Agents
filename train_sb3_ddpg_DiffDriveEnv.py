import gym
import panda_gym
from time import sleep

from stable_baselines3 import DDPG, HerReplayBuffer
from rl.diff_drive_GymEnv.DiffDrive_env import DiffDrive_Env


mode = 3


# For env and reward options refer to https://panda-gym.readthedocs.io/en/latest/usage/environments.html
# env = gym.make('PandaReach-v2', render=True) # the environment return a reward if and only if the task is completed,

# Taking Random actions in the env
if mode == 1:
    # env = gym.make('PandaReachDense-v2', render=True) # the closer the agent is to completing the task, the higher the reward.
    # env = gym.make('Pendulum-v1', g=9.81)
    config_path = 'rl/config/config_DDPG_DiffDriveEnv.yaml'
    env = DiffDrive_Env(config_path)
    env.seed(42)

    observation = env.reset()

    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, info = env.step(action)
        sleep(1/60)

        if terminated:
            observation = env.reset()

    env.close()

# DDPG training 
if mode == 2:
    mode2 = "train"

    if mode2 == "train":
        # env = gym.make("PandaReach-v2")
        # env = gym.make("PandaReachDense-v2")
        # env = gym.make('Pendulum-v1', g=9.81)
        config_path = 'rl/config/config_DDPG_DiffDriveEnv.yaml'
        env = DiffDrive_Env(config_path)
        env.seed(42)
        # Train model
        # replay_buffer = HerReplayBuffer(env, 1_000_000, n_sampled_goal=0)
        model = DDPG(policy="MlpPolicy", env=env, verbose=1, \
            tensorboard_log="./tb_logs/", \
            learning_rate=1e-3, learning_starts=0, batch_size=100, \
            # replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            # replay_buffer_kwargs=dict(
            #     n_sampled_goal=0,
            #     goal_selection_strategy="future",
            #     online_sampling=True,
            #     max_episode_length=50,
            # ),
            gradient_steps=100,
            policy_kwargs={
                'net_arch': {
                    'pi': [400, 300],
                    'qf': [400, 300]
                    # 'pi': [8, 16],
                    # 'qf': [8, 16]
                }
            },
            gamma=0.9999,
            seed=42
    )
        print(model.policy)
        model.learn(500_000, log_interval=1, tb_log_name="sb3_DDPG_DiffDriveEnv", progress_bar=True)

        model.save("DDPG_DiffDriveEnv") # Save model
    
        mode2 = "test" # move on to testing the trained model


if mode == 3 or mode2 == "test":
    # env = gym.make("PandaReach-v2", render=True)
    # env = gym.make("PandaReachDense-v2", render=True)
    # env = gym.make('Pendulum-v1', g=9.81)
    config_path = 'rl/config/config_DDPG_DiffDriveEnv.yaml'
    env = DiffDrive_Env(config_path)
    env.seed(42)

    # Test model
    model = DDPG.load("DDPG_DiffDriveEnv", env=env) # load model

    obs = env.reset()
    env.render()
    while True:
        action, _hidden_state = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)

        obs = next_obs

        if True:
            print(f'obs: {obs}, action: {action}, reward: {reward}')

        sleep(1/60)
        env.render()

        if done:
            obs = env.reset()
            env.render()





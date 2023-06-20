import gym
import panda_gym
from time import sleep


mode = 2


# For env and reward options refer to https://panda-gym.readthedocs.io/en/latest/usage/environments.html
# env = gym.make('PandaReach-v2', render=True) # the environment return a reward if and only if the task is completed,

# Taking Random actions in the env
if mode == 1:
    # env = gym.make('PandaReachDense-v2', render=True) # the closer the agent is to completing the task, the higher the reward.
    env = gym.make('Pendulum-v1', g=9.81)
    env.seed(42)

    observation = env.reset()

    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, info = env.step(action)
        sleep(1/60)

        if terminated:
            observation = env.reset()

    env.close()

# PPO training 
if mode == 2:
    from stable_baselines3 import PPO, HerReplayBuffer

    mode2 = "test"

    if mode2 == "train":
        # env = gym.make("PandaReach-v2")
        # env = gym.make("PandaReachDense-v2")
        env = gym.make('Pendulum-v1', g=9.81)
        env.seed(42)
        print(env.observation_space)
        # Train model
        # replay_buffer = HerReplayBuffer(env, 1_000_000, n_sampled_goal=0)
        model = PPO(policy="MlpPolicy", env=env, verbose=1, \
            tensorboard_log="./tb_logs/", \
            learning_rate=3e-4, n_steps=2048, batch_size=64, \
            # replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            # replay_buffer_kwargs=dict(
            #     n_sampled_goal=0,
            #     goal_selection_strategy="future",
            #     online_sampling=True,
            #     max_episode_length=50,
            # ),
            n_epochs=10,
            policy_kwargs={
                'net_arch': {
                    'pi': [64, 64],
                    'vf': [64, 64]
                }
            },
            gamma=0.9999,
            seed=42
        )
        print(model.policy)
        model.learn(400_000, log_interval=1, tb_log_name="sb3_PPO_pendulum", progress_bar=True)

        model.save("PPO_pendulum") # Save model
    
        mode2 = "test" # move on to testing the trained model

    if mode2 == "test":
        # env = gym.make("PandaReach-v2", render=True)
        # env = gym.make("PandaReachDense-v2", render=True)
        env = gym.make('Pendulum-v1', g=9.81)
        env.seed(42)

        # Test model
        model = PPO.load("PPO_pendulum", env=env) # load model

        obs = env.reset()
        env.render()
        while True:
            action, _hidden_state = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)

            obs = next_obs

            sleep(1/60)
            env.render()

            if done:
                obs = env.reset()
                env.render()





from rl.diff_drive_GymEnv.DiffDrive_env import DiffDrive_Env
import numpy as np
import time


class HardCodedController:
    """
    A hard coded naive deterministic policy which operates on ground truth position and 
    orientation observations to navigate to target location.
    """
    def __init__(self):
        pass

    def get_action(self, observation, env):
        agent_pos = observation[:2]
        target_pos = np.array([env.target_x, env.target_y])
        v = target_pos - agent_pos

        agent_theta = observation[2]
        theta_diff = np.arctan2(v[1], v[0])
        angular_vel_action = (theta_diff - agent_theta) / env._step_duration
        linear_vel_action = np.clip(np.linalg.norm(v) / env._step_duration, -1, 1)
        action = np.array([linear_vel_action, angular_vel_action])
        return action



if __name__ == "__main__":
    # Read in parameters from config.yaml
    config_path = 'rl/config/config_HardCodedController_DiffDrive_env.yaml'
    env = DiffDrive_Env(config_path)
    agent = HardCodedController()

    observation = env.reset()
    env.render()
    while 1:
        if env.config['random_agent']: # use random agent
            action = env.action_space.sample() # [linear_vel, angular_vel]
        else: # use hard coded controller
            action = agent.get_action(observation, env)

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
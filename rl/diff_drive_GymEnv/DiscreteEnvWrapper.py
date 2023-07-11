from rl.diff_drive_GymEnv.DiffDrive_env import DiffDrive_Env
from math import pi
import numpy as np
from gym import spaces

class Discrete_DiffDrive_Env(DiffDrive_Env):
    """
    Wraps the passed in DiffDrive_env instance to an environment which supports discrete actions space.
    """
    def __init__(self, *args):
        super().__init__(*args)
        if False:
            self.action_space = spaces.Discrete(4) # {0, 1, 2, 3}
        else:
            self.action_space = spaces.MultiDiscrete([2,2]) # [[0, 1], [0, 1]], i.e. [[up, down], [left, right]], e.g. [up, left]

        self.rotation_angle_in_radians = (pi / 180) * 10 # agent can turn 10 degrees to left or right at every step
        self.linear_vel = 0.5 # linear velocity of the agent at every step
    
    def convert_discrete_action_to_continuous_action(self, disc_action):
        """
        convertes the given discrete action to continuous action.
        disc_action:
            action : [0, 1, 2, 3], i.e. [turn left, go forward, turn right, stay]
        cont_actions:
            action : [linear_vel,  angular_velocity]
        """
        if disc_action == 0: # left
            cont_action = [0.0 , self.rotation_angle_in_radians]
        elif disc_action == 1: # forward
            cont_action = [self.linear_vel, 0.0]
        elif disc_action == 2: # right
            cont_action = [0.0 , -self.rotation_angle_in_radians]
        elif disc_action == 3: # stay
            cont_action = [0.0 , 0.0]
        else:
            raise Exception(f'{disc_action} is not a valid action. It should be from [0, 1, 2].')
        
        return np.array(cont_action)

    def step(self, disc_action):
        """
        Converts the given discrete action to continuous action and takes a step in the env.
        disc_action:
            action : [0, 1, 2], i.e. [left, forward, right]
        """
        if False:
            # action : [0, 1, 2], i.e. [left, forward, right]
            assert len(disc_action) == 1
            cont_action = self.convert_discrete_action_to_continuous_action(disc_action)
            return super().step(cont_action)
        elif False:
            # action : [0, 1, 2, 3], i.e. [left, right, up, down]
            assert len(disc_action) == 1
            # Apply action on the robot:
            # calculate position of the robot after the displacements
            delta_x = 0.0
            delta_y = 0.0
            if disc_action == 0: # left
                delta_x = -1.0
                delta_y = 0.0
            elif disc_action == 1: # right
                delta_x = 1.0
                delta_y = 0.0
            elif disc_action == 2: # up
                delta_x = 0.0
                delta_y = 1.0
            elif disc_action == 3: # down
                delta_x = 0.0
                delta_y = -1.0
            else:
                raise Exception(f'{disc_action} is not a valid action. It should be from [0, 1, 2].')

            delta_x = self._step_duration * delta_x
            delta_y = self._step_duration * delta_y
            self.x += delta_x
            self.y += delta_y

            # Simulate real time action taking
            if self.config['real_time']:
                from time import sleep
                sleep(self._step_duration)

            # Get observation
            theta_diff = np.arctan2(self.target_y - self.x, self.target_x - self.x)
            observation = np.array([self.x, self.y, theta_diff, self.target_x, self.target_y])

            # Calculate Reward
            from rl.diff_drive_GymEnv.env_utils import get_reward
            target_state = np.array([self.target_x, self.target_y])
            reward = get_reward(observation, target_state)

            # Check done
            done = False
            # check if max num steps for an episode is reached
            if self.cur_iteration == (self.max_episode_length - 1):
                done = True
            # check if robot is close to target within the threshold values
            d = [observation[0] - target_state[0], observation[1] - target_state[1]]
            l2_distance_to_target = np.linalg.norm(d)
            if l2_distance_to_target <= self._distance_threshold: # goal state is reached
                done = True

            # Get info
            info = {
                'observation': observation,
                'target_state': target_state,
                'iteration': self.cur_iteration,
                'verbose': self.config['verbose']
            }

            self.cur_iteration += 1

            return observation, reward, done, info
        else:
            # action : [[0, 1], [0, 1]], i.e. [[up, down], [left, right]], e.g. [up, left]
            assert len(disc_action) == 2
            # Apply action on the robot:
            # calculate position of the robot after the displacements
            delta_x = 0.0
            delta_y = 0.0
            if disc_action[0] == 0: # up
                delta_y = 1.0
            elif disc_action[0] == 1: # down
                delta_y = -1.0

            if disc_action[1] == 0: # left
                delta_x = -1.0
            elif disc_action[1] == 1: # right
                delta_x = 1.0

            delta_x = self._step_duration * delta_x
            delta_y = self._step_duration * delta_y
            self.x += delta_x
            self.y += delta_y

            # Simulate real time action taking
            if self.config['real_time']:
                from time import sleep
                sleep(self._step_duration)

            # Get observation
            theta_diff = np.arctan2(self.target_y - self.x, self.target_x - self.x)
            observation = np.array([self.x, self.y, theta_diff, self.target_x, self.target_y])

            # Calculate Reward
            from rl.diff_drive_GymEnv.env_utils import get_reward
            target_state = np.array([self.target_x, self.target_y])
            reward = get_reward(observation, target_state)

            # Check done
            done = False
            # check if max num steps for an episode is reached
            if self.cur_iteration == (self.max_episode_length - 1):
                done = True
            # check if robot is close to target within the threshold values
            d = [observation[0] - target_state[0], observation[1] - target_state[1]]
            l2_distance_to_target = np.linalg.norm(d)
            if l2_distance_to_target <= self._distance_threshold: # goal state is reached
                done = True

            # Get info
            info = {
                'observation': observation,
                'target_state': target_state,
                'iteration': self.cur_iteration,
                'verbose': self.config['verbose']
            }

            self.cur_iteration += 1

            return observation, reward, done, info

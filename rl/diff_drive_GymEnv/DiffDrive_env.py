from gym import Env, spaces
import numpy as np
from rl.diff_drive_GymEnv.env_utils import get_observation, get_reward, get_target_state, read_yaml_config
from time import sleep
import matplotlib.pyplot as plt


class DiffDrive_Env(Env):
    def __init__(self, config_path):
        # Read in parameters from config.yaml
        self.config = read_yaml_config(config_path)
        self.render_mode = self.config['render_mode']

        # Set gym spaces
        self.action_space = spaces.Box(-1 * np.ones((2, )), np.ones((2, )), dtype=np.float32)
        self.observation_space = spaces.Box(-1 * np.ones((5, )), np.ones((5, )), dtype=np.float32)

        self._step_duration = self.config['step_duration'] * (10 ** -9) # how long each individual action is taken for. (i.e. time btw observations). In seconds
        self.max_episode_length = self.config['max_episode_length'] # max num steps before terminating the episode
        self._distance_threshold = self.config['distance_threshold'] # if robot is withing the reach of target smaller than this threshold value then goal state is reached
        self.cur_iteration = 0

        # Robot state wrt absolute world frame
        self.x = 0 # in meters
        self.y = 0 # in meters
        self.theta = 0 # in radians

        # target location wrt absolute world frame
        self.target_x = 5
        self.target_y = 5

        if 'draw_coordinates' in self.config['render_mode']:
            # Rendering position
            plt.ion() # activate interactive mode
            self.fig, self.ax = plt.subplots(figsize=(4, 5))
            # self.fig = plt.figure()
            # self.ax = self.fig.add_subplot(111)
            self.line1, = self.ax.plot(self.x, self.y, ".b")
            self.line1, = self.ax.plot(self.target_x, self.target_y, "*r")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # line1, = ax.scatter()

    def reset(self, target_x=None, target_y=None):
        # Reset robot location in the world
        self.x = 0 # in meters
        self.y = 0 # in meters
        self.theta = 0 # in radians

        # target location wrt absolute world frame
        if (target_x is not None) and (target_y is not None):
            self.target_x = target_x
            self.target_y = target_y
        else:
            self.target_x = 5
            self.target_y = 5

        self.cur_iteration = 0

        observation = get_observation(self.x, self.y, self.theta, self.target_x, self.target_y)

        if 'draw_coordinates' in self.config['render_mode']:
            # Rendering position
            plt.close()
            plt.ion() # activate interactive mode
            self.fig, self.ax = plt.subplots(figsize=(4, 5))
            self.line1, = self.ax.plot(self.x, self.y, ".b")
            self.line1, = self.ax.plot(self.target_x, self.target_y, "*r")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        return observation

    def step(self, action):
        """
        Inputs:
            action : [linear_vel,  angular_velocity]
        """
        # Apply action on the robot:
        # calculate orientation of the robot after the displacements
        delta_theta = self._step_duration * action[1]
        self.theta += delta_theta
        # calculate position of the robot after the displacements
        delta_x = self._step_duration * action[0] * np.cos(self.theta)
        delta_y = self._step_duration * action[0] * np.sin(self.theta)
        self.x += delta_x
        self.y += delta_y

        # Simulate real time action taking
        if self.config['real_time']:
            sleep(self._step_duration)

        # Get observation
        observation = get_observation(self.x, self.y, self.theta, self.target_x, self.target_y)

        # Calculate Reward
        target_state = get_target_state(self.target_x, self.target_y)
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

    def render(self, mode="human"):
        if 'draw_coordinates' in self.config['render_mode']:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.plot(self.x, self.y, ".b")
            plt.show(block=False)

    def close(self):
        pass

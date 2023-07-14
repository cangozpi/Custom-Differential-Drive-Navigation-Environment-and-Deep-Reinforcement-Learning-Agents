from gym import Env, spaces
import numpy as np
from rl.diff_drive_GymEnv.env_utils import get_observation, get_reward, get_target_state, read_yaml_config
from time import sleep
import matplotlib.pyplot as plt


class DiffDrive_Env(Env):
    def __init__(self, config_path):
        super().__init__()
        # Read in parameters from config.yaml
        self.config = read_yaml_config(config_path)
        self.render_mode = self.config['render_mode']

        self.use_diff_drive_kinematics = self.config['use_diff_drive_kinematics'] if 'use_diff_drive_kinematics' in self.config else False
        self.l = self.config['l'] # distance btw the centers of the two wheels

        # Set gym spaces
        self.action_space = spaces.Box(-1 * np.ones((2, )), np.ones((2, )), dtype=np.float32)
        self.observation_space = spaces.Box(-float('inf') * np.ones((5, )), float('inf')*np.ones((5, )), dtype=np.float32)

        self._step_duration = self.config['step_duration'] * (10 ** -9) # how long each individual action is taken for. (i.e. time btw observations). In seconds
        self.max_episode_length = self.config['max_episode_length'] # max num steps before terminating the episode
        self._distance_threshold = self.config['distance_threshold'] # if robot is withing the reach of target smaller than this threshold value then goal state is reached
        self.cur_iteration = 0

        # Robot state wrt absolute world frame
        self.x = 0 # in meters
        self.y = 0 # in meters
        self.theta = 0 # in radians

        # target location wrt absolute world frame
        self.target_x = 1.5
        self.target_y = 1.5

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
            self.target_x = 1.5
            self.target_y = 1.5

        self.cur_iteration = 0

        observation = get_observation(self.x, self.y, self.theta, self.target_x, self.target_y).astype(self.observation_space.dtype)

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
            if self.use_diff_drive_kinematics == False:
                action : [linear_vel,  angular_velocity]
            else:
                action : [v_l (i.e. left wheel linear velocity),  v_r (i.e right wheel linear velocity)]
        """
        if self.use_diff_drive_kinematics:
            # Calculate Agent position after taking the action
            # For an explanation of the equations, refer to: https://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf
            v = action
            v_l = v[0]
            v_r = v[1]
            l = self.l
            theta = self.theta
            x = self.x
            y = self.y
            delta_t = self._step_duration

            if v_r == v_l: # moves straight (special case)
                out = np.array([
                    [x + (v_l * np.cos(theta) * delta_t)],
                    [y + (v_l * np.sin(theta) * delta_t)],
                    [theta] # note that theta does not change (there is no rotation when v_l = v_r)
                ]) # [3, 1]

            else:
                R = (l / 2) * ((v_l + v_r) / (v_r - v_l)) 
                w = (v_r - v_l) / l

                ICC_x = x - (R * np.sin(theta))
                ICC_y = y + (R * np.cos(theta))

                M_rotation = np.array([
                    [np.cos(w * delta_t), -np.sin(w * delta_t), 0],
                    [np.sin(w * delta_t), np.cos(w * delta_t), 0],
                    [0, 0, 1]
                ]) # [3, 3]

                M_translatedToOrigin = np.array([
                    [x - ICC_x],
                    [y - ICC_y],
                    [theta]
                ]) # [3, 1]

                M_translationToOriginalPosition = np.array([
                    [ICC_x],
                    [ICC_y],
                    [w * delta_t]
                ]) # [3, 1]

                out = np.matmul(M_rotation, M_translatedToOrigin) + M_translationToOriginalPosition # [3, 1]

            x_prime = out[0, 0]
            y_prime = out[1, 0]
            theta_prime = out[2, 0]

            # Update agent position
            self.x = x_prime
            self.y = y_prime
            self.theta = theta_prime

        else:
            # Apply action on the robot:
            # calculate orientation of the robot after the displacements
            delta_theta = self._step_duration * action[1]
            self.theta += delta_theta
            # calculate position of the robot after the displacements
            delta_x = self._step_duration * action[0] * np.cos(self.theta)
            delta_y = self._step_duration * action[0] * np.sin(self.theta)
            self.x += delta_x
            self.y += delta_y


        # Get observation
        observation = get_observation(self.x, self.y, self.theta, self.target_x, self.target_y).astype(self.observation_space.dtype)

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


        # Simulate real time action taking
        if self.config['real_time']:
            sleep(self._step_duration)

        self.cur_iteration += 1
        self.render()

        return observation, reward, done, info

    def render(self, mode="human"):
        if 'draw_coordinates' in self.config['render_mode']:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.plot(self.x, self.y, ".b")
            plt.show(block=False)

    def close(self):
        pass

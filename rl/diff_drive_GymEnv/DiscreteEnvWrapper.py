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
        self.action_space = spaces.Discrete(4) # {0, 1, 2, 3} , (i.e. left_rotation, forward_move, right_rotation, stay)

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
            action : [0, 1, 2, 3], i.e. [left_rotation, forward_move, right_rotation, stay]
        """
        assert disc_action < 4, f'{disc_action} is not supported'
        cont_action = self.convert_discrete_action_to_continuous_action(disc_action)
        return super().step(cont_action)
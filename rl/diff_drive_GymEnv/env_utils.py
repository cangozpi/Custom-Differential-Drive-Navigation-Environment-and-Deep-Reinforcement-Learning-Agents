import numpy as np
import yaml

def read_yaml_config(config_path):
    """
    Inputs:
        config_path (str): path to config.yaml file
    Outputs:
        config (dict): parsed config.yaml parameters
    """
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_observation(agent_x, agent_y, agent_theta, target_x, target_y):
    observation = [agent_x, agent_y, agent_theta, target_x, target_y]
    return np.array(observation)

def get_reward(observation, target, distance_threshold=0.1):
    # Check for distance to target
    d = [observation[0] - target[0], observation[1] - target[1]]
    l2_distance_to_target = np.linalg.norm(d)

    # Check if robot is close to target within the threshold values
    if l2_distance_to_target <= distance_threshold: # goal state is reached
        return 100 # target reach bonus

    # L2 distance cost reward
    return -l2_distance_to_target

def get_target_state(target_x, target_y):
    return np.array([target_x, target_y])

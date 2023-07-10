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

def get_reward(observation, target):
    # L2 distance cost reward
    d = [observation[0] - target[0], observation[1] - target[1]]
    l2_distance = np.linalg.norm(d)
    return -l2_distance

def get_target_state(target_x, target_y):
    return np.array([target_x, target_y])

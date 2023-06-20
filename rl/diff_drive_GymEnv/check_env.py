from DiffDrive_env import DiffDrive_Env

verbose = True

def main():
    # Initialize Diff_drive environment
    env = DiffDrive_Env()

    cur_episode = 0
    while True:
        # Samplee a random action
        action = env.action_space.sample()

        # Take the action
        observation, reward, done, info = env.step(action)

        # Print info
        if verbose:
            print(f'episode: {cur_episode}, iteration: {env.cur_iteration}/{env.max_episode_length} | action:{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f} | robot_state: {observation[0]:.2f}, {observation[1]:.2f}, {observation[2]:.2f} | target_state: {env.target_x:.2f}, {env.target_y:.2f} | done: {done:.2f}')

        if done:
            observation = env.reset()
            cur_episode += 1



if __name__ == "__main__":
    main()
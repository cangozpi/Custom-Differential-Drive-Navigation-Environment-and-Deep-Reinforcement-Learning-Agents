# Custom Differential Drive Navigation Environment and Deep Reinforcement Learning Agents

This repository implements a toy differential drive OpenAI Gym environment, along with some deep reinforcement learning algorithms to train agents on it. It supports custom implementations (pytorch) of the rl algorithms along with some implementations from sb3 which can easily be extended. Environment supports both discrete and continuous action spaces.

---

### Running:
Change the parameters in _rl/config/*.yaml_ files for setting the hyperparameters of the rl algorithm, configuring the simulation options, and alternating between training and testing your agents. Under _/rl/_ you will find deep rl algorithms which you can try to use for solving the algorithm.
Run results are logged into tensorboard which will create _logs_tensorboard_ directory (for custom implementations), or _tb_logs_ (for sb3 implementations).

* Sample run for DQN training:
    ```bash
    python3 rl/DQN/train_my_DQN_DiffDriveEnv.py
    ```
    To see logs:
    ```bash
    tensorboard --logdir logs_tensorboard
    ```

* Sample run for DDPG training:
    ```bash
    python3 rl/DDPG/train_my_DDPG_DiffDriveEnv.py
    ```
    To see logs:
    ```bash
    tensorboard --logdir logs_tensorboard
    ```

---

### Supported Algorithms:
1. DQN (pytorch)
2. DDPG (pytorch)
3. TD3 (pytorch)
4. VPG (pytorch)
5. PPO (sb3)
6. HardCodedAgent, (i.e. baseline) (numpy)

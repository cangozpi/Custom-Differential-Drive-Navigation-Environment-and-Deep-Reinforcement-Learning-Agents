## Differential Drive Simulation and Deep Deterministic Policy Gradients (DDPG)
---
A very simple differential drive Deep Reinforcement Learning simulation using OpenAI Gym and torch.

---

* Configure the DDPG model hyperparameters and some simulation parameters from _config/config\_DDPG\_DiffDrive\_env.yaml_ file.

* __To view Tensorboard Logs__:
    ```python
    python3 rl/DDPG/train_DDPG.py
    ```
    Training/Test stats are logged at the _logs\_tensorboard/_ directory.

* __To train DDPG agent__:
    ```python
    python3 rl/DDPG/train_DDPG.py
    ```
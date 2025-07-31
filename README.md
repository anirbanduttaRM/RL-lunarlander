🛰️ RL-LunarLander — Deep Q-Learning for LunarLander-v3
This project implements a Deep Q-Network (DQN) agent to solve the LunarLander-v3 environment using PyTorch. The agent learns to autonomously land a spacecraft with precision through reinforcement learning.

🚀 Project Highlights
✅ DQN from scratch using PyTorch with stable training

🧠 Customizable policy networks

🔁 Experience replay buffer and soft target updates

🗂️ Session-labeled logs for each training run

📊 Tracks performance: episode reward, loss, epsilon, and more

⚙️ Modular design for rapid experimentation

🌌 About LunarLander-v3
LunarLander-v3 is part of the classic control suite, simulating a lunar module attempting to land between flags.
Observations include:

Horizontal/vertical position

Velocities

Angle and angular velocity

Left/right leg contact indicators

Actions:

Do nothing

Fire left engine

Fire main engine

Fire right engine

The agent must land softly, upright, and centrally — without crashing.

📦 Dependencies
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Main libraries:

gymnasium (or gym if you're using Gym Classic)

torch

numpy

matplotlib

tqdm

🏗️ Project Structure
File / Folder	Description
train.py	Main training loop and hyperparameters
dqn_agent.py	DQN agent with policy/target networks
replay_buffer.py	Experience replay memory
logs/	CSV logs per training session
models/	Saved PyTorch model checkpoints

▶️ Run Training
bash
Copy
Edit
python train.py
Edit train.py to tweak:

MAX_EPISODES, BATCH_SIZE, GAMMA, TAU, EPSILON_DECAY, etc.

Model checkpoints and logs are saved every few episodes.

📊 Sample Training Log (Auto-Saved)
yaml
Copy
Edit
Episode,EpLenMean,EpRewMean,Loss,Epsilon
1420,920,201.3,0.019,0.08
Monitors reward improvement, loss convergence, and epsilon decay.

🔍 Key Experiment Takeaways
Decreasing tau slows target network updates → more stable but slower to adapt

Larger batch_size stabilizes gradients but needs more compute

Deep custom policy networks (e.g., 128-256-128) accelerate convergence

Logging each run allows comparison across different hyperparameters

🧪 Future Enhancements
 Add Double DQN for overestimation reduction

 Integrate Dueling Networks

 Add Prioritized Experience Replay

 Export models for web/demo inference

 Add interactive plots (e.g., using Plotly or WandB)

📜 License
MIT License


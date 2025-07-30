# ==========================================
# 🚀 TRAINING A DQN AGENT ON LUNARLANDER-V3
# ==========================================

# 🔧 Import the Gymnasium environment and Stable Baselines 3 DQN algorithm
import os                                     # For creating folders
import gymnasium as gym                       # For the game environment
from stable_baselines3 import DQN             # Deep Q-Network algorithm
from stable_baselines3.common.callbacks import CheckpointCallback  # To save checkpoints

# ------------------------------------------
# 🧱 STEP 1: Create the LunarLander Environment
# ------------------------------------------
# LunarLander-v3 is a 2D physics simulation where the goal is to land a lunar module softly.
# No rendering is needed here since we're just training (no visuals).
env = gym.make("LunarLander-v3")              # Environment handles simulation logic

# ------------------------------------------
# 💾 STEP 2: Set Up Checkpointing
# ------------------------------------------
# This saves intermediate models every 100,000 steps to avoid loss during crashes or long runs.
# Checkpoints are saved in ./checkpoints with filenames like dqn_lander_100000_steps.zip
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)    # Make the folder if it doesn't exist

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,                        # Save every 100,000 steps
    save_path=checkpoint_dir,                 # Directory to save checkpoints
    name_prefix="dqn_lander"                  # Filename prefix for saved models
)

# ------------------------------------------
# 🤖 STEP 3: Initialize the DQN Agent
# ------------------------------------------
# MlpPolicy = Multi-Layer Perceptron policy (fully connected neural net)
# Additional config improves stability for long training runs
model = DQN(
    "MlpPolicy",                              # Neural network type
    env,                                      # LunarLander environment
    learning_rate=1e-4,                       # Smaller learning rate for smoother learning
    buffer_size=100_000,                      # Replay buffer size
    learning_starts=10_000,                   # Start training after collecting 10k steps
    batch_size=64,                            # Batch size for updates
    tau=1.0,                                  # Soft update coefficient for target network
    gamma=0.99,                               # Discount factor
    train_freq=4,                             # Train every 4 steps
    target_update_interval=1_000,             # Target network update interval
    verbose=1                                 # Show training logs
)

# ------------------------------------------
# 🏋️ STEP 4: Train the Agent
# ------------------------------------------
# The agent will interact with the environment for 1,000,000 timesteps.
# Models will be checkpointed every 100k steps during training.
# 💡 Estimated time:
#    - Mid-range CPU (no GPU): ~2–4 hours
#    - High-end CPU: ~1–2 hours
#    - GPU (RTX 3060+): ~20–45 mins
#    - To resume, load a checkpoint and continue learning
model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback
)

# ------------------------------------------
# 💾 STEP 5: Save the Final Trained Model
# ------------------------------------------
# This saves two files:
#   - dqn_lander_final.zip (model weights and config)
#   - Additional metadata for loading later
model.save("dqn_lander_final")                # This creates dqn_lander_final.zip

# ------------------------------------------
# ✅ DONE
# ------------------------------------------
print("✅ Training complete. Final model saved as dqn_lander_final.zip")

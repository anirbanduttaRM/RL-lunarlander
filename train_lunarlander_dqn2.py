# ==========================================
# 🚀 TRAINING A DQN AGENT ON LUNARLANDER-V3
# ==========================================

# 🔧 Import the Gymnasium environment and Stable Baselines 3 DQN algorithm
import os                                     # For creating folders and file paths
import gymnasium as gym                       # For the LunarLander game environment
from stable_baselines3 import DQN             # Deep Q-Network implementation from Stable Baselines3
from stable_baselines3.common.callbacks import CheckpointCallback  # Auto-save checkpoints during training
from datetime import datetime                 # For timestamping training runs

# ------------------------------------------
# 🧱 STEP 1: Create the LunarLander Environment
# ------------------------------------------
# LunarLander-v3 is a 2D physics simulation. The goal is to land the module gently between the flags.
# The agent receives rewards based on landing quality and fuel efficiency.
env = gym.make("LunarLander-v3")              # Instantiates the game simulation

# ------------------------------------------
# 🏷️ STEP 2: Create a Timestamped Log Header
# ------------------------------------------
# Add a visual log marker before each training session so logs are easy to track
training_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("\n" + "=" * 60)
print(f"🚀 NEW TRAINING RUN STARTED — {training_id}")
print("=" * 60 + "\n")

# ------------------------------------------
# 💾 STEP 3: Set Up Checkpointing
# ------------------------------------------
# CheckpointCallback saves model files periodically during training to avoid losing progress.
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)    # Creates the folder if it doesn't already exist

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,                        # Save every 100,000 steps
    save_path=checkpoint_dir,                # Directory to store checkpoints
    name_prefix=f"dqn_lander_{training_id}"   # Include training timestamp in saved file name
)

# ------------------------------------------
# 🤖 STEP 4: Initialize the DQN Agent
# ------------------------------------------
# We use MlpPolicy (a fully connected neural network) for function approximation.
# These hyperparameters are optimized for better stability on LunarLander-v3.
model = DQN(
    "MlpPolicy",                              # Type of policy network to use (Multi-Layer Perceptron)
    env,                                      # The training environment
    learning_rate=1e-4,                       # Smaller LR helps prevent unstable learning
    buffer_size=100_000,                      # Size of the replay memory
    learning_starts=10_000,                   # Warm-up period before learning begins
    batch_size=64,                            # Number of samples per training batch
    tau=0.1,                                  # Target network update smoothing coefficient
    gamma=0.99,                               # Discount factor for future rewards
    train_freq=4,                             # Agent trains every 4 steps
    target_update_interval=1_000,             # Update target network every 1000 training steps
    verbose=1                                 # Show detailed logs during training
)

# ------------------------------------------
# 🏋️ STEP 5: Train the Agent
# ------------------------------------------
# This is the main loop where the agent improves over time by interacting with the environment.
# Training takes place over 1 million time steps. Checkpoints are saved regularly.
# 💡 TIP: You can later reload checkpoints and continue training from there.
model.learn(
    total_timesteps=1_000_000,                # Total number of time steps to train
    callback=checkpoint_callback              # Save model at intervals
)

# ------------------------------------------
# 💾 STEP 6: Save the Final Trained Model
# ------------------------------------------
# After training completes, we save the final model for later evaluation or reuse.
model.save(f"dqn_lander_final_{training_id}")  # Save the model with a timestamped name

# ------------------------------------------
# ✅ DONE
# ------------------------------------------
print("\n✅ Training complete.")
print(f"📦 Final model saved as: dqn_lander_final_{training_id}.zip")
print(f"🧠 Checkpoints available in: {checkpoint_dir}")

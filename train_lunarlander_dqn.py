# ==========================================
# 🚀 TRAINING A DQN AGENT ON LUNARLANDER-V2
# ==========================================

# 🔧 Import the Gymnasium environment and Stable Baselines 3 DQN algorithm
import gymnasium as gym                      # For the game environment
from stable_baselines3 import DQN            # Deep Q-Network algorithm

# ------------------------------------------
# 🧱 STEP 1: Create the LunarLander Environment
# ------------------------------------------
# LunarLander-v2 is a 2D physics simulation where the goal is to land a lunar module softly.
# No rendering is needed here since we're just training (no visuals).
env = gym.make("LunarLander-v3")           # Environment handles simulation logic

# ------------------------------------------
# 🤖 STEP 2: Initialize the DQN Agent
# ------------------------------------------
# MlpPolicy = Multi-Layer Perceptron policy (fully connected neural net)
# verbose=1 lets us see training progress in the terminal
model = DQN("MlpPolicy", env, verbose=1)

# ------------------------------------------
# 🏋️ STEP 3: Train the Agent
# ------------------------------------------
# The agent will interact with the environment for 200,000 timesteps.
# This is where most of the time is spent.
# 💡 Estimated time:
#    - Mid-range CPU (no GPU): ~20–40 mins
#    - High-end CPU: ~10–20 mins
#    - GPU (RTX 3060+): ~3–10 mins
#    - For quick tests, try 10_000 steps instead of 200_000
model.learn(total_timesteps=200_000)

# ------------------------------------------
# 💾 STEP 4: Save the Trained Model
# ------------------------------------------
# This saves two files:
#   - lunarlander_dqn.zip (model weights and config)
#   - Additional metadata for loading later
model.save("lunarlander_dqn")                # This creates lunarlander_dqn.zip

# ------------------------------------------
# ✅ DONE
# ------------------------------------------
print("✅ Model trained and saved as lunarlander_dqn.zip")

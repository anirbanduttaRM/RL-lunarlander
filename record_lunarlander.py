# record_lunarlander.py

# Import the necessary modules

# Gymnasium provides the LunarLander-v2 environment and other RL environments
import gymnasium as gym

# imageio is used to create an animated GIF from individual image frames
import imageio

# Import the Deep Q-Network implementation from stable-baselines3
from stable_baselines3 import DQN

# ------------------------------------------------------------------------------
# STEP 1: Load the trained model
# ------------------------------------------------------------------------------

# Load the previously trained DQN model from file.
# This loads the network architecture and the learned weights (policy).
# Make sure 'lunarlander_dqn.zip' exists in the same directory.
model = DQN.load("lunarlander_dqn")

# ------------------------------------------------------------------------------
# STEP 2: Initialize the environment with RGB frame rendering
# ------------------------------------------------------------------------------

# Create a new instance of the LunarLander-v2 environment with RGB array rendering.
# This will render each frame as a NumPy RGB image (instead of using a GUI window).
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Reset the environment to get the initial observation.
# This starts a new episode. The second returned value is 'info', which we ignore here.
obs, _ = env.reset()

# Create an empty list to store rendered RGB frames from each time step.
frames = []

# ------------------------------------------------------------------------------
# STEP 3: Run the agent in the environment and collect frames
# ------------------------------------------------------------------------------

# Run the agent for up to 500 steps (or until the episode ends earlier)
for _ in range(500):

    # Render the current frame and store it.
    # `env.render()` returns an RGB image of the current state (W x H x 3).
    frames.append(env.render())

    # Predict the next action using the trained model given the current observation.
    # `deterministic=True` ensures the action is always the same for the same state.
    action, _ = model.predict(obs, deterministic=True)

    # Apply the action to the environment and receive the next observation and info.
    # - obs: the new state
    # - reward: reward received after taking action
    # - done: True if the episode ends due to success/failure
    # - truncated: True if the episode ends due to time limits
    # - info: diagnostic metadata (ignored here)
    obs, reward, done, truncated, _ = env.step(action)

    # If the episode is over (either crashed or landed), exit the loop
    if done or truncated:
        break

# Close the environment to release memory and resources
env.close()

# ------------------------------------------------------------------------------
# STEP 4: Save the recorded frames as a GIF
# ------------------------------------------------------------------------------

# Use imageio to convert the list of RGB frames into an animated GIF.
# duration=1/30 → frame rate of ~30 FPS (fast enough for smooth playback)
imageio.mimsave("lander_run.gif", frames, duration=1/30)

# Print a message to confirm that the episode was recorded successfully
print("✅ Episode recorded to lander_run.gif")

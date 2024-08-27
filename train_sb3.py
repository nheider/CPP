import gymnasium as gym
from stable_baselines3 import PPO
import stable_baselines3
import RL_CPP 
import torch

from stable_baselines3.common.env_checker import check_env


device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else 
                       "cpu")


# Create the environment
env = gym.make("FieldEnv-v0")

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=0, device=device)


def render_agent(): 
    i = 0 
    terminated = False 
    truncated = False
    obs, _ = env.reset()
    while not terminated and not truncated and i <= 1000:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(f"Test Episode ended after", i, "steps")
            obs, _ = env.reset()
        i += 1


iterations = 20
for i in range(iterations): 
    model.learn(total_timesteps=2048, progress_bar=True)
    render_agent()

# Save the trained model
#model.save("ppo_pendulum")

# Test the trained agent


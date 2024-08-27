

import torch
import gymnasium as gym
from gymnasium import vector
from train import Agent
from RL_CPP.wrappers import VisualizeWrapper


def evaluate_agent(checkpoint_path, env_id, num_episodes=10, device="cpu"):
    """
    Loads the agent from a checkpoint and evaluates its performance on the environment.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        env_id (str): ID of the environment to evaluate the agent on.
        num_episodes (int): Number of episodes to run for evaluation.
        device (str): Device to use for the evaluation (default is "cpu").

    Returns:
        List[float]: List of episode returns.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # The agent expects a vector of environments, so we make a vector of 1 env 
    vis_env = VisualizeWrapper(gym.make(env_id))

    env = gym.make(env_id)
    
    # Create the agent and load the checkpoint
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()

    # Evaluate the agent
    episodic_returns = []
    for _ in range(num_episodes):
        observation, info = vis_env.reset()
        done = False
        total_reward = 0

        with torch.no_grad(): 
            while not done:
                action, _, _, _ = agent.get_action_and_value(torch.Tensor(observation).unsqueeze(0).to(device))
                observation, reward, terminated, truncated, info = vis_env.step(action.cpu().numpy()[0])
                print("TEST ENV:", terminated, truncated)
                total_reward += reward
                done = terminated or truncated

            episodic_returns.append(total_reward)
            env.close()

    return episodic_returns

checkpoint_path = "runs/FieldEnv-v0__train__1__1724252964/checkpoints/train_1.cleanrl_model"
env_id = "FieldEnv-v0"
episode_returns = evaluate_agent(checkpoint_path, env_id, num_episodes=10)
print(f"Test episode return: {sum(episode_returns) / len(episode_returns)}")
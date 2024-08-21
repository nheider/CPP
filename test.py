import torch
import train
import gymnasium as gym
import tyro 
import time 
from dataclasses import dataclass
import os 


@dataclass
class Args:
        # Algorithm specific arguments
        env_id: str = "FieldEnv-v0"
        """the id of the environment"""
        total_timesteps: int = 1000000
        """total timesteps of the experiments"""
        learning_rate: float = 3e-4
        """the learning rate of the optimizer"""
        num_envs: int = 1
        """the number of parallel game environments"""
        num_steps: int = 2048
        """the number of steps to run in each environment per policy rollout"""
        anneal_lr: bool = True
        """Toggle learning rate annealing for policy and value networks"""
        gamma: float = 0.99 # default 99
        """the discount factor gamma"""
        capture_video: bool = False
        """test"""
        exp_name: str = os.path.basename(__file__)[: -len(".py")]
        seed: int = 0



def load_and_test_agent(model_path, env_name, num_episodes=10):
    # Create the environment
    args = tyro.cli(Args)
    env_id = args.env_id

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    env = gym.make(env_id)

    def make_env(env_id, idx, capture_video, run_name, gamma):
        def thunk():
            env = gym.make(env_id)
            return env
        return thunk


    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    
    # Create an instance of the agent
    agent = train.Agent(envs)  # Adjust this if your Agent class requires any arguments
    
    # Load the model
    agent.load_state_dict(torch.load(model_path))
    agent.eval()  # Set to evaluation mode
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)  # Ensure this method is implemented in your agent class
            next_state, reward, done, _, _ = env.step(action)  # Adjust if env.step() returns more or fewer values
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward}")
    
    env.close()
    return avg_reward

# Usage
model_path = "runs/FieldEnv-v0__train__1__1724228970/checkpoints/train_5.cleanrl_model"
env_name = "FieldEnv-v0"
average_reward = load_and_test_agent(model_path, env_name)

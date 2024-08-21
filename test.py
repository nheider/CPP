import torch
import train
import gymnasium as gym

def load_and_test_agent(model_path, env_name, num_episodes=10):
    # Create the environment
    env = gym.make(env_name)
    
    # Load the model
    agent = train.Agent # Adjust as needed
    agent.load_state_dict(torch.load(model_path))
    agent.eval()  # Set to evaluation mode
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)  # Implement this method in your agent class
            next_state, reward, done, _ = env.step(action, visualization = True)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward}")
    
    env.close()
    return avg_reward

# Usage
model_path = "path/to/your/checkpoint.cleanrl_model"
env_name = "FieldEnv-v0" 
average_reward = load_and_test_agent(model_path, env_name)
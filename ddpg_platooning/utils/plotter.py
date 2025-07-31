import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(episode_rewards, critic_losses=None, actor_losses=None,
                          reward_components=None, save_path="training_results.png"):
    """
    Plot training results for the DDPG platooning experiment.
    
    Args:
        episode_rewards: List of rewards per episode
        critic_losses: List of critic loss values
        actor_losses: List of actor loss values
        reward_components: List of dicts containing 'main_rewards', 'jerk_penalties', 'RTG_values'
        save_path: File path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # Losses
    if critic_losses:
        axes[0, 1].plot(critic_losses, label='Critic Loss', alpha=0.7)
    if actor_losses:
        axes[0, 1].plot(actor_losses, label='Actor Loss', alpha=0.7)
    axes[0, 1].legend()
    axes[0, 1].set_title('Training Losses')
    axes[0, 1].grid(True)

    # Reward components
    if reward_components:
        main_rewards = [np.mean(ep.get('main_rewards', [0])) for ep in reward_components[-100:]]
        jerk_penalties = [np.mean(ep.get('jerk_penalties', [0])) for ep in reward_components[-100:]]
        axes[1, 0].plot(main_rewards, label='Main Reward', alpha=0.7)
        axes[1, 0].plot(jerk_penalties, label='Jerk Penalty', alpha=0.7)
        axes[1, 0].legend()
        axes[1, 0].set_title('Reward Components (Last 100 Episodes)')
        axes[1, 0].grid(True)

        rtg_values = [np.mean(ep.get('RTG_values', [0])) for ep in reward_components[-100:]]
        axes[1, 1].plot(rtg_values, alpha=0.7)
        axes[1, 1].axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='RTG_min')
        axes[1, 1].axhline(y=4.0, color='r', linestyle='--', alpha=0.5, label='RTG_max')
        axes[1, 1].legend()
        axes[1, 1].set_title('Relative Time Gap (Last 100 Episodes)')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Training plots saved to {save_path}")

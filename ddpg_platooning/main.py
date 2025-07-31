import argparse
import logging
import torch
import numpy as np

from .sumo_generator import SUMONetworkGenerator
from .reward_function import PaperCompliantRewardFunction
from .environment import PaperCompliantPlatoonEnvironment
from .agent import PaperCompliantDDPGAgent
from .trainer import PlatoonTrainer
from .tester import PlatoonTester


def main():
    parser = argparse.ArgumentParser(description='Modular DDPG Vehicle Platooning (Paper Compliant)')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--model', type=str, default='paper_compliant_ddpg.pth', help='Model file path')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=== Modular DDPG Vehicle Platooning ===")
    logger.info("Paper: Multi-Task Vehicle Platoon Control: A Deep Deterministic Policy Gradient Approach")

    # Generate SUMO files
    logger.info("Generating SUMO network files...")
    generator = SUMONetworkGenerator()
    net_file = generator.create_network()
    rou_file = generator.create_route_file()
    cfg_file = generator.create_config_file(net_file, rou_file)
    logger.info(f"SUMO files created in: {generator.output_dir}")

    # Reward function
    reward_function = PaperCompliantRewardFunction(
        L=3.2, d_d=4.0, delta_T=0.25,
        RTG_min=2.0, RTG_max=4.0, alpha=0.1,
        a_max_acc=3.5, a_max_dec=-3.5
    )

    # Environment
    env = PaperCompliantPlatoonEnvironment(
        cfg_file, use_gui=args.gui, max_steps=100, reward_function=reward_function
    )

    # Agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PaperCompliantDDPGAgent(
        state_dim=4, action_dim=1, max_action=3.5,
        lr_actor=1e-4, lr_critic=1e-3, gamma=0.99,
        tau=0.005, device=device
    )
    logger.info(f"Using device: {device}")

    # Train or test
    if args.train:
        logger.info(f"Starting training for {args.episodes} episodes...")
        trainer = PlatoonTrainer(env, agent, logger)
        results = trainer.train(
            episodes=args.episodes,
            batch_size=args.batch_size,
            model_path=args.model
        )
        trainer.plot_training_results("paper_compliant_training_results.png")

        if results['episode_rewards']:
            logger.info(f"Average reward: {np.mean(results['episode_rewards']):.2f}")
            logger.info(f"Best reward: {np.max(results['episode_rewards']):.2f}")
            logger.info(f"Final reward: {results['episode_rewards'][-1]:.2f}")
            logger.info(f"Final noise std: {agent.noise_std:.3f}")

    elif args.test:
        logger.info("Starting testing of trained model...")
        tester = PlatoonTester(env, agent, logger)
        tester.test(num_episodes=5, model_path=args.model)

    else:
        logger.warning("No mode selected. Use --train or --test.")
        parser.print_help()

    env.close()
    logger.info("Execution completed.")


if __name__ == "__main__":
    main()

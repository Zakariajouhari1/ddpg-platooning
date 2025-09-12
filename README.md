# Multi-Task Vehicle Platoon Control: A Deep Deterministic Policy Gradient Approach

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![SUMO](https://img.shields.io/badge/SUMO-1.8+-green.svg)](https://sumo.dlr.de/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository implements a **Deep Deterministic Policy Gradient (DDPG)** based vehicle platooning system that addresses three critical aspects of autonomous vehicle coordination in a single unified controller. The implementation transforms theoretical research into practical code, enabling vehicles to autonomously form and maintain platoons while ensuring safety and efficiency.

**Based on**: *"Multi-Task Vehicle Platoon Control: A Deep Deterministic Policy Gradient Approach"* by Berahman et al. (2022)

## Key Features

### 🚗 **Multi-Task Control**
- **Speed Tracking**: Follows leading vehicle's dynamic speed profiles with minimal error
- **Gap Maintenance**: Maintains constant 4m inter-vehicle distance during all maneuvers  
- **Gap Closing/Opening**: Handles platoon joining (15s) and leaving (7s) maneuvers safely
- **Unified Architecture**: Single DDPG controller manages all three tasks simultaneously

### 🧠 **Advanced DDPG Implementation**
- **Actor-Critic Networks**: 6-layer architecture with [400, 300, 200, 50] neurons
- **Experience Replay**: 1M buffer size for stable learning
- **3-Step TD Learning**: Enhanced temporal difference for better predictions
- **Target Networks**: Soft updates (τ=0.005) for training stability

### 🛣️ **Realistic Simulation Environment**
- **SUMO Integration**: Full integration with Simulation of Urban Mobility
- **TraCI Interface**: Real-time vehicle control and monitoring
- **Highway Scenarios**: Realistic traffic conditions and vehicle dynamics
- **Performance Metrics**: Comprehensive evaluation including string stability

### 📊 **Proven Performance**
- **Superior Gap Control**: 40cm max error vs 65cm (CACC)
- **Better Speed Tracking**: 334.13 m/s total error vs 369.31 m/s (CACC)
- **String Stability**: Perturbations decay within 10 seconds
- **Zero Collisions**: Extensive testing with safety guarantees

## Repository Structure

\`\`\`
ddpg-platooning/
├── ddpg_platooning/          # Core DDPG implementation
│   ├── ddpg_agent.py         # Main DDPG agent with training/evaluation
│   ├── environment.py        # Platooning environment wrapper
│   ├── networks.py           # Actor-critic neural networks
│   ├── replay_buffer.py      # Experience replay implementation
│   └── utils.py              # Reward functions and utilities
├── sumo_files/               # SUMO simulation configuration
│   ├── highway.net.xml       # Highway network topology
│   ├── routes.rou.xml        # Vehicle route definitions
│   └── config.sumocfg        # Main SUMO configuration
├── models/                   # Pre-trained models and checkpoints
│   ├── paper_compliant_ddpg.pth  # Validated model from paper
│   └── checkpoints/          # Training checkpoints
├── results/                  # Training logs and evaluation results
│   ├── training_curves/      # Reward convergence plots
│   └── evaluation_metrics/   # Performance analysis
├── docs/                     # Additional documentation
│   ├── INSTALLATION.md       # Detailed setup guide
│   ├── API_REFERENCE.md      # Code documentation
│   └── TRAINING.md           # Training procedures
└── requirements.txt          # Python dependencies
\`\`\`

## Installation

### Prerequisites

**1. Python 3.8+**
\`\`\`bash
python --version  # Ensure 3.8 or higher
\`\`\`

**2. SUMO Traffic Simulator**
\`\`\`bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# macOS (Homebrew)
brew install sumo

# Windows
# Download from: https://sumo.dlr.de/docs/Downloads.php
# Add SUMO/bin to your PATH
\`\`\`

**3. Verify SUMO Installation**
\`\`\`bash
sumo --version  # Should show SUMO version
\`\`\`

### Setup

**1. Clone Repository**
\`\`\`bash
git clone https://github.com/Zakariajouhari1/ddpg-platooning.git
cd ddpg-platooning
\`\`\`

**2. Install Dependencies**
\`\`\`bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
\`\`\`

**3. Configure SUMO Environment**
\`\`\`bash
# Linux/macOS
export SUMO_HOME="/usr/share/sumo"

# Windows (adjust path to your SUMO installation)
set SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
\`\`\`

**4. Verify Installation**
\`\`\`bash
python -c "import traci; print('TraCI imported successfully')"
\`\`\`

## Quick Start

### Training from Scratch
\`\`\`python
from ddpg_platooning.ddpg_agent import DDPGAgent
from ddpg_platooning.environment import PlatooningEnvironment

# Initialize environment
env = PlatooningEnvironment(
    sumo_config="sumo_files/config.sumocfg",
    num_vehicles=8,
    desired_gap=4.0,
    max_steps=100
)

# Create DDPG agent
agent = DDPGAgent(
    state_dim=4,
    action_dim=1,
    max_action=3.5,
    lr_actor=1e-4,
    lr_critic=1e-4
)

# Train for 3000 episodes (paper configuration)
agent.train(env, episodes=3000, save_interval=500)
\`\`\`

### Using Pre-trained Model
\`\`\`python
# Load paper-compliant model
agent = DDPGAgent.load_model("models/paper_compliant_ddpg.pth")

# Evaluate performance
results = agent.evaluate(env, episodes=100)
print(f"Average Reward: {results['avg_reward']:.2f}")
print(f"Success Rate: {results['success_rate']:.1%}")
\`\`\`

### Running Evaluation Scenarios
\`\`\`python
# Gap closing/opening test
results_gap = agent.evaluate_gap_maneuvers(env)

# Speed tracking test  
results_speed = agent.evaluate_speed_tracking(env)

# String stability test
results_stability = agent.evaluate_string_stability(env)
\`\`\`

## Algorithm Details

### State Space (4D)
- **d_(i-1,i)**: Inter-vehicle distance (meters)
- **e_i**: Gap error from desired 4m distance (meters)  
- **v_i**: Current vehicle speed (m/s)
- **v_leader**: Leading vehicle speed (m/s)

### Action Space
- **Continuous**: Acceleration ∈ [-3.5, 3.5] m/s²
- **Mapped**: From tanh output [-1, 1] to acceleration range

### Multi-Objective Reward Function
\`\`\`python
# Core reward components
if gap_error_increased:
    reward = -1.0  # Punishment for wrong direction
elif RTG_min <= relative_time_gap <= RTG_max:
    reward = percentage_error_deviation  # Reward improvement
else:
    reward = -RTG_error  # Punishment for poor speed control

# Add comfort penalty
reward += jerk_penalty * alpha

# Collision avoidance
if collision_detected:
    reward = -10.0  # Major punishment
\`\`\`

### Network Architecture
**Actor Network**: State → Action
- Input: 4D state vector
- Hidden: [400, 300, 200, 50] with ReLU
- Output: 1D action with Tanh

**Critic Network**: State + Action → Q-Value  
- Input: 4D state + 1D action
- Hidden: [400, 300, 200, 50] with ReLU
- Output: Scalar Q-value

## Performance Benchmarks

| Metric | DDPG (Ours) | CACC Baseline | Improvement |
|--------|-------------|---------------|-------------|
| **Total Distance Gap Error** | 25,627 m | 25,846 m | **0.8% better** |
| **Total Speed Difference** | 334.13 m/s | 369.31 m/s | **9.5% better** |
| **Maximum Gap Error** | **40 cm** | 65 cm | **38% better** |
| **Gap Closing Time** | **15 seconds** | >30 seconds | **50% faster** |
| **Gap Opening Time** | **7 seconds** | >15 seconds | **53% faster** |
| **String Stability Recovery** | **10 seconds** | >20 seconds | **50% faster** |

## Configuration

### Hyperparameters
\`\`\`python
CONFIG = {
    # Network parameters
    'actor_lr': 1e-4,
    'critic_lr': 1e-4, 
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 256,
    'buffer_size': 1_000_000,
    
    # Environment parameters  
    'max_acceleration': 3.5,
    'max_deceleration': -3.5,
    'desired_gap': 4.0,
    'vehicle_length': 3.2,
    'simulation_step': 0.25,
    
    # Reward parameters
    'RTG_max': 4.0,
    'RTG_min': 2.0, 
    'alpha': 0.1,  # Jerk penalty weight
}
\`\`\`

## Citation

```bibtex
@article{berahman2022multi,
  title={Multi-Task Vehicle Platoon Control: A Deep Deterministic Policy Gradient Approach},
  author={Berahman, Mehran and Rostami-Shahrbabaki, Majid and Bogenberger, Klaus},
  journal={Future Transportation},
  volume={2},
  number={4},
  pages={1028--1046},
  year={2022},
  publisher={MDPI},
  doi={10.3390/futuretransp2040057}
}

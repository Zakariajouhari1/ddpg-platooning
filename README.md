# 🧠 DDPG-Based Autonomous Vehicle Platooning

A **research-oriented implementation** of **Deep Deterministic Policy Gradient (DDPG)** for autonomous vehicle platooning in highway scenarios. This project focuses on **continuous control**, **inter-vehicle coordination**, and **safe gap regulation** using reinforcement learning.

> Designed for reproducible experiments and academic research using **SUMO + TraCI**.

---

## 🔍 Why This Project Exists

Classical control approaches struggle to generalize across dynamic traffic scenarios. This project explores whether **model-free deep reinforcement learning** can:

* Maintain safe inter-vehicle distances
* Match leader speed smoothly
* Reduce oscillations and string instability
* Scale from 2 vehicles to multi-vehicle platoons

The implementation is intentionally **minimal, transparent, and modifiable** to support research extensions.

---

## ⚙️ Core Features

* ✅ Continuous action space (acceleration control)
* ✅ DDPG with Actor–Critic neural networks
* ✅ SUMO traffic simulator integration via TraCI
* ✅ Configurable platoon size
* ✅ Deterministic evaluation mode
* ✅ Training curves and logging

---

## 🧩 System Overview

```
Agent (DDPG)
   │
   ▼
Action (Acceleration)
   │
   ▼
SUMO Environment ──► State (speed, gap, relative velocity)
   ▲
   │
Reward (gap error, collision penalty, smoothness)
```

---

## 📦 Requirements

* Python ≥ 3.8
* SUMO ≥ 1.9
* PyTorch
* NumPy
* Matplotlib

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure SUMO is accessible:

```bash
export SUMO_HOME=/path/to/sumo
export PATH=$SUMO_HOME/bin:$PATH
```

---

## 🚀 Running the Project

### Train the Agent

```bash
python train.py
```

Training parameters (episodes, noise, buffer size, etc.) can be modified directly in the script for clarity and experimentation.

### Evaluate a Trained Model

```bash
python evaluate.py
```

Evaluation runs without exploration noise and visualizes platoon behavior in SUMO.

---

## 🧪 State, Action, Reward Design

### State Vector

* Ego vehicle speed
* Distance to leader
* Relative velocity

### Action

* Continuous longitudinal acceleration

### Reward Function

Encourages:

* Target gap tracking
* Speed matching
* Smooth control

Penalizes:

* Collisions
* Unsafe gaps
* Aggressive acceleration

---

## 📊 Results

The trained agent successfully:

* Maintains stable gaps
* Tracks leader velocity
* Avoids collisions in standard highway scenarios

Plots and logs are generated during training for analysis.

---

## 📁 Project Structure

```
ddpg-platooning/
├── train.py          # Training loop
├── evaluate.py       # Policy evaluation
├── agent/            # Actor–Critic networks
├── environment/      # SUMO interface
├── utils/            # Replay buffer, noise
└── requirements.txt
```

---

## 📚 References

* Lillicrap et al., *Continuous Control with Deep Reinforcement Learning*
* SUMO Traffic Simulator Documentation
* TraCI API Reference

---

## 🔮 Future Work

* Multi-agent decentralized learning
* Hybrid model-based + RL control
* String stability metrics
* Curriculum learning
* Comparison with PID / MPC baselines

---

## 🧠 Research Use

This repository is suitable for:

* Academic experiments
* RL benchmarking
* Control–learning hybrid research
* Conference paper extensions

If you build upon this work, a citation would be appreciated.

---

## 📜 License

MIT License

---

## ✉️ Contact

**Zakaria Jouhari**
Electrical Engineering / Autonomous Systems
GitHub: [https://github.com/Zakariajouhari1](https://github.com/Zakariajouhari1)

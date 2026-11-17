# Intelligent Agent Navigation in the Wumpus World via Reinforcement Learning (Q-Learning)

An object-oriented, modular Python implementation of the classic **Wumpus World** Artificial Intelligence benchmark solved using **Model-Free Tabular Q-Learning**, complete with sensory percept simulation, step-by-step trajectory visualization, and multi-episode statistical benchmarking.

---

## 📌 Project Overview

The **Wumpus World** is a canonical testbed in Computational Intelligence and Artificial Intelligence for evaluating sequential decision-making under uncertainty and partial observability. In this project, we formulate the $4 \times 4$ grid environment as a **Markov Decision Process (MDP)** and train an autonomous agent from scratch—with zero prior knowledge of game rules or hazard locations—to locate the gold, avoid lethal Pits and the Wumpus, and return safely to the exit cell `(0,0)`.

### Key Features

- **Object-Oriented Environment (`WumpusWorld`):** Supports both the classic Russell & Norvig canonical benchmark layout and procedural randomized map generation with guaranteed start-zone safety.
- **Full Sensory Percept Engine:** Simulates orthogonal `Stench` and `Breeze`, local `Glitter`, boundary `Bump`, and global `Scream` events.
- **Tabular Q-Learning Agent (`QLearningAgent`):** Implements Temporal-Difference (TD) Bellman updates, exponential $\epsilon$-greedy exploration decay, random tie-breaking, and sensory-guided reflexes.
- **Publication-Quality Visualizations:** Generates dual-panel convergence curves (rolling reward & win rate vs. $\epsilon$), multi-frame trajectory storyboards, and comparative performance bar charts.
- **Quantitative Evaluation Suite:** Benchmarks the learned policy across $200$ episodes against a uniform random baseline and tests cross-map generalization on $200$ unseen procedural worlds.

---

## 🗂️ Repository Structure

```text
├── Wumpus_World_QLearning.ipynb   # Main Jupyter Notebook with academic explanations, code, and plots
└── README.md                      # Project documentation and experimental summary
```

---

## ⚙️ Installation & Setup

Ensure you have **Python 3.8+** installed along with the required scientific computing libraries:

```bash
pip install numpy matplotlib jupyter

```

To launch the notebook:

```bash
jupyter notebook Wumpus_World_QLearning.ipynb

```

---

## 🧠 Methodology & MDP Formulation

### State & Action Spaces

- **State Representation ($s_t$):**

$$s_t = \langle \text{row}, \text{col}, \text{has-gold}, \text{wumpus-alive}, \text{Breeze}, \text{Stench}, \text{Glitter} \rangle$$

- **Action Space ($\mathcal{A}$):** `UP (0)`, `DOWN (1)`, `LEFT (2)`, `RIGHT (3)`, `GRAB (4)`, `SHOOT (5)`

### Bellman Optimality Update

At each step, the agent updates its action-value table $Q(s, a)$ via:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big[ r_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') - Q(s_t, a_t) \Big]$$

- **Learning Rate ($\alpha$):** `0.15`
- **Discount Factor ($\gamma$):** `0.95`
- **Exploration Decay ($\epsilon$):** Exponentially decayed from `1.0` to `0.02` (`decay = 0.996`)

### Reward Structure

| Event / Action                 | Reward  | Purpose                                            |
| ------------------------------ | ------- | -------------------------------------------------- |
| Escape to `(0,0)` with Gold    | `+1000` | Primary win objective                              |
| Grabbing the Gold              | `+250`  | Intermediate reward shaping for faster convergence |
| Killing the Wumpus             | `+50`   | Threat elimination incentive                       |
| Step Taken                     | `-1`    | Enforces shortest-path optimality                  |
| Firing the Arrow               | `-10`   | Penalizes unnecessary resource expenditure         |
| Wall Bump / Invalid Grab       | `-5`    | Discourages invalid actions                        |
| Pit or Living Wumpus Collision | `-1000` | Terminal hazard penalty                            |

---

## 📊 Experimental Results ($N = 200$ Evaluation Episodes)

| Agent Configuration                   | Win Rate (%)        | Gold Retrieval (%) | Survival Rate (%)    | Mean Score       |
| ------------------------------------- | ------------------- | ------------------ | -------------------- | ---------------- |
| **Random Baseline (Canonical Map)**   | ~0.5%               | ~2.0%              | ~1.5%                | ~ -960.0         |
| **Tabular Q-Learning (Trained Map)**  | **100.0%**          | **100.0%**         | **100.0%**           | **+1238.0**      |
| **Percept Q-Agent (200 Unseen Maps)** | Improved vs. Random | Moderate           | Significantly Higher | Substantial Gain |

_(Note: Exact unseen-map metrics depend on procedural pit density and random seed initialization.)_

---

## 🚀 Future Improvements

- **Deep Q-Networks (DQN) & DRQN:** Incorporating recurrent neural architectures (LSTM/GRU) to maintain episodic memory across time steps for full POMDP generalization on unseen $N \times N$ grids.
- **Neuro-Symbolic Safety Shielding:** Integrating propositional logic constraints to mask provably fatal moves during early exploration.

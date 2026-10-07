# 🤖 MECORA AUTONAV

### AI-Based Autonomous Navigation & Obstacle Avoidance System

MECORA AUTONAV is an **AI-powered autonomous navigation system** designed to simulate how a mobile robot can perceive its environment, make navigation decisions, avoid obstacles, and reach a target autonomously.

The project combines **Machine Learning, virtual sensors, robotics control, and a safety layer** inside a 2D simulation environment — without requiring physical robot hardware.

---

## 🚀 Project Overview

The objective of MECORA AUTONAV is to develop a software-based autonomous robot capable of:

* 🧠 Making navigation decisions using a neural network
* 👁️ Perceiving obstacles through virtual sensors
* 🎯 Navigating toward a target
* 🚧 Avoiding obstacles
* 🛡️ Using a safety controller to override unsafe ML decisions
* 🌍 Operating in randomized environments
* 📊 Evaluating performance through automated benchmarks

### System Architecture

```text
                  ┌─────────────────────┐
                  │   Virtual Sensors   │
                  │                     │
                  │  Left / Front /     │
                  │       Right         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Target Direction  │
                  │   Target Distance   │
                  │   Previous Action   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    V5 Neural Net    │
                  │    MLP Classifier   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Navigation Action   │
                  │                     │
                  │ FORWARD             │
                  │ FORWARD_LEFT        │
                  │ FORWARD_RIGHT       │
                  │ TURN_LEFT           │
                  │ TURN_RIGHT          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Safety Controller  │
                  │                     │
                  │ Collision Prevention│
                  │ Emergency Override  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Autonomous Robot  │
                  └──────────┬──────────┘
                             │
                             ▼
                         🎯 TARGET
```

---

# 🧠 Machine Learning

MECORA AUTONAV V5 uses a **Multi-Layer Perceptron (MLP)** classifier from Scikit-learn.

### Model

```text
Pipeline
 ├── StandardScaler
 └── MLPClassifier
      ├── Hidden Layer: 128
      ├── Hidden Layer: 64
      └── Hidden Layer: 32
```

### Configuration

```text
Activation:        ReLU
Optimizer:         Adam
Learning rate:     0.001
Maximum iterations: 700
Early stopping:    Enabled
Validation:        15%
Random state:      42
```

---

# 📡 Virtual Sensors

The robot uses three simulated distance sensors:

```text
              FRONT
                │
                │
          ┌─────┴─────┐
          │   🤖      │
          └───────────┘
          ╱           ╲
       LEFT           RIGHT
```

The sensors provide information about nearby obstacles.

### Model inputs

The V5 model receives:

| Input             | Description                    |
| ----------------- | ------------------------------ |
| `left_sensor`     | Distance detected on the left  |
| `front_sensor`    | Distance detected in front     |
| `right_sensor`    | Distance detected on the right |
| `target_distance` | Distance to target             |
| `target_angle`    | Direction of target            |
| `previous_action` | Previous navigation action     |

Sensor noise is also introduced during dataset generation to make the model more robust to imperfect measurements.

---

# 🎮 Navigation Actions

The neural network predicts one of five actions:

```text
FORWARD
FORWARD_LEFT
FORWARD_RIGHT
TURN_LEFT
TURN_RIGHT
```

The predicted action is then passed through the safety controller before being applied to the robot.

---

# 🛡️ Safety Controller

The neural network is not allowed to blindly control the robot.

A separate safety layer monitors the environment and can override the ML decision when a collision becomes likely.

```text
ML Decision
     │
     ▼
Safety Controller
     │
     ├── Safe → Execute ML action
     │
     └── Dangerous → Override
                         │
                         ▼
                  Safe corrective action
```

This hybrid architecture combines:

**Machine Learning + Deterministic Safety**

which is particularly useful for autonomous robotics systems.

---

# 📊 V5 Benchmark

The V5 model was evaluated over **100 randomized simulation episodes**.

| Metric                   |    Result |
| ------------------------ | --------: |
| Episodes                 |       100 |
| Successful episodes      |       100 |
| Failed episodes          |         0 |
| **Success rate**         |  **100%** |
| Average collisions       |  **0.03** |
| Average steps            | **95.25** |
| Average path length      |  **6.83** |
| Average safety overrides | **15.23** |
| Average final distance   |  **0.41** |
| Average successful steps | **95.25** |

### Result

> 🏆 **100% success rate across 100 randomized episodes**

The benchmark demonstrates that the V5 system can consistently reach its target while maintaining a low collision rate.

---

# 🧪 Simulation

MECORA AUTONAV provides a 2D simulation environment containing:

* 🤖 Autonomous robot
* 🎯 Target position
* 🧱 Randomized obstacles
* 📡 Virtual distance sensors
* 🛡️ Safety controller
* 🧠 Neural-network decision system
* 📈 Performance evaluation

The environment can be randomized to test the navigation system under different conditions.

---

# 📁 Project Structure

```text
MECORA-AUTONAV/
│
├── data/
│   └── robot_training_data_v5.csv
│
├── docs/
│   └── architecture.md
│
├── results/
│   └── benchmark_v5.txt
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── generate_dataset_v5.py
├── train_model_v5.py
├── evaluate_model_v5.py
│
├── robot_sim_v5.py
│
└── robot_ml_model_v5.joblib
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ahmedbendag31-creator/MECORA-AUTONAV.git
```

Enter the project:

```bash
cd MECORA-AUTONAV
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Simulator

Launch the autonomous robot:

```bash
python robot_sim_v5.py
```

### Controls

| Key   | Function                    |
| ----- | --------------------------- |
| `L`   | ML Autonomous Mode          |
| `M`   | Manual Mode                 |
| `R`   | Generate random environment |
| `ESC` | Exit                        |
| `↑`   | Move forward                |
| `←`   | Turn left                   |
| `→`   | Turn right                  |

---

# 📊 Run the Benchmark

To evaluate the V5 system:

```bash
python evaluate_model_v5.py
```

The evaluator runs multiple randomized episodes and reports:

* Success rate
* Failure rate
* Collisions
* Steps
* Path length
* Safety overrides
* Final distance

---

# 🧠 Generate Training Data

To generate a new V5 dataset:

```bash
python generate_dataset_v5.py
```

The generated dataset is stored in:

```text
data/robot_training_data_v5.csv
```

---

# 🎓 Train the Model

Train the neural network using:

```bash
python train_model_v5.py
```

The trained model is saved as:

```text
robot_ml_model_v5.joblib
```

---

# 📈 Development Evolution

MECORA AUTONAV was developed progressively through several versions.

```text
V1
 │
 ├── Basic robot navigation
 │
 ▼
V2
 │
 ├── Improved obstacle avoidance
 │
 ▼
V3
 │
 ├── Virtual sensors
 │
 ▼
V4
 │
 ├── Machine-learning navigation
 │
 ▼
V5
 │
 ├── Multi-input neural network
 ├── Sensor noise
 ├── Randomized environments
 ├── Five navigation actions
 ├── Safety controller
 └── Automated benchmarking
```

---

# 🌐 Web Version

A browser-based version of MECORA AUTONAV is also being developed.

The web version is intended to provide:

* Interactive robot simulation
* Browser-based visualization
* Real-time telemetry
* Virtual sensors
* Autonomous navigation
* Random environments
* Portfolio demonstration

The long-term objective is to make MECORA AUTONAV accessible directly from a web browser without requiring Python installation.

---

# 🔬 Future Development

Planned improvements include:

* [ ] Browser implementation of the actual V5 neural network
* [ ] Real-time model inference in JavaScript
* [ ] More complex environments
* [ ] Dynamic obstacles
* [ ] Reinforcement learning
* [ ] Path planning algorithms
* [ ] A* / Dijkstra integration
* [ ] ROS integration
* [ ] Real robot deployment
* [ ] LiDAR simulation
* [ ] Camera-based perception
* [ ] Embedded implementation
* [ ] Hardware prototype

---

# 🛠️ Technologies

* **Python**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Joblib**
* **Machine Learning**
* **Neural Networks**
* **Robotics Simulation**
* **Autonomous Navigation**
* **Control Systems**

---

# 🎯 Project Objectives

MECORA AUTONAV is part of a broader goal of exploring the intersection of:

```text
       🤖 Robotics
           +
       🧠 AI / ML
           +
       ⚙️ Control
           +
       💻 Software
           +
       📡 Embedded Systems
```

The project is designed as a foundation for future autonomous robotics and embedded-AI applications.

---

# 👨‍💻 Author

**Ahmed Bendag**

Mechatronics Engineering Student

Interested in:

* 🤖 Robotics
* 🧠 Artificial Intelligence
* 📊 Machine Learning
* ⚙️ Mechatronics
* 📡 Embedded Systems
* 🚗 Autonomous Systems

---

# ⭐ Project Status

**MECORA AUTONAV V5 — Completed & Benchmark Validated**

```text
Model trained                  ✓
Virtual sensors                ✓
Obstacle avoidance             ✓
Safety controller              ✓
Randomized environments        ✓
Automated benchmark            ✓
100-episode evaluation         ✓
100% benchmark success         ✓
Web version                    🚧 In development
Real hardware                  🔜 Future
```

---

## 📜 License

This project is currently intended as a personal educational and engineering portfolio project.

---

⭐ If you find this project interesting, consider starring the repository.

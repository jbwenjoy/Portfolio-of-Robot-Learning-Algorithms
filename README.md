# Portfolio-of-Robotics-Algorithms

My personal portfolio of robot learning algorithms, from course materials in 517 and 650 at UPenn.

* This repository is continuously updated with new algorithms and implementations as I learn them.

* Last updated: 2024-Apr-6

## Table of Contents

1. Optimal Control and Trajectory Optimization: LQR, MPC, Energy Shaping, iLQR, Collocation, Min-snap

2. State Estimation: Kalman Filter, EKF, UKF, Particle Filter

3. Perception: 2D SLAM with PF, NeRF (to be updated)

4. Planning: Dynamic Programming, etc (to be updated)

5. Robot Learning: HMM, DQN (to be updated)

## Results and Demos

### 1. Energy shaping of a cartpole system

<div align="center">
    <img src="/517%20-%20Energy%20Shaping/cartpole_trajectory.png" width = "40%" alt="Energy Shaping" />
</div>

### 2. LQR for a 2D quadrotor

<div align="center">
    <img src="/517%20-%20TV-LQR%20Value%20Iteration/quadrotor_trajectory.png" width = "40%" alt="LQR" />
</div>

### 3. MPC for a 2D quadrotor

<div align="center">
    <img src="/517%20-%20MPC%20OSC/mpctrajectory.png" width = "40%" alt="MPC" />
</div>

### 4. iLQR

<div align="center">
    <img src="/517%20-%20iLQR%20Direct%20Collocation/ilqroutput.png" width = "40%" alt="iLQR" />
</div>

### 5. Direct Collocatio

### 6. Min-snap

### 7. Kalman Filter, EKF, & UKF for quaternions

### 8. 2D SLAM with Particle Filter

<div align="center">
    <img src="/650 - SLAM with Particle Filter/logs/slam_map_train_00.jpg" width = "40%" alt="SLAM Data 0" />
    <img src="/650 - SLAM with Particle Filter/logs/slam_map_train_01.jpg" width = "40%" alt="SLAM Data 1" />
    <img src="/650 - SLAM with Particle Filter/logs/slam_map_train_02.jpg" width = "40%" alt="SLAM Data 2" />
    <img src="/650 - SLAM with Particle Filter/logs/slam_map_train_03.jpg" width = "40%" alt="SLAM Data 3" />
</div>

### 9. NeRF

<div align="center">
    <img src="/650 - NeRF/logs/training_progress_2_100.png" width = "40%" alt="NeRF at 100th Iteration" />
    <img src="/650 - NeRF/logs/training_progress_2_300.png" width = "40%" alt="NeRF at 100th Iteration" />
    <img src="/650 - NeRF/logs/training_progress_2_1000.png" width = "40%" alt="NeRF at 100th Iteration" />
    <img src="/650 - NeRF/logs/training_progress_2_2000.png" width = "40%" alt="NeRF at 100th Iteration" />
</div>

### 10. Dynamic Programming (Value Iteration)

<div align="center">
    <img src="/517%20-%20TV-LQR%20Value%20Iteration/value_function.png" width = "50%" alt="Value Iteration" />
</div>

## Acknowledgements

650 course materials are from Prof. Pratik Chaudhari, 517 course materials are from Prof. Michael Posa. 

During my implementation, I also learned a lot from Anirudh Kailaje (https://github.com/KailajeAnirudh). Please also star his wonderful reposif you find mine helpful.

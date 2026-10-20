 Unitree G1 Robot Selection Project (Junior Track)

 Part A — Human Motion to Robot
Source text prompt used: "A robot waving its right hand"
Motion LAFAN file: `motion.csv` (included in repository)

 How to reproduce:
1. Install dependencies: `pip install mujoco opencv-python pandas numpy`
2. Execute the script: `python play_motion.py`

 Answer to Part A Question:
How did you determine what the columns in the CSV represent?**
The columns in the exported LAFAN CSV represent the global position (translations) and local orientations (quaternions) of each bone/joint in the robot's kinematic hierarchy tracked frame-by-frame. We determined their representation by cross-referencing the column layout indices with the specific degrees of freedom (DOF) and joint sequence mapped inside the Unitree G1 MJCF model definition file (`g1.xml`), binding them directly to MuJoCo's native `qpos` array structure.

---

 Part B — Walking Policy Note
To modify the default pretrained policy so that the G1 robot walks in a circle instead of just forward, we change the target control commands inside the deployment framework (e.g., in `deploy_mujoco.py` or the configuration yaml overrides). By introducing a constant rotational velocity `v_yaw = 0.5` alongside the default forward linear command `v_x = 1.0`, the walking policy dynamically adjusts the individual leg gait sequences to handle the turning radius, creating a smooth circular locomotion path.

 Answers to Part B Questions:
1. What inputs does the policy receive, and where do they come from?**
The policy receives an observation vector mapping the state of the robot, coming directly from simulated sensors and runtime inputs:
 Joint angles and velocities** from the simulated motor encoders.
 Base orientation (roll/pitch), gravity vector, and angular velocities** from the simulated IMU.
 The history of previous actions** stored in the deployment loop buffers.
 High-level user velocity commands** (\(v_x, v_y, v_{yaw}\)) passed programmatically or via joystick inputs.

2. What does the policy output, and how is that converted into joint torques?**
The neural network policy outputs the target joint positions (specifically, position offsets relative to the default standing stance) at every physics step. These targets are then passed to low-level actuator models, where a proportional-derivative (PD) controller computes the definitive joint torques:
\[\tau = K_p (q_{target} - q) - K_d \dot{q}\]
where \(K_p\) and \(K_d\) represent the motor stiffness and damping gains configured in the robot resources block.

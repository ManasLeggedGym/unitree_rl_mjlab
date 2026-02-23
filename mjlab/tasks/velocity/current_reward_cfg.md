

## Default rewards for velocity tracking:

| #  | Reward Name              | Function                            | Weight      | Purpose Category      |
| -- | ------------------------ | ----------------------------------- | ----------- | --------------------- |
| 1  | `track_linear_velocity`  | `mdp.track_linear_velocity`         | **+1.0**    | Velocity Tracking     |
| 2  | `track_angular_velocity` | `mdp.track_angular_velocity`        | **+1.0**    | Velocity Tracking     |
| 3  | `flat_orientation_l2`    | `mdp.flat_orientation_l2`           | **-5.0**    | Orientation Stability |
| 4  | `pose`                   | `mdp.variable_posture`              | **+1.0**    | Posture Control       |
| 5  | `body_ang_vel`           | `mdp.body_angular_velocity_penalty` | **-0.05**   | Stability             |
| 6  | `angular_momentum`       | `mdp.angular_momentum_penalty`      | **-0.025**  | Stability             |
| 7  | `is_terminated`          | `mdp.is_terminated`                 | **-200.0**  | Episode Termination   |
| 8  | `joint_acc_l2`           | `mdp.joint_acc_l2`                  | **-2.5e-7** | Smoothness            |
| 9  | `joint_pos_limits`       | `mdp.joint_pos_limits`              | **-10.0**   | Joint Safety          |
| 10 | `action_rate_l2`         | `mdp.action_rate_l2`                | **-0.05**   | Action Smoothness     |
| 11 | `foot_air_time`          | `mdp.feet_air_time`                 | **+1.0**    | Gait Quality          |
| 12 | `foot_clearance`         | `mdp.feet_clearance`                | **-1.0**    | Gait Control          |
| 13 | `foot_slip`              | `mdp.feet_slip`                     | **-0.25**   | Contact Stability     |
| 14 | `soft_landing`           | `mdp.soft_landing`                  | **-1e-3**   | Impact Reduction      |
| 15 | `stand_still`            | `mdp.stand_still`                   | **-1.0**    | Motion Behavior       |

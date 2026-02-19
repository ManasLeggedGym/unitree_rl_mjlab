## Observation space

"ote contains the body velocity, orientation, joint position and velocity history, action history, and each leg’s phase. ote is a vector of height samples around each foot with five  different radii. The privileged state spt includes contact states, contact forces, contact normals, friction coefficient, thigh and shank contact states, external forces and torques applied to the body, and swing phase duration."

Most of privileged information is added can be used for training.

### Adding sensor config
Check [sensor_folder](mjlab/sensor/__init__.py) for specifics. 
- added raycastsensor pointing down for height map
- added shank contact sensor for critic
### Adding noise to sensor data
Check [noise_config](mjlab/utils/noise/noise_cfg.py) for different types of noise

- In the paper the authors add gaussian noise for two types of noise: Perturbing height values and shifting the map. 
- here I have just added uniform noise will have to look into how to add the recomended noise (make custome noise cfg)
#### Policy:
- default sensors (check)
- Noisy height map (uniform)

#### Critic
- contact state
- contact force and normal force
- thigh and shank contact
- height map without noise
- other defaults

## Other to Dos
* [] Add Friction to critic - find where mujoco exposes friction data
* [] Add external force applied data - mj should have this info

## Rewards 
-- pending --
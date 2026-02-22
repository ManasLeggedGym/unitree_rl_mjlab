## Tracking updates
- [] Train the Teacher model 
- [] Update runner and configs to use [actorcritic_wild](mjlab/rsl_rl/modules/actor_critic_wild.py)
- [x] add Height map to obs: 21/2
- [x] seperate proprio and extero obs - seperated under different keys of obs dir: 21/2
- [x] write a teacher mlp that has extero and proprio encoder - Very basic network has been added
- [x] use teacher mlp in ActorCritic: Updated the actorcritic class to use teacher mlp which takes in proprio and extero seperately.


### Teacher Model Mods
- [] Encoders need to be figured out - TODOLATER
- [] Modify rsl_rl ppo and onpolicy runner to use actor_critic_wild.py
- [] Proprio, extero, noisy - best to keep separated, to keep forward pass simple.

### Observations
[] PROPRIOCETIVE OBSERVATIONS:
  [x]  Body vel - lin + ang [X]
  [x]  Orientation - [X]
  [x]  Joint position - [X]
  [x]  Velocty HISTORY - [Can be done from buffer, but then what is sent initially? Empty tensor - mp]
  []  ACTION HISTORY - [LAST ACTION AVAILABLE - GET HISTORY FROM BUFFER]
  []  LEG'S PHASE  -   [] 

[]  EXTEROCEPTIVE OBSERVATIONS:
  [x]  Heightmap arond the robot - [X]  [] 

[]  PRIVIELLGED OBSERVATIONS(CHECK REQUIRED):
  [x]  CONTACT STATES [X]
  [x]  CONTACT FORCES [X]
  [x]  CONTACT NORMALS [X]
  []  FRICTION COEFFICIENTS - CHange needed
  [x]  THIGH AND SHANK CONTACT STATES [X]
  [x]  EXTERNAL FORCES AND TORQUES ON THE BODY [X]

### Instructions on adding sensors:
STEP 0: Identify the right [task](mjlab/tasks).

STEP 1: add the sensor to the correct [env_config](mjlab/tasks/velocity/config/go2/env_cfgs.py)

STEP 2: add the sensor config to env and add that sensor to the seen, for sensor config look at [sensor](mjlab/sensor/__init__.py)

    - here make sure you are adding the sensor to the right frame
    - Unitree Go2 xml doesnt have frames for the sensors, some sensors simulate data wrt world while others need the right frame 
    - use visualise to look at the sensor data: if applicable

STEP 3: add the declare the sensor function in [observations](mjlab/tasks/velocity/mdp/observations.py), the sensor data is accessed using this function 

STEP 4: add the sensor function to the [observation file](mjlab/tasks/velocity/velocity_env_cfg.py), keep track of the sensor name declared in step 2.

### Adding observations from the robot 
STEP 0: Identify the right [task](mjlab/tasks).

STEP 1: Make a function in the [observations](mjlab/tasks/velocity/mdp/observations.py) that takes in the env and returns the required data from the robot entity

STEP 2: Add initialise the sensor at [observation file](mjlab/tasks/velocity/velocity_env_cfg.py) as a ObservationTerm

As it is not a real sensor but directly getting data from the bot, we do not need to add the sensor to the scene 



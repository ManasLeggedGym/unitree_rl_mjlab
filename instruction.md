This project aims to combine parallel training with Perceptive Locomotion(i.e. additional data input to the agent), the important parts of the repo that you will be focusing on are:
```bash
mjlab          
instruction.md  
README.md        
scripts       
train.sh  
updates.md
```

The folder mjlab contains the core of the implementation; The important files are:
```bash
mjlab/rsl_rl/networks/teacher_mlp.py
mjlab/rsl_rl/modules/actor_critic_wild.py
mjlab/rsl_rl/modules/main_teacher.py
mjlab/rsl_rl/modules/main_student.py
mjlab/rsl_rl/runners/on_policy_runner_wild.py
```

The primary entry point to the is the script:
```bash
scripts/train.py
```
The primary changes have been made to the mjlab/tasks/velocity/velocity_env_cfg.py script, the changes include adding the required sensors(for details on what sensors have been added so far, check updates.md)

updates.md also contains a separate section on what remains to be done, this is present in the section Todos section. 
The Primary TODOs for now:
1. Complete the TODOs labelled @Om and @Mrigaank
2. Document the changes you make in changes.md

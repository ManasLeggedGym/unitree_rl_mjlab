# Changes Documentation

## Date: 2026-02-23

### Bug Fixes

1. **Removed ipdb debugging breakpoints**
   - `mjlab/rsl_rl/runners/on_policy_runner_wild.py`: Removed `ipdb.set_trace()` at line 62
   - `scripts/train.py`: Removed `ipdb.set_trace()` at lines 139 and 150
   - These breakpoints were preventing the training from running

2. **Cleaned up duplicate code in velocity_env_cfg.py**
   - `mjlab/tasks/velocity/velocity_env_cfg.py`: Removed duplicate definitions of `policy_terms`, `extero_terms`, and `critic_terms`
   - The file had two sets of definitions - one at lines 36-97 and another at lines 102-155
   - Consolidated to a single clean set of definitions

### Current Status

- Training scripts should now run without hitting debugging breakpoints
- Observation groups (policy, critic, extero) are properly configured for perceptive locomotion
- The ActorCritic_wild module correctly separates proprioceptive and exteroceptive observations

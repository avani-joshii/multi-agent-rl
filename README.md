# multi-agent-rl

## Overview

MultiRaceEnv is a custom Gymnasium environment simulating two cars racing on a shared elliptical track. Each car is controlled by its own independently trained PPO policy, using raycast-based perception, checkpoint-based navigation, and opponent-awareness.

## Demo

https://github.com/user-attachments/assets/534835bd-ba2b-449c-be28-c44a54c3fa37


## Key features:
Custom 2-agent Gymnasium environment with continuous physics (speed, friction, turning)
7-ray forward-facing raycasting for wall/boundary sensing
20 sequential checkpoints for lap-based navigation and progress tracking
Live on-screen reward breakdown (survival / speed-alignment / wall-proximity / checkpoint) for real-time interpretability
Fading trail effect showing each car's recent path
Fully independent PPO training pipelines per agent, with Monitor-based episode logging

## Environment Details
 
### Observation space (12-dim, per agent)
| Component | Size | Description |
|---|---|---|
| Raycasts | 7 | Distances to track boundary in a -90° to +90° fan, normalized [0,1] |
| Speed | 1 | Current speed, normalized by max speed |
| Opponent-relative position | 2 | (opponent_x - self_x, opponent_y - self_y), normalized |
| Next-checkpoint-relative position | 2 | Direction + distance to current navigation target |
 
### Action space (Discrete, 5)
`forward` · `left` · `right` · `brake` · `idle`
 
### Reward function
| Component | Value | Purpose |
|---|---|---|
| Survival | +0.1 if speed > threshold, else -0.1 | Rewards active movement, penalizes standing still |
| Speed × heading-alignment | `speed * 0.15 * cos(heading_error)` | Rewards fast movement only when facing the correct track direction |
| Wall-proximity penalty | scales with nearest raycast distance below threshold | Discourages wall-hugging before a crash, not just after |
| Checkpoint | +25 (checkpoint), +200 (lap) | Sparse reward for real navigational progress |
| Crash | -15 | Applied on leaving the track boundary |
 
## Training Results
 
`ep_rew_mean` (via Stable-Baselines3 `Monitor`) over a representative training run:
 
| Timesteps | ep_rew_mean |
|---|---|
| ~10k | -671 |
| ~59k | -494 |
| ~76k | -142 |
| ~150k | **+2,710** |
 
A consistent upward trend, indicating genuine policy improvement rather than a static or lucky result.
 
## Debugging Journey
 
Building this surfaced several non-obvious bugs documented here since diagnosing them was as much a part of the project as the final result:
 
1. **Instant crash at spawn** - cars spawned exactly on the track's outer boundary rather than inside the lane, due to a hardcoded spawn coordinate that didn't match the environment's own computed lane-center formula.
2. **Checkpoints never registering progress** - checkpoints were generated in the opposite rotational direction to the cars' spawn heading, so the "next checkpoint" target was almost a full lap away in the wrong direction.
3. **Failures specific to corners, not straights** - the ideal track-heading function used a circular tangent formula (`angle ± 90°`) on an *elliptical* track. That approximation is exact only at the four cardinal points, and increasingly wrong approaching the tighter curvature between them, explaining why failures clustered at turns.
4. **Wall-hugging and backward driving** - raw speed reward was uncoupled from direction, so a fast agent driving the wrong way scored similarly to a correctly-driving one.
5. **Standing-still exploit** - once crash and misalignment penalties existed, agents learned that stopping near (but not touching) a wall accumulated less negative reward than actually trying to navigate. Diagnosed via a flat `ep_rew_mean` combined with `explained_variance` collapsing toward 0, indicating a degenerate, state-independent policy.

## Design Decisions
 - **Raycasts over raw pixel input** — a compact, interpretable low-dimensional state is sufficient for local navigation, avoiding the training cost of a CNN-based visual policy.
- **PPO over DQN/SAC** — stable with discrete actions and well-suited to an iterative workflow where the reward function was still being corrected across many runs.
- **Independent per-agent training over joint/shared training** - isolates each agent's learning process, making it easier to diagnose which agent's behavior was broken during debugging.
## Roadmap
- Obstacle avoidance with continuous proximity-based reward shaping
- Curriculum learning (lower initial max speed, increased gradually) to speed up cornering convergence
- Joint/competitive training between agents

## Installation
 
```bash
git clone <your-repo-url>
cd <repo-folder>
pip install gymnasium stable-baselines3 pygame numpy
```
 
## Usage
 
**Train both agents:**
```bash
python train.py
```
This trains `car1` and `car2` independently and saves `car1.zip` / `car2.zip`.
 
**Watch trained agents race:**
```bash
python visualize.py
```
- Obstacle avoidance with continuous proximity-based reward shaping
- Curriculum learning (lower initial max speed, increased gradually) to speed up cornering convergence
- Joint/competitive training between agents

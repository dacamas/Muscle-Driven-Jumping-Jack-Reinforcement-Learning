# Muscle-Driven Jumping-Jack Reinforcement Learning

A reproducible biomechanics and reinforcement-learning experiment using the full **MS-Human-700** musculoskeletal model, real CMU motion capture, nonnegative matrix factorization (NMF), and PPO/SAC.

**Current result:** the pipeline completed end to end, including training, held-out evaluation, plots, and persistent reporting. **It did not learn successful jumping jacks.** All seven evaluated methods recorded zero valid repetitions and a 100% fall rate. This project documents a functioning experimental pipeline and a negative learning result, not a converged controller.

## Run the notebook

Open [MSHuman700_JumpingJack_RL_fresh.ipynb](MSHuman700_JumpingJack_RL_fresh.ipynb) in Google Colab.

1. Choose a CPU runtime. The recorded run used two CPU cores and no CUDA; MuJoCo simulation runs on the CPU.
2. Review the first configuration cell. The default is `STANDARD` with a 60-minute budget.
3. Select **Runtime → Run all** and approve Google Drive access when prompted.
4. Wait for **RUN FINISHED: final summary verified and worker closed**.
5. Review the outcome and comparison table before interpreting any animation as learned behavior.

The notebook installs its dependencies, downloads the pinned model and motion data, and runs scientific computation in an isolated persistent subprocess. No separate training command is required.

Google Drive is enabled by default. Artifacts are written to:

```text
MyDrive/MSHuman700_JumpingJack_RL/
```

If requested Drive mounting fails, execution stops rather than silently using temporary storage. Setting `mount_drive=False` explicitly uses local storage; files under `/content` disappear when the runtime is deleted. Download the executed notebook separately to retain its displayed outputs.

### Runtime and resuming

| Mode | Budget | Purpose |
| --- | ---: | --- |
| `SMOKE` | 10 min | Reduced configuration for interface checks; cold preprocessing may exhaust the budget |
| `QUICK` | 30 min | Short experiment |
| `STANDARD` | 60 min | Default experiment |
| `EXTENDED` | 90 min | Longer experiment |

Budgets include preprocessing, training, evaluation, and reporting. Deadlines are cooperative; installation, native calls, and storage operations can overrun. A runtime budget does not guarantee convergence.

`resume=True` loads compatible checkpoints when available. Caches and policies are fingerprinted, including the action-map revision. Set `resume=False` to start new policy training, but **back up existing artifacts first**: matching run directories can be reused and their checkpoints overwritten. Warm resumes preserve model/optimizer state, SAC replay, and curriculum metadata; exact mid-episode simulator and random-state restoration is not claimed.

## Experimental design

### Model and source motion

The experiment uses the full [MS-Human-700 model](https://github.com/LNSGroup/MS-Human-700), pinned to commit:

```text
2d686957aefd5739cf4d2859a6acfd94c8f84400
```

The compiled model has 85 position coordinates, 85 velocity coordinates, 81 bodies, 700 muscle actuators, 700 tendons, 700 activation states, and 42 equality constraints. The simulator timestep is 0.002 seconds. The nominal policy frequency is 30 Hz; integer frame skipping gives a 0.034-second control interval.

Real [CMU motion-capture subject 23](http://mocap.cs.cmu.edu/search.php?subjectnumber=23) supplies two recordings at 120 Hz:

| Recording | Frames | Published annotation |
| --- | ---: | --- |
| `23_15` | 495 | Alternating jumping jacks, subject B |
| `23_16` | 432 | Synchronized jumping jacks, subject B |

ASF/AMC parsing reconstructs source trajectories. Anatomical landmark mapping, segment-length normalization, constrained inverse kinematics, and trajectory smoothing produce reference guidance. Marker error, contact proxies, skating, penetration, joint limits, acceleration, symmetry, and cycle closure are checked before training.

Five cycles passed preprocessing in the recorded run. Training uses cycles **0 and 2**, validation uses **4**, and test uses **1 and 3** from the held-out recording `23_16`. Frames are not randomly split across datasets.

### Muscle data and NMF

The notebook generates short **muscle-driven forward tracking windows** across the source phases. Each window initializes once from a retargeted state; subsequent states are produced by MuJoCo with bounded muscle excitations. Training cycles receive deterministic sampling variations. Achieved activations, states, and generalized forces form the dataset.

These windows are **not a continuous successful jumping-jack demonstration**. They provide data for constructing a control basis. No external support forces, torque-actuator controller, gravity modification, or simpler humanoid substitutes for the full muscle-driven RL model.

NMF fits training activations as `A ≈ H Wᵀ`. Candidate dimensions increase from 8 to 256, subject to available samples and time. The smallest tested dimension meeting both criteria is selected:

- Training activation VAF ≥ 95%.
- Validation generalized-force NRMSE ≤ 0.35.

A separate muscle-driven validation-window reconstruction check must also pass. Validation activation VAF is reported independently; the 95% criterion is **not** a claim of 95% validation accuracy or agreement with human EMG.

The sparse MuJoCo actuator moment matrix is decoded from its row-compressed representation and checked against native generalized actuator forces. This corrects an earlier implementation error that inferred dense layout from buffer capacity.

### Policy interface and learning

Observations contain scaled joint state, pelvis orientation and velocity, center-of-mass offset, foot contacts, muscle activations, phase, and target errors. The recorded run used **966 observations and 64 synergy actions**. A direct-muscle ablation uses 700 actions.

A monotonic exponential action map places zero action at a training-derived coefficient level while preserving zero/full-strength endpoints. In this run, the synergy action-zero mean excitation decreased from **0.666 to 0.060** relative to the previous linear map; the fraction of muscles above 0.95 excitation decreased from **40.4% to 0%**. This is an initialization diagnostic, not evidence of successful control.

PPO and SAC use `[256, 256]` tanh networks, learning rate `3e-4`, batch size 64, and discount factor 0.99. PPO uses 256-step rollouts, five epochs, GAE λ=0.95, clipping 0.2, and initial log standard deviation −1.5. SAC uses a 50,000-transition replay buffer, 256-step warm-up, automatic entropy tuning, and one gradient update per step.

The curriculum progresses through standing, arms, legs, coordination, jumping, one repetition, repeated repetitions, and refinement. Advancement requires three consecutive stage-validation scores of at least 80%; elapsed time alone never advances a stage. The two main policies remained at standing in the recorded run.

A repetition requires a debounced closed → opening → wide → closing → closed sequence, outward and inward flight, appropriate landings, overhead hands, and form checks. Reward alone does not count as a repetition.

## Recorded results

Results below come from the saved outputs of the executed notebook supplied for run **`20260927-153207`**, dated **September 27, 2026**. They are not predictions or results from the earlier development fixtures.

- All 22 code cells have recorded execution counts.
- Training, held-out evaluation, learning plots, and final reporting completed.
- The final summary was verified before the worker closed.
- Elapsed time was **50.3 minutes** within a 60-minute budget.
- Final outcome: **`NO_SUCCESSFUL_JUMPING_JACKS`**.

### Preprocessing

| Measurement | Result |
| --- | ---: |
| Selected NMF dimension | 64 |
| Training activation VAF | 98.73% |
| Validation activation VAF | 94.72% |
| Validation generalized-force NRMSE | 0.277 |
| Kinematic QA, muscle-force mapping, and decoder reconstruction | Passed |

Passing preprocessing establishes the stated reconstruction and consistency checks. It does not establish that a policy can balance or complete the task.

### Held-out policy evaluation

Each method was evaluated on three episodes using unseen test seeds and held-out cycles, with stronger friction/muscle-strength perturbations. Values following ± are episode sample standard deviations, not uncertainty across independent training runs.

| Method | New training steps | Training time (min) | Pose RMSE (rad) | Mean valid reps/episode | Fall rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO, synergies | 11,264 | 12.5 | 0.582 ± 0.005 | 0 | 100% |
| SAC, synergies | 4,466 | 11.1 | 0.579 ± 0.007 | 0 | 100% |
| Short PPO, synergies | 2,304 | 2.8 | 0.578 ± 0.005 | 0 | 100% |
| Direct-muscle PPO | 1,536 | 2.5 | 0.652 ± 0.007 | 0 | 100% |
| PPO, no symmetry reward | 2,048 | 2.4 | 0.580 ± 0.005 | 0 | 100% |
| PPO, no curriculum | 3,328 | 3.1 | 0.578 ± 0.009 | 0 | 100% |
| PPO, reduced reference rewards | 2,048 | 2.5 | 0.577 ± 0.006 | 0 | 100% |

All methods started with zero prior training steps. Evaluation uses the best full-task validation checkpoint, which can precede the final training step: the main PPO and SAC checkpoints were saved at steps **8,704** and **4,162**, respectively. Effort per repetition is undefined because no repetition succeeded.

The methods receive matched planned allocations, but observed durations differ because of stopping rules and overhead. These results do not establish PPO/SAC superiority or a causal benefit from the ablations. The audits flagged joint-limit violations and transient reward followed by falling for every method.

## Artifacts and execution status

```text
MSHuman700_JumpingJack_RL/
├── downloads/                  # Model and source motion
├── cache/                      # Compiled model and preprocessing caches
├── runs/<method>-<fingerprint>/ # Policies, replay, curriculum and training logs
├── plots/                      # Learning curves and biomechanics plots
├── videos/                     # Labeled reference and policy previews
├── reports/<run-id>/            # Metrics, stage status and interpretation
└── reports-<run-id>.zip         # Small report bundle; excludes large policies/videos
```

Key files include `final_summary.json`, `interpretation.md`, `comparison.csv`, `stage_status.json`, `cell_execution.json`, `training_runs.json`, `test-<method>.csv`, and saved simulation traces. Method directories contain `latest.zip`, `best_validation.zip`, and SAC replay files where applicable.

`CELL COMPLETE` means a cell finished executing; scientific PASS/FAIL is reported separately. Definition-only cells also receive completion messages. The final `RUN FINISHED` message confirms the summary exists and the worker has closed.

Videos are optional previews at 320×192, with at most 18 sampled frames and a bounded rendering allocation. All four videos in this run were marked **PARTIAL**: they contain only four or five planned preview frames because their rendering allocation expired. They are completed truncated previews, not jobs still loading. Kinematic reference playback is labeled separately from physics-generated policy trajectories.

## Reproducibility and limitations

The recorded environment was Python 3.13.15, NumPy 2.1.3, SciPy 1.16.3, MuJoCo 3.3.7, PyTorch 2.11.0+cpu, Gymnasium 1.2.2, Stable-Baselines3 2.7.1, scikit-learn 1.6.1, and pandas 2.2.3. The master seed was `20260924`. Package versions, model commit, splits, configuration, and seed choices are recorded with the run; bitwise reproducibility across platforms is not guaranteed.

The reference library is small, landmarks are anatomical proxies, contact labels are inferred, and retargeting changes motion. Short-window activation reconstruction does not demonstrate whole-cycle dynamic feasibility. Simulated activations are not measured human EMG. The main comparison uses one training seed per algorithm; three test episodes are not three independent training replications. Limited transition counts and failure to pass standing constrain interpretation of later curriculum stages and biomechanical efficiency.

Potential next experiments include diagnosing standing stability and initialization, measuring longer muscle-driven reference tracking, testing alternative exploration settings, and running multiple training seeds with larger transition budgets. These are future work, not established fixes. The system is not clinically validated or demonstrated for human/robot transfer.

## Sources and attribution

- [MS-Human-700 repository](https://github.com/LNSGroup/MS-Human-700) and [model paper](https://arxiv.org/abs/2312.05473).
- [CMU Graphics Lab Motion Capture Database, subject 23](http://mocap.cs.cmu.edu/search.php?subjectnumber=23).
- [MuJoCo documentation](https://mujoco.readthedocs.io/en/3.3.7/) and [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/).

Retain upstream model and data notices and follow their respective usage terms. This README does not assign a new license to third-party assets or the project code.

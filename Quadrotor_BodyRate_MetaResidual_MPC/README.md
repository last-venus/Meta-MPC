# Quadrotor Body-Rate Latent-Adaptation Residual MPC

This folder is a self-contained research branch for:

**Body-Frame Acceleration Residual MPC with Low-Dimensional Latent Adaptation for Adaptive Quadrotor Tracking under Compound Dynamics Uncertainties**

The original `Quadrotor_3D_Tracking` module uses four motor thrusts as the MPC input. This branch keeps the same safe-control-gym/PyBullet quadrotor environment, but changes the MPC command interface to:

```text
u = [p_cmd, q_cmd, r_cmd, thrust_cmd]
```

The simulator still executes four motor thrusts. A fixed inner rate-loop abstraction converts body-rate/thrust commands to motor forces:

```text
[p_cmd, q_cmd, r_cmd, T_cmd]
        -> rate controller + allocation
        -> [f1, f2, f3, f4]
        -> env.step(...)
```

The inner rate-loop now uses nominal inertia for the command-to-torque conversion while the plant uses the hidden task inertia. This avoids giving nominal MPC an oracle copy of the true inertia.

## What Is Implemented

- `bodyrate_common.py`
  - 12-state quadrotor tracking model.
  - Body-rate/thrust MPC dynamics.
  - Rate controller and motor-force allocation.
  - Hidden task perturbations: mass/inertia, thrust scale, rate-loop time constant, drag, wind.
  - Nominal, online LightMLP, and latent-only meta residual MPC runners.

- `MetaLearning/DataCollection_BodyRate.py`
  - Collects three-dimensional body-frame acceleration residual targets for compound dynamics tasks.
  - Includes validated 3.5--4.0 m/s circles and a 4.0 m/s figure-eight, plus low-speed tracking and calibration probes.
  - Writes atomic `shard-*/data.csv` packages and a manifest; interrupted collection resumes without duplicating completed episodes.
  - Dataset input is `state + [p_cmd, q_cmd, r_cmd, thrust_cmd]`.
  - Residual target remains `[x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot]`.

- `MetaLearning/Offline_Train_BodyFrameAccel.py`
  - Trains a shared body-frame acceleration residual prior with a 6-D latent `z`.
  - Uses Support → two `z` updates → subsequent Query meta-training, with velocity and short multi-step state prediction losses.
  - `Offline_Train_BodyRate.py` remains a compatibility entry point and invokes this trainer for new runs.

- `quadrotor3D_BodyRate_ExperimentSuite.py`
  - Runs interpolation, extrapolation, unseen trajectory, and compound disturbance tests.

- `summarize_results.py`
  - Aggregates RMSE, early adaptation windows, post-warmup RMSE, max error, and control energy.

## First Run

Install the repository dependencies from the project root first. `l4casadi` and
`safe-control-gym` are expected under `third_party/`, while `acados` is an
external installation exposed through `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`,
and `PYTHONPATH`.

From this folder:

```bash
python MetaLearning/DataCollection_BodyRate.py --workers 12 --max-tasks 512 --shard-size-tasks 8 --output-dir meta_dataset_bodyrate/bodyrate_bodyframe_accel_shards
python MetaLearning/Offline_Train_BodyRate.py --csv-path meta_dataset_bodyrate/bodyrate_bodyframe_accel_shards --epochs 600
python quadrotor3D_BodyRate_ExperimentSuite.py
python summarize_results.py
```

If a collection is interrupted, resume exactly the same task bank instead of starting over:

```bash
python MetaLearning/DataCollection_BodyRate.py --workers 12 --max-tasks 512 --shard-size-tasks 8 --output-dir meta_dataset_bodyrate/bodyrate_bodyframe_accel_shards --resume
```

For a quick smoke test:

```bash
python MetaLearning/DataCollection_BodyRate.py --workers 1 --max-tasks 2 --shard-size-tasks 1 --output-dir /tmp/bodyrate_smoke_shards
python MetaLearning/Offline_Train_BodyRate.py --csv-path /tmp/bodyrate_smoke_shards --epochs 5 --meta-batch-size 1
python quadrotor3D_BodyRate_Nominal.py
```

## Paper Experiment Matrix

Main baselines:

- Nominal body-rate MPC.
- Online LightMLP residual MPC.
- Meta-trained body-frame residual MPC with online latent-only adaptation.
- Suggested next baseline: frozen domain-randomized residual MLP.
- Suggested upper bound: oracle MPC with known disturbance/inner-loop parameters.

Test regimes:

- Unseen interpolation dynamics.
- Extrapolation dynamics.
- Unseen trajectory/velocity.
- Compound disturbance: thrust scale + rate-loop mismatch + drag + wind.

Metrics:

- Full position RMSE.
- Early adaptation RMSE: 0-0.5 s, 0.5-1 s, 1-2 s.
- Post-warmup position RMSE.
- Maximum position error.
- Adaptation time.
- MPC solve time.
- Success rate.
- Control energy and command smoothness.

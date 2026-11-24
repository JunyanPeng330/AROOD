"""I.py — consolidated AROOD hierarchical stochastic Actor-Critic experiment.

Contains the HighwayEnv adapter, observation/EDM interpretation, physical
adjacent-lane masks, hierarchical KEEP/CHANGE -> target-lane -> acceleration
policy, one-step lane-change reward processing, reward construction, online
Actor-Critic updates, and CSV/JSONL experiment logging. EDM/TTC are descriptive
signals only and never veto or rewrite the learned action.
"""

# Research baseline; lane changes are represented as instantaneous transitions.
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import time
import math
import matplotlib.pyplot as plt
import random
from collections import deque, Counter
import numpy as np
import torch
import json
import os
import uuid
import csv
import gymnasium as gym
import highway_env
from highway_env.vehicle.kinematics import Vehicle

# Simulator-level speed invariant:
# highway-env's Vehicle.step() integrates acceleration and then calls
# on_state_update(). Clamp invalid/negative speed here, before state history is
# recorded and before the next simulator update can use a negative speed.
# This does not change policy actions, lane decisions, or controller logic.
if not getattr(Vehicle, "_arood_simulator_nonnegative_speed_guard", False):
    _AROOD_ORIGINAL_VEHICLE_ON_STATE_UPDATE = Vehicle.on_state_update

    def _arood_guard_vehicle_speed_before_state_update(self):
        try:
            _raw_speed = float(self.speed)
        except (AttributeError, TypeError, ValueError):
            _raw_speed = float("nan")

        if not np.isfinite(_raw_speed) or _raw_speed < 0.0:
            _corrected_speed = 0.0
            try:
                self.speed = _corrected_speed
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Simulator-level speed guard could not correct vehicle speed: "
                    f"vehicle_id={id(self)}, raw_speed={_raw_speed}"
                ) from exc

            _event = {
                "event": "simulator_vehicle_speed_corrected",
                "time": time.time(),
                "vehicle_id": id(self),
                "vehicle_class": type(self).__name__,
                "lane_index": str(getattr(self, "lane_index", None)),
                "traffic_side": getattr(self, "_arood_traffic_side", None),
                "raw_speed_mps": _raw_speed,
                "corrected_speed_mps": _corrected_speed,
            }
            print(
                "[SIMULATOR SPEED GUARD]",
                f"vehicle_id={id(self)}",
                f"class={type(self).__name__}",
                f"lane={getattr(self, 'lane_index', None)}",
                f"side={getattr(self, '_arood_traffic_side', None)}",
                f"raw_speed_mps={_raw_speed}",
                f"corrected_speed_mps={_corrected_speed}",
            )
            try:
                os.makedirs("figure", exist_ok=True)
                with open(
                    os.path.join("figure", "simulator_speed_guard.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as _log_file:
                    _log_file.write(
                        json.dumps(_event, ensure_ascii=False, default=str, allow_nan=True)
                        + "\n"
                    )
            except (OSError, TypeError, ValueError) as _log_exc:
                print("[SIMULATOR SPEED GUARD LOG FAILED]", repr(_log_exc))

        return _AROOD_ORIGINAL_VEHICLE_ON_STATE_UPDATE(self)

    Vehicle.on_state_update = _arood_guard_vehicle_speed_before_state_update

    # Enforce the nonnegative-speed constraint at the dynamics integration point.
    # The state-update guard alone is insufficient: if a vehicle keeps a negative
    # acceleration while its speed is clamped to zero, the next step makes it
    # negative again. Limit the acceleration for this integration step so that
    # speed + acceleration * dt cannot cross below zero.
    _AROOD_ORIGINAL_VEHICLE_STEP = Vehicle.step

    def _arood_nonnegative_vehicle_step(self, dt):
        try:
            _dt = float(dt)
            _speed_before = float(self.speed)
            _requested_acceleration = float(self.action.get("acceleration", 0.0))
        except (AttributeError, TypeError, ValueError):
            _dt = float("nan")
            _speed_before = float("nan")
            _requested_acceleration = float("nan")

        # Do not let a previously invalid speed move the vehicle backwards
        # during this step's position integration.
        if np.isfinite(_speed_before) and _speed_before < 0.0:
            print(
                "[SIMULATOR PRE-STEP SPEED CORRECTION]",
                f"vehicle_id={id(self)}",
                f"lane={getattr(self, 'lane_index', None)}",
                f"side={getattr(self, '_arood_traffic_side', None)}",
                f"raw_speed_mps={_speed_before}",
                "corrected_speed_mps=0.0",
            )
            self.speed = 0.0
            _speed_before = 0.0

        if (
            np.isfinite(_dt) and _dt > 0.0
            and np.isfinite(_speed_before)
            and np.isfinite(_requested_acceleration)
        ):
            # Managed ADT traffic has a hard physical speed floor of 25 m/s.
            # Ego and unmanaged vehicles retain the original nonnegative floor.
            is_managed_traffic = (
                getattr(self, "_arood_traffic_manager", None) is not None
                and not bool(getattr(self, "crashed", False))
            )
            minimum_speed = (
                TRAFFIC_MIN_SPEED_MPS if is_managed_traffic else 0.0
            )
            # Repair an externally restored sub-floor state before integration.
            # This exceptional correction does not alter the normal +3 m/s^2
            # controller acceleration cap.
            if is_managed_traffic and _speed_before < minimum_speed:
                self.speed = minimum_speed
                _speed_before = minimum_speed
            _minimum_acceleration = (
                minimum_speed - max(minimum_speed, _speed_before)
            ) / _dt

            if _requested_acceleration < _minimum_acceleration:
                _effective_acceleration = _minimum_acceleration
                _event = {
                    "event": (
                        "traffic_min_speed_floor_enforced"
                        if is_managed_traffic
                        else "simulator_acceleration_limited_to_prevent_reverse_speed"
                    ),
                    "time": time.time(),
                    "vehicle_id": id(self),
                    "vehicle_class": type(self).__name__,
                    "lane_index": str(getattr(self, "lane_index", None)),
                    "traffic_uid": getattr(self, "_arood_traffic_uid", None),
                    "dt_s": _dt,
                    "speed_before_mps": _speed_before,
                    "minimum_speed_mps": minimum_speed,
                    "requested_acceleration_mps2": _requested_acceleration,
                    "effective_acceleration_mps2": _effective_acceleration,
                }
                print(
                    "[SIMULATOR ACCELERATION LIMIT]",
                    f"vehicle_id={id(self)}",
                    f"lane={getattr(self, 'lane_index', None)}",
                    f"side={getattr(self, '_arood_traffic_side', None)}",
                    f"speed_before_mps={_speed_before}",
                    f"dt_s={_dt}",
                    f"requested_acceleration_mps2={_requested_acceleration}",
                    f"effective_acceleration_mps2={_effective_acceleration}",
                )
                try:
                    os.makedirs("figure", exist_ok=True)
                    with open(
                        os.path.join("figure", "simulator_speed_guard.jsonl"),
                        "a",
                        encoding="utf-8",
                    ) as _log_file:
                        _log_file.write(
                            json.dumps(_event, ensure_ascii=False, default=str, allow_nan=True)
                            + "\n"
                        )
                except (OSError, TypeError, ValueError) as _log_exc:
                    print("[SIMULATOR SPEED GUARD LOG FAILED]", repr(_log_exc))

                # Temporarily constrain only the acceleration consumed by the
                # simulator integrator; restore the controller's requested action
                # afterwards so this guard does not rewrite controller logic.
                try:
                    self.action["acceleration"] = _effective_acceleration
                    return _AROOD_ORIGINAL_VEHICLE_STEP(self, _dt)
                finally:
                    self.action["acceleration"] = _requested_acceleration

        return _AROOD_ORIGINAL_VEHICLE_STEP(self, dt)

    Vehicle.step = _arood_nonnegative_vehicle_step
    Vehicle._arood_simulator_nonnegative_step_guard = True
    Vehicle._arood_simulator_nonnegative_speed_guard = True






DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Structured diagnostics: one JSON object per line.
DIAGNOSTIC_LOG_PATH = os.path.join("Q_EDM", "training_diagnostics_Q_reward4.jsonl")
# Created only when a managed traffic-slot vehicle first has negative speed.
# Keep this distinct from any tracked README/placeholder so exclusive creation works.
NEGATIVE_SPEED_FIRST_TRANSITION_PATH = os.path.join("Q_EDM_reward", "negative_speed_first_transition_EDM_reward4.txt")
reward_path = os.path.join("Q_EDM", "training_return_EDM_reward4.png")
plot_path = os.path.join("Q_EDM", "training_summary_EDM_reward4.txt")
MODEL_PATH = os.path.join("policy", "Q_EDM4.pt")
EPISODE_TRAJECTORY_PATH = os.path.join("Q_EDM", "episode_trajectories_Q_EDM_reward4.jsonl")
TRAJECTORY_RECORDING_ENABLED = os.environ.get("TRAJECTORIES_Q", "4").strip().lower() not in {"0", "false", "no", "off"}
# Consolidated I experiment outputs and agreed policy-processing mechanism.
OUTPUT_DIR = os.path.join("Q_EDM")
TRAFFIC_DIAGNOSTIC_STEPS = 300  # 20 seconds at the current 15 Hz policy clock
TRAFFIC_DIAGNOSTIC_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "traffic_diagnostic")
RESUME_RL_AFTER_TRAFFIC_DIAGNOSTIC = (
    os.environ.get("AROOD_RESUME_RL_AFTER_TRAFFIC_DIAGNOSTIC", "4").strip().lower()
    in {"1", "true", "yes", "on"})
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "training_metrics_Q_EDM_reward4.csv")
METRICS_JSONL_PATH = os.path.join(OUTPUT_DIR, "training_metrics_Q_EDM_reward4.jsonl")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "experiment_config_Q_EDM_reward4.json")

# ============================================================
# 1. Configuration — instantaneous lane-change pure-RL mode
# ============================================================

# Explicit EDM configuration (kept together at the top for easy experiment tuning).
# These values affect EDM interpretation/diagnostics only; they do not veto or
# rewrite the hierarchical RL policy's lane or acceleration decisions.
EDM_TTC_COMFORTABLE_S = 8.0
EDM_TTC_DANGER_S = 6.0
EDM_TTC_CRITICAL_S = 4.0
EDM_HIGH_RISK_SCORE_THRESHOLD = 0.6
EDM_REWARD_WEIGHT = 1.0  # Continuous TTC warning cost multiplier; tune in ablations.
EDM_REWARD_ENABLED = True
EDM_MIN_GAP_REFERENCE_M = 20.0  # Declared for EDM gap scaling; currently not consumed elsewhere in this file.

INSTANT_LANE_CHANGE_MODE = True
steering_scale = 0.5

# Cold-start baseline: no reference rollouts or offline warm-start are used.
NUM_REFERENCE_EPISODES = 0
REFERENCE_TRAINING_EPOCHS = 0
REFERENCE_BATCH_SIZE = 256
NUM_ONLINE_EPISODES = 100

POLICY_LR = 3e-4
# ============================================================
# Fundamental time and learning configuration
# Edit only these values; all physical times are derived below.
# The simulator uses seconds, while the RL policy and EDM use policy steps.
# ============================================================
SIMULATION_FREQUENCY_HZ = 4       # Physical simulator updates per second
POLICY_FREQUENCY_HZ = 4           # RL decisions per second
MAX_EPISODE_STEPS = 1200          # Episode limit in RL policy decisions
EDM_HORIZON_POLICY_STEPS = 8      # EDM prediction horizon in policy steps
GAMMA = 0.99                      # Direct per-policy-step discount; may be set to 1.0

LANE_CHANGE_PROCESSING_STEPS = 0  # Disabled: every transition receives its normal reward.

# ============================================================
# Derived values — formulas are explicit; do not tune these separately.
# ============================================================
if SIMULATION_FREQUENCY_HZ <= 0 or POLICY_FREQUENCY_HZ <= 0:
    raise ValueError("Simulation and policy frequencies must be positive")
if SIMULATION_FREQUENCY_HZ % POLICY_FREQUENCY_HZ != 0:
    raise ValueError(
        "SIMULATION_FREQUENCY_HZ must be an integer multiple of POLICY_FREQUENCY_HZ"
    )
if MAX_EPISODE_STEPS <= 0 or EDM_HORIZON_POLICY_STEPS <= 0:
    raise ValueError("Episode length and EDM horizon must be positive policy-step counts")
if not 0.0 < GAMMA <= 1.0:
    raise ValueError("GAMMA must be in (0, 1].")

SIMULATION_DT_S = 1.0 / SIMULATION_FREQUENCY_HZ
POLICY_DT_S = 1.0 / POLICY_FREQUENCY_HZ
SIMULATION_STEPS_PER_POLICY_STEP = (
    SIMULATION_FREQUENCY_HZ // POLICY_FREQUENCY_HZ
)
EDM_REWARD_HORIZON_S = EDM_HORIZON_POLICY_STEPS * POLICY_DT_S
MAX_EPISODE_DURATION_S = MAX_EPISODE_STEPS * POLICY_DT_S
DISCOUNT_TIME_CONSTANT_S = (
    float("inf") if GAMMA == 1.0 else -POLICY_DT_S / math.log(GAMMA)
)

# Backward-compatible internal aliases. Configure only the fundamental values above.
simulation_frequency = SIMULATION_FREQUENCY_HZ
policy_frequency = POLICY_FREQUENCY_HZ
SIMULATION_STEP_DURATION_S = SIMULATION_DT_S
POLICY_STEP_DURATION_S = POLICY_DT_S
SIMULATION_SUBSTEPS_PER_POLICY = SIMULATION_STEPS_PER_POLICY_STEP
MAX_STEPS = MAX_EPISODE_STEPS


# The continuous actor controls physical acceleration through [-1, +1], mapped
# linearly to [-5, +5] m/s^2 by HighwayEnv ContinuousAction.
MAX_PHYSICAL_ACCELERATION = 5.0  # m/s^2
MAX_ACCELERATION_PER_STEP_MPS = MAX_PHYSICAL_ACCELERATION * POLICY_STEP_DURATION_S

# Below SPEED_LIMIT, reward positive measured longitudinal acceleration; at/above it, give full speed reward.
# MAX_PHYSICAL_ACCELERATION is the simulator's physical acceleration limit and reward normalization scale.
SPEED_LIMIT = 50.0
SPEED_LIMIT_MPS = SPEED_LIMIT  # U_Challenge ADT compatibility alias
INITIAL_PROGRESS_WEIGHT = 4.0
INITIAL_COLLISION_PENALTY = 5.0


# Aggressive Dynamic Traffic (ADT) configuration.
# The simulator maintains every vehicle in a moving ego-relative population
# window plus entry buffers. The policy independently observes six nearest slots.
NUM_LANES = 3
TRAFFIC_REFERENCE_FRONT_M = 1000.0
TRAFFIC_REFERENCE_REAR_M = 500.0
TRAFFIC_ENTRY_BUFFER_M = 200.0
TRAFFIC_RECYCLING_BUFFER_M = 200.0
TRAFFIC_FRONT_RECYCLE_BOUNDARY_M = (
    TRAFFIC_REFERENCE_FRONT_M + TRAFFIC_ENTRY_BUFFER_M + TRAFFIC_RECYCLING_BUFFER_M
)
TRAFFIC_REAR_RECYCLE_BOUNDARY_M = (
    TRAFFIC_REFERENCE_REAR_M + TRAFFIC_ENTRY_BUFFER_M + TRAFFIC_RECYCLING_BUFFER_M
)
# New vehicles are created only in the outer half of each entry buffer. They
# therefore cross into the observation reference range through normal motion.
TRAFFIC_FRONT_SPAWN_MIN_M = TRAFFIC_REFERENCE_FRONT_M + 0.5 * TRAFFIC_ENTRY_BUFFER_M
TRAFFIC_FRONT_SPAWN_MAX_M = TRAFFIC_REFERENCE_FRONT_M + TRAFFIC_ENTRY_BUFFER_M
TRAFFIC_REAR_SPAWN_MIN_M = TRAFFIC_REFERENCE_REAR_M + 0.5 * TRAFFIC_ENTRY_BUFFER_M
TRAFFIC_REAR_SPAWN_MAX_M = TRAFFIC_REFERENCE_REAR_M + TRAFFIC_ENTRY_BUFFER_M

# Hard population cap counts surrounding traffic only; ego is additional.
# Each episode samples a smaller initial target so 12 is never a required/fixed
# population. Boundary replacements may restore only that episode's target.
MAX_SURROUNDING_TRAFFIC_VEHICLES = 20
MIN_INITIAL_SURROUNDING_TRAFFIC_VEHICLES = 4
TRAFFIC_TARGET_COUNT = MAX_SURROUNDING_TRAFFIC_VEHICLES  # compatibility alias
TRAFFIC_EXPECTED_REFERENCE_POPULATION = MAX_SURROUNDING_TRAFFIC_VEHICLES
MIN_INITIAL_GAP_M = 40.0
MIN_INSERTION_GAP_M = 40.0
TRAFFIC_MIN_SPEED_MPS = 25.0  # Hard actual-speed floor for every non-crashed background vehicle
TRAFFIC_MAX_SPEED_MPS = 50.0

# Ordinary background-traffic dynamics. Emergency braking is a separate,
# infrequent response to an imminent closing conflict.
IDM_MAX_ACCEL_MPS2 = 3.0
IDM_COMFORTABLE_BRAKING_MPS2 = 3.0
ADT_EMERGENCY_BRAKING_MPS2 = 6.0
IDM_MIN_GAP_M = 20.0
IDM_TIME_HEADWAY_S = 1.5
IDM_ACCEL_EXPONENT = 4.0

ADT_PROFILE_MIX = (
    ("normal", 0.25),
    ("aggressive", 0.35),
    ("unstable", 0.25),
    ("slow", 0.15),
)
ADT_TAILGATE_HEADWAY_S = 0.70
ADT_TAILGATE_MIN_GAP_M = 3.0
ADT_AGGRESSIVE_MAX_ACCEL_MPS2 = 3.0  # compatibility alias; hard cap is 3.0
ADT_AGGRESSIVE_BRAKING_MPS2 = 3.0  # ordinary braking; emergency cap is separate
ADT_OVERTAKER_LANE_CHANGE_RATE_PER_S = 0.20
ADT_UNSTABLE_LANE_CHANGE_RATE_PER_S = 0.12
ADT_LANE_CHANGE_COOLDOWN_S = 1.2
ADT_LANE_CHANGE_FRONT_GAP_M = 7.0
ADT_LANE_CHANGE_REAR_GAP_M = 7.0
ADT_LANE_CHANGE_FRONT_TTC_S = 1.0
ADT_LANE_CHANGE_REAR_TTC_S = 0.9
ADT_LANE_CHANGE_IMPROVEMENT_MPS = 0.8
ADT_LANE_CHANGE_TRIGGER_GAP_M = 55.0
ADT_SUDDEN_BRAKE_EVENT_RATE_PER_S = 0.02
ADT_SUDDEN_BRAKE_ACCEL_MPS2 = -5.5
ADT_SUDDEN_BRAKE_DURATION_S = 0.5
ADT_UNSTABLE_ACCEL_EVENT_RATE_PER_S = 0.04
ADT_UNSTABLE_ACCEL_PULSE_MPS2 = 3.0
ADT_SPEED_VARIATION_MPS = 3.0

PROFILE_DESIRED_SPEED_RANGES_MPS = {"normal": (28.0, 42.0), "aggressive": (36.0, 50.0), "unstable": (25.0, 45.0), "slow": (25.0, 32.0),}
PROFILE_IDM_HEADWAY_S = {"normal": 1.5, "aggressive": 1.0, "unstable": 1.2, "slow": 1.8,}
PROFILE_IDM_MIN_GAP_M = {"normal": 5.0, "aggressive": 3.0, "unstable": 3.5, "slow": 6.0,}
PROFILE_LANE_CHANGE_RATE_PER_S = {"normal": 0.06, "aggressive": ADT_OVERTAKER_LANE_CHANGE_RATE_PER_S, "unstable": ADT_UNSTABLE_LANE_CHANGE_RATE_PER_S, "slow": 0.03,}

INITIAL_SPAWN_MAX_ATTEMPTS = 50000
BOUNDARY_SPAWN_MAX_ATTEMPTS = 200


NUM_SURROUNDING_VEHICLES = 6
VEHICLE_FEATURES = 7
NUM_VEHICLE_ROWS = NUM_SURROUNDING_VEHICLES + 1
VEHICLE_STATE_DIM = NUM_VEHICLE_ROWS * VEHICLE_FEATURES
CONTEXT_DIM = 5
# Per lane: six physical measurements plus front/rear EDM risk scores and
# normalized categorical severity codes. These are observations only.
# EDM thresholds are declared in the top-level configuration block above.
EDM_FEATURES_PER_LANE = 4
SAFETY_FEATURES_PER_LANE = 6 + EDM_FEATURES_PER_LANE
SAFETY_FEATURES_DIM = NUM_LANES * SAFETY_FEATURES_PER_LANE
STATE_DIM = VEHICLE_STATE_DIM + CONTEXT_DIM + SAFETY_FEATURES_DIM
ACTION_DIM = 2  # execution action [normalized learned acceleration, binary KEEP/CHANGE]
POLICY_ACTION_DIM = 2  # binary KEEP, CHANGE decision; acceleration is a conditional continuous head

HIDDEN_DIM = 128

# Episode-level evaluator settings. The placeholder is deterministic and does not
# call a network or pretend to be a real LLM.
EVALUATOR_BATCH_SIZE = 10
EVALUATOR_MIN_EPISODES = 10
EVALUATOR_MAX_SAFETY_WEIGHT = 3.0
EVALUATOR_MIN_SAFETY_WEIGHT = 0.5


ENABLE_EPISODE_EVALUATOR = False  # fixed reward for a clean baseline; no automatic reward adaptation

# ============================================================
# 1. Reproducibility
# ============================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 2. Utility Functions
# ============================================================

# Persistent, append-only episode trajectory archive. Each JSONL record is
# self-contained enough to replay the actual recorded vehicle states without
# running the policy or stepping a new simulator.

TRAJECTORY_RUN_ID = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _trajectory_jsonable(value):
    """Convert numpy/torch values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _trajectory_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trajectory_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _trajectory_jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return _trajectory_jsonable(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def snapshot_simulator_vehicles(simulator):
    """Capture actual world coordinates/kinematics for every vehicle at this instant."""
    base_env = simulator.env.unwrapped
    ego = getattr(base_env, "vehicle", None)
    vehicles = list(getattr(getattr(base_env, "road", None), "vehicles", []) or [])
    if ego is not None and all(vehicle is not ego for vehicle in vehicles):
        vehicles.insert(0, ego)
    records = []
    for index, vehicle in enumerate(vehicles):
        position = np.asarray(getattr(vehicle, "position", [np.nan, np.nan]), dtype=float).reshape(-1)
        lane_index = getattr(vehicle, "lane_index", None)
        try:
            lane_id = int(lane_index[2])
        except (TypeError, IndexError, ValueError):
            lane_id = None
        records.append({
            "vehicle_index": int(index),
            "is_ego": bool(vehicle is ego),
            "vehicle_class": type(vehicle).__name__,
            "vehicle_id": getattr(vehicle, "id", None),
            "position_xy_m": position[:2].tolist(),
            "lane_id": lane_id,
            "lane_index": lane_index,
            "speed_mps": float(getattr(vehicle, "speed", 0.0)),
            "heading_rad": float(getattr(vehicle, "heading", 0.0)),
            "crashed": bool(getattr(vehicle, "crashed", False)),
            "length_m": float(getattr(vehicle, "LENGTH", getattr(vehicle, "length", 5.0))),
            "width_m": float(getattr(vehicle, "WIDTH", getattr(vehicle, "width", 2.0))),
        })
    return records


def record_episode_start(simulator, phase, episode_number, initial_state):
    """Append episode metadata and the exact initial scene; does not affect simulation."""
    if not TRAJECTORY_RECORDING_ENABLED:
        return
    record = {
        "record_type": "episode_start",
        "schema_version": 1,
        "run_id": TRAJECTORY_RUN_ID,
        "phase": str(phase),
        "episode_number": int(episode_number),
        "episode_key": f"{TRAJECTORY_RUN_ID}:{phase}:{int(episode_number)}",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed_base": int(SEED),
        "max_steps": int(getattr(simulator, "max_steps", MAX_STEPS)),
        "episode_duration_s": float(MAX_EPISODE_DURATION_S),
        "policy_frequency_hz": int(policy_frequency),
        "policy_step_duration_s": float(POLICY_STEP_DURATION_S),
        "simulation_frequency_hz": int(simulation_frequency),
        "simulation_step_duration_s": float(SIMULATION_STEP_DURATION_S),
        "simulation_substeps_per_policy_action": int(SIMULATION_SUBSTEPS_PER_POLICY),
        "episode_duration_s": float(MAX_EPISODE_DURATION_S),
        "discount_gamma_per_policy_step": float(GAMMA),
        "traffic_reference_front_m": float(TRAFFIC_REFERENCE_FRONT_M),
        "traffic_reference_rear_m": float(TRAFFIC_REFERENCE_REAR_M),
        "traffic_front_recycle_boundary_m": float(TRAFFIC_FRONT_RECYCLE_BOUNDARY_M),
        "traffic_rear_recycle_boundary_m": float(TRAFFIC_REAR_RECYCLE_BOUNDARY_M),
        "traffic_max_surrounding_vehicles": int(MAX_SURROUNDING_TRAFFIC_VEHICLES),
        "traffic_episode_population_target": int(getattr(
            simulator, "_traffic_population_target", 0
        )),
        "traffic_actual_initial_population": int(
            len(simulator.env.unwrapped.road.vehicles) - 1
        ),
        "num_lanes": int(NUM_LANES),
        "managed_surrounding_vehicle_slots": int(NUM_SURROUNDING_VEHICLES),
        "speed_limit_mps": float(SPEED_LIMIT),
        "initial_state": np.asarray(initial_state, dtype=np.float32).tolist(),
        "vehicles": snapshot_simulator_vehicles(simulator),
    }
    append_trajectory_record(record)


def record_episode_transition(
    simulator, phase, episode_number, step_number, state_before, state_after,
    vehicles_before, sampled_action, executed_action, selected_target_lane,
    valid_action_mask, target_lane_mask, safety_info, step_info, reward,
    reward_components=None, done=False, terminated=False, truncated=False
):
    """Append one transition with pre/post world snapshots and decision context."""
    if not TRAJECTORY_RECORDING_ENABLED:
        return
    record = {
        "record_type": "transition",
        "schema_version": 1,
        "run_id": TRAJECTORY_RUN_ID,
        "phase": str(phase),
        "episode_number": int(episode_number),
        "episode_key": f"{TRAJECTORY_RUN_ID}:{phase}:{int(episode_number)}",
        "step": int(step_number),
        "simulated_time_s": float(step_number * POLICY_STEP_DURATION_S),
        "physics_simulation_step": int(step_number * SIMULATION_SUBSTEPS_PER_POLICY),
        "state_before": np.asarray(state_before, dtype=np.float32).tolist(),
        "state_after": np.asarray(state_after, dtype=np.float32).tolist(),
        "vehicles_before": vehicles_before,
        "vehicles_after": snapshot_simulator_vehicles(simulator),
        "policy_sampled_action": sampled_action,
        "executed_high_level_action": executed_action,
        "selected_target_lane": int(selected_target_lane),
        "valid_action_mask": valid_action_mask,
        "valid_target_lane_mask": target_lane_mask,
        "safety_info": safety_info,
        "step_info": step_info,
        "reward": float(reward),
        "reward_components": reward_components or {},
        "done": bool(done),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "ego_lane_change_stats": dict(getattr(simulator, "_lane_change_stats", {})),
    }
    append_trajectory_record(record)


def append_trajectory_record(record):
    """Append and flush a single JSONL record so completed steps survive interruption."""
    if not TRAJECTORY_RECORDING_ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(EPISODE_TRAJECTORY_PATH) or ".", exist_ok=True)
        with open(EPISODE_TRAJECTORY_PATH, "a", encoding="utf-8") as archive:
            archive.write(json.dumps(
                _trajectory_jsonable(record), ensure_ascii=False,
                allow_nan=False, separators=(",", ":")
            ) + "\n")
            archive.flush()
    except (OSError, TypeError, ValueError) as exc:
        print("[TRAJECTORY RECORD WARNING]", f"error={exc}", f"path={EPISODE_TRAJECTORY_PATH}")


def write_diagnostic_event(event, simulator=None, **fields):
    """Append one strict-JSON diagnostic record; logging failures never stop RL."""
    record = {
        "record_type": "diagnostic_event",
        "schema_version": 1,
        "run_id": TRAJECTORY_RUN_ID,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": str(event),
        "phase": getattr(simulator, "_trace_phase", None) if simulator is not None else None,
        "episode": getattr(simulator, "_trace_episode", None) if simulator is not None else None,
        "simulator_step": getattr(simulator, "step_count", None) if simulator is not None else None,
    }
    record.update(fields)
    '''
    try:
        os.makedirs(os.path.dirname(DIAGNOSTIC_LOG_PATH) or ".", exist_ok=True)
        with open(DIAGNOSTIC_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(
                _trajectory_jsonable(record),
                ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ) + "\n")
            log_file.flush()
    except (OSError, TypeError, ValueError) as exc:
        print("[DIAGNOSTIC LOG WARNING]", f"event={event}", f"path={DIAGNOSTIC_LOG_PATH}", f"error={exc}")
    '''

def _safe_float_for_diagnostic(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else repr(value)
    except (TypeError, ValueError):
        return repr(value)


def lane_action_index_to_delta(lane_action_index):
    """Map categorical policy index 0/1/2 to semantic relative lane delta -1/0/+1."""
    index = int(np.clip(np.rint(float(lane_action_index)), 0, 2))
    return index - 1


def lane_action_delta_to_index(lane_action_delta):
    """Map semantic relative lane delta -1/0/+1 to categorical policy index 0/1/2."""
    delta = int(np.clip(np.rint(float(lane_action_delta)), -1, 1))
    return delta + 1


def relative_lane_delta_to_absolute_lane(current_lane_id, lane_action_delta):
    """Convert LEFT/KEEP/RIGHT intent to absolute Highway lane ID.

    This highway-v0 road graph numbers lanes in the opposite direction to the
    semantic convention: absolute lane IDs increase toward the left. Therefore
    LEFT (-1) maps to current_lane_id + 1, KEEP (0) to current_lane_id, and
    RIGHT (+1) to current_lane_id - 1. Absolute IDs remain geometry identifiers.
    """
    current_lane_id = int(current_lane_id)
    delta = int(np.clip(np.rint(float(lane_action_delta)), -1, 1))
    return current_lane_id - delta


def to_tensor(x):
    """Convert input to a float32 tensor on DEVICE."""
    if isinstance(x, torch.Tensor):
        return x.to(device=DEVICE, dtype=torch.float32)

    return torch.as_tensor(
        np.asarray(x),
        dtype=torch.float32,
        device=DEVICE
    )




class ActorCriticPolicy(nn.Module):
    """Conditional hybrid actor: KEEP/CHANGE -> target lane -> continuous acceleration."""
    def __init__(self, state_dim, action_dim, hidden_dim, steering_scale=None):
        super().__init__()
        if action_dim != 2:
            raise ValueError("Expected execution action [acceleration, binary KEEP/CHANGE]")
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.action_logits = nn.Linear(hidden_dim, 2)
        self.lane_rank_logits = nn.Linear(hidden_dim, NUM_LANES)
        accel_context_dim = hidden_dim + 2 + NUM_LANES
        self.acceleration_mean = nn.Sequential(
            nn.Linear(accel_context_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.acceleration_log_std = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                    nn.Linear(hidden_dim // 2, 1))
        self.optimizer = Adam(self.parameters(), lr=POLICY_LR)

    def features(self, state):
        x = to_tensor(state)
        if x.ndim == 1: x = x.unsqueeze(0)
        return self.shared(x)

    def action_distributions(self, state, valid_action_mask=None):
        logits = self.action_logits(self.features(state))
        if valid_action_mask is not None:
            mask = torch.as_tensor(valid_action_mask, dtype=torch.bool, device=logits.device)
            if mask.ndim == 1: mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and logits.shape[0] > 1: mask = mask.expand(logits.shape[0], -1)
            if mask.shape != logits.shape: raise ValueError(f"Binary mask shape {mask.shape} != logits {logits.shape}")
            if not mask.any(dim=-1).all(): raise ValueError("Every state needs a valid binary action")
            logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def lane_distributions(self, state, valid_target_lane_mask=None):
        logits = self.lane_rank_logits(self.features(state))
        if valid_target_lane_mask is not None:
            mask = torch.as_tensor(valid_target_lane_mask, dtype=torch.bool, device=logits.device)
            if mask.ndim == 1: mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and logits.shape[0] > 1: mask = mask.expand(logits.shape[0], -1)
            if mask.shape != logits.shape: raise ValueError(f"Target mask shape {mask.shape} != logits {logits.shape}")
            empty = ~mask.any(dim=-1)
            if empty.any():
                mask = mask.clone()
                mask[empty] = True
            logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def acceleration_distribution(self, state, binary_action, target_lane):
        """Gaussian over pre-tanh acceleration, conditioned on both lane decisions."""
        features = self.features(state)
        batch = features.shape[0]
        binary = torch.as_tensor(binary_action, dtype=torch.long, device=features.device).reshape(-1)
        if binary.numel() == 1 and batch > 1: binary = binary.expand(batch)
        target = torch.as_tensor(target_lane, dtype=torch.long, device=features.device).reshape(-1)
        if target.numel() == 1 and batch > 1: target = target.expand(batch)
        binary = binary.clamp(0, 1)
        target = target.clamp(0, NUM_LANES - 1)
        context = torch.cat([
            features,
            F.one_hot(binary, num_classes=2).to(features.dtype),
            F.one_hot(target, num_classes=NUM_LANES).to(features.dtype)
        ], dim=-1)
        mean = self.acceleration_mean(context).squeeze(-1)
        log_std = self.acceleration_log_std.clamp(-3.0, 0.5).expand_as(mean)
        return torch.distributions.Normal(mean, log_std.exp())

    @staticmethod
    def squashed_normal_log_prob(distribution, pre_tanh_value):
        z = torch.as_tensor(pre_tanh_value, dtype=distribution.mean.dtype,
                            device=distribution.mean.device).reshape_as(distribution.mean)
        return distribution.log_prob(z) - torch.log(1.0 - torch.tanh(z).pow(2) + 1e-6)

    def act(self, state, deterministic=False, valid_action_mask=None,
            valid_target_lane_mask=None, current_lane_id=None, return_aux=False):
        """Sample KEEP/CHANGE, then target lane if needed, then conditional acceleration."""
        dist = self.action_distributions(state, valid_action_mask)
        sampled_binary = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        binary_action = int(sampled_binary[0].detach().cpu().item())
        current = int(current_lane_id) if current_lane_id is not None else 0
        target = current
        if binary_action == 1 and valid_target_lane_mask is not None:
            tm = torch.as_tensor(valid_target_lane_mask, dtype=torch.bool, device=dist.logits.device)
            if tm.ndim == 1: tm = tm.unsqueeze(0)
            if tm.shape[0] == 1 and dist.logits.shape[0] > 1: tm = tm.expand(dist.logits.shape[0], -1)
            change_mask = tm.clone()
            if 0 <= current < change_mask.shape[-1]: change_mask[:, current] = False
            if change_mask.any(dim=-1).all():
                lane_dist = self.lane_distributions(state, change_mask)
                lane_index = torch.argmax(lane_dist.logits, dim=-1) if deterministic else lane_dist.sample()
                target = int(lane_index[0].detach().cpu().item())
            else:
                binary_action, target = 0, current
        accel_dist = self.acceleration_distribution(state, binary_action, target)
        pre_tanh = accel_dist.mean if deterministic else accel_dist.sample()
        action = np.asarray([
            float(torch.tanh(pre_tanh[0]).detach().cpu().item()), float(binary_action)
        ], dtype=np.float32)
        aux = {
            "pre_tanh_accel": float(pre_tanh[0].detach().cpu().item()),
            "selected_target_lane": int(target),
            "acceleration_log_prob": float(self.squashed_normal_log_prob(accel_dist, pre_tanh)[0].detach().cpu().item())
        }
        return (action, aux) if return_aux else action

    def __call__(self, state, valid_action_mask=None, valid_target_lane_mask=None, current_lane_id=None):
        return self.act(state, valid_action_mask=valid_action_mask,
                        valid_target_lane_mask=valid_target_lane_mask,
                        current_lane_id=current_lane_id)

    def value(self, state):
        return self.critic(self.features(state)).squeeze(-1)


def lane_policy_diagnostic(policy, state, valid_action_mask, sampled_action, phase, episode, step,
                           selected_target_lane=-1, valid_target_lane_mask=None, current_lane_id=None):
    with torch.no_grad():
        x = to_tensor(state)
        if x.ndim == 1: x = x.unsqueeze(0)
        h = policy.shared(x)
        lane_logits = policy.lane_rank_logits(h)
        if valid_target_lane_mask is not None:
            lm = torch.as_tensor(valid_target_lane_mask, dtype=torch.bool, device=lane_logits.device)
            if lm.ndim == 1: lm = lm.unsqueeze(0)
            if lm.shape[0] == 1 and lane_logits.shape[0] > 1: lm = lm.expand(lane_logits.shape[0], -1)
            lane_probs = torch.softmax(lane_logits.masked_fill(~lm, -1e9), dim=-1) if lm.any(dim=-1).all() else torch.zeros_like(lane_logits)
        else:
            lane_probs = torch.softmax(lane_logits, dim=-1)
    i = int(np.clip(np.rint(float(sampled_action[1])), 0, 1))
    print("[LANE RANKING DECISION]", f"phase={phase}", f"episode={episode}", f"step={step}",
          f"current_lane={current_lane_id}",
          f"valid_target_lane_mask={None if valid_target_lane_mask is None else np.asarray(valid_target_lane_mask,dtype=bool).tolist()}",
          f"target_lane_probs={np.round(lane_probs.squeeze(0).cpu().numpy(),4).tolist()}",
          f"selected_target_lane={selected_target_lane}",
          f"derived_action={('KEEP','CHANGE')[i]}")

# ============================================================
# 6. Highway-env Simulator Adapter
# ============================================================


def ttc_to_risk(ttc_seconds):
    """Compatibility wrapper around the single canonical EDM TTC risk scale."""
    return edm_ttc_risk_score(ttc_seconds)


def edm_ttc_risk_score(ttc_seconds):
    """Map TTC to a warning score: 0 at comfortable TTC, 1 at danger TTC.

    The score rises linearly from zero at EDM_TTC_COMFORTABLE_S to one at
    EDM_TTC_DANGER_S, then remains saturated at one for more dangerous TTCs.
    """
    try:
        ttc = float(ttc_seconds)
    except (TypeError, ValueError):
        return 1.0
    if np.isnan(ttc):
        return 1.0
    if np.isinf(ttc):
        return 0.0
    warning_range = EDM_TTC_COMFORTABLE_S - EDM_TTC_DANGER_S
    if warning_range <= 0.0:
        raise ValueError(
            "EDM_TTC_COMFORTABLE_S must be greater than EDM_TTC_DANGER_S"
        )
    return float(np.clip(
        (EDM_TTC_COMFORTABLE_S - ttc) / warning_range, 0.0, 1.0
    ))


def interpret_edm_ttc(ttc_seconds):
    """Translate TTC into an interpretable band; this never controls actions."""
    try:
        ttc = float(ttc_seconds)
    except (TypeError, ValueError):
        return {"category": "UNKNOWN", "severity": -1, "risk_score": 1.0}
    if np.isnan(ttc):
        return {"category": "UNKNOWN", "severity": -1, "risk_score": 1.0}
    if np.isinf(ttc):
        return {"category": "NO_CLOSING_THREAT", "severity": 0, "risk_score": 0.0}
    if ttc < EDM_TTC_CRITICAL_S:
        category, severity = "CRITICAL", 3
    elif ttc <= EDM_TTC_DANGER_S:
        category, severity = "DANGER", 2
    elif ttc < EDM_TTC_COMFORTABLE_S:
        category, severity = "CAUTION", 1
    else:
        category, severity = "COMFORTABLE", 0
    return {
        "category": category,
        "severity": severity,
        "risk_score": edm_ttc_risk_score(ttc),
    }


def estimate_edm_action_warning(
    edm_pre_action_by_lane,
    current_lane_id,
    selected_target_lane,
    lane_change_action,
    ego_speed_mps,
    normalized_acceleration,
    horizon_s=EDM_REWARD_HORIZON_S,
):
    """Estimate short-horizon TTC warning for the lane interaction implied by the action.

    KEEP evaluates front/rear interactions in the current lane. CHANGE evaluates
    front/rear lane-entry interactions in the selected target lane. The selected
    acceleration projects ego speed and bumper gaps to the configured horizon
    endpoint, assuming constant neighbour speeds. This is an endpoint projection,
    not a sequential multi-step traffic rollout. The returned value is a continuous
    TTC risk proxy, not a calibrated probability.
    """
    current_lane_id = int(current_lane_id)
    target_lane_id = int(selected_target_lane)
    changing = bool(int(round(float(lane_change_action))) == 1 and target_lane_id != current_lane_id)
    evaluated_lane_id = target_lane_id if changing else current_lane_id
    lane_values = (
        edm_pre_action_by_lane.get(evaluated_lane_id)
        or edm_pre_action_by_lane.get(str(evaluated_lane_id))
        or {}
    )
    dt = max(0.0, float(horizon_s))
    ego_speed = max(0.0, float(ego_speed_mps))
    ego_acceleration = float(np.clip(normalized_acceleration, -1.0, 1.0)) * MAX_PHYSICAL_ACCELERATION
    ego_speed_end = max(0.0, ego_speed + ego_acceleration * dt)

    def projected_ttc(gap_key, neighbour_speed_key, side):
        try:
            gap = float(lane_values.get(gap_key, float("inf")))
            neighbour_speed = float(lane_values.get(neighbour_speed_key))
        except (TypeError, ValueError):
            return float("inf"), 0.0
        if not np.isfinite(gap) or not np.isfinite(neighbour_speed):
            return float("inf"), 0.0
        if side == "front":
            gap_end = gap + (neighbour_speed - ego_speed) * dt - 0.5 * ego_acceleration * dt * dt
            closing_speed_end = ego_speed_end - neighbour_speed
        else:
            gap_end = gap + (ego_speed - neighbour_speed) * dt + 0.5 * ego_acceleration * dt * dt
            closing_speed_end = neighbour_speed - ego_speed_end
        if gap_end <= 0.0:
            ttc_end = 0.0
        elif closing_speed_end > 1e-6:
            ttc_end = gap_end / closing_speed_end
        else:
            ttc_end = float("inf")
        return float(ttc_end), float(edm_ttc_risk_score(ttc_end))

    front_ttc, front_risk = projected_ttc("front_gap_m", "front_speed_mps", "front")
    rear_ttc, rear_risk = projected_ttc("rear_gap_m", "rear_speed_mps", "rear")
    warning_score = float(np.clip(max(front_risk, rear_risk), 0.0, 1.0))
    return {
        "enabled": bool(EDM_REWARD_ENABLED),
        "action_type": "CHANGE" if changing else "KEEP",
        "current_lane_id": current_lane_id,
        "evaluated_lane_id": evaluated_lane_id,
        "risk_basis": "target_lane_entry_front_rear" if changing else "current_lane_front_rear",
        "horizon_s": dt,
        "ego_acceleration_mps2": ego_acceleration,
        "front_projected_ttc_s": front_ttc,
        "rear_projected_ttc_s": rear_ttc,
        "front_warning_score": front_risk,
        "rear_warning_score": rear_risk,
        "warning_score": warning_score,
        "warning_score_semantics": "continuous_TTC_risk_proxy_not_calibrated_probability",
    }


class HighwaySimulator:
    """Highway-env wrapper with ADT background traffic and six ego observation slots.

    The simulator maintains a distributed dynamic traffic population. The
    policy observes only the nearest front/rear vehicle in each of three lanes.
    State includes vehicle rows, ego-relative context, six physical lane
    metrics plus four EDM interpretation features for each of the roles
    [current, left, right]. State dimension is derived from configuration.
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 max_steps=MAX_STEPS, render_mode=None):
        if gym is None or highway_env is None:
            raise ImportError(
                "Highway-env dependencies are missing. Install with: "
                "pip install highway-env gymnasium"
            )
        if state_dim != STATE_DIM or action_dim != ACTION_DIM:
            raise ValueError(f"HighwaySimulator expects STATE_DIM={STATE_DIM} and execution ACTION_DIM={ACTION_DIM}")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = int(max_steps)
        if self.max_steps != MAX_EPISODE_STEPS:
            raise ValueError(
                f"max_steps={self.max_steps} does not match "
                f"MAX_EPISODE_STEPS={MAX_EPISODE_STEPS}"
            )
        self.step_count = 0
        self.render_mode = render_mode
        self.env = gym.make(
            "highway-v0",
            render_mode=render_mode,
            config={
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": NUM_VEHICLE_ROWS,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-1000.0, 1000.0],
                        "y": [-12.0, 12.0],
                        "vx": [-40.0, 40.0],
                        "vy": [-10.0, 10.0],
                    },
                    "absolute": False,
                    "normalize": True,
                    "order": "sorted",
                },
                "action": {"type": "ContinuousAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.1745329, 0.1745329],
                },

                "lanes_count": NUM_LANES,
                # Background traffic is created exclusively by the continuous
                # population manager after reset; disable HighwayEnv random cars.
                "vehicles_count": 0,
                "duration": MAX_EPISODE_DURATION_S,
                "offroad_terminal": False,

                # HighwayEnv advances an integer number of physical updates per
                # policy action: 15 Hz / 5 Hz = 3 simulator updates per decision.
                "simulation_frequency": simulation_frequency,
                "policy_frequency": policy_frequency,

            },
        )



        self.state = None



    def _encode_state(self, obs):
        obs = np.asarray(obs, dtype=np.float32)


        rows = np.zeros((NUM_VEHICLE_ROWS, VEHICLE_FEATURES), dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] < VEHICLE_FEATURES:
            raise ValueError(f"Unexpected Highway observation shape: {obs.shape}")
        n = min(NUM_VEHICLE_ROWS, obs.shape[0])
        rows[:n] = obs[:n, :VEHICLE_FEATURES]
        if n < NUM_VEHICLE_ROWS:
            print("[OBSERVATION WARNING]", f"expected {NUM_VEHICLE_ROWS} rows, got {n}")

        #print("copy the raw_obs to rows:", rows)

        vehicle = self.env.unwrapped.vehicle
        lane_id = 0.0
        try:
            lane_id = float(vehicle.lane_index[2])
        except (TypeError, IndexError, AttributeError):
            pass
        # Do not expose the absolute lane ID as a policy feature. Encode only
        # whether a left/right adjacent lane physically exists.
        left_candidate_exists = float(lane_id > 0)
        right_candidate_exists = float(lane_id < NUM_LANES - 1)
        context = np.asarray([
            np.clip(float(vehicle.speed) / max(SPEED_LIMIT, 1e-6), 0.0, 1.5),
            left_candidate_exists,
            right_candidate_exists,
            float(bool(getattr(vehicle, "crashed", False))),
            min(self.step_count / max(1, self.max_steps), 1.0),
        ], dtype=np.float32)
        state = np.concatenate([rows.reshape(-1), context])

        #print("at the end of _encode_state(), the state becomes: ", state)

        safety = np.zeros(SAFETY_FEATURES_DIM, dtype=np.float32)
        try:
            metrics = self._min_ttc_by_lane()
            keys = ["front_gap", "rear_gap", "front_ttc", "rear_ttc",
                    "front_speed", "rear_speed"]
            scales = [100.0, 100.0, 10.0, 10.0, 40.0, 40.0]
            # Ego-relative lane roles prevent the appended EDM block from
            # hard-coding physical lane 0/1/2 as different semantic categories:
            # [current lane, left candidate, right candidate]. Missing boundary
            # candidates retain neutral sentinel values and are separately
            # identified by the two availability bits in context.
            role_lanes = [lane_id, lane_id - 1, lane_id + 1]
            for role_index, physical_lane_id in enumerate(role_lanes):
                vals = metrics.get(physical_lane_id, {}) if 0 <= physical_lane_id < NUM_LANES else {}
                base = role_index * SAFETY_FEATURES_PER_LANE
                for j, (key, scale) in enumerate(zip(keys, scales)):
                    value = vals.get(key, scale)
                    value = float(value) if value is not None else scale
                    if not np.isfinite(value):
                        value = scale * 1.5
                    safety[base + j] = np.clip(value / scale, -1.0, 1.5)
                front_edm = interpret_edm_ttc(vals.get("front_ttc", float("inf")))
                rear_edm = interpret_edm_ttc(vals.get("rear_ttc", float("inf")))
                safety[base + 6] = front_edm["risk_score"]
                safety[base + 7] = rear_edm["risk_score"]
                safety[base + 8] = (
                    1.0 if front_edm["severity"] < 0 else front_edm["severity"] / 3.0
                )
                safety[base + 9] = (
                    1.0 if rear_edm["severity"] < 0 else rear_edm["severity"] / 3.0
                )
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            pass
        return np.nan_to_num(np.concatenate([state,safety]),nan=0.0,posinf=1.0,neginf=-1.0).astype(np.float32)




    def reset(self, seed=None):

        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)

        # ADT uses a randomized, capped continuous population and side-specific
        # boundary recycling; the six observation slots do not limit world traffic.
        print(
            "[ADT TRAFFIC SCENARIO]",
            f"maximum_surrounding_population={MAX_SURROUNDING_TRAFFIC_VEHICLES}",
            f"initial_population_range={MIN_INITIAL_SURROUNDING_TRAFFIC_VEHICLES}-{MAX_SURROUNDING_TRAFFIC_VEHICLES}",
            f"reference_front=+{TRAFFIC_REFERENCE_FRONT_M:.0f}m",
            f"reference_rear=-{TRAFFIC_REFERENCE_REAR_M:.0f}m",
            f"entry_buffer={TRAFFIC_ENTRY_BUFFER_M:.0f}m",
            f"recycle_front=+{TRAFFIC_FRONT_RECYCLE_BOUNDARY_M:.0f}m",
            f"recycle_rear=-{TRAFFIC_REAR_RECYCLE_BOUNDARY_M:.0f}m",
            f"desired_speed_range={TRAFFIC_MIN_SPEED_MPS:.0f}-{TRAFFIC_MAX_SPEED_MPS:.0f}m/s",
            f"ordinary_acceleration_cap={IDM_MAX_ACCEL_MPS2:.1f}m/s2",
            f"ordinary_braking_cap={IDM_COMFORTABLE_BRAKING_MPS2:.1f}m/s2",
            f"emergency_braking_cap={ADT_EMERGENCY_BRAKING_MPS2:.1f}m/s2",
            f"profile_mix={dict(ADT_PROFILE_MIX)}",
            f"simulation_frequency={simulation_frequency}Hz",
            "ego_observation_slots=6",
            "traffic_speed_floor_applies_to_desired_and_spawn_speeds_only",
        )

        self.step_count = 0
        # One collision event is recorded per uninterrupted contact episode.
        # The simulator crash latch is still cleared after each update so the
        # RL episode can continue, but sustained overlap is not re-counted as
        # a fresh impact on every subsequent policy step.
        self._collision_contact_active = False
        base_env = self.env.unwrapped
        self._initialize_continuous_traffic()
        obs = base_env.observation_type.observe()

        # Store the target lane once at the beginning of this episode.
        # Do NOT recalculate it at every time step.
        base_env = self.env.unwrapped
        ego_vehicle = base_env.vehicle

        self.target_lane_index = ego_vehicle.lane_index
        self._lane_change_source_index = None
        self._lane_change_active = False
        self._lane_change_cooldown = 0
        self._lane_change_stats = {
            "requested": 0, "accepted": 0, "rejected": 0,
            "started": 0, "completed": 0, "timed_out": 0,
            "completed_LEFT": 0, "completed_RIGHT": 0,
        }
        self._lane_change_started_step = None
        self._lane_maneuver_sequence = 0
        self._active_lane_maneuver_id = None
        self._active_lane_request_trace_id = None
        self._current_lane_request_trace_id = None
        self._lane_change_timeout_logged = False
        self._lane_change_stable_steps = 0
        self._lane_change_stats["aborted"] = 0

        print(
            "[TARGET LANE]",
            self.target_lane_index,
            "initial_position=",
            np.asarray(ego_vehicle.position).round(3),
        )

        self.state = self._encode_state(obs)
        self._assert_continuous_traffic_invariant("reset")
        return self.state.copy()




    # All vehicles are projected onto the same reference lane coordinate.
    # In highway-v0, the three lanes share the same road segment and longitudinal
    # axis; using one reference lane avoids comparing independently projected s
    # values from different lane geometries.
    def _lane_longitudinal_position(self, vehicle, lane_index):
        lane = self.env.unwrapped.road.network.get_lane(lane_index)
        longitudinal, _ = lane.local_coordinates(np.asarray(vehicle.position, dtype=float))
        return float(longitudinal)

    def _road_longitudinal_position(self, vehicle):
        """Return longitudinal s in one shared road-coordinate frame."""
        ego_lane_index = self.env.unwrapped.vehicle.lane_index
        reference_lane_index = (ego_lane_index[0], ego_lane_index[1], 0)
        return self._lane_longitudinal_position(vehicle, reference_lane_index)

    def _lane_index_for_id(self, lane_id):
        ego_lane_index = self.env.unwrapped.vehicle.lane_index
        return (ego_lane_index[0], ego_lane_index[1], int(lane_id))

    def _relevant_lane_ids(self, include_active_lane_change=True):
        """Return the single canonical lane set relevant to ego safety/reward.

        Always include the physically occupied lane. While a lane change is
        active, also include its source and target lanes. Adjacent lanes are
        evaluated separately only when considering a candidate lane change.
        """
        ego = self.env.unwrapped.vehicle
        lane_ids = {int(ego.lane_index[2])}
        if include_active_lane_change and getattr(self, "_lane_change_active", False):
            for lane_index in (
                getattr(self, "_lane_change_source_index", None),
                getattr(self, "target_lane_index", None),
            ):
                if lane_index is not None:
                    try:
                        lane_ids.add(int(lane_index[2]))
                    except (TypeError, ValueError, IndexError):
                        pass
        return lane_ids

    def _min_ttc_by_lane(self):
        """Return the nearest front/rear vehicle and TTC for every lane.

        TTC is based on signed bumper-to-bumper clearance and positive closing
        speed. A present vehicle whose body overlaps the ego has TTC=0 even if
        the relative speed is currently non-closing. A missing or non-closing
        vehicle has TTC=inf. This method is diagnostic/state information only:
        it does not veto, replace, or modify the RL agent's action.

        EDM consumes this same snapshot, ensuring its front/rear risk
        interpretation uses the same neighbours and bumper gaps.
        """
        base_env = self.env.unwrapped
        road = base_env.road
        ego = base_env.vehicle
        lanes_count = int(base_env.config["lanes_count"])
        ego_lane = int(ego.lane_index[2])
        ego_speed = max(0.0, float(ego.speed))
        ego_s = float(self._road_longitudinal_position(ego))
        ego_length = float(getattr(ego, "LENGTH", getattr(ego, "length", 5.0)))

        lane_ttc = {
            lane_id: {
                "front_ttc": float("inf"), "rear_ttc": float("inf"),
                "min_ttc": float("inf"),
                "front_gap": float("inf"), "rear_gap": float("inf"),
                "front_speed": None, "rear_speed": None,
                "nearest_front_ttc": float("inf"),
                "front_center_gap": float("inf"),
                "rear_center_gap": float("inf"),
                "front_vehicle_id": None, "rear_vehicle_id": None,
            }
            for lane_id in range(lanes_count)
        }

        nearest = {
            (lane_id, side): None
            for lane_id in range(lanes_count)
            for side in ("front", "rear")
        }

        for other in list(road.vehicles):
            if other is ego or bool(getattr(other, "crashed", False)):
                continue
            try:
                lane_id = int(other.lane_index[2])
                if lane_id not in lane_ttc:
                    continue
                other_s = float(self._road_longitudinal_position(other))
                other_speed = max(0.0, float(other.speed))
                other_length = float(getattr(
                    other, "LENGTH", getattr(other, "length", 5.0)
                ))
            except (AttributeError, TypeError, ValueError, IndexError, KeyError):
                continue
            if not (np.isfinite(other_s) and np.isfinite(other_speed)
                    and np.isfinite(other_length) and other_length > 0.0):
                continue

            center_gap = other_s - ego_s
            if (
                center_gap > TRAFFIC_REFERENCE_FRONT_M
                or center_gap < -TRAFFIC_REFERENCE_REAR_M
            ):
                continue
            if center_gap == 0.0:
                # Longitudinal overlap/alignment is represented as an immediate
                # collision-risk signal on both sides of this lane.
                overlap_gap = -0.5 * (ego_length + other_length)
                values = lane_ttc[lane_id]
                values.update({
                    "front_gap": overlap_gap, "rear_gap": overlap_gap,
                    "front_speed": other_speed, "rear_speed": other_speed,
                    "front_ttc": 0.0, "rear_ttc": 0.0,
                    "nearest_front_ttc": 0.0,
                    "front_center_gap": 0.0, "rear_center_gap": 0.0,
                    "front_vehicle_id": id(other), "rear_vehicle_id": id(other),
                })
                continue

            side = "front" if center_gap > 0.0 else "rear"
            bumper_gap = abs(center_gap) - 0.5 * (ego_length + other_length)
            # Select by nearest longitudinal centre, matching the six-slot
            # front/rear observation convention; retain signed bumper clearance.
            distance = abs(center_gap)
            key = (lane_id, side)
            current = nearest[key]
            if current is None or distance < current["distance"]:
                nearest[key] = {
                    "distance": distance,
                    "center_gap": center_gap,
                    "bumper_gap": bumper_gap,
                    "speed": other_speed,
                    "vehicle_id": id(other),
                }

        for (lane_id, side), neighbor in nearest.items():
            if neighbor is None:
                continue
            values = lane_ttc[lane_id]
            gap = float(neighbor["bumper_gap"])
            other_speed = float(neighbor["speed"])
            if gap <= 0.0:
                ttc = 0.0
            elif side == "front":
                closing_speed = ego_speed - other_speed
                ttc = gap / closing_speed if closing_speed > 1e-6 else float("inf")
            else:
                closing_speed = other_speed - ego_speed
                ttc = gap / closing_speed if closing_speed > 1e-6 else float("inf")

            if side == "front":
                values.update({
                    "front_gap": gap,
                    "front_speed": other_speed,
                    "front_ttc": float(ttc),
                    "nearest_front_ttc": float(ttc),
                    "front_center_gap": float(neighbor["distance"]),
                    "front_vehicle_id": neighbor["vehicle_id"],
                })
            else:
                values.update({
                    "rear_gap": gap,
                    "rear_speed": other_speed,
                    "rear_ttc": float(ttc),
                    "rear_center_gap": float(neighbor["distance"]),
                    "rear_vehicle_id": neighbor["vehicle_id"],
                })

        for values in lane_ttc.values():
            values["min_ttc"] = min(values["front_ttc"], values["rear_ttc"])
            values["front_risk"] = ttc_to_risk(values["front_ttc"])
            values["rear_risk"] = ttc_to_risk(values["rear_ttc"])
            values["lane_risk"] = max(values["front_risk"], values["rear_risk"])

        # Empty slots are valid: the reference window does not force the
        # simulator to manufacture a front/rear vehicle in every lane.
        return lane_ttc

    def _edm_interpretation_by_lane(self, lane_metrics=None):
        """Readable per-lane EDM output from the canonical neighbour/TTC snapshot."""
        lane_metrics = self._min_ttc_by_lane() if lane_metrics is None else lane_metrics
        result = {}
        for lane_id, values in sorted(lane_metrics.items()):
            front = interpret_edm_ttc(values.get("front_ttc", float("inf")))
            rear = interpret_edm_ttc(values.get("rear_ttc", float("inf")))
            result[int(lane_id)] = {
                "front_ttc_s": float(values.get("front_ttc", float("inf"))),
                "rear_ttc_s": float(values.get("rear_ttc", float("inf"))),
                "front_gap_m": float(values.get("front_gap", float("inf"))),
                "rear_gap_m": float(values.get("rear_gap", float("inf"))),
                "front_speed_mps": values.get("front_speed"),
                "rear_speed_mps": values.get("rear_speed"),
                "front_vehicle_id": values.get("front_vehicle_id"),
                "rear_vehicle_id": values.get("rear_vehicle_id"),
                "front_category": front["category"],
                "rear_category": rear["category"],
                "front_severity": int(front["severity"]),
                "rear_severity": int(rear["severity"]),
                "front_risk_score": float(front["risk_score"]),
                "rear_risk_score": float(rear["risk_score"]),
                "lane_max_risk_score": float(max(front["risk_score"], rear["risk_score"])),
            }
        return result

    # Legacy TTC/speed-matching intervention code removed; EDM remains diagnostic/state input only.


    def _snapshot_managed_traffic_slots(self):
        """Capture every ADT background vehicle around one native env.step()."""
        base_env = self.env.unwrapped
        ego = getattr(base_env, "vehicle", None)
        snapshot = {}
        for vehicle in list(getattr(getattr(base_env, "road", None), "vehicles", [])):
            if vehicle is ego:
                continue
            side = getattr(vehicle, "_arood_traffic_side", getattr(vehicle, "_arood_adt_profile", "dynamic"))
            try:
                speed = float(vehicle.speed)
            except (AttributeError, TypeError, ValueError):
                speed = float("nan")
            try:
                position = np.asarray(vehicle.position, dtype=float).tolist()
            except (AttributeError, TypeError, ValueError):
                position = None
            try:
                velocity = np.asarray(vehicle.velocity, dtype=float).tolist()
            except (AttributeError, TypeError, ValueError):
                velocity = None
            action = getattr(vehicle, "action", None)
            if isinstance(action, dict):
                action = {str(k): (float(v) if isinstance(v, (int, float, np.number)) else repr(v))
                          for k, v in action.items()}
            else:
                action = repr(action)
            key = id(vehicle)
            snapshot[key] = {
                "vehicle_id": key,
                "lane_index": repr(getattr(vehicle, "lane_index", None)),
                "slot_side": side,
                "speed_mps": speed,
                "position": position,
                "velocity": velocity,
                "heading_rad": _safe_float_for_diagnostic(getattr(vehicle, "heading", None)),
                "action": action,
                "crashed": bool(getattr(vehicle, "crashed", False)),
            }
        return snapshot

    def _record_first_negative_speed_transition(self, before_snapshot, after_snapshot,
                                                physical_action, env_info=None):
        """Persist the first observed negative-speed transition without hiding it via clamping."""
        incidents = []
        for vehicle_id, after in after_snapshot.items():
            after_speed = after.get("speed_mps", float("nan"))
            before = before_snapshot.get(vehicle_id)
            before_speed = before.get("speed_mps", float("nan")) if before else None
            became_negative = (
                np.isfinite(after_speed) and after_speed < 0.0
                and before is not None and np.isfinite(before_speed) and before_speed >= 0.0
            )
            already_negative = (
                np.isfinite(after_speed) and after_speed < 0.0
                and (before is None or not np.isfinite(before_speed) or before_speed < 0.0)
            )
            if became_negative or already_negative:
                incidents.append({
                    "transition_type": "nonnegative_to_negative" if became_negative else "negative_first_observed",
                    "before": before,
                    "after": after,
                })

        if not incidents:
            return

        report = {
            "event": "FIRST_NEGATIVE_MANAGED_TRAFFIC_SPEED",
            "phase": getattr(self, "_trace_phase", None),
            "episode": getattr(self, "_trace_episode", None),
            "policy_step": int(getattr(self, "step_count", 0)),
            "physics_simulation_step": int(
                getattr(self, "step_count", 0) * SIMULATION_SUBSTEPS_PER_POLICY
            ),
            "simulated_time_s": float(
                getattr(self, "step_count", 0) * POLICY_STEP_DURATION_S
            ),
            "native_call": "self.env.step(physical_action)",
            "physical_action_applied_to_ego": np.asarray(physical_action, dtype=float).tolist(),
            "env_info": repr(env_info),
            "incident_count_in_first_observed_step": len(incidents),
            "incidents": incidents,
            "note": "Captured immediately after env.step and before post-step sanitization or slot maintenance.",
        }

        path = NEGATIVE_SPEED_FIRST_TRANSITION_PATH
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # Exclusive creation ensures later incidents cannot overwrite the first report.
            with open(path, "x", encoding="utf-8") as log_file:
                log_file.write(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=True))
                log_file.write("\n")
            print("[FIRST NEGATIVE TRAFFIC SPEED CAPTURED]", f"path={path}",
                  f"incident_count={len(incidents)}")
        except FileExistsError:
            pass
        except (OSError, TypeError, ValueError) as exc:
            print("[NEGATIVE SPEED DIAGNOSTIC WRITE FAILED]", f"path={path}", f"error={exc}")


        # Also write to the existing JSONL diagnostic stream, regardless of whether
        # the first-incident text file already exists.
        report_fields = dict(report)
        report_fields.pop("event", None)
        write_diagnostic_event("negative_managed_traffic_speed_observed", self, **report_fields)


    def step(self, action, sampled_action=None, safety_info=None):
        """Execute one policy action over the configured simulator substeps.

        The policy chooses KEEP or CHANGE and selects a physically adjacent target.
        No TTC/EDM safety shield, speed controller, or action override is applied.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Expected hybrid action [acceleration, lane_action], got {action.shape}")
        incoming_action = action.copy()

        # Pre-action traffic context: descriptive only, never an action veto.
        edm_pre_action_metrics = self._min_ttc_by_lane()
        edm_pre_action = self._edm_interpretation_by_lane(edm_pre_action_metrics)
        lane_before_action = int(self.env.unwrapped.vehicle.lane_index[2])

        # Pass the sampled policy action through unchanged.
        action = incoming_action.copy()
        safety_info = dict(safety_info or {})
        safety_info["simulator_input_action"] = incoming_action.copy()
        safety_info["executed_high_level_action"] = action.copy()

        base_env = self.env.unwrapped
        ego = base_env.vehicle
        source_lane_id = int(ego.lane_index[2])
        acceleration = float(np.clip(action[0], -1.0, 1.0))
        lane_action_id = int(np.clip(np.rint(action[1]), 0, 1))
        lane_name = ("KEEP", "CHANGE")[lane_action_id]
        requested_target_lane_id = int(safety_info.get("executed_target_lane_id", safety_info.get("requested_target_lane_id", source_lane_id)))

        lane_changed = False
        lane_rejection_reason = safety_info.get("lane_change_rejection_reason", "none")
        if lane_action_id == 1 and requested_target_lane_id != source_lane_id:
            if not safety_info.get("lane_change_blocked", False):
                try:
                    source_lane_index = tuple(ego.lane_index)
                    source_lane = base_env.road.network.get_lane(source_lane_index)
                    longitudinal, _ = source_lane.local_coordinates(np.asarray(ego.position, dtype=float))
                    target_lane_index = (
                        source_lane_index[0], source_lane_index[1], int(requested_target_lane_id)
                    )
                    target_lane = base_env.road.network.get_lane(target_lane_index)
                    # Apply the lane transition to the actual Highway-env vehicle
                    # before env.step(), not just to the wrapper's bookkeeping.
                    # Preserve longitudinal position and speed; lateral position
                    # and heading are snapped to the target lane centre/tangent.
                    target_position = np.asarray(
                        target_lane.position(longitudinal, 0.0), dtype=float
                    )
                    target_heading = float(target_lane.heading_at(longitudinal))
                    ego.position = target_position
                    ego.heading = target_heading
                    ego.lane_index = tuple(target_lane_index)

                    # ControlledVehicle implementations may also maintain an
                    # internal target_lane_index. Synchronize it when present so
                    # the next native vehicle update cannot follow a stale target.
                    if hasattr(ego, "target_lane_index"):
                        ego.target_lane_index = tuple(target_lane_index)

                    self.target_lane_index = tuple(target_lane_index)
                    self.target_lane = target_lane
                    self._lane_change_active = False
                    self._lane_change_source_index = None
                    self._lane_change_started_step = None
                    self._lane_change_stable_steps = 0
                    self._lane_change_cooldown = 0
                    self._active_lane_maneuver_id = None
                    self._active_lane_request_trace_id = None
                    lane_changed = True
                    stats = getattr(self, "_lane_change_stats", None)
                    if stats is not None:
                        stats["completed"] = int(stats.get("completed", 0)) + 1
                        direction = "LEFT" if requested_target_lane_id > source_lane_id else "RIGHT"
                        stats["completed_" + direction] = int(stats.get("completed_" + direction, 0)) + 1
                    write_diagnostic_event(
                        "instant_lane_change", self,
                        source_lane=source_lane_id,
                        target_lane=int(requested_target_lane_id),
                        direction="LEFT" if requested_target_lane_id > source_lane_id else "RIGHT",
                        outcome="COMPLETED_INSTANTLY",
                        longitudinal_position=float(longitudinal),
                        request_id=safety_info.get("lane_request_trace_id"),
                    )
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    lane_action_id = 0
                    lane_name = "KEEP"
                    action[1] = 0.0
                    lane_rejection_reason = "instant_target_lane_error:" + str(exc)
                    safety_info["lane_change_rejection_reason"] = lane_rejection_reason
                    safety_info["lane_change_blocked"] = True
                    print("[INSTANT LANE CHANGE ERROR]", lane_rejection_reason)
            else:
                lane_action_id = 0
                lane_name = "KEEP"
                action[1] = 0.0
        elif lane_action_id in (1, 2):
            # A lane-change action without a selected safe target fails closed.
            lane_action_id = 0
            lane_name = "KEEP"
            action[1] = 0.0
            lane_rejection_reason = "change_action_missing_safe_target"
            safety_info["lane_change_rejection_reason"] = lane_rejection_reason
            safety_info["lane_change_blocked"] = True

        # Do not replace the policy's acceleration with a hand-designed
        # speed-match/emergency-braking action. The simulator's vehicle dynamics
        # and action bounds determine the resulting motion.
        acceleration = float(np.clip(acceleration, -1.0, 1.0))
        speed_before = float(ego.speed)
        lane_before_env_update = int(ego.lane_index[2])
        position_before_env_update = np.asarray(ego.position, dtype=np.float64).copy()
        policy_dt = 1.0 / float(base_env.config["policy_frequency"])
        physical_action = np.asarray([acceleration, 0.0], dtype=np.float32)

        # Sanitize the complete background population before the native update,
        # then again immediately after it. The post-update pass is essential
        # because highway-env may integrate a traffic vehicle into a negative speed during
        # this very update; slot maintenance and safety evaluation must never see
        # that invalid value.
        self._sanitize_all_traffic_speeds("pre_env_step")
        traffic_speeds_before = self._snapshot_managed_traffic_slots()
        # Preserve the exact collision-proximate scene before native physics can
        # alter/remove a vehicle and make its TTC appear to reset to infinity.
        edm_pre_env_step_metrics = self._min_ttc_by_lane()
        edm_pre_env_step = self._edm_interpretation_by_lane(edm_pre_env_step_metrics)
        vehicles_pre_env_step = snapshot_simulator_vehicles(self)
        crashed_before_step = bool(getattr(ego, "crashed", False))
        obs, env_reward, native_terminated, truncated, env_info = self.env.step(physical_action)
        # Preserve the post-physics collision scene before sanitization and
        # traffic maintenance can remove the crashed background vehicle.
        post_native_collision_snapshot = (
            snapshot_simulator_vehicles(self)
            if bool(getattr(base_env.vehicle, "crashed", False))
            else None
        )
        # This check must run BEFORE post-step sanitization: it records the first
        # negative value and the vehicle's action/kinematics before any clamp or
        # slot replacement can obscure the cause.
        traffic_speeds_after = self._snapshot_managed_traffic_slots()
        self._record_first_negative_speed_transition(
            traffic_speeds_before, traffic_speeds_after, physical_action, env_info
        )
        self._sanitize_all_traffic_speeds("post_env_step")
        ego_after = base_env.vehicle
        speed_after = float(ego_after.speed)
        lane_after_env_update = int(ego_after.lane_index[2])
        position_after_env_update = np.asarray(ego_after.position, dtype=np.float64).copy()

        # Collision is an episode-terminal event in this experiment. Count the
        # onset once, preserve the native crashed state, and stop immediately;
        # there are no post-collision overlap transitions in the RL trajectory.
        raw_collision_contact = bool(getattr(ego_after, "crashed", False))
        collision_contact_was_active = bool(
            getattr(self, "_collision_contact_active", False)
        )
        collision = bool(raw_collision_contact and not collision_contact_was_active)
        collision_contact_continuing = bool(
            raw_collision_contact and collision_contact_was_active
        )
        self._collision_contact_active = raw_collision_contact
        ego_crashed_after_step = raw_collision_contact
        # Also terminate on the observed crash flag in case a particular
        # highway-env version does not propagate it through native_terminated.
        effective_terminated = bool(native_terminated or raw_collision_contact)

        if lane_changed and lane_after_env_update != int(requested_target_lane_id):
            print(
                "[INSTANT LANE STATE WARNING]",
                f"expected_target_lane={int(requested_target_lane_id)}",
                f"lane_after_env_update={lane_after_env_update}",
                f"position_before_env_update={position_before_env_update.tolist()}",
                f"position_after_env_update={position_after_env_update.tolist()}",
                "reason=underlying_env_did_not_preserve_instantaneous_lane_transition",
            )
            write_diagnostic_event(
                "instant_lane_state_mismatch", self,
                source_lane=int(source_lane_id),
                requested_target_lane=int(requested_target_lane_id),
                lane_before_env_update=int(lane_before_env_update),
                lane_after_env_update=int(lane_after_env_update),
                position_before_env_update=position_before_env_update.tolist(),
                position_after_env_update=position_after_env_update.tolist(),
                speed_before_env_update=float(speed_before),
                speed_after_env_update=float(speed_after),
            )

        if collision:
            print("[COLLISION EGO][INSTANT MODE]",
                  f"position={np.asarray(ego_after.position, dtype=float).tolist()}",
                  f"lane={int(ego_after.lane_index[2])}",
                  f"speed={speed_after:.3f}")

        self.step_count += 1
        self._maintain_continuous_traffic()
        obs = base_env.observation_type.observe()
        next_state = self._encode_state(obs)
        self._assert_continuous_traffic_invariant("instant_step")

        # Shared TTC risk is logged after the simulator update and traffic-slot
        # maintenance. It does not intervene in the action or enter the reward.
        ttc_risk_snapshot = self._min_ttc_by_lane()
        ttc_risk_diagnostics = {}
        for lane_id, lane_values in sorted(ttc_risk_snapshot.items()):
            ttc_risk_diagnostics[int(lane_id)] = {
                "front_ttc_s": float(lane_values.get("front_ttc", float("inf"))),
                "rear_ttc_s": float(lane_values.get("rear_ttc", float("inf"))),
                "front_risk": float(lane_values.get("front_risk", 0.0)),
                "rear_risk": float(lane_values.get("rear_risk", 0.0)),
                "lane_risk": float(lane_values.get("lane_risk", 0.0)),
            }
        edm_post_step = self._edm_interpretation_by_lane(ttc_risk_snapshot)
        current_lane_risk = ttc_risk_diagnostics.get(int(ego_after.lane_index[2]), {})
        lane_risk_summary = " | ".join(
            f"L{lane_id}:F={v['front_ttc_s']:.2f}s/{v['front_risk']:.2f},"
            f"R={v['rear_ttc_s']:.2f}s/{v['rear_risk']:.2f},"
            f"MAX={v['lane_risk']:.2f}"
            for lane_id, v in ttc_risk_diagnostics.items()
        )
        print(
            "[TTC RISK]",
            f"step={self.step_count}",
            f"current_lane={int(ego_after.lane_index[2])}",
            f"current_lane_max_risk={float(current_lane_risk.get('lane_risk', 0.0)):.3f}",
            f"lanes=({lane_risk_summary})",
        )

        info = dict(env_info or {})
        info.update({
            "safety_violation": collision,
            "collision": collision,
            # "terminated" is the effective RL terminal flag. Native collision
            # termination is retained separately for auditability.
            "terminated": bool(effective_terminated),
            "native_terminated": bool(native_terminated),
            "collision_event": bool(collision),
            "collision_event_onset": bool(collision),
            "collision_contact_active": bool(raw_collision_contact),
            "collision_contact_continuing": bool(collision_contact_continuing),
            "collision_contact_was_active": bool(collision_contact_was_active),
            "ego_crashed_after_step": bool(ego_crashed_after_step),
            "crashed_before_step": bool(crashed_before_step),
            "collision_continuation_enabled": False,
            "collision_terminates_episode": bool(raw_collision_contact),
            # Present only on collision transitions. Captured immediately after
            # native physics and before collided traffic is removed/recycled.
            "collision_vehicle_snapshot_after_native_step": post_native_collision_snapshot,
            "truncated": bool(truncated),
            "env_reward": float(env_reward),
            "lane_action_id": lane_action_id,
            "lane_action_index": lane_action_id,
            "lane_action_delta": (int(requested_target_lane_id - source_lane_id) if lane_changed else 0),
            "lane_action_name": lane_name,
            "current_lane_id": source_lane_id,
            "requested_target_lane_id": requested_target_lane_id,
            "accepted_target_lane_id": int(ego_after.lane_index[2]),
            "lane_before_env_update": int(lane_before_env_update),
            "lane_after_env_update": int(lane_after_env_update),
            "position_before_env_update": position_before_env_update.copy(),
            "position_after_env_update": position_after_env_update.copy(),
            "lane_change_completed_instantly": bool(lane_changed),
            "one_env_update_per_policy_action": bool(SIMULATION_SUBSTEPS_PER_POLICY == 1),
            "simulator_updates_per_policy_action": int(SIMULATION_SUBSTEPS_PER_POLICY),
            "simulation_step_duration_s": float(SIMULATION_STEP_DURATION_S),
            "policy_step_duration_s": float(POLICY_STEP_DURATION_S),
            "lane_change_rejection_reason": lane_rejection_reason,
            "sampled_policy_action": (np.asarray(sampled_action, dtype=np.float32).copy()
                                      if sampled_action is not None else incoming_action.copy()),
            "policy_action": np.asarray([acceleration, lane_action_id], dtype=np.float32),
            "executed_policy_action": np.asarray([acceleration, lane_action_id], dtype=np.float32),
            "executed_action": physical_action.copy(),
            "executed_acceleration": float(acceleration),
            "executed_steering": 0.0,
            "policy_action_context": safety_info,
            "instant_lane_change_mode": True,
            "observed_acceleration": (speed_after - speed_before) / policy_dt,
            "ttc_risk_diagnostics": ttc_risk_diagnostics,
            "ttc_risk_aggregation": "max_front_rear",
            "ttc_risk_thresholds_s": {
                "comfortable": float(EDM_TTC_COMFORTABLE_S),
                "danger": float(EDM_TTC_DANGER_S),
                "critical": float(EDM_TTC_CRITICAL_S),
            },
            "edm_pre_action_by_lane": edm_pre_action,
            "edm_pre_env_step_by_lane": edm_pre_env_step,
            "edm_post_step_by_lane": edm_post_step,
            "edm_thresholds_s": {
                "comfortable_min": float(EDM_TTC_COMFORTABLE_S),
                "danger": float(EDM_TTC_DANGER_S),
                "critical": float(EDM_TTC_CRITICAL_S),
            },
            "edm_interpretation_only": True,
            "edm_action_intervention": False,
            "lane_before_action": int(lane_before_action),
            "vehicles_pre_env_step": vehicles_pre_env_step,
            "collision_snapshot_source": "pre_env_step_before_collision_processing",
        })
        self.state = next_state
        return next_state.copy(), float(env_reward), bool(effective_terminated or truncated), info

    def get_lane_information(self):
        base_env = self.env.unwrapped
        ego = base_env.vehicle
        road_network = base_env.road.network
        lane_index = road_network.get_closest_lane_index(
            ego.position,
            ego.heading,
        )
        lane = road_network.get_lane(lane_index)
        _, lateral_position = lane.local_coordinates(ego.position)
        return {
            "on_road": bool(ego.on_road),
            "lane_deviation": abs(float(lateral_position)),
            "lane_index": lane_index,
        }



    def close(self):
        self.env.close()






# ---------------------------------------------------------------------------
# ADT background traffic ported from U_Challenge.py.
# The existing P.py EDM encoder remains the state representation.
# ---------------------------------------------------------------------------

class ContinuousTrafficVehicle(Vehicle):
    """Background vehicle with an ADT profile and bounded longitudinal control."""

    def act(self):
        manager = getattr(self, "_arood_traffic_manager", None)
        if manager is None:
            acceleration = 0.0
        else:
            acceleration = manager.adt_vehicle_acceleration(self)

        # Individual ordinary acceleration varies slightly by vehicle, but no
        # background vehicle can exceed the agreed +3.0 m/s^2 cap. The lower
        # emergency bound is used only when the controller identifies an
        # imminent closing conflict or a rare unstable-driver event.
        max_acceleration = IDM_MAX_ACCEL_MPS2
        self.action = {
            "steering": 0.0,
            "acceleration": float(np.clip(
                acceleration, -ADT_EMERGENCY_BRAKING_MPS2, max_acceleration
            )),
        }


# ---------------------------------------------------------------------------
# 3. Continuous traffic population and boundary management
# ---------------------------------------------------------------------------

_ORIGINAL_ROAD_LONGITUDINAL_POSITION = HighwaySimulator._road_longitudinal_position
_ORIGINAL_STATE_ENCODER = HighwaySimulator._encode_state


def _road_s(self, vehicle):
    return float(_ORIGINAL_ROAD_LONGITUDINAL_POSITION(self, vehicle))


def _vehicle_length(vehicle):
    return float(getattr(vehicle, "LENGTH", getattr(vehicle, "length", 5.0)))


def _traffic_in_lane(self, lane_id, exclude=None):
    road = self.env.unwrapped.road
    result = []
    for vehicle in list(road.vehicles):
        if vehicle is exclude:
            continue
        try:
            if int(vehicle.lane_index[2]) == int(lane_id):
                result.append(vehicle)
        except (AttributeError, TypeError, IndexError, ValueError):
            continue
    return result


def _idm_acceleration(self, vehicle):
    """IDM acceleration using the nearest preceding vehicle in the same lane."""
    ego_s = _road_s(self, vehicle)
    lane_id = int(vehicle.lane_index[2])
    speed = max(0.0, float(vehicle.speed))
    desired_speed = float(getattr(vehicle, "_arood_desired_speed", SPEED_LIMIT_MPS))
    desired_speed = max(TRAFFIC_MIN_SPEED_MPS, min(desired_speed, TRAFFIC_MAX_SPEED_MPS))

    front = None
    front_gap = float("inf")
    for other in _traffic_in_lane(self, lane_id, exclude=vehicle):
        gap_center = _road_s(self, other) - ego_s
        if gap_center <= 0.0:
            continue
        bumper_gap = gap_center - 0.5 * (_vehicle_length(vehicle) + _vehicle_length(other))
        if bumper_gap < front_gap:
            front_gap = bumper_gap
            front = other

    free_road_term = (speed / max(desired_speed, 1e-3)) ** IDM_ACCEL_EXPONENT
    if front is None or not np.isfinite(front_gap):
        return IDM_MAX_ACCEL_MPS2 * (1.0 - free_road_term)

    gap = max(front_gap, 0.5)
    front_speed = max(0.0, float(front.speed))
    closing_speed = speed - front_speed
    max_accel = IDM_MAX_ACCEL_MPS2
    comfortable_braking = IDM_COMFORTABLE_BRAKING_MPS2
    root = math.sqrt(max_accel * comfortable_braking)
    dynamic_gap = (
        speed * IDM_TIME_HEADWAY_S
        + speed * closing_speed / max(2.0 * root, 1e-3)
    )
    desired_gap = IDM_MIN_GAP_M + max(0.0, dynamic_gap)
    interaction_term = (desired_gap / gap) ** 2
    acceleration = max_accel * (
        1.0 - free_road_term - interaction_term
    )
    return float(np.clip(
        acceleration, -comfortable_braking, max_accel
    ))


def _spawn_traffic_vehicle(self, lane_id, longitudinal, speed):
    road = self.env.unwrapped.road
    lane_index = self._lane_index_for_id(int(lane_id))
    speed = float(np.clip(
        speed, TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS
    ))
    vehicle = ContinuousTrafficVehicle.make_on_lane(
        road, lane_index, longitudinal=float(longitudinal), speed=speed
    )
    vehicle._arood_traffic_manager = self
    profile_names = [item[0] for item in ADT_PROFILE_MIX]
    profile_weights = [item[1] for item in ADT_PROFILE_MIX]
    vehicle._arood_adt_profile = random.choices(
        profile_names, weights=profile_weights, k=1
    )[0]
    profile = vehicle._arood_adt_profile
    headway = PROFILE_IDM_HEADWAY_S[profile]
    min_gap = PROFILE_IDM_MIN_GAP_M[profile]
    desired_low, desired_high = PROFILE_DESIRED_SPEED_RANGES_MPS[profile]
    vehicle._arood_idm_headway_s = headway
    vehicle._arood_idm_min_gap_m = min_gap
    sampled_desired_speed = random.uniform(desired_low, desired_high)
    sampled_desired_speed += random.uniform(
        -ADT_SPEED_VARIATION_MPS, ADT_SPEED_VARIATION_MPS
    )
    vehicle._arood_desired_speed = float(np.clip(
        sampled_desired_speed, TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS
    ))
    # Avoid spawning a vehicle with a large artificial speed mismatch relative
    # to its own desired speed. The simulator integration guard also enforces
    # TRAFFIC_MIN_SPEED_MPS as a hard floor while the vehicle is not crashed.
    initial_speed_low = max(
        TRAFFIC_MIN_SPEED_MPS, vehicle._arood_desired_speed - 4.0
    )
    initial_speed_high = min(
        TRAFFIC_MAX_SPEED_MPS, vehicle._arood_desired_speed + 4.0
    )
    vehicle.speed = float(np.clip(speed, initial_speed_low, initial_speed_high))
    vehicle._arood_lane_change_cooldown = 0
    vehicle._arood_brake_steps_remaining = 0
    vehicle._arood_traffic_uid = f"traffic-{time.time_ns()}-{random.randrange(1_000_000)}"
    vehicle.MAX_SPEED = TRAFFIC_MAX_SPEED_MPS
    road.vehicles.append(vehicle)
    return vehicle


def _safe_spawn_position(self, lane_id, longitudinal, minimum_gap):
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    # Keep an insertion buffer around ego in every lane, not only the
    # currently occupied lane, so an adjacent-lane vehicle cannot begin
    # immediately alongside ego and create an artificial lane-change hazard.
    if abs(longitudinal - ego_s) < minimum_gap:
        return False
    for other in _traffic_in_lane(self, lane_id):
        gap = abs(_road_s(self, other) - longitudinal)
        if gap < minimum_gap:
            return False
    return True


def _initialize_continuous_traffic(self):
    """Initialize a randomized population whose episode target never exceeds 12."""
    env = self.env.unwrapped
    ego = env.vehicle
    env.road.vehicles = [ego]
    ego_s = _road_s(self, ego)

    self._traffic_population_target = random.randint(
        MIN_INITIAL_SURROUNDING_TRAFFIC_VEHICLES,
        MAX_SURROUNDING_TRAFFIC_VEHICLES,
    )
    spawned = 0
    attempts = 0
    longitudinal_min = -TRAFFIC_REAR_RECYCLE_BOUNDARY_M
    longitudinal_max = TRAFFIC_FRONT_RECYCLE_BOUNDARY_M
    while (
        spawned < self._traffic_population_target
        and attempts < INITIAL_SPAWN_MAX_ATTEMPTS
        and spawned < MAX_SURROUNDING_TRAFFIC_VEHICLES
    ):
        attempts += 1
        lane_id = random.randrange(NUM_LANES)
        longitudinal = ego_s + random.uniform(longitudinal_min, longitudinal_max)
        if not _safe_spawn_position(self, lane_id, longitudinal, MIN_INITIAL_GAP_M):
            continue
        speed = random.uniform(TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS)
        _spawn_traffic_vehicle(self, lane_id, longitudinal, speed)
        spawned += 1

    if spawned < self._traffic_population_target:
        print(
            "[TRAFFIC INITIALIZATION NOTE]",
            f"requested_target={self._traffic_population_target}",
            f"actual_population={spawned}",
            "reason=spawn_attempt_limit_or_spacing",
        )

    self._traffic_spawn_count = spawned
    self._traffic_removal_count = 0
    self._traffic_pending_replacements_by_side = {"front": 0, "rear": 0}
    self._adt_background_lane_change_count = 0
    self._adt_slot_replacement_count = 0
    self._adt_slot_observation_count = 0
    self._adt_last_slot_uids = None
    self._adt_profile_counts = {
        name: sum(getattr(v, "_arood_adt_profile", "normal") == name
                  for v in env.road.vehicles if v is not ego)
        for name, _ in ADT_PROFILE_MIX
    }
    print(
        "[ADT TRAFFIC INITIALIZED]",
        f"vehicles={spawned}",
        f"episode_target={self._traffic_population_target}",
        f"hard_max={MAX_SURROUNDING_TRAFFIC_VEHICLES}",
        f"profiles={self._adt_profile_counts}",
        f"reference=+{TRAFFIC_REFERENCE_FRONT_M:.0f}/-{TRAFFIC_REFERENCE_REAR_M:.0f}m",
        f"recycling=+{TRAFFIC_FRONT_RECYCLE_BOUNDARY_M:.0f}/-{TRAFFIC_REAR_RECYCLE_BOUNDARY_M:.0f}m",
        "six_nearest_slots=observation_only",
    )



def _adt_vehicle_acceleration(self, vehicle):
    """Profile-conditioned IDM with a restart rule and rare emergency braking."""
    profile = getattr(vehicle, "_arood_adt_profile", "normal")
    speed = max(0.0, float(vehicle.speed))
    desired_speed = float(getattr(vehicle, "_arood_desired_speed", SPEED_LIMIT_MPS))
    desired_speed = float(np.clip(
        desired_speed, TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS
    ))
    lane_id = int(vehicle.lane_index[2])
    vehicle_s = _road_s(self, vehicle)
    front = None
    front_gap = float("inf")
    for other in _traffic_in_lane(self, lane_id, exclude=vehicle):
        delta = _road_s(self, other) - vehicle_s
        if delta <= 0.0:
            continue
        gap = delta - 0.5 * (_vehicle_length(vehicle) + _vehicle_length(other))
        if gap < front_gap:
            front, front_gap = other, gap

    max_accel = IDM_MAX_ACCEL_MPS2
    comfortable_braking = IDM_COMFORTABLE_BRAKING_MPS2
    headway = float(getattr(vehicle, "_arood_idm_headway_s", IDM_TIME_HEADWAY_S))
    min_gap = float(getattr(vehicle, "_arood_idm_min_gap_m", IDM_MIN_GAP_M))
    free_term = (speed / max(desired_speed, 1e-3)) ** IDM_ACCEL_EXPONENT
    closing_speed = 0.0
    emergency_braking = False

    if front is None or not np.isfinite(front_gap):
        acceleration = max_accel * (1.0 - free_term)
    else:
        gap = max(front_gap, 0.5)
        front_speed = max(0.0, float(front.speed))
        closing_speed = speed - front_speed
        root = math.sqrt(max_accel * comfortable_braking)
        dynamic_gap = (
            speed * headway
            + speed * closing_speed / max(2.0 * root, 1e-3)
        )
        desired_gap = min_gap + max(0.0, dynamic_gap)
        acceleration = max_accel * (
            1.0 - free_term - (desired_gap / gap) ** 2
        )

        closing_ttc = (
            gap / closing_speed if closing_speed > 1e-6 else float("inf")
        )
        emergency_braking = bool(
            closing_speed > 0.5
            and (
                closing_ttc < 1.5
                or gap < max(3.5, 0.75 * min_gap)
            )
        )

        # Restart a stopped vehicle once its leader has cleared enough space.
        if speed < 0.5 and gap > max(12.0, min_gap + 8.0):
            acceleration = max(acceleration, min(1.5, max_accel))

    dt = 1.0 / max(1.0, float(simulation_frequency))
    if profile == "unstable":
        remaining = int(getattr(vehicle, "_arood_brake_steps_remaining", 0))
        if remaining > 0:
            vehicle._arood_brake_steps_remaining = remaining - 1
            acceleration = min(acceleration, ADT_SUDDEN_BRAKE_ACCEL_MPS2)
            emergency_braking = True
        elif random.random() < ADT_SUDDEN_BRAKE_EVENT_RATE_PER_S * dt:
            vehicle._arood_brake_steps_remaining = max(
                1, int(round(ADT_SUDDEN_BRAKE_DURATION_S / dt))
            )
            acceleration = min(acceleration, ADT_SUDDEN_BRAKE_ACCEL_MPS2)
            emergency_braking = True
        elif random.random() < ADT_UNSTABLE_ACCEL_EVENT_RATE_PER_S * dt:
            acceleration = max(acceleration, ADT_UNSTABLE_ACCEL_PULSE_MPS2)

    self.adt_try_lane_change(vehicle)
    vehicle._arood_emergency_brake_active = bool(emergency_braking)
    lower_bound = (
        -ADT_EMERGENCY_BRAKING_MPS2
        if emergency_braking else -comfortable_braking
    )
    return float(np.clip(acceleration, lower_bound, max_accel))


def _adt_try_lane_change(self, vehicle):
    """Occasional independent lane changes with explicit collision-avoidance gaps."""
    profile = getattr(vehicle, "_arood_adt_profile", "normal")
    cooldown = int(getattr(vehicle, "_arood_lane_change_cooldown", 0))
    if cooldown > 0:
        vehicle._arood_lane_change_cooldown = cooldown - 1
        return False

    current_lane = int(vehicle.lane_index[2])
    current_s = _road_s(self, vehicle)
    current_front = None
    current_front_gap = float("inf")
    for other in _traffic_in_lane(self, current_lane, exclude=vehicle):
        delta = _road_s(self, other) - current_s
        if delta <= 0.0:
            continue
        gap = delta - 0.5 * (_vehicle_length(vehicle) + _vehicle_length(other))
        if gap < current_front_gap:
            current_front, current_front_gap = other, gap

    dt = 1.0 / max(1.0, float(simulation_frequency))
    rate = PROFILE_LANE_CHANGE_RATE_PER_S.get(profile, 0.0)
    should_consider = (
        current_front is not None
        and current_front_gap < ADT_LANE_CHANGE_TRIGGER_GAP_M
        and float(current_front.speed) + ADT_LANE_CHANGE_IMPROVEMENT_MPS
            < float(vehicle.speed)
    )
    if not should_consider or random.random() >= rate * dt:
        return False

    start_node, end_node, _ = vehicle.lane_index
    candidates = []
    for target_lane in (current_lane - 1, current_lane + 1):
        if target_lane not in range(NUM_LANES):
            continue
        front, rear = None, None
        front_gap, rear_gap = float("inf"), float("inf")
        for other in _traffic_in_lane(self, target_lane, exclude=vehicle):
            delta = _road_s(self, other) - current_s
            bumper_gap = abs(delta) - 0.5 * (
                _vehicle_length(vehicle) + _vehicle_length(other)
            )
            if delta > 0.0 and bumper_gap < front_gap:
                front, front_gap = other, bumper_gap
            elif delta < 0.0 and bumper_gap < rear_gap:
                rear, rear_gap = other, bumper_gap

        front_ttc = float("inf")
        rear_ttc = float("inf")
        if front is not None and float(vehicle.speed) > float(front.speed):
            front_ttc = max(front_gap, 0.0) / max(
                float(vehicle.speed) - float(front.speed), 1e-6
            )
        if rear is not None and float(rear.speed) > float(vehicle.speed):
            rear_ttc = max(rear_gap, 0.0) / max(
                float(rear.speed) - float(vehicle.speed), 1e-6
            )
        if front_gap < ADT_LANE_CHANGE_FRONT_GAP_M:
            continue
        if rear_gap < ADT_LANE_CHANGE_REAR_GAP_M:
            continue
        if front_ttc < ADT_LANE_CHANGE_FRONT_TTC_S:
            continue
        if rear_ttc < ADT_LANE_CHANGE_REAR_TTC_S:
            continue
        front_speed = float(front.speed) if front is not None else SPEED_LIMIT_MPS
        candidates.append((front_speed, target_lane))
    if not candidates:
        return False

    candidates.sort(reverse=True)
    best_front_speed, target_lane = candidates[0]
    if best_front_speed < float(current_front.speed) + ADT_LANE_CHANGE_IMPROVEMENT_MPS:
        return False

    # Apply a discrete background-traffic lane transition. The target lane's
    # own geometry is used, preserving the road coordinate and heading.
    target_lane_index = (start_node, end_node, int(target_lane))
    lane = self.env.unwrapped.road.network.get_lane(target_lane_index)
    vehicle.lane_index = target_lane_index
    vehicle.position = np.asarray(lane.position(current_s, 0.0), dtype=float)
    vehicle.heading = float(lane.heading_at(current_s))
    vehicle._arood_lane_change_cooldown = int(round(
        ADT_LANE_CHANGE_COOLDOWN_S * simulation_frequency
    ))
    self._adt_background_lane_change_count = int(
        getattr(self, "_adt_background_lane_change_count", 0)
    ) + 1
    return True


def _adt_slot_diagnostics(self, slots):
    """Count identity changes in the six slots, considering only the reference window."""
    current = {}
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    for lane_id in range(NUM_LANES):
        for side in ("front", "rear"):
            candidates = []
            for vehicle in list(self.env.unwrapped.road.vehicles):
                if vehicle is ego or int(vehicle.lane_index[2]) != lane_id:
                    continue
                delta = _road_s(self, vehicle) - ego_s
                if delta > TRAFFIC_REFERENCE_FRONT_M or delta < -TRAFFIC_REFERENCE_REAR_M:
                    continue
                if (side == "front" and delta > 0.0) or (side == "rear" and delta < 0.0):
                    candidates.append((abs(delta), getattr(
                        vehicle, "_arood_traffic_uid", str(id(vehicle))
                    )))
            candidates.sort(key=lambda item: item[0])
            current[f"lane{lane_id}_{side}"] = candidates[0][1] if candidates else None

    previous = getattr(self, "_adt_last_slot_uids", None)
    if previous is not None:
        self._adt_slot_replacement_count += sum(
            previous.get(key) != current.get(key) for key in current
        )
    self._adt_slot_observation_count += 1
    self._adt_last_slot_uids = current


def _spawn_boundary_replacement(self, side):
    """Spawn a replacement in the appropriate outer entry buffer, never in-view."""
    if side not in ("front", "rear"):
        raise ValueError(f"Unknown traffic entry side: {side}")
    ego = self.env.unwrapped.vehicle
    road = self.env.unwrapped.road
    current_population = sum(vehicle is not ego for vehicle in road.vehicles)
    episode_target = int(getattr(
        self, "_traffic_population_target", MAX_SURROUNDING_TRAFFIC_VEHICLES
    ))
    if current_population >= min(episode_target, MAX_SURROUNDING_TRAFFIC_VEHICLES):
        self._traffic_pending_replacements_by_side[side] = 0
        return False

    ego_s = _road_s(self, ego)
    ego_speed = max(0.0, float(ego.speed))

    if side == "front":
        speed = max(
            TRAFFIC_MIN_SPEED_MPS,
            ego_speed - random.uniform(2.0, 8.0),
        )
        longitudinal_offset = random.uniform(
            TRAFFIC_FRONT_SPAWN_MIN_M, TRAFFIC_FRONT_SPAWN_MAX_M
        )
    else:
        speed = min(
            TRAFFIC_MAX_SPEED_MPS,
            ego_speed + random.uniform(2.0, 6.0),
        )
        longitudinal_offset = -random.uniform(
            TRAFFIC_REAR_SPAWN_MIN_M, TRAFFIC_REAR_SPAWN_MAX_M
        )

    for _ in range(BOUNDARY_SPAWN_MAX_ATTEMPTS):
        lane_id = random.randrange(NUM_LANES)
        longitudinal = ego_s + longitudinal_offset
        if not _safe_spawn_position(self, lane_id, longitudinal, MIN_INSERTION_GAP_M):
            # Re-sample the location on the next attempt, retaining the same side.
            if side == "front":
                longitudinal_offset = random.uniform(
                    TRAFFIC_FRONT_SPAWN_MIN_M, TRAFFIC_FRONT_SPAWN_MAX_M
                )
            else:
                longitudinal_offset = -random.uniform(
                    TRAFFIC_REAR_SPAWN_MIN_M, TRAFFIC_REAR_SPAWN_MAX_M
                )
            continue
        _spawn_traffic_vehicle(self, lane_id, longitudinal, speed)
        self._traffic_spawn_count += 1
        self._traffic_pending_replacements_by_side[side] = max(
            0, self._traffic_pending_replacements_by_side[side] - 1
        )
        return True
    return False


def _maintain_continuous_traffic(self):
    """Retain traffic through the asymmetric reference window and entry buffers."""
    road = self.env.unwrapped.road
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    kept = [ego]
    removed = 0

    for vehicle in list(road.vehicles):
        if vehicle is ego:
            continue
        # A collided background vehicle is removed rather than forced to
        # maintain the speed floor or recycled into the traffic stream.
        if bool(getattr(vehicle, "crashed", False)):
            removed += 1
            print(
                "[ADT COLLISION REMOVAL]",
                f"traffic_uid={getattr(vehicle, '_arood_traffic_uid', str(id(vehicle)))}",
                f"lane={getattr(vehicle, 'lane_index', None)}",
                f"speed_mps={float(getattr(vehicle, 'speed', 0.0)):.3f}",
                "replacement=none",
            )
            continue

        try:
            s = _road_s(self, vehicle)
            speed = float(vehicle.speed)
            lane_id = int(vehicle.lane_index[2])
        except (AttributeError, TypeError, ValueError, IndexError):
            removed += 1
            continue

        if not np.isfinite(speed) or speed < TRAFFIC_MIN_SPEED_MPS:
            vehicle.speed = TRAFFIC_MIN_SPEED_MPS
        if lane_id not in range(NUM_LANES):
            removed += 1
            continue

        relative_s = s - ego_s
        if relative_s > TRAFFIC_FRONT_RECYCLE_BOUNDARY_M:
            removed += 1
            self._traffic_pending_replacements_by_side["front"] += 1
            continue
        if relative_s < -TRAFFIC_REAR_RECYCLE_BOUNDARY_M:
            removed += 1
            self._traffic_pending_replacements_by_side["rear"] += 1
            continue
        kept.append(vehicle)

    road.vehicles = kept
    self._traffic_removal_count += removed

    # Refill only the same side from which a vehicle left. Spawn positions are
    # outside the observation window, so new vehicles enter by natural motion.
    target_population = min(
        int(getattr(self, "_traffic_population_target", MAX_SURROUNDING_TRAFFIC_VEHICLES)),
        MAX_SURROUNDING_TRAFFIC_VEHICLES,
    )
    for side in ("front", "rear"):
        while (
            self._traffic_pending_replacements_by_side[side] > 0
            and (len(road.vehicles) - 1) < target_population
        ):
            if not _spawn_boundary_replacement(self, side):
                break

    return {
        "population": len(road.vehicles) - 1,
        "spawned_total": int(self._traffic_spawn_count),
        "removed_total": int(self._traffic_removal_count),
        "pending_replacements": int(sum(self._traffic_pending_replacements_by_side.values())),
        "pending_front_replacements": int(self._traffic_pending_replacements_by_side["front"]),
        "pending_rear_replacements": int(self._traffic_pending_replacements_by_side["rear"]),
    }


def _assert_continuous_traffic_invariant(self, where):
    road = self.env.unwrapped.road
    ego = self.env.unwrapped.vehicle
    vehicles = [v for v in road.vehicles if v is not ego]
    if len(vehicles) > MAX_SURROUNDING_TRAFFIC_VEHICLES:
        raise RuntimeError(
            f"Surrounding traffic cap exceeded at {where}: "
            f"{len(vehicles)} > {MAX_SURROUNDING_TRAFFIC_VEHICLES}."
        )
    if any(not np.isfinite(float(v.speed)) or float(v.speed) < 0.0 for v in vehicles):
        raise RuntimeError(f"Invalid background vehicle speed at {where}.")
    if any(
        not bool(getattr(v, "crashed", False))
        and float(v.speed) < TRAFFIC_MIN_SPEED_MPS - 1e-6
        for v in vehicles
    ):
        offenders = [
            {
                "uid": getattr(v, "_arood_traffic_uid", str(id(v))),
                "speed_mps": float(v.speed),
                "lane": int(v.lane_index[2]),
            }
            for v in vehicles
            if not bool(getattr(v, "crashed", False))
            and float(v.speed) < TRAFFIC_MIN_SPEED_MPS - 1e-6
        ]
        raise RuntimeError(
            f"Background traffic violated hard minimum speed at {where}: {offenders}"
        )
    if any(int(v.lane_index[2]) not in range(NUM_LANES) for v in vehicles):
        raise RuntimeError(f"Background vehicle has an invalid lane at {where}.")
    ego_s = _road_s(self, ego)
    for vehicle in vehicles:
        relative_s = _road_s(self, vehicle) - ego_s
        if relative_s > TRAFFIC_FRONT_RECYCLE_BOUNDARY_M + 1e-6:
            raise RuntimeError(f"Vehicle beyond front recycling boundary at {where}.")
        if relative_s < -TRAFFIC_REAR_RECYCLE_BOUNDARY_M - 1e-6:
            raise RuntimeError(f"Vehicle beyond rear recycling boundary at {where}.")
    return True


def _sanitize_all_traffic_speeds(self, where):
    for vehicle in list(self.env.unwrapped.road.vehicles):
        if vehicle is self.env.unwrapped.vehicle:
            continue
        speed = float(getattr(vehicle, "speed", 0.0))
        if not bool(getattr(vehicle, "crashed", False)) and (
            not np.isfinite(speed) or speed < TRAFFIC_MIN_SPEED_MPS
        ):
            # Defensive repair for externally altered/restored states. Normal
            # operation is constrained earlier at the Vehicle.step integrator.
            vehicle.speed = TRAFFIC_MIN_SPEED_MPS
        elif not np.isfinite(speed) or speed < 0.0:
            vehicle.speed = 0.0


def _snapshot_all_traffic(self):
    records = []
    ego = self.env.unwrapped.vehicle
    for vehicle in list(self.env.unwrapped.road.vehicles):
        if vehicle is ego:
            continue
        records.append({
            "uid": getattr(vehicle, "_arood_traffic_uid", str(id(vehicle))),
            "lane": int(vehicle.lane_index[2]),
            "s": _road_s(self, vehicle),
            "speed": float(vehicle.speed),
        })
    return records


def _no_negative_transition_record(self, *args, **kwargs):
    return None


# ---------------------------------------------------------------------------
# 4. Exact six-slot observation, independent of full traffic population
# ---------------------------------------------------------------------------

def _six_slot_state(self, obs=None):
    """Encode nearest front/rear vehicle in each lane, in a fixed lane order."""
    env = self.env.unwrapped
    ego = env.vehicle
    ego_s = _road_s(self, ego)
    ego_heading = float(ego.heading)
    ego_velocity = np.asarray([
        float(ego.speed) * math.cos(ego_heading),
        float(ego.speed) * math.sin(ego_heading),
    ])

    slots = {(lane, side): None for lane in range(NUM_LANES)
             for side in ("front", "rear")}
    nearest = {(lane, side): float("inf") for lane in range(NUM_LANES)
               for side in ("front", "rear")}

    for vehicle in list(env.road.vehicles):
        if vehicle is ego:
            continue
        try:
            lane_id = int(vehicle.lane_index[2])
            if lane_id not in range(NUM_LANES):
                continue
            signed_gap = _road_s(self, vehicle) - ego_s
        except (AttributeError, TypeError, ValueError, IndexError):
            continue

        # The buffer vehicles are physically simulated but are not yet visible
        # to the policy/EDM until they cross into the reference window.
        if (
            signed_gap > TRAFFIC_REFERENCE_FRONT_M
            or signed_gap < -TRAFFIC_REFERENCE_REAR_M
        ):
            continue
        side = "front" if signed_gap > 0.0 else "rear"
        distance = abs(signed_gap)
        if distance < nearest[(lane_id, side)]:
            nearest[(lane_id, side)] = distance
            slots[(lane_id, side)] = vehicle

    feature_ranges = {
        "x": (-1000.0, 1000.0),
        "y": (-12.0, 12.0),
        "vx": (-40.0, 40.0),
        "vy": (-10.0, 10.0),
        "cos_h": (-1.0, 1.0),
        "sin_h": (-1.0, 1.0),
    }

    def encode_vehicle(vehicle):
        if vehicle is None:
            return np.zeros(VEHICLE_FEATURES, dtype=np.float32)
        pos_delta = np.asarray(vehicle.position, dtype=float) - np.asarray(ego.position, dtype=float)
        c, s = math.cos(ego_heading), math.sin(ego_heading)
        local_x = c * pos_delta[0] + s * pos_delta[1]
        local_y = -s * pos_delta[0] + c * pos_delta[1]

        other_heading = float(vehicle.heading)
        other_velocity = np.asarray([
            float(vehicle.speed) * math.cos(other_heading),
            float(vehicle.speed) * math.sin(other_heading),
        ])
        relative_velocity = other_velocity - ego_velocity
        local_vx = c * relative_velocity[0] + s * relative_velocity[1]
        local_vy = -s * relative_velocity[0] + c * relative_velocity[1]

        raw = {
            "presence": 1.0,
            "x": local_x,
            "y": local_y,
            "vx": local_vx,
            "vy": local_vy,
            "cos_h": math.cos(other_heading - ego_heading),
            "sin_h": math.sin(other_heading - ego_heading),
        }
        encoded = [1.0]
        for key in ("x", "y", "vx", "vy", "cos_h", "sin_h"):
            lo, hi = feature_ranges[key]
            normalized = 2.0 * (raw[key] - lo) / (hi - lo) - 1.0
            encoded.append(float(np.clip(normalized, -1.0, 1.0)))
        return np.asarray(encoded, dtype=np.float32)

    rows = [encode_vehicle(ego)]
    for lane_id in range(NUM_LANES):
        rows.append(encode_vehicle(slots[(lane_id, "front")]))
        rows.append(encode_vehicle(slots[(lane_id, "rear")]))

    _adt_slot_diagnostics(self, slots)

    # The original encoder appends context and per-lane safety features.
    return _ORIGINAL_STATE_ENCODER(
        self, np.asarray(rows, dtype=np.float32)
    )


# ---------------------------------------------------------------------------


# Bind ADT traffic management to P.py's existing HighwaySimulator.
HighwaySimulator._lane_index_for_id = lambda self, lane_id: (
    self.env.unwrapped.vehicle.lane_index[0],
    self.env.unwrapped.vehicle.lane_index[1],
    int(lane_id),
)
HighwaySimulator._road_longitudinal_position = _road_s
HighwaySimulator.idm_acceleration = _idm_acceleration
HighwaySimulator.adt_vehicle_acceleration = _adt_vehicle_acceleration
HighwaySimulator.adt_try_lane_change = _adt_try_lane_change
HighwaySimulator._spawn_traffic_vehicle = _spawn_traffic_vehicle
HighwaySimulator._initialize_continuous_traffic = _initialize_continuous_traffic
HighwaySimulator._maintain_continuous_traffic = _maintain_continuous_traffic
HighwaySimulator._assert_continuous_traffic_invariant = _assert_continuous_traffic_invariant
HighwaySimulator._sanitize_all_traffic_speeds = _sanitize_all_traffic_speeds
HighwaySimulator._encode_state = _six_slot_state


def create_highway_simulator(render_mode=None):
    return HighwaySimulator(render_mode=render_mode)


def create_carla_simulator():
    """Backward-compatible factory; currently returns Highway-env, not CARLA."""
    return create_highway_simulator()





# ============================================================
# 8. Create Initial Policy
# ============================================================
def create_initial_policy(
    state_dim,
    action_dim,
    hidden_dim,
    scale_of_steering,
):
    return ActorCriticPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        steering_scale=scale_of_steering,
    ).to(DEVICE)

# ============================================================
# 9. Collect Reference Rollouts
# ============================================================

def make_lane_request_counters():
    """Counters for each lane-change direction at distinct decision stages."""
    return {
        direction: {"requested": 0, "mask_allowed": 0, "target_selected": 0, "completed": 0}
        for direction in ("LEFT", "RIGHT")
    }


def make_action_mask_counters():
    """Measure high-level lane-change opportunities independently of choices."""
    return {"steps": 0, "change_valid": 0, "only_keep": 0}


def record_action_mask(counters, valid_action_mask):
    mask = np.asarray(valid_action_mask, dtype=bool).reshape(-1)
    counters["steps"] += 1
    change_valid = bool(mask[1] or mask[2]) if mask.size > 2 else (bool(mask[1]) if mask.size > 1 else False)
    counters["change_valid"] += int(change_valid)
    counters["only_keep"] += int(not change_valid)


def print_action_mask_counters(tag, counters):
    steps = max(1, counters["steps"])
    print(f"[{tag} ACTION OPPORTUNITIES]",
          f"steps={counters['steps']}",
          f"change_valid={counters['change_valid']} ({100*counters['change_valid']/steps:.1f}%)",
          f"only_keep={counters['only_keep']} ({100*counters['only_keep']/steps:.1f}%)")



def _physical_lane_action_mask(self):
    """Binary [KEEP, CHANGE] mask based only on physical lane existence."""
    current = int(self.env.unwrapped.vehicle.lane_index[2])
    lane_count = min(int(self.env.unwrapped.config.get("lanes_count", NUM_LANES)), NUM_LANES)
    candidates = [lane for lane in (current - 1, current + 1) if 0 <= lane < lane_count]
    self._last_valid_lane_candidates = list(candidates)
    return np.asarray([True, bool(candidates)], dtype=bool)


def _physical_target_lane_mask(self):
    """Absolute simulator lane IDs; only existing immediately adjacent lanes are valid."""
    current = int(self.env.unwrapped.vehicle.lane_index[2])
    lane_count = min(int(self.env.unwrapped.config.get("lanes_count", NUM_LANES)), NUM_LANES)
    mask = np.zeros(NUM_LANES, dtype=bool)
    for lane in (current - 1, current + 1):
        if 0 <= lane < lane_count:
            mask[lane] = True
    self._last_valid_lane_candidates = np.flatnonzero(mask).astype(int).tolist()
    return mask


def resolve_policy_lane_action(self, sampled_action, selected_target_lane=None):
    """Attach the policy-selected adjacent lane without modifying its action."""
    action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
    if action.size != 2:
        raise ValueError("Expected [normalized acceleration, binary KEEP/CHANGE]")
    current = int(self.env.unwrapped.vehicle.lane_index[2])
    requested = int(np.rint(float(action[1])))
    if requested not in (0, 1):
        raise ValueError(f"Invalid binary lane action: {requested}")
    mask = _physical_target_lane_mask(self)
    selected = current if selected_target_lane is None else int(selected_target_lane)
    valid = bool(0 <= selected < NUM_LANES and mask[selected])
    if requested == 1 and not valid:
        raise ValueError(
            f"Policy selected physically unavailable target lane {selected}; "
            f"current={current}, valid={np.flatnonzero(mask).tolist()}"
        )
    target = selected if requested == 1 else current
    info = {
        "current_lane_id": current, "requested_target_lane_id": int(selected),
        "selected_target_lane_id": int(selected), "executed_target_lane_id": int(target),
        "policy_requested_lane_action": "CHANGE" if requested else "KEEP",
        "executed_lane_action": "CHANGE" if requested else "KEEP",
        "policy_requested_binary_action": "CHANGE" if requested else "KEEP",
        "executed_binary_action": "CHANGE" if requested else "KEEP",
        "lane_change_blocked": False, "lane_change_rejection_reason": "none",
        "lane_request_direction": "LEFT" if target > current else "RIGHT" if target < current else "NONE",
        "lane_request_source_lane": current, "lane_request_target_lane": int(target),
        "lane_request_decision": "TARGET_SELECTED" if requested else "KEEP",
        "target_selector_mode": "physical_adjacent_lane_mask_only",
        "overridden": False, "emergency_braking": False,
        "acceleration_overridden": False, "edm_action_intervention": False,
        "valid_adjacent_lane_ids": np.flatnonzero(mask).astype(int).tolist(),
        "sampled_action": action.copy(), "policy_action": action.copy(),
        "proposed_policy_acceleration": float(action[0]), "executed_acceleration": float(action[0]),
    }
    return action.copy(), info



HighwaySimulator.get_valid_lane_action_mask = _physical_lane_action_mask
HighwaySimulator.get_valid_target_lane_mask = _physical_target_lane_mask
HighwaySimulator.resolve_policy_lane_action = resolve_policy_lane_action




def append_episode_metrics_csv(row):
    """Append one episode using a stable, self-describing CSV header."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    serializable = {
        key: (json.dumps(value, ensure_ascii=False, sort_keys=True)
              if isinstance(value, (dict, list, tuple)) else value)
        for key, value in row.items()
    }
    fields = list(serializable.keys())
    existing_fields = None
    if os.path.exists(METRICS_CSV_PATH) and os.path.getsize(METRICS_CSV_PATH) > 0:
        with open(METRICS_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            old_rows = list(reader)
        if existing_fields != fields:
            merged = list(existing_fields) + [name for name in fields if name not in existing_fields]
            with open(METRICS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=merged)
                writer.writeheader()
                writer.writerows(old_rows)
            fields = merged
    with open(METRICS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if os.path.getsize(METRICS_CSV_PATH) == 0:
            writer.writeheader()
        writer.writerow(serializable)
    with open(METRICS_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable, ensure_ascii=False, allow_nan=True) + "\n")


def _aggregate_episode_metrics(metrics):
    if not metrics:
        return {
            "episode_count": 0, "collision_rate": 0.0, "lane_change_rate": 0.0,
            "safe_travel_dis": 0.0, "safe_step_count": 0.0,
        }
    return {
        "episode_count": len(metrics),
        "collision_rate": float(np.mean([m["collision_rate"] for m in metrics])),
        "lane_change_rate": float(np.mean([m["lane_change_rate"] for m in metrics])),
        "safe_travel_dis": float(np.mean([m["safe_travel_dis"] for m in metrics])),
        "safe_step_count": float(np.mean([m["safe_step_count"] for m in metrics])),
        "mean_speed_mps": float(np.mean([m.get("mean_speed_mps", 0.0) for m in metrics])),
        "mean_speed_ratio": float(np.mean([m.get("mean_speed_ratio", 0.0) for m in metrics])),
        "mean_speed_error_mps": float(np.mean([m.get("mean_speed_error_mps", SPEED_LIMIT) for m in metrics])),
        "speed_target_fraction_90pct": float(np.mean([m.get("speed_target_fraction_90pct", 0.0) for m in metrics])),
        "no_safe_target_lane_fraction": float(np.mean([m.get("no_safe_target_lane_fraction", 0.0) for m in metrics])),
        "valid_target_lane_fraction": float(np.mean([m.get("valid_target_lane_fraction", 0.0) for m in metrics])),
        "mean_return": float(np.mean([m.get("episode_return", 0.0) for m in metrics])),
        "mean_step_reward": float(np.mean([m.get("mean_reward_per_step", 0.0) for m in metrics])),
    }


def build_evaluator_prompt(batch_metrics, reward_specification):
    """Build the exact prompt payload that a future local LLM can consume."""
    return {
        "role": (
            "Evaluate an RL highway-driving agent. Do not control the vehicle, "
            "override actions, or invent observations."
        ),
        "principles": [
            "Prioritize reaching and maintaining the road speed limit when traffic permits.",
            "Maintain safe following distance and avoid collisions.",
            "Do not reward or penalize a lane change merely because it occurs; judge its value through speed/progress and safety outcomes.",
            "Use TTC and minimum-gap thresholds only to reject unsafe adjacent lanes, not to rank safe lanes.",
            "Consider both front-vehicle and rear-vehicle conditions.",
            "Treat TTC thresholds as evidence, not universal optimal boundaries.",
            "Distinguish predicted risk from observed collision or near-collision outcomes.",
            "Recommend no change when the evidence is insufficient.",
        ],
        "metrics": batch_metrics,
        "current_reward_specification": {
            key: value for key, value in reward_specification.items()
            if isinstance(value, (str, int, float, bool))
        },
        "required_output": {
            "diagnosis": "Short evidence-grounded explanation",
            "recommendation": "One bounded reward change or no change",
            "reward_updates": "Only approved reward keys and bounded values",
            "expected_tradeoff": "Expected benefit and possible downside",
            "validation": "Metrics to compare over the next episode batch",
        },
    }


class PromptBasedEvaluator:
    """Transparent template/rule placeholder for a future local-LLM backend.

    It consumes aggregate episode metrics and returns structured suggestions.
    It never receives an action callback and cannot intervene in the simulator.
    """
    backend_name = "template_placeholder_not_a_real_llm"

    def evaluate(self, batch_metrics, reward_specification):
        prompt_payload = build_evaluator_prompt(batch_metrics, reward_specification)
        metrics = batch_metrics
        updates = {}
        diagnosis = "No persistent weakness was strong enough to justify adaptation."
        recommendation = "No change"
        expected_tradeoff = "Keeping the current objective avoids unnecessary reward drift."

        collision_rate = float(metrics.get("collision_rate", 0.0))
        lane_change_rate = float(metrics.get("lane_change_rate", 0.0))
        safe_steps = float(metrics.get("safe_step_count", 0.0))

        current_safety_weight = float(reward_specification.get("safety_weight", INITIAL_COLLISION_PENALTY))

        if int(metrics.get("episode_count", 0)) >= EVALUATOR_MIN_EPISODES and collision_rate >= 0.20:
            new_value = min(
                EVALUATOR_MAX_SAFETY_WEIGHT,
                max(EVALUATOR_MIN_SAFETY_WEIGHT, current_safety_weight + max(0.05, 0.10 * current_safety_weight)),
            )
            if new_value > current_safety_weight:
                updates["safety_weight"] = round(float(new_value), 6)
                diagnosis = (
                    f"Collisions occurred in {collision_rate:.0%} of the evaluation episodes. "
                    "The current collision penalty may be too weak relative to progress."
                )
                recommendation = "Increase the collision penalty modestly."
                expected_tradeoff = "May reduce risky behavior, but could also make learning more conservative."
        return {
            "backend": self.backend_name,
            "prompt_payload": prompt_payload,
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "reward_updates": updates,
            "expected_tradeoff": expected_tradeoff,
            "metrics": metrics,
        }


def _adaptation_validation_passed(current_metrics, baseline_metrics):
    """Reject an update if safety metrics materially deteriorate in the next batch."""
    collision_worsened = (
        current_metrics["collision_rate"] > baseline_metrics["collision_rate"] + 0.05
    )
    steps_base = max(float(baseline_metrics["safe_step_count"]), 1.0)
    steps_dropped = current_metrics["safe_step_count"] < 0.80 * steps_base
    distance_base = float(baseline_metrics["safe_travel_dis"])
    distance_dropped = (
        distance_base > 1.0
        and current_metrics["safe_travel_dis"] < 0.80 * distance_base
    )
    return not (collision_worsened or steps_dropped or distance_dropped), {
        "collision_worsened": bool(collision_worsened),
        "safe_step_count_dropped_over_20pct": bool(steps_dropped),
        "safe_travel_distance_dropped_over_20pct": bool(distance_dropped),
    }



def record_lane_request_decision(counters, sampled_action, valid_action_mask, safety_info):
    """Count binary CHANGE requests and use the selected target to infer direction."""
    action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
    if action.size < 2 or int(np.clip(np.rint(float(action[1])), 0, 1)) != 1: return
    current, target = safety_info.get("current_lane_id"), safety_info.get("requested_target_lane_id")
    if current is None or target is None or int(current) == int(target): return
    direction = "LEFT" if int(target) > int(current) else "RIGHT"
    counters[direction]["requested"] += 1
    mask = np.asarray(valid_action_mask, dtype=bool).reshape(-1)
    if mask.size > 1 and mask[1]: counters[direction]["mask_allowed"] += 1
    if not safety_info.get("lane_change_blocked", False): counters[direction]["target_selected"] += 1




def sync_lane_completion_counters(counters, simulator, seen_completed):
    """Record newly completed lane changes, attributed by their actual direction."""
    stats = getattr(simulator, "_lane_change_stats", {})
    for direction in ("LEFT", "RIGHT"):
        current = int(stats.get(f"completed_{direction}", 0))
        delta = max(0, current - seen_completed.get(direction, 0))
        counters[direction]["completed"] += delta
        seen_completed[direction] = current


def print_lane_request_counters(tag, counters):
    for direction in ("LEFT", "RIGHT"):
        values = counters[direction]
        print(
            f"[{tag} {direction} COUNTERS]",
            f"requested={values['requested']}",
            f"mask_allowed={values['mask_allowed']}",
            f"target_selected={values['target_selected']}",
            f"completed={values['completed']}",
        )



def calculate_min_ttc(simulator):
    """
    Return minimum TTC, in seconds, across relevant lanes.
    Uses HighwaySimulator._min_ttc_by_lane() as the single
    source of TTC values.
    """
    env = simulator.env.unwrapped
    ego = env.vehicle

    # Canonical lane relevance shared with the safety filter.
    physical_lane_id = int(ego.lane_index[2])
    relevant_lane_ids = simulator._relevant_lane_ids()
    lane_ttc = simulator._min_ttc_by_lane()

    values = []
    for lane_id in relevant_lane_ids:
        lane_values = lane_ttc.get(lane_id)
        if lane_values is None:
            continue

        # Include both front and rear threats in the reward metric.
        values.append(float(lane_values["front_ttc"]))
        values.append(float(lane_values["rear_ttc"]))

    min_ttc = min(values, default=float("inf"))
    return min_ttc, sum(np.isfinite(v) for v in values)


def construct_reward(reward_specification, embedding_model=None):
    """Reward acceleration below the speed limit and a fixed 1.0 at/above it.

    The separate speed reward is disabled. The unchanged collision penalty is
    applied on collision, and collision terminates the episode.
    """
    def reward_function(
        behavior_description, safety_violation, action, lane_deviation, offroad,
        min_ttc, lane_change_allowed=True, ego_speed=0.0, ego_speed_change_per_step_mps=0.0,
        safety_info=None, policy_sampled_action=None, reward_diagnostics=None,
        edm_warning_score=0.0, edm_warning_details=None,
    ):
        max_acceleration_per_step = float(
            reward_specification.get(
                "max_acceleration_per_step_mps", MAX_ACCELERATION_PER_STEP_MPS
            )
        )
        if not np.isfinite(max_acceleration_per_step) or max_acceleration_per_step <= 0.0:
            raise ValueError("max_acceleration_per_step_mps must be finite and positive.")
        speed_limit = float(reward_specification.get("speed_limit_mps", SPEED_LIMIT))
        if not np.isfinite(speed_limit) or speed_limit <= 0.0:
            raise ValueError("speed_limit_mps must be a finite positive value.")

        raw_speed_change = float(ego_speed_change_per_step_mps)
        clipped_speed_change = float(np.clip(
            raw_speed_change, -max_acceleration_per_step, max_acceleration_per_step
        ))
        current_speed = float(ego_speed)
        # Below the speed limit, retain the normal normalized positive-
        # acceleration reward. At/above the limit, assign exactly 1.0,
        # including when maintaining speed or decelerating.
        if current_speed >= speed_limit:
            acceleration_reward = 1.0
        else:
            acceleration_reward = float((2 * abs(clipped_speed_change) + clipped_speed_change) / (3.0 * max_acceleration_per_step))

        # Speed utility remains an EDM observation for the policy; it is not
        # added as a second scalar reward in this first controlled experiment.
        speed_reward = 0.0

        # Use the configured terminal collision penalty only on an actual
        # collision. The previous active line accidentally charged
        # 100 * acceleration_reward on every transition, regardless of collision.
        collision_penalty = (
            max(0.0, float(reward_specification.get("safety_weight", INITIAL_COLLISION_PENALTY)))
            if bool(safety_violation) else 0.0
        )

        edm_warning_score = float(np.clip(
            float(edm_warning_score) if EDM_REWARD_ENABLED else 0.0, 0.0, 1.0
        ))
        edm_warning_weight = max(
            0.0, float(reward_specification.get("edm_reward_weight", EDM_REWARD_WEIGHT))
        )
        edm_warning_penalty = edm_warning_weight * edm_warning_score

        reward = float(acceleration_reward - edm_warning_penalty - collision_penalty)
        if reward_diagnostics is not None:
            reward_diagnostics.update({
                "reward_objective": "acceleration_reward_at_or_below_speed_limit_minus_unchanged_collision_penalty",
                "acceleration_reward": acceleration_reward,
                "ego_speed_change_per_step_mps": raw_speed_change,
                "clipped_speed_change_per_step_mps": clipped_speed_change,
                "max_acceleration_per_step_mps": float(max_acceleration_per_step),
                "speed_reward": 0.0,
                "speed_reward_enabled": False,
                "edm_warning_enabled": bool(EDM_REWARD_ENABLED),
                "edm_warning_score": edm_warning_score,
                "edm_warning_weight": edm_warning_weight,
                "edm_warning_penalty": edm_warning_penalty,
                "edm_warning_details": edm_warning_details or {},
                "ego_speed_mps": current_speed,
                "speed_limit_mps": speed_limit,
                "safety_penalty": float(collision_penalty),
                "collision_penalty": float(collision_penalty),
                "offroad_penalty": 0.0,  # Off-road remains a terminal safeguard, not a reward term.
                "collision_terminates_episode": bool(safety_violation),
                "total_reward": reward,
            })
        print(
            f"[REWARD DEBUG] acceleration_reward={acceleration_reward:+.3f}, "
            f"speed_change_per_step={raw_speed_change:+.4f}m/s "
            f"(clipped={clipped_speed_change:+.4f}), "
            f"speed_reward=DISABLED, "
            f"edm_warning={edm_warning_score:.3f}*{edm_warning_weight:.2f}="
            f"{edm_warning_penalty:.3f}, "
            f"speed={current_speed:.3f}m/s, limit={speed_limit:.3f}m/s, "
            f"collision_penalty={collision_penalty:.3f}, total={reward:+.3f}"
        )
        return reward
    return reward_function


def compute_reward(
    reward_function,
    behavior_description,
    safety_violation,
    action,
    lane_deviation,
    offroad,
    min_ttc=float("inf"),
    lane_change_allowed=True,
    ego_speed=0.0,
    ego_speed_change_per_step_mps=0.0,
    safety_info=None,
    policy_sampled_action=None,
    reward_diagnostics=None,
    edm_warning_score=0.0,
    edm_warning_details=None,
):
    return reward_function(
        behavior_description=behavior_description,
        safety_violation=safety_violation,
        action=action,
        lane_deviation=lane_deviation,
        offroad=offroad,
        min_ttc=min_ttc,
        lane_change_allowed=lane_change_allowed,
        ego_speed=ego_speed,
        ego_speed_change_per_step_mps=ego_speed_change_per_step_mps,
        safety_info=safety_info,
        policy_sampled_action=policy_sampled_action,
        reward_diagnostics=reward_diagnostics,
        edm_warning_score=edm_warning_score,
        edm_warning_details=edm_warning_details,
    )



# ============================================================
# 21. RL Policy Update
# ============================================================
def update_policy(policy, state, action, reward, next_state, done,
                  valid_action_mask=None, terminated=None, pre_tanh_accel=None,
                  selected_target_lane=-1, valid_target_lane_mask=None,
                  current_lane_id=None, lane_action_overridden=False,
                  acceleration_overridden=False):
    """
    One-step actor-critic update with intervention-aware actor credit assignment.

    The critic always learns from the observed transition. The actor only gets
    a policy-gradient term for action dimensions that were actually allowed
    through the safety filter. The sampled action is retained for computing
    the likelihood of the policy's original sample.
    """
    st, nst = to_tensor(state).reshape(1, -1), to_tensor(next_state).reshape(1, -1)
    act, rew = to_tensor(action).reshape(1, -1), to_tensor([reward])
    value = policy.value(st)
    with torch.no_grad():
        nv = policy.value(nst)
        terminal = bool(done) if terminated is None else bool(terminated)
        target = rew if terminal else rew + GAMMA * nv
    adv = target - value
    critic_loss = F.mse_loss(value, target)

    # Use the sampled action to evaluate the policy distribution that generated it.
    sampled_lane_action = int(np.clip(
        np.rint(float(act[0, 1].detach().cpu().item())), 0, 1
    ))
    current = int(current_lane_id) if current_lane_id is not None else 0
    sampled_target = (
        int(selected_target_lane)
        if sampled_lane_action == 1 and int(selected_target_lane) >= 0
        else current
    )
    sampled_target = int(np.clip(sampled_target, 0, NUM_LANES - 1))

    binary_dist = policy.action_distributions(
        st, torch.as_tensor(valid_action_mask, dtype=torch.bool, device=DEVICE).reshape(1, -1)
        if valid_action_mask is not None else None
    )
    binary_log_prob = binary_dist.log_prob(torch.as_tensor(
        [sampled_lane_action], dtype=torch.long, device=DEVICE
    ))

    target_log_prob = torch.zeros(1, dtype=torch.float32, device=DEVICE)
    if (not lane_action_overridden and sampled_lane_action == 1
            and int(selected_target_lane) >= 0 and valid_target_lane_mask is not None):
        lane_mask = torch.as_tensor(
            valid_target_lane_mask, dtype=torch.bool, device=DEVICE
        ).reshape(1, -1).clone()
        if current_lane_id is not None and 0 <= int(current_lane_id) < lane_mask.shape[-1]:
            lane_mask[:, int(current_lane_id)] = False
        if lane_mask.any(dim=-1).all():
            lane_dist = policy.lane_distributions(st, lane_mask)
            target_log_prob = lane_dist.log_prob(torch.as_tensor(
                [int(selected_target_lane)], dtype=torch.long, device=DEVICE
            ))

    accel_value = (
        float(pre_tanh_accel) if pre_tanh_accel is not None else
        float(np.arctanh(np.clip(
            float(act[0, 0].detach().cpu().item()), -0.999, 0.999
        )))
    )
    pre_tanh = torch.as_tensor([accel_value], dtype=torch.float32, device=DEVICE)
    accel_dist = policy.acceleration_distribution(st, sampled_lane_action, sampled_target)
    acceleration_log_prob = policy.squashed_normal_log_prob(accel_dist, pre_tanh)

    # This is a joint, hierarchically conditioned policy: acceleration is
    # conditioned on the sampled lane decision. If the filter changes ANY
    # part of the sampled action, the observed return belongs to the filtered
    # action, not the sampled joint action. Therefore skip the entire actor
    # likelihood-ratio term for this transition rather than crediting a
    # partially mismatched action. The critic still learns from the real
    # state transition and reward.
    actor_log_prob = torch.zeros(1, dtype=torch.float32, device=DEVICE)
    actor_update_valid = not (
        bool(lane_action_overridden) or bool(acceleration_overridden)
    )
    if actor_update_valid:
        actor_log_prob = binary_log_prob + target_log_prob + acceleration_log_prob

    actor_loss = -(actor_log_prob * adv.detach()).mean()
    loss = actor_loss + critic_loss
    policy.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    policy.optimizer.step()
    return {
        "actor_loss": float(actor_loss.detach().item()),
        "critic_loss": float(critic_loss.detach().item()),
        "advantage": float(adv.detach().mean().item()),
        "binary_log_prob": float(binary_log_prob.detach().mean().item()),
        "target_lane_log_prob": float(target_log_prob.detach().mean().item()),
        "acceleration_log_prob": float(acceleration_log_prob.detach().mean().item()),
        "actor_update_valid": bool(actor_update_valid),
    }






# ============================================================
# 22. Save Policy
# ============================================================


def save_policy(policy, path):
    if not isinstance(policy, nn.Module):
        raise TypeError("save_policy() expected a PyTorch policy model, " f"but received {type(policy).__name__}.")

    torch.save(policy.state_dict(), path)

    print(f"Policy saved to: {path}")





# ============================================================
# No embedding model is loaded in this fixed-reward variant.
# ============================================================




# ============================================================
# Legacy reference-rollout helpers are retained for reproducibility only; cold-start training does not call them.
# ============================================================
def build_fixed_reward_function():
    reward_specification = {
        "name": "pure_rl_episode_adaptive_reward",
        "target_behavior": (
            "Below the speed limit, reward positive speed change per policy step using "
            "(abs(delta_v_step) + delta_v_step) / (2 * max_acceleration_per_step). "
            "At or above the speed limit, acceleration_reward is exactly 1.0. "
            "The separate speed reward remains disabled. Apply a continuous action-conditioned "
            "EDM TTC warning cost over one policy interval, plus the configured terminal collision "
            "penalty. EDM does not veto or rewrite actions. Do not include a progress reward or "
            "lane-change reward/cost. Do not add an off-road reward penalty."
        ),
        "safety_weight": INITIAL_COLLISION_PENALTY,
        "max_acceleration_per_step_mps": float(MAX_ACCELERATION_PER_STEP_MPS),
        "policy_step_duration_s": float(POLICY_STEP_DURATION_S),
        "speed_limit_mps": float(SPEED_LIMIT),
        "speed_reward_enabled": False,
        "collision_terminates_episode": True,
        "edm_reward_enabled": bool(EDM_REWARD_ENABLED),
        "edm_reward_weight": float(EDM_REWARD_WEIGHT),
        "edm_reward_horizon_s": float(EDM_REWARD_HORIZON_S),
        "edm_reward_semantics": "action_conditioned_continuous_TTC_risk_proxy_not_calibrated_collision_probability",
    }
    reward_function = construct_reward(reward_specification)
    # The closure and evaluator share this mutable specification. Only the
    # episode-level adaptation module may update it, between episodes.
    reward_function.reward_specification = reward_specification
    return reward_function


def _hybrid_action_log_probability(policy, states, actions, pre_tanh_accels, masks,
                                  selected_target_lanes=None, target_lane_masks=None,
                                  current_lane_ids=None):
    """Joint log-likelihood for lane action, target lane, and conditional acceleration."""
    binary_actions = torch.clamp(torch.round(actions[:, 1]).long(), 0, 1)
    binary_dist = policy.action_distributions(states, torch.as_tensor(masks, dtype=torch.bool, device=DEVICE))
    binary_log_prob = binary_dist.log_prob(binary_actions)
    target_log_prob = torch.zeros_like(binary_log_prob)
    batch = actions.shape[0]
    currents = (torch.as_tensor(current_lane_ids, dtype=torch.long, device=DEVICE).reshape(-1)
                if current_lane_ids is not None else torch.zeros(batch, dtype=torch.long, device=DEVICE))
    lanes = (torch.as_tensor(selected_target_lanes, dtype=torch.long, device=DEVICE).reshape(-1)
             if selected_target_lanes is not None else currents.clone())
    effective_targets = torch.where((binary_actions == 1) & (lanes >= 0), lanes, currents).clamp(0, NUM_LANES - 1)
    if selected_target_lanes is not None and target_lane_masks is not None and current_lane_ids is not None:
        lane_masks = torch.as_tensor(target_lane_masks, dtype=torch.bool, device=DEVICE).clone()
        rows = torch.arange(lane_masks.shape[0], device=DEVICE)
        valid_current = (currents >= 0) & (currents < lane_masks.shape[1])
        lane_masks[rows[valid_current], currents[valid_current]] = False
        change_rows = (binary_actions == 1) & (lanes >= 0) & (lanes < lane_masks.shape[1])
        if change_rows.any():
            selected_rows = rows[change_rows]
            selected_mask = lane_masks[change_rows]
            has_candidates = selected_mask.any(dim=-1)
            if has_candidates.any():
                candidate_rows = selected_rows[has_candidates]
                lane_dist = policy.lane_distributions(states[candidate_rows], selected_mask[has_candidates])
                target_log_prob[candidate_rows] = lane_dist.log_prob(lanes[candidate_rows])
    pre_tanh = torch.as_tensor(pre_tanh_accels, dtype=actions.dtype, device=DEVICE).reshape(-1)
    accel_dist = policy.acceleration_distribution(states, binary_actions, effective_targets)
    acceleration_log_prob = policy.squashed_normal_log_prob(accel_dist, pre_tanh)
    return binary_log_prob + target_log_prob + acceleration_log_prob




def collect_reference_rollouts(simulator, policy, reward_function):
    """
    Collect complete policy trajectories without updating the policy.
    These trajectories are subsequently used for reward/return-weighted
    policy warm-start, not for OOD detection.
    """
    trajectories = []
    total_transitions = 0
    reference_total_lane_counters = make_lane_request_counters()
    reference_total_mask_counters = make_action_mask_counters()
    old_phase = getattr(simulator, "_trace_phase", None)

    for episode in range(NUM_REFERENCE_EPISODES):
        simulator._trace_phase = "reference_collection"
        simulator._trace_episode = episode + 1
        state = np.asarray(
            simulator.reset(), dtype=np.float32
        )
        record_episode_start(simulator, "reference_collection", episode + 1, state)
        lane_change_processing_remaining = 0
        trajectory = {
            "states": [], "actions": [], "pre_tanh_accels": [], "masks": [],
            "selected_target_lanes": [], "target_lane_masks": [], "current_lane_ids": [],
            "rewards": [], "next_states": [], "terminated": []
        }
        lane_counters = make_lane_request_counters()
        action_mask_counters = make_action_mask_counters()
        seen_completed = {
            "LEFT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_LEFT", 0)),
            "RIGHT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_RIGHT", 0)),
        }

        for step_index in range(MAX_STEPS):
            valid_mask = np.asarray(
                simulator.get_valid_lane_action_mask(), dtype=bool
            )
            record_action_mask(action_mask_counters, valid_mask)
            target_lane_mask = np.asarray(simulator.get_valid_target_lane_mask(), dtype=bool)
            current_lane_id = int(simulator.env.unwrapped.vehicle.lane_index[2])
            sampled_action, policy_aux = policy.act(
                state, deterministic=False, valid_action_mask=valid_mask,
                valid_target_lane_mask=target_lane_mask,
                current_lane_id=current_lane_id, return_aux=True,
            )
            sampled_action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
            pre_tanh_accel = float(policy_aux.get("pre_tanh_accel", 0.0))
            selected_target_lane = int(policy_aux.get("selected_target_lane", -1))
            policy_action, safety_info = simulator.resolve_policy_lane_action(
                sampled_action, selected_target_lane=selected_target_lane
            )
            policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
            record_lane_request_decision(
                lane_counters, sampled_action, valid_mask, safety_info or {}
            )
            previous_speed_mps = float(simulator.env.unwrapped.vehicle.speed)
            vehicles_before = snapshot_simulator_vehicles(simulator)
            next_state, _, done, step_info = simulator.step(
                policy_action,
                    sampled_action=sampled_action,
                safety_info=safety_info,
            )
            next_state = np.asarray(next_state, dtype=np.float32)
            sync_lane_completion_counters(lane_counters, simulator, seen_completed)

            min_ttc, _ = calculate_min_ttc(simulator)
            safety_violation = bool(
                step_info.get("safety_violation", False)
                if isinstance(step_info, dict) else step_info
            )
            lane_info = simulator.get_lane_information()
            offroad = not bool(lane_info.get("on_road", True))
            lane_deviation = float(lane_info.get("lane_deviation", 0.0))
            if offroad:
                done = True

            reward_safety_info = dict(safety_info or {})


            reward_safety_info["lane_change_accepted"] = bool(step_info.get("lane_change_completed_instantly", False)
                                                              ) if isinstance(step_info, dict) else False


            reward = compute_reward(
                reward_function=reward_function,
                behavior_description="",
                safety_violation=safety_violation,
                action=np.asarray(
                    step_info.get("executed_policy_action", policy_action)
                    if isinstance(step_info, dict) else policy_action,
                    dtype=np.float32,
                ),
                lane_deviation=lane_deviation,
                offroad=offroad,
                min_ttc=min_ttc,
                lane_change_allowed=True,
                ego_speed=float(simulator.env.unwrapped.vehicle.speed),
                ego_speed_change_per_step_mps=(
                    float(simulator.env.unwrapped.vehicle.speed) - previous_speed_mps
                ),
                safety_info=reward_safety_info,
                policy_sampled_action=sampled_action,
                reward_diagnostics={},
            )

            if lane_change_processing_remaining > 0:
                reward = 0.0
                lane_change_processing_remaining -= 1
            if bool(step_info.get("lane_change_completed_instantly", False)):
                lane_change_processing_remaining = LANE_CHANGE_PROCESSING_STEPS

            trajectory["states"].append(state.copy())
            trajectory["actions"].append(sampled_action.copy())
            trajectory["pre_tanh_accels"].append(pre_tanh_accel)
            trajectory["masks"].append(valid_mask.copy())
            trajectory["selected_target_lanes"].append(int(selected_target_lane))
            trajectory["target_lane_masks"].append(target_lane_mask.copy())
            trajectory["current_lane_ids"].append(int(current_lane_id))
            trajectory["rewards"].append(float(reward))
            trajectory["next_states"].append(next_state.copy())
            is_terminated = (
                bool(step_info.get("terminated", False) or offroad)
                if isinstance(step_info, dict) else bool(done or offroad)
            )
            trajectory["terminated"].append(is_terminated)
            record_episode_transition(
                simulator=simulator,
                phase="reference_collection",
                episode_number=episode + 1,
                step_number=step_index + 1,
                state_before=state,
                state_after=next_state,
                vehicles_before=vehicles_before,
                sampled_action=sampled_action.tolist(),
                executed_action=np.asarray(
                    step_info.get("executed_policy_action", policy_action)
                    if isinstance(step_info, dict) else policy_action,
                    dtype=np.float32,
                ).reshape(-1).tolist(),
                selected_target_lane=selected_target_lane,
                valid_action_mask=valid_mask.tolist(),
                target_lane_mask=target_lane_mask.tolist(),
                safety_info=safety_info,
                step_info=step_info,
                reward=reward,
                done=done,
                terminated=is_terminated,
                truncated=bool(step_info.get("truncated", False)) if isinstance(step_info, dict) else False,
            )
            total_transitions += 1
            state = next_state
            if done:
                break

        # Monte-Carlo discounted returns, computed separately per episode.
        returns = np.zeros(len(trajectory["rewards"]), dtype=np.float32)
        running_return = 0.0
        for i in range(len(returns) - 1, -1, -1):
            running_return = trajectory["rewards"][i] + GAMMA * running_return
            returns[i] = running_return
        trajectory["returns"] = returns
        trajectories.append(trajectory)
        print(
            f"[REFERENCE COLLECTION] episode={episode + 1}/{NUM_REFERENCE_EPISODES} "
            f"steps={len(returns)} return={float(returns[0]) if len(returns) else 0.0:+.3f}"
        )
        print_lane_request_counters("REFERENCE EPISODE", lane_counters)
        print_action_mask_counters("REFERENCE EPISODE", action_mask_counters)
        for direction in ("LEFT", "RIGHT"):
            for counter_name in ("requested", "mask_allowed", "target_selected", "completed"):
                reference_total_lane_counters[direction][counter_name] += lane_counters[direction][counter_name]
        for counter_name in reference_total_mask_counters:
            reference_total_mask_counters[counter_name] += action_mask_counters[counter_name]

    print_lane_request_counters("REFERENCE TOTAL", reference_total_lane_counters)
    print_action_mask_counters("REFERENCE TOTAL", reference_total_mask_counters)
    simulator._trace_phase = old_phase or "pure_rl"
    if total_transitions == 0:
        raise RuntimeError("Reference collection produced no transitions.")

    # Persist the exact data used for warm-start so experiments are reproducible.
    os.makedirs("data", exist_ok=True)
    flattened = {
        key: np.concatenate([np.asarray(traj[key]) for traj in trajectories], axis=0)
        for key in (
            "states", "actions", "pre_tanh_accels", "masks",
            "selected_target_lanes", "target_lane_masks", "current_lane_ids",
            "rewards", "next_states", "terminated", "returns"
        )
    }
    reference_data_path = os.path.join("data", "binary_lane_ranked_reference_rollouts.npz")
    np.savez_compressed(reference_data_path, **flattened)
    print(
        f"[REFERENCE COLLECTION COMPLETE] episodes={len(trajectories)} "
        f"transitions={total_transitions} saved={reference_data_path}"
    )
    return flattened


def warm_start_policy_from_reference(policy, reference_data):
    """
    Reward/return-weighted offline warm-start.

    The actor learns to increase likelihood of collected actions with higher
    estimated advantage; the critic regresses to discounted trajectory returns.
    This is deliberately separate from the online one-step Actor-Critic update.
    """
    states = to_tensor(reference_data["states"])
    actions = to_tensor(reference_data["actions"])
    pre_tanh = to_tensor(reference_data["pre_tanh_accels"]).reshape(-1)
    masks = torch.as_tensor(reference_data["masks"], dtype=torch.bool, device=DEVICE)
    selected_target_lanes = torch.as_tensor(reference_data["selected_target_lanes"], dtype=torch.long, device=DEVICE)
    target_lane_masks = torch.as_tensor(reference_data["target_lane_masks"], dtype=torch.bool, device=DEVICE)
    current_lane_ids = torch.as_tensor(reference_data["current_lane_ids"], dtype=torch.long, device=DEVICE)
    returns = to_tensor(reference_data["returns"]).reshape(-1)

    with torch.no_grad():
        baseline = policy.value(states).reshape(-1)
        advantages = returns - baseline
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-6
            )
        # Exponentiated advantage weights are bounded to avoid one trajectory
        # dominating the warm-start.
        weights = torch.exp(advantages.clamp(-2.0, 2.0) / 2.0)
        weights = weights / (weights.mean() + 1e-6)

    count = states.shape[0]
    batch_size = min(REFERENCE_BATCH_SIZE, count)
    for epoch in range(REFERENCE_TRAINING_EPOCHS):
        permutation = torch.randperm(count, device=DEVICE)
        actor_losses, critic_losses = [], []
        for start in range(0, count, batch_size):
            idx = permutation[start:start + batch_size]
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_pre_tanh = pre_tanh[idx]
            batch_masks = masks[idx]
            batch_target_lanes = selected_target_lanes[idx]
            batch_target_masks = target_lane_masks[idx]
            batch_current_lanes = current_lane_ids[idx]
            batch_returns = returns[idx]
            batch_weights = weights[idx]

            log_prob = _hybrid_action_log_probability(
                policy, batch_states, batch_actions, batch_pre_tanh, batch_masks,
                batch_target_lanes, batch_target_masks, batch_current_lanes
            )
            actor_loss = -(batch_weights.detach() * log_prob).mean()
            predicted_returns = policy.value(batch_states).reshape(-1)
            critic_loss = F.mse_loss(predicted_returns, batch_returns)
            loss = actor_loss + critic_loss

            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            policy.optimizer.step()
            actor_losses.append(float(actor_loss.detach().item()))
            critic_losses.append(float(critic_loss.detach().item()))

        print(
            f"[REFERENCE WARM-START] epoch={epoch + 1}/{REFERENCE_TRAINING_EPOCHS} "
            f"actor_loss={np.mean(actor_losses):+.5f} "
            f"critic_loss={np.mean(critic_losses):.5f}"
        )

    print(
        "[REFERENCE WARM-START COMPLETE] The reference dataset has updated "
        "the actor and critic; online Actor-Critic training starts next."
    )


# ============================================================
# Online Actor-Critic training from a randomly initialized policy — no OOD
# ============================================================
def run_traffic_diagnostic(simulator, steps=TRAFFIC_DIAGNOSTIC_STEPS):
    """Run a short non-learning drive and export complete traffic trajectories."""
    os.makedirs(TRAFFIC_DIAGNOSTIC_OUTPUT_DIR, exist_ok=True)
    trajectory_csv = os.path.join(
        TRAFFIC_DIAGNOSTIC_OUTPUT_DIR, "traffic_vehicle_trajectories.csv"
    )
    population_csv = os.path.join(
        TRAFFIC_DIAGNOSTIC_OUTPUT_DIR, "traffic_population_by_step.csv"
    )

    simulator._trace_phase = "traffic_diagnostic"
    simulator._trace_episode = 1
    simulator.reset(seed=SEED + 10000)
    base_env = simulator.env.unwrapped
    print(
        "[TRAFFIC DIAGNOSTIC START]",
        f"steps={int(steps)}",
        f"duration_s={int(steps) * POLICY_STEP_DURATION_S:.1f}",
        f"surrounding_population={len(base_env.road.vehicles) - 1}",
        f"hard_max={MAX_SURROUNDING_TRAFFIC_VEHICLES}",
        "ego_control=KEEP + neutral_acceleration",
        "rl_updates=DISABLED",
    )

    trajectory_fields = [
        "step", "time_s", "vehicle_uid", "is_ego", "lane_id",
        "position_x_m", "position_y_m", "longitudinal_s_m",
        "speed_mps", "acceleration_mps2", "heading_rad",
        "profile", "desired_speed_mps", "max_acceleration_mps2",
        "ordinary_braking_mps2", "emergency_brake_active", "crashed",
    ]
    population_fields = [
        "step", "time_s", "surrounding_vehicle_count", "ego_lane",
        "ego_speed_mps", "spawned_total", "removed_total",
        "pending_replacements", "collision",
    ]

    with open(trajectory_csv, "w", newline="", encoding="utf-8") as trajectory_file, \
         open(population_csv, "w", newline="", encoding="utf-8") as population_file:
        trajectory_writer = csv.DictWriter(trajectory_file, fieldnames=trajectory_fields)
        population_writer = csv.DictWriter(population_file, fieldnames=population_fields)
        trajectory_writer.writeheader()
        population_writer.writeheader()

        def write_snapshot(step_number, collision=False):
            road = simulator.env.unwrapped.road
            ego_vehicle = simulator.env.unwrapped.vehicle
            current_time = step_number * POLICY_STEP_DURATION_S
            traffic_count = sum(vehicle is not ego_vehicle for vehicle in road.vehicles)
            population_writer.writerow({
                "step": step_number,
                "time_s": round(current_time, 4),
                "surrounding_vehicle_count": traffic_count,
                "ego_lane": int(ego_vehicle.lane_index[2]),
                "ego_speed_mps": float(ego_vehicle.speed),
                "spawned_total": int(getattr(simulator, "_traffic_spawn_count", 0)),
                "removed_total": int(getattr(simulator, "_traffic_removal_count", 0)),
                "pending_replacements": int(sum(
                    getattr(simulator, "_traffic_pending_replacements_by_side", {}).values()
                )),
                "collision": bool(collision),
            })
            for vehicle in list(road.vehicles):
                position = np.asarray(vehicle.position, dtype=float)
                longitudinal_s = float(simulator._road_longitudinal_position(vehicle))
                trajectory_writer.writerow({
                    "step": step_number,
                    "time_s": round(current_time, 4),
                    "vehicle_uid": (
                        "EGO" if vehicle is ego_vehicle else
                        getattr(vehicle, "_arood_traffic_uid", str(id(vehicle)))
                    ),
                    "is_ego": vehicle is ego_vehicle,
                    "lane_id": int(vehicle.lane_index[2]),
                    "position_x_m": float(position[0]),
                    "position_y_m": float(position[1]),
                    "longitudinal_s_m": longitudinal_s,
                    "speed_mps": float(vehicle.speed),
                    "acceleration_mps2": float(
                        getattr(vehicle, "action", {}).get("acceleration", 0.0)
                    ),
                    "heading_rad": float(vehicle.heading),
                    "profile": "ego" if vehicle is ego_vehicle else
                        getattr(vehicle, "_arood_adt_profile", "unknown"),
                    "desired_speed_mps": (
                        "" if vehicle is ego_vehicle else
                        float(getattr(vehicle, "_arood_desired_speed", np.nan))
                    ),
                    "max_acceleration_mps2": (
                        "" if vehicle is ego_vehicle else
                        float(IDM_MAX_ACCEL_MPS2)
                    ),
                    "ordinary_braking_mps2": (
                        "" if vehicle is ego_vehicle else
                        float(IDM_COMFORTABLE_BRAKING_MPS2)
                    ),
                    "emergency_brake_active": (
                        False if vehicle is ego_vehicle else
                        bool(getattr(vehicle, "_arood_emergency_brake_active", False))
                    ),
                    "crashed": bool(getattr(vehicle, "crashed", False)),
                })

        write_snapshot(0)
        completed_steps = 0
        collision_seen = False
        for step_number in range(1, int(steps) + 1):
            # Normalized acceleration=0 and binary action=KEEP.
            next_state, _, done, info = simulator.step(
                np.asarray([0.0, 0.0], dtype=np.float32),
                sampled_action=np.asarray([0.0, 0.0], dtype=np.float32),
                safety_info={},
            )
            _ = next_state
            collision_seen = bool(
                info.get("collision", False) if isinstance(info, dict) else False
            )
            write_snapshot(step_number, collision=collision_seen)
            completed_steps = step_number
            if done:
                print(
                    "[TRAFFIC DIAGNOSTIC STOP]",
                    f"step={step_number}",
                    f"collision={collision_seen}",
                    "reason=simulator_terminal",
                )
                break

    final_population = len(simulator.env.unwrapped.road.vehicles) - 1
    simulator._assert_continuous_traffic_invariant("traffic_diagnostic_complete")
    print(
        "[TRAFFIC DIAGNOSTIC COMPLETE]",
        f"steps={completed_steps}",
        f"final_population={final_population}",
        f"max_population={MAX_SURROUNDING_TRAFFIC_VEHICLES}",
        f"collision={collision_seen}",
        f"trajectory_csv={os.path.abspath(trajectory_csv)}",
        f"population_csv={os.path.abspath(population_csv)}",
    )
    return {
        "steps": completed_steps,
        "collision": collision_seen,
        "trajectory_csv": trajectory_csv,
        "population_csv": population_csv,
        "final_population": final_population,
    }


def train_pure_rl(simulator, policy):
    reward_history = []
    mean_step_reward_history = []
    episode_times = []
    collision_history = []
    lane_change_history = []
    # Episode-level evaluation metrics:
    # episode_collision_indicator: 1 if any collision occurs, otherwise 0.
    # lane_change_rate: completed lane changes divided by policy decisions.
    # safe_travel_dis: road-longitudinal distance accumulated on collision-free transitions.
    # safe_step_count: completed collision-free transitions before first collision.
    collision_rate_history = []
    lane_change_rate_history = []
    safe_travel_dis_history = []
    safe_step_count_history = []
    emergency_braking_rate_history = []
    lane_change_rejection_rate_history = []
    acceleration_override_rate_history = []
    overall_intervention_rate_history = []
    interventions_per_km_history = []
    actor_loss_history = []
    critic_loss_history = []
    episode_metric_history = []
    evaluator = PromptBasedEvaluator()
    reward_specification = None
    pending_adaptation = None

    reward_function = build_fixed_reward_function()
    reward_specification = reward_function.reward_specification

    os.makedirs(os.path.dirname(reward_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Append episode rows; this experiment has its own output directory and
    # does not touch the historical J_DDPG or V experiment logs.
    with open(CONFIG_PATH, "w", encoding="utf-8") as _config_file:
        json.dump({
            "algorithm": "hierarchical_stochastic_actor_critic",
            "training_initialization": "cold_start_random_policy_initialization",
            "training_mode": "normal_online_actor_critic_training",
            "training_episodes": NUM_ONLINE_EPISODES,
            "reference_rollout_collection_enabled": False,
            "reference_offline_warm_start_enabled": False,
            "policy": "KEEP_CHANGE_then_masked_adjacent_target_then_conditional_squashed_gaussian_acceleration",
            "state_dim": STATE_DIM, "policy_action_dim": POLICY_ACTION_DIM,
            "execution_action_dim": ACTION_DIM, "num_lanes": NUM_LANES,
            "simulation_frequency_hz": simulation_frequency,
            "simulation_step_duration_s": SIMULATION_STEP_DURATION_S,
            "policy_frequency_hz": policy_frequency,
            "policy_step_duration_s": POLICY_STEP_DURATION_S,
            "simulation_substeps_per_policy_action": SIMULATION_SUBSTEPS_PER_POLICY,
            "episode_duration_s": MAX_EPISODE_DURATION_S,
            "max_policy_steps_per_episode": MAX_STEPS,
            "discount_gamma_per_policy_step": GAMMA,
            "discount_time_constant_s": DISCOUNT_TIME_CONSTANT_S,
            "traffic_reference_front_m": TRAFFIC_REFERENCE_FRONT_M,
            "traffic_reference_rear_m": TRAFFIC_REFERENCE_REAR_M,
            "traffic_entry_buffer_m": TRAFFIC_ENTRY_BUFFER_M,
            "traffic_recycling_buffer_m": TRAFFIC_RECYCLING_BUFFER_M,
            "traffic_front_recycle_boundary_m": TRAFFIC_FRONT_RECYCLE_BOUNDARY_M,
            "traffic_rear_recycle_boundary_m": TRAFFIC_REAR_RECYCLE_BOUNDARY_M,
            "traffic_max_surrounding_vehicles": MAX_SURROUNDING_TRAFFIC_VEHICLES,
            "traffic_initial_population_min": MIN_INITIAL_SURROUNDING_TRAFFIC_VEHICLES,
            "traffic_initial_population_max": MAX_SURROUNDING_TRAFFIC_VEHICLES,
            "traffic_population_target_is_random_per_episode": True,
            "traffic_minimum_desired_and_initial_speed_mps": TRAFFIC_MIN_SPEED_MPS,
            "traffic_ordinary_max_acceleration_mps2": IDM_MAX_ACCEL_MPS2,
            "traffic_ordinary_max_braking_mps2": IDM_COMFORTABLE_BRAKING_MPS2,
            "traffic_emergency_braking_magnitude_mps2": ADT_EMERGENCY_BRAKING_MPS2,
            "traffic_stopped_vehicle_restart_rule": True,
            "six_nearest_slots_are_observation_only": True,
            "boundary_only_vehicle_replacement": True,
            "speed_limit_mps": SPEED_LIMIT,
            "max_physical_acceleration_mps2": MAX_PHYSICAL_ACCELERATION,
            "max_acceleration_per_step_mps": MAX_ACCELERATION_PER_STEP_MPS,
            "acceleration_reward_formula": "(abs(delta_v_step) + delta_v_step) / (2 * max_acceleration_per_step)",
            "speed_reward_formula": "(current_speed - speed_limit) / speed_limit",
            "lane_change_processing_steps": LANE_CHANGE_PROCESSING_STEPS,
            "edm_ttc_thresholds_s": {
                "comfortable": EDM_TTC_COMFORTABLE_S,
                "danger": EDM_TTC_DANGER_S,
                "critical": EDM_TTC_CRITICAL_S,
            },
            "edm_high_risk_score_threshold": EDM_HIGH_RISK_SCORE_THRESHOLD,
            "edm_exposure_metric_snapshot": "edm_pre_action_by_lane",
            "collision_counting": "one_event_per_uninterrupted_collision_contact_episode",
            "collision_contact_steps_logged_separately": True,
            "edm_is_interpretation_only": True,
            "edm_action_intervention": False,
            "lane_mask": "physical_adjacent_lanes_only; no TTC/EDM veto",
            "lane_state_representation": "ego_relative_roles_current_left_right; absolute_lane_id_not_exposed_in_context",
            "episode_evaluator_enabled": ENABLE_EPISODE_EVALUATOR,
            "metrics_csv": METRICS_CSV_PATH,
        }, _config_file, indent=2, ensure_ascii=False)
    '''
    try:
        os.makedirs(os.path.dirname(DIAGNOSTIC_LOG_PATH) or ".", exist_ok=True)
        with open(DIAGNOSTIC_LOG_PATH, "w", encoding="utf-8"):
            pass
    except OSError as exc:
        print("[DIAGNOSTIC LOG WARNING]", f"path={DIAGNOSTIC_LOG_PATH}", f"error={exc}")
    '''

    # Cold-start: do not collect reference trajectories and do not perform
    # offline actor/critic warm-start. The randomly initialized policy enters
    # online simulator interaction immediately and learns only from its own
    # on-policy transitions through the existing Actor-Critic updates.
    print("\n[COLD START] Reference rollout collection DISABLED.")
    print("[COLD START] Offline reference warm-start DISABLED.")
    print("[ONLINE RL] Learning begins immediately from the randomly initialized policy; updates occur on every collected transition.")

    for episode in range(NUM_ONLINE_EPISODES):
        simulator._trace_phase = "pure_rl"
        simulator._trace_episode = episode + 1
        start_time = time.time()
        state = simulator.reset()
        record_episode_start(simulator, "pure_rl", episode + 1, state)
        episode_return = 0.0
        episode_steps = 0
        lane_change_processing_remaining = 0
        collisions = 0  # new collision-contact onsets, not repeated contact steps
        collision_contact_steps = 0  # transitions spent in collision contact
        first_collision_step = None
        safe_travel_dis = 0.0
        # Measure road-longitudinal displacement, not Euclidean XY displacement:
        # instantaneous lane changes move laterally and must not add fake distance.
        previous_ego_s = float(simulator._road_longitudinal_position(simulator.env.unwrapped.vehicle))
        episode_actor_losses = []
        episode_critic_losses = []
        episode_speed_samples = []
        lane_occupancy_counts = {lane: 0 for lane in range(NUM_LANES)}
        edm_current_lane_risk_sum = 0.0
        edm_current_lane_high_risk_steps = 0
        lane_change_processing_step_count = 0
        no_safe_target_lane_steps = 0
        valid_target_lane_steps = 0
        lane_counters = make_lane_request_counters()
        intervention_counts = {
            "emergency_braking_steps": 0,
            "lane_change_requests": 0,
            "lane_change_rejections": 0,
            "acceleration_override_steps": 0,
            "steps_with_any_intervention": 0,
            "intervention_events": 0,
        }
        safe_intervention_steps = 0
        seen_completed = {
            "LEFT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_LEFT", 0)),
            "RIGHT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_RIGHT", 0)),
        }

        for t in range(MAX_STEPS):
            valid_mask = simulator.get_valid_lane_action_mask()
            simulator._trace_valid_action_mask = np.asarray(valid_mask, dtype=bool).tolist()

            target_lane_mask = np.asarray(simulator.get_valid_target_lane_mask(), dtype=bool)
            # The target mask contains only physically existing adjacent lanes;
            # the policy handles KEEP separately in its binary distribution.
            if bool(target_lane_mask.any()):
                valid_target_lane_steps += 1
            else:
                no_safe_target_lane_steps += 1
            current_lane_id = int(simulator.env.unwrapped.vehicle.lane_index[2])
            lane_occupancy_counts[current_lane_id] += 1
            sampled_action, policy_aux = policy.act(
                state, deterministic=False, valid_action_mask=valid_mask,
                valid_target_lane_mask=target_lane_mask,
                current_lane_id=current_lane_id, return_aux=True,
            )
            sampled_action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
            pre_tanh_accel = float(policy_aux.get("pre_tanh_accel", 0.0))
            selected_target_lane = int(policy_aux.get("selected_target_lane", -1))
            lane_policy_diagnostic(
                policy, state, valid_mask, sampled_action, phase="pure_rl",
                episode=episode + 1, step=t + 1, selected_target_lane=selected_target_lane,
                valid_target_lane_mask=target_lane_mask,
                current_lane_id=current_lane_id,
            )
            policy_action, safety_info = simulator.resolve_policy_lane_action(
                sampled_action, selected_target_lane=selected_target_lane
            )
            policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
            record_lane_request_decision(lane_counters, sampled_action, valid_mask, safety_info)

            # Count safety-layer interventions on online-training steps. An acceleration
            # override includes front-threat emergency braking OR rear-threat escape
            # acceleration; emergency_braking_steps is the front-threat subset.
            requested_change = int(np.clip(np.rint(float(sampled_action[1])), 0, 1)) == 1
            emergency_intervention = bool(safety_info.get("emergency_braking", False))
            lane_change_rejected = bool(requested_change and safety_info.get("lane_change_blocked", False))
            acceleration_override = bool(safety_info.get("acceleration_overridden", False))
            any_intervention = bool(emergency_intervention or lane_change_rejected or acceleration_override)
            intervention_counts["emergency_braking_steps"] += int(emergency_intervention)
            intervention_counts["lane_change_requests"] += int(requested_change)
            intervention_counts["lane_change_rejections"] += int(lane_change_rejected)
            intervention_counts["acceleration_override_steps"] += int(acceleration_override)
            intervention_counts["steps_with_any_intervention"] += int(any_intervention)
            intervention_events_this_step = (
                int(emergency_intervention) + int(lane_change_rejected) + int(acceleration_override)
            )
            intervention_counts["intervention_events"] += intervention_events_this_step
            intervention_step_this_step = int(any_intervention)
            previous_speed_mps = float(simulator.env.unwrapped.vehicle.speed)
            vehicles_before = snapshot_simulator_vehicles(simulator)

            next_state, simulator_reward, done, step_info = simulator.step(
                policy_action,
                    sampled_action=sampled_action,
                safety_info=safety_info,
            )
            next_state = np.asarray(next_state, dtype=np.float32)
            executed_action = np.asarray(
                step_info.get("executed_policy_action", policy_action),
                dtype=np.float32,
            ).reshape(-1)

            # EDM exposure is measured from the exact pre-action snapshot
            # captured by simulator.step(), not from a later reconstructed
            # scene. The lane is the ego's lane before the policy decision.
            pre_action_edm = step_info.get("edm_pre_action_by_lane", {}) if isinstance(step_info, dict) else {}
            pre_action_lane_edm = pre_action_edm.get(current_lane_id, pre_action_edm.get(str(current_lane_id), {}))
            current_lane_risk = float(pre_action_lane_edm.get("lane_max_risk_score", 0.0))
            if not np.isfinite(current_lane_risk):
                current_lane_risk = 1.0
            current_lane_risk = float(np.clip(current_lane_risk, 0.0, 1.0))
            edm_current_lane_risk_sum += current_lane_risk
            if current_lane_risk >= EDM_HIGH_RISK_SCORE_THRESHOLD:
                edm_current_lane_high_risk_steps += 1

            min_ttc, closing_vehicle_count = calculate_min_ttc(simulator)
            collision_occurred = bool(
                step_info.get("collision", step_info.get("safety_violation", False))
                if isinstance(step_info, dict) else step_info
            )
            collisions += int(collision_occurred)
            collision_contact_steps += int(
                bool(step_info.get("collision_contact_active", collision_occurred))
                if isinstance(step_info, dict) else collision_occurred
            )
            current_ego_s = float(
                simulator._road_longitudinal_position(simulator.env.unwrapped.vehicle)
            )
            if collision_occurred:
                if first_collision_step is None:
                    # 1-based index: a collision on the first transition is step 1.
                    first_collision_step = episode_steps + 1
                # Do not count the collision transition as safely travelled distance.
            elif first_collision_step is None:
                # Count only transitions before the first collision in both the
                # safe-distance and safe-intervention numerators.
                safe_travel_dis += max(0.0, current_ego_s - previous_ego_s)
                safe_intervention_steps += intervention_step_this_step
            previous_ego_s = current_ego_s

            safety_violation = collision_occurred
            lane_info = simulator.get_lane_information()
            offroad = not bool(lane_info.get("on_road", True))
            lane_deviation = float(lane_info.get("lane_deviation", 0.0))
            if offroad:
                done = True

            reward_components = {}
            reward_safety_info = dict(safety_info or {})

            reward_safety_info["lane_change_accepted"] = bool(
                safety_info.get("lane_request_decision") == "TARGET_SELECTED"
            )
            reward_safety_info["lane_change_completed"] = bool(
                step_info.get("lane_change_completed_instantly", False)
            ) if isinstance(step_info, dict) else False

            current_speed_mps = float(simulator.env.unwrapped.vehicle.speed)
            episode_speed_samples.append(current_speed_mps)

            # Use the exact pre-action EDM snapshot captured inside simulator.step().
            # KEEP evaluates current-lane front/rear interactions; CHANGE evaluates
            # front/rear lane-entry interactions in the selected target lane.
            # This only changes reward; it never vetoes or rewrites the policy action.
            pre_action_edm = (
                step_info.get("edm_pre_action_by_lane", {})
                if isinstance(step_info, dict) else {}
            )
            reward_lane_id = int(
                step_info.get("accepted_target_lane_id", current_lane_id)
                if int(round(float(executed_action[1]))) == 1
                else current_lane_id
            )
            edm_warning_details = estimate_edm_action_warning(
                edm_pre_action_by_lane=pre_action_edm,
                current_lane_id=current_lane_id,
                selected_target_lane=reward_lane_id,
                lane_change_action=executed_action[1],
                ego_speed_mps=previous_speed_mps,
                normalized_acceleration=executed_action[0],
                horizon_s=EDM_REWARD_HORIZON_S,
            )
            edm_warning_score = float(edm_warning_details["warning_score"])
            reward_components["edm_warning"] = edm_warning_details

            reward = compute_reward(
                reward_function=reward_function,
                behavior_description="",
                safety_violation=safety_violation,
                action=executed_action,
                lane_deviation=lane_deviation,
                offroad=offroad,
                min_ttc=min_ttc,
                lane_change_allowed=True,
                ego_speed=current_speed_mps,
                ego_speed_change_per_step_mps=(
                    current_speed_mps - previous_speed_mps
                ),
                safety_info=reward_safety_info,
                policy_sampled_action=sampled_action,
                reward_diagnostics=reward_components,
                edm_warning_score=edm_warning_score,
                edm_warning_details=edm_warning_details,
            )

            # Suspend the complete reward for exactly one policy transition
            # immediately following an executed instantaneous lane change.
            if lane_change_processing_remaining > 0:
                reward = 0.0
                lane_change_processing_step_count += 1
                reward_components["lane_change_processing_step"] = True
                reward_components["reward_suspended"] = True
                lane_change_processing_remaining -= 1
            else:
                reward_components["lane_change_processing_step"] = False
                reward_components["reward_suspended"] = False
            if bool(step_info.get("lane_change_completed_instantly", False)):
                lane_change_processing_remaining = LANE_CHANGE_PROCESSING_STEPS

            record_episode_transition(
                simulator=simulator,
                phase="pure_rl",
                episode_number=episode + 1,
                step_number=t + 1,
                state_before=state,
                state_after=next_state,
                vehicles_before=vehicles_before,
                sampled_action=sampled_action.tolist(),
                executed_action=executed_action.tolist(),
                selected_target_lane=selected_target_lane,
                valid_action_mask=np.asarray(valid_mask, dtype=bool).tolist(),
                target_lane_mask=target_lane_mask.tolist(),
                safety_info=safety_info,
                step_info=step_info,
                reward=reward,
                reward_components=reward_components,
                done=done,
                terminated=bool(step_info.get("terminated", False) or offroad)
                    if isinstance(step_info, dict) else bool(done or offroad),
                truncated=bool(step_info.get("truncated", False))
                    if isinstance(step_info, dict) else False,
            )

            update_stats = update_policy(
                policy=policy,
                state=state,
                action=sampled_action,
                reward=reward,
                next_state=next_state,
                done=done,
                valid_action_mask=valid_mask,
                terminated=bool(step_info.get("terminated", False) or offroad)
                    if isinstance(step_info, dict) else bool(done or offroad),
                pre_tanh_accel=pre_tanh_accel,
                selected_target_lane=selected_target_lane,
                valid_target_lane_mask=target_lane_mask,
                current_lane_id=current_lane_id,
            )
            episode_actor_losses.append(update_stats["actor_loss"])
            episode_critic_losses.append(update_stats["critic_loss"])

            sync_lane_completion_counters(lane_counters, simulator, seen_completed)
            episode_return += float(reward)
            episode_steps += 1
            write_diagnostic_event(
                "transition", simulator,
                step=t + 1,
                sampled_action=sampled_action.tolist(),
                executed_action=executed_action.tolist(),
                valid_action_mask=np.asarray(valid_mask, dtype=bool).tolist(),
                safety_info=safety_info,
                reward=float(reward),
                reward_components=reward_components,
                min_ttc=float(min_ttc),
                closing_vehicle_count=int(closing_vehicle_count),
                safety_violation=safety_violation,
                offroad=offroad,
                lane_change_stats=dict(getattr(simulator, "_lane_change_stats", {})),
                actor_loss=float(update_stats["actor_loss"]),
                critic_loss=float(update_stats["critic_loss"]),
            )
            state = next_state
            if done:
                break

        elapsed = time.time() - start_time
        episode_simulated_duration_s = episode_steps * POLICY_STEP_DURATION_S
        mean_step_reward = episode_return / max(episode_steps, 1)
        mean_reward_per_simulated_second = (
            episode_return / episode_simulated_duration_s
            if episode_simulated_duration_s > 0.0 else 0.0
        )
        speed_array = np.asarray(episode_speed_samples, dtype=np.float64)
        mean_speed_mps = float(np.mean(speed_array)) if speed_array.size else 0.0
        max_speed_mps = float(np.max(speed_array)) if speed_array.size else 0.0
        mean_speed_ratio = mean_speed_mps / max(SPEED_LIMIT, 1e-6)
        speed_target_fraction = float(np.mean(speed_array >= 0.90 * SPEED_LIMIT)) if speed_array.size else 0.0
        mean_speed_error_mps = float(np.mean(np.maximum(0.0, SPEED_LIMIT - speed_array))) if speed_array.size else float(SPEED_LIMIT)
        no_safe_target_lane_fraction = no_safe_target_lane_steps / max(episode_steps, 1)
        valid_target_lane_fraction = valid_target_lane_steps / max(episode_steps, 1)
        reward_history.append(episode_return)
        mean_step_reward_history.append(mean_step_reward)
        episode_times.append(elapsed)
        collision_history.append(collisions)
        lane_stats = dict(getattr(simulator, "_lane_change_stats", {}))
        lane_change_history.append(lane_stats)

        # collision_rate is a per-episode indicator (0/1), not a per-step rate.
        # safe_step_count excludes the collision transition itself.
        collision_rate = float(first_collision_step is not None)
        completed_lane_changes = int(lane_stats.get("completed", 0))
        lane_change_rate = completed_lane_changes / max(episode_steps, 1)
        lane_changes_per_minute = (
            completed_lane_changes / (episode_simulated_duration_s / 60.0)
            if episode_simulated_duration_s > 0.0 else 0.0
        )
        lane_changes_per_safe_km = (
            completed_lane_changes / (safe_travel_dis / 1000.0)
            if safe_travel_dis > 1e-9 else 0.0
        )
        safe_step_count = int(
            first_collision_step - 1 if first_collision_step is not None else episode_steps
        )
        collision_rate_history.append(collision_rate)
        lane_change_rate_history.append(lane_change_rate)
        safe_travel_dis_history.append(float(safe_travel_dis))
        safe_step_count_history.append(safe_step_count)

        # Step rates use policy decisions as denominator. Rejection rate uses
        # lane-change attempts. Distance is stored in metres and converted to km
        # only for the interpretable interventions-per-km metric.
        episode_step_denominator = max(episode_steps, 1)
        emergency_braking_rate = intervention_counts["emergency_braking_steps"] / episode_step_denominator
        lane_change_rejection_rate = (
            intervention_counts["lane_change_rejections"] / intervention_counts["lane_change_requests"]
            if intervention_counts["lane_change_requests"] else 0.0
        )
        acceleration_override_rate = intervention_counts["acceleration_override_steps"] / episode_step_denominator
        overall_intervention_rate = intervention_counts["steps_with_any_intervention"] / episode_step_denominator
        distance_km = float(safe_travel_dis) / 1000.0
        interventions_per_km = safe_intervention_steps / distance_km if distance_km > 1e-9 else 0.0
        emergency_braking_rate_history.append(float(emergency_braking_rate))
        lane_change_rejection_rate_history.append(float(lane_change_rejection_rate))
        acceleration_override_rate_history.append(float(acceleration_override_rate))
        overall_intervention_rate_history.append(float(overall_intervention_rate))
        interventions_per_km_history.append(float(interventions_per_km))

        actor_loss_history.append(float(np.mean(episode_actor_losses)) if episode_actor_losses else 0.0)
        critic_loss_history.append(float(np.mean(episode_critic_losses)) if episode_critic_losses else 0.0)

        print_lane_request_counters("PURE RL EPISODE", lane_counters)
        print(
            f"Episode {episode + 1:03d}/{NUM_ONLINE_EPISODES} | "
            f"steps={episode_steps} | return={episode_return:.3f} | "
            f"mean_step_reward={mean_step_reward:+.4f} | collisions={collisions} "
            f"(contact_steps={collision_contact_steps}) | "
            f"mean_speed={mean_speed_mps:.2f}m/s ({mean_speed_ratio:.1%} target) | "
            f"speed>=90%target={speed_target_fraction:.1%} | "
            f"no_safe_lane={no_safe_target_lane_fraction:.1%} | "
            f"episode_had_collision={bool(collision_rate)} | "
            f"lane_changes_per_100_steps={100.0 * lane_change_rate:.2f} | "
            f"lane_changes_per_minute={lane_changes_per_minute:.2f} | "
            f"lane_changes_per_safe_km={lane_changes_per_safe_km:.2f} | "
            f"simulated_duration={episode_simulated_duration_s:.1f}s | "
            f"collision_free_distance={safe_travel_dis:.2f}m | "
            f"collision_free_steps={safe_step_count} | "
            f"emergency_braking_rate={emergency_braking_rate:.4f} | "
            f"lane_change_rejection_rate={lane_change_rejection_rate:.4f} "
            f"({intervention_counts['lane_change_rejections']}/{intervention_counts['lane_change_requests']} requests) | "
            f"acceleration_override_rate={acceleration_override_rate:.4f} | "
            f"overall_intervention_rate={overall_intervention_rate:.4f} | "
            f"intervention_steps_per_safe_km={interventions_per_km:.3f} "
            f"(steps={safe_intervention_steps}, distance={safe_travel_dis:.1f}m) | "
            f"lane_requested={lane_stats.get('requested', 0)} | "
            f"lane_accepted={lane_stats.get('accepted', 0)} | "
            f"lane_completed={lane_stats.get('completed', 0)} | "
            f"actor_loss={actor_loss_history[-1]:.5f} | "
            f"critic_loss={critic_loss_history[-1]:.5f} | elapsed={elapsed:.1f}s"
        )
        episode_metrics = {
            "episode": int(episode + 1),
            "episode_return": float(episode_return),
            "episode_simulated_duration_s": float(episode_simulated_duration_s),
            "mean_reward_per_step": float(mean_step_reward),
            "mean_reward_per_simulated_second": float(mean_reward_per_simulated_second),
            "mean_speed_mps": mean_speed_mps,
            "max_speed_mps": max_speed_mps,
            "mean_speed_ratio": mean_speed_ratio,
            "mean_speed_error_mps": mean_speed_error_mps,
            "speed_target_fraction_90pct": speed_target_fraction,
            "no_safe_target_lane_fraction": no_safe_target_lane_fraction,
            "valid_target_lane_fraction": valid_target_lane_fraction,
            "steps": int(episode_steps),
            "collisions": int(collisions),
            "collision_contact_steps": int(collision_contact_steps),
            "collision_rate": float(collision_rate),
            "lane_change_rate": float(lane_change_rate),
            "lane_changes_per_minute": float(lane_changes_per_minute),
            "lane_changes_per_safe_km": float(lane_changes_per_safe_km),
            "safe_travel_dis": float(safe_travel_dis),
            "safe_step_count": int(safe_step_count),
            "emergency_braking_rate": float(emergency_braking_rate),
            "lane_change_rejection_rate": float(lane_change_rejection_rate),
            "acceleration_override_rate": float(acceleration_override_rate),
            "overall_intervention_rate": float(overall_intervention_rate),
            "interventions_per_km": float(interventions_per_km),
            "intervention_counts": dict(intervention_counts),
            "collision_free_intervention_steps": int(safe_intervention_steps),
            "mean_actor_loss": float(actor_loss_history[-1]),
            "mean_critic_loss": float(critic_loss_history[-1]),
            "lane_occupancy_counts": dict(lane_occupancy_counts),
            "lane_occupancy_fraction": {
                str(lane): float(count / max(episode_steps, 1))
                for lane, count in lane_occupancy_counts.items()
            },
            "mean_current_lane_edm_risk": float(edm_current_lane_risk_sum / max(episode_steps, 1)),
            "current_lane_high_risk_exposure_steps": int(edm_current_lane_high_risk_steps),
            "current_lane_high_risk_exposure_fraction": float(edm_current_lane_high_risk_steps / max(episode_steps, 1)),
            "edm_high_risk_score_threshold": float(EDM_HIGH_RISK_SCORE_THRESHOLD),
            "edm_exposure_snapshot": "edm_pre_action_by_lane",
            "lane_change_processing_steps": int(lane_change_processing_step_count),
        }
        episode_metric_history.append(episode_metrics)
        append_episode_metrics_csv(episode_metrics)

        # Periodic policy checkpoints protect long normal-training runs.
        checkpoint_interval = max(1, int(os.environ.get("AROOD_CHECKPOINT_INTERVAL", "10")))
        if (episode + 1) % checkpoint_interval == 0:
            save_policy(policy, MODEL_PATH)

        write_diagnostic_event(
            "episode_summary", simulator,
            episode_return=float(episode_return),
            mean_reward_per_step=float(mean_step_reward),
            mean_speed_mps=mean_speed_mps,
            max_speed_mps=max_speed_mps,
            mean_speed_ratio=mean_speed_ratio,
            mean_speed_error_mps=mean_speed_error_mps,
            speed_target_fraction_90pct=speed_target_fraction,
            no_safe_target_lane_fraction=no_safe_target_lane_fraction,
            valid_target_lane_fraction=valid_target_lane_fraction,
            steps=int(episode_steps),
            collisions=int(collisions),
            collision_contact_steps=int(collision_contact_steps),
            collision_rate=float(collision_rate),
            lane_change_rate=float(lane_change_rate),
            safe_travel_dis=float(safe_travel_dis),
            safe_step_count=int(safe_step_count),
            emergency_braking_rate=float(emergency_braking_rate),
            lane_change_rejection_rate=float(lane_change_rejection_rate),
            acceleration_override_rate=float(acceleration_override_rate),
            overall_intervention_rate=float(overall_intervention_rate),
            interventions_per_km=float(interventions_per_km),
            intervention_counts=dict(intervention_counts),
            collision_free_intervention_steps=int(safe_intervention_steps),
            first_collision_step=first_collision_step,
            lane_change_stats=lane_stats,
            lane_request_counters=lane_counters,
            mean_actor_loss=actor_loss_history[-1],
            mean_critic_loss=critic_loss_history[-1],
        )

        # Evaluate only between episodes, never inside the action loop.
        # Any applied update remains pending until the next batch validates it.
        batch_boundary = (
            (episode + 1) % EVALUATOR_BATCH_SIZE == 0
            or episode + 1 == NUM_ONLINE_EPISODES
        )
        if ENABLE_EPISODE_EVALUATOR and batch_boundary:
            current_batch = _aggregate_episode_metrics(
                episode_metric_history[-EVALUATOR_BATCH_SIZE:]
            )

            if pending_adaptation is not None:
                passed, validation_details = _adaptation_validation_passed(
                    current_batch, pending_adaptation["baseline_metrics"]
                )
                if passed:
                    validation_status = "accepted"
                else:
                    for key, old_value in pending_adaptation["previous_values"].items():
                        reward_specification[key] = old_value
                    validation_status = "rolled_back"
                write_diagnostic_event(
                    "reward_adaptation_validation", simulator,
                    status=validation_status,
                    proposed_updates=pending_adaptation["updates"],
                    baseline_metrics=pending_adaptation["baseline_metrics"],
                    validation_metrics=current_batch,
                    validation_details=validation_details,
                    reward_specification=dict(reward_specification),
                )
                print(
                    "[EVALUATOR ADAPTATION VALIDATION]",
                    f"status={validation_status}",
                    f"details={validation_details}",
                )
                pending_adaptation = None

            evaluator_result = evaluator.evaluate(
                current_batch, reward_specification
            )
            proposed_updates = dict(evaluator_result["reward_updates"])
            can_validate_later = episode + 1 < NUM_ONLINE_EPISODES
            applied_updates = {}

            if proposed_updates and can_validate_later:
                previous_values = {
                    key: float(reward_specification[key])
                    for key in proposed_updates
                }
                for key, value in proposed_updates.items():
                    if key == "safety_weight":
                        value = float(np.clip(
                            value, EVALUATOR_MIN_SAFETY_WEIGHT,
                            EVALUATOR_MAX_SAFETY_WEIGHT
                        ))
                    else:
                        continue
                    reward_specification[key] = value
                    applied_updates[key] = value

                if applied_updates:
                    pending_adaptation = {
                        "updates": dict(applied_updates),
                        "previous_values": previous_values,
                        "baseline_metrics": current_batch,
                    }

            evaluator_event = {
                "backend": evaluator_result["backend"],
                "episode_batch": current_batch,
                "diagnosis": evaluator_result["diagnosis"],
                "recommendation": evaluator_result["recommendation"],
                "proposed_updates": proposed_updates,
                "applied_updates": applied_updates,
                "pending_validation": bool(applied_updates),
                "final_batch_unapplied": bool(proposed_updates and not can_validate_later),
                "expected_tradeoff": evaluator_result["expected_tradeoff"],
                "prompt_payload": evaluator_result["prompt_payload"],
                "reward_specification_after_call": dict(reward_specification),
            }
            write_diagnostic_event("episode_level_evaluator", simulator, **evaluator_event)
            print(
                "[EPISODE-LEVEL EVALUATOR]",
                f"backend={evaluator_result['backend']}",
                f"batch_episodes={current_batch['episode_count']}",
                f"collision_rate={current_batch['collision_rate']:.3f}",
                f"lane_change_rate={current_batch['lane_change_rate']:.4f}",
                f"diagnosis={evaluator_result['diagnosis']}",
                f"recommendation={evaluator_result['recommendation']}",
                f"proposed_updates={proposed_updates}",
                f"applied_updates={applied_updates}",
                f"reward_specification={reward_specification}",
            )

    # Preserve the existing training-curve workflow.
    episodes = np.arange(len(reward_history))
    plt.figure()
    plt.plot(episodes, reward_history, alpha=0.35, label="raw episode return")
    if len(reward_history) >= 3:
        window = min(10, len(reward_history))
        smoothed = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, len(reward_history)), smoothed,
                 linewidth=2.0, label=f"{window}-episode moving average")
    plt.xlabel("episode")
    plt.ylabel("total episode return")
    plt.title("Pure RL training return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_path)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()






    print("\\n[FINAL EPISODE METRICS — means across episodes]")
    print(f"episode_collision_incidence={float(np.mean(collision_rate_history)) if collision_rate_history else 0.0:.4f}")
    print(f"lane_changes_per_100_steps={100.0 * (float(np.mean(lane_change_rate_history)) if lane_change_rate_history else 0.0):.2f}")
    print(f"collision_free_distance={float(np.mean(safe_travel_dis_history)) if safe_travel_dis_history else 0.0:.2f} m/episode")
    print(f"collision_free_steps={float(np.mean(safe_step_count_history)) if safe_step_count_history else 0.0:.2f} steps/episode")
    print(f"emergency_braking_rate={float(np.mean(emergency_braking_rate_history)) if emergency_braking_rate_history else 0.0:.4f} (per-step rate)")
    print(f"lane_change_rejection_rate={float(np.mean(lane_change_rejection_rate_history)) if lane_change_rejection_rate_history else 0.0:.4f} (per-request rate)")
    print(f"acceleration_override_rate={float(np.mean(acceleration_override_rate_history)) if acceleration_override_rate_history else 0.0:.4f} (per-step rate)")
    print(f"overall_intervention_rate={float(np.mean(overall_intervention_rate_history)) if overall_intervention_rate_history else 0.0:.4f} (per-step rate)")
    print(f"intervention_steps_per_collision_free_km={float(np.mean(interventions_per_km_history)) if interventions_per_km_history else 0.0:.3f}")


    with open(plot_path, "w", encoding="utf-8") as f:
        f.write(f"episode_collision_incidence={float(np.mean(collision_rate_history)) if collision_rate_history else 0.0:.4f}"+ "\n")
        f.write(f"lane_changes_per_policy_step={float(np.mean(lane_change_rate_history)) if lane_change_rate_history else 0.0:.4f}"+ "\n")
        f.write(f"collision_free_distance_m={float(np.mean(safe_travel_dis_history)) if safe_travel_dis_history else 0.0:.2f} per episode"+ "\n")
        f.write(f"collision_free_steps={float(np.mean(safe_step_count_history)) if safe_step_count_history else 0.0:.2f} per episode"+ "\n")
        f.write(f"emergency_braking_rate={float(np.mean(emergency_braking_rate_history)) if emergency_braking_rate_history else 0.0:.4f} (per-step rate)"+ "\n")
        f.write(f"lane_change_rejection_rate={float(np.mean(lane_change_rejection_rate_history)) if lane_change_rejection_rate_history else 0.0:.4f} (per-request rate)"+ "\n")
        f.write(f"acceleration_override_rate={float(np.mean(acceleration_override_rate_history)) if acceleration_override_rate_history else 0.0:.4f} (per-step rate)"+ "\n")
        f.write(f"overall_intervention_rate={float(np.mean(overall_intervention_rate_history)) if overall_intervention_rate_history else 0.0:.4f} (per-step rate)"+ "\n")
        f.write(f"intervention_steps_per_collision_free_km={float(np.mean(interventions_per_km_history)) if interventions_per_km_history else 0.0:.3f}"+ "\n")
        f.write("reward:\n" + str(reward_history) + "\n")
        f.write("collision:\n" + str(collision_history) + "\n")
        f.write("collision_rate:\n" + str(collision_rate_history) + "\n")
        f.write("safe_travel_distance:\n" + str(safe_travel_dis_history) + "\n")
        f.write("safe_step_count:\n" + str(safe_step_count_history) + "\n")
        f.write("lane_change:\n" + str(lane_change_history) + "\n")
        f.write("episode_times:\n" + str(episode_times) + "\n")
        f.write("lane_change_rate:\n" + str(lane_change_rate_history) + "\n")
        f.write("mean_step_reward:\n" + str(mean_step_reward_history) + "\n")



    save_policy(policy, MODEL_PATH)
    return {
        "policy": policy,
        "reward_history": reward_history,
        "mean_step_reward_history": mean_step_reward_history,
        "collision_history": collision_history,
        "lane_change_history": lane_change_history,
        "collision_rate": collision_rate_history,
        "lane_change_rate": lane_change_rate_history,
        "safe_travel_dis": safe_travel_dis_history,
        "safe_step_count": safe_step_count_history,
        "emergency_braking_rate": emergency_braking_rate_history,
        "lane_change_rejection_rate": lane_change_rejection_rate_history,
        "acceleration_override_rate": acceleration_override_rate_history,
        "overall_intervention_rate": overall_intervention_rate_history,
        "interventions_per_km": interventions_per_km_history,
    }


def check_model_device(model, name):
    devices = {
        str(parameter.device)
        for parameter in model.parameters()
    }

    print(f"{name}: {devices}")

    assert all(
        parameter.device == DEVICE
        for parameter in model.parameters()
    ), f"{name} has parameters on the wrong device"


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Selected DEVICE:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Startup: NORMAL ONLINE RL TRAINING ({NUM_ONLINE_EPISODES} episodes); no diagnostic-only phase.")
    print(f"Dimensions: state={STATE_DIM}, policy_action={POLICY_ACTION_DIM} [KEEP/CHANGE], execution_action={ACTION_DIM}.")
    print("Control: hierarchical stochastic policy selects KEEP/CHANGE, then a physically valid adjacent target, then conditional acceleration.")
    print(f"EDM reward: enabled={EDM_REWARD_ENABLED}, TTC warning weight={EDM_REWARD_WEIGHT:.3f}, horizon={EDM_HORIZON_POLICY_STEPS} policy step(s) / {EDM_REWARD_HORIZON_S:.3f}s; no action veto or acceleration override.")
    print("Episode termination: ENABLED on the first collision; no post-collision overlap transitions.")
    print("Lane-change processing: DISABLED (no reward suspension after lane changes).")
    print(f"CSV metrics: {METRICS_CSV_PATH}")
    print(f"Simulator: {NUM_LANES} lanes, randomized surrounding traffic={MIN_INITIAL_SURROUNDING_TRAFFIC_VEHICLES}-{MAX_SURROUNDING_TRAFFIC_VEHICLES} (hard maximum excludes ego), six nearest observation slots.")
    print(f"Timing: simulation={SIMULATION_FREQUENCY_HZ}Hz, policy={POLICY_FREQUENCY_HZ}Hz, "
        f"{SIMULATION_STEPS_PER_POLICY_STEP} simulator updates/policy action, "
        f"episode={MAX_EPISODE_STEPS} policy steps ({MAX_EPISODE_DURATION_S:.1f}s).")
    discount_tau_text = (
        "infinite" if math.isinf(DISCOUNT_TIME_CONSTANT_S)
        else f"{DISCOUNT_TIME_CONSTANT_S:.2f}s"
    )
    print(f"RL discount: gamma={GAMMA:.6f} per {POLICY_STEP_DURATION_S:.3f}s decision (discount time constant={discount_tau_text}).")
    print(f"Traffic window: +{TRAFFIC_REFERENCE_FRONT_M:.0f}m front / -{TRAFFIC_REFERENCE_REAR_M:.0f}m rear; entry buffer={TRAFFIC_ENTRY_BUFFER_M:.0f}m; recycling=+{TRAFFIC_FRONT_RECYCLE_BOUNDARY_M:.0f}/-{TRAFFIC_REAR_RECYCLE_BOUNDARY_M:.0f}m.")
    print(
        f"ADT: max surrounding vehicles={MAX_SURROUNDING_TRAFFIC_VEHICLES}, "
        f"randomized initial population={MIN_INITIAL_SURROUNDING_TRAFFIC_VEHICLES}-"
        f"{MAX_SURROUNDING_TRAFFIC_VEHICLES}, desired cruising speed="
        f"{TRAFFIC_MIN_SPEED_MPS:.0f}-{TRAFFIC_MAX_SPEED_MPS:.0f}m/s, "
        f"profile mix={dict(ADT_PROFILE_MIX)}."
    )
    print(
        f"Ego acceleration bounds: ±{MAX_PHYSICAL_ACCELERATION:.1f} m/s²; "
        f"background ordinary acceleration/braking: ±{IDM_MAX_ACCEL_MPS2:.1f} m/s²; "
        f"background emergency braking: -{ADT_EMERGENCY_BRAKING_MPS2:.1f} m/s²; "
        f"traffic desired-speed floor: {TRAFFIC_MIN_SPEED_MPS:.1f} m/s."
    )
    print(f"Episode evaluator enabled: {ENABLE_EPISODE_EVALUATOR} (fixed reward baseline).")
    #print(f"Diagnostics: {DIAGNOSTIC_LOG_PATH}; trajectories: {EPISODE_TRAJECTORY_PATH if TRAJECTORY_RECORDING_ENABLED else 'disabled'}.")
    print(f"[REWARD CONFIG] acceleration reward + EDM TTC warning cost (weight={EDM_REWARD_WEIGHT:.2f}) + terminal collision penalty; speed utility remains an observation; collision terminates episode.")
    simulator = create_highway_simulator()
    print("[ONLINE RL] Starting policy initialization and Actor-Critic training directly.")
    policy = create_initial_policy(
        state_dim=STATE_DIM,
        action_dim=POLICY_ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        scale_of_steering=steering_scale,
    )
    check_model_device(policy, "Policy")
    results = train_pure_rl(simulator, policy)
    print("Pure RL training finished. Policy:", MODEL_PATH)
    print(
        "Recorded episode archive:",
        os.path.abspath(EPISODE_TRAJECTORY_PATH)
        if TRAJECTORY_RECORDING_ENABLED else "disabled (AROOD_RECORD_TRAJECTORIES=0)"
    )


if __name__ == "__main__":
    main()








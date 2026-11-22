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
            _speed_for_bound = max(0.0, _speed_before)
            _minimum_acceleration = -_speed_for_bound / _dt
            if _requested_acceleration < _minimum_acceleration:
                _effective_acceleration = _minimum_acceleration
                _event = {
                    "event": "simulator_acceleration_limited_to_prevent_reverse_speed",
                    "time": time.time(),
                    "vehicle_id": id(self),
                    "vehicle_class": type(self).__name__,
                    "lane_index": str(getattr(self, "lane_index", None)),
                    "traffic_side": getattr(self, "_arood_traffic_side", None),
                    "dt_s": _dt,
                    "speed_before_mps": _speed_before,
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

# ============================================================
# 1. Configuration — instantaneous lane-change diagnostic mode
# ============================================================
INSTANT_LANE_CHANGE_MODE = True
steering_scale = 0.5
offroad_penalty = 0.0  # No separate off-road reward penalty

# Instantaneous lane-transition abstraction: one policy decision advances exactly
# one simulator update. There are no intermediate lateral-motion substeps.
policy_frequency = 15
simulation_frequency = 15

lane_keeping_weight = 0.20

# Below SPEED_LIMIT, reward positive measured longitudinal acceleration; at/above it, give full speed reward.
# MAX_PHYSICAL_ACCELERATION is the simulator's physical acceleration limit and reward normalization scale.
# Lane selection is judged separately; TTC/gap stay in the independent safety shield.
SPEED_LIMIT = 50.0
SPEED_LIMIT_MPS = SPEED_LIMIT  # U_Challenge ADT compatibility alias
INITIAL_PROGRESS_WEIGHT = 4.0
INITIAL_COLLISION_PENALTY = 5.0
# Off-road is not a reward-shaping term: the lane-action mask restricts
# policy lane choices to the three valid lanes. A physical off-road state is
# still handled as a defensive episode terminal in the training loops.
INITIAL_OFFROAD_PENALTY = 0.0

# TTC thresholds are used by the safety shield, not reward shaping.
ttc_critical_threshold = 3.0
ttc_danger_threshold = 5.0
ttc_caution_threshold = 7.0

SPEED_MATCH_TTC = 5.0
SPEED_MATCH_GAP_TOLERANCE = 10.0  # metres; avoid needless overrides for tiny gap deviations
# Execution-time safety shield (TTC is measured in seconds).
SAFETY_OVERRIDE_TTC = 2.0       # Existing emergency-intervention threshold (unchanged)
TTC_RISK_CRITICAL = 2.0         # Shared continuous-risk scale: full risk at/below this TTC
TTC_RISK_WARNING = 5.0          # Shared continuous-risk scale: zero risk at/above this TTC
# Candidate-lane TTC thresholds. Keep front TTC at least as conservative as rear TTC.
LANE_CHANGE_MIN_FRONT_TTC = 6.0
LANE_CHANGE_MIN_REAR_TTC = 4.0
assert LANE_CHANGE_MIN_FRONT_TTC >= LANE_CHANGE_MIN_REAR_TTC, (
    "Design constraint violated: minimum front TTC must be >= minimum rear TTC."
)
LANE_CHANGE_PREDICTION_HORIZON = 1.5  # Retained for transition/diagnostic predictions (seconds)
LANE_CHANGE_TTC_MARGIN = 0.5     # Extra rear-TTC margin during the transition
LANE_CHANGE_MIN_GAP = 15.0        # Independent minimum bumper-gap safeguard (metres)

# Maneuver completion uses geometric alignment over consecutive policy decisions.
LANE_CHANGE_COMPLETE_LATERAL_TOL = 0.45   # metres from target-lane centre
LANE_CHANGE_COMPLETE_HEADING_TOL = 0.12   # radians, wrapped heading error
LANE_CHANGE_COMPLETE_STABLE_STEPS = 2     # consecutive checks before completion
# Relaxed fallback is used only after a long maneuver timeout.
LANE_CHANGE_TIMEOUT_LATERAL_TOL = 0.90
LANE_CHANGE_TIMEOUT_HEADING_TOL = 0.20
LANE_CHANGE_RECOVERY_COOLDOWN_STEPS = 2
LANE_CHANGE_ACTIVE_ABORT_TTC = 2.0
LANE_CHANGE_ACTIVE_ABORT_GAP = 6.0

SAFETY_OVERRIDE_REWARD_PENALTY = 1.0  # Applied only for emergency TTC braking, not routine speed matching
LANE_REJECTION_REWARD_PENALTY = 0.05  # Small signal when an unsafe lane-change request is rejected
SHIELD_ACCEL_INTERVENTION_WEIGHT = 0.15  # Modest cost for routine acceleration overrides

# Traffic scenario diversity for training.
# Randomized traffic gaps create varied lane-change opportunities.

REFERENCE_GAP_MIN = 20.0
REFERENCE_GAP_MAX = 500.0
RANDOMIZED_FRONT_GAP_MIN = 50.0
RANDOMIZED_FRONT_GAP_MAX = 500.0
RANDOMIZED_REAR_GAP_MIN = 20.0
RANDOMIZED_REAR_GAP_MAX = 500.0

# A lane change is beneficial only when its expected combined utility improves
# meaningfully over staying in the current lane across a physical-time horizon.
# The weights are explicit experiment parameters and sum to one.
BENEFIT_PREDICTION_HORIZON_SECONDS = 5.0
BENEFIT_PREDICTION_SAMPLES = 11
BENEFIT_SAFETY_WEIGHT = 0.40
BENEFIT_EFFICIENCY_WEIGHT = 0.40
BENEFIT_STABILITY_WEIGHT = 0.20
BENEFIT_MIN_UTILITY_IMPROVEMENT = 0.1
BENEFIT_SAFETY_GAP_REFERENCE = 50.0



# Aggressive Dynamic Traffic (ADT) simulator configuration.
# The simulator maintains a distributed background population; the policy
# observation remains exactly six surrounding slots (front/rear in each lane).
TRAFFIC_TARGET_COUNT = 12
REFERENCE_RANGE_M = 500.0
RECYCLING_BOUNDARY_M = 700.0
INITIAL_PLACEMENT_RANGE_M = 350.0
MIN_INITIAL_GAP_M = 50.0
MIN_INSERTION_GAP_M = 35.0

FRONT_COVERAGE_MAX_DISTANCE_M = 450.0
FRONT_COVERAGE_SPAWN_MIN_DISTANCE_M = 180.0
FRONT_COVERAGE_SPAWN_MAX_DISTANCE_M = 420.0
FRONT_COVERAGE_SPEED_OFFSET_MIN_MPS = 2.0
FRONT_COVERAGE_SPEED_OFFSET_MAX_MPS = 7.0
TRAFFIC_MIN_SPEED_MPS = 10.0
TRAFFIC_MAX_SPEED_MPS = 50.0

IDM_MAX_ACCEL_MPS2 = 2.0
IDM_COMFORTABLE_BRAKING_MPS2 = 3.5
IDM_MIN_GAP_M = 5.0
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
ADT_AGGRESSIVE_MAX_ACCEL_MPS2 = 3.2
ADT_AGGRESSIVE_BRAKING_MPS2 = 5.5
ADT_OVERTAKER_LANE_CHANGE_RATE_PER_S = 0.75
ADT_UNSTABLE_LANE_CHANGE_RATE_PER_S = 0.35
ADT_LANE_CHANGE_COOLDOWN_S = 1.2
ADT_LANE_CHANGE_FRONT_GAP_M = 7.0
ADT_LANE_CHANGE_REAR_GAP_M = 7.0
ADT_LANE_CHANGE_FRONT_TTC_S = 1.0
ADT_LANE_CHANGE_REAR_TTC_S = 0.9
ADT_LANE_CHANGE_IMPROVEMENT_MPS = 0.8
ADT_LANE_CHANGE_TRIGGER_GAP_M = 55.0
ADT_SUDDEN_BRAKE_EVENT_RATE_PER_S = 0.12
ADT_SUDDEN_BRAKE_ACCEL_MPS2 = -5.0
ADT_SUDDEN_BRAKE_DURATION_S = 0.8
ADT_UNSTABLE_ACCEL_EVENT_RATE_PER_S = 0.10
ADT_UNSTABLE_ACCEL_PULSE_MPS2 = 3.0
ADT_SPEED_VARIATION_MPS = 9.0

PROFILE_DESIRED_SPEED_RANGES_MPS = {
    "normal": (28.0, 42.0), "aggressive": (36.0, 50.0),
    "unstable": (24.0, 49.0), "slow": (16.0, 29.0),
}
PROFILE_IDM_HEADWAY_S = {
    "normal": 1.5, "aggressive": 0.70, "unstable": 0.85, "slow": 1.8,
}
PROFILE_IDM_MIN_GAP_M = {
    "normal": 5.0, "aggressive": 3.0, "unstable": 3.5, "slow": 6.0,
}
PROFILE_LANE_CHANGE_RATE_PER_S = {
    "normal": 0.0, "aggressive": ADT_OVERTAKER_LANE_CHANGE_RATE_PER_S,
    "unstable": ADT_UNSTABLE_LANE_CHANGE_RATE_PER_S, "slow": 0.0,
}

INITIAL_SPAWN_MAX_ATTEMPTS = 10000
LANE_COVERAGE_SPAWN_MAX_ATTEMPTS = 80
BOUNDARY_SPAWN_MAX_ATTEMPTS = 60
BOUNDARY_SPAWN_SPEED_OFFSET_MIN_MPS = 2.0
BOUNDARY_SPAWN_SPEED_OFFSET_MAX_MPS = 8.0
BOUNDARY_SPAWN_DISTANCE_BUFFER_M = 10.0
BOUNDARY_SPAWN_EXTRA_RANGE_M = 100.0

Traffic_relevant_range = REFERENCE_RANGE_M
TRAFFIC_SPEED_MIN = TRAFFIC_MIN_SPEED_MPS
TRAFFIC_SPEED_MAX = TRAFFIC_MAX_SPEED_MPS

# Legacy helper compatibility only; the active simulator now uses ADT profiles.
TRAFFIC_SPEED_MATCHED_PROB = 0.60
TRAFFIC_SPEED_POSITION_CHALLENGE_PROB = 0.25
TRAFFIC_SPEED_MATCHED_DELTA_MIN = -3.0
TRAFFIC_SPEED_MATCHED_DELTA_MAX = 3.0
TRAFFIC_FRONT_SLOW_DELTA_MIN = -8.0
TRAFFIC_FRONT_SLOW_DELTA_MAX = -3.0
TRAFFIC_REAR_FAST_DELTA_MIN = 3.0
TRAFFIC_REAR_FAST_DELTA_MAX = 6.0
TRAFFIC_SPEED_BROAD_DELTA_MIN = -8.0
TRAFFIC_SPEED_BROAD_DELTA_MAX = 6.0
LANE_CHANGE_COST = 0.0
LANE_CHANGE_COOLDOWN_STEPS = 0

SIMULATION_FREQUENCY_HZ = 15
POLICY_FREQUENCY_HZ = 15
simulation_frequency = SIMULATION_FREQUENCY_HZ
policy_frequency = POLICY_FREQUENCY_HZ

# Speed-matching controller
DESIRED_TIME_HEADWAY = 3   # seconds
MIN_FOLLOWING_GAP = 10.0      # metres
SPEED_MATCH_KV = 0.5         # 1/second
SPEED_MATCH_KD = 0.08        # 1/second^2

MAX_PHYSICAL_ACCELERATION = 5.0  # m/s^2; matches ContinuousAction acceleration_range [-5, 5].
POLICY_STEP_DURATION_S = 1.0 / float(policy_frequency)
MAX_ACCELERATION_PER_STEP_MPS = MAX_PHYSICAL_ACCELERATION * POLICY_STEP_DURATION_S


NUM_LANES = 3
NUM_SURROUNDING_VEHICLES = 6
VEHICLE_FEATURES = 7
NUM_VEHICLE_ROWS = NUM_SURROUNDING_VEHICLES + 1
VEHICLE_STATE_DIM = NUM_VEHICLE_ROWS * VEHICLE_FEATURES
CONTEXT_DIM = 5
# Per lane: six physical measurements plus front/rear EDM risk scores and
# normalized categorical severity codes. These are observations only.
EDM_TTC_COMFORTABLE_S = 6.0
EDM_TTC_DANGER_S = 4.0
EDM_TTC_CRITICAL_S = 2.0
# A transition is counted as high-risk exposure when the pre-action lane EDM
# risk score reaches this threshold. This is independent of category labels.
EDM_HIGH_RISK_SCORE_THRESHOLD = 0.5
EDM_MIN_GAP_REFERENCE_M = 15.0
EDM_FEATURES_PER_LANE = 4
SAFETY_FEATURES_PER_LANE = 6 + EDM_FEATURES_PER_LANE
SAFETY_FEATURES_DIM = NUM_LANES * SAFETY_FEATURES_PER_LANE
STATE_DIM = VEHICLE_STATE_DIM + CONTEXT_DIM + SAFETY_FEATURES_DIM
ACTION_DIM = 2  # execution action [normalized learned acceleration, binary KEEP/CHANGE]
POLICY_ACTION_DIM = 2  # binary KEEP, CHANGE decision; acceleration is a conditional continuous head

HIDDEN_DIM = 128


# Cold-start baseline: no reference rollouts or offline warm-start are used.
NUM_REFERENCE_EPISODES = 0
REFERENCE_TRAINING_EPOCHS = 0
REFERENCE_BATCH_SIZE = 256
NUM_ONLINE_EPISODES = 50

MAX_STEPS = 500


TTC_PROXY_THRESHOLD = 0.2


POLICY_LR = 3e-4


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Structured diagnostics: one JSON object per line.
DIAGNOSTIC_LOG_PATH = os.path.join("I_hierarchical_output", "training_diagnostics_P.jsonl")

# Created only when a managed traffic-slot vehicle first has negative speed.
# Keep this distinct from any tracked README/placeholder so exclusive creation works.
NEGATIVE_SPEED_FIRST_TRANSITION_PATH = os.path.join("P_hierarchical_output", "negative_speed_first_transition.txt")

# Episode-level evaluator settings. The placeholder is deterministic and does not
# call a network or pretend to be a real LLM.
EVALUATOR_BATCH_SIZE = 10
EVALUATOR_MIN_EPISODES = 10
EVALUATOR_MAX_SAFETY_WEIGHT = 3.0
EVALUATOR_MIN_SAFETY_WEIGHT = 0.5
# Lane-change frequency is monitored diagnostically. A tiny cost is applied
# only when a lane change completes; rejected requests are not charged this cost.


reward_path = os.path.join("P_hierarchical_output", "training_return.png")
#reward_per_step_path = 'figure/Re_E_new_per_step_' + '2.png'
plot_path = os.path.join("P_hierarchical_output", "training_summary.txt")
MODEL_PATH = os.path.join("policy", "P_hierarchical_actor_critic.pt")
EPISODE_TRAJECTORY_PATH = os.path.join("P_hierarchical_output", "episode_trajectories_P.jsonl")
TRAJECTORY_RECORDING_ENABLED = os.environ.get("AROOD_RECORD_TRAJECTORIES_P", "1").strip().lower() not in {"0", "false", "no", "off"}

# Consolidated I experiment outputs and agreed policy-processing mechanism.
OUTPUT_DIR = os.path.join("P_hierarchical_output")
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "training_metrics_P.csv")
METRICS_JSONL_PATH = os.path.join(OUTPUT_DIR, "training_metrics_P.jsonl")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "experiment_config_P.json")
LANE_CHANGE_PROCESSING_STEPS = 0  # disabled: every transition receives its normal reward
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
        "policy_frequency_hz": int(policy_frequency),
        "simulation_frequency_hz": int(simulation_frequency),
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
    """Map TTC in seconds to a continuous [0, 1] risk score.

    This is a diagnostic signal only in this revision: it does not modify
    reward construction, policy actions, or existing safety thresholds.
    Missing/non-closing traffic (TTC=inf) maps to zero risk; invalid NaN
    values conservatively map to full risk.
    """
    try:
        ttc = float(ttc_seconds)
    except (TypeError, ValueError):
        return 1.0
    if np.isnan(ttc):
        return 1.0
    if ttc <= TTC_RISK_CRITICAL:
        return 1.0
    if ttc >= TTC_RISK_WARNING or np.isinf(ttc):
        return 0.0
    return float(np.clip(
        (TTC_RISK_WARNING - ttc) / (TTC_RISK_WARNING - TTC_RISK_CRITICAL),
        0.0, 1.0,
    ))


def edm_ttc_risk_score(ttc_seconds):
    """Continuous descriptive risk score: 0 at 6s, 1 at 0s or below."""
    try:
        ttc = float(ttc_seconds)
    except (TypeError, ValueError):
        return 1.0
    if np.isnan(ttc):
        return 1.0
    if np.isinf(ttc):
        return 0.0
    return float(np.clip(
        (EDM_TTC_COMFORTABLE_S - ttc) / EDM_TTC_COMFORTABLE_S, 0.0, 1.0
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
    elif ttc < EDM_TTC_DANGER_S:
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


class HighwaySimulator:
    """Highway-env wrapper with six managed traffic vehicles.

    State includes vehicle rows, ego-relative context, six physical lane
    metrics plus four EDM interpretation features for each of the roles
    [current, left, right]. State dimension is derived from configuration.
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 max_steps=MAX_STEPS, render_mode=None, vehicles_count=TRAFFIC_TARGET_COUNT):
        if gym is None or highway_env is None:
            raise ImportError(
                "Highway-env dependencies are missing. Install with: "
                "pip install highway-env gymnasium"
            )
        if state_dim != STATE_DIM or action_dim != ACTION_DIM:
            raise ValueError(f"HighwaySimulator expects STATE_DIM={STATE_DIM} and execution ACTION_DIM={ACTION_DIM}")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0
        self.render_mode = render_mode
        self.vehicles_count = TRAFFIC_TARGET_COUNT
        self.traffic_front_gap = 45.0
        self.traffic_rear_gap = 45.0
        self.randomize_traffic_gaps = False
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

                "lanes_count": 3,
                "vehicles_count": TRAFFIC_TARGET_COUNT,
                "duration": max_steps,
                "offroad_terminal": False,

                # Instantaneous lane transitions are applied by the adapter before
                # this single environment update. Keep both frequencies equal so
                # one policy decision maps to exactly one simulator update.
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




    def reset(self, seed=None, randomize_traffic_gaps=True):

        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)

        # ADT uses a distributed population and continuous boundary recycling;
        # randomize_traffic_gaps is retained only for call compatibility.
        self.randomize_traffic_gaps = False
        self.traffic_front_gap = float("nan")
        self.traffic_rear_gap = float("nan")
        print(
            "[ADT TRAFFIC SCENARIO]",
            f"target_population={TRAFFIC_TARGET_COUNT}",
            f"initial_placement=±{INITIAL_PLACEMENT_RANGE_M:.0f}m",
            f"reference_range=±{REFERENCE_RANGE_M:.0f}m",
            f"recycling_boundary=±{RECYCLING_BOUNDARY_M:.0f}m",
            f"speed_range={TRAFFIC_MIN_SPEED_MPS:.0f}-{TRAFFIC_MAX_SPEED_MPS:.0f}m/s",
            f"profile_mix={dict(ADT_PROFILE_MIX)}",
            f"simulation_frequency={SIMULATION_FREQUENCY_HZ}Hz",
            "ego_observation_slots=6",
        )

        self.step_count = 0
        # One collision event is recorded per uninterrupted contact episode.
        # The simulator crash latch is still cleared after each update so the
        # RL episode can continue, but sustained overlap is not re-counted as
        # a fresh impact on every subsequent policy step.
        self._collision_contact_active = False
        base_env = self.env.unwrapped
        self._initialize_six_vehicle_traffic()
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
        self._assert_traffic_invariant("reset")
        return self.state.copy()




    # All vehicles are projected onto the same reference lane coordinate.
    # In highway-v0, the three lanes share the same road segment and longitudinal
    # axis; using one reference lane avoids comparing independently projected s
    # values from different lane geometries.
    TRAFFIC_RELEVANT_RANGE = Traffic_relevant_range  # metres ahead/behind; farther vehicles are recycled
    TRAFFIC_CROSSING_EPSILON = 0.5  # metres; avoids unstable front/rear classification at ego

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

    def _sample_surrounding_speed(self, ego_speed, side):
        """Sample traffic speed relative to ego, with position-aware diversity.

        Mixture:
          60% speed-matched: delta-v in [-3, +3] m/s;
          25% position-specific challenge: slower front or faster rear;
          15% broad diversity: delta-v in [-8, +6] m/s.
        The final absolute speed is clipped to simulator bounds.
        """
        ego_speed = float(ego_speed)
        draw = random.random()

        if draw < TRAFFIC_SPEED_MATCHED_PROB:
            delta_v = random.uniform(
                TRAFFIC_SPEED_MATCHED_DELTA_MIN,
                TRAFFIC_SPEED_MATCHED_DELTA_MAX,
            )
            regime = "matched"
        elif draw < TRAFFIC_SPEED_MATCHED_PROB + TRAFFIC_SPEED_POSITION_CHALLENGE_PROB:
            if side == "front":
                delta_v = random.uniform(
                    TRAFFIC_FRONT_SLOW_DELTA_MIN,
                    TRAFFIC_FRONT_SLOW_DELTA_MAX,
                )
            else:
                delta_v = random.uniform(
                    TRAFFIC_REAR_FAST_DELTA_MIN,
                    TRAFFIC_REAR_FAST_DELTA_MAX,
                )
            regime = "position_challenge"
        else:
            delta_v = random.uniform(
                TRAFFIC_SPEED_BROAD_DELTA_MIN,
                TRAFFIC_SPEED_BROAD_DELTA_MAX,
            )
            regime = "broad"

        requested_speed = ego_speed + delta_v
        speed = float(np.clip(
            requested_speed, TRAFFIC_SPEED_MIN, TRAFFIC_SPEED_MAX
        ))
        return speed, float(delta_v), regime

    def _spawn_slot_vehicle(self, lane_id, side):
        """Spawn a replacement at the configured gap with ego-relative speed."""
        base_env = self.env.unwrapped
        ego = base_env.vehicle
        lane_index = self._lane_index_for_id(lane_id)
        # Compute s in the target lane's own parameterization for placement;
        # classification and distance comparisons use the shared road frame.
        ego_s_target_lane = self._lane_longitudinal_position(ego, lane_index)
        # Keep the existing independent per-lane/per-side gap randomization.
        if self.randomize_traffic_gaps:
            if side == "front":
                gap = random.uniform(
                    RANDOMIZED_FRONT_GAP_MIN, RANDOMIZED_FRONT_GAP_MAX)
            else:
                gap = random.uniform(
                    RANDOMIZED_REAR_GAP_MIN, RANDOMIZED_REAR_GAP_MAX)
        else:
            gap = self.traffic_front_gap if side == "front" else self.traffic_rear_gap
        longitudinal = ego_s_target_lane + gap if side == "front" else ego_s_target_lane - gap

        # Generate speed from ego speed rather than independently sampling
        # the full [TRAFFIC_SPEED_MIN, TRAFFIC_SPEED_MAX] range. Do not apply
        # the old front/rear pairwise speed constraint: it biased the intended
        # ego-relative mixture and could override position-aware sampling.
        speed, delta_v, speed_regime = self._sample_surrounding_speed(
            ego_speed=float(ego.speed), side=side
        )
        traffic_vehicle = Vehicle.make_on_lane(
            base_env.road, lane_index, longitudinal=longitudinal, speed=speed
        )
        traffic_vehicle._arood_traffic_side = side
        traffic_vehicle._arood_speed_delta_from_ego = delta_v
        traffic_vehicle._arood_speed_regime = speed_regime
        base_env.road.vehicles.append(traffic_vehicle)
        print(
            "[TRAFFIC SPAWN]",
            f"lane={lane_id}", f"side={side}",
            f"gap={gap:.1f}m", f"range={self.TRAFFIC_RELEVANT_RANGE:.1f}m",
            f"ego_speed={float(ego.speed):.2f}m/s",
            f"speed={speed:.2f}m/s", f"delta_v_sample={delta_v:+.2f}m/s",
            f"regime={speed_regime}",
            f"id={id(traffic_vehicle)}",
        )
        return traffic_vehicle

    def _initialize_six_vehicle_traffic(self):
        """Start with exactly one front and one rear vehicle in each lane."""
        base_env = self.env.unwrapped
        ego = base_env.vehicle
        base_env.road.vehicles = [vehicle for vehicle in base_env.road.vehicles if vehicle is ego]
        for lane_id in range(NUM_LANES):
            self._spawn_slot_vehicle(lane_id, "front")
            self._spawn_slot_vehicle(lane_id, "rear")
        self._assert_traffic_invariant("initialized")

    def _maintain_six_vehicle_traffic(self):
        """Maintain nearest in-range front/rear neighbors in every lane.

        A vehicle qualifies only when its lane is valid, it is within the
        longitudinal relevance range, and it is strictly on one side of ego.
        When it crosses ego, leaves range, changes into an invalid lane, or
        becomes a duplicate in a slot, it is removed/reclassified and the
        missing slot is immediately replenished.
        """
        road = self.env.unwrapped.road
        ego = self.env.unwrapped.vehicle
        ego_s = self._road_longitudinal_position(ego)
        slots = {(lane, side): [] for lane in range(NUM_LANES) for side in ("front", "rear")}
        valid_vehicle_ids = set()

        for vehicle in list(road.vehicles):
            if vehicle is ego:
                continue
            try:
                lane_id = int(vehicle.lane_index[2])
                if lane_id not in range(NUM_LANES):
                    continue
                longitudinal = self._road_longitudinal_position(vehicle)
                signed_gap = longitudinal - ego_s
            except (AttributeError, TypeError, ValueError, IndexError):
                continue

            distance = abs(signed_gap)
            if distance > self.TRAFFIC_RELEVANT_RANGE:
                continue
            if abs(signed_gap) <= self.TRAFFIC_CROSSING_EPSILON:
                # Avoid assigning a nearly overlapping vehicle inconsistently
                # to front/rear from tiny numerical sign changes.
                continue

            side = "front" if signed_gap > 0 else "rear"
            slots[(lane_id, side)].append((distance, vehicle))
            valid_vehicle_ids.add(id(vehicle))

        keep_ids = set()
        for slot_key, candidates in slots.items():
            candidates.sort(key=lambda item: item[0])
            if candidates:
                nearest = candidates[0][1]
                keep_ids.add(id(nearest))
                if len(candidates) > 1:
                    print("[TRAFFIC DUPLICATE SLOT]", f"slot={slot_key}",
                          f"removed={len(candidates) - 1}")

        # Remove vehicles that crossed ego, left the relevant range, changed
        # to an unsupported lane, or lost a nearest-neighbor slot.
        road.vehicles = [
            vehicle for vehicle in road.vehicles
            if vehicle is ego or id(vehicle) in keep_ids
        ]

        # Replenish each missing slot immediately. Each spawn is placed using
        # the target lane's longitudinal coordinate, while slot selection uses
        # the shared road-coordinate frame above.
        for lane_id in range(NUM_LANES):
            for side in ("front", "rear"):
                if not slots[(lane_id, side)]:
                    self._spawn_slot_vehicle(lane_id, side)

        self._assert_traffic_invariant("maintained")

    def _assert_traffic_invariant(self, where):
        ego = self.env.unwrapped.vehicle
        ego_s = self._road_longitudinal_position(ego)
        counts = {(lane, side): 0 for lane in range(NUM_LANES) for side in ("front", "rear")}
        traffic_count = 0
        for vehicle in list(self.env.unwrapped.road.vehicles):
            if vehicle is ego:
                continue
            traffic_count += 1
            try:
                lane_id = int(vehicle.lane_index[2])
                signed_gap = self._road_longitudinal_position(vehicle) - ego_s
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
            if lane_id in range(NUM_LANES) and abs(signed_gap) > self.TRAFFIC_CROSSING_EPSILON:
                if abs(signed_gap) <= self.TRAFFIC_RELEVANT_RANGE:
                    counts[(lane_id, "front" if signed_gap > 0 else "rear")] += 1
        expected_slots = NUM_LANES * 2
        occupied_slots = sum(count > 0 for count in counts.values())
        invariant_ok = (
            traffic_count == expected_slots
            and occupied_slots == expected_slots
            and all(count == 1 for count in counts.values())
        )
        if not invariant_ok:
            print("[TRAFFIC INVARIANT WARNING] Expected exactly one in-range vehicle in every lane/side slot.")

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

    def _lane_change_principle_reasons(self, candidate_lane, lane_ttc, current_lane=None, ego_speed=None):
        """Return blocking reasons for a candidate lane change.

        Core target-lane safety rules:
          1) Candidate front TTC must meet LANE_CHANGE_MIN_FRONT_TTC.
          2) Candidate rear TTC must meet LANE_CHANGE_MIN_REAR_TTC.
          3) Front TTC threshold is constrained to be >= rear TTC threshold.
          4) Front/rear bumper gaps must satisfy the independent minimum-gap
             safeguard whenever a corresponding vehicle exists.

        TTC is calculated from bumper gap and positive closing speed. If a
        vehicle is not closing on the ego, its TTC is infinite. An absent
        vehicle is also treated as infinite TTC, but its gap is not fabricated.
        """
        base_env = self.env.unwrapped
        ego = base_env.vehicle
        current_lane = int(ego.lane_index[2] if current_lane is None else current_lane)
        ego_speed = max(0.0, float(ego.speed if ego_speed is None else ego_speed))

        if candidate_lane not in lane_ttc:
            print("[MASK DEBUG][INVALID CANDIDATE]",
                  f"candidate_lane={candidate_lane}",
                  f"available_lane_ids={sorted(lane_ttc.keys())}")
            return ["target_lane_traffic_data_unavailable"]

        if LANE_CHANGE_MIN_FRONT_TTC < LANE_CHANGE_MIN_REAR_TTC:
            # Guard at the decision point as well as in the configuration.
            raise ValueError("Minimum front TTC must be >= minimum rear TTC.")

        values = lane_ttc[candidate_lane]
        front_ttc = float(values.get("front_ttc", float("inf")))
        rear_ttc = float(values.get("rear_ttc", float("inf")))
        front_gap = float(values.get("front_gap", float("inf")))
        rear_gap = float(values.get("rear_gap", float("inf")))
        front_speed = values.get("front_speed")
        rear_speed = values.get("rear_speed")

        reasons = []
        if front_ttc < LANE_CHANGE_MIN_FRONT_TTC:
            reasons.append("candidate_front_ttc_below_threshold")
        if rear_ttc < LANE_CHANGE_MIN_REAR_TTC:
            reasons.append("candidate_rear_ttc_below_threshold")
        if values.get("front_vehicle_id") is not None and front_gap < LANE_CHANGE_MIN_GAP:
            reasons.append("candidate_front_gap_below_minimum")
        if values.get("rear_vehicle_id") is not None and rear_gap < LANE_CHANGE_MIN_GAP:
            reasons.append("candidate_rear_gap_below_minimum")

        return reasons

    def _lane_benefit_utility(self, lane_values):
        """Estimate safety, efficiency, and stability over a physical-time horizon.

        Traffic vehicles are assumed to maintain constant speed during this
        short-horizon estimate. This is a lane-ranking signal, not a safety
        shield: immediate TTC and minimum-gap constraints are checked separately.
        """
        ego = self.env.unwrapped.vehicle
        ego_speed = max(0.0, float(ego.speed))
        horizon = float(BENEFIT_PREDICTION_HORIZON_SECONDS)
        sample_count = max(2, int(BENEFIT_PREDICTION_SAMPLES))
        times = np.linspace(0.0, horizon, sample_count)

        front_speed_raw = lane_values.get("front_speed")
        rear_speed_raw = lane_values.get("rear_speed")
        front_gap = float(lane_values.get("front_gap", float("inf")))
        rear_gap = float(lane_values.get("rear_gap", float("inf")))
        front_present = front_speed_raw is not None and np.isfinite(front_gap)
        rear_present = rear_speed_raw is not None and np.isfinite(rear_gap)
        front_speed = float(front_speed_raw) if front_present else SPEED_LIMIT
        rear_speed = float(rear_speed_raw) if rear_present else ego_speed

        safety_series = []
        efficiency_series = []
        gap_reference = max(1.0, float(BENEFIT_SAFETY_GAP_REFERENCE))
        headway = max(0.1, float(DESIRED_TIME_HEADWAY))
        following_gap = max(0.0, float(MIN_FOLLOWING_GAP))
        speed_reference = max(1.0, float(SPEED_LIMIT))

        for elapsed in times:
            predicted_front_gap = (
                max(0.0, front_gap + (front_speed - ego_speed) * float(elapsed))
                if front_present else float("inf")
            )
            predicted_rear_gap = (
                max(0.0, rear_gap + (ego_speed - rear_speed) * float(elapsed))
                if rear_present else float("inf")
            )

            front_safety = (
                float(np.clip(predicted_front_gap / gap_reference, 0.0, 1.0))
                if front_present else 1.0
            )
            rear_safety = (
                float(np.clip(predicted_rear_gap / gap_reference, 0.0, 1.0))
                if rear_present else 1.0
            )
            safety_series.append(min(front_safety, rear_safety))

            # Estimate the speed that can be sustained behind the front vehicle
            # while preserving the desired time headway.
            if front_present:
                sustainable_speed = front_speed + max(
                    0.0, predicted_front_gap - following_gap
                ) / headway
            else:
                sustainable_speed = speed_reference
            efficiency_series.append(
                float(np.clip(sustainable_speed / speed_reference, 0.0, 1.0))
            )

        safety_score = float(np.mean(safety_series))
        efficiency_score = float(np.mean(efficiency_series))

        # Stability rewards a lane whose predicted safety/efficiency profile
        # does not fluctuate sharply over the evaluation horizon.
        profile_variation = (
            float(np.std(safety_series)) + float(np.std(efficiency_series))
        ) / 2.0
        stability_score = float(np.clip(1.0 - profile_variation, 0.0, 1.0))

        utility = (
            BENEFIT_SAFETY_WEIGHT * safety_score
            + BENEFIT_EFFICIENCY_WEIGHT * efficiency_score
            + BENEFIT_STABILITY_WEIGHT * stability_score
        )
        return {
            "utility": float(utility),
            "safety_score": safety_score,
            "efficiency_score": efficiency_score,
            "stability_score": stability_score,
            "prediction_horizon_s": horizon,
        }

    def _safe_lane_change_candidates(self, lane_ttc=None):
        """Return adjacent lanes that are safe and meaningfully beneficial.

        TTC and minimum-gap thresholds determine admissibility only. They do not
        rank lanes or require an adjacent lane to have a larger front TTC than
        the current lane. If no adjacent lane is safe, return the current lane
        as a compatibility fallback; target-lane masks exclude it for CHANGE.
        """
        base_env = self.env.unwrapped
        vehicle = base_env.vehicle
        if lane_ttc is None:
            lane_ttc = self._min_ttc_by_lane()
        start_node, end_node, current_lane = vehicle.lane_index
        available_lanes = base_env.road.network.graph[start_node][end_node]
        valid_ids = set(range(len(available_lanes)))
        current_lane = int(current_lane)
        current_values = lane_ttc.get(current_lane, {})
        current_profile = self._lane_benefit_utility(current_values)
        safe_adjacent = []

        # A real current-lane rear emergency can bypass the benefit threshold,
        # but never the hard safety checks applied to the destination lane.
        rear_assessment = self._rear_escape_assessment(lane_ttc)
        current_front_ttc = float(current_values.get("front_ttc", float("inf")))
        current_front_gap = float(current_values.get("front_gap", float("inf")))
        current_front_emergency = (
            current_front_ttc <= SAFETY_OVERRIDE_TTC
            or current_front_gap <= MIN_FOLLOWING_GAP
        )
        emergency_benefit_override = bool(
            rear_assessment.get("rear_escape_needed", False)
            or current_front_emergency
        )

        for candidate_lane in (current_lane - 1, current_lane + 1):
            if candidate_lane not in valid_ids or candidate_lane not in lane_ttc:
                continue
            reasons = self._lane_change_principle_reasons(
                candidate_lane, lane_ttc, current_lane=current_lane,
                ego_speed=float(vehicle.speed),
            )
            if reasons:
                continue

            values = lane_ttc[candidate_lane]
            candidate_profile = self._lane_benefit_utility(values)
            utility_improvement = (
                float(candidate_profile["utility"])
                - float(current_profile["utility"])
            )

            if (
                not emergency_benefit_override
                and utility_improvement < BENEFIT_MIN_UTILITY_IMPROVEMENT
            ):
                continue

            safe_adjacent.append({
                "target_lane": int(candidate_lane),
                "utility": float(utility_improvement),
                "candidate_utility": float(candidate_profile["utility"]),
                "current_lane_utility": float(current_profile["utility"]),
                "utility_improvement": float(utility_improvement),
                "candidate_safety_score": float(candidate_profile["safety_score"]),
                "candidate_efficiency_score": float(candidate_profile["efficiency_score"]),
                "candidate_stability_score": float(candidate_profile["stability_score"]),
                "current_safety_score": float(current_profile["safety_score"]),
                "current_efficiency_score": float(current_profile["efficiency_score"]),
                "current_stability_score": float(current_profile["stability_score"]),
                "prediction_horizon_s": float(BENEFIT_PREDICTION_HORIZON_SECONDS),
                "benefit_threshold_bypassed_for_emergency": emergency_benefit_override,
                "front_gap": float(values.get("front_gap", float("inf"))),
                "rear_gap": float(values.get("rear_gap", float("inf"))),
                "front_ttc": float(values.get("front_ttc", float("inf"))),
                "rear_ttc": float(values.get("rear_ttc", float("inf"))),
                "front_speed": values.get("front_speed"),
                "rear_speed": values.get("rear_speed"),
                "selection_principle": (
                    "emergency_safe_lane_selection"
                    if emergency_benefit_override
                    else "persistent_safety_efficiency_stability_improvement"
                ),
                "is_current_lane": False,
            })

        if safe_adjacent:
            selected_candidates = safe_adjacent
        else:
            selected_candidates = [{
                "target_lane": current_lane,
                "utility": 0.0,
                "candidate_utility": float(current_profile["utility"]),
                "current_lane_utility": float(current_profile["utility"]),
                "utility_improvement": 0.0,
                "candidate_safety_score": float(current_profile["safety_score"]),
                "candidate_efficiency_score": float(current_profile["efficiency_score"]),
                "candidate_stability_score": float(current_profile["stability_score"]),
                "current_safety_score": float(current_profile["safety_score"]),
                "current_efficiency_score": float(current_profile["efficiency_score"]),
                "current_stability_score": float(current_profile["stability_score"]),
                "prediction_horizon_s": float(BENEFIT_PREDICTION_HORIZON_SECONDS),
                "benefit_threshold_bypassed_for_emergency": False,
                "front_gap": float(current_values.get("front_gap", float("inf"))),
                "rear_gap": float(current_values.get("rear_gap", float("inf"))),
                "front_ttc": float(current_values.get("front_ttc", float("inf"))),
                "rear_ttc": float(current_values.get("rear_ttc", float("inf"))),
                "front_speed": current_values.get("front_speed"),
                "rear_speed": current_values.get("rear_speed"),
                "selection_principle": "current_lane_fallback_no_beneficial_safe_adjacent",
                "is_current_lane": True,
            }]

        return selected_candidates

    def _select_best_safe_lane_candidate(self, lane_ttc=None):
        """Choose the highest-utility safe target lane, or None if none is safe."""
        candidates = self._safe_lane_change_candidates(lane_ttc)
        if not candidates:
            return None
        selected = max(candidates, key=lambda item: (item["utility"], -item["target_lane"]))
        return selected

    def get_valid_lane_action_mask(self):
        """Two-action mask: [KEEP, CHANGE]; CHANGE is valid if any safe target exists."""
        vehicle = self.env.unwrapped.vehicle
        if self._lane_change_active:
            mask = np.array([True, False], dtype=bool)
            return mask

        lane_ttc = self._min_ttc_by_lane()
        candidates = self._safe_lane_change_candidates(lane_ttc)
        safe_adjacent = [
            item for item in candidates
            if int(item["target_lane"]) != int(vehicle.lane_index[2])
        ]
        mask = np.array([True, bool(safe_adjacent)], dtype=bool)
        return mask

    def _min_ttc_by_lane(self):
        """Return the nearest front/rear vehicle and TTC for every lane.

        TTC is based on signed bumper-to-bumper clearance and positive closing
        speed. A present vehicle whose body overlaps the ego has TTC=0 even if
        the relative speed is currently non-closing. A missing or non-closing
        vehicle has TTC=inf. This method is diagnostic/state information only:
        it does not veto, replace, or modify the RL agent's action.

        The same snapshot is consumed by EDM so that TTC and lane utility do
        not independently select different neighbours or use different gaps.
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

        occupied_slots = sum(
            lane_ttc[lane_id][f"{side}_vehicle_id"] is not None
            for lane_id in range(lanes_count)
            for side in ("front", "rear")
        )
        if occupied_slots != NUM_SURROUNDING_VEHICLES:
            write_diagnostic_event(
                "traffic_slot_snapshot_incomplete", self,
                occupied_slots=int(occupied_slots),
                expected_slots=NUM_SURROUNDING_VEHICLES,
                lane_slots={
                    str(lane_id): {
                        "front_vehicle_id": lane_ttc[lane_id]["front_vehicle_id"],
                        "rear_vehicle_id": lane_ttc[lane_id]["rear_vehicle_id"],
                    }
                    for lane_id in range(lanes_count)
                },
            )
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

    def _speed_matching_acceleration(self, front_speed, front_gap):
        """
        Return normalized acceleration in [-1, 1].
        Uses both relative speed and desired following distance.
        """
        ego = self.env.unwrapped.vehicle
        ego_speed = max(0.0, float(ego.speed))
        front_speed = max(0.0, float(front_speed))
        front_gap = max(0.0, float(front_gap))

        desired_gap = MIN_FOLLOWING_GAP + DESIRED_TIME_HEADWAY * ego_speed

        # Positive relative speed means the front vehicle is faster.
        speed_error = front_speed - ego_speed
        gap_error = front_gap - desired_gap

        # Physical acceleration in m/s^2.
        physical_acceleration = SPEED_MATCH_KV * speed_error + SPEED_MATCH_KD * gap_error

        physical_acceleration = float(np.clip(physical_acceleration, -MAX_PHYSICAL_ACCELERATION, MAX_PHYSICAL_ACCELERATION,))

        # Highway ContinuousAction expects normalized acceleration.
        normalized_acceleration = physical_acceleration / MAX_PHYSICAL_ACCELERATION

        return float(np.clip(normalized_acceleration, -1.0, 1.0))





    def _legacy_filter_ttc_action(self, sampled_action):
        """
        Apply TTC-based emergency braking and lane-change safety checks.
        Hybrid policy representation:
            action[0] = normalized acceleration in [-1, 1]
            action[1] = categorical lane-action index (0/1/2)

        Semantic relative lane action:
            index 0 -> delta -1 = LEFT
            index 1 -> delta  0 = KEEP
            index 2 -> delta +1 = RIGHT

        Absolute Highway lane IDs remain simulator geometry IDs. Low-level
        executed steering is computed separately by the steering controller.
        Requires _min_ttc_by_lane() to return:
            {lane_id: {
                    "front_ttc": float,
                    "rear_ttc": float,
                    "min_ttc": float,}}
        Returns:
            filtered_action: np.ndarray
            safety_info: dict
        """

        # Identify each filter invocation so logs from different steps
        # can be correlated.
        self._ttc_filter_call_count = getattr(self, "_ttc_filter_call_count", 0) + 1
        ttc_filter_call_id = self._ttc_filter_call_count

        # Normalize the hybrid action to a 1-D pair:
        # [normalized_acceleration, lane_action]
        if hasattr(sampled_action, "detach"):
            sampled_action = sampled_action.detach()
            if hasattr(sampled_action, "cpu"):
                sampled_action = sampled_action.cpu()
            if hasattr(sampled_action, "numpy"):
                sampled_action = sampled_action.numpy()

        sampled_array = np.asarray(sampled_action, dtype=np.float32)
        original_shape = sampled_array.shape

        sampled_action = sampled_array.reshape(-1)

        if sampled_action.size != 2:
            raise ValueError(
                "filter_ttc_action expects exactly two values: "
                "[normalized_acceleration, lane_action]. "
                f"Received shape={original_shape}, "
                f"value={sampled_action.tolist()}"
            )



        sampled_action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)


        if sampled_action.size != 2:
            raise ValueError(
                f"Expected action with 2 elements, "
                f"received shape {sampled_action.shape}"
            )

        filtered_action = sampled_action.copy()

        base_env = self.env.unwrapped
        vehicle = base_env.vehicle
        lanes_count = int(base_env.config["lanes_count"])

        emergency_threshold = float(SAFETY_OVERRIDE_TTC)
        lane_change_threshold = float(LANE_CHANGE_MIN_REAR_TTC)

        # ---------------------------------------------------------
        # 1. Assess current-lane emergency risk.
        # ---------------------------------------------------------
        lane_ttc_current = self._min_ttc_by_lane()

        current_lane = int(vehicle.lane_index[2])

        default_ttc = {
            "front_ttc": float("inf"),
            "rear_ttc": float("inf"),
            "min_ttc": float("inf"),
        }

        current_values = lane_ttc_current.get(current_lane, default_ttc)

        # Maneuver-aware safety: assess both lanes on every policy decision.
        # A lane change is a commitment over several steps, not a fresh one-step
        # decision. Do not abort into a source lane whose rear gap is collapsing.
        active_lane_safety_event = None
        active_maneuver_progress = None
        active_source_rear_critical = False
        active_target_rear_critical = False
        active_target_front_critical = False
        if self._lane_change_active and self._lane_change_source_index is not None:
            try:
                source_lane_id_active = int(self._lane_change_source_index[2])
                target_lane_id_active = int(self.target_lane_index[2])
                source_values_active = lane_ttc_current.get(source_lane_id_active, default_ttc)
                target_values_active = lane_ttc_current.get(target_lane_id_active, default_ttc)

                source_front_ttc_active = float(source_values_active.get("front_ttc", float("inf")))
                source_rear_ttc_active = float(source_values_active.get("rear_ttc", float("inf")))
                target_front_ttc_active = float(target_values_active.get("front_ttc", float("inf")))
                target_rear_ttc_active = float(target_values_active.get("rear_ttc", float("inf")))
                source_front_gap_active = float(source_values_active.get("front_gap", float("inf")))
                source_rear_gap_active = float(source_values_active.get("rear_gap", float("inf")))
                target_front_gap_active = float(target_values_active.get("front_gap", float("inf")))
                target_rear_gap_active = float(target_values_active.get("rear_gap", float("inf")))
                ego_speed_active = float(vehicle.speed)

                target_lane_active_geometry = base_env.road.network.get_lane(self.target_lane_index)
                _, target_lateral_active = target_lane_active_geometry.local_coordinates(vehicle.position)
                source_lane_active_geometry = base_env.road.network.get_lane(self._lane_change_source_index)
                _, source_lateral_active = source_lane_active_geometry.local_coordinates(vehicle.position)
                source_is_nearer_active = abs(float(source_lateral_active)) < abs(float(target_lateral_active))
                active_maneuver_progress = "SOURCE_SIDE" if source_is_nearer_active else "TARGET_SIDE"

                active_source_rear_critical = (
                    source_rear_ttc_active <= LANE_CHANGE_ACTIVE_ABORT_TTC
                    or source_rear_gap_active <= LANE_CHANGE_ACTIVE_ABORT_GAP
                )
                active_target_rear_critical = (
                    target_rear_ttc_active <= LANE_CHANGE_ACTIVE_ABORT_TTC
                    or target_rear_gap_active <= LANE_CHANGE_ACTIVE_ABORT_GAP
                )
                active_target_front_critical = target_front_ttc_active <= SAFETY_OVERRIDE_TTC

                source_lane_safe_for_abort = (
                    source_front_ttc_active > SAFETY_OVERRIDE_TTC
                    and source_rear_ttc_active > LANE_CHANGE_ACTIVE_ABORT_TTC
                    and source_front_gap_active >= LANE_CHANGE_ACTIVE_ABORT_GAP
                    and source_rear_gap_active >= LANE_CHANGE_ACTIVE_ABORT_GAP
                )
                target_lane_safe_to_complete = (
                    target_front_ttc_active > SAFETY_OVERRIDE_TTC
                    and target_front_gap_active >= 0.0
                    and not (
                        target_rear_ttc_active <= 0.5
                        or target_rear_gap_active <= 0.0
                    )
                )

                # Recovery rule:
                # 1) If the target lane becomes dangerous while ego is still
                #    nearer the source, return only if the source is safe.
                # 2) If source rear pressure makes returning unsafe, do NOT
                #    switch target back to the source lane. If the target is
                #    viable, keep tracking it; if both lanes are dangerous,
                #    retain the target trajectory and use longitudinal control.
                should_recover_to_source = (
                    active_target_rear_critical
                    and source_is_nearer_active
                    and source_lane_safe_for_abort
                )
                if should_recover_to_source:
                    aborted_target = tuple(self.target_lane_index)
                    source_target = tuple(self._lane_change_source_index)
                    aborted_maneuver_id = self._active_lane_maneuver_id
                    aborted_request_id = self._active_lane_request_trace_id
                    self.target_lane_index = source_target
                    self._lane_change_active = False
                    self._lane_change_cooldown = int(max(
                        LANE_CHANGE_COOLDOWN_STEPS,
                        LANE_CHANGE_RECOVERY_COOLDOWN_STEPS,
                    ))
                    self._lane_change_source_index = None
                    self._lane_change_started_step = None
                    self._active_lane_maneuver_id = None
                    self._active_lane_request_trace_id = None
                    self._lane_change_stable_steps = 0
                    self._lane_change_timeout_logged = False
                    if getattr(self, "_lane_change_stats", None) is not None:
                        self._lane_change_stats["aborted"] = self._lane_change_stats.get("aborted", 0) + 1
                    active_lane_safety_event = "RECOVER_TO_SOURCE_TARGET_REAR_CRITICAL_SOURCE_SAFE"
                    write_diagnostic_event(
                        "lane_maneuver_safety_decision", self,
                        maneuver_id=aborted_maneuver_id,
                        request_id=aborted_request_id,
                        decision="RECOVER_TO_SOURCE",
                        reason=active_lane_safety_event,
                        progress=active_maneuver_progress,
                        source_lane=source_lane_id_active,
                        target_lane=target_lane_id_active,
                        source_rear_ttc=source_rear_ttc_active,
                        source_rear_gap=source_rear_gap_active,
                        target_rear_ttc=target_rear_ttc_active,
                        target_rear_gap=target_rear_gap_active,
                    )
                elif active_target_rear_critical and not source_lane_safe_for_abort:
                    active_lane_safety_event = (
                        "CONTINUE_TARGET_SOURCE_UNSAFE"
                        if target_lane_safe_to_complete
                        else "KEEP_TARGET_TRAJECTORY_BOTH_LANES_RISKY"
                    )
                    write_diagnostic_event(
                        "lane_maneuver_safety_decision", self,
                        maneuver_id=getattr(self, "_active_lane_maneuver_id", None),
                        request_id=getattr(self, "_active_lane_request_trace_id", None),
                        decision="CONTINUE_TARGET",
                        reason=active_lane_safety_event,
                        progress=active_maneuver_progress,
                        source_lane=source_lane_id_active,
                        target_lane=target_lane_id_active,
                        source_front_ttc=source_front_ttc_active,
                        source_rear_ttc=source_rear_ttc_active,
                        source_rear_gap=source_rear_gap_active,
                        target_front_ttc=target_front_ttc_active,
                        target_rear_ttc=target_rear_ttc_active,
                        target_rear_gap=target_rear_gap_active,
                    )
                elif active_target_front_critical:
                    # A target-front hazard is not solved by blindly switching
                    # lanes. Keep the target plan stable and request braking
                    # through the normal front-TTC emergency controller below.
                    active_lane_safety_event = "TARGET_FRONT_CRITICAL_CONTINUE_TARGET_AND_BRAKE"

                print(
                    "[MANEUVER-AWARE SAFETY]",
                    f"maneuver_id={getattr(self, '_active_lane_maneuver_id', None)}",
                    f"progress={active_maneuver_progress}",
                    f"source_lane={source_lane_id_active}",
                    f"target_lane={target_lane_id_active}",
                    f"source_front_ttc={source_front_ttc_active:.3f}",
                    f"source_rear_ttc={source_rear_ttc_active:.3f}",
                    f"source_rear_gap={source_rear_gap_active:.3f}",
                    f"target_front_ttc={target_front_ttc_active:.3f}",
                    f"target_rear_ttc={target_rear_ttc_active:.3f}",
                    f"target_rear_gap={target_rear_gap_active:.3f}",
                    f"event={active_lane_safety_event}",
                )

            except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
                print("[ACTIVE LANE SAFETY CHECK WARNING]", f"error={exc}")

        current_front_ttc = float(current_values["front_ttc"])
        current_rear_ttc = float(current_values["rear_ttc"])
        current_lane_ttc = min(current_front_ttc, current_rear_ttc)

        # Information about the nearest vehicle ahead.
        front_gap = float(current_values.get("front_gap", float("inf")))
        front_speed_raw = current_values.get("front_speed", None)
        nearest_front_ttc = float(current_values.get("nearest_front_ttc", float("inf")))

        front_vehicle_present = front_speed_raw is not None
        front_closing_speed = (
            float(vehicle.speed) - float(front_speed_raw)
            if front_speed_raw is not None else 0.0
        )
        # Positive means the ego is closing on the lead vehicle; negative means
        # the lead vehicle is pulling away and the gap is naturally increasing.
        rear_pressure = current_rear_ttc <= ttc_danger_threshold
        selected_front_vehicle_id = current_values.get("front_vehicle_id")

        # Use the current lane and, during an active lane change,
        # the source and target lanes for emergency monitoring.
        # Monitor all lanes relevant to the ego's current trajectory.
        relevant_lane_ids = self._relevant_lane_ids()

        # Emergency braking is triggered by a closing vehicle AHEAD in a lane
        # occupied by, or actively entering, the ego. A fast-approaching REAR
        # vehicle is relevant to lane-change rejection, but should not by itself
        # trigger maximum braking in the current lane.
        relevant_front_ttc_values = []

        for lane_id in relevant_lane_ids:
            values = lane_ttc_current.get(lane_id)
            if values is None:
                continue
            relevant_front_ttc_values.append(float(values["front_ttc"]))

        min_relevant_front_ttc = min(
            relevant_front_ttc_values,
            default=float("inf"),
        )
        emergency_braking = (min_relevant_front_ttc <= SAFETY_OVERRIDE_TTC)


        # Level 1: adaptive car-following / speed matching.
        desired_gap = MIN_FOLLOWING_GAP + DESIRED_TIME_HEADWAY * max(0.0, float(vehicle.speed))
        gap_trigger_threshold = desired_gap - SPEED_MATCH_GAP_TOLERANCE

        # TTC is the urgent closing-risk signal. A short gap alone should not
        # force a speed-matching override when the lead vehicle is pulling away:
        # in that case the gap is already increasing, so let the policy continue.
        # If the gap is short and the ego is not closing, regulate headway.
        front_ttc_trigger = nearest_front_ttc <= SPEED_MATCH_TTC
        gap_and_closing_trigger = (
            front_gap < gap_trigger_threshold
            and front_closing_speed >= -0.1
        )

        # If a vehicle is rapidly approaching from behind, avoid routine
        # speed-matching overrides when the front TTC is not time-critical.
        # This preserves the policy's opportunity to maintain speed or accelerate;
        # it does NOT override emergency braking for a dangerous lead vehicle.
        rear_pressure_suppresses_routine_matching = (
            rear_pressure and nearest_front_ttc > SPEED_MATCH_TTC
        )
        # If the sampled policy action requests a lane change that passes all
        # four lane-selection principles, routine headway regulation should not
        # cancel the acceleration needed to complete that maneuver. Emergency
        # braking remains independent and cannot be suppressed.
        requested_lane_value = int(np.clip(np.rint(float(sampled_action[1])), 0, 1))
        safe_lane_change_request = False
        if requested_lane_value == 1 and not self._lane_change_active:
            lane_snapshot_for_speed_match = self._min_ttc_by_lane()
            safe_lane_change_request = bool(self._safe_lane_change_candidates(lane_snapshot_for_speed_match))

        speed_match_active = (
            front_vehicle_present
            and (front_ttc_trigger or gap_and_closing_trigger)
            and not rear_pressure_suppresses_routine_matching
            and not safe_lane_change_request
        )


        controller_accel = None



        rear_threat_response_active = False

        if emergency_braking:
            # Critical TTC: request maximum braking.
            filtered_action[0] = -1.0
            speed_match_active = False

        elif (
            current_rear_ttc <= ttc_danger_threshold
            and current_values.get("rear_speed") is not None
            and float(current_values["rear_speed"]) > float(vehicle.speed) + 0.1
            # Do not accelerate away from a rear threat into a close lead vehicle.
            and (
                not front_vehicle_present
                or (
                    nearest_front_ttc > SPEED_MATCH_TTC
                    and front_gap >= gap_trigger_threshold
                )
            )
            and (
                not self._lane_change_active
                or active_source_rear_critical
                or active_lane_safety_event is not None
            )
        ):
            # Rear-threat response: do not merely suppress speed matching.
            # When a fast rear vehicle is closing, modestly accelerate to reduce
            # relative closing speed, unless a critical front TTC already requires
            # braking. The ordinary speed-limit stage still bounds this command.
            rear_ttc_now = float(current_rear_ttc)
            rear_escape_accel = 0.25 if rear_ttc_now > 1.5 else 0.45
            policy_accel_before_rear_response = float(filtered_action[0])
            filtered_action[0] = max(policy_accel_before_rear_response, rear_escape_accel)
            speed_match_active = False
            rear_threat_response_active = True
            print(
                "[REAR-THREAT RESPONSE]",
                f"rear_ttc={rear_ttc_now:.3f}",
                f"rear_gap={float(current_values.get('rear_gap', float('inf'))):.3f}",
                f"rear_speed={float(current_values['rear_speed']):.3f}",
                f"ego_speed={float(vehicle.speed):.3f}",
                f"policy_accel={policy_accel_before_rear_response:+.3f}",
                f"escape_accel={rear_escape_accel:+.3f}",
                f"executed_accel={float(filtered_action[0]):+.3f}",
                f"active_maneuver={bool(self._lane_change_active)}",
                f"active_event={active_lane_safety_event}",
            )
            write_diagnostic_event(
                "rear_threat_response", self,
                rear_ttc=rear_ttc_now,
                rear_gap=float(current_values.get("rear_gap", float("inf"))),
                rear_speed=float(current_values["rear_speed"]),
                ego_speed=float(vehicle.speed),
                policy_accel=policy_accel_before_rear_response,
                executed_accel=float(filtered_action[0]),
                active_maneuver=bool(self._lane_change_active),
                active_event=active_lane_safety_event,
            )

        elif speed_match_active:
            # Speed matching may cap positive acceleration, but it must never
            # cancel a stronger braking request sampled by the RL policy.
            # This preserves the policy's negative acceleration while still
            # preventing unnecessary acceleration toward a close lead vehicle.
            controller_accel = self._speed_matching_acceleration(
                front_speed=float(front_speed_raw),
                front_gap=front_gap,
            )
            policy_accel_before_speed_match = float(filtered_action[0])
            filtered_action[0] = min(
                policy_accel_before_speed_match,
                float(controller_accel),
            )
            print(
                "[SPEED MATCH ARBITRATION]",
                f"policy_accel={policy_accel_before_speed_match:+.6f}",
                f"controller_accel={float(controller_accel):+.6f}",
                f"executed_accel={float(filtered_action[0]):+.6f}",
                f"policy_braking_preserved={policy_accel_before_speed_match < 0.0 and filtered_action[0] == policy_accel_before_speed_match}",
            )



        # DIAGNOSTIC A: identify changes from emergency braking
        # or speed matching.
        accel_after_ttc_control = float(filtered_action[0])

        # ---------------------------------------------------------
        # 2. Decode high-level KEEP/CHANGE and select a target separately.
        # ---------------------------------------------------------
        policy_requested_lane_action = int(np.clip(np.rint(float(sampled_action[1])), 0, 1))
        requested_lane_action = policy_requested_lane_action
        requested_lane_name = ("KEEP", "CHANGE")[policy_requested_lane_action]
        requested_lane_delta = 0
        candidate_lane = current_lane
        filtered_lane_action = requested_lane_action
        filtered_lane_name = requested_lane_name
        lane_change_rejection_reasons = []
        lane_change_blocked = False
        candidate_front_ttc = float("inf")
        candidate_rear_ttc = float("inf")
        candidate_front_gap = float("inf")
        candidate_rear_gap = float("inf")
        candidate_front_gap_limit = LANE_CHANGE_MIN_GAP
        candidate_rear_gap_limit = LANE_CHANGE_MIN_GAP
        candidate_lane_available = False
        selected_candidate = None

        if policy_requested_lane_action == 1:
            if getattr(self, "_lane_change_stats", None) is not None:
                self._lane_change_stats["requested"] += 1
            if self._lane_change_active:
                lane_change_rejection_reasons.append("lane_change_already_in_progress")
            if self._lane_change_cooldown > 0:
                lane_change_rejection_reasons.append("lane_change_cooldown")
            if not lane_change_rejection_reasons:
                selected_candidate = self._select_best_safe_lane_candidate(lane_ttc_current)
                if selected_candidate is None:
                    lane_change_rejection_reasons.append("no_safe_adjacent_lane")
                else:
                    candidate_lane = int(selected_candidate["target_lane"])
                    candidate_lane_available = True
                    requested_lane_delta = int(candidate_lane - current_lane)
                    requested_lane_name = "CHANGE_LEFT" if candidate_lane > current_lane else "CHANGE_RIGHT"
                    candidate_values = lane_ttc_current[candidate_lane]
                    candidate_front_ttc = float(candidate_values.get("front_ttc", float("inf")))
                    candidate_rear_ttc = float(candidate_values.get("rear_ttc", float("inf")))
                    candidate_front_gap = float(candidate_values.get("front_gap", float("inf")))
                    candidate_rear_gap = float(candidate_values.get("rear_gap", float("inf")))
                    lane_change_rejection_reasons.extend(self._lane_change_principle_reasons(
                        candidate_lane, lane_ttc_current, current_lane=current_lane,
                        ego_speed=float(vehicle.speed),
                    ))
                    print("[LANE-CHANGE SAFETY SNAPSHOT]",
                          f"filter_call={ttc_filter_call_id}",
                          f"ego_lane={current_lane}",
                          f"target_lane={candidate_lane}",
                          f"direction={requested_lane_name}",
                          f"front_gap_m={candidate_front_gap:.3f}",
                          f"rear_gap_m={candidate_rear_gap:.3f}",
                          f"front_ttc_s={candidate_front_ttc:.3f}",
                          f"rear_ttc_s={candidate_rear_ttc:.3f}",
                          f"utility_improvement={selected_candidate.get('utility_improvement', selected_candidate['utility']):.3f}",
                           f"candidate_utility={selected_candidate.get('candidate_utility', 0.0):.3f}",
                           f"benefit_horizon_s={selected_candidate.get('prediction_horizon_s', BENEFIT_PREDICTION_HORIZON_SECONDS):.1f}s")

            lane_change_blocked = bool(lane_change_rejection_reasons)
            if lane_change_blocked:
                filtered_action[1] = 0.0
                filtered_lane_action = 0
                filtered_lane_name = "KEEP"
                candidate_lane = current_lane
                requested_lane_delta = 0
            else:
                filtered_lane_action = 1
                filtered_lane_name = "CHANGE"

        lane_change_rejection_reason = (
            ",".join(lane_change_rejection_reasons)
            if lane_change_rejection_reasons else "none"
        )
        if policy_requested_lane_action == 1 and getattr(self, "_lane_change_stats", None) is not None:
            if lane_change_blocked:
                self._lane_change_stats["rejected"] += 1
            else:
                self._lane_change_stats["accepted"] += 1

        # ---------------------------------------------------------
        # 3B. Apply the speed limit to the filtered action.
        # This must happen before dynamics prediction and execution.
        # ---------------------------------------------------------
        # Apply the speed limit before dynamics prediction and execution.
        speed_limit_override = False
        current_speed = float(vehicle.speed)
        max_speed = float(getattr(vehicle, "MAX_SPEED", 40.0))


        policy_dt = 1.0 / float(base_env.config["policy_frequency"])
        if current_speed < -0.1:
            raise RuntimeError(
                f"Invalid reverse ego speed: {current_speed:.3f} m/s. "
                "Reset the episode and inspect the speed dynamics."
            )

        # DIAGNOSTIC B: inspect the action before the speed-floor constraint.
        accel_before_speed_floor = float(filtered_action[0])

        min_normalized_accel = None

        if current_speed >= 0.0 and filtered_action[0] < 0.0:
            # Normalized action u maps to physical acceleration a = u * 5 m/s^2.
            # Keep predicted speed at or above 0.05 m/s over one policy interval.
            min_normalized_accel = (
                (0.05 - current_speed)
                / (MAX_PHYSICAL_ACCELERATION * policy_dt)
            )

            if filtered_action[0] < min_normalized_accel:
                filtered_action[0] = float(np.clip(min_normalized_accel, -1.0, 1.0))

        accel_after_speed_floor = float(filtered_action[0])

        print(
            "[FILTER STAGE B]",
            f"speed={current_speed:.6f}",
            f"policy_dt={policy_dt:.6f}",
            f"accel_before={accel_before_speed_floor:.6f}",
            f"min_normalized_accel={min_normalized_accel}",
            f"accel_units=normalized[-1,1]",
            f"physical_accel_before_mps2={accel_before_speed_floor * MAX_PHYSICAL_ACCELERATION:+.3f}",
            f"physical_accel_after_mps2={accel_after_speed_floor * MAX_PHYSICAL_ACCELERATION:+.3f}",
            f"accel_after={accel_after_speed_floor:.6f}",
            f"changed={abs(accel_after_speed_floor - accel_before_speed_floor) > 1e-6}",
        )


        # Highway action acceleration range is [-5, 5] m/s^2.
        policy_dt = 1.0 / float(policy_frequency)
        max_physical_acceleration = 5.0

        if filtered_action[0] > 0.0:
            predicted_speed = (
                    current_speed
                    + filtered_action[0]
                    * max_physical_acceleration
                    * policy_dt
            )

            if predicted_speed > max_speed:
                max_allowed_normalized_accel = (max_speed - current_speed) / (max_physical_acceleration * policy_dt)

                filtered_action[0] = float(np.clip(max_allowed_normalized_accel, 0.0, 1.0))
                speed_limit_override = True

        filtered_action[1] = float(np.clip(np.rint(filtered_action[1]), 0, 1))




        # ---------------------------------------------------------
        # 4. Diagnostics.
        # ---------------------------------------------------------
        safety_info = {
            "current_lane": current_lane,
            "current_speed": float(vehicle.speed),
            "candidate_lane": candidate_lane,

            "requested_lane_action": requested_lane_action,  # categorical index
            "requested_lane_name": requested_lane_name,
            "requested_lane_action_index": policy_requested_lane_action,
            "requested_lane_delta": requested_lane_delta,
            "filtered_lane_action": filtered_lane_action,  # categorical index
            "filtered_lane_name": filtered_lane_name,
            "filtered_lane_delta": int(requested_lane_delta if filtered_lane_action == 1 else 0),
            "lane_action_semantics": "0=KEEP,1=CHANGE",
            "target_lane_selection": selected_candidate,
            "current_lane_id": current_lane,
            "requested_target_lane_id": candidate_lane,

            "current_front_ttc": current_front_ttc,
            "current_rear_ttc": current_rear_ttc,
            "current_lane_ttc": float(current_lane_ttc),

            "candidate_front_ttc": float(candidate_front_ttc),
            "candidate_rear_ttc": float(candidate_rear_ttc),
            "candidate_front_gap": float(candidate_front_gap),
            "candidate_rear_gap": float(candidate_rear_gap),
            "candidate_front_gap_limit": float(candidate_front_gap_limit),
            "candidate_rear_gap_limit": float(candidate_rear_gap_limit),
            "candidate_lane_available": bool(candidate_lane_available),
            "lane_change_principles_passed": bool(
                policy_requested_lane_action == 0 or not lane_change_rejection_reasons
            ),
            "lane_change_rejection_reasons": list(lane_change_rejection_reasons),
            "lane_change_rejection_reason": lane_change_rejection_reason,

            "emergency_braking": bool(emergency_braking),
            "lane_change_blocked": bool(lane_change_blocked),
            "active_lane_safety_event": active_lane_safety_event,
            "active_maneuver_progress": active_maneuver_progress,
            "active_source_rear_critical": bool(active_source_rear_critical),
            "active_target_rear_critical": bool(active_target_rear_critical),
            "active_target_front_critical": bool(active_target_front_critical),
            "rear_threat_response": bool(rear_threat_response_active),

            "emergency_threshold": emergency_threshold,
            "lane_change_threshold": lane_change_threshold,

            "sampled_action": sampled_action.copy(),

            "speed_limit_override": bool(speed_limit_override),
            "filtered_action": filtered_action.copy(),
            # Keep a broad flag for diagnostics, but do not use it as a blanket reward penalty.
            "overridden": bool(emergency_braking or speed_match_active or lane_change_blocked or speed_limit_override),
            "emergency_intervention": bool(emergency_braking),
            "routine_controller_intervention": bool(speed_match_active and not emergency_braking),
            "front_gap": front_gap,
            "front_speed": (
                float(front_speed_raw)
                if front_speed_raw is not None
                else None),
            "nearest_front_ttc": nearest_front_ttc,
            "speed_match_active": bool(speed_match_active),
            # Distinguish routine longitudinal regulation from a genuine safety intervention.
            "intervention_reason": (
                "emergency_ttc_braking" if emergency_braking
                else f"lane_change_rejected:{lane_change_rejection_reason}" if lane_change_blocked
                else "speed_matching" if speed_match_active
                else "speed_limit" if speed_limit_override
                else "none"
            ),
            "reward_safety_penalty": (
                SAFETY_OVERRIDE_REWARD_PENALTY if emergency_braking
                else LANE_REJECTION_REWARD_PENALTY if lane_change_blocked
                else 0.0
            ),
        }



        # Keep one request-level trace for each CHANGE decision. The post-step
        # [TTC RISK] snapshot separately reports the resulting physical risk.
        safety_info["ttc_filter_call_id"] = ttc_filter_call_id
        safety_info["safety_filter_applied"] = True




        # Structured request-level trace: one ID per sampled high-level CHANGE decision.
        # This intentionally does not equate a request with a unique physical maneuver.
        if policy_requested_lane_action == 1:
            request_trace_id = int(getattr(self, "_lane_request_trace_sequence", 0)) + 1
            self._lane_request_trace_sequence = request_trace_id
            rejection_reasons = list(lane_change_rejection_reasons)
            decision_status = "REJECTED" if lane_change_blocked else "ACCEPTED"
            safety_info.update({
                "lane_request_trace_id": request_trace_id,
                "lane_request_direction": requested_lane_name,
                "lane_request_source_lane": int(current_lane),
                "lane_request_target_lane": int(candidate_lane),
                "lane_request_decision": decision_status,
                "lane_request_rejection_reasons": rejection_reasons,
                "lane_request_candidate_front_ttc": float(candidate_front_ttc),
                "lane_request_candidate_rear_ttc": float(candidate_rear_ttc),
                "lane_request_candidate_front_gap": float(candidate_front_gap),
                "lane_request_candidate_rear_gap": float(candidate_rear_gap),
                "lane_request_candidate_front_gap_limit": float(candidate_front_gap_limit),
                "lane_request_candidate_rear_gap_limit": float(candidate_rear_gap_limit),
            })
            write_diagnostic_event(
                "lane_request_decision", self,
                request_id=request_trace_id,
                direction=requested_lane_name,
                decision=decision_status,
                rejection_reasons=rejection_reasons,
                source_lane=int(current_lane),
                target_lane=int(candidate_lane),
                candidate_front_ttc=float(candidate_front_ttc),
                candidate_rear_ttc=float(candidate_rear_ttc),
                candidate_front_gap=float(candidate_front_gap),
                candidate_rear_gap=float(candidate_rear_gap),
                candidate_front_gap_limit=float(candidate_front_gap_limit),
                candidate_rear_gap_limit=float(candidate_rear_gap_limit),
                filter_call_id=ttc_filter_call_id,
                valid_action_mask=getattr(self, "_trace_valid_action_mask", None),
                sampled_action=np.asarray(sampled_action).tolist(),
                filtered_action=np.asarray(filtered_action).tolist(),
            )
            print(
                "[LANE REQUEST TRACE]",
                f"id={request_trace_id}",
                f"decision={decision_status}",
                f"direction={requested_lane_name}",
                f"source_lane={current_lane}",
                f"target_lane={candidate_lane}",
                f"reasons={rejection_reasons or ['none']}",
                f"front_ttc={candidate_front_ttc:.3f}",
                f"rear_ttc={candidate_rear_ttc:.3f}",
                f"front_gap={candidate_front_gap:.3f}",
                f"rear_gap={candidate_rear_gap:.3f}",
                f"front_gap_limit={candidate_front_gap_limit:.3f}",
                f"rear_gap_limit={candidate_rear_gap_limit:.3f}",
                f"active_maneuver={bool(self._lane_change_active)}",
                f"cooldown={int(self._lane_change_cooldown)}",
            )
        return filtered_action, safety_info





#???????????????????????????????????????????????????????????????????????????????????????????????????????????
    def _update_lane_target(self, lane_action_id):
        """
        Legacy persistent-target helper for the non-instantaneous controller.

        The active instantaneous KEEP/CHANGE policy selects its safe target in
        filter_ttc_action() and executes it in step(), so this legacy controller
        must not reinterpret the binary action as LEFT/KEEP/RIGHT.
        """
        if INSTANT_LANE_CHANGE_MODE:
            return

        base_env = self.env.unwrapped
        ego = base_env.vehicle
        road = base_env.road

        lane_action_id = int(np.clip(lane_action_id, 0, 2))
        lane_action_delta = lane_action_index_to_delta(lane_action_id)

        # Evaluate geometric completion in the target lane's own coordinates.
        target_lane = road.network.get_lane(self.target_lane_index)
        longitudinal, lateral = target_lane.local_coordinates(ego.position)
        target_heading = float(target_lane.heading_at(longitudinal))
        heading_error = (float(ego.heading) - target_heading + np.pi) % (2.0 * np.pi) - np.pi

        target_lane_id = int(self.target_lane_index[2])
        physical_lane_id = int(ego.lane_index[2])
        source_lane_id_for_check = (
            int(self._lane_change_source_index[2])
            if self._lane_change_source_index is not None else physical_lane_id
        )
        target_is_nearest = True
        if self._lane_change_source_index is not None:
            source_lane_geometry = road.network.get_lane(self._lane_change_source_index)
            source_s, source_lateral = source_lane_geometry.local_coordinates(ego.position)
            target_is_nearest = abs(float(lateral)) <= abs(float(source_lateral))

        target_reached = (
            abs(float(lateral)) <= LANE_CHANGE_COMPLETE_LATERAL_TOL
            and abs(float(heading_error)) <= LANE_CHANGE_COMPLETE_HEADING_TOL
            and (physical_lane_id == target_lane_id or target_is_nearest)
        )

        if self._lane_change_active:
            elapsed_before_completion = (
                int(self.step_count - self._lane_change_started_step)
                if self._lane_change_started_step is not None else 0
            )
            # If the strict centre-alignment test has not completed a long
            # maneuver, accept a modest residual offset only when the target
            # lane is clearly the nearer lane and heading is already aligned.
            timeout_reached = (
                elapsed_before_completion >= 25
                and abs(float(lateral)) <= LANE_CHANGE_TIMEOUT_LATERAL_TOL
                and abs(float(heading_error)) <= LANE_CHANGE_TIMEOUT_HEADING_TOL
                and target_is_nearest
            )
            if target_reached or timeout_reached:
                self._lane_change_stable_steps = (
                    int(getattr(self, "_lane_change_stable_steps", 0)) + 1
                )
            else:
                self._lane_change_stable_steps = 0

            if self._lane_change_stable_steps >= LANE_CHANGE_COMPLETE_STABLE_STEPS:
                completed_steps = (
                    int(self.step_count - self._lane_change_started_step)
                    if self._lane_change_started_step is not None else None
                )
                source_lane_id = int(self._lane_change_source_index[2]) if self._lane_change_source_index is not None else int(ego.lane_index[2])
                target_lane_id = int(self.target_lane_index[2])
                completed_direction = "LEFT" if target_lane_id > source_lane_id else "RIGHT"
                completed_maneuver_id = self._active_lane_maneuver_id
                completed_request_id = self._active_lane_request_trace_id
                source_lane_completed = source_lane_id
                completion_mode = "GEOMETRIC" if target_reached else "TIMEOUT_TOLERANCE"
                self._lane_change_active = False
                self._lane_change_cooldown = int(max(
                    LANE_CHANGE_COOLDOWN_STEPS,
                    LANE_CHANGE_RECOVERY_COOLDOWN_STEPS if timeout_reached else 0,
                ))
                self._lane_change_stable_steps = 0
                self._lane_change_source_index = None
                self._lane_change_started_step = None
                self._active_lane_maneuver_id = None
                self._active_lane_request_trace_id = None
                self._lane_change_timeout_logged = False
                if getattr(self, "_lane_change_stats", None) is not None:
                    self._lane_change_stats["completed"] += 1
                    self._lane_change_stats[f"completed_{completed_direction}"] += 1
                write_diagnostic_event(
                    "lane_maneuver_terminal", self,
                    maneuver_id=completed_maneuver_id,
                    request_id=completed_request_id,
                    outcome="COMPLETED",
                    completion_mode=completion_mode,
                    lateral_error=float(lateral),
                    heading_error=float(heading_error),
                    stable_steps=LANE_CHANGE_COMPLETE_STABLE_STEPS,
                    direction=completed_direction,
                    source_lane=source_lane_completed,
                    target_lane=target_lane_id,
                    physical_lane=getattr(ego, "lane_index", None),
                    steps_since_start=completed_steps,
                    cooldown_started=self._lane_change_cooldown,
                )
                print(
                    "[LANE MANEUVER TERMINAL]",
                    f"maneuver_id={completed_maneuver_id}",
                    f"request_id={completed_request_id}",
                    "outcome=COMPLETED",
                    f"completion_mode={completion_mode}",
                    f"lateral_error={float(lateral):+.3f}",
                    f"heading_error={float(heading_error):+.3f}",
                    f"stable_steps={LANE_CHANGE_COMPLETE_STABLE_STEPS}",
                    f"direction={completed_direction}",
                    f"source_lane={source_lane_completed}",
                    f"target_lane={target_lane_id}",
                    f"physical_lane={getattr(ego, 'lane_index', None)}",
                    f"steps_since_start={completed_steps}",
                    f"cooldown_started={self._lane_change_cooldown}",
                    f"stats={getattr(self, '_lane_change_stats', {})}",
                )
            elif lane_action_delta != 0:
                print(
                    "[LANE REQUEST DURING MANEUVER]",
                    f"active_maneuver_id={self._active_lane_maneuver_id}",
                    f"requested_action={lane_action_delta:+d}",
                    "outcome=IGNORED_WHILE_ACTIVE",
                )
            elapsed_steps = (
                int(self.step_count - self._lane_change_started_step)
                if self._lane_change_started_step is not None else 0
            )
            if elapsed_steps >= 25 and not self._lane_change_timeout_logged:
                self._lane_change_timeout_logged = True
                if getattr(self, "_lane_change_stats", None) is not None:
                    self._lane_change_stats["timed_out"] += 1
                write_diagnostic_event(
                    "lane_maneuver_timeout_warning", self,
                    maneuver_id=self._active_lane_maneuver_id,
                    direction=(
                        "LEFT" if self._lane_change_source_index is not None
                        and int(self.target_lane_index[2]) > int(self._lane_change_source_index[2])
                        else "RIGHT" if self._lane_change_source_index is not None else "UNKNOWN"
                    ),
                    source_lane=self._lane_change_source_index,
                    target_lane=self.target_lane_index,
                    elapsed_steps=elapsed_steps,
                    lateral_error=float(lateral),
                    outcome="STILL_ACTIVE_NOT_COMPLETED",
                )
                print(
                    "[LANE MANEUVER TIMEOUT WARNING]",
                    f"maneuver_id={self._active_lane_maneuver_id}",
                    f"direction={'LEFT' if int(self.target_lane_index[2]) > int(self._lane_change_source_index[2]) else 'RIGHT' if self._lane_change_source_index is not None else 'UNKNOWN'}",
                    f"source_lane={self._lane_change_source_index}",
                    f"target_lane={self.target_lane_index}",
                    f"elapsed_steps={elapsed_steps}",
                    f"lateral_error={float(lateral):+.3f}",
                    "outcome=STILL_ACTIVE_NOT_COMPLETED",
                )
            return

        if self._lane_change_cooldown > 0:
            if lane_action_delta != 0:
                print(
                    "[LANE TARGET REJECTED]",
                    "reason=lane_change_cooldown",
                    f"cooldown_remaining={self._lane_change_cooldown}",
                )
            self._lane_change_cooldown -= 1
            return

        # Keep-lane action does not change the target lane.
        if lane_action_delta == 0:
            return

        # In this Highway configuration, lane IDs increase in the
        # direction of increasing lateral position. Verify this in
        # your rendered simulator before relying on left/right labels.




        start_node, end_node, current_lane_id = self.target_lane_index
        current_lane_id = int(current_lane_id)

        # Find the actual lanes available on this road segment.
        available_lanes = road.network.graph[start_node][end_node]

        valid_lane_ids = set(range(len(available_lanes)))

        # In this configuration, lane IDs should be 0, 1, and 2.
        candidate_lane_id = relative_lane_delta_to_absolute_lane(
            current_lane_id, lane_action_delta
        )

        if candidate_lane_id not in valid_lane_ids:
            print(
                "[LANE CHANGE REJECTED] "
                f"current_lane={current_lane_id}, "
                f"candidate_lane={candidate_lane_id}, "
                f"valid_lane_ids={sorted(valid_lane_ids)}"
            )
            return

        candidate_lane_index = (start_node, end_node, candidate_lane_id,)

        # Preserve the lane being left for TTC monitoring.
        self._lane_change_source_index = self.target_lane_index

        self.target_lane_index = candidate_lane_index
        self._lane_change_active = True
        self._lane_change_started_step = int(self.step_count)
        self._lane_maneuver_sequence = int(getattr(self, "_lane_maneuver_sequence", 0)) + 1
        self._active_lane_maneuver_id = self._lane_maneuver_sequence
        self._active_lane_request_trace_id = getattr(self, "_current_lane_request_trace_id", None)
        self._lane_change_timeout_logged = False
        self._lane_change_stable_steps = 0
        if getattr(self, "_lane_change_stats", None) is not None:
            self._lane_change_stats["started"] = self._lane_change_stats.get("started", 0) + 1

        write_diagnostic_event(
            "lane_maneuver_started", self,
            maneuver_id=self._active_lane_maneuver_id,
            request_id=getattr(self, "_current_lane_request_trace_id", None),
            direction="LEFT" if candidate_lane_id > current_lane_id else "RIGHT",
            source_lane=current_lane_id,
            target_lane=candidate_lane_id,
            start_step=self._lane_change_started_step,
        )
        print(
            "[LANE MANEUVER STARTED]",
            f"maneuver_id={self._active_lane_maneuver_id}",
            f"request_id={self._active_lane_request_trace_id}",
            f"direction={'LEFT' if candidate_lane_id > current_lane_id else 'RIGHT'}",
            f"source_lane={current_lane_id}",
            f"target_lane={candidate_lane_id}",
            f"start_step={self._lane_change_started_step}",
        )
        print(
            "[LANE CHANGE REQUEST]",
            f"policy_lane_action_index={lane_action_id}",
            f"relative_lane_delta={lane_action_delta:+d}",
            f"current_absolute_lane_id={current_lane_id}",
            f"target_absolute_lane_id={candidate_lane_id}",
            f"target_lane={self.target_lane_index}",
            f"started_step={self._lane_change_started_step}",
            f"stats={getattr(self, '_lane_change_stats', {})}",
        )




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
            "simulator_step": int(getattr(self, "step_count", 0)),
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


    def _sanitize_traffic_slot_speeds(self, where):
        """Prevent managed traffic-slot vehicles from retaining negative/non-finite speed.

        highway-env updates surrounding vehicles during env.step(). A traffic
        vehicle that acquires a negative scalar speed must be clamped immediately
        after that update, before slot maintenance or safety-mask evaluation.
        """
        base_env = self.env.unwrapped
        ego = base_env.vehicle
        for traffic_vehicle in list(base_env.road.vehicles):
            if traffic_vehicle is ego:
                continue
            if getattr(traffic_vehicle, "_arood_traffic_side", None) not in ("front", "rear"):
                continue
            try:
                raw_speed = float(traffic_vehicle.speed)
            except (AttributeError, TypeError, ValueError):
                raw_speed = float("nan")
            if np.isfinite(raw_speed) and raw_speed >= 0.0:
                continue

            safe_speed = 0.0
            try:
                traffic_vehicle.speed = safe_speed
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Cannot clamp invalid traffic-slot speed at {where}: "
                    f"vehicle_id={id(traffic_vehicle)}, speed={raw_speed}"
                ) from exc

            # Keep the kinematic velocity consistent with the clamped scalar
            # speed. Zeroing velocity avoids one more backward displacement
            # from a stale negative velocity vector on the next update.
            if hasattr(traffic_vehicle, "velocity"):
                try:
                    velocity = np.asarray(traffic_vehicle.velocity, dtype=float)
                    traffic_vehicle.velocity = np.zeros_like(velocity)
                except (AttributeError, TypeError, ValueError):
                    pass

            print(
                "[TRAFFIC SLOT SPEED CLAMP]",
                f"where={where}",
                f"lane={getattr(traffic_vehicle, 'lane_index', None)}",
                f"side={getattr(traffic_vehicle, '_arood_traffic_side', None)}",
                f"vehicle_id={id(traffic_vehicle)}",
                f"raw_speed_mps={raw_speed}",
                f"clamped_speed_mps={safe_speed}",
            )
            write_diagnostic_event(
                "traffic_slot_speed_clamped", self,
                where=where,
                vehicle_id=id(traffic_vehicle),
                lane_index=str(getattr(traffic_vehicle, "lane_index", None)),
                side=getattr(traffic_vehicle, "_arood_traffic_side", None),
                raw_speed_mps=raw_speed,
                clamped_speed_mps=safe_speed,
            )

    def step(self, action, apply_safety_filter=False, sampled_action=None, safety_info=None):
        """Execute the proposed hybrid action with instantaneous lane changes.

        The policy chooses KEEP or CHANGE. For CHANGE, the separate target-lane
        selector supplies an adjacent lane. The upstream safety layer may reject
        the request or adjust acceleration; this method performs the accepted lane
        relocation instantaneously before one native simulator update.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Expected hybrid action [acceleration, lane_action], got {action.shape}")
        incoming_action = action.copy()

        # Pre-action traffic context: descriptive only, never an action veto.
        edm_pre_action_metrics = self._min_ttc_by_lane()
        edm_pre_action = self._edm_interpretation_by_lane(edm_pre_action_metrics)
        lane_before_action = int(self.env.unwrapped.vehicle.lane_index[2])

        if apply_safety_filter:
            # Backward-compatible argument name only: this version's selector
            # does not override actions. It only chooses a target lane for CHANGE.
            action, safety_info = self.filter_ttc_action(incoming_action.copy())
            action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
            filter_applied_here = False
        else:
            action = incoming_action.copy()
            safety_info = dict(safety_info or {})
            filter_applied_here = False

        safety_info["simulator_input_action"] = incoming_action.copy()
        safety_info["executed_high_level_action"] = action.copy()
        safety_info["safety_filter_applied_in_step"] = filter_applied_here
        safety_info["safety_filter_applied"] = bool(
            safety_info.get("safety_filter_applied", False) or filter_applied_here
        )

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

        # Sanitize managed traffic before the native update, then again
        # immediately after it. The post-update pass is essential because
        # highway-env may integrate a slot vehicle into a negative speed during
        # this very update; slot maintenance and safety evaluation must never see
        # that invalid value.
        self._sanitize_traffic_slot_speeds("pre_env_step")
        traffic_speeds_before = self._snapshot_managed_traffic_slots()
        # Preserve the exact collision-proximate scene before native physics can
        # alter/remove a vehicle and make its TTC appear to reset to infinity.
        edm_pre_env_step_metrics = self._min_ttc_by_lane()
        edm_pre_env_step = self._edm_interpretation_by_lane(edm_pre_env_step_metrics)
        vehicles_pre_env_step = snapshot_simulator_vehicles(self)
        crashed_before_step = bool(getattr(ego, "crashed", False))
        obs, env_reward, native_terminated, truncated, env_info = self.env.step(physical_action)
        # This check must run BEFORE post-step sanitization: it records the first
        # negative value and the vehicle's action/kinematics before any clamp or
        # slot replacement can obscure the cause.
        traffic_speeds_after = self._snapshot_managed_traffic_slots()
        self._record_first_negative_speed_transition(
            traffic_speeds_before, traffic_speeds_after, physical_action, env_info
        )
        self._sanitize_traffic_slot_speeds("post_env_step")
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
        self._maintain_six_vehicle_traffic()
        obs = base_env.observation_type.observe()
        next_state = self._encode_state(obs)
        self._assert_traffic_invariant("instant_step")

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
            "one_env_update_per_policy_action": True,
            "lane_change_rejection_reason": lane_rejection_reason,
            "sampled_policy_action": (np.asarray(sampled_action, dtype=np.float32).copy()
                                      if sampled_action is not None else incoming_action.copy()),
            "policy_action": np.asarray([acceleration, lane_action_id], dtype=np.float32),
            "filtered_policy_action": np.asarray([acceleration, lane_action_id], dtype=np.float32),
            "executed_action": physical_action.copy(),
            "executed_acceleration": float(acceleration),
            "executed_steering": 0.0,
            "ttc_safety_filter": safety_info,  # legacy field name; now non-intervening assessment
            "episode_evaluator_assessment": {
                "target_selector_mode": safety_info.get("target_selector_mode"),
                "lane_change_candidate_front_ttc": safety_info.get("lane_change_candidate_front_ttc"),
                "lane_change_candidate_rear_ttc": safety_info.get("lane_change_candidate_rear_ttc"),
                "lane_change_candidate_utility": safety_info.get("lane_change_candidate_utility"),
            },
            "instant_lane_change_mode": True,
            "observed_acceleration": (speed_after - speed_before) / policy_dt,
            "ttc_risk_diagnostics": ttc_risk_diagnostics,
            "ttc_risk_aggregation": "max_front_rear",
            "ttc_risk_thresholds_s": {
                "critical": float(TTC_RISK_CRITICAL),
                "warning": float(TTC_RISK_WARNING),
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

    def get_lane_keeping_steering(
            self,
            k_lateral=0.08,
            k_heading=0.8,
            max_steering= 0.02,
    ):
        """
        Return normalized steering to keep the ego vehicle
        near the target lane selected during reset().

        The returned steering is in [-1, 1], matching the
        ContinuousAction interface.
        """
        base_env = self.env.unwrapped
        ego_vehicle = base_env.vehicle
        road = base_env.road

        target_lane_index = getattr(self, "target_lane_index", ego_vehicle.lane_index)
        lane = road.network.get_lane(target_lane_index)
        longitudinal, lateral = lane.local_coordinates(ego_vehicle.position)
        target_heading = lane.heading_at(longitudinal)
        heading_error = (ego_vehicle.heading - target_heading + np.pi) % (2.0 * np.pi) - np.pi

        # Feedback controller: correct lateral and heading errors.
        raw_steering = -k_lateral * lateral - k_heading * heading_error
        steering = float(np.clip(raw_steering, -max_steering, max_steering))
        print(
            "[STEERING DEBUG]",
            f"lateral={lateral:.3f}",
            f"heading_error={heading_error:.3f}",
            f"raw_steering={raw_steering:.4f}",
            f"executed_normalized_steering={steering:.4f}",
        )

        return steering, float(lateral), float(heading_error)


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
        profile = getattr(self, "_arood_adt_profile", "normal")
        max_braking = (
            ADT_AGGRESSIVE_BRAKING_MPS2
            if profile in ("aggressive", "unstable")
            else IDM_COMFORTABLE_BRAKING_MPS2
        )
        max_acceleration = (
            ADT_AGGRESSIVE_MAX_ACCEL_MPS2
            if profile == "aggressive"
            else IDM_MAX_ACCEL_MPS2
        )
        self.action = {
            "steering": 0.0,
            "acceleration": float(np.clip(
                acceleration, -max_braking, max_acceleration
            )),
        }


# ---------------------------------------------------------------------------
# 3. Replace the old six-vehicle traffic manager
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
    root = math.sqrt(
        IDM_MAX_ACCEL_MPS2 * IDM_COMFORTABLE_BRAKING_MPS2
    )
    dynamic_gap = (
        speed * IDM_TIME_HEADWAY_S
        + speed * closing_speed / max(2.0 * root, 1e-3)
    )
    desired_gap = IDM_MIN_GAP_M + max(0.0, dynamic_gap)
    interaction_term = (desired_gap / gap) ** 2
    acceleration = IDM_MAX_ACCEL_MPS2 * (
        1.0 - free_road_term - interaction_term
    )
    return float(np.clip(
        acceleration, -IDM_COMFORTABLE_BRAKING_MPS2,
        IDM_MAX_ACCEL_MPS2
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
    sampled_desired_speed += random.uniform(-ADT_SPEED_VARIATION_MPS, ADT_SPEED_VARIATION_MPS)
    vehicle._arood_desired_speed = float(np.clip(
        sampled_desired_speed, TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS
    ))
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
    """Create a distributed population; never use six-slot replacement spawning."""
    env = self.env.unwrapped
    ego = env.vehicle
    env.road.vehicles = [ego]
    ego_s = _road_s(self, ego)

    spawned = 0
    attempts = 0
    while spawned < TRAFFIC_TARGET_COUNT and attempts < INITIAL_SPAWN_MAX_ATTEMPTS:
        attempts += 1
        lane_id = random.randrange(NUM_LANES)
        longitudinal = ego_s + random.uniform(
            -INITIAL_PLACEMENT_RANGE_M, INITIAL_PLACEMENT_RANGE_M
        )
        if not _safe_spawn_position(self, lane_id, longitudinal, MIN_INITIAL_GAP_M):
            continue
        speed = random.uniform(TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS)
        _spawn_traffic_vehicle(self, lane_id, longitudinal, speed)
        spawned += 1

    if spawned < TRAFFIC_TARGET_COUNT:
        raise RuntimeError(
            f"Could initialize only {spawned}/{TRAFFIC_TARGET_COUNT} traffic vehicles. "
            "Increase the placement range or reduce the minimum initial spacing."
        )

    self._traffic_population_target = TRAFFIC_TARGET_COUNT
    self._traffic_spawn_count = spawned
    self._traffic_removal_count = 0
    self._traffic_pending_replacements = 0
    self._adt_front_coverage_spawn_count = 0
    self._adt_front_coverage_attempt_failures = 0
    # Fill any lane that lacks a front vehicle before the first observation.
    _ensure_lane_front_coverage(self)
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
        f"profiles={self._adt_profile_counts}",
        f"profile_targets={dict(ADT_PROFILE_MIX)}",
        f"reference=±{REFERENCE_RANGE_M:.0f}m",
        f"recycling=±{RECYCLING_BOUNDARY_M:.0f}m",
        "ego_observation_slots=6",
    )



def _adt_vehicle_acceleration(self, vehicle):
    """Profile-conditioned IDM plus occasional, time-scaled braking events."""
    profile = getattr(vehicle, "_arood_adt_profile", "normal")
    speed = max(0.0, float(vehicle.speed))
    desired_speed = float(getattr(vehicle, "_arood_desired_speed", SPEED_LIMIT_MPS))
    desired_speed = float(np.clip(
        desired_speed, TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS
    ))
    lane_id = int(vehicle.lane_index[2])
    ego_s = _road_s(self, vehicle)
    front = None
    front_gap = float("inf")
    for other in _traffic_in_lane(self, lane_id, exclude=vehicle):
        delta = _road_s(self, other) - ego_s
        if delta <= 0.0:
            continue
        gap = delta - 0.5 * (_vehicle_length(vehicle) + _vehicle_length(other))
        if gap < front_gap:
            front, front_gap = other, gap

    headway = float(getattr(vehicle, "_arood_idm_headway_s", IDM_TIME_HEADWAY_S))
    min_gap = float(getattr(vehicle, "_arood_idm_min_gap_m", IDM_MIN_GAP_M))
    max_accel = (
        ADT_AGGRESSIVE_MAX_ACCEL_MPS2
        if profile == "aggressive"
        else IDM_MAX_ACCEL_MPS2
    )
    comfortable_braking = (
        ADT_AGGRESSIVE_BRAKING_MPS2
        if profile in ("aggressive", "unstable") else IDM_COMFORTABLE_BRAKING_MPS2
    )
    free_term = (speed / max(desired_speed, 1e-3)) ** IDM_ACCEL_EXPONENT
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

    # Unstable drivers occasionally brake hard or accelerate in short pulses.
    # Event rates are scaled by simulation dt rather than policy-step count.
    dt = 1.0 / max(1.0, float(SIMULATION_FREQUENCY_HZ))
    if profile == "unstable":
        remaining = int(getattr(vehicle, "_arood_brake_steps_remaining", 0))
        if remaining > 0:
            vehicle._arood_brake_steps_remaining = remaining - 1
            acceleration = min(acceleration, ADT_SUDDEN_BRAKE_ACCEL_MPS2)
        elif random.random() < ADT_SUDDEN_BRAKE_EVENT_RATE_PER_S * dt:
            vehicle._arood_brake_steps_remaining = max(
                1, int(round(ADT_SUDDEN_BRAKE_DURATION_S / dt))
            )
            acceleration = min(acceleration, ADT_SUDDEN_BRAKE_ACCEL_MPS2)
        elif random.random() < ADT_UNSTABLE_ACCEL_EVENT_RATE_PER_S * dt:
            acceleration = max(acceleration, ADT_UNSTABLE_ACCEL_PULSE_MPS2)

    self.adt_try_lane_change(vehicle)
    return float(np.clip(acceleration, -comfortable_braking, max_accel))


def _adt_try_lane_change(self, vehicle):
    """Occasional independent lane changes with explicit collision-avoidance gaps."""
    profile = getattr(vehicle, "_arood_adt_profile", "normal")
    if getattr(vehicle, "_arood_front_coverage_guard", False):
        return False
    if profile not in ("aggressive", "unstable"):
        return False
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

    dt = 1.0 / max(1.0, float(SIMULATION_FREQUENCY_HZ))
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
        ADT_LANE_CHANGE_COOLDOWN_S * SIMULATION_FREQUENCY_HZ
    ))
    self._adt_background_lane_change_count = int(
        getattr(self, "_adt_background_lane_change_count", 0)
    ) + 1
    return True


def _adt_slot_diagnostics(self, slots):
    """Count changes in the identities occupying the six observation slots."""
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


def _lane_has_front_vehicle(self, lane_id, ego_s):
    """Whether a lane has a front vehicle inside the active reference range."""
    for vehicle in _traffic_in_lane(self, lane_id):
        delta = _road_s(self, vehicle) - ego_s
        if 0.0 < delta <= FRONT_COVERAGE_MAX_DISTANCE_M:
            return True
    return False


def _ensure_lane_front_coverage(self):
    """Ensure all three lanes have a front vehicle within the reference range.

    Coverage vehicles are placed 180–420m ahead, subject to safe spacing.
    They are protected from background lane changes so one lane cannot be
    repeatedly emptied by the traffic manager's own coverage vehicles.
    """
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    ego_speed = max(0.0, float(ego.speed))
    inserted = 0

    for lane_id in range(NUM_LANES):
        if _lane_has_front_vehicle(self, lane_id, ego_s):
            continue

        # A slower front vehicle creates a future closing/acceleration decision
        # while remaining far enough away to avoid an artificial immediate hazard.
        speed = max(
            TRAFFIC_MIN_SPEED_MPS,
            ego_speed - random.uniform(
                FRONT_COVERAGE_SPEED_OFFSET_MIN_MPS,
                FRONT_COVERAGE_SPEED_OFFSET_MAX_MPS,
            ),
        )
        speed = min(speed, TRAFFIC_MAX_SPEED_MPS)

        spawned = False
        for _ in range(LANE_COVERAGE_SPAWN_MAX_ATTEMPTS):
            longitudinal = ego_s + random.uniform(
                FRONT_COVERAGE_SPAWN_MIN_DISTANCE_M,
                FRONT_COVERAGE_SPAWN_MAX_DISTANCE_M,
            )
            if not _safe_spawn_position(
                self, lane_id, longitudinal, MIN_INSERTION_GAP_M
            ):
                continue

            vehicle = _spawn_traffic_vehicle(self, lane_id, longitudinal, speed)
            vehicle._arood_front_coverage_guard = True
            self._traffic_spawn_count += 1
            self._adt_front_coverage_spawn_count += 1
            inserted += 1
            spawned = True
            break

        if not spawned:
            self._adt_front_coverage_attempt_failures += 1

    return inserted


def _spawn_boundary_replacement(self):
    """Insert only beyond the active reference range, with a safe-gap check."""
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    ego_speed = max(0.0, float(ego.speed))

    # A slower vehicle is placed ahead so the ego can approach it; a faster
    # vehicle is placed behind so it can approach the ego. Both enter gradually.
    if random.random() < 0.5:
        side = 1.0
        speed = max(TRAFFIC_MIN_SPEED_MPS, ego_speed - random.uniform(BOUNDARY_SPAWN_SPEED_OFFSET_MIN_MPS, BOUNDARY_SPAWN_SPEED_OFFSET_MAX_MPS))
    else:
        side = -1.0
        speed = min(TRAFFIC_MAX_SPEED_MPS, ego_speed + random.uniform(2.0, 6.0))

    for _ in range(BOUNDARY_SPAWN_MAX_ATTEMPTS):
        lane_id = random.randrange(NUM_LANES)
        longitudinal = ego_s + side * random.uniform(
            REFERENCE_RANGE_M + BOUNDARY_SPAWN_EXTRA_RANGE_M, RECYCLING_BOUNDARY_M - BOUNDARY_SPAWN_DISTANCE_BUFFER_M
        )
        if not _safe_spawn_position(self, lane_id, longitudinal, MIN_INSERTION_GAP_M):
            continue
        _spawn_traffic_vehicle(self, lane_id, longitudinal, speed)
        self._traffic_spawn_count += 1
        self._traffic_pending_replacements = max(
            0, self._traffic_pending_replacements - 1
        )
        return True
    return False


def _maintain_continuous_traffic(self):
    """Keep every vehicle until it crosses the outer recycling boundary."""
    road = self.env.unwrapped.road
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    kept = [ego]
    removed = 0

    for vehicle in list(road.vehicles):
        if vehicle is ego:
            continue
        try:
            s = _road_s(self, vehicle)
            speed = float(vehicle.speed)
            lane_id = int(vehicle.lane_index[2])
        except (AttributeError, TypeError, ValueError, IndexError):
            removed += 1
            continue

        if not np.isfinite(speed) or speed < 0.0:
            vehicle.speed = max(0.0, speed if np.isfinite(speed) else 0.0)
        if lane_id not in range(NUM_LANES):
            removed += 1
            continue

        if abs(s - ego_s) > RECYCLING_BOUNDARY_M:
            removed += 1
            continue

        # No vehicle is removed merely because it is outside the six slots or
        # outside the ±500m active reference interval.
        kept.append(vehicle)

    road.vehicles = kept
    if removed:
        self._traffic_removal_count += removed
        self._traffic_pending_replacements += removed

    # Retry deferred replacements on subsequent steps if no safe insertion gap
    # is currently available. No vehicle is inserted close to the ego.
    while self._traffic_pending_replacements > 0:
        if not _spawn_boundary_replacement(self):
            break

    # Refill lane-specific front coverage after removals and background
    # lane changes. This runs every simulator step, independent of ego lane.
    front_coverage_inserted = _ensure_lane_front_coverage(self)

    return {
        "population": len(road.vehicles) - 1,
        "front_coverage_inserted": int(front_coverage_inserted),
        "front_coverage_spawned_total": int(self._adt_front_coverage_spawn_count),
        "front_coverage_failures": int(self._adt_front_coverage_attempt_failures),
        "spawned_total": int(self._traffic_spawn_count),
        "removed_total": int(self._traffic_removal_count),
        "pending_replacements": int(self._traffic_pending_replacements),
    }


def _assert_continuous_traffic_invariant(self, where):
    road = self.env.unwrapped.road
    ego = self.env.unwrapped.vehicle
    vehicles = [v for v in road.vehicles if v is not ego]
    if any(not np.isfinite(float(v.speed)) or float(v.speed) < 0.0 for v in vehicles):
        raise RuntimeError(f"Invalid background vehicle speed at {where}.")
    if any(int(v.lane_index[2]) not in range(NUM_LANES) for v in vehicles):
        raise RuntimeError(f"Background vehicle has an invalid lane at {where}.")
    return True


def _sanitize_all_traffic_speeds(self, where):
    for vehicle in list(self.env.unwrapped.road.vehicles):
        if vehicle is self.env.unwrapped.vehicle:
            continue
        speed = float(getattr(vehicle, "speed", 0.0))
        if not np.isfinite(speed) or speed < 0.0:
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
HighwaySimulator._initialize_six_vehicle_traffic = _initialize_continuous_traffic
HighwaySimulator._maintain_six_vehicle_traffic = _maintain_continuous_traffic
HighwaySimulator._assert_traffic_invariant = _assert_continuous_traffic_invariant
HighwaySimulator._sanitize_traffic_slot_speeds = _sanitize_all_traffic_speeds
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



# ============================================================
# Predictive current-lane rear-threat feasibility
# ============================================================

REAR_ESCAPE_PREDICTION_HORIZON = 2.0
REAR_ESCAPE_PREDICTION_SAMPLES = 21
REAR_ESCAPE_MIN_GAP = float(LANE_CHANGE_ACTIVE_ABORT_GAP)


def _rear_escape_assessment(self, lane_ttc=None):
    """Test whether maximum front-safe acceleration can preserve a rear gap.

    This is a decision-support signal, not permission to violate adjacent-lane
    safety constraints. Traffic vehicles are assumed to maintain constant
    speed over the short prediction horizon (conservative acceleration
    uncertainty can be added in a later experiment).
    """
    if lane_ttc is None:
        lane_ttc = self._min_ttc_by_lane()

    ego = self.env.unwrapped.vehicle
    current_lane = int(ego.lane_index[2])
    values = lane_ttc.get(current_lane, {})
    rear_speed = values.get("rear_speed")
    rear_gap = float(values.get("rear_gap", float("inf")))
    rear_ttc = float(values.get("rear_ttc", float("inf")))
    ego_speed = max(0.0, float(ego.speed))

    result = {
        "rear_threat": False,
        "rear_escape_needed": False,
        "rear_ttc": rear_ttc,
        "rear_gap": rear_gap,
        "rear_speed": rear_speed,
        "max_front_safe_acceleration_mps2": 0.0,
        "predicted_min_rear_gap_m": rear_gap,
        "predicted_min_front_gap_m": float(values.get("front_gap", float("inf"))),
        "prediction_horizon_s": float(REAR_ESCAPE_PREDICTION_HORIZON),
        "reason": "no_imminent_rear_threat",
    }

    if rear_speed is None or float(rear_speed) <= ego_speed + 0.1 or rear_ttc > ttc_danger_threshold:
        return result

    result["rear_threat"] = True
    front_speed = values.get("front_speed")
    front_gap = float(values.get("front_gap", float("inf")))
    rear_speed = float(rear_speed)
    horizon = float(REAR_ESCAPE_PREDICTION_HORIZON)
    times = np.linspace(0.0, horizon, int(REAR_ESCAPE_PREDICTION_SAMPLES))

    # highway-env's continuous longitudinal action is physically bounded to
    # approximately +/-5 m/s^2; MAX_PHYSICAL_ACCELERATION is a reward scale.
    physical_accel_limit = float(min(
        5.0, max(0.0, float(getattr(ego, "MAX_ACCELERATION", 5.0)))
    ))
    feasible_accelerations = []
    best_rear_min_gap = -float("inf")
    best_acceleration = 0.0
    best_front_min_gap = front_gap

    for acceleration in np.linspace(0.0, physical_accel_limit, 51):
        ego_positions = ego_speed * times + 0.5 * acceleration * times ** 2
        predicted_rear_gap = rear_gap + (ego_speed - rear_speed) * times + 0.5 * acceleration * times ** 2
        min_rear_gap = float(np.min(predicted_rear_gap))

        if front_speed is not None and np.isfinite(front_gap):
            predicted_front_gap = (
                front_gap
                + (float(front_speed) - ego_speed) * times
                - 0.5 * acceleration * times ** 2
            )
            min_front_gap = float(np.min(predicted_front_gap))
            front_safe = min_front_gap >= float(MIN_FOLLOWING_GAP)
        else:
            min_front_gap = float("inf")
            front_safe = True

        if front_safe:
            feasible_accelerations.append((float(acceleration), min_rear_gap, min_front_gap))
            if min_rear_gap > best_rear_min_gap:
                best_rear_min_gap = min_rear_gap
                best_acceleration = float(acceleration)
                best_front_min_gap = min_front_gap

    # Zero acceleration should normally remain feasible unless the front gap
    # is already below the minimum. In that case, avoid treating acceleration
    # as safe and leave the acceleration decision to the ordinary front shield.
    if feasible_accelerations:
        max_front_safe_acceleration = max(item[0] for item in feasible_accelerations)
        # Re-evaluate rear clearance using the strongest acceleration that
        # remains front-safe, rather than choosing acceleration solely by rear gap.
        chosen = max(item for item in feasible_accelerations if item[0] == max_front_safe_acceleration)
        max_front_safe_acceleration, predicted_rear_min_gap, predicted_front_min_gap = chosen
    else:
        max_front_safe_acceleration = 0.0
        predicted_rear_min_gap = rear_gap + (ego_speed - rear_speed) * horizon
        predicted_front_min_gap = front_gap

    escape_needed = predicted_rear_min_gap < REAR_ESCAPE_MIN_GAP
    result.update({
        "rear_escape_needed": bool(escape_needed),
        "max_front_safe_acceleration_mps2": float(max_front_safe_acceleration),
        "predicted_min_rear_gap_m": float(predicted_rear_min_gap),
        "predicted_min_front_gap_m": float(predicted_front_min_gap),
        "reason": (
            "max_front_safe_acceleration_insufficient"
            if escape_needed else "rear_threat_manageable_by_front_safe_acceleration"
        ),
    })
    return result


def _select_safe_lane_with_rear_escape_priority(self, lane_ttc=None):
    """Keep learned target ranking normally; prioritize rear clearance in escape cases."""
    if lane_ttc is None:
        lane_ttc = self._min_ttc_by_lane()
    candidates = [
        item for item in self._safe_lane_change_candidates(lane_ttc)
        if int(item["target_lane"]) != int(self.env.unwrapped.vehicle.lane_index[2])
    ]
    if not candidates:
        return None

    assessment = self._rear_escape_assessment(lane_ttc)
    if assessment["rear_escape_needed"]:
        selected = max(
            candidates,
            key=lambda item: (
                float(item.get("rear_ttc", float("inf"))),
                float(item.get("rear_gap", float("inf"))),
                float(item.get("front_ttc", float("inf"))),
            ),
        )
        selected = dict(selected)
        selected["selection_principle"] = "rear_escape_priority_then_front_rear_safety"
        print(
            "[REAR-ESCAPE TARGET PRIORITY]",
            f"safe_targets={[int(c['target_lane']) for c in candidates]}",
            f"selected={int(selected['target_lane'])}",
            f"selected_rear_ttc={float(selected.get('rear_ttc', float('inf'))):.3f}",
            f"selected_rear_gap={float(selected.get('rear_gap', float('inf'))):.3f}",
        )
        return selected

    selected = max(
        candidates,
        key=lambda item: (
            float(item.get("utility_improvement", item["utility"])),
            -int(item["target_lane"]),
        ),
    )
    return selected


# ============================================================
# Non-intervening lane selector and prompt-ready episode evaluator
# ============================================================

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


def _non_intervening_lane_selector(self, sampled_action, selected_target_lane=None):
    """Resolve a physically valid target; never use TTC/EDM to veto or alter policy actions."""
    action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
    if action.size != 2:
        raise ValueError("Expected [normalized acceleration, binary KEEP/CHANGE]")
    current = int(self.env.unwrapped.vehicle.lane_index[2])
    requested = int(np.clip(np.rint(float(action[1])), 0, 1))
    mask = _physical_target_lane_mask(self)
    selected = current if selected_target_lane is None else int(selected_target_lane)
    valid = bool(0 <= selected < NUM_LANES and mask[selected])
    # This is a defensive interface invariant, not a risk-based rejection.
    # The policy's target distribution is already masked to these physical candidates.
    executed = requested == 1 and valid
    target = selected if executed else current
    info = {
        "current_lane_id": current,
        "requested_target_lane_id": int(selected),
        "selected_target_lane_id": int(selected),
        "executed_target_lane_id": int(target),
        "policy_requested_lane_action": "CHANGE" if requested else "KEEP",
        "executed_lane_action": "CHANGE" if executed else "KEEP",
        "policy_requested_binary_action": "CHANGE" if requested else "KEEP",
        "executed_binary_action": "CHANGE" if executed else "KEEP",
        "lane_change_blocked": bool(requested and not valid),
        "lane_change_rejection_reason": "invalid_physical_target" if requested and not valid else "none",
        "lane_request_direction": "LEFT" if target > current else "RIGHT" if target < current else "NONE",
        "lane_request_source_lane": current,
        "lane_request_target_lane": int(target),
        "lane_request_decision": "TARGET_SELECTED" if executed else "KEEP",
        "target_selector_mode": "physical_adjacent_lane_mask_only",
        "safety_filter_applied": False,
        "overridden": False,
        "emergency_braking": False,
        "acceleration_overridden": False,
        "edm_action_intervention": False,
        "valid_adjacent_lane_ids": np.flatnonzero(mask).astype(int).tolist(),
        "sampled_action": action.copy(),
        "filtered_action": np.asarray([float(np.clip(action[0], -1.0, 1.0)), float(int(executed))], dtype=np.float32),
        "proposed_policy_acceleration": float(action[0]),
        "executed_acceleration": float(np.clip(action[0], -1.0, 1.0)),
    }
    return info["filtered_action"], info


HighwaySimulator._rear_escape_assessment = _rear_escape_assessment
HighwaySimulator._select_best_safe_lane_candidate = _select_safe_lane_with_rear_escape_priority
HighwaySimulator.get_valid_lane_action_mask = _physical_lane_action_mask
HighwaySimulator.get_valid_target_lane_mask = _physical_target_lane_mask
HighwaySimulator.filter_ttc_action = _non_intervening_lane_selector




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
            acceleration_reward = float(
                (abs(clipped_speed_change) + clipped_speed_change)
                / (2.0 * max_acceleration_per_step)
            )

        # The separate speed reward is deliberately disabled for this run.
        speed_reward = 0.0

        collision_penalty = (
            max(0.0, float(reward_specification.get("safety_weight", INITIAL_COLLISION_PENALTY)))
            if bool(safety_violation) else 0.0
        )
        reward = float(acceleration_reward - collision_penalty)
        if reward_diagnostics is not None:
            reward_diagnostics.update({
                "reward_objective": "acceleration_reward_at_or_below_speed_limit_minus_unchanged_collision_penalty",
                "acceleration_reward": acceleration_reward,
                "ego_speed_change_per_step_mps": raw_speed_change,
                "clipped_speed_change_per_step_mps": clipped_speed_change,
                "max_acceleration_per_step_mps": float(max_acceleration_per_step),
                "speed_reward": 0.0,
                "speed_reward_enabled": False,
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
        target = rew if terminal else rew + 0.99 * nv
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
            "The separate speed reward is disabled. Apply the existing collision penalty "
            "unchanged and terminate the episode immediately on collision. Do not include "
            "a progress reward or lane-change reward/cost. Do not add an off-road reward "
            "penalty. EDM/TTC remain diagnostic only."
        ),
        "safety_weight": INITIAL_COLLISION_PENALTY,
        "max_acceleration_per_step_mps": float(MAX_ACCELERATION_PER_STEP_MPS),
        "policy_step_duration_s": float(POLICY_STEP_DURATION_S),
        "speed_limit_mps": float(SPEED_LIMIT),
        "speed_reward_enabled": False,
        "collision_terminates_episode": True,
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
            simulator.reset(randomize_traffic_gaps=True), dtype=np.float32
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
            filtered_action, safety_info = simulator.filter_ttc_action(
                sampled_action, selected_target_lane=selected_target_lane
            )
            filtered_action = np.asarray(filtered_action, dtype=np.float32).reshape(-1)
            record_lane_request_decision(
                lane_counters, sampled_action, valid_mask, safety_info or {}
            )
            previous_speed_mps = float(simulator.env.unwrapped.vehicle.speed)
            vehicles_before = snapshot_simulator_vehicles(simulator)
            next_state, _, done, step_info = simulator.step(
                filtered_action,
                apply_safety_filter=False,
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
                    step_info.get("filtered_policy_action", filtered_action)
                    if isinstance(step_info, dict) else filtered_action,
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
                    step_info.get("filtered_policy_action", filtered_action)
                    if isinstance(step_info, dict) else filtered_action,
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
            running_return = trajectory["rewards"][i] + 0.99 * running_return
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
            "reference_rollout_collection_enabled": False,
            "reference_offline_warm_start_enabled": False,
            "policy": "KEEP_CHANGE_then_masked_adjacent_target_then_conditional_squashed_gaussian_acceleration",
            "state_dim": STATE_DIM, "policy_action_dim": POLICY_ACTION_DIM,
            "execution_action_dim": ACTION_DIM, "num_lanes": NUM_LANES,
            "simulation_frequency_hz": simulation_frequency,
            "policy_frequency_hz": policy_frequency,
            "speed_limit_mps": SPEED_LIMIT,
            "max_physical_acceleration_mps2": MAX_PHYSICAL_ACCELERATION,
            "policy_step_duration_s": POLICY_STEP_DURATION_S,
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
    print("[ONLINE RL] Learning begins immediately from the randomly initialized policy.")

    for episode in range(NUM_ONLINE_EPISODES):
        simulator._trace_phase = "pure_rl"
        simulator._trace_episode = episode + 1
        start_time = time.time()
        state = simulator.reset(randomize_traffic_gaps=True)
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
            filtered_action, safety_info = simulator.filter_ttc_action(
                sampled_action, selected_target_lane=selected_target_lane
            )
            filtered_action = np.asarray(filtered_action, dtype=np.float32).reshape(-1)
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
                filtered_action,
                apply_safety_filter=False,
                sampled_action=sampled_action,
                safety_info=safety_info,
            )
            next_state = np.asarray(next_state, dtype=np.float32)
            executed_action = np.asarray(
                step_info.get("filtered_policy_action", filtered_action),
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
        mean_step_reward = episode_return / max(episode_steps, 1)
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
        lane_change_rate = float(lane_stats.get("completed", 0)) / max(episode_steps, 1)
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
            "mean_reward_per_step": float(mean_step_reward),
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
    print("Training: COLD START — random policy initialization, no reference rollouts, no offline warm-start; online Actor-Critic updates begin immediately.")
    print(f"Dimensions: state={STATE_DIM}, policy_action={POLICY_ACTION_DIM} [KEEP/CHANGE], execution_action={ACTION_DIM}.")
    print("Control: hierarchical stochastic policy selects KEEP/CHANGE, then a physically valid adjacent target, then conditional acceleration.")
    print("EDM/TTC: interpretation and diagnostics only; no action veto or acceleration override.")
    print("Episode termination: ENABLED on the first collision; no post-collision overlap transitions.")
    print("Lane-change processing: DISABLED (no reward suspension after lane changes).")
    print(f"CSV metrics: {METRICS_CSV_PATH}")
    print(f"Simulator: {NUM_LANES} lanes, {NUM_SURROUNDING_VEHICLES} managed traffic slots, {policy_frequency} Hz policy / {simulation_frequency} Hz simulation.")
    print(f"Acceleration bounds: ±{MAX_PHYSICAL_ACCELERATION:.1f} m/s²; speed limit: {SPEED_LIMIT:.1f} m/s.")
    print(f"Episode evaluator enabled: {ENABLE_EPISODE_EVALUATOR} (fixed reward baseline).")
    #print(f"Diagnostics: {DIAGNOSTIC_LOG_PATH}; trajectories: {EPISODE_TRAJECTORY_PATH if TRAJECTORY_RECORDING_ENABLED else 'disabled'}.")
    print("[REWARD CONFIG] speed reward DISABLED; acceleration reward is 1.0 at/above the speed limit; unchanged collision penalty; collision terminates episode.")
    simulator = create_highway_simulator()
    simulator.reset(seed=SEED)
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








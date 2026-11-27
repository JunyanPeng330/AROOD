"""
Q_EDM_ActorCritic.py

New AROOD baseline:
- Preserves the continuous three-lane ADT efficiency-testing simulator from
  diagnostics/Q_criticEDM.py.
- Replaces six raw surrounding-vehicle policy slots with a 6D ego state.
- Builds a fixed-order 12D EDM representation (three absolute lanes x
  front/rear TTC-to-clearance and physical bumper GAP).
- Supplies the identical [ego state, EDM] representation to Actor and Critic.
- EDM performs measurement only: no safety scores, reward shaping, horizon,
  action veto, action override, or lane recommendation.
- The Actor learns LEFT / KEEP / RIGHT and continuous acceleration.
- The only action mask enforces physical road boundaries.
- Native HighwayEnv geometry remains authoritative for collisions.

This is a separate experiment file; Q_criticEDM.py is not modified.
"""

"""I.py — consolidated AROOD hierarchical stochastic Actor-Critic experiment.

Contains the HighwayEnv adapter, observation/EDM interpretation, physical
adjacent-lane masks, hierarchical KEEP/CHANGE -> target-lane -> acceleration
policy, one-step lane-change reward processing, reward construction, online
Actor-Critic updates, and CSV/JSONL experiment logging. Instantaneous EDM TTC/GAP
features condition the Critic only; they never veto or rewrite the learned action.
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

        # Keep managed background traffic below the ego target speed.
        if (
            getattr(self, "_arood_traffic_manager", None) is not None
            and not bool(getattr(self, "crashed", False))
        ):
            try:
                _managed_speed = float(self.speed)
                _traffic_role = getattr(self, "_arood_traffic_role", "front")
                _role_speed_max = (
                    TRAFFIC_REAR_MAX_SPEED_MPS
                    if _traffic_role == "rear" else TRAFFIC_MAX_SPEED_MPS
                )
                if np.isfinite(_managed_speed) and _managed_speed > _role_speed_max:
                    self.speed = float(_role_speed_max)
            except (AttributeError, TypeError, ValueError):
                pass

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






# Shared physical geometry: every vehicle, including ego, is exactly 5 m long.
VEHICLE_LENGTH_M = 5.0
VEHICLE_HALF_LENGTH_M = VEHICLE_LENGTH_M / 2.0

# Keep HighwayEnv native collision geometry consistent with the distance model.
Vehicle.LENGTH = VEHICLE_LENGTH_M


def _longitudinal_bumper_gap(center_delta_m):
    """Signed bumper-to-bumper clearance for equal-length 5 m vehicles."""
    return abs(float(center_delta_m)) - 2.0 * VEHICLE_HALF_LENGTH_M


def _vehicle_length_width(vehicle):
    """Return finite positive body dimensions in metres."""
    try:
        length = float(getattr(vehicle, "LENGTH", VEHICLE_LENGTH_M))
    except (TypeError, ValueError):
        length = VEHICLE_LENGTH_M
    try:
        width = float(getattr(vehicle, "WIDTH", getattr(vehicle, "width", 2.0)))
    except (TypeError, ValueError):
        width = 2.0
    if not np.isfinite(length) or length <= 0.0:
        length = VEHICLE_LENGTH_M
    if not np.isfinite(width) or width <= 0.0:
        width = 2.0
    return length, width


def _oriented_vehicle_bodies_overlap(vehicle_a, vehicle_b, tolerance=1e-9):
    """Exact oriented-rectangle intersection test using the separating-axis theorem."""
    try:
        pa = np.asarray(vehicle_a.position, dtype=float).reshape(2)
        pb = np.asarray(vehicle_b.position, dtype=float).reshape(2)
        ha = float(getattr(vehicle_a, "heading", 0.0))
        hb = float(getattr(vehicle_b, "heading", 0.0))
    except (AttributeError, TypeError, ValueError):
        return False
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))
            and np.isfinite(ha) and np.isfinite(hb)):
        return False

    la, wa = _vehicle_length_width(vehicle_a)
    lb, wb = _vehicle_length_width(vehicle_b)

    def axes(heading):
        forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
        lateral = np.asarray([-math.sin(heading), math.cos(heading)], dtype=float)
        return forward, lateral

    a_axes = axes(ha)
    b_axes = axes(hb)
    displacement = pb - pa
    for axis in (*a_axes, *b_axes):
        radius_a = (0.5 * la * abs(float(np.dot(a_axes[0], axis)))
                    + 0.5 * wa * abs(float(np.dot(a_axes[1], axis))))
        radius_b = (0.5 * lb * abs(float(np.dot(b_axes[0], axis)))
                    + 0.5 * wb * abs(float(np.dot(b_axes[1], axis))))
        if abs(float(np.dot(displacement, axis))) > radius_a + radius_b + tolerance:
            return False
    return True


def _vehicle_longitudinal_half_extent(vehicle, reference_heading):
    """Project an oriented vehicle rectangle onto the road longitudinal axis."""
    length, width = _vehicle_length_width(vehicle)
    heading = float(getattr(vehicle, "heading", reference_heading))
    relative_heading = heading - float(reference_heading)
    return (
        0.5 * length * abs(math.cos(relative_heading))
        + 0.5 * width * abs(math.sin(relative_heading))
    )


def _vehicle_overlaps_lane_corridor(road, vehicle, lane_index):
    """Whether the vehicle body intersects a lane's lateral corridor.

    The test uses the oriented vehicle rectangle's lateral projection, not
    `vehicle.lane_index`, so vehicles straddling lane boundaries are relevant
    to every lane corridor their bodies actually enter.
    """
    try:
        lane = road.network.get_lane(lane_index)
        position = np.asarray(vehicle.position, dtype=float)
        longitudinal, lateral = lane.local_coordinates(position)


# ============================================================
# 1. ALL EXPERIMENT CONFIGURATION
# Edit this section only. Derived values are calculated below.
# ============================================================

# --- Experiment identity and outputs ---
EXPERIMENT_NAME = "AROOD_EDM_ActorCritic"
OUTPUT_DIR = "Q_EDM_ActorCritic"
MODEL_PATH = os.path.join("policy", "Q_EDM_ActorCritic.pt")
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "training_metrics.csv")
METRICS_JSONL_PATH = os.path.join(OUTPUT_DIR, "training_metrics.jsonl")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "experiment_config.json")
EPISODE_TRAJECTORY_PATH = os.path.join(OUTPUT_DIR, "episode_trajectories.jsonl")
DIAGNOSTIC_LOG_PATH = os.path.join(OUTPUT_DIR, "diagnostics.jsonl")
NEGATIVE_SPEED_FIRST_TRANSITION_PATH = os.path.join(OUTPUT_DIR, "negative_speed_transition.txt")
TRAJECTORY_RECORDING_ENABLED = os.environ.get("AROOD_RECORD_TRAJECTORIES", "1").strip().lower() not in {"0", "false", "no", "off"}
TRAJECTORY_RUN_ID = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"

# --- Reproducibility ---
SEED = 42

# --- Training ---
NUM_ONLINE_EPISODES = 100
MAX_EPISODE_STEPS = 1000                 # RL decisions per episode
GAMMA = 1.0                              # Discount per policy decision
POLICY_LR = 3e-4
HIDDEN_DIM = 128
VALUE_LOSS_WEIGHT = 0.5
MAX_GRAD_NORM = 1.0
INITIAL_ACCELERATION_LOG_STD = -0.5
CHECKPOINT_INTERVAL_EPISODES = 10

# --- Simulator and policy timing ---
SIMULATION_FREQUENCY_HZ = 15             # Physical simulator updates per second
POLICY_FREQUENCY_HZ = 15                 # Agent decisions per second
NUM_LANES = 3
SPEED_LIMIT_MPS = 40.0
MAX_PHYSICAL_ACCELERATION_MPS2 = 5.0
INSTANT_LANE_CHANGE_MODE = True
LANE_CHANGE_PROCESSING_STEPS = 0

# --- Vehicle geometry and EDM semantics ---
VEHICLE_LENGTH_M = 5.0
VEHICLE_WIDTH_M = 2.0
TTC_CLEARANCE_MARGIN_M = 5.0
EDM_TTC_INPUT_MAX_S = 30.0
EDM_GAP_INPUT_MAX_M = 200.0
EDM_MISSING_VALUE = 1.1                 # Distinct from present/non-closing (1.0)
EDM_LANE_COUNT = NUM_LANES
EDM_FEATURES_PER_LANE = 4                # front TTC, front GAP, rear TTC, rear GAP
EDM_FEATURE_DIM = EDM_LANE_COUNT * EDM_FEATURES_PER_LANE
EGO_STATE_DIM = 6                        # speed, acceleration, 3-lane one-hot, episode progress
STATE_DIM = EGO_STATE_DIM
ACTION_DIM = 2                            # simulator adapter: [normalized acceleration, KEEP/CHANGE]

# --- Reward: actual environmental outcomes only; no EDM-derived terms ---
COLLISION_PENALTY_MULTIPLIER = 10.0
ACCELERATION_REWARD_AT_OR_ABOVE_SPEED_LIMIT = 1.0
SPEED_REWARD_ENABLED = False

# --- Existing efficiency-testing ADT traffic environment (preserved) ---
TRAFFIC_REFERENCE_FRONT_M = 600.0
TRAFFIC_REFERENCE_REAR_M = 400.0
TRAFFIC_ENTRY_BUFFER_M = 200.0
TRAFFIC_RECYCLING_BUFFER_M = 400.0
TRAFFIC_CROSS_LANE_STAGGER_MIN_M = 35.0
TRAFFIC_FRONT_RECYCLE_BOUNDARY_M = TRAFFIC_REFERENCE_FRONT_M + TRAFFIC_ENTRY_BUFFER_M + TRAFFIC_RECYCLING_BUFFER_M
TRAFFIC_REAR_RECYCLE_BOUNDARY_M = TRAFFIC_REFERENCE_REAR_M + TRAFFIC_ENTRY_BUFFER_M
TRAFFIC_FRONT_SPAWN_MIN_M = TRAFFIC_REFERENCE_FRONT_M + 0.5 * TRAFFIC_ENTRY_BUFFER_M
TRAFFIC_FRONT_SPAWN_MAX_M = TRAFFIC_FRONT_RECYCLE_BOUNDARY_M - 100.0
TRAFFIC_REAR_SPAWN_MIN_M = TRAFFIC_REFERENCE_REAR_M + 0.5 * TRAFFIC_ENTRY_BUFFER_M
TRAFFIC_REAR_SPAWN_MAX_M = TRAFFIC_REFERENCE_REAR_M + TRAFFIC_ENTRY_BUFFER_M

MAX_SURROUNDING_TRAFFIC_VEHICLES = None
TRAFFIC_TARGET_COUNT = None
TRAFFIC_EXPECTED_REFERENCE_POPULATION = None
TRAFFIC_INITIAL_FRONT_VEHICLES_PER_LANE = 4
TRAFFIC_MIN_FRONT_VEHICLES_PER_LANE = 1
TRAFFIC_INITIAL_REAR_VEHICLES_PER_LANE = 1
MIN_INITIAL_GAP_M = 40.0
MIN_INSERTION_GAP_M = 40.0
TRAFFIC_MIN_SPEED_MPS = 25.0
TRAFFIC_MAX_SPEED_MPS = 36.0
TRAFFIC_REAR_MAX_SPEED_MPS = 50.0

TRAFFIC_MAX_ACCEL_KMH_PER_S = 10.0
IDM_MAX_ACCEL_MPS2 = TRAFFIC_MAX_ACCEL_KMH_PER_S / 3.6
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
ADT_AGGRESSIVE_MAX_ACCEL_MPS2 = IDM_MAX_ACCEL_MPS2
ADT_AGGRESSIVE_BRAKING_MPS2 = 3.0
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
ADT_UNSTABLE_ACCEL_PULSE_MPS2 = IDM_MAX_ACCEL_MPS2
ADT_SPEED_VARIATION_MPS = 3.0

PROFILE_DESIRED_SPEED_RANGES_MPS = {
    "normal": (29.0, 35.0),
    "aggressive": (33.0, 36.0),
    "unstable": (27.0, 35.0),
    "slow": (25.0, 30.0),
}
PROFILE_REAR_DESIRED_SPEED_RANGES_MPS = {
    "normal": (28.0, 42.0),
    "aggressive": (36.0, 50.0),
    "unstable": (25.0, 45.0),
    "slow": (25.0, 32.0),
}
PROFILE_IDM_HEADWAY_S = {"normal": 1.5, "aggressive": 1.0, "unstable": 1.2, "slow": 1.8}
PROFILE_IDM_MIN_GAP_M = {"normal": 5.0, "aggressive": 3.0, "unstable": 3.5, "slow": 6.0}
PROFILE_LANE_CHANGE_RATE_PER_S = {
    "normal": 0.06,
    "aggressive": ADT_OVERTAKER_LANE_CHANGE_RATE_PER_S,
    "unstable": ADT_UNSTABLE_LANE_CHANGE_RATE_PER_S,
    "slow": 0.03,
}
INITIAL_SPAWN_MAX_ATTEMPTS = 50000
BOUNDARY_SPAWN_MAX_ATTEMPTS = 200

# --- Simulator observation internals (not policy traffic observations) ---
NUM_SURROUNDING_VEHICLES = 6
VEHICLE_FEATURES = 7
NUM_VEHICLE_ROWS = NUM_SURROUNDING_VEHICLES + 1

# ============================================================
# Derived configuration. Do not tune these independently.
# ============================================================
if SIMULATION_FREQUENCY_HZ <= 0 or POLICY_FREQUENCY_HZ <= 0:
    raise ValueError("Simulation and policy frequencies must be positive.")
if SIMULATION_FREQUENCY_HZ % POLICY_FREQUENCY_HZ != 0:
    raise ValueError("SIMULATION_FREQUENCY_HZ must be an integer multiple of POLICY_FREQUENCY_HZ.")
if MAX_EPISODE_STEPS <= 0 or NUM_ONLINE_EPISODES <= 0:
    raise ValueError("Episode counts and episode steps must be positive.")
if not 0.0 < GAMMA <= 1.0:
    raise ValueError("GAMMA must be in (0, 1].")
if not 0.0 < SPEED_LIMIT_MPS:
    raise ValueError("SPEED_LIMIT_MPS must be positive.")

SIMULATION_DT_S = 1.0 / SIMULATION_FREQUENCY_HZ
POLICY_DT_S = 1.0 / POLICY_FREQUENCY_HZ
SIMULATION_STEPS_PER_POLICY_STEP = SIMULATION_FREQUENCY_HZ // POLICY_FREQUENCY_HZ
MAX_EPISODE_DURATION_S = MAX_EPISODE_STEPS * POLICY_DT_S
DISCOUNT_TIME_CONSTANT_S = float("inf") if GAMMA == 1.0 else -POLICY_DT_S / math.log(GAMMA)
MAX_ACCELERATION_PER_POLICY_STEP_MPS = MAX_PHYSICAL_ACCELERATION_MPS2 * POLICY_DT_S

# Compatibility aliases required by the retained HighwayEnv/ADT implementation.
SPEED_LIMIT = SPEED_LIMIT_MPS
MAX_PHYSICAL_ACCELERATION = MAX_PHYSICAL_ACCELERATION_MPS2
MAX_ACCELERATION_PER_STEP_MPS = MAX_ACCELERATION_PER_POLICY_STEP_MPS
MAX_STEPS = MAX_EPISODE_STEPS
simulation_frequency = SIMULATION_FREQUENCY_HZ
policy_frequency = POLICY_FREQUENCY_HZ
SIMULATION_STEP_DURATION_S = SIMULATION_DT_S
POLICY_STEP_DURATION_S = POLICY_DT_S
SIMULATION_SUBSTEPS_PER_POLICY = SIMULATION_STEPS_PER_POLICY_STEP
TRAFFIC_UPDATE_FREQUENCY_HZ = SIMULATION_FREQUENCY_HZ
TRAFFIC_UPDATE_DT_S = SIMULATION_DT_S
SPEED_LIMIT_MPS = SPEED_LIMIT
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
            "length_m": VEHICLE_LENGTH_M,
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
        "traffic_update_frequency_hz": int(TRAFFIC_UPDATE_FREQUENCY_HZ),
        "traffic_update_step_duration_s": float(TRAFFIC_UPDATE_DT_S),
        "simulation_substeps_per_policy_action": int(SIMULATION_SUBSTEPS_PER_POLICY),
        "episode_duration_s": float(MAX_EPISODE_DURATION_S),
        "discount_gamma_per_policy_step": float(GAMMA),
        "traffic_reference_front_m": float(TRAFFIC_REFERENCE_FRONT_M),
        "traffic_reference_rear_m": float(TRAFFIC_REFERENCE_REAR_M),
        "traffic_front_recycle_boundary_m": float(TRAFFIC_FRONT_RECYCLE_BOUNDARY_M),
        "traffic_rear_recycle_boundary_m": float(TRAFFIC_REAR_RECYCLE_BOUNDARY_M),
        "traffic_max_surrounding_vehicles": None,
        "traffic_population_cap_enabled": False,
        "traffic_episode_population_target": None,
        "traffic_design": "uncapped_staggered_overtaking_with_per_lane_front_coverage",
        "traffic_initial_front_vehicles_per_lane": int(TRAFFIC_INITIAL_FRONT_VEHICLES_PER_LANE),
        "traffic_initial_rear_vehicles_per_lane": int(TRAFFIC_INITIAL_REAR_VEHICLES_PER_LANE),
        "traffic_front_max_speed_mps": float(TRAFFIC_MAX_SPEED_MPS),
        "traffic_rear_max_speed_mps": float(TRAFFIC_REAR_MAX_SPEED_MPS),
        "traffic_actual_initial_population": int(
            len(simulator.env.unwrapped.road.vehicles) - 1
        ),
        "num_lanes": int(NUM_LANES),
        "managed_surrounding_vehicle_slots": int(NUM_SURROUNDING_VEHICLES),
        "speed_limit_mps": float(SPEED_LIMIT),
        "ttc_clearance_margin_m": float(TTC_CLEARANCE_MARGIN_M),
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






# ============================================================
# 2. EDM-conditioned Actor-Critic
# Both Actor and Critic receive the identical [ego state, 12 EDM features].
# The only action mask is physical road-boundary validity.
# ============================================================

def build_actor_lane_mask(current_lane_id):
    """Mask only lane changes that would leave the physical three-lane road."""
    current_lane_id = int(current_lane_id)
    # Categorical indices 0/1/2 mean LEFT/KEEP/RIGHT = lane delta -1/0/+1.
    return np.asarray([
        0 <= relative_lane_delta_to_absolute_lane(current_lane_id, -1) < NUM_LANES,
        True,
        0 <= relative_lane_delta_to_absolute_lane(current_lane_id, +1) < NUM_LANES,
    ], dtype=np.bool_)


def combine_ego_and_edm(ego_state, edm_features):
    ego = to_tensor(ego_state)
    edm = to_tensor(edm_features)
    if ego.ndim == 1:
        ego = ego.unsqueeze(0)
    if edm.ndim == 1:
        edm = edm.unsqueeze(0)
    if ego.shape[0] == 1 and edm.shape[0] > 1:
        ego = ego.expand(edm.shape[0], -1)
    if edm.shape[0] == 1 and ego.shape[0] > 1:
        edm = edm.expand(ego.shape[0], -1)
    if ego.shape[-1] != EGO_STATE_DIM or edm.shape[-1] != EDM_FEATURE_DIM:
        raise ValueError(
            f"Expected ego/EDM dimensions {EGO_STATE_DIM}/{EDM_FEATURE_DIM}; "
            f"received {ego.shape[-1]}/{edm.shape[-1]}."
        )
    if ego.shape[0] != edm.shape[0]:
        raise ValueError("Ego state and EDM batch sizes do not match.")
    return torch.cat([ego, edm], dim=-1)


class EDMActorCritic(nn.Module):
    """One shared EDM-conditioned representation with Actor and Critic heads."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(EGO_STATE_DIM + EDM_FEATURE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.lane_head = nn.Linear(HIDDEN_DIM, 3)
        self.acceleration_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + 3, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        self.acceleration_log_std = nn.Parameter(
            torch.tensor([INITIAL_ACCELERATION_LOG_STD], dtype=torch.float32)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        self.optimizer = Adam(self.parameters(), lr=POLICY_LR)

    def latent(self, ego_state, edm_features):
        return self.encoder(combine_ego_and_edm(ego_state, edm_features))

    def lane_distribution(self, latent, valid_lane_mask):
        logits = self.lane_head(latent)
        mask = torch.as_tensor(valid_lane_mask, dtype=torch.bool, device=logits.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and logits.shape[0] > 1:
            mask = mask.expand(logits.shape[0], -1)
        if mask.shape != logits.shape or not mask.any(dim=-1).all():
            raise ValueError("Invalid physical lane-action mask.")
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))

    def acceleration_distribution(self, latent, lane_action_index):
        lane_index = torch.as_tensor(
            lane_action_index, dtype=torch.long, device=latent.device
        ).reshape(-1)
        if lane_index.numel() == 1 and latent.shape[0] > 1:
            lane_index = lane_index.expand(latent.shape[0])
        lane_one_hot = F.one_hot(lane_index.clamp(0, 2), num_classes=3).to(latent.dtype)
        mean = self.acceleration_head(torch.cat([latent, lane_one_hot], dim=-1)).squeeze(-1)
        log_std = self.acceleration_log_std.clamp(-3.0, 0.5).expand_as(mean)
        return torch.distributions.Normal(mean, log_std.exp())

    def value(self, ego_state, edm_features):
        latent = self.latent(ego_state, edm_features)
        return self.critic_head(latent).squeeze(-1)

    @staticmethod
    def squashed_log_probability(normal_dist, pre_tanh):
        log_prob = normal_dist.log_prob(pre_tanh)
        correction = torch.log(1.0 - torch.tanh(pre_tanh).pow(2) + 1e-6)
        return log_prob - correction

    def act(self, ego_state, edm_features, current_lane_id, deterministic=False):
        latent = self.latent(ego_state, edm_features)
        valid_mask = build_actor_lane_mask(current_lane_id)
        lane_dist = self.lane_distribution(latent, valid_mask)
        lane_index = (
            torch.argmax(lane_dist.logits, dim=-1)
            if deterministic else lane_dist.sample()
        )
        accel_dist = self.acceleration_distribution(latent, lane_index)
        pre_tanh = accel_dist.mean if deterministic else accel_dist.sample()
        acceleration = torch.tanh(pre_tanh)
        log_prob = (
            lane_dist.log_prob(lane_index)
            + self.squashed_log_probability(accel_dist, pre_tanh)
        )
        lane_action_delta = int(lane_index[0].detach().cpu().item()) - 1
        action = np.asarray(
            [float(acceleration[0].detach().cpu().item()), lane_action_delta],
            dtype=np.float32,
        )
        aux = {
            "lane_action_index": int(lane_index[0].detach().cpu().item()),
            "lane_action_delta": lane_action_delta,
            "pre_tanh_acceleration": float(pre_tanh[0].detach().cpu().item()),
            "log_probability": float(log_prob[0].detach().cpu().item()),
            "valid_lane_action_mask": valid_mask.tolist(),
        }
        return action, aux

    def evaluate_sample(self, ego_state, edm_features, current_lane_id,
                        lane_action_index, pre_tanh_acceleration):
        latent = self.latent(ego_state, edm_features)
        lane_mask = build_actor_lane_mask(current_lane_id)
        lane_dist = self.lane_distribution(latent, lane_mask)
        lane_index = torch.as_tensor(
            [int(lane_action_index)], dtype=torch.long, device=DEVICE
        )
        accel_dist = self.acceleration_distribution(latent, lane_index)
        pre_tanh = torch.as_tensor(
            [float(pre_tanh_acceleration)], dtype=torch.float32, device=DEVICE
        )
        log_prob = (
            lane_dist.log_prob(lane_index)
            + self.squashed_log_probability(accel_dist, pre_tanh)
        )
        value = self.critic_head(latent).squeeze(-1)
        return log_prob, value


class HighwaySimulator:
    """Highway-env wrapper with ADT traffic and six ego observation slots.

    The Actor observation contains vehicle kinematics and ego-relative context.
    Instantaneous TTC/GAP features for current, left and right lane roles are
    exposed separately through get_critic_edm_features() for the Critic only.
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
                # policy action: 10 Hz / 5 Hz = 2 simulator updates per decision.
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

        # EDM is excluded from the Actor observation and supplied separately
        # to the Critic as instantaneous TTC/GAP features.
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


    def _install_simulator_rate_traffic_maintenance(self):
        """Run population-boundary maintenance after every physical road update."""
        road = self.env.unwrapped.road
        if bool(getattr(road, "_arood_traffic_maintenance_wrapper_installed", False)):
            return
        original_road_step = road.step

        def _road_step_then_maintain(dt):
            result = original_road_step(dt)
            self._maintain_continuous_traffic()
            return result

        road.step = _road_step_then_maintain
        road._arood_traffic_maintenance_wrapper_installed = True

    def reset(self, seed=None):

        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)

        # Overtaking scenario uses uncapped, staggered traffic with guaranteed
        # front-vehicle coverage; six observation slots remain independent of
        # the total number of vehicles in the world.
        print(
            "[ADT TRAFFIC SCENARIO]",
            "population_cap=NONE",
            f"initial_front_vehicles_per_lane={TRAFFIC_INITIAL_FRONT_VEHICLES_PER_LANE}",
            f"minimum_front_vehicles_per_lane={TRAFFIC_MIN_FRONT_VEHICLES_PER_LANE}",
            f"reference_front=+{TRAFFIC_REFERENCE_FRONT_M:.0f}m",
            f"reference_rear=-{TRAFFIC_REFERENCE_REAR_M:.0f}m",
            f"entry_buffer={TRAFFIC_ENTRY_BUFFER_M:.0f}m",
            f"recycle_front=+{TRAFFIC_FRONT_RECYCLE_BOUNDARY_M:.0f}m",
            f"recycle_rear=-{TRAFFIC_REAR_RECYCLE_BOUNDARY_M:.0f}m",
            f"front_desired_speed_range={TRAFFIC_MIN_SPEED_MPS:.0f}-{TRAFFIC_MAX_SPEED_MPS:.0f}m/s",
        f"rear_desired_speed_range={TRAFFIC_MIN_SPEED_MPS:.0f}-{TRAFFIC_REAR_MAX_SPEED_MPS:.0f}m/s",
            f"ordinary_acceleration_cap={IDM_MAX_ACCEL_MPS2:.1f}m/s2",
            f"ordinary_braking_cap={IDM_COMFORTABLE_BRAKING_MPS2:.1f}m/s2",
            f"emergency_braking_cap={ADT_EMERGENCY_BRAKING_MPS2:.1f}m/s2",
            f"profile_mix={dict(ADT_PROFILE_MIX)}",
            f"simulation_frequency={simulation_frequency}Hz",
            "ego_observation_slots=6",
            "traffic_speed_floor_applies_to_all_non_crashed_managed_vehicles",
        )

        self.step_count = 0
        # One collision event is recorded per uninterrupted contact episode.
        # The simulator crash latch is still cleared after each update so the
        # RL episode can continue, but sustained overlap is not re-counted as
        # a fresh impact on every subsequent policy step.
        self._collision_contact_active = False
        base_env = self.env.unwrapped
        self._install_simulator_rate_traffic_maintenance()
        # Start the ego at its 40 m/s desired cruising speed so the slower
        # traffic creates an immediate, repeatable overtaking task.
        base_env.vehicle.LENGTH = VEHICLE_LENGTH_M
        base_env.vehicle.speed = float(SPEED_LIMIT_MPS)
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
        """Return geometrically relevant front/rear neighbours and physical/buffer TTC.

        Candidate vehicles are included in a lane only when their oriented
        bodies intersect that lane's lateral corridor. The nearest candidate
        is selected by longitudinal bumper clearance, not centre distance.

        `front_ttc`/`rear_ttc` remain physical longitudinal TTC estimates.
        `*_ttc_buffer` is the time until the additional configured clearance
        reserve is reached. Neither metric replaces HighwayEnv's actual
        oriented-body collision detection.
        """
        base_env = self.env.unwrapped
        road = base_env.road
        ego = base_env.vehicle
        lanes_count = int(base_env.config["lanes_count"])
        ego_lane = int(ego.lane_index[2])
        ego_speed = max(0.0, float(ego.speed))
        ego_s = float(self._road_longitudinal_position(ego))
        ego.LENGTH = VEHICLE_LENGTH_M

        lane_ttc = {
            lane_id: {
                "front_ttc": float("inf"), "rear_ttc": float("inf"),
                "front_ttc_buffer": float("inf"), "rear_ttc_buffer": float("inf"),
                "min_ttc": float("inf"),
                "front_gap": float("inf"), "rear_gap": float("inf"),
                "front_ttc_effective_gap": float("inf"),
                "rear_ttc_effective_gap": float("inf"),
                "front_speed": None, "rear_speed": None,
                "nearest_front_ttc": float("inf"),
                "front_center_gap": float("inf"),
                "rear_center_gap": float("inf"),
                "front_vehicle_id": None, "rear_vehicle_id": None,
                "front_body_overlap": False, "rear_body_overlap": False,
                "front_lane_corridor_overlap": False,
                "rear_lane_corridor_overlap": False,
                "front_lateral_overlap": False,
                "rear_lateral_overlap": False,
            }
            for lane_id in range(lanes_count)
        }

        nearest = {
            (lane_id, side): None
            for lane_id in range(lanes_count)
            for side in ("front", "rear")
        }
        reference_lane_index = (ego.lane_index[0], ego.lane_index[1], 0)
        reference_lane = road.network.get_lane(reference_lane_index)

        for other in list(road.vehicles):
            if other is ego or bool(getattr(other, "crashed", False)):
                continue
            try:
                other_s = float(self._road_longitudinal_position(other))
                other_speed = max(0.0, float(other.speed))
                other.LENGTH = VEHICLE_LENGTH_M
                other_half_extent = _vehicle_longitudinal_half_extent(
                    other, float(reference_lane.heading_at(other_s))
                )
            except (AttributeError, TypeError, ValueError, IndexError, KeyError):
                continue
            if not (np.isfinite(other_s) and np.isfinite(other_speed)):
                continue

            center_gap = other_s - ego_s
            if (
                center_gap > TRAFFIC_REFERENCE_FRONT_M
                or center_gap < -TRAFFIC_REFERENCE_REAR_M
            ):
                continue

            # Longitudinally aligned centres are retained in both diagnostics,
            # avoiding an arbitrary front/rear classification for alongside cars.
            candidate_sides = (
                ("front", "rear") if abs(center_gap) <= 1e-9
                else (("front",) if center_gap > 0.0 else ("rear",))
            )
            ego_heading = float(reference_lane.heading_at(ego_s))
            ego_half_extent = _vehicle_longitudinal_half_extent(ego, ego_heading)
            physical_gap = abs(center_gap) - ego_half_extent - other_half_extent

            for lane_id in range(lanes_count):
                lane_index = self._lane_index_for_id(lane_id)
                if not _vehicle_overlaps_lane_corridor(road, other, lane_index):
                    continue

                lane = road.network.get_lane(lane_index)
                other_longitudinal, other_lateral = lane.local_coordinates(
                    np.asarray(other.position, dtype=float)
                )
                lane_heading = float(lane.heading_at(other_longitudinal))
                ego_heading_in_lane = float(lane.heading_at(ego_s))
                ego_length, ego_width = _vehicle_length_width(ego)
                other_length, other_width = _vehicle_length_width(other)
                ego_relative_heading = float(ego.heading) - ego_heading_in_lane
                other_relative_heading = float(other.heading) - lane_heading
                ego_lateral_half_extent = (
                    0.5 * ego_width * abs(math.cos(ego_relative_heading))
                    + 0.5 * ego_length * abs(math.sin(ego_relative_heading))
                )
                other_lateral_half_extent = (
                    0.5 * other_width * abs(math.cos(other_relative_heading))
                    + 0.5 * other_length * abs(math.sin(other_relative_heading))
                )
                # For a candidate lane, assume the ego completes its
                # instantaneous lane transition and is centred in that lane.
                lateral_body_overlap = bool(
                    abs(float(other_lateral))
                    <= ego_lateral_half_extent + other_lateral_half_extent
                )

                # Select the smallest signed bumper clearance among vehicles
                # whose bodies actually enter this lane's lateral corridor.
                for key_side in candidate_sides:
                    key = (lane_id, key_side)
                    current = nearest[key]
                    if current is None or physical_gap < current["physical_gap"]:
                        nearest[key] = {
                            "physical_gap": float(physical_gap),
                            "center_gap": float(center_gap),
                            "speed": float(other_speed),
                            "vehicle_id": id(other),
                            "body_overlap": bool(_oriented_vehicle_bodies_overlap(ego, other)),
                            "lane_corridor_overlap": True,
                            "lateral_overlap": lateral_body_overlap,
                        }

        def _ttc_from_gap(gap, closing_speed):
            # A nonpositive clearance means contact/boundary breach is already
            # present; report zero even when relative speed is not closing.
            if gap <= 0.0:
                return 0.0
            if closing_speed <= 1e-6:
                return float("inf")
            return float(gap / closing_speed)

        for (lane_id, side), neighbor in nearest.items():
            if neighbor is None:
                continue
            values = lane_ttc[lane_id]
            physical_gap = float(neighbor["physical_gap"])
            effective_gap = physical_gap - float(TTC_CLEARANCE_MARGIN_M)
            other_speed = float(neighbor["speed"])
            if side == "front":
                closing_speed = ego_speed - other_speed
            else:
                closing_speed = other_speed - ego_speed

            physical_ttc = _ttc_from_gap(physical_gap, closing_speed)
            buffer_ttc = _ttc_from_gap(effective_gap, closing_speed)

            if side == "front":
                values.update({
                    "front_gap": physical_gap,
                    "front_ttc_effective_gap": effective_gap,
                    "front_speed": other_speed,
                    "front_ttc": physical_ttc,
                    "front_ttc_buffer": buffer_ttc,
                    "nearest_front_ttc": physical_ttc,
                    "front_center_gap": abs(float(neighbor["center_gap"])),
                    "front_vehicle_id": neighbor["vehicle_id"],
                    "front_body_overlap": neighbor["body_overlap"],
                    "front_lane_corridor_overlap": neighbor["lane_corridor_overlap"],
                    "front_lateral_overlap": neighbor["lateral_overlap"],
                })
            else:
                values.update({
                    "rear_gap": physical_gap,
                    "rear_ttc_effective_gap": effective_gap,
                    "rear_speed": other_speed,
                    "rear_ttc": physical_ttc,
                    "rear_ttc_buffer": buffer_ttc,
                    "rear_center_gap": abs(float(neighbor["center_gap"])),
                    "rear_vehicle_id": neighbor["vehicle_id"],
                    "rear_body_overlap": neighbor["body_overlap"],
                    "rear_lane_corridor_overlap": neighbor["lane_corridor_overlap"],
                    "rear_lateral_overlap": neighbor["lateral_overlap"],
                })

        for values in lane_ttc.values():
            values["min_ttc"] = min(values["front_ttc"], values["rear_ttc"])
        return lane_ttc

    def get_critic_edm_features(self):
        """Normalized instantaneous TTC/GAP for current, left and right lanes.

        Order per role: front TTC-to-5m-clearance, front physical bumper GAP,
        rear TTC-to-5m-clearance, rear physical bumper GAP. Infinite TTC/GAP
        maps to the configured maximum; no risk score or future rollout exists.
        """
        metrics = self._min_ttc_by_lane()
        ego_lane = int(self.env.unwrapped.vehicle.lane_index[2])
        features = []
        for lane_id in (ego_lane, ego_lane + 1, ego_lane - 1):
            values = metrics.get(lane_id, {})
            for ttc_key, gap_key in (
                ("front_ttc_buffer", "front_gap"),
                ("rear_ttc_buffer", "rear_gap"),
            ):
                try:
                    ttc = float(values.get(ttc_key, float("inf")))
                except (TypeError, ValueError):
                    ttc = float("inf")
                try:
                    gap = float(values.get(gap_key, float("inf")))
                except (TypeError, ValueError):
                    gap = float("inf")
                if not np.isfinite(ttc):
                    ttc = EDM_TTC_INPUT_MAX_S
                if not np.isfinite(gap):
                    gap = EDM_GAP_INPUT_MAX_M
                features.extend((
                    float(np.clip(ttc / EDM_TTC_INPUT_MAX_S, 0.0, 1.0)),
                    float(np.clip(gap / EDM_GAP_INPUT_MAX_M, -1.0, 1.0)),
                ))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (EDM_FEATURE_DIM,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"Invalid instantaneous EDM feature vector: {result}")
        return result

    def _edm_interpretation_by_lane(self, lane_metrics=None):
        """Raw TTC/GAP snapshot for diagnostics; no categories or risk scores."""
        lane_metrics = self._min_ttc_by_lane() if lane_metrics is None else lane_metrics
        return {
            int(lane_id): {
                "front_ttc_s": float(values.get("front_ttc", float("inf"))),
                "rear_ttc_s": float(values.get("rear_ttc", float("inf"))),
                "front_ttc_to_margin_s": float(values.get("front_ttc_buffer", float("inf"))),
                "rear_ttc_to_margin_s": float(values.get("rear_ttc_buffer", float("inf"))),
                "front_bumper_gap_m": float(values.get("front_gap", float("inf"))),
                "rear_bumper_gap_m": float(values.get("rear_gap", float("inf"))),
                "front_gap_after_margin_m": float(values.get("front_ttc_effective_gap", float("inf"))),
                "rear_gap_after_margin_m": float(values.get("rear_ttc_effective_gap", float("inf"))),
                "front_body_overlap": bool(values.get("front_body_overlap", False)),
                "rear_body_overlap": bool(values.get("rear_body_overlap", False)),
                "front_vehicle_id": values.get("front_vehicle_id"),
                "rear_vehicle_id": values.get("rear_vehicle_id"),
            }
            for lane_id, values in sorted(lane_metrics.items())
        }

    # EDM provides instantaneous measurements only; it never predicts trajectories or intervenes.


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
                    longitudinal, _ = source_lane.local_coordinates(
                        np.asarray(ego.position, dtype=float)
                    )
                    speed_before_lateral_snap = float(ego.speed)
                    target_lane_index = (
                        source_lane_index[0], source_lane_index[1], int(requested_target_lane_id)
                    )
                    target_lane = base_env.road.network.get_lane(target_lane_index)

                    # Split the maneuver into independent dimensions:
                    #   * longitudinal: preserve s and speed exactly at the instant
                    #     of the lane decision; normal vehicle physics integrates
                    #     acceleration and forward motion in env.step() below.
                    #   * lateral: snap directly to the target lane centre now;
                    #     there is no lateral interpolation or maneuver duration.
                    # This changes the ego's lane occupancy before native physics,
                    # so the normal collision geometry sees the destination lane.
                    target_position = np.asarray(
                        target_lane.position(longitudinal, 0.0), dtype=float
                    )
                    target_heading = float(target_lane.heading_at(longitudinal))
                    ego.position = target_position
                    ego.heading = target_heading
                    ego.lane_index = tuple(target_lane_index)

                    # Verify that the lateral snap did not introduce a longitudinal
                    # jump or alter speed. The subsequent env.step() is responsible
                    # for all normal longitudinal integration.
                    target_longitudinal_after_snap, target_lateral_after_snap = (
                        target_lane.local_coordinates(np.asarray(ego.position, dtype=float))
                    )
                    longitudinal_snap_error_m = float(
                        target_longitudinal_after_snap - longitudinal
                    )
                    speed_snap_error_mps = float(ego.speed) - speed_before_lateral_snap
                    if abs(longitudinal_snap_error_m) > 1e-6 or abs(speed_snap_error_mps) > 1e-9:
                        raise RuntimeError(
                            "Instantaneous lateral lane-change altered longitudinal "
                            f"state: ds={longitudinal_snap_error_m:.9g}m, "
                            f"dV={speed_snap_error_mps:.9g}m/s"
                        )

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
                        longitudinal_snap_error_m=float(longitudinal_snap_error_m),
                        lateral_position_after_snap_m=float(target_lateral_after_snap),
                        speed_before_lateral_snap_mps=float(speed_before_lateral_snap),
                        speed_snap_error_mps=float(speed_snap_error_mps),
                        lateral_transition="instantaneous",
                        longitudinal_dynamics="normal_simulator_integration",
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
        # Steering is deliberately zero: lateral lane occupancy was already
        # changed instantaneously above. Keep the learned acceleration untouched
        # so longitudinal motion follows the ordinary HighwayEnv dynamics.
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
        # Population maintenance runs after each 10 Hz Road.step(), rather
        # than once per 5 Hz policy decision.
        obs = base_env.observation_type.observe()
        next_state = self._encode_state(obs)
        self._assert_continuous_traffic_invariant("instant_step")

        # Instantaneous post-step TTC/GAP diagnostics only.
        edm_post_step = self._edm_interpretation_by_lane(self._min_ttc_by_lane())

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
            "traffic_update_frequency_hz": int(TRAFFIC_UPDATE_FREQUENCY_HZ),
            "traffic_update_step_duration_s": float(TRAFFIC_UPDATE_DT_S),
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
            "instant_lane_change_mode": bool(INSTANT_LANE_CHANGE_MODE),
            "lateral_transition_mode": "instantaneous",
            "longitudinal_motion_mode": "normal_acceleration_dynamics",
            "observed_acceleration": (speed_after - speed_before) / policy_dt,
            "edm_pre_action_by_lane": edm_pre_action,
            "edm_pre_env_step_by_lane": edm_pre_env_step,
            "edm_post_step_by_lane": edm_post_step,
            "edm_role": "instantaneous_ttc_and_bumper_gap_features_for_critic_only",
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
# Instantaneous EDM features are now supplied separately to the Critic; the
# Actor state contains only vehicle kinematics and non-EDM context.
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
    role_speed_max = (
        TRAFFIC_REAR_MAX_SPEED_MPS
        if getattr(vehicle, "_arood_traffic_role", "front") == "rear"
        else TRAFFIC_MAX_SPEED_MPS
    )
    desired_speed = max(TRAFFIC_MIN_SPEED_MPS, min(desired_speed, role_speed_max))

    front = None
    front_gap = float("inf")
    for other in _traffic_in_lane(self, lane_id, exclude=vehicle):
        gap_center = _road_s(self, other) - ego_s
        if gap_center <= 0.0:
            continue
        bumper_gap = _longitudinal_bumper_gap(gap_center)
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


def _spawn_traffic_vehicle(self, lane_id, longitudinal, speed, traffic_role="front"):
    road = self.env.unwrapped.road
    lane_index = self._lane_index_for_id(int(lane_id))
    if traffic_role not in ("front", "rear"):
        raise ValueError(f"Unknown traffic role: {traffic_role}")
    role_speed_max = (
        TRAFFIC_REAR_MAX_SPEED_MPS if traffic_role == "rear"
        else TRAFFIC_MAX_SPEED_MPS
    )
    speed = float(np.clip(speed, TRAFFIC_MIN_SPEED_MPS, role_speed_max))
    vehicle = ContinuousTrafficVehicle.make_on_lane(
        road, lane_index, longitudinal=float(longitudinal), speed=speed
    )
    vehicle.LENGTH = VEHICLE_LENGTH_M
    vehicle._arood_traffic_manager = self
    profile_names = [item[0] for item in ADT_PROFILE_MIX]
    profile_weights = [item[1] for item in ADT_PROFILE_MIX]
    vehicle._arood_adt_profile = random.choices(
        profile_names, weights=profile_weights, k=1
    )[0]
    profile = vehicle._arood_adt_profile
    headway = PROFILE_IDM_HEADWAY_S[profile]
    min_gap = PROFILE_IDM_MIN_GAP_M[profile]
    desired_ranges = (
        PROFILE_REAR_DESIRED_SPEED_RANGES_MPS
        if traffic_role == "rear" else PROFILE_DESIRED_SPEED_RANGES_MPS
    )
    desired_low, desired_high = desired_ranges[profile]
    vehicle._arood_idm_headway_s = headway
    vehicle._arood_idm_min_gap_m = min_gap
    sampled_desired_speed = random.uniform(desired_low, desired_high)
    sampled_desired_speed += random.uniform(
        -ADT_SPEED_VARIATION_MPS, ADT_SPEED_VARIATION_MPS
    )
    vehicle._arood_traffic_role = traffic_role
    vehicle._arood_desired_speed = float(np.clip(
        sampled_desired_speed, TRAFFIC_MIN_SPEED_MPS, role_speed_max
    ))
    # Avoid spawning a vehicle with a large artificial speed mismatch relative
    # to its own desired speed. The simulator integration guard also enforces
    # TRAFFIC_MIN_SPEED_MPS as a hard floor while the vehicle is not crashed.
    initial_speed_low = max(
        TRAFFIC_MIN_SPEED_MPS, vehicle._arood_desired_speed - 4.0
    )
    initial_speed_high = min(
        role_speed_max, vehicle._arood_desired_speed + 4.0
    )
    vehicle.speed = float(np.clip(speed, initial_speed_low, initial_speed_high))
    vehicle._arood_lane_change_cooldown = 0
    vehicle._arood_brake_steps_remaining = 0
    vehicle._arood_traffic_uid = f"traffic-{time.time_ns()}-{random.randrange(1_000_000)}"
    vehicle.MAX_SPEED = role_speed_max
    road.vehicles.append(vehicle)
    return vehicle


def _safe_spawn_position(self, lane_id, longitudinal, minimum_gap, enforce_cross_lane_stagger=True):
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    # Keep an insertion buffer around ego in every lane, not only the
    # currently occupied lane, so an adjacent-lane vehicle cannot begin
    # immediately alongside ego and create an artificial lane-change hazard.
    ego_bumper_gap = _longitudinal_bumper_gap(longitudinal - ego_s)
    if ego_bumper_gap < minimum_gap:
        return False
    for other in list(self.env.unwrapped.road.vehicles):
        if other is ego:
            continue
        try:
            other_s = _road_s(self, other)
            other_lane_id = int(other.lane_index[2])
        except (AttributeError, TypeError, ValueError, IndexError):
            continue
        if other_lane_id == int(lane_id):
            gap = _longitudinal_bumper_gap(other_s - longitudinal)
            if gap < minimum_gap:
                return False
        elif (
            enforce_cross_lane_stagger
            and abs(other_s - float(longitudinal)) < TRAFFIC_CROSS_LANE_STAGGER_MIN_M
        ):
            # Initial traffic is explicitly staggered. Boundary replacement
            # prioritizes restoring lane coverage and uses a broad random
            # spawn interval instead of imposing a rigid cross-lane barrier.
            return False
    return True


def _initialize_continuous_traffic(self):
    """Create staggered front queues in all lanes without a population cap.

    Four initial vehicles per lane are placed ahead of the ego at staggered
    longitudinal offsets. Subsequent maintenance guarantees at least one
    front vehicle in every lane. Rear-side traffic continues to use the normal
    rear-boundary recycling path rather than being forced into the front queue.
    """
    env = self.env.unwrapped
    ego = env.vehicle
    env.road.vehicles = [ego]
    ego_s = _road_s(self, ego)

    lane_base_offsets = {0: 100.0, 1: 150.0, 2: 200.0}
    spawned = 0
    for lane_id in range(NUM_LANES):
        for queue_index in range(TRAFFIC_INITIAL_FRONT_VEHICLES_PER_LANE):
            offset = lane_base_offsets[lane_id] + 280.0 * queue_index
            longitudinal = ego_s + offset + random.uniform(-12.0, 12.0)
            if not _safe_spawn_position(
                self, lane_id, longitudinal, MIN_INITIAL_GAP_M
            ):
                longitudinal = ego_s + offset
            if not _safe_spawn_position(
                self, lane_id, longitudinal, MIN_INITIAL_GAP_M
            ):
                raise RuntimeError(
                    "Could not initialize staggered overtaking traffic: "
                    f"lane={lane_id}, offset={offset:.1f}m"
                )
            speed = random.uniform(TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS)
            _spawn_traffic_vehicle(self, lane_id, longitudinal, speed, traffic_role="front")
            spawned += 1

    # Rear traffic is deliberately ordinary ADT traffic: one vehicle per lane
    # begins behind the ego and retains the wider 25-50 m/s profile range.
    rear_offsets = {0: 120.0, 1: 240.0, 2: 360.0}
    for lane_id in range(NUM_LANES):
        for rear_index in range(TRAFFIC_INITIAL_REAR_VEHICLES_PER_LANE):
            offset = rear_offsets[lane_id] + 180.0 * rear_index
            longitudinal = ego_s - offset
            if not _safe_spawn_position(
                self, lane_id, longitudinal, MIN_INITIAL_GAP_M
            ):
                raise RuntimeError(
                    "Could not initialize ordinary rear traffic: "
                    f"lane={lane_id}, offset={offset:.1f}m"
                )
            speed = random.uniform(TRAFFIC_MIN_SPEED_MPS, TRAFFIC_REAR_MAX_SPEED_MPS)
            _spawn_traffic_vehicle(
                self, lane_id, longitudinal, speed, traffic_role="rear"
            )
            spawned += 1

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
    front_coverage = {
        lane_id: sum(
            _road_s(self, v) > ego_s and int(v.lane_index[2]) == lane_id
            for v in env.road.vehicles if v is not ego
        )
        for lane_id in range(NUM_LANES)
    }
    if any(
        count < TRAFFIC_MIN_FRONT_VEHICLES_PER_LANE
        for count in front_coverage.values()
    ):
        raise RuntimeError(f"Initial front-vehicle coverage failed: {front_coverage}")
    print(
        "[OVERTAKING TRAFFIC INITIALIZED]",
        f"vehicles={spawned}",
        f"front_vehicles_per_lane={front_coverage}",
        f"front_speed_range={TRAFFIC_MIN_SPEED_MPS:.0f}-{TRAFFIC_MAX_SPEED_MPS:.0f}m/s",
        f"rear_speed_range={TRAFFIC_MIN_SPEED_MPS:.0f}-{TRAFFIC_REAR_MAX_SPEED_MPS:.0f}m/s",
        f"ego_target_speed={SPEED_LIMIT_MPS:.0f}m/s",
        f"profiles={self._adt_profile_counts}",
        f"reference=+{TRAFFIC_REFERENCE_FRONT_M:.0f}/-{TRAFFIC_REFERENCE_REAR_M:.0f}m",
        "population_cap=none; layout=staggered front queues",
    )


def _adt_vehicle_acceleration(self, vehicle):
    """Profile-conditioned IDM with a restart rule and rare emergency braking."""
    profile = getattr(vehicle, "_arood_adt_profile", "normal")
    speed = max(0.0, float(vehicle.speed))
    desired_speed = float(getattr(vehicle, "_arood_desired_speed", SPEED_LIMIT_MPS))
    role_speed_max = (
        TRAFFIC_REAR_MAX_SPEED_MPS
        if getattr(vehicle, "_arood_traffic_role", "front") == "rear"
        else TRAFFIC_MAX_SPEED_MPS
    )
    desired_speed = float(np.clip(
        desired_speed, TRAFFIC_MIN_SPEED_MPS, role_speed_max
    ))
    lane_id = int(vehicle.lane_index[2])
    vehicle_s = _road_s(self, vehicle)
    front = None
    front_gap = float("inf")
    for other in _traffic_in_lane(self, lane_id, exclude=vehicle):
        delta = _road_s(self, other) - vehicle_s
        if delta <= 0.0:
            continue
        gap = _longitudinal_bumper_gap(delta)
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
        gap = _longitudinal_bumper_gap(delta)
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
        parallel_blocked = False
        for other in _traffic_in_lane(self, target_lane, exclude=vehicle):
            delta = _road_s(self, other) - current_s
            # Keep background traffic longitudinally staggered when changing
            # lanes; do not create an artificial side-by-side blocking pair.
            if abs(delta) < TRAFFIC_CROSS_LANE_STAGGER_MIN_M:
                parallel_blocked = True
                continue
            bumper_gap = _longitudinal_bumper_gap(delta)
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
        if parallel_blocked:
            continue
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
    """Spawn an uncapped replacement at the matching distant road boundary.

    Front exits are replaced at the distant front boundary, preferring the
    lane with the fewest vehicles ahead. Rear exits retain ordinary rear-side
    replacement and are not diverted into the overtaking queue.
    """
    if side not in ("front", "rear"):
        raise ValueError(f"Unknown traffic exit side: {side}")
    ego = self.env.unwrapped.vehicle
    road = self.env.unwrapped.road
    ego_s = _road_s(self, ego)

    side_counts = {lane: 0 for lane in range(NUM_LANES)}
    for vehicle in road.vehicles:
        if vehicle is ego:
            continue
        try:
            lane = int(vehicle.lane_index[2])
            delta = _road_s(self, vehicle) - ego_s
            is_relevant_side_vehicle = (
                delta > 0.0
                and float(vehicle.speed) < SPEED_LIMIT_MPS
                and float(getattr(vehicle, "_arood_desired_speed", TRAFFIC_REAR_MAX_SPEED_MPS))
                    < SPEED_LIMIT_MPS
                if side == "front" else delta < 0.0
            )
            if lane in side_counts and is_relevant_side_vehicle:
                side_counts[lane] += 1
        except (AttributeError, TypeError, ValueError, IndexError):
            continue

    lane_order = sorted(range(NUM_LANES), key=lambda lane: (side_counts[lane], lane))
    side_speed_max = (
        TRAFFIC_MAX_SPEED_MPS if side == "front" else TRAFFIC_REAR_MAX_SPEED_MPS
    )
    speed = random.uniform(TRAFFIC_MIN_SPEED_MPS, side_speed_max)
    if side == "front":
        spawn_min, spawn_max = TRAFFIC_FRONT_SPAWN_MIN_M, TRAFFIC_FRONT_SPAWN_MAX_M
    else:
        spawn_min, spawn_max = TRAFFIC_REAR_SPAWN_MIN_M, TRAFFIC_REAR_SPAWN_MAX_M

    for attempt in range(BOUNDARY_SPAWN_MAX_ATTEMPTS):
        lane_id = lane_order[attempt % len(lane_order)]
        direction = 1.0 if side == "front" else -1.0
        longitudinal = ego_s + direction * random.uniform(spawn_min, spawn_max)
        if not _safe_spawn_position(
            self, lane_id, longitudinal, MIN_INSERTION_GAP_M,
            enforce_cross_lane_stagger=False,
        ):
            continue
        _spawn_traffic_vehicle(
            self, lane_id, longitudinal, speed, traffic_role=side
        )
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

    # Restore the experiment's defining invariant: every lane retains at
    # least one slower front vehicle. New vehicles are created only at the
    # distant front boundary and enter the visible region through normal motion.
    front_counts = {lane: 0 for lane in range(NUM_LANES)}
    for vehicle in road.vehicles:
        if vehicle is ego:
            continue
        try:
            lane_id = int(vehicle.lane_index[2])
            is_slower_front = (
                _road_s(self, vehicle) > ego_s
                and float(vehicle.speed) < SPEED_LIMIT_MPS
                and float(getattr(vehicle, "_arood_desired_speed", TRAFFIC_REAR_MAX_SPEED_MPS))
                    < SPEED_LIMIT_MPS
            )
            if lane_id in front_counts and is_slower_front:
                front_counts[lane_id] += 1
        except (AttributeError, TypeError, ValueError, IndexError):
            continue

    for lane_id in range(NUM_LANES):
        while front_counts[lane_id] < TRAFFIC_MIN_FRONT_VEHICLES_PER_LANE:
            if not _spawn_boundary_replacement(self, "front"):
                break
            front_counts = {lane: 0 for lane in range(NUM_LANES)}
            for vehicle in road.vehicles:
                if vehicle is ego:
                    continue
                try:
                    lane = int(vehicle.lane_index[2])
                    is_slower_front = (
                        _road_s(self, vehicle) > ego_s
                        and float(vehicle.speed) < SPEED_LIMIT_MPS
                        and float(getattr(vehicle, "_arood_desired_speed", TRAFFIC_REAR_MAX_SPEED_MPS))
                            < SPEED_LIMIT_MPS
                    )
                    if lane in front_counts and is_slower_front:
                        front_counts[lane] += 1
                except (AttributeError, TypeError, ValueError, IndexError):
                    continue

    # Rear-side exits are replenished normally, independently of front coverage.
    while self._traffic_pending_replacements_by_side["rear"] > 0:
        if not _spawn_boundary_replacement(self, "rear"):
            break
    # Passed front vehicles also feed the rolling front queue, without a cap.
    while self._traffic_pending_replacements_by_side["front"] > 0:
        if not _spawn_boundary_replacement(self, "front"):
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
    if any(not np.isfinite(float(v.speed)) or float(v.speed) < 0.0 for v in vehicles):
        raise RuntimeError(f"Invalid background vehicle speed at {where}.")

    offenders = [
        {
            "uid": getattr(v, "_arood_traffic_uid", str(id(v))),
            "speed_mps": float(v.speed),
            "lane": int(v.lane_index[2]),
        }
        for v in vehicles
        if not bool(getattr(v, "crashed", False))
        and (
            float(v.speed) < TRAFFIC_MIN_SPEED_MPS - 1e-6
            or float(v.speed) > (
                TRAFFIC_REAR_MAX_SPEED_MPS
                if getattr(v, "_arood_traffic_role", "front") == "rear"
                else TRAFFIC_MAX_SPEED_MPS
            ) + 1e-6
        )
    ]
    if offenders:
        raise RuntimeError(
            f"Background traffic violated configured speed range at {where}: {offenders}"
        )
    if any(int(v.lane_index[2]) not in range(NUM_LANES) for v in vehicles):
        raise RuntimeError(f"Background vehicle has an invalid lane at {where}.")

    ego_s = _road_s(self, ego)
    front_counts = {lane: 0 for lane in range(NUM_LANES)}
    for vehicle in vehicles:
        relative_s = _road_s(self, vehicle) - ego_s
        if relative_s > TRAFFIC_FRONT_RECYCLE_BOUNDARY_M + 1e-6:
            raise RuntimeError(f"Vehicle beyond front recycling boundary at {where}.")
        if relative_s < -TRAFFIC_REAR_RECYCLE_BOUNDARY_M - 1e-6:
            raise RuntimeError(f"Vehicle beyond rear recycling boundary at {where}.")
        if (
            relative_s > 0.0
            and float(vehicle.speed) < SPEED_LIMIT_MPS
            and float(getattr(vehicle, "_arood_desired_speed", TRAFFIC_REAR_MAX_SPEED_MPS))
                < SPEED_LIMIT_MPS
        ):
            front_counts[int(vehicle.lane_index[2])] += 1

    missing_lanes = [
        lane for lane, count in front_counts.items()
        if count < TRAFFIC_MIN_FRONT_VEHICLES_PER_LANE
    ]
    if missing_lanes:
        raise RuntimeError(
            f"Front-vehicle coverage invariant failed at {where}: "
            f"front_counts={front_counts}, missing_lanes={missing_lanes}"
        )
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




# Bind the preserved continuous ADT traffic implementation to HighwaySimulator.
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



# ============================================================
# 3. Ego state and fixed-lane-order EDM feature encoders
# ============================================================

def _encode_ego_state(self, obs=None):
    ego = self.env.unwrapped.vehicle
    lane_id = int(ego.lane_index[2])
    try:
        raw_acceleration = float(getattr(ego, "action", {}).get("acceleration", 0.0))
    except (AttributeError, TypeError, ValueError):
        raw_acceleration = 0.0
    if not np.isfinite(raw_acceleration):
        raw_acceleration = 0.0
    state = np.asarray([
        np.clip(float(ego.speed) / SPEED_LIMIT_MPS, 0.0, 1.5),
        np.clip(raw_acceleration / MAX_PHYSICAL_ACCELERATION_MPS2, -1.0, 1.0),
        float(lane_id == 0),
        float(lane_id == 1),
        float(lane_id == 2),
        min(float(self.step_count) / max(1, self.max_steps), 1.0),
    ], dtype=np.float32)
    if state.shape != (EGO_STATE_DIM,) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"Invalid ego state representation: {state}")
    return state


def _get_edm_features_fixed_absolute_lane_order(self):
    """Return 12 features in fixed order: lane 0, lane 1, lane 2."""
    lane_metrics = self._min_ttc_by_lane()
    features = []
    for lane_id in range(NUM_LANES):
        lane = lane_metrics.get(lane_id, {})
        for side in ("front", "rear"):
            gap_key = f"{side}_gap"
            ttc_key = f"{side}_ttc_buffer"
            vehicle_key = f"{side}_vehicle_id"
            # A missing vehicle is distinct from a present, non-closing vehicle.
            if lane.get(vehicle_key) is None:
                features.extend([EDM_MISSING_VALUE, EDM_MISSING_VALUE])
                continue
            raw_ttc = float(lane.get(ttc_key, float("inf")))
            raw_gap = float(lane.get(gap_key, float("inf")))
            if not np.isfinite(raw_ttc):
                normalized_ttc = 1.0
            else:
                normalized_ttc = float(np.clip(raw_ttc / EDM_TTC_INPUT_MAX_S, 0.0, 1.0))
            if not np.isfinite(raw_gap):
                normalized_gap = 1.0
            else:
                normalized_gap = float(np.clip(raw_gap / EDM_GAP_INPUT_MAX_M, -1.0, 1.0))
            features.extend([normalized_ttc, normalized_gap])
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (EDM_FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"Invalid EDM feature vector: {result}")
    return result


# Replace the legacy six-slot policy observation and relative-lane EDM ordering.
HighwaySimulator._encode_state = _encode_ego_state
HighwaySimulator.get_critic_edm_features = _get_edm_features_fixed_absolute_lane_order
HighwaySimulator.get_edm_features = _get_edm_features_fixed_absolute_lane_order

# Preserve the original ADT traffic manager and its boundary-only replenishment.
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

def create_highway_simulator(render_mode=None):
    return HighwaySimulator(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_steps=MAX_EPISODE_STEPS,
        render_mode=render_mode,
    )


# ============================================================
# 4. Reward and one-step Actor-Critic update
# ============================================================

def calculate_environment_reward(collision, speed_after_mps, speed_change_mps):
    max_delta_v = max(MAX_ACCELERATION_PER_POLICY_STEP_MPS, 1e-8)
    clipped_delta_v = float(np.clip(speed_change_mps, -max_delta_v, max_delta_v))
    if float(speed_after_mps) >= SPEED_LIMIT_MPS:
        acceleration_reward = ACCELERATION_REWARD_AT_OR_ABOVE_SPEED_LIMIT
    else:
        acceleration_reward = float(
            (abs(clipped_delta_v) + clipped_delta_v) / (2.0 * max_delta_v)
        )
    collision_penalty = (
        COLLISION_PENALTY_MULTIPLIER * acceleration_reward if collision else 0.0
    )
    reward = float(acceleration_reward - collision_penalty)
    return reward, {
        "acceleration_reward": float(acceleration_reward),
        "collision_penalty": float(collision_penalty),
        "speed_reward": 0.0,
        "speed_change_mps": float(speed_change_mps),
        "clipped_speed_change_mps": clipped_delta_v,
        "total_reward": reward,
    }


def update_actor_critic(policy, state, edm, action_aux, current_lane_id,
                        reward, next_state, next_edm, done):
    log_prob, value = policy.evaluate_sample(
        state, edm, current_lane_id,
        action_aux["lane_action_index"],
        action_aux["pre_tanh_acceleration"],
    )
    with torch.no_grad():
        next_value = policy.value(next_state, next_edm)
        target = torch.as_tensor([reward], dtype=torch.float32, device=DEVICE)
        if not done:
            target = target + GAMMA * next_value
    advantage = target - value
    actor_loss = -(log_prob * advantage.detach()).mean()
    critic_loss = F.mse_loss(value, target)
    total_loss = actor_loss + VALUE_LOSS_WEIGHT * critic_loss

    policy.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
    policy.optimizer.step()
    return {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "advantage": float(advantage.detach().mean().cpu().item()),
    }


# ============================================================
# 5. Metrics and training
# ============================================================

def _append_metrics_csv(row):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exists = os.path.exists(METRICS_CSV_PATH)
    with open(METRICS_CSV_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(simulator, policy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    config_record = {
        key: value for key, value in globals().copy().items()
        if key.isupper() and isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
    }
    config_record.update({
        "architecture": "EDM_conditioned_shared_encoder_actor_critic",
        "actor_input": "ego_state_plus_fixed_absolute_lane_order_12D_EDM",
        "critic_input": "ego_state_plus_fixed_absolute_lane_order_12D_EDM",
        "actor_receives_edm": True,
        "critic_receives_edm": True,
        "edm_reward_shaping": False,
        "edm_action_intervention": False,
        "edm_prediction_horizon": False,
        "lane_action_mask": "physical_road_boundaries_only",
        "collision_detection": "native_HighwayEnv_vehicle_body_geometry",
        "traffic_environment_source": "Q_criticEDM.py diagnostics branch, preserved ADT simulator",
    })
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(_trajectory_jsonable(config_record), handle, indent=2, ensure_ascii=False)

    episode_rows = []
    reward_history = []
    for episode in range(NUM_ONLINE_EPISODES):
        simulator._trace_phase = "edm_actor_critic"
        simulator._trace_episode = episode + 1
        state = simulator.reset(seed=SEED + episode)
        edm = simulator.get_edm_features()
        record_episode_start(simulator, "edm_actor_critic", episode + 1, state)

        episode_return = 0.0
        speed_sum = 0.0
        safe_travel_distance_m = 0.0
        safe_steps = 0
        lane_changes = 0
        collision_count = 0
        actor_losses = []
        critic_losses = []
        start_time = time.time()
        steps = 0

        for step_index in range(MAX_EPISODE_STEPS):
            ego = simulator.env.unwrapped.vehicle
            current_lane = int(ego.lane_index[2])
            speed_before = float(ego.speed)
            road_s_before = float(simulator._road_longitudinal_position(ego))
            vehicles_before = snapshot_simulator_vehicles(simulator)

            sampled_action, action_aux = policy.act(
                state, edm, current_lane, deterministic=False
            )
            acceleration = float(sampled_action[0])
            lane_delta = int(sampled_action[1])
            target_lane = relative_lane_delta_to_absolute_lane(current_lane, lane_delta)

            # The simulator adapter retains its original [acceleration, KEEP/CHANGE]
            # interface. The Actor itself selects the actual relative lane direction.
            simulator_action = np.asarray(
                [acceleration, 1.0 if lane_delta != 0 else 0.0], dtype=np.float32
            )
            next_state, _, done, info = simulator.step(
                simulator_action,
                sampled_action=sampled_action,
                safety_info={
                    "requested_target_lane_id": int(target_lane),
                    "executed_target_lane_id": int(target_lane),
                    "lane_change_blocked": False,
                    "lane_request_trace_id": None,
                },
            )

            next_ego = simulator.env.unwrapped.vehicle
            speed_after = float(next_ego.speed)
            collision = bool(info.get("collision", False))
            speed_change = speed_after - speed_before
            reward, reward_components = calculate_environment_reward(
                collision, speed_after, speed_change
            )
            next_edm = simulator.get_edm_features()

            road_s_after = float(simulator._road_longitudinal_position(next_ego))
            transition_distance = max(0.0, road_s_after - road_s_before)
            if not collision:
                safe_travel_distance_m += transition_distance
                safe_steps += 1
            if bool(info.get("lane_change_completed_instantly", False)):
                lane_changes += 1
            if collision:
                collision_count += 1

            losses = update_actor_critic(
                policy, state, edm, action_aux, current_lane,
                reward, next_state, next_edm, bool(done),
            )
            actor_losses.append(losses["actor_loss"])
            critic_losses.append(losses["critic_loss"])

            episode_return += reward
            speed_sum += speed_after
            steps = step_index + 1

            if TRAJECTORY_RECORDING_ENABLED:
                append_trajectory_record({
                    "record_type": "transition",
                    "schema_version": 2,
                    "run_id": TRAJECTORY_RUN_ID,
                    "phase": "edm_actor_critic",
                    "episode_number": episode + 1,
                    "step": steps,
                    "ego_state_before": np.asarray(state, dtype=np.float32).tolist(),
                    "ego_state_after": np.asarray(next_state, dtype=np.float32).tolist(),
                    "edm_features_before": np.asarray(edm, dtype=np.float32).tolist(),
                    "edm_features_after": np.asarray(next_edm, dtype=np.float32).tolist(),
                    "vehicles_before": vehicles_before,
                    "vehicles_after": snapshot_simulator_vehicles(simulator),
                    "sampled_action": sampled_action.tolist(),
                    "lane_action_index": action_aux["lane_action_index"],
                    "lane_action_delta": lane_delta,
                    "target_lane": target_lane,
                    "reward": reward,
                    "reward_components": reward_components,
                    "collision": collision,
                    "safe_distance_increment_m": transition_distance if not collision else 0.0,
                    "done": bool(done),
                })

            state, edm = next_state, next_edm
            if done:
                break

        elapsed = time.time() - start_time
        row = {
            "episode": episode + 1,
            "episode_return": float(episode_return),
            "steps": int(steps),
            "collisions": int(collision_count),
            "collision_incidence": int(collision_count > 0),
            "safe_travel_distance_m": float(safe_travel_distance_m),
            "safe_steps": int(safe_steps),
            "mean_speed_mps": float(speed_sum / max(steps, 1)),
            "lane_changes": int(lane_changes),
            "lane_changes_per_100_steps": float(100.0 * lane_changes / max(steps, 1)),
            "mean_actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "elapsed_seconds": float(elapsed),
        }
        episode_rows.append(row)
        reward_history.append(episode_return)
        _append_metrics_csv(row)
        with open(METRICS_JSONL_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        print(
            f"[EP {episode + 1:03d}/{NUM_ONLINE_EPISODES}] "
            f"return={episode_return:.2f} | collisions={collision_count} | "
            f"safe_distance={safe_travel_distance_m:.1f}m | "
            f"mean_speed={row['mean_speed_mps']:.2f}m/s | "
            f"lane_changes={lane_changes} | steps={steps} | {elapsed:.1f}s"
        )

        if (episode + 1) % CHECKPOINT_INTERVAL_EPISODES == 0:
            torch.save(policy.state_dict(), MODEL_PATH)

    torch.save(policy.state_dict(), MODEL_PATH)
    if reward_history:
        plt.figure()
        plt.plot(np.arange(1, len(reward_history) + 1), reward_history, alpha=0.45)
        if len(reward_history) >= 5:
            window = min(10, len(reward_history))
            moving = np.convolve(
                reward_history, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                np.arange(window, len(reward_history) + 1), moving,
                linewidth=2.0, label=f"{window}-episode moving average",
            )
            plt.legend()
        plt.xlabel("Episode")
        plt.ylabel("Episode return")
        plt.title("AROOD EDM-conditioned Actor-Critic")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "training_return.png"))
        plt.close()

    summary = {
        "episodes": len(episode_rows),
        "mean_collision_incidence": float(np.mean([r["collision_incidence"] for r in episode_rows])) if episode_rows else 0.0,
        "mean_safe_travel_distance_m": float(np.mean([r["safe_travel_distance_m"] for r in episode_rows])) if episode_rows else 0.0,
        "mean_speed_mps": float(np.mean([r["mean_speed_mps"] for r in episode_rows])) if episode_rows else 0.0,
        "mean_lane_changes": float(np.mean([r["lane_changes"] for r in episode_rows])) if episode_rows else 0.0,
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[TRAINING SUMMARY]", summary)
    print("[MODEL]", os.path.abspath(MODEL_PATH))
    return {"policy": policy, "episodes": episode_rows, "summary": summary}


def main():
    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"Architecture: Actor/Critic both receive {EGO_STATE_DIM} ego features + "
        f"{EDM_FEATURE_DIM} fixed-order EDM features."
    )
    print(
        f"Timing: simulation={SIMULATION_FREQUENCY_HZ}Hz, policy={POLICY_FREQUENCY_HZ}Hz, "
        f"episode={MAX_EPISODE_STEPS} decisions ({MAX_EPISODE_DURATION_S:.1f}s), gamma={GAMMA}."
    )
    print(
        f"ADT: uncapped traffic; initial front vehicles/lane={TRAFFIC_INITIAL_FRONT_VEHICLES_PER_LANE}; "
        f"minimum front vehicles/lane={TRAFFIC_MIN_FRONT_VEHICLES_PER_LANE}; "
        f"traffic speed={TRAFFIC_MIN_SPEED_MPS:.1f}-{TRAFFIC_MAX_SPEED_MPS:.1f}m/s "
        f"(front), rear max={TRAFFIC_REAR_MAX_SPEED_MPS:.1f}m/s."
    )
    print(
        f"EDM: 3 lanes x [front TTC-to-{TTC_CLEARANCE_MARGIN_M:.1f}m, front physical GAP, "
        "rear TTC-to-margin, rear physical GAP]. No risk scores, reward shaping, "
        "prediction horizon, action veto or override."
    )
    print("Lane-action masking: physical road boundaries only.")
    print("Collision: native simulator geometry; actual collision terminates the episode.")
    simulator = create_highway_simulator()
    policy = EDMActorCritic().to(DEVICE)
    print(
        f"Input dimensions: ego={EGO_STATE_DIM}, EDM={EDM_FEATURE_DIM}, "
        f"combined={EGO_STATE_DIM + EDM_FEATURE_DIM}."
    )
    train(simulator, policy)


if __name__ == "__main__":
    main()

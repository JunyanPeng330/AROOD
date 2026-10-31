# PURE RL EXPERIMENT 3: CONTINUE TRAINING AFTER COLLISIONS
# Based on D_Pure_RL.py. Collision events incur the unchanged fixed penalty,
# but collision alone does not terminate an episode. Episodes end at the normal
# time limit (or a genuine non-collision terminal condition).
# No TTC/gap-based action masking, action rejection, acceleration override,
# emergency braking, or episode-level reward adaptation is active in this file.
# The binary lane-action distribution remains separate from target-lane selection.
# Continuous acceleration is sampled by a learned head conditioned on the lane decision.
# The initial evaluator is a transparent placeholder, not a real LLM.

# Research baseline; instantaneous lane changes are not physically realistic lateral trajectories.
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import random
from collections import deque, Counter
import numpy as np
import torch
import json
import os
import uuid

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
policy_frequency = 5
simulation_frequency = 10

lane_keeping_weight = 0.20

# Below SPEED_LIMIT, reward positive measured longitudinal acceleration; at/above it, give full speed reward.
# MAX_PHYSICAL_ACCELERATION normalizes below-limit acceleration reward to [0, 1].
# Lane selection is learned; TTC/gap are retained only as observation/diagnostic data.
SPEED_LIMIT = 40.0
INITIAL_PROGRESS_WEIGHT = 4.0
INITIAL_COLLISION_PENALTY = 5.0
# No off-road reward term: the policy chooses only among the three valid lanes.
INITIAL_OFFROAD_PENALTY = 0.0

# TTC and gap are retained as observation/diagnostic measurements only.

# Maneuver completion uses geometric alignment over consecutive policy decisions.
LANE_CHANGE_COMPLETE_LATERAL_TOL = 0.45   # metres from target-lane centre
LANE_CHANGE_COMPLETE_HEADING_TOL = 0.12   # radians, wrapped heading error
LANE_CHANGE_COMPLETE_STABLE_STEPS = 2     # consecutive checks before completion
# Relaxed fallback is used only after a long maneuver timeout.
LANE_CHANGE_TIMEOUT_LATERAL_TOL = 0.90
LANE_CHANGE_TIMEOUT_HEADING_TOL = 0.20
LANE_CHANGE_RECOVERY_COOLDOWN_STEPS = 2

# Traffic scenario diversity for training.
# Randomized traffic gaps create varied lane-change opportunities.

REFERENCE_GAP_MIN = 20.0
REFERENCE_GAP_MAX = 200.0
RANDOMIZED_FRONT_GAP_MIN = 20.0
RANDOMIZED_FRONT_GAP_MAX = 200.0
RANDOMIZED_REAR_GAP_MIN = 20.0
RANDOMIZED_REAR_GAP_MAX = 200.0
TRAFFIC_SPEED_VARIATION = 10.0



'''
In _maintain_six_vehicle_traffic(), it calculates the absolute longitudinal distance between each vehicle and ego.
- Within 120 m: the vehicle can qualify for a front or rear slot.
- Beyond 120 m: the vehicle is excluded from slot selection and removed from the managed traffic list.
- Crossing the ego: if the longitudinal separation is within the 0.5 m crossing tolerance,
the vehicle is not assigned to either side.
- Missing slot: _spawn_slot_vehicle(lane_id, side) creates a replacement for that slot.
'''
# Simulator triggers a replacement of surrounding vehicle
Traffic_relevant_range = 200

# Ego-relative surrounding-vehicle speed distribution (m/s).
# Each new traffic vehicle is sampled relative to the current ego speed.
TRAFFIC_SPEED_MIN = 15.0
TRAFFIC_SPEED_MAX = 40.0
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

# Lane changes are not intrinsically rewarded or penalized.
LANE_CHANGE_COST = 0.0

# Instantaneous lane changes complete within the current policy transition.
# This cooldown constant is retained for legacy helper paths only; the active
# instantaneous step path clears maneuver state immediately after relocation.
LANE_CHANGE_COOLDOWN_STEPS = 0


MAX_PHYSICAL_ACCELERATION = 8.0  # m/s^2


NUM_LANES = 3
NUM_SURROUNDING_VEHICLES = 6
VEHICLE_FEATURES = 7
NUM_VEHICLE_ROWS = NUM_SURROUNDING_VEHICLES + 1
VEHICLE_STATE_DIM = NUM_VEHICLE_ROWS * VEHICLE_FEATURES
CONTEXT_DIM = 4
SAFETY_FEATURES_PER_LANE = 6
SAFETY_FEATURES_DIM = NUM_LANES * SAFETY_FEATURES_PER_LANE
STATE_DIM = VEHICLE_STATE_DIM + CONTEXT_DIM + SAFETY_FEATURES_DIM
ACTION_DIM = 2  # execution action [normalized learned acceleration, binary KEEP/CHANGE]
POLICY_ACTION_DIM = 2  # binary KEEP, CHANGE decision; acceleration is a conditional continuous head

HIDDEN_DIM = 128


NUM_REFERENCE_EPISODES = 300
REFERENCE_TRAINING_EPOCHS = 200
REFERENCE_BATCH_SIZE = 256
NUM_ONLINE_EPISODES = 100

MAX_STEPS = 200


TTC_PROXY_THRESHOLD = 0.2


POLICY_LR = 3e-4


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Structured diagnostics: one JSON object per line.

#DIAGNOSTIC_LOG_PATH = "training_diagnostics_binary_lane_ranked.jsonl"

# Created only when a managed traffic-slot vehicle first has negative speed.
# Keep this distinct from any tracked README/placeholder so exclusive creation works.
NEGATIVE_SPEED_FIRST_TRANSITION_PATH = os.path.join("figure", "negative_speed_first_transition.txt")

# Fixed-reward pure-RL baseline: no episode-level evaluator or reward adaptation.
# only when a lane change completes; rejected requests are not charged this cost.


reward_path = 'figure/Re_Pure_RL_' + '3.png'
reward_per_step_path = 'figure/Re_Pure_RL_per_step_' + '3.png'
plot_path = 'figure/plot_Pure_RL_' + '3.txt'
MODEL_PATH = 'policy/Pure_RL_' + '3.pt'
EPISODE_TRAJECTORY_PATH = 'figure/episode_trajectories_Pure_RL_' + '3.jsonl'
TRAJECTORY_RECORDING_ENABLED = os.environ.get("AROOD_RECORD_TRAJECTORIES_Pure_RL_", "3").strip().lower() not in {"0", "false", "no", "off"}


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
    """Append one structured diagnostic event; logging failures must not stop RL."""
    record = {
        "event": str(event),
        "phase": getattr(simulator, "_trace_phase", None) if simulator is not None else None,
        "episode": getattr(simulator, "_trace_episode", None) if simulator is not None else None,
        "simulator_step": getattr(simulator, "step_count", None) if simulator is not None else None,
    }
    record.update(fields)
    '''
    try:
        with open(DIAGNOSTIC_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False, default=str, allow_nan=True) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        print("[DIAGNOSTIC LOG WARNING]", f"event={event}", f"error={exc}")
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


class HighwaySimulator:
    """Highway-env wrapper with six managed traffic vehicles.

    State is 53-D: seven 7-feature vehicle rows (ego + six traffic vehicles)
    and four context values.
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 max_steps=MAX_STEPS, render_mode=None, vehicles_count=NUM_SURROUNDING_VEHICLES):
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
        self.vehicles_count = NUM_SURROUNDING_VEHICLES
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
                    "absolute": False,
                    "normalize": True,
                    "order": "sorted",
                },
                "action": {"type": "ContinuousAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.1745329, 0.1745329],
                },

                "lanes_count": 3,
                "vehicles_count": NUM_SURROUNDING_VEHICLES,
                "duration": max_steps,
                "offroad_terminal": False,
                # A collision is a penalized transition, not an episode boundary.
                "crash_ends_episode": False,

                # Instantaneous lane transitions are applied by the adapter before
                # this single environment update. Keep both frequencies equal so
                # one policy decision maps to exactly one simulator update.
                "simulation_frequency": policy_frequency,
                "policy_frequency": policy_frequency,

            },
        )



        # Keep this invariant explicit even if a highway-env version applies
        # defaults differently from the configuration dictionary.
        self.env.unwrapped.config["crash_ends_episode"] = False
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
        context = np.asarray([
            np.clip(float(vehicle.speed) / 40.0, 0.0, 1.5),
            np.clip(lane_id / 2.0, 0.0, 1.0),
            float(bool(getattr(vehicle, "crashed", False))),
            min(self.step_count / max(1, self.max_steps), 1.0),
        ], dtype=np.float32)
        state = np.concatenate([rows.reshape(-1), context])

        #print("at the end of _encode_state(), the state becomes: ", state)

        safety=np.zeros(SAFETY_FEATURES_DIM,dtype=np.float32)
        try:
            metrics=self._min_ttc_by_lane()
            keys=["front_gap","rear_gap","front_ttc","rear_ttc","front_speed","rear_speed"]
            scales=[100.0,100.0,10.0,10.0,40.0,40.0]
            for lane_id in range(NUM_LANES):
                vals=metrics.get(lane_id,{})
                for j,(key,scale) in enumerate(zip(keys,scales)):
                    value=vals.get(key,scale); value=float(value) if value is not None else scale
                    if not np.isfinite(value): value=scale*1.5
                    safety[lane_id*SAFETY_FEATURES_PER_LANE+j]=np.clip(value/scale,-1.0,1.5)
        except (AttributeError,KeyError,TypeError,ValueError,IndexError): pass
        return np.nan_to_num(np.concatenate([state,safety]),nan=0.0,posinf=1.0,neginf=-1.0).astype(np.float32)




    def reset(self, seed=None, randomize_traffic_gaps=False):

        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)

        # In randomized reference scenarios, each lane/side slot samples its
        # own gap during spawning (and when that slot is replenished). Keep the
        # configured distribution-shift gaps untouched when randomization is off.
        self.randomize_traffic_gaps = bool(randomize_traffic_gaps)
        if self.randomize_traffic_gaps:
            self.traffic_front_gap = random.uniform(
                RANDOMIZED_FRONT_GAP_MIN, RANDOMIZED_FRONT_GAP_MAX)
            self.traffic_rear_gap = random.uniform(
                RANDOMIZED_REAR_GAP_MIN, RANDOMIZED_REAR_GAP_MAX)
        print("[TRAFFIC SCENARIO]",
              f"front_gap_range={RANDOMIZED_FRONT_GAP_MIN:.1f}-{RANDOMIZED_FRONT_GAP_MAX:.1f}m",
              f"rear_gap_range={RANDOMIZED_REAR_GAP_MIN:.1f}-{RANDOMIZED_REAR_GAP_MAX:.1f}m",
              f"gap_sampling={'independent_per_lane_and_side' if self.randomize_traffic_gaps else 'configured_baseline_setting'}",
              f"speed_variation=±{TRAFFIC_SPEED_VARIATION:.1f}m/s",
              f"randomized={self.randomize_traffic_gaps}")

        self.step_count = 0
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





    def _min_ttc_by_lane(self):
        """Return one dynamic front/rear neighbor per lane and its TTC.

        The traffic abstraction is exactly six dynamic slots: nearest front and
        nearest rear vehicle in each of the three lanes. Slot identity is based
        on current lane and longitudinal position, not on a vehicle's original
        spawn label. Distances use a shared road-longitudinal coordinate and
        bumper-to-bumper clearance. Speeds are read directly from the simulator.
        """
        base_env = self.env.unwrapped
        road = base_env.road
        ego = base_env.vehicle

        ego_lane = int(ego.lane_index[2])
        ego_speed = float(ego.speed)
        ego_s = self._road_longitudinal_position(ego)
        lanes_count = int(base_env.config["lanes_count"])
        ego_length = float(getattr(ego, "LENGTH", getattr(ego, "length", 5.0)))

        lane_ttc = {
            lane_id: {
                "front_ttc": float("inf"),
                "rear_ttc": float("inf"),
                "min_ttc": float("inf"),
                "front_gap": float("inf"),
                "front_speed": None,
                "nearest_front_ttc": float("inf"),
                "front_center_gap": float("inf"),
                "front_vehicle_id": None,
                "rear_gap": float("inf"),
                "rear_speed": None,
                "rear_center_gap": float("inf"),
                "rear_vehicle_id": None,
            }
            for lane_id in range(lanes_count)
        }

        # Find the nearest vehicle on each side of ego in each lane. The
        # maintenance routine normally guarantees one vehicle per slot; nearest
        # selection also makes this function robust while diagnosing violations.
        nearest = {
            (lane_id, side): None
            for lane_id in range(lanes_count)
            for side in ("front", "rear")
        }

        for other in list(road.vehicles):
            if other is ego or bool(getattr(other, "crashed", False)):
                continue
            try:
                other_lane = int(other.lane_index[2])
                if other_lane not in lane_ttc:
                    continue
                other_s = self._road_longitudinal_position(other)
                other_speed = float(other.speed)
                other_length = float(getattr(
                    other, "LENGTH", getattr(other, "length", 5.0)
                ))
            except (AttributeError, TypeError, ValueError, IndexError, KeyError):
                continue

            center_gap = other_s - ego_s
            half_length_sum = 0.5 * (ego_length + other_length)
            if center_gap > 0.0:
                side = "front"
                bumper_gap = center_gap - half_length_sum
                distance = center_gap
            elif center_gap < 0.0:
                side = "rear"
                bumper_gap = -center_gap - half_length_sum
                distance = -center_gap
            else:
                # Exact longitudinal alignment is unsafe in both directions.
                lane_values = lane_ttc[other_lane]
                lane_values["front_gap"] = -half_length_sum
                lane_values["rear_gap"] = -half_length_sum
                lane_values["front_speed"] = other_speed
                lane_values["rear_speed"] = other_speed
                lane_values["front_center_gap"] = 0.0
                lane_values["rear_center_gap"] = 0.0
                lane_values["front_vehicle_id"] = id(other)
                lane_values["rear_vehicle_id"] = id(other)
                lane_values["front_ttc"] = 0.0
                lane_values["rear_ttc"] = 0.0
                lane_values["nearest_front_ttc"] = 0.0
                continue

            slot_key = (other_lane, side)
            current = nearest[slot_key]
            if current is None or distance < current["distance"]:
                nearest[slot_key] = {
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
            bumper_gap = float(neighbor["bumper_gap"])
            neighbor_speed = float(neighbor["speed"])

            if side == "front":
                # Front TTC: bumper gap / (ego speed - front speed), only
                # when ego is actually closing on the vehicle ahead.
                if bumper_gap <= 0.0:
                    front_ttc = 0.0
                elif ego_speed > neighbor_speed:
                    front_ttc = bumper_gap / (ego_speed - neighbor_speed)
                else:
                    front_ttc = float("inf")

                values["front_gap"] = bumper_gap
                values["front_speed"] = neighbor_speed
                values["front_ttc"] = front_ttc
                values["nearest_front_ttc"] = front_ttc
                values["front_center_gap"] = float(neighbor["distance"])
                values["front_vehicle_id"] = neighbor["vehicle_id"]
            else:
                # Rear TTC: bumper gap / (rear speed - ego speed), only
                # when the rear vehicle is actually closing on ego.
                if bumper_gap <= 0.0:
                    rear_ttc = 0.0
                elif neighbor_speed > ego_speed:
                    rear_ttc = bumper_gap / (neighbor_speed - ego_speed)
                else:
                    rear_ttc = float("inf")

                values["rear_gap"] = bumper_gap
                values["rear_speed"] = neighbor_speed
                values["rear_ttc"] = rear_ttc
                values["rear_center_gap"] = float(neighbor["distance"])
                values["rear_vehicle_id"] = neighbor["vehicle_id"]

        for values in lane_ttc.values():
            values["min_ttc"] = min(values["front_ttc"], values["rear_ttc"])

        # Diagnostic check: six slots should be occupied exactly once.
        occupied_slots = sum(
            nearest[(lane_id, side)] is not None
            or lane_ttc[lane_id]["front_center_gap"] == 0.0
            or lane_ttc[lane_id]["rear_center_gap"] == 0.0
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











#???????????????????????????????????????????????????????????????????????????????????????????????????????????
    def _update_lane_target(self, lane_action_id):
        """
        Legacy persistent-target helper for the non-instantaneous controller.

        The active instantaneous pure-RL policy selects a physical target lane
        through its learned target-lane head, so this legacy helper is unused.
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
        """Capture managed-slot vehicle state immediately around one native env.step()."""
        base_env = self.env.unwrapped
        ego = getattr(base_env, "vehicle", None)
        snapshot = {}
        for vehicle in list(getattr(getattr(base_env, "road", None), "vehicles", [])):
            side = getattr(vehicle, "_arood_traffic_side", None)
            if vehicle is ego or side not in ("front", "rear"):
                continue
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

    def step(self, action, sampled_action=None, safety_info=None):
        """Execute the policy-selected hybrid action with instantaneous lane changes."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Expected hybrid action [acceleration, lane_action], got {action.shape}")
        incoming_action = action.copy()

        action = incoming_action.copy()
        safety_info = dict(safety_info or {})
        safety_info["simulator_input_action"] = incoming_action.copy()
        safety_info["executed_high_level_action"] = action.copy()
        safety_info["safety_filter_applied_in_step"] = False
        safety_info["safety_filter_applied"] = False

        base_env = self.env.unwrapped
        ego = base_env.vehicle
        source_lane_id = int(ego.lane_index[2])
        acceleration = float(np.clip(action[0], -1.0, 1.0))
        lane_action_id = int(np.clip(np.rint(action[1]), 0, 1))
        lane_name = ("KEEP", "CHANGE")[lane_action_id]
        requested_target_lane_id = int(safety_info.get("executed_target_lane_id", safety_info.get("requested_target_lane_id", source_lane_id)))

        lane_changed = False
        lane_rejection_reason = "none"
        if lane_action_id == 1:
            adjacent_lanes = self._pure_rl_adjacent_lane_ids()
            if requested_target_lane_id not in adjacent_lanes:
                raise RuntimeError(
                    "Pure-RL action invariant violated: CHANGE must execute the policy-selected "
                    f"physically adjacent target lane; current={source_lane_id}, "
                    f"target={requested_target_lane_id}, adjacent={adjacent_lanes}. "
                    "No safety-based fallback or action substitution is permitted."
                )
            try:
                source_lane_index = tuple(ego.lane_index)
                source_lane = base_env.road.network.get_lane(source_lane_index)
                longitudinal, _ = source_lane.local_coordinates(np.asarray(ego.position, dtype=float))
                target_lane_index = (
                    source_lane_index[0], source_lane_index[1], int(requested_target_lane_id)
                )
                target_lane = base_env.road.network.get_lane(target_lane_index)
                # Instantaneous lane relocation is the selected simulator abstraction.
                target_position = np.asarray(target_lane.position(longitudinal, 0.0), dtype=float)
                target_heading = float(target_lane.heading_at(longitudinal))
                ego.position = target_position
                ego.heading = target_heading
                ego.lane_index = tuple(target_lane_index)
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
                raise RuntimeError(
                    "Pure-RL simulator could not execute the policy-selected physical lane change: "
                    f"source={source_lane_id}, target={requested_target_lane_id}"
                ) from exc

        # Do not replace the policy's acceleration with a hand-designed
        # controller-generated action. Only the policy's normalized action bound
        # and the simulator's native vehicle dynamics determine the motion.
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
        # Track crash flags around the native update. highway-env keeps
        # Vehicle.crashed=True after impact, so that flag alone is a state,
        # not a per-transition collision-event signal.
        road_vehicles_before = list(getattr(base_env.road, "vehicles", []) or [])
        crashed_before = {id(vehicle): bool(getattr(vehicle, "crashed", False))
                          for vehicle in road_vehicles_before}
        ego_crashed_before = bool(getattr(ego, "crashed", False))
        obs, env_reward, terminated, truncated, env_info = self.env.step(physical_action)
        # Explicitly neutralize highway-env's collision terminal signal at the
        # adapter boundary. Some highway-env versions still return terminated=True
        # for ego.crashed even when crash_ends_episode=False is in the config.
        # Preserve truncation/time-limit signals, and do not alter the collision
        # state or the fixed collision penalty applied by the RL reward function.
        native_terminated = bool(terminated)
        native_truncated = bool(truncated)
        native_termination_overridden = False
        if bool(getattr(base_env.vehicle, "crashed", False)) and bool(terminated):
            terminated = False
            native_termination_overridden = True
            print(
                "[COLLISION TERMINATION OVERRIDE]",
                "native_terminated=True",
                "effective_terminated=False",
                f"truncated={native_truncated}",
                "reason=continue_episode_after_ego_collision",
            )
        road_vehicles_after = list(getattr(base_env.road, "vehicles", []) or [])
        newly_crashed_vehicle_ids = [
            id(vehicle) for vehicle in road_vehicles_after
            if bool(getattr(vehicle, "crashed", False))
            and not crashed_before.get(id(vehicle), False)
        ]
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
        collision = bool(getattr(ego_after, "crashed", False))
        collision_event = bool(
            (collision and not ego_crashed_before)
            or any(vehicle_id != id(ego_after) for vehicle_id in newly_crashed_vehicle_ids)
        )

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

        info = dict(env_info or {})
        info.update({
            "safety_violation": collision_event,
            "collision": collision,
            "collision_event": collision_event,
            "newly_crashed_vehicle_count": int(len(newly_crashed_vehicle_ids)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "native_terminated": native_terminated,
            "native_truncated": native_truncated,
            "native_termination_overridden": native_termination_overridden,
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
            "policy_action_audit": safety_info,
            "episode_evaluator_assessment": {
                "target_selector_mode": safety_info.get("target_selector_mode"),
                "lane_change_candidate_front_ttc": safety_info.get("lane_change_candidate_front_ttc"),
                "lane_change_candidate_rear_ttc": safety_info.get("lane_change_candidate_rear_ttc"),
                "lane_change_candidate_utility": safety_info.get("lane_change_candidate_utility"),
            },
            "instant_lane_change_mode": True,
            "observed_acceleration": (speed_after - speed_before) / policy_dt,
        })
        self.state = next_state
        print("[INSTANT STEP]",
              f"source_lane={source_lane_id}",
              f"requested={lane_name}",
              f"target_lane={requested_target_lane_id}",
              f"lane_changed={lane_changed}",
              f"executed_lane={int(ego_after.lane_index[2])}",
              f"accel={acceleration:.3f}",
              f"collision_state={collision}",
            f"collision_event={collision_event}")
        effective_done = bool(terminated or truncated)
        print(
            "[EPISODE END FLAGS]",
            f"native_terminated={native_terminated}",
            f"effective_terminated={bool(terminated)}",
            f"native_truncated={native_truncated}",
            f"effective_done={effective_done}",
            f"collision={collision}",
        )
        return next_state.copy(), float(env_reward), effective_done, info

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
# Pure-RL physical lane masks and identity action pass-through
# ============================================================

def _pure_rl_adjacent_lane_ids(self):
    """Return physically existing adjacent lanes only; never inspect TTC or gaps."""
    base = self.env.unwrapped
    ego = base.vehicle
    start_node, end_node, current_lane = ego.lane_index
    lane_count = len(base.road.network.graph[start_node][end_node])
    current_lane = int(current_lane)
    return [lane for lane in (current_lane - 1, current_lane + 1)
            if 0 <= lane < min(lane_count, NUM_LANES)]


def _pure_rl_lane_action_mask(self):
    """Binary [KEEP, CHANGE] mask; CHANGE is available if a physical adjacent lane exists."""
    adjacent = self._pure_rl_adjacent_lane_ids()
    mask = np.asarray([True, bool(adjacent)], dtype=bool)
    print("[PURE RL ACTION MASK]", f"mask={mask.tolist()}",
          f"physical_adjacent_lanes={adjacent}", "safety_filter=disabled")
    return mask


def _pure_rl_target_lane_mask(self):
    """Allow the learned target-lane head to choose any physically adjacent lane."""
    mask = np.zeros(NUM_LANES, dtype=bool)
    adjacent = self._pure_rl_adjacent_lane_ids()
    for lane_id in adjacent:
        mask[int(lane_id)] = True
    print("[PURE RL TARGET MASK]", f"mask={mask.tolist()}",
          f"physical_adjacent_lanes={adjacent}", "TTC_gap_filter=disabled")
    return mask


def _pure_rl_action_passthrough(self, sampled_action, selected_target_lane=None):
    """Pass policy actions through unchanged; validate only physical lane topology."""
    action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
    if action.size != 2:
        raise ValueError("Expected [normalized acceleration, binary KEEP/CHANGE action]")
    requested = int(np.clip(np.rint(float(action[1])), 0, 1))
    current = int(self.env.unwrapped.vehicle.lane_index[2])
    adjacent = self._pure_rl_adjacent_lane_ids()
    if requested == 1:
        if selected_target_lane is None:
            raise RuntimeError("RL selected CHANGE without a target lane; check the policy target mask.")
        target = int(selected_target_lane)
        if target not in adjacent:
            raise RuntimeError(
                f"RL selected physically unavailable/non-adjacent lane {target}; "
                f"current={current}, available_adjacent={adjacent}. No safety fallback is applied."
            )
    else:
        target = current

    info = {
        "current_lane_id": current,
        "requested_target_lane_id": target,
        "selected_target_lane_id": int(selected_target_lane) if selected_target_lane is not None else current,
        "executed_target_lane_id": target,
        "policy_requested_lane_action": ("KEEP", "CHANGE")[requested],
        "executed_lane_action": ("KEEP", "CHANGE")[requested],
        "policy_requested_binary_action": ("KEEP", "CHANGE")[requested],
        "executed_binary_action": ("KEEP", "CHANGE")[requested],
        "lane_change_blocked": False,
        "lane_change_rejection_reason": "none",
        "lane_request_direction": ("LEFT" if target > current else "RIGHT" if target < current else "NONE"),
        "lane_request_source_lane": current,
        "lane_request_target_lane": target,
        "lane_request_decision": "TARGET_SELECTED" if requested == 1 else "KEEP",
        "target_selector_mode": "rl_policy_with_physical_lane_boundary_mask_only",
        "safety_filter_applied": False,
        "overridden": False,
        "emergency_braking": False,
        "acceleration_overridden": False,
        "intervention_reason": "none",
        "sampled_action": action.copy(),
        "filtered_action": action.copy(),
        "proposed_policy_acceleration": float(action[0]),
        "executed_acceleration": float(action[0]),
    }
    print("[PURE RL ACTION PASSTHROUGH]", f"requested={info['policy_requested_lane_action']}",
          f"target={target}", f"acceleration={float(action[0]):+.4f}",
          "action_intervention=none")
    return action.copy(), info


HighwaySimulator._pure_rl_adjacent_lane_ids = _pure_rl_adjacent_lane_ids
HighwaySimulator.get_valid_lane_action_mask = _pure_rl_lane_action_mask
HighwaySimulator.get_valid_target_lane_mask = _pure_rl_target_lane_mask
HighwaySimulator.prepare_policy_action = _pure_rl_action_passthrough




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
        "no_adjacent_lane_fraction": float(np.mean([m.get("no_adjacent_lane_fraction", 0.0) for m in metrics])),
        "valid_adjacent_lane_fraction": float(np.mean([m.get("valid_adjacent_lane_fraction", 0.0) for m in metrics])),
        "mean_return": float(np.mean([m.get("episode_return", 0.0) for m in metrics])),
        "mean_step_reward": float(np.mean([m.get("mean_reward_per_step", 0.0) for m in metrics])),
    }


# Episode-level reward evaluators are intentionally absent from this pure-RL baseline.


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

    # Canonical lane relevance for diagnostic TTC measurement only.
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
    lane_details = {
        lane_id: {
            "front_vehicle_id": lane_ttc.get(lane_id, {}).get("front_vehicle_id"),
            "front_ttc_s": lane_ttc.get(lane_id, {}).get("front_ttc", float("inf")),
            "front_gap_m": lane_ttc.get(lane_id, {}).get("front_gap", float("inf")),
            "rear_vehicle_id": lane_ttc.get(lane_id, {}).get("rear_vehicle_id"),
            "rear_ttc_s": lane_ttc.get(lane_id, {}).get("rear_ttc", float("inf")),
            "rear_gap_m": lane_ttc.get(lane_id, {}).get("rear_gap", float("inf")),
        }
        for lane_id in sorted(relevant_lane_ids)
    }
    print(
        "[TTC REWARD INPUT]",
        f"source=post_step",
        f"physical_lane={physical_lane_id}",
        f"relevant_lanes={sorted(relevant_lane_ids)}",
        f"lane_details={lane_details}",
        f"min_ttc_s={min_ttc:.3f}",
        f"finite_ttc_count={sum(np.isfinite(v) for v in values)}",
    )

    return min_ttc, sum(np.isfinite(v) for v in values)





def construct_reward(reward_specification, embedding_model=None):
    """Fixed pure-RL reward: normalized forward progress minus a fixed collision penalty.

    Lane-change cost is fixed at zero. TTC and gap are diagnostic observations
    only and do not enter reward construction or action execution.
    """
    def reward_function(
        behavior_description, safety_violation, action, lane_deviation, offroad,
        min_ttc, lane_change_allowed=True, ego_speed=0.0, ego_acceleration_mps2=0.0,
        safety_info=None, policy_sampled_action=None, reward_diagnostics=None,
    ):
        target_speed = max(
            1e-6, float(reward_specification.get("target_speed", SPEED_LIMIT))
        )
        ego_speed = max(0.0, float(ego_speed))
        # Normalized speed is a direct per-step proxy for sustained forward progress.
        progress_reward = float((ego_speed - target_speed)/target_speed)      #float(np.clip(ego_speed / target_speed, 0.0, 1.0))
        collision_penalty = (
            max(0.0, float(reward_specification.get("collision_penalty_weight", INITIAL_COLLISION_PENALTY)))
            if bool(safety_violation) else 0.0
        )
        # Charge only a completed lane change, never a rejected request or KEEP.
        lane_change_completed = bool(
            isinstance(safety_info, dict) and safety_info.get("lane_change_accepted", False)
        )
        lane_change_cost_weight = max(
            0.0, float(reward_specification.get("lane_change_cost", LANE_CHANGE_COST))
        )
        lane_change_cost = lane_change_cost_weight if lane_change_completed else 0.0
        reward = float(progress_reward - collision_penalty - lane_change_cost)
        if reward_diagnostics is not None:
            reward_diagnostics.update({
                "reward_objective": "normalized_forward_progress_minus_collision_and_lane_change_cost",
                "progress_reward": progress_reward,
                "speed_reward": progress_reward,
                "ego_speed_mps": float(ego_speed),
                "target_speed_mps": float(target_speed),
                "safety_penalty": float(collision_penalty),
                "collision_penalty": float(collision_penalty),
                "lane_change_completed": bool(lane_change_completed),
                "lane_change_cost_weight": float(lane_change_cost_weight),
                "lane_change_cost": float(lane_change_cost),
                "offroad_penalty": 0.0,
                "total_reward": reward,
            })
        print(
            f"[REWARD DEBUG] progress_reward={progress_reward:.3f}, "
            f"speed={ego_speed:.2f}m/s, target_speed={target_speed:.2f}m/s, "
            f"collision_penalty={collision_penalty:.3f}, "
            f"lane_change_cost={lane_change_cost:.3f}, total={reward:.3f}"
        )
        print("[PURE RL] TTC/gap are diagnostic only; collision penalty is fixed.")
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
    ego_acceleration_mps2=0.0,
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
        ego_acceleration_mps2=ego_acceleration_mps2,
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
                  current_lane_id=None):
    """Update lane choice, conditional target lane, and learned acceleration jointly."""
    st, nst = to_tensor(state).reshape(1, -1), to_tensor(next_state).reshape(1, -1)
    act, rew = to_tensor(action).reshape(1, -1), to_tensor([reward])
    value = policy.value(st)
    with torch.no_grad():
        nv = policy.value(nst)
        terminal = bool(done) if terminated is None else bool(terminated)
        target = rew if terminal else rew + 0.99 * nv
    adv = target - value
    critic_loss = F.mse_loss(value, target)
    binary_action = int(np.clip(np.rint(float(act[0, 1].detach().cpu().item())), 0, 1))
    current = int(current_lane_id) if current_lane_id is not None else 0
    effective_target = int(selected_target_lane) if binary_action == 1 and int(selected_target_lane) >= 0 else current
    effective_target = int(np.clip(effective_target, 0, NUM_LANES - 1))
    binary_dist = policy.action_distributions(
        st, torch.as_tensor(valid_action_mask, dtype=torch.bool, device=DEVICE).reshape(1, -1)
        if valid_action_mask is not None else None
    )
    binary_log_prob = binary_dist.log_prob(torch.as_tensor([binary_action], dtype=torch.long, device=DEVICE))
    target_log_prob = torch.zeros(1, dtype=torch.float32, device=DEVICE)
    if binary_action == 1 and int(selected_target_lane) >= 0 and valid_target_lane_mask is not None:
        lane_mask = torch.as_tensor(valid_target_lane_mask, dtype=torch.bool, device=DEVICE).reshape(1, -1).clone()
        if current_lane_id is not None and 0 <= int(current_lane_id) < lane_mask.shape[-1]:
            lane_mask[:, int(current_lane_id)] = False
        if lane_mask.any(dim=-1).all():
            lane_dist = policy.lane_distributions(st, lane_mask)
            target_log_prob = lane_dist.log_prob(torch.as_tensor([int(selected_target_lane)], dtype=torch.long, device=DEVICE))
    accel_value = (float(pre_tanh_accel) if pre_tanh_accel is not None else
                   float(np.arctanh(np.clip(float(act[0, 0].detach().cpu().item()), -0.999, 0.999))))
    pre_tanh = torch.as_tensor([accel_value], dtype=torch.float32, device=DEVICE)
    accel_dist = policy.acceleration_distribution(st, binary_action, effective_target)
    acceleration_log_prob = policy.squashed_normal_log_prob(accel_dist, pre_tanh)
    action_log_prob = binary_log_prob + target_log_prob + acceleration_log_prob
    actor_loss = -(action_log_prob * adv.detach()).mean()
    loss = actor_loss + critic_loss
    policy.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    policy.optimizer.step()
    return {"actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "advantage": float(adv.detach().mean().item()),
            "binary_log_prob": float(binary_log_prob.detach().mean().item()),
            "target_lane_log_prob": float(target_log_prob.detach().mean().item()),
            "acceleration_log_prob": float(acceleration_log_prob.detach().mean().item())}





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
# Phase I: collect reference rollouts, then learn from them
# ============================================================
def build_fixed_reward_function():
    reward_specification = {
        "name": "pure_rl_fixed_reward",
        "target_behavior": (
            "Reward sustained forward progress using normalized ego speed, with a strong penalty for collisions. "
            "The collision penalty is fixed throughout training. Lane-change cost is zero. "
            "TTC and gap are diagnostic observations only; the RL policy may choose any physically "
            "adjacent lane without an independent safety controller."
        ),
        "collision_penalty_weight": INITIAL_COLLISION_PENALTY,
        "target_speed": SPEED_LIMIT,
        "lane_change_cost": float(LANE_CHANGE_COST),
    }
    reward_function = construct_reward(reward_specification)
    # Immutable during this pure-RL experiment: no evaluator or adaptive reward updates.
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
            filtered_action, safety_info = simulator.prepare_policy_action(
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
                sampled_action=sampled_action,
                safety_info=safety_info,
            )
            reference_executed_action = np.asarray(
                step_info.get("filtered_policy_action", filtered_action), dtype=np.float32
            ).reshape(-1)
            if not np.allclose(reference_executed_action, sampled_action, atol=1e-6, rtol=0.0):
                raise RuntimeError(
                    "Pure-RL invariant violated during reference collection: "
                    f"sampled_action={sampled_action.tolist()}, "
                    f"executed_action={reference_executed_action.tolist()}"
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
                ego_acceleration_mps2=(
                    float(simulator.env.unwrapped.vehicle.speed) - previous_speed_mps
                ) * float(simulation_frequency),
                safety_info=reward_safety_info,
                policy_sampled_action=sampled_action,
                reward_diagnostics={},
            )

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
# Phase II: online Actor-Critic fine-tuning — no OOD
# ============================================================
def train_pure_rl(simulator, policy):
    reward_history = []
    mean_step_reward_history = []
    episode_times = []
    collision_history = []
    lane_change_history = []
    # Episode-level evaluation metrics:
    # collision_rate: 1 if at least one collision event occurs in the episode.
    # collisions: count of newly detected collision events, not persistent crash flags.
    # safe_travel_dis: collision-free distance accumulated before the first collision.
    # safe_step_count: first collision-event step, or episode length if none occurs.
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
    reward_function = build_fixed_reward_function()
    reward_specification = reward_function.reward_specification

    os.makedirs(os.path.dirname(reward_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    '''
    try:
        with open(DIAGNOSTIC_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")
    except OSError as exc:
        print("[DIAGNOSTIC LOG WARNING]", exc)
    '''

    # Phase I: collect trajectories with the initial policy and update it
    # from those returns before starting ordinary online RL.
    reference_data = collect_reference_rollouts(simulator, policy, reward_function)

    print('((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((')
    print('((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((')
    print('========================================================================================================================')


    warm_start_policy_from_reference(policy, reference_data)

    for episode in range(NUM_ONLINE_EPISODES):
        simulator._trace_phase = "pure_rl"
        simulator._trace_episode = episode + 1
        start_time = time.time()
        state = simulator.reset(randomize_traffic_gaps=True)
        record_episode_start(simulator, "pure_rl", episode + 1, state)
        episode_return = 0.0
        episode_steps = 0
        collisions = 0
        first_collision_step = None
        safe_travel_dis = 0.0
        previous_ego_position = np.asarray(simulator.env.unwrapped.vehicle.position, dtype=np.float64).copy()
        episode_actor_losses = []
        episode_critic_losses = []
        episode_speed_samples = []
        no_adjacent_lane_steps = 0
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
        seen_completed = {
            "LEFT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_LEFT", 0)),
            "RIGHT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_RIGHT", 0)),
        }

        for t in range(MAX_STEPS):
            valid_mask = simulator.get_valid_lane_action_mask()
            simulator._trace_valid_action_mask = np.asarray(valid_mask, dtype=bool).tolist()

            target_lane_mask = np.asarray(simulator.get_valid_target_lane_mask(), dtype=bool)
            # Count physical lane-change opportunities, not TTC-approved opportunities.
            if int(target_lane_mask.sum()) > 0:
                valid_target_lane_steps += 1
            else:
                no_adjacent_lane_steps += 1
            current_lane_id = int(simulator.env.unwrapped.vehicle.lane_index[2])
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
            filtered_action, safety_info = simulator.prepare_policy_action(
                sampled_action, selected_target_lane=selected_target_lane
            )
            filtered_action = np.asarray(filtered_action, dtype=np.float32).reshape(-1)
            record_lane_request_decision(lane_counters, sampled_action, valid_mask, safety_info)

            # Audit-only counters. In this baseline all intervention flags must remain
            # false because the policy action is passed through unchanged.
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
            intervention_counts["intervention_events"] += (
                int(emergency_intervention) + int(lane_change_rejected) + int(acceleration_override)
            )
            previous_speed_mps = float(simulator.env.unwrapped.vehicle.speed)
            vehicles_before = snapshot_simulator_vehicles(simulator)

            next_state, simulator_reward, done, step_info = simulator.step(
                filtered_action,
                sampled_action=sampled_action,
                safety_info=safety_info,
            )
            next_state = np.asarray(next_state, dtype=np.float32)
            executed_action = np.asarray(
                step_info.get("filtered_policy_action", filtered_action),
                dtype=np.float32,
            ).reshape(-1)
            if not np.allclose(executed_action, sampled_action, atol=1e-6, rtol=0.0):
                raise RuntimeError(
                    "Pure-RL invariant violated during online training: "
                    f"sampled_action={sampled_action.tolist()}, "
                    f"executed_action={executed_action.tolist()}"
                )

            min_ttc, closing_vehicle_count = calculate_min_ttc(simulator)
            # Count collision EVENTS, not the persistent ego.crashed state.
            collision_occurred = bool(
                step_info.get("collision_event", step_info.get("safety_violation", False))
                if isinstance(step_info, dict) else step_info
            )
            collisions += int(collision_occurred)
            current_ego_position = np.asarray(
                simulator.env.unwrapped.vehicle.position, dtype=np.float64
            ).copy()
            if collision_occurred:
                if first_collision_step is None:
                    # 1-based index: a collision on the first transition is step 1.
                    first_collision_step = episode_steps + 1
                # Do not count the collision transition as safely travelled distance.
            elif first_collision_step is None:
                # Once the first collision has occurred, later motion is not counted
                # as safely travelled distance even if the episode continues.
                safe_travel_dis += float(
                    np.linalg.norm(current_ego_position - previous_ego_position)
                )
            previous_ego_position = current_ego_position

            safety_violation = collision_occurred
            lane_info = simulator.get_lane_information()
            offroad = not bool(lane_info.get("on_road", True))
            lane_deviation = float(lane_info.get("lane_deviation", 0.0))
            if offroad:
                done = True

            reward_components = {}
            reward_safety_info = dict(safety_info or {})

            reward_safety_info["lane_change_accepted"] = bool(

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
                ego_acceleration_mps2=(
                    current_speed_mps - previous_speed_mps
                ) * float(simulation_frequency),
                safety_info=reward_safety_info,
                policy_sampled_action=sampled_action,
                reward_diagnostics=reward_components,
            )

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
        no_adjacent_lane_fraction = no_adjacent_lane_steps / max(episode_steps, 1)
        valid_adjacent_lane_fraction = valid_target_lane_steps / max(episode_steps, 1)
        reward_history.append(episode_return)
        mean_step_reward_history.append(mean_step_reward)
        episode_times.append(elapsed)
        collision_history.append(collisions)
        lane_stats = dict(getattr(simulator, "_lane_change_stats", {}))
        lane_change_history.append(lane_stats)

        # A collision-free episode contributes 0 to collision_rate. For an episode
        # without collision, safe_step_count is the full number of completed steps.
        collision_rate = float(first_collision_step is not None)
        collisions_per_1000_steps = float(collisions) / max(episode_steps, 1) * 1000.0
        lane_change_rate = float(lane_stats.get("completed", 0)) / max(episode_steps, 1)
        safe_step_count = int(
            first_collision_step if first_collision_step is not None else episode_steps
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
        interventions_per_km = intervention_counts["intervention_events"] / distance_km if distance_km > 1e-9 else 0.0
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
            f"mean_step_reward={mean_step_reward:+.4f} | collisions={collisions} | "
            f"mean_speed={mean_speed_mps:.2f}m/s ({mean_speed_ratio:.1%} target) | "
            f"speed>=90%target={speed_target_fraction:.1%} | "
            f"no_safe_lane={no_adjacent_lane_fraction:.1%} | "
            f"collision_rate={collision_rate:.0f} | "
            f"collision_events_per_1000_steps={collisions_per_1000_steps:.2f} | "
            f"lane_change_rate={lane_change_rate:.4f} | "
            f"safe_travel_dis={safe_travel_dis:.2f}m | "
            f"safe_step_count={safe_step_count} | "
            f"emergency_braking_rate={emergency_braking_rate:.4f} | "
            f"lane_change_rejection_rate={lane_change_rejection_rate:.4f} "
            f"({intervention_counts['lane_change_rejections']}/{intervention_counts['lane_change_requests']} requests) | "
            f"acceleration_override_rate={acceleration_override_rate:.4f} | "
            f"overall_intervention_rate={overall_intervention_rate:.4f} | "
            f"interventions_per_km={interventions_per_km:.3f} (safe distance={safe_travel_dis:.2f}m) | "
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
            "no_adjacent_lane_fraction": no_adjacent_lane_fraction,
            "valid_adjacent_lane_fraction": valid_adjacent_lane_fraction,
            "steps": int(episode_steps),
            "collisions": int(collisions),
            "collisions_per_1000_steps": float(collisions_per_1000_steps),
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
        }
        episode_metric_history.append(episode_metrics)

        write_diagnostic_event(
            "episode_summary", simulator,
            episode_return=float(episode_return),
            mean_reward_per_step=float(mean_step_reward),
            mean_speed_mps=mean_speed_mps,
            max_speed_mps=max_speed_mps,
            mean_speed_ratio=mean_speed_ratio,
            mean_speed_error_mps=mean_speed_error_mps,
            speed_target_fraction_90pct=speed_target_fraction,
            no_adjacent_lane_fraction=no_adjacent_lane_fraction,
            valid_adjacent_lane_fraction=valid_adjacent_lane_fraction,
            steps=int(episode_steps),
            collisions=int(collisions),
            collisions_per_1000_steps=float(collisions_per_1000_steps),
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
            first_collision_step=first_collision_step,
            lane_change_stats=lane_stats,
            lane_request_counters=lane_counters,
            mean_actor_loss=actor_loss_history[-1],
            mean_critic_loss=critic_loss_history[-1],
        )

        # Pure-RL baseline: reward specification remains fixed for every episode.
        assert reward_specification["collision_penalty_weight"] == INITIAL_COLLISION_PENALTY

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

    plt.figure()
    plt.plot(episodes, mean_step_reward_history, alpha=0.35, label="mean reward/step")
    if len(mean_step_reward_history) >= 3:
        window = min(10, len(mean_step_reward_history))
        smoothed = np.convolve(mean_step_reward_history, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, len(mean_step_reward_history)), smoothed,
                 linewidth=2.0, label=f"{window}-episode moving average")
    plt.xlabel("episode")
    plt.ylabel("mean reward per step")
    plt.title("Pure RL reward per step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_per_step_path)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()




    print("\\n[FINAL EPISODE METRICS]")
    print(f"collision_rate={float(np.mean(collision_rate_history)) if collision_rate_history else 0.0:.4f} (episodes with >=1 collision event)")
    total_online_steps = int(sum(m.get("steps", 0) for m in episode_metric_history))
    total_collision_events = int(sum(m.get("collisions", 0) for m in episode_metric_history))
    print(f"collision_events={total_collision_events} across {total_online_steps} steps")
    print(f"collisions_per_1000_steps={1000.0 * total_collision_events / max(total_online_steps, 1):.3f}")
    print(f"lane_change_rate={float(np.mean(lane_change_rate_history)) if lane_change_rate_history else 0.0:.4f}")
    print(f"safe_travel_dis={float(np.mean(safe_travel_dis_history)) if safe_travel_dis_history else 0.0:.2f} m/episode (mean)")
    print(f"safe_step_count={float(np.mean(safe_step_count_history)) if safe_step_count_history else 0.0:.2f} steps/episode (mean)")
    print(f"emergency_braking_rate={float(np.mean(emergency_braking_rate_history)) if emergency_braking_rate_history else 0.0:.4f} (per-step rate)")
    print(f"lane_change_rejection_rate={float(np.mean(lane_change_rejection_rate_history)) if lane_change_rejection_rate_history else 0.0:.4f} (per-request rate)")
    print(f"acceleration_override_rate={float(np.mean(acceleration_override_rate_history)) if acceleration_override_rate_history else 0.0:.4f} (per-step rate)")
    print(f"overall_intervention_rate={float(np.mean(overall_intervention_rate_history)) if overall_intervention_rate_history else 0.0:.4f} (per-step rate)")
    print(f"interventions_per_km={float(np.mean(interventions_per_km_history)) if interventions_per_km_history else 0.0:.3f} (safe_travel_dis is measured in metres)")


    with open(plot_path, "w", encoding="utf-8") as f:
        f.write(f"collision_rate={float(np.mean(collision_rate_history)) if collision_rate_history else 0.0:.4f} (episodes with >=1 collision event)"+ "\n")
        total_online_steps = int(sum(m.get("steps", 0) for m in episode_metric_history))
        total_collision_events = int(sum(m.get("collisions", 0) for m in episode_metric_history))
        f.write(f"collision_events={total_collision_events} across {total_online_steps} steps\n")
        f.write(f"collisions_per_1000_steps={1000.0 * total_collision_events / max(total_online_steps, 1):.3f}\n")
        f.write(f"lane_change_rate={float(np.mean(lane_change_rate_history)) if lane_change_rate_history else 0.0:.4f}"+ "\n")
        f.write(f"safe_travel_dis={float(np.mean(safe_travel_dis_history)) if safe_travel_dis_history else 0.0:.2f} m/episode (mean)"+ "\n")
        f.write(f"safe_step_count={float(np.mean(safe_step_count_history)) if safe_step_count_history else 0.0:.2f} steps/episode (mean)"+ "\n")
        f.write(f"emergency_braking_rate={float(np.mean(emergency_braking_rate_history)) if emergency_braking_rate_history else 0.0:.4f} (per-step rate)"+ "\n")
        f.write(f"lane_change_rejection_rate={float(np.mean(lane_change_rejection_rate_history)) if lane_change_rejection_rate_history else 0.0:.4f} (per-request rate)"+ "\n")
        f.write(f"acceleration_override_rate={float(np.mean(acceleration_override_rate_history)) if acceleration_override_rate_history else 0.0:.4f} (per-step rate)"+ "\n")
        f.write(f"overall_intervention_rate={float(np.mean(overall_intervention_rate_history)) if overall_intervention_rate_history else 0.0:.4f} (per-step rate)"+ "\n")
        f.write(f"interventions_per_km={float(np.mean(interventions_per_km_history)) if interventions_per_km_history else 0.0:.3f} (safe_travel_dis is measured in metres)"+ "\n")
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
    print("PURE RL EXPERIMENT 3: learned KEEP/CHANGE, target lane and acceleration; collisions are penalized but do not terminate episodes.")
    print(f"State dimension: {STATE_DIM}; policy actions: {POLICY_ACTION_DIM}; execution envelope: {ACTION_DIM}")
    print("Architecture: binary KEEP/CHANGE policy, physical-boundary-only target-lane mask, learned conditional acceleration.")
    print("Safety mode: diagnostic-only; TTC/gap do not mask actions, reject lane changes, or override acceleration.")
    print("Reward mode: fixed forward-progress and 5.0 collision-event penalty; collision does not terminate the episode.")
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








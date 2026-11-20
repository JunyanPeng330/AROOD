"""
H_EDM_PureRL.py — EDM-informed pure-RL autonomous-driving experiment.

Based on the I.py simulator module. Keeps the six-slot observation,
continuous ADT background traffic, physical lane-validity mask, collision
termination, and actor-critic implementation. EDM lane scores are appended
to the RL state as decision-support features only. The RL policy chooses
KEEP/CHANGE, the target adjacent lane, and continuous acceleration. TTC and
EDM do not override or veto the policy's decision.
"""

import os
import math
import time
import random
import json
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

import I as base


# ---------------------------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------------------------

SEED = 42
NUM_LANES = 3
SPEED_LIMIT_MPS = 50.0

# Traffic population and spatial management
TRAFFIC_TARGET_COUNT = 12
REFERENCE_RANGE_M = 500.0
RECYCLING_BOUNDARY_M = 700.0
INITIAL_PLACEMENT_RANGE_M = 350.0
MIN_INITIAL_GAP_M = 50.0
MIN_INSERTION_GAP_M = 35.0

# Maintain at least one front vehicle in every lane, not only the ego lane.
# Coverage vehicles are inserted well inside the ±500m observation range.
FRONT_COVERAGE_MAX_DISTANCE_M = 450.0
FRONT_COVERAGE_SPAWN_MIN_DISTANCE_M = 180.0
FRONT_COVERAGE_SPAWN_MAX_DISTANCE_M = 420.0
FRONT_COVERAGE_SPEED_OFFSET_MIN_MPS = 2.0
FRONT_COVERAGE_SPEED_OFFSET_MAX_MPS = 7.0
TRAFFIC_MIN_SPEED_MPS = 10.0
TRAFFIC_MAX_SPEED_MPS = 50.0

# Background IDM parameters (normal-driver baseline)
IDM_MAX_ACCEL_MPS2 = 2.0
IDM_COMFORTABLE_BRAKING_MPS2 = 3.5
IDM_MIN_GAP_M = 5.0
IDM_TIME_HEADWAY_S = 1.5
IDM_ACCEL_EXPONENT = 4.0

# Aggressive Dynamic Traffic (ADT): proportions of mutually exclusive profiles.
# These are vehicle-type proportions, not per-step event probabilities.
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
# Lane-change rates are per simulated second; attempts require an incentive
# and a target lane that passes the configured minimum gap/TTC checks.
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

# Desired-speed distributions by driver personality (m/s).
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

# Traffic-manager retry limits and boundary replacement geometry.
INITIAL_SPAWN_MAX_ATTEMPTS = 10000
LANE_COVERAGE_SPAWN_MAX_ATTEMPTS = 80
BOUNDARY_SPAWN_MAX_ATTEMPTS = 60
BOUNDARY_SPAWN_SPEED_OFFSET_MIN_MPS = 2.0
BOUNDARY_SPAWN_SPEED_OFFSET_MAX_MPS = 8.0
BOUNDARY_SPAWN_DISTANCE_BUFFER_M = 10.0
BOUNDARY_SPAWN_EXTRA_RANGE_M = 100.0

# Ego longitudinal capability and EDM configuration.
EGO_MAX_ACCEL_MPS2 = 5.0
EGO_MAX_BRAKING_MPS2 = 5.0
EGO_COMFORTABLE_BRAKING_MPS2 = 4.0
EGO_MIN_GAP_M = 5.0
EGO_TIME_HEADWAY_S = 1.5

# EDM is a continuous, predictive lane-evaluation signal, not a hard safety filter.
# It evaluates front and rear risk separately, then combines the worst risk.
# Safety has priority over sustainable acceleration efficiency.
EDM_SAFETY_WEIGHT = 0.70
EDM_ACCELERATION_WEIGHT = 0.30
EDM_TTC_CRITICAL_S = 4.0
EDM_TTC_WARNING_S = 6.0
EDM_GAP_CRITICAL_M = 5.0
EDM_GAP_WARNING_M = 50.0

# Reward: positive acceleration utility + non-negative speed utility + collision penalty.
# No explicit lane-change cost: one subsequent processing decision creates
# the intended implicit temporal cost. Both ordinary rewards are suspended
# during the lane-change processing decision.
LANE_CHANGE_COST_WEIGHT = 0.0
SPEED_REWARD_WEIGHT = 1.0
COLLISION_PENALTY = 5
LANE_CHANGE_PROCESSING_STEPS = 1

# EDM prediction is independent of lane-change processing.
EDM_PREDICTION_HORIZON_S = 2.0

# Temporal resolution (Hz). Change these at the top for frequency experiments.
# One RL decision spans 1 / POLICY_FREQUENCY_HZ seconds, regardless of how many
# simulator substeps occur inside that interval.
SIMULATION_FREQUENCY_HZ = 15
POLICY_FREQUENCY_HZ = 15
POLICY_DECISION_INTERVAL_S = 1.0 / float(POLICY_FREQUENCY_HZ)
MAX_ACCEL_DELTA_V_PER_DECISION_MPS = EGO_MAX_ACCEL_MPS2 * POLICY_DECISION_INTERVAL_S

MAX_BRAKE_DELTA_V_PER_DECISION_MPS = EGO_MAX_BRAKING_MPS2 * POLICY_DECISION_INTERVAL_S
LANE_CHANGE_PROCESSING_EFFECTIVE_DURATION_S = (
    LANE_CHANGE_PROCESSING_STEPS / float(POLICY_FREQUENCY_HZ)
)


# Training
NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 1500
DISCOUNT = 0.99
POLICY_LR = 3e-4
HIDDEN_DIM = 128

# Keep I.py's six-slot state layout and append three EDM relative-utility
# features (left/current/right absolute lane IDs). The current lane has utility 0.
BASE_STATE_DIM = base.STATE_DIM
STATE_DIM = BASE_STATE_DIM + NUM_LANES
base.STATE_DIM = STATE_DIM
ACTION_DIM = base.ACTION_DIM
POLICY_ACTION_DIM = base.POLICY_ACTION_DIM

MODEL_PATH = os.path.join("policy", "J_3_Diagnostics.pt")
PLOT_PATH = os.path.join("figure", "J_3_Diagnostics_return.png")
PLOT_TXT_PATH = os.path.join("figure", "plot_J_3_Diagnostics.txt")
METRICS_PATH = os.path.join("figure", "J_3_Diagnostics_episode_metrics.jsonl")
base.EPISODE_TRAJECTORY_PATH = os.path.join("figure", "J_3_Diagnostics_trajectories.jsonl")
EDM_ACTION_DIAGNOSTIC_PATH = os.path.join("figure", "J_3_edm_to_rl_action_diagnostics.jsonl")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not np.isclose(EDM_SAFETY_WEIGHT + EDM_ACCELERATION_WEIGHT, 1.0):
    raise ValueError("EDM component weights must sum to 1.0")
if POLICY_FREQUENCY_HZ <= 0 or SIMULATION_FREQUENCY_HZ <= 0:
    raise ValueError("Policy and simulation frequencies must be positive")
if (
    not isinstance(LANE_CHANGE_PROCESSING_STEPS, int)
    or LANE_CHANGE_PROCESSING_STEPS < 0
):
    raise ValueError("Lane-change processing steps must be a non-negative integer")


# ---------------------------------------------------------------------------
# 2. Background traffic vehicle
# ---------------------------------------------------------------------------

class ContinuousTrafficVehicle(base.Vehicle):
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

_ORIGINAL_ROAD_LONGITUDINAL_POSITION = base.HighwaySimulator._road_longitudinal_position
_ORIGINAL_STATE_ENCODER = base.HighwaySimulator._encode_state


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
            return np.zeros(base.VEHICLE_FEATURES, dtype=np.float32)
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
# 5. Pure-RL lane action masks: physical reachability only
# ---------------------------------------------------------------------------

def _valid_candidate_lanes(self):
    """Return all physically reachable adjacent lanes, with no traffic filtering.

    The mask encodes road topology only: the ego may KEEP its current lane or
    CHANGE one lane left/right when that lane exists. TTC, gaps, predicted
    acceleration, lane utility, and emergency rules do not remove candidates.
    """
    ego = self.env.unwrapped.vehicle
    current_lane = int(ego.lane_index[2])
    lane_count = int(self.env.unwrapped.config.get("lanes_count", NUM_LANES))
    return [
        lane_id for lane_id in (current_lane - 1, current_lane + 1)
        if 0 <= lane_id < lane_count
    ]


def _valid_lane_action_mask(self):
    """Binary policy mask [KEEP, CHANGE], constrained only by road boundaries."""
    return np.asarray([True, bool(_valid_candidate_lanes(self))], dtype=bool)


def _valid_target_lane_mask(self):
    """Allow the policy to learn among every physically reachable adjacent lane."""
    mask = np.zeros(NUM_LANES, dtype=bool)
    for lane_id in _valid_candidate_lanes(self):
        if 0 <= int(lane_id) < NUM_LANES:
            mask[int(lane_id)] = True
    return mask


def _validate_policy_action(self, sampled_action, selected_target_lane=None):
    """Pass policy actions through; validate only lane topology, never traffic safety."""
    action = np.asarray(sampled_action, dtype=np.float32).reshape(-1).copy()
    if action.size != 2:
        raise ValueError("Expected [normalized acceleration, binary KEEP/CHANGE].")

    ego = self.env.unwrapped.vehicle
    current_lane = int(ego.lane_index[2])
    requested = int(np.clip(np.rint(action[1]), 0, 1))
    candidates = _valid_candidate_lanes(self)
    target_lane = current_lane
    executed = requested
    invalid_target = False

    if requested == 1:
        try:
            selected = int(selected_target_lane)
        except (TypeError, ValueError):
            selected = current_lane
        if selected not in candidates:
            # Defensive fallback for malformed/external callers. The policy's
            # target mask prevents this during normal training.
            executed = 0
            invalid_target = True
        else:
            target_lane = selected

    action[0] = float(np.clip(action[0], -1.0, 1.0))
    action[1] = float(executed)
    request_id = None
    if requested:
        request_id = int(getattr(self, "_lane_request_trace_sequence", 0)) + 1
        self._lane_request_trace_sequence = request_id

    info = {
        "current_lane_id": current_lane,
        "candidate_lanes": [int(x) for x in candidates],
        "requested_target_lane_id": int(
            selected_target_lane if selected_target_lane is not None else current_lane
        ),
        "selected_target_lane_id": int(
            selected_target_lane if selected_target_lane is not None else current_lane
        ),
        "executed_target_lane_id": int(target_lane),
        "policy_requested_lane_action": "CHANGE" if requested else "KEEP",
        "executed_lane_action": "CHANGE" if executed else "KEEP",
        "lane_change_blocked": bool(invalid_target),
        "lane_change_rejection_reason": (
            "target_outside_physical_adjacent_lanes" if invalid_target else "none"
        ),
        "lane_request_trace_id": request_id,
        "action_validation_mode": "physical_lane_reachability_only",
        "traffic_safety_filter_applied": False,
        "ttc_filter_applied": False,
        "gap_filter_applied": False,
        "acceleration_override_applied": False,
        "emergency_braking": False,
        "safety_filter_applied": False,
        "overridden": bool(invalid_target),
        "sampled_action": np.asarray(sampled_action, dtype=np.float32).copy(),
        "filtered_action": action.copy(),
        "proposed_policy_acceleration": float(sampled_action[0]),
        "executed_acceleration": float(action[0]),
    }
    return action, info


# ---------------------------------------------------------------------------
# 6. EDM: continuous lane scores supplied to RL (not a safety shield)
# ---------------------------------------------------------------------------

def _linear_risk(value, critical, warning):
    """Map a small TTC/gap to risk 1 and a large TTC/gap to risk 0."""
    if not np.isfinite(value):
        return 0.0
    if warning <= critical:
        raise ValueError("EDM warning threshold must exceed its critical threshold")
    return float(np.clip((warning - float(value)) / (warning - critical), 0.0, 1.0))


def _nearest_lane_neighbors(self, lane_id, ego_s, ego, lane_ttc=None):
    """Return the exact front/rear neighbours used by the canonical TTC snapshot."""
    if lane_ttc is None:
        lane_ttc = self._min_ttc_by_lane()[int(lane_id)]
    vehicles_by_id = {
        id(vehicle): vehicle
        for vehicle in list(self.env.unwrapped.road.vehicles)
    }
    front = vehicles_by_id.get(lane_ttc.get("front_vehicle_id"))
    rear = vehicles_by_id.get(lane_ttc.get("rear_vehicle_id"))
    return (
        front,
        float(lane_ttc.get("front_gap", float("inf"))),
        rear,
        float(lane_ttc.get("rear_gap", float("inf"))),
    )


def _counterfactual_lane_acceleration(ego, front, front_gap):
    """IDM-like acceleration estimate if ego were travelling in this lane."""
    speed = max(0.0, float(ego.speed))
    desired_speed = max(SPEED_LIMIT_MPS, 1e-3)
    max_accel = EGO_MAX_ACCEL_MPS2
    comfortable_braking = EGO_COMFORTABLE_BRAKING_MPS2
    free_term = (speed / desired_speed) ** IDM_ACCEL_EXPONENT
    if front is None:
        acceleration = max_accel * (1.0 - free_term)
    else:
        front_speed = max(0.0, float(front.speed))
        closing_speed = speed - front_speed
        root = math.sqrt(max_accel * comfortable_braking)
        dynamic_gap = (
            speed * EGO_TIME_HEADWAY_S
            + speed * closing_speed / max(2.0 * root, 1e-6)
        )
        desired_gap = EGO_MIN_GAP_M + max(0.0, dynamic_gap)
        acceleration = max_accel * (
            1.0 - free_term - (desired_gap / max(front_gap, 0.5)) ** 2
        )
    return float(np.clip(acceleration, -EGO_MAX_BRAKING_MPS2, EGO_MAX_ACCEL_MPS2))


def compute_edm_lane_scores(self):
    """Evaluate every lane using predictive front/rear safety and sustainable efficiency.

    Safety uses the worse of current TTC/gap risk and constant-velocity gap risk
    projected across EDM_PREDICTION_HORIZON_S. Efficiency uses the more
    conservative of current and horizon-projected IDM acceleration. The final
    utility keeps the configured safety/efficiency weighting.
    """
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    ego_speed = max(0.0, float(ego.speed))
    horizon = max(0.0, float(EDM_PREDICTION_HORIZON_S))
    scores, details = [], {}
    ttc_snapshot = self._min_ttc_by_lane()

    for lane_id in range(NUM_LANES):
        # Use the simulator's canonical TTC/gap snapshot. EDM and TTC therefore
        # share both the selected neighbour and its signed bumper clearance.
        lane_ttc = ttc_snapshot[int(lane_id)]
        front, front_gap, rear, rear_gap = _nearest_lane_neighbors(
            self, lane_id, ego_s, ego, lane_ttc=lane_ttc
        )
        front_ttc = float(lane_ttc["front_ttc"])
        rear_ttc = float(lane_ttc["rear_ttc"])
        front_speed = (
            max(0.0, float(front.speed)) if front is not None else ego_speed
        )
        rear_speed = (
            max(0.0, float(rear.speed)) if rear is not None else ego_speed
        )
        # Keep the gaps exactly aligned with the TTC calculation, including
        # negative clearances when vehicle bodies overlap.
        front_gap = float(lane_ttc["front_gap"])
        rear_gap = float(lane_ttc["rear_gap"])

        # Constant-velocity prediction over the independent EDM prediction horizon.
        # Positive values are bumper-to-bumper clearances in metres.
        predicted_front_gap = (
            max(0.0, front_gap + (front_speed - ego_speed) * horizon)
            if front is not None else float("inf")
        )
        predicted_rear_gap = (
            max(0.0, rear_gap + (ego_speed - rear_speed) * horizon)
            if rear is not None else float("inf")
        )

        front_ttc_risk = _linear_risk(
            front_ttc, EDM_TTC_CRITICAL_S, EDM_TTC_WARNING_S
        )
        rear_ttc_risk = _linear_risk(
            rear_ttc, EDM_TTC_CRITICAL_S, EDM_TTC_WARNING_S
        )
        front_gap_risk = max(
            _linear_risk(front_gap if front is not None else float("inf"),
                         EDM_GAP_CRITICAL_M, EDM_GAP_WARNING_M),
            _linear_risk(predicted_front_gap,
                         EDM_GAP_CRITICAL_M, EDM_GAP_WARNING_M),
        )
        rear_gap_risk = max(
            _linear_risk(rear_gap if rear is not None else float("inf"),
                         EDM_GAP_CRITICAL_M, EDM_GAP_WARNING_M),
            _linear_risk(predicted_rear_gap,
                         EDM_GAP_CRITICAL_M, EDM_GAP_WARNING_M),
        )
        front_risk = max(front_ttc_risk, front_gap_risk)
        rear_risk = max(rear_ttc_risk, rear_gap_risk)
        safety_utility = float(np.clip(
            1.0 - max(front_risk, rear_risk), 0.0, 1.0
        ))

        current_accel = _counterfactual_lane_acceleration(
            ego, front, front_gap
        )
        projected_accel = _counterfactual_lane_acceleration(
            ego, front, predicted_front_gap
        ) if front is not None else current_accel
        sustainable_accel = min(current_accel, projected_accel)
        acceleration_utility = float(np.clip(
            (sustainable_accel + EGO_MAX_BRAKING_MPS2)
            / (EGO_MAX_ACCEL_MPS2 + EGO_MAX_BRAKING_MPS2),
            0.0, 1.0,
        ))

        score = (
            EDM_SAFETY_WEIGHT * safety_utility
            + EDM_ACCELERATION_WEIGHT * acceleration_utility
        )
        score = float(np.clip(score, 0.0, 1.0))
        scores.append(score)
        details[int(lane_id)] = {
            "score": score,
            "safety_utility": safety_utility,
            "front_safety_risk": float(front_risk),
            "rear_safety_risk": float(rear_risk),
            "front_ttc_risk": float(front_ttc_risk),
            "rear_ttc_risk": float(rear_ttc_risk),
            "front_gap_risk": float(front_gap_risk),
            "rear_gap_risk": float(rear_gap_risk),
            "acceleration_utility": acceleration_utility,
            "current_predicted_acceleration_mps2": float(current_accel),
            "horizon_predicted_acceleration_mps2": float(projected_accel),
            "sustainable_predicted_acceleration_mps2": float(sustainable_accel),
            "front_gap_m": float(front_gap),
            "rear_gap_m": float(rear_gap),
            "predicted_front_gap_m": float(predicted_front_gap),
            "predicted_rear_gap_m": float(predicted_rear_gap),
            "front_ttc_s": float(front_ttc),
            "rear_ttc_s": float(rear_ttc),
            "prediction_horizon_s": float(horizon),
        }

    self._last_edm_lane_details = details
    return np.asarray(scores, dtype=np.float32)


# Preserve the original six-slot/context state and append three relative EDM features.
_BASE_SIX_SLOT_STATE = _six_slot_state

def _edm_augmented_state(self, obs=None):
    base_state = np.asarray(_BASE_SIX_SLOT_STATE(self, obs), dtype=np.float32).reshape(-1)
    scores = compute_edm_lane_scores(self)
    current_lane = int(self.env.unwrapped.vehicle.lane_index[2])
    # Relative utilities make the opportunity (or disadvantage) of changing
    # lanes explicit: current lane is zero; positive values favour alternatives.
    scores = np.asarray(scores, dtype=np.float32).reshape(NUM_LANES)
    if current_lane < 0 or current_lane >= NUM_LANES:
        raise ValueError(f"Invalid current lane ID for EDM state: {current_lane}")

    # Express each lane's utility relative to the lane currently occupied.
    # This keeps the current-lane feature at zero and preserves all three lanes.
    relative_utilities = np.clip(
        scores - float(scores[current_lane]), -1.0, 1.0
    ).astype(np.float32)

    augmented_state = np.concatenate(
        (base_state, relative_utilities)
    ).astype(np.float32)

    if augmented_state.size != STATE_DIM:
        raise ValueError(
            f"EDM state dimension mismatch: expected {STATE_DIM}, "
            f"got {augmented_state.size}"
        )
    return augmented_state


# ---------------------------------------------------------------------------
# 7. Reward: acceleration utility + non-negative speed utility + collision penalty
# ---------------------------------------------------------------------------

def calculate_reward(
    actual_delta_v, current_speed, lane_change_executed, collision,
    lane_change_processing_step=False,
):
    """Combine non-negative acceleration and speed rewards with collision cost.

    Acceleration reward preserves the existing per-decision design:
        R_acc = max(delta_v, 0) / max_delta_v_per_decision
    and remains 1.0 at/above the speed limit, as in the previous model.

    Speed reward is independently non-negative:
        R_speed = clip(current_speed / speed_limit, 0, 1)

    Both ordinary rewards are suspended during the lane-change processing
    decision. The collision penalty remains active, including during processing.
    """
    if lane_change_processing_step:
        acceleration_reward = 0.0
        speed_reward = 0.0
    else:
        if float(current_speed) >= SPEED_LIMIT_MPS - 1e-3:
            acceleration_reward = 1.0
        else:
            denominator = max(float(MAX_ACCEL_DELTA_V_PER_DECISION_MPS), 1e-8)
            delta_v = float(actual_delta_v)
            positive_delta_v = (abs(delta_v) + delta_v) / 2.0
            acceleration_reward = float(np.clip(
                positive_delta_v / denominator, 0.0, 1.0
            ))

        speed_reward = float(SPEED_REWARD_WEIGHT) * float(np.clip(
            float(current_speed) / max(float(SPEED_LIMIT_MPS), 1e-8),
            0.0, 1.0,
        ))

    collision_cost = -float(COLLISION_PENALTY) if collision else 0.0
    total = acceleration_reward + speed_reward + collision_cost
    delta_v = float(actual_delta_v)
    positive_delta_v = (abs(delta_v) + delta_v) / 2.0
    return total, {
        "acceleration_reward": float(acceleration_reward),
        "speed_reward": float(speed_reward),
        "speed_reward_weight": float(SPEED_REWARD_WEIGHT),
        "speed_reward_raw": float(np.clip(
            float(current_speed) / max(float(SPEED_LIMIT_MPS), 1e-8),
            0.0, 1.0,
        )),
        "ordinary_reward_suspended": bool(lane_change_processing_step),
        "lane_change_processing_step": bool(lane_change_processing_step),
        "actual_delta_v_mps_per_decision": delta_v,
        "positive_delta_v_mps_per_decision": float(positive_delta_v),
        "max_acceleration_delta_v_mps_per_decision": float(MAX_ACCEL_DELTA_V_PER_DECISION_MPS),
        "policy_decision_interval_s": float(POLICY_DECISION_INTERVAL_S),
        "lane_change_cost": 0.0,
        "lane_change_executed": bool(lane_change_executed),
        "collision_penalty": float(collision_cost),
        "total_reward": float(total),
    }


# ---------------------------------------------------------------------------
# 7. Bind the new mechanisms to the retained HighwaySimulator
# ---------------------------------------------------------------------------

base.HighwaySimulator._lane_index_for_id = lambda self, lane_id: (
    self.env.unwrapped.vehicle.lane_index[0],
    self.env.unwrapped.vehicle.lane_index[1],
    int(lane_id),
)
base.HighwaySimulator._road_longitudinal_position = _road_s
base.HighwaySimulator.idm_acceleration = _idm_acceleration
base.HighwaySimulator.adt_vehicle_acceleration = _adt_vehicle_acceleration
base.HighwaySimulator.adt_try_lane_change = _adt_try_lane_change
base.HighwaySimulator._spawn_traffic_vehicle = _spawn_traffic_vehicle
base.HighwaySimulator._initialize_six_vehicle_traffic = _initialize_continuous_traffic
base.HighwaySimulator._maintain_six_vehicle_traffic = _maintain_continuous_traffic
base.HighwaySimulator._assert_traffic_invariant = _assert_continuous_traffic_invariant
base.HighwaySimulator._sanitize_traffic_slot_speeds = _sanitize_all_traffic_speeds
base.HighwaySimulator._snapshot_managed_traffic_slots = _snapshot_all_traffic
base.HighwaySimulator._record_first_negative_speed_transition = _no_negative_transition_record
base.HighwaySimulator._encode_state = _edm_augmented_state
base.HighwaySimulator.get_valid_candidate_lanes = _valid_candidate_lanes
base.HighwaySimulator.get_valid_lane_action_mask = _valid_lane_action_mask
base.HighwaySimulator.get_valid_target_lane_mask = _valid_target_lane_mask
base.HighwaySimulator.validate_policy_action = _validate_policy_action

# Extend the original state with EDM scores; retain the original action dimensions and policy implementation.
base.POLICY_LR = POLICY_LR
base.HIDDEN_DIM = HIDDEN_DIM
base.MAX_STEPS = MAX_STEPS_PER_EPISODE

# The base class captured its original state_dim as a Python default argument.
# Pass the augmented dimension explicitly through this factory.
def create_edm_highway_simulator(render_mode=None):
    return base.HighwaySimulator(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_steps=MAX_STEPS_PER_EPISODE,
        render_mode=render_mode,
    )


# ---------------------------------------------------------------------------
# 8. EDM-to-RL information-flow diagnostics (logging only; no action authority)
# ---------------------------------------------------------------------------

def _policy_distribution_diagnostics(
    policy, state, valid_action_mask, target_lane_mask,
    sampled_action, selected_target_lane, pre_tanh_accel,
):
    """Read policy distributions without sampling or changing the policy action."""
    with torch.no_grad():
        features = policy.features(state)
        raw_action_logits = policy.action_logits(features)
        raw_action_probs = torch.softmax(raw_action_logits, dim=-1)
        masked_action_dist = policy.action_distributions(state, valid_action_mask)

        raw_lane_logits = policy.lane_rank_logits(features)
        raw_lane_probs = torch.softmax(raw_lane_logits, dim=-1)
        masked_lane_dist = policy.lane_distributions(state, target_lane_mask)

        binary_action = int(np.clip(np.rint(float(sampled_action[1])), 0, 1))
        target_lane = int(np.clip(selected_target_lane, 0, NUM_LANES - 1))
        acceleration_dist = policy.acceleration_distribution(
            state, binary_action, target_lane
        )

    def _one_row(tensor):
        return tensor.detach().cpu().numpy().reshape(-1).astype(float).tolist()

    return {
        "binary_action_raw_logits": _one_row(raw_action_logits),
        "binary_action_raw_probabilities": _one_row(raw_action_probs),
        "binary_action_masked_probabilities": _one_row(masked_action_dist.probs),
        "target_lane_raw_logits": _one_row(raw_lane_logits),
        "target_lane_raw_probabilities": _one_row(raw_lane_probs),
        "target_lane_masked_probabilities": _one_row(masked_lane_dist.probs),
        "acceleration_pre_tanh_mean": float(acceleration_dist.mean[0].item()),
        "acceleration_std_pre_tanh": float(acceleration_dist.stddev[0].item()),
        "acceleration_mean_after_tanh": float(torch.tanh(acceleration_dist.mean[0]).item()),
        "sampled_pre_tanh_acceleration": float(pre_tanh_accel),
        "sampled_normalized_acceleration": float(sampled_action[0]),
    }


def _ttc_snapshot_for_diagnostics(simulator):
    """Capture the canonical pre/post TTC snapshot without modifying it."""
    snapshot = simulator._min_ttc_by_lane()
    fields = (
        "front_gap", "rear_gap", "front_ttc", "rear_ttc",
        "front_speed", "rear_speed", "front_risk", "rear_risk", "lane_risk",
        "front_vehicle_id", "rear_vehicle_id",
    )
    return {
        str(lane_id): {key: values.get(key) for key in fields}
        for lane_id, values in sorted(snapshot.items())
    }


def _observation_information_diagnostics(state, current_lane, edm_scores, ttc_snapshot):
    """Decode the exact TTC/gap and EDM features supplied to the RL policy."""
    state_array = np.asarray(state, dtype=np.float32).reshape(-1)
    ttc_start = int(base.VEHICLE_STATE_DIM + base.CONTEXT_DIM)
    ttc_end = ttc_start + int(base.SAFETY_FEATURES_DIM)
    ttc_features = state_array[ttc_start:ttc_end].reshape(
        NUM_LANES, base.SAFETY_FEATURES_PER_LANE
    )
    edm_features = state_array[-NUM_LANES:]

    keys = ("front_gap", "rear_gap", "front_ttc", "rear_ttc", "front_speed", "rear_speed")
    scales = (100.0, 100.0, 10.0, 10.0, 40.0, 40.0)
    expected_ttc_features = np.zeros_like(ttc_features)
    for lane_id in range(NUM_LANES):
        lane_values = ttc_snapshot[str(lane_id)]
        for feature_index, (key, scale) in enumerate(zip(keys, scales)):
            value = lane_values.get(key)
            value = scale if value is None else float(value)
            if not np.isfinite(value):
                value = scale * 1.5
            expected_ttc_features[lane_id, feature_index] = np.clip(
                value / scale, -1.0, 1.5
            )

    scores = np.asarray(edm_scores, dtype=np.float32).reshape(NUM_LANES)
    expected_edm_features = np.clip(
        scores - float(scores[int(current_lane)]), -1.0, 1.0
    )
    ttc_error = float(np.max(np.abs(ttc_features - expected_ttc_features)))
    edm_error = float(np.max(np.abs(edm_features - expected_edm_features)))

    return {
        "state_dimension": int(state_array.size),
        "state_ttc_gap_feature_order": list(keys),
        "state_ttc_gap_features_normalized": ttc_features.tolist(),
        "expected_ttc_gap_features_from_current_world": expected_ttc_features.tolist(),
        "ttc_gap_observation_max_abs_error": ttc_error,
        "ttc_gap_observation_matches_current_world": bool(ttc_error <= 1e-5),
        "state_edm_relative_utilities": edm_features.astype(float).tolist(),
        "expected_edm_relative_utilities": expected_edm_features.astype(float).tolist(),
        "edm_observation_max_abs_error": edm_error,
        "edm_observation_matches_current_scores": bool(edm_error <= 1e-5),
        "policy_received_ttc_gap_features": True,
        "policy_received_edm_relative_utility_features": True,
    }


def _append_edm_action_diagnostic(record):
    """Append one strict JSON record; diagnostics must never interrupt training."""
    try:
        os.makedirs(os.path.dirname(EDM_ACTION_DIAGNOSTIC_PATH) or ".", exist_ok=True)
        safe_record = base._trajectory_jsonable(record)
        with open(EDM_ACTION_DIAGNOSTIC_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(
                safe_record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ) + "\n")
            log_file.flush()
    except (OSError, TypeError, ValueError) as exc:
        print(
            "[EDM ACTION DIAGNOSTIC LOG WARNING]",
            f"path={EDM_ACTION_DIAGNOSTIC_PATH}", f"error={exc}",
        )


# ---------------------------------------------------------------------------
# 9. New concise online actor-critic training loop
# ---------------------------------------------------------------------------

def train(simulator, policy):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    episode_returns = []
    metrics_history = []

    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        simulator._trace_phase = "H_EDM_PureRL"
        simulator._trace_episode = episode + 1
        state = np.asarray(simulator.reset(seed=SEED + episode), dtype=np.float32)
        base.record_episode_start(
            simulator, "H_EDM_PureRL", episode + 1, state
        )

        episode_return = 0.0
        collisions = 0
        lane_changes = 0
        speeds = []
        acceleration_rewards = []
        lane_costs = []
        edm_score_history = []
        safe_distance = 0.0
        safe_steps = 0
        previous_position = np.asarray(
            simulator.env.unwrapped.vehicle.position, dtype=float
        ).copy()
        actor_losses = []
        critic_losses = []
        rejection_count = 0
        change_requests = 0
        lane_change_processing_steps = 0
        lane_change_processing_steps_remaining = 0
        edm_current_lane_warning_steps = 0
        edm_warning_change_requests = 0
        edm_warning_keep_requests = 0
        edm_warning_deceleration_commands = 0

        # Use road-longitudinal coordinates for distance metrics. Euclidean
        # position deltas can include lateral lane displacement and distort
        # the reported travel distance.
        previous_road_s = _road_s(
            simulator, simulator.env.unwrapped.vehicle
        )

        for step in range(MAX_STEPS_PER_EPISODE):
            ego = simulator.env.unwrapped.vehicle
            current_lane = int(ego.lane_index[2])
            in_lane_change_processing = lane_change_processing_steps_remaining > 0
            if in_lane_change_processing:
                valid_action_mask = np.asarray([True, False], dtype=bool)
                target_lane_mask = np.zeros(NUM_LANES, dtype=bool)
                target_lane_mask[current_lane] = True
            else:
                valid_action_mask = simulator.get_valid_lane_action_mask()
                target_lane_mask = simulator.get_valid_target_lane_mask()
            edm_scores = compute_edm_lane_scores(simulator)
            edm_score_history.append(edm_scores.copy())
            # Capture the pre-action values now. The next_state encoder will
            # refresh _last_edm_lane_details after the simulator update.
            edm_details_before = deepcopy(getattr(simulator, "_last_edm_lane_details", {}))
            ttc_before = _ttc_snapshot_for_diagnostics(simulator)
            observation_diagnostics = _observation_information_diagnostics(
                state, current_lane, edm_scores, ttc_before
            )
            current_lane_edm = edm_details_before.get(current_lane, {})
            current_lane_warning_risk = max(
                float(current_lane_edm.get("front_safety_risk", 0.0)),
                float(current_lane_edm.get("rear_safety_risk", 0.0)),
            )
            current_lane_warning = bool(current_lane_warning_risk > 0.0)

            sampled_action, aux = policy.act(
                state,
                deterministic=False,
                valid_action_mask=valid_action_mask,
                valid_target_lane_mask=target_lane_mask,
                current_lane_id=current_lane,
                return_aux=True,
            )
            selected_target_lane = int(aux.get("selected_target_lane", current_lane))
            if int(round(float(sampled_action[1]))) == 1:
                change_requests += 1
            policy_diagnostics = _policy_distribution_diagnostics(
                policy=policy,
                state=state,
                valid_action_mask=valid_action_mask,
                target_lane_mask=target_lane_mask,
                sampled_action=sampled_action,
                selected_target_lane=selected_target_lane,
                pre_tanh_accel=aux.get("pre_tanh_accel", 0.0),
            )

            filtered_action, safety_info = simulator.validate_policy_action(
                sampled_action,
                selected_target_lane=selected_target_lane,
            )
            if safety_info.get("lane_change_blocked", False):
                rejection_count += 1

            speed_before = float(ego.speed)
            road_s_before = _road_s(simulator, ego)
            lane_before_step = int(ego.lane_index[2])
            vehicles_before = (
                base.snapshot_simulator_vehicles(simulator)
                if base.TRAJECTORY_RECORDING_ENABLED else []
            )
            next_state, _, simulator_done, step_info = simulator.step(
                filtered_action,
                apply_safety_filter=False,
                sampled_action=sampled_action,
                safety_info=safety_info,
            )

            # Collision is an episode-terminal event for this experiment.
            # I.HighwaySimulator records the collision and clears the transient
            # crash latch so the simulator can continue if requested. This
            # training loop treats the event as terminal and resets immediately:
            # record the first collision, mark it terminal for the RL update, and
            # reset before another simulator action is taken.
            step_info = dict(step_info or {})
            collision = bool(step_info.get("collision", False))
            episode_terminated = bool(
                collision or step_info.get("terminated", False)
            )
            episode_truncated = bool(
                step_info.get("truncated", False) and not episode_terminated
            )
            episode_done = bool(
                simulator_done or episode_terminated or episode_truncated
            )

            if collision:
                step_info.update({
                    "terminated": True,
                    "truncated": False,
                    "episode_terminated_by_collision": True,
                    "termination_reason": "collision",
                    "collision_continuation_enabled": False,
                })
            else:
                step_info["terminated"] = episode_terminated
                step_info["truncated"] = episode_truncated
                step_info["episode_terminated_by_collision"] = False
                if episode_done:
                    step_info["termination_reason"] = (
                        "time_limit" if episode_truncated else "environment_terminal"
                    )

            ego_after_step = simulator.env.unwrapped.vehicle
            # I.HighwaySimulator._encode_state() has already recomputed EDM
            # details for the post-action world while creating next_state.
            edm_details_after = deepcopy(getattr(simulator, "_last_edm_lane_details", {}))
            edm_scores_after = [
                float(edm_details_after.get(lane_id, {}).get("score", 0.0))
                for lane_id in range(NUM_LANES)
            ]
            ttc_after = step_info.get("ttc_risk_diagnostics", {})
            lane_after_step = int(ego_after_step.lane_index[2])
            lane_change_executed = bool(lane_after_step != lane_before_step)
            speed = float(ego_after_step.speed)
            actual_delta_v = speed - speed_before
            reward, reward_components = calculate_reward(
                actual_delta_v,
                speed,
                lane_change_executed,
                collision,
                lane_change_processing_step=in_lane_change_processing,
            )

            # The initiation step keeps its normal reward. The following
            # processing steps mask CHANGE and suspend ordinary reward.
            if in_lane_change_processing:
                lane_change_processing_steps += 1
                lane_change_processing_steps_remaining = max(
                    0, lane_change_processing_steps_remaining - 1
                )
            elif lane_change_executed and not collision:
                lane_change_processing_steps_remaining = LANE_CHANGE_PROCESSING_STEPS

            step_info.update({
                "lane_change_processing_active": bool(in_lane_change_processing),
                "lane_change_processing_step": bool(in_lane_change_processing),
                "lane_change_processing_steps_remaining": int(
                    lane_change_processing_steps_remaining
                ),
                "lane_change_processing_duration_s": float(
                    LANE_CHANGE_PROCESSING_EFFECTIVE_DURATION_S
                ),
                "lane_change_processing_total_steps": int(
                    LANE_CHANGE_PROCESSING_STEPS
                ),
                "ordinary_reward_suspended": bool(in_lane_change_processing),
            })
            step_info["edm_lane_scores_pre_action"] = {
                str(lane_id): float(edm_scores[lane_id])
                for lane_id in range(NUM_LANES)
            }
            step_info["edm_lane_details_pre_action"] = edm_details_before
            step_info["edm_lane_scores_post_action"] = {
                str(lane_id): float(edm_scores_after[lane_id])
                for lane_id in range(NUM_LANES)
            }
            step_info["edm_lane_details_post_action"] = edm_details_after
            step_info["policy_selected_target_lane"] = int(selected_target_lane)

            if collision:
                collisions += 1
            if lane_change_executed:
                lane_changes += 1
            if current_lane_warning:
                edm_current_lane_warning_steps += 1
                if int(round(float(sampled_action[1]))) == 1:
                    edm_warning_change_requests += 1
                else:
                    edm_warning_keep_requests += 1
                if float(sampled_action[0]) < 0.0:
                    edm_warning_deceleration_commands += 1
            if not in_lane_change_processing:
                acceleration_rewards.append(reward_components["acceleration_reward"])
            lane_costs.append(0.0)

            # Measure forward road progress in metres, not 2-D Euclidean
            # displacement. This excludes lateral lane-change displacement.
            road_s_after = _road_s(
                simulator, simulator.env.unwrapped.vehicle
            )
            step_distance_m = max(0.0, float(road_s_after - road_s_before))
            if not collision:
                safe_distance += step_distance_m
                safe_steps += 1
            previous_road_s = road_s_after
            speed = float(simulator.env.unwrapped.vehicle.speed)
            speeds.append(speed)

            # One row per decision connects the exact observation, EDM/TTC
            # warning, policy distribution, sampled action and subsequent result.
            current_lane_post_ttc = ttc_after.get(
                lane_after_step, ttc_after.get(str(lane_after_step), {})
            )
            diagnostic_record = {
                "record_type": "edm_to_rl_action_diagnostic",
                "schema_version": 1,
                "run_id": base.TRAJECTORY_RUN_ID,
                "episode": int(episode + 1),
                "step": int(step + 1),
                "simulator_step_after_transition": int(getattr(simulator, "step_count", -1)),
                "pre_action": {
                    "ego_lane": int(current_lane),
                    "ego_speed_mps": float(speed_before),
                    "lane_change_processing_active": bool(in_lane_change_processing),
                    "ttc_by_lane": ttc_before,
                    "edm_scores_by_lane": {
                        str(lane_id): float(edm_scores[lane_id])
                        for lane_id in range(NUM_LANES)
                    },
                    "edm_details_by_lane": edm_details_before,
                    "current_lane_warning": current_lane_warning,
                    "current_lane_warning_risk": float(current_lane_warning_risk),
                    "any_lane_warning": bool(any(
                        max(
                            float(values.get("front_safety_risk", 0.0)),
                            float(values.get("rear_safety_risk", 0.0)),
                        ) > 0.0
                        for values in edm_details_before.values()
                    )),
                },
                "rl_observation": observation_diagnostics,
                "policy": {
                    "valid_action_mask_keep_change": np.asarray(valid_action_mask, dtype=bool).tolist(),
                    "valid_target_lane_mask": np.asarray(target_lane_mask, dtype=bool).tolist(),
                    "distribution_diagnostics": policy_diagnostics,
                    "sampled_action": np.asarray(sampled_action, dtype=np.float32).tolist(),
                    "selected_target_lane": int(selected_target_lane),
                    "action_after_physical_validation": np.asarray(filtered_action, dtype=np.float32).tolist(),
                    "validation_info": safety_info,
                },
                "post_action_outcome": {
                    "ego_lane": int(lane_after_step),
                    "ego_speed_mps": float(speed),
                    "actual_delta_v_mps": float(actual_delta_v),
                    "lane_change_executed": bool(lane_change_executed),
                    "collision": bool(collision),
                    "terminated": bool(episode_terminated),
                    "termination_reason": step_info.get("termination_reason"),
                    "ttc_risk_by_lane_after_action": ttc_after,
                    "current_lane_ttc_after_action": current_lane_post_ttc,
                    "edm_scores_by_lane_after_action": {
                        str(lane_id): float(edm_scores_after[lane_id])
                        for lane_id in range(NUM_LANES)
                    },
                    "edm_details_by_lane_after_action": edm_details_after,
                    "reward": float(reward),
                    "reward_components": reward_components,
                },
            }
            _append_edm_action_diagnostic(diagnostic_record)

            if current_lane_warning and (
                current_lane_warning_risk >= 0.5 or collision
            ):
                print(
                    "[EDM->RL RESPONSE]",
                    f"episode={episode + 1}", f"step={step + 1}",
                    f"lane={current_lane}",
                    f"warning_risk={current_lane_warning_risk:.3f}",
                    f"P(CHANGE)={policy_diagnostics['binary_action_masked_probabilities'][1]:.3f}",
                    f"sampled_lane_action={'CHANGE' if int(round(float(sampled_action[1]))) else 'KEEP'}",
                    f"target_lane={selected_target_lane}",
                    f"accel_norm={float(sampled_action[0]):+.3f}",
                    f"delta_v={actual_delta_v:+.3f}m/s",
                    f"collision={collision}",
                )

            base.record_episode_transition(
                simulator=simulator,
                phase="H_EDM_PureRL",
                episode_number=episode + 1,
                step_number=step + 1,
                state_before=state,
                state_after=next_state,
                vehicles_before=vehicles_before,
                sampled_action=sampled_action.tolist(),
                executed_action=filtered_action.tolist(),
                selected_target_lane=selected_target_lane,
                valid_action_mask=np.asarray(valid_action_mask, dtype=bool).tolist(),
                target_lane_mask=np.asarray(target_lane_mask, dtype=bool).tolist(),
                safety_info=safety_info,
                step_info=step_info,
                reward=reward,
                reward_components=reward_components,
                done=episode_done,
                terminated=episode_terminated,
                truncated=episode_truncated,
            )

            update = base.update_policy(
                policy=policy,
                state=state,
                action=sampled_action,
                reward=reward,
                next_state=next_state,
                done=episode_done,
                valid_action_mask=valid_action_mask,
                terminated=episode_terminated,
                pre_tanh_accel=aux.get("pre_tanh_accel"),
                selected_target_lane=selected_target_lane,
                valid_target_lane_mask=target_lane_mask,
                current_lane_id=current_lane,
            )
            actor_losses.append(update["actor_loss"])
            critic_losses.append(update["critic_loss"])
            episode_return += reward
            state = np.asarray(next_state, dtype=np.float32)

            if episode_done:
                break

        steps = step + 1
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        collision_rate = float(collisions > 0)
        lane_change_rate = lane_changes / max(steps, 1)
        speed_target_fraction = (
            float(np.mean(np.asarray(speeds) >= 0.9 * SPEED_LIMIT_MPS))
            if speeds else 0.0
        )
        row = {
            "episode": episode + 1,
            "steps": steps,
            "terminated_by_collision": bool(collisions > 0),
            "termination_reason": (
                "collision" if collisions > 0
                else step_info.get("termination_reason", "max_steps")
            ),
            "return": float(episode_return),
            "collisions": int(collisions),
            "collision_rate": collision_rate,
            "lane_changes": int(lane_changes),
            "lane_change_rate": float(lane_change_rate),
            "lane_change_processing_steps": int(lane_change_processing_steps),
            "lane_change_processing_step_ratio": float(
                lane_change_processing_steps / max(steps, 1)
            ),
            "lane_change_processing_duration_s": float(
                LANE_CHANGE_PROCESSING_EFFECTIVE_DURATION_S
            ),
            "lane_change_processing_total_steps": int(LANE_CHANGE_PROCESSING_STEPS),
            "lane_change_requests": int(change_requests),
            "lane_change_rejections": int(rejection_count),
            "edm_current_lane_warning_steps": int(edm_current_lane_warning_steps),
            "edm_warning_change_requests": int(edm_warning_change_requests),
            "edm_warning_keep_requests": int(edm_warning_keep_requests),
            "edm_warning_deceleration_commands": int(edm_warning_deceleration_commands),
            "lane_change_rejection_rate": (
                rejection_count / change_requests if change_requests else 0.0
            ),
            "mean_speed_mps": mean_speed,
            "speed_at_least_90pct_limit": speed_target_fraction,
            "safe_travel_distance_m": float(safe_distance),
            "safe_steps": int(safe_steps),
            "safe_step_ratio": float(safe_steps / max(steps, 1)),
            "mean_acceleration_reward": (
                float(np.mean(acceleration_rewards)) if acceleration_rewards else 0.0
            ),
            "mean_lane_change_cost": 0.0,
            "mean_edm_left_score": (
                float(np.mean([x[0] for x in edm_score_history])) if edm_score_history else 0.0
            ),
            "mean_edm_current_score": (
                float(np.mean([x[1] for x in edm_score_history])) if edm_score_history else 0.0
            ),
            "mean_edm_right_score": (
                float(np.mean([x[2] for x in edm_score_history])) if edm_score_history else 0.0
            ),
            "mean_actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "traffic_population": len(simulator.env.unwrapped.road.vehicles) - 1,
            "adt_background_lane_changes": int(getattr(
                simulator, "_adt_background_lane_change_count", 0
            )),
            "adt_six_slot_replacements": int(getattr(
                simulator, "_adt_slot_replacement_count", 0
            )),
            "adt_six_slot_replacement_rate_per_slot_observation": (
                float(getattr(simulator, "_adt_slot_replacement_count", 0))
                / max(6 * int(getattr(simulator, "_adt_slot_observation_count", 0)), 1)
            ),
            "adt_slot_observations": int(getattr(
                simulator, "_adt_slot_observation_count", 0
            )),
            "traffic_profile_counts": {
                name: sum(
                    getattr(vehicle, "_arood_adt_profile", "normal") == name
                    for vehicle in simulator.env.unwrapped.road.vehicles
                    if vehicle is not simulator.env.unwrapped.vehicle
                )
                for name, _ in ADT_PROFILE_MIX
            },
            "traffic_spawned_total": int(getattr(simulator, "_traffic_spawn_count", 0)),
            "traffic_removed_total": int(getattr(simulator, "_traffic_removal_count", 0)),
            "adt_front_coverage_spawned": int(getattr(
                simulator, "_adt_front_coverage_spawn_count", 0
            )),
            "adt_front_coverage_failures": int(getattr(
                simulator, "_adt_front_coverage_attempt_failures", 0
            )),
            "adt_lanes_with_front_vehicle": sum(
                _lane_has_front_vehicle(
                    simulator, lane_id,
                    _road_s(simulator, simulator.env.unwrapped.vehicle)
                )
                for lane_id in range(NUM_LANES)
            ),
            "episode_time_seconds": float(time.time() - episode_start_time),
        }
        metrics_history.append(row)
        episode_returns.append(float(episode_return))

        with open(METRICS_PATH, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[H EDM EPISODE {episode + 1:03d}/{NUM_EPISODES}] "
            f"steps={steps} return={episode_return:+.3f} "
            f"collisions={collisions} collision_rate={collision_rate:.0f} "
            f"mean_speed={mean_speed:.2f}m/s "
            f"speed>=90%limit={speed_target_fraction:.1%} "
            f"lane_changes={lane_changes} "
            f"change_rejection={rejection_count}/{change_requests} "
            f"safe_distance={safe_distance:.1f}m "
            f"safe_steps={safe_steps}/{steps} "
            f"traffic={row['traffic_population']} "
            f"bg_lane_changes={row['adt_background_lane_changes']} "
            f"slot_replacements={row['adt_six_slot_replacements']} "
            f"slot_replacement_rate={row['adt_six_slot_replacement_rate_per_slot_observation']:.3%} "
            f"spawned={row['traffic_spawned_total']} "
            f"removed={row['traffic_removed_total']} "
            f"front_refills={row['adt_front_coverage_spawned']} "
            f"front_lanes={row['adt_lanes_with_front_vehicle']}/3 "
            f"front_refill_failures={row['adt_front_coverage_failures']}"
        )

    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"[U POLICY SAVED] {MODEL_PATH}")

    plt.figure()
    plt.plot(episode_returns, alpha=0.4, label="episode return")
    if len(episode_returns) >= 5:
        window = min(5, len(episode_returns))
        moving = np.convolve(
            episode_returns, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            np.arange(window - 1, len(episode_returns)),
            moving, linewidth=2.0, label=f"{window}-episode average"
        )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("U — Pure RL under ADT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    # Save a compact text summary and the complete per-episode histories,
    # following the plot.txt workflow used in V_old.py.
    def _mean_metric(key):
        values = [float(item[key]) for item in metrics_history if key in item]
        return float(np.mean(values)) if values else 0.0

    with open(PLOT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"collision_rate={_mean_metric('collision_rate'):.4f}\n")
        f.write(f"lane_change_rate={_mean_metric('lane_change_rate'):.4f}\n")
        f.write(
            f"safe_travel_distance={_mean_metric('safe_travel_distance_m'):.2f} "
            "m/episode (mean)\n"
        )
        f.write(f"steps={_mean_metric('steps'):.2f} steps/episode (mean)\n")
        f.write(
            f"lane_change_rejection_rate={_mean_metric('lane_change_rejection_rate'):.4f} "
            "(per-request rate)\n"
        )
        f.write(f"mean_speed={_mean_metric('mean_speed_mps'):.2f} m/s\n")
        f.write(
            f"speed_at_least_90pct_limit={_mean_metric('speed_at_least_90pct_limit'):.4f}\n"
        )
        f.write(
            f"mean_acceleration_reward={_mean_metric('mean_acceleration_reward'):.4f}\n"
        )
        f.write(f"mean_lane_change_cost={_mean_metric('mean_lane_change_cost'):.4f}\n")
        f.write(
            f"mean_actor_loss={_mean_metric('mean_actor_loss'):.6f}\n"
        )
        f.write(
            f"mean_critic_loss={_mean_metric('mean_critic_loss'):.6f}\n"
        )
        f.write(
            f"episode_time_seconds={_mean_metric('episode_time_seconds'):.2f} "
            "seconds/episode (mean)\n"
        )
        f.write("reward:\n" + str(episode_returns) + "\n")
        f.write(
            "collision:\n"
            + str([int(item["collisions"]) for item in metrics_history]) + "\n"
        )
        f.write(
            "collision_rate:\n"
            + str([float(item["collision_rate"]) for item in metrics_history]) + "\n"
        )
        f.write(
            "safe_travel_distance:\n"
            + str([float(item["safe_travel_distance_m"]) for item in metrics_history])
            + "\n"
        )
        f.write(
            "safe_step_count:\n"
            + str([int(item["safe_steps"]) for item in metrics_history]) + "\n"
        )
        f.write(
            "lane_change:\n"
            + str([int(item["lane_changes"]) for item in metrics_history]) + "\n"
        )
        f.write(
            "episode_times:\n"
            + str([float(item["episode_time_seconds"]) for item in metrics_history])
            + "\n"
        )
        f.write(
            "lane_change_rate:\n"
            + str([float(item["lane_change_rate"]) for item in metrics_history])
            + "\n"
        )
        f.write(
            "mean_step_reward:\n"
            + str([
                float(item["return"]) / max(int(item["steps"]), 1)
                for item in metrics_history
            ]) + "\n"
        )
        f.write(
            "mean_speed_mps:\n"
            + str([float(item["mean_speed_mps"]) for item in metrics_history])
            + "\n"
        )
        f.write(
            "lane_change_rejection_rate:\n"
            + str([
                float(item["lane_change_rejection_rate"])
                for item in metrics_history
            ]) + "\n"
        )
        f.write(
            "traffic_population:\n"
            + str([int(item["traffic_population"]) for item in metrics_history])
            + "\n"
        )

    print(f"[PLOT TXT SAVED] {PLOT_TXT_PATH}")

    return {"policy": policy, "episode_returns": episode_returns,
            "metrics": metrics_history}


def main():
    print("[H_EDM] EDM-informed pure RL with Aggressive Dynamic Traffic (ADT)")
    print(f"Reference range: ±{REFERENCE_RANGE_M:.0f}m")
    print(f"Recycling boundary: ±{RECYCLING_BOUNDARY_M:.0f}m")
    print(f"Target background population: {TRAFFIC_TARGET_COUNT}")
    print(f"Background speed range: {TRAFFIC_MIN_SPEED_MPS:.1f}-{TRAFFIC_MAX_SPEED_MPS:.1f} m/s")
    print(f"Profile desired-speed ranges: {PROFILE_DESIRED_SPEED_RANGES_MPS}")
    print(f"Profile lane-change rates: {PROFILE_LANE_CHANGE_RATE_PER_S}")
    print(f"ADT profile mix: {dict(ADT_PROFILE_MIX)}")
    print(
        "Traffic design: 15 baseline vehicles plus lane-aware front coverage; "
        "all 3 lanes are maintained with a front vehicle inside ±450m"
    )
    print(
        "ADT behaviors: interactive IDM, aggressive overtaking/lane changes, "
        "unstable acceleration/braking, and slower vehicles"
    )
    print(
        f"Temporal resolution: simulation={SIMULATION_FREQUENCY_HZ}Hz, "
        f"policy={POLICY_FREQUENCY_HZ}Hz"
    )

    # Set the base simulator frequencies and speed limit before construction.
    base.simulation_frequency = SIMULATION_FREQUENCY_HZ
    base.policy_frequency = POLICY_FREQUENCY_HZ
    base.SPEED_LIMIT = SPEED_LIMIT_MPS

    print("Ego observation: fixed six slots (nearest front/rear in each lane)")
    print("Lane decision: RL chooses KEEP/CHANGE; EDM ranks all lanes and CHANGE selects the best physically adjacent lane")
    print(f"EDM weights: safety={EDM_SAFETY_WEIGHT:.2f}, acceleration={EDM_ACCELERATION_WEIGHT:.2f}; TTC critical/warning={EDM_TTC_CRITICAL_S:.1f}/{EDM_TTC_WARNING_S:.1f}s; gap critical/warning={EDM_GAP_CRITICAL_M:.1f}/{EDM_GAP_WARNING_M:.1f}m")
    print(f"EDM prediction horizon={EDM_PREDICTION_HORIZON_S:.2f}s; state features are relative lane utilities (candidate score minus current-lane score)")
    print("Safety intervention: none; only physical lane-existence validity is enforced")
    print("Reward: non-negative per-decision acceleration utility + clipped non-negative speed utility (0 to 1); both suspended during lane-change processing; collision penalty terminates episode")
    print(f"Decision interval={POLICY_DECISION_INTERVAL_S:.6f}s; max delta-v/decision={MAX_ACCEL_DELTA_V_PER_DECISION_MPS:.6f}m/s")
    print(
        f"Lane-change processing={LANE_CHANGE_PROCESSING_STEPS} policy step(s) "
        f"({LANE_CHANGE_PROCESSING_EFFECTIVE_DURATION_S:.6f}s at "
        f"{POLICY_FREQUENCY_HZ}Hz); exactly this many subsequent decisions "
        "suspend ordinary rewards and mask CHANGE"
    )
    print(f"State dimension: base={BASE_STATE_DIM} + EDM={NUM_LANES} -> {STATE_DIM}")
    print(f"EDM-to-RL action diagnostics: {EDM_ACTION_DIAGNOSTIC_PATH}")
    print("Diagnostics: pre-action TTC/EDM, exact observation features, policy probabilities, sampled action, post-action TTC/EDM and collision; logging only")
    print(f"PyTorch={torch.__version__} device={base.DEVICE}")

    simulator = create_edm_highway_simulator()
    simulator.max_steps = MAX_STEPS_PER_EPISODE
    policy = base.create_initial_policy(
        state_dim=STATE_DIM,
        action_dim=POLICY_ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        scale_of_steering=0.5,
    )
    base.check_model_device(policy, "U policy")
    train(simulator, policy)
    simulator.close()


if __name__ == "__main__":
    main()

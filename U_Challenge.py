"""
U_Challenge.py — pure-RL challenge environment with Aggressive Dynamic Traffic (ADT).

The ego policy, six-slot observation, reward, safety filter, and actor-critic
training implementation are retained. ADT changes only background-vehicle
behavior: heterogeneous driving profiles, autonomous lane changes, tailgating,
fast overtaking, and occasional controlled hard braking.
"""

import os
import math
import time
import random
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

import V as base


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

# Legacy lane-potential configuration (inactive in pure-RL mode)
POTENTIAL_HORIZON_S = 3.0
POTENTIAL_DT_S = 0.2
EGO_MAX_ACCEL_MPS2 = 5.0
EGO_COMFORTABLE_BRAKING_MPS2 = 4.0
EGO_MIN_GAP_M = 15.0
EGO_TIME_HEADWAY_S = 1.5
MIN_ACCELERATION_GAIN_MPS2 = 0.25

# Legacy traffic-safety thresholds (inactive for ego action selection)
MIN_LANE_CHANGE_FRONT_GAP_M = 25.0
MIN_LANE_CHANGE_REAR_GAP_M = 25.0
MIN_LANE_CHANGE_FRONT_TTC_S = 6.0
MIN_LANE_CHANGE_REAR_TTC_S = 4.0
EMERGENCY_FRONT_GAP_M = 10.0
EMERGENCY_FRONT_TTC_S = 3.0

# Reward
LANE_CHANGE_COST_WEIGHT = 0.2
COLLISION_PENALTY = 5.0

# Temporal resolution (Hz)
# Keep these equal: the inherited HighwaySimulator applies one instantaneous
# ego lane decision per environment step, so 1 policy decision corresponds to
# 1 simulator update. Background ADT events are scaled by simulation dt.
SIMULATION_FREQUENCY_HZ = 15
POLICY_FREQUENCY_HZ = 15

# Training
NUM_EPISODES = 30
MAX_STEPS_PER_EPISODE = 1500
DISCOUNT = 0.99
POLICY_LR = 3e-4
HIDDEN_DIM = 128

# Keep the six-slot state layout from V_old.py.
STATE_DIM = base.STATE_DIM
ACTION_DIM = base.ACTION_DIM
POLICY_ACTION_DIM = base.POLICY_ACTION_DIM

MODEL_PATH = os.path.join("policy", "U_Challenge_4.pt")
PLOT_PATH = os.path.join("figure", "U_Challenge_4_return.png")
PLOT_TXT_PATH = os.path.join("figure", "plot_U_Challenge_4.txt")
METRICS_PATH = os.path.join("figure", "U_Challenge_4_episode_metrics.jsonl")
base.EPISODE_TRAJECTORY_PATH = os.path.join("figure", "U_Challenge_4_trajectories.jsonl")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# 6. Reward: measured acceleration, executed lane-change cost, collision cost
# ---------------------------------------------------------------------------

def calculate_reward(actual_acceleration, lane_change_executed, collision):
    acceleration_reward = float(np.clip(
        float(actual_acceleration) / EGO_MAX_ACCEL_MPS2, -1.0, 1.0
    ))
    lane_change_cost = (
        -LANE_CHANGE_COST_WEIGHT * abs(acceleration_reward)
        if lane_change_executed else 0.0
    )
    collision_cost = COLLISION_PENALTY if collision else 0.0
    total = acceleration_reward + lane_change_cost - collision_cost
    return total, {
        "acceleration_reward": acceleration_reward,
        "actual_acceleration_mps2": float(actual_acceleration),
        "lane_change_cost": float(lane_change_cost),
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
base.HighwaySimulator._encode_state = _six_slot_state
base.HighwaySimulator.get_valid_candidate_lanes = _valid_candidate_lanes
base.HighwaySimulator.get_valid_lane_action_mask = _valid_lane_action_mask
base.HighwaySimulator.get_valid_target_lane_mask = _valid_target_lane_mask
base.HighwaySimulator.validate_policy_action = _validate_policy_action

# Keep the original state/action dimensions and the policy implementation.
base.POLICY_LR = POLICY_LR
base.HIDDEN_DIM = HIDDEN_DIM
base.MAX_STEPS = MAX_STEPS_PER_EPISODE


# ---------------------------------------------------------------------------
# 8. New concise online actor-critic training loop
# ---------------------------------------------------------------------------

def train(simulator, policy):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    episode_returns = []
    metrics_history = []

    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        simulator._trace_phase = "U_ADT_pure_RL"
        simulator._trace_episode = episode + 1
        state = np.asarray(simulator.reset(seed=SEED + episode), dtype=np.float32)
        base.record_episode_start(
            simulator, "U_ADT_pure_RL", episode + 1, state
        )

        episode_return = 0.0
        collisions = 0
        lane_changes = 0
        speeds = []
        acceleration_rewards = []
        lane_costs = []
        safe_distance = 0.0
        safe_steps = 0
        previous_position = np.asarray(
            simulator.env.unwrapped.vehicle.position, dtype=float
        ).copy()
        actor_losses = []
        critic_losses = []
        rejection_count = 0
        change_requests = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            ego = simulator.env.unwrapped.vehicle
            current_lane = int(ego.lane_index[2])
            valid_action_mask = simulator.get_valid_lane_action_mask()
            target_lane_mask = simulator.get_valid_target_lane_mask()

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

            filtered_action, safety_info = simulator.validate_policy_action(
                sampled_action,
                selected_target_lane=selected_target_lane,
            )
            if safety_info.get("lane_change_blocked", False):
                rejection_count += 1

            vehicles_before = (
                base.snapshot_simulator_vehicles(simulator)
                if base.TRAJECTORY_RECORDING_ENABLED else []
            )
            next_state, _, done, step_info = simulator.step(
                filtered_action,
                apply_safety_filter=False,
                sampled_action=sampled_action,
                safety_info=safety_info,
            )

            actual_acceleration = float(
                step_info.get("observed_acceleration", 0.0)
            )
            lane_change_executed = bool(
                step_info.get("lane_change_completed_instantly", False)
            )
            collision = bool(step_info.get("collision", False))
            reward, reward_components = calculate_reward(
                actual_acceleration,
                lane_change_executed,
                collision,
            )

            if collision:
                collisions += 1
            if lane_change_executed:
                lane_changes += 1
            acceleration_rewards.append(reward_components["acceleration_reward"])
            lane_costs.append(reward_components["lane_change_cost"])

            new_position = np.asarray(
                simulator.env.unwrapped.vehicle.position, dtype=float
            )
            step_distance = float(np.linalg.norm(new_position - previous_position))
            if not collision:
                safe_distance += step_distance
                safe_steps += 1
            previous_position = new_position
            speed = float(simulator.env.unwrapped.vehicle.speed)
            speeds.append(speed)

            base.record_episode_transition(
                simulator=simulator,
                phase="U_ADT_pure_RL",
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
                done=done,
                terminated=bool(step_info.get("terminated", False)),
                truncated=bool(step_info.get("truncated", False)),
            )

            update = base.update_policy(
                policy=policy,
                state=state,
                action=sampled_action,
                reward=reward,
                next_state=next_state,
                done=done,
                valid_action_mask=valid_action_mask,
                terminated=bool(step_info.get("terminated", False)),
                pre_tanh_accel=aux.get("pre_tanh_accel"),
                selected_target_lane=selected_target_lane,
                valid_target_lane_mask=target_lane_mask,
                current_lane_id=current_lane,
            )
            actor_losses.append(update["actor_loss"])
            critic_losses.append(update["critic_loss"])
            episode_return += reward
            state = np.asarray(next_state, dtype=np.float32)

            if done:
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
            "return": float(episode_return),
            "collisions": int(collisions),
            "collision_rate": collision_rate,
            "lane_changes": int(lane_changes),
            "lane_change_rate": float(lane_change_rate),
            "lane_change_requests": int(change_requests),
            "lane_change_rejections": int(rejection_count),
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
            "mean_lane_change_cost": (
                float(np.mean(lane_costs)) if lane_costs else 0.0
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
            f"[U ADT EPISODE {episode + 1:03d}/{NUM_EPISODES}] "
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
    print("[U] Pure RL with Compact Interactive Dynamic Traffic (ADT)")
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

    # V_old.py reads these module-level values when constructing HighwaySimulator.
    # Set them before create_highway_simulator() so the top-level V2 settings
    # are the actual frequencies used by the environment.
    base.simulation_frequency = SIMULATION_FREQUENCY_HZ
    base.policy_frequency = POLICY_FREQUENCY_HZ

    print("Ego observation: fixed six slots (nearest front/rear in each lane)")
    print("Lane selection: pure RL over physically reachable adjacent lanes only; no TTC/gap or acceleration-ranking filter")
    print("Background traffic: ADT enabled; ego action is controlled by RL without a traffic-safety shield")
    print("Reward: normalized measured acceleration - 0.2*|acceleration reward| for executed lane changes - collision penalty")
    print(f"PyTorch={torch.__version__} device={base.DEVICE}")

    simulator = base.create_highway_simulator()
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

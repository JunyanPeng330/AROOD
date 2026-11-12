"""
V2.py — continuous-traffic, acceleration-oriented RL experiment.

This is a new experiment file. It imports V.py only to reuse the tested
HighwayEnv wrapper's instantaneous lane-transition execution, actor-critic
network, trajectory utilities, and policy-update implementation. It replaces
the six-vehicle traffic manager, lane utility, safety selection, reward, and
training loop defined in V.py.

Run:
    python V2.py

The six-slot observation is retained, but the simulator maintains a larger
traffic population. Background traffic uses bounded IDM longitudinal control.
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
SPEED_LIMIT_MPS = 40.0

# Traffic population and spatial management
TRAFFIC_TARGET_COUNT = 20
REFERENCE_RANGE_M = 500.0
RECYCLING_BOUNDARY_M = 700.0
INITIAL_PLACEMENT_RANGE_M = 470.0
MIN_INITIAL_GAP_M = 32.0
MIN_INSERTION_GAP_M = 35.0
TRAFFIC_MIN_SPEED_MPS = 15.0
TRAFFIC_MAX_SPEED_MPS = 42.0

# Background IDM parameters
IDM_MAX_ACCEL_MPS2 = 2.0
IDM_COMFORTABLE_BRAKING_MPS2 = 3.5
IDM_MIN_GAP_M = 5.0
IDM_TIME_HEADWAY_S = 1.5
IDM_ACCEL_EXPONENT = 4.0

# Ego acceleration-potential prediction
POTENTIAL_HORIZON_S = 3.0
POTENTIAL_DT_S = 0.2
EGO_MAX_ACCEL_MPS2 = 5.0
EGO_COMFORTABLE_BRAKING_MPS2 = 4.0
EGO_MIN_GAP_M = 5.0
EGO_TIME_HEADWAY_S = 1.5
MIN_ACCELERATION_GAIN_MPS2 = 0.25

# Independent lane-change safety constraints
MIN_LANE_CHANGE_FRONT_GAP_M = 15.0
MIN_LANE_CHANGE_REAR_GAP_M = 15.0
MIN_LANE_CHANGE_FRONT_TTC_S = 6.0
MIN_LANE_CHANGE_REAR_TTC_S = 4.0
EMERGENCY_FRONT_GAP_M = 8.0
EMERGENCY_FRONT_TTC_S = 1.5

# Reward
LANE_CHANGE_COST_WEIGHT = 0.2
COLLISION_PENALTY = 5.0

# Temporal resolution (Hz)
# Keep these equal in V2: the inherited HighwaySimulator applies one
# instantaneous lane decision per environment step, so 1 policy decision
# corresponds to 1 simulator update. Change both together unless the action
# execution wrapper is redesigned for multiple simulator substeps.
SIMULATION_FREQUENCY_HZ = 15
POLICY_FREQUENCY_HZ = 15

# Training
NUM_EPISODES = 50
MAX_STEPS_PER_EPISODE = 500
DISCOUNT = 0.99
POLICY_LR = 3e-4
HIDDEN_DIM = 128

# Keep the six-slot state layout from V.py.
STATE_DIM = base.STATE_DIM
ACTION_DIM = base.ACTION_DIM
POLICY_ACTION_DIM = base.POLICY_ACTION_DIM

MODEL_PATH = os.path.join("policy", "V2_acceleration_oriented.pt")
PLOT_PATH = os.path.join("figure", "V2_training_return.png")
METRICS_PATH = os.path.join("figure", "V2_episode_metrics.jsonl")
base.EPISODE_TRAJECTORY_PATH = os.path.join("figure", "V2_trajectories.jsonl")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 2. Background traffic vehicle
# ---------------------------------------------------------------------------

class ContinuousTrafficVehicle(base.Vehicle):
    """A background vehicle with bounded IDM longitudinal control."""

    def act(self):
        manager = getattr(self, "_arood_traffic_manager", None)
        if manager is None:
            acceleration = 0.0
        else:
            acceleration = manager.idm_acceleration(self)

        self.action = {
            "steering": 0.0,
            "acceleration": float(np.clip(
                acceleration, -IDM_COMFORTABLE_BRAKING_MPS2,
                IDM_MAX_ACCEL_MPS2
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
    vehicle._arood_desired_speed = float(np.clip(
        random.uniform(32.0, SPEED_LIMIT_MPS),
        TRAFFIC_MIN_SPEED_MPS, TRAFFIC_MAX_SPEED_MPS
    ))
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
    while spawned < TRAFFIC_TARGET_COUNT and attempts < 10000:
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
    print(
        "[CONTINUOUS TRAFFIC INITIALIZED]",
        f"vehicles={spawned}",
        f"active_reference=±{REFERENCE_RANGE_M:.0f}m",
        f"recycling_boundary=±{RECYCLING_BOUNDARY_M:.0f}m",
        "observation_slots=6",
    )


def _spawn_boundary_replacement(self):
    """Insert only beyond the active reference range, with a safe-gap check."""
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    ego_speed = max(0.0, float(ego.speed))

    # A slower vehicle is placed ahead so the ego can approach it; a faster
    # vehicle is placed behind so it can approach the ego. Both enter gradually.
    if random.random() < 0.5:
        side = 1.0
        speed = max(TRAFFIC_MIN_SPEED_MPS, ego_speed - random.uniform(2.0, 6.0))
    else:
        side = -1.0
        speed = min(TRAFFIC_MAX_SPEED_MPS, ego_speed + random.uniform(2.0, 6.0))

    for _ in range(60):
        lane_id = random.randrange(NUM_LANES)
        longitudinal = ego_s + side * random.uniform(
            REFERENCE_RANGE_M + 100.0, RECYCLING_BOUNDARY_M - 10.0
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

    return {
        "population": len(road.vehicles) - 1,
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

    # The original encoder appends context and per-lane safety features.
    return _ORIGINAL_STATE_ENCODER(
        self, np.asarray(rows, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# 5. Lane safety and predictive acceleration-potential evaluator
# ---------------------------------------------------------------------------

def _lane_neighbors(self, lane_id):
    ego = self.env.unwrapped.vehicle
    ego_s = _road_s(self, ego)
    front = None
    rear = None
    front_gap = float("inf")
    rear_gap = float("inf")

    for vehicle in _traffic_in_lane(self, lane_id):
        delta = _road_s(self, vehicle) - ego_s
        bumper_gap = abs(delta) - 0.5 * (
            _vehicle_length(ego) + _vehicle_length(vehicle)
        )
        if delta > 0.0 and bumper_gap < front_gap:
            front, front_gap = vehicle, bumper_gap
        elif delta < 0.0 and bumper_gap < rear_gap:
            rear, rear_gap = vehicle, bumper_gap

    ego_speed = max(0.0, float(ego.speed))
    front_ttc = float("inf")
    rear_ttc = float("inf")
    if front is not None and ego_speed > float(front.speed):
        front_ttc = max(front_gap, 0.0) / max(
            ego_speed - float(front.speed), 1e-6
        )
    if rear is not None and float(rear.speed) > ego_speed:
        rear_ttc = max(rear_gap, 0.0) / max(
            float(rear.speed) - ego_speed, 1e-6
        )

    return {
        "front": front, "rear": rear,
        "front_gap": float(front_gap), "rear_gap": float(rear_gap),
        "front_ttc": float(front_ttc), "rear_ttc": float(rear_ttc),
        "front_speed": float(front.speed) if front is not None else SPEED_LIMIT_MPS,
        "rear_speed": float(rear.speed) if rear is not None else 0.0,
    }


def _lane_is_safe(self, lane_id):
    current_lane = int(self.env.unwrapped.vehicle.lane_index[2])
    if int(lane_id) == current_lane:
        return True, "current_lane"

    values = _lane_neighbors(self, lane_id)
    reasons = []
    if values["front"] is not None:
        if values["front_gap"] < MIN_LANE_CHANGE_FRONT_GAP_M:
            reasons.append("front_gap")
        if values["front_ttc"] < MIN_LANE_CHANGE_FRONT_TTC_S:
            reasons.append("front_ttc")
    if values["rear"] is not None:
        if values["rear_gap"] < MIN_LANE_CHANGE_REAR_GAP_M:
            reasons.append("rear_gap")
        if values["rear_ttc"] < MIN_LANE_CHANGE_REAR_TTC_S:
            reasons.append("rear_ttc")
    return len(reasons) == 0, ",".join(reasons) if reasons else "safe"


def _predict_lane_acceleration(self, lane_id):
    """Roll out ego IDM response for a candidate lane over a short horizon."""
    ego = self.env.unwrapped.vehicle
    ego_speed = max(0.0, float(ego.speed))
    desired_speed = min(SPEED_LIMIT_MPS, float(getattr(ego, "MAX_SPEED", SPEED_LIMIT_MPS)))
    neighbors = _lane_neighbors(self, lane_id)
    front = neighbors["front"]
    front_speed = (
        max(0.0, float(front.speed)) if front is not None else desired_speed
    )
    gap = float(neighbors["front_gap"])
    dt = POTENTIAL_DT_S
    steps = max(1, int(round(POTENTIAL_HORIZON_S / dt)))
    acceleration_samples = []

    for _ in range(steps):
        free_term = (ego_speed / max(desired_speed, 1e-3)) ** IDM_ACCEL_EXPONENT
        if front is None:
            interaction_term = 0.0
        else:
            safe_dynamic_gap = (
                ego_speed * EGO_TIME_HEADWAY_S
                + ego_speed * (ego_speed - front_speed)
                / max(2.0 * math.sqrt(EGO_MAX_ACCEL_MPS2 * EGO_COMFORTABLE_BRAKING_MPS2), 1e-3)
            )
            desired_gap = EGO_MIN_GAP_M + max(0.0, safe_dynamic_gap)
            interaction_term = (desired_gap / max(gap, 0.5)) ** 2

        acceleration = EGO_MAX_ACCEL_MPS2 * (
            1.0 - free_term - interaction_term
        )
        acceleration = float(np.clip(
            acceleration, -EGO_COMFORTABLE_BRAKING_MPS2,
            EGO_MAX_ACCEL_MPS2
        ))
        if ego_speed >= desired_speed and acceleration > 0.0:
            acceleration = 0.0

        acceleration_samples.append(acceleration)
        next_speed = max(0.0, ego_speed + acceleration * dt)
        if front is not None:
            gap = max(0.1, gap + (front_speed - 0.5 * (ego_speed + next_speed)) * dt)
        ego_speed = min(desired_speed, next_speed)

    return float(np.mean(acceleration_samples))


def _rank_safe_lanes(self):
    ego = self.env.unwrapped.vehicle
    current = int(ego.lane_index[2])
    current_potential = _predict_lane_acceleration(self, current)
    candidates = []

    for lane_id in (current - 1, current + 1):
        if lane_id not in range(NUM_LANES):
            continue
        safe, reason = _lane_is_safe(self, lane_id)
        if not safe:
            continue
        potential = _predict_lane_acceleration(self, lane_id)
        gain = potential - current_potential
        candidates.append({
            "target_lane": int(lane_id),
            "current_lane": current,
            "current_acceleration_potential": float(current_potential),
            "candidate_acceleration_potential": float(potential),
            "acceleration_gain": float(gain),
            "safety": reason,
        })

    candidates.sort(key=lambda item: item["candidate_acceleration_potential"], reverse=True)
    beneficial = [
        item for item in candidates
        if item["acceleration_gain"] >= MIN_ACCELERATION_GAIN_MPS2
    ]
    return current_potential, beneficial, candidates


def _valid_lane_action_mask(self):
    _, beneficial, _ = _rank_safe_lanes(self)
    return np.asarray([True, bool(beneficial)], dtype=bool)


def _valid_target_lane_mask(self):
    _, beneficial, _ = _rank_safe_lanes(self)
    mask = np.zeros(NUM_LANES, dtype=bool)
    if beneficial:
        # Exactly one target is eligible: the safest candidate among those
        # that pass hard safety checks and provide the greatest acceleration.
        mask[int(beneficial[0]["target_lane"])] = True
    return mask


def _filter_acceleration_and_lane_action(self, sampled_action, selected_target_lane=None):
    action = np.asarray(sampled_action, dtype=np.float32).reshape(-1).copy()
    if action.size != 2:
        raise ValueError("Expected [normalized acceleration, binary KEEP/CHANGE].")

    ego = self.env.unwrapped.vehicle
    current = int(ego.lane_index[2])
    requested = int(np.clip(np.rint(action[1]), 0, 1))
    current_values = _lane_neighbors(self, current)
    emergency = (
        current_values["front"] is not None
        and (
            current_values["front_gap"] <= EMERGENCY_FRONT_GAP_M
            or current_values["front_ttc"] <= EMERGENCY_FRONT_TTC_S
        )
    )

    current_potential, beneficial, all_safe_candidates = _rank_safe_lanes(self)
    best = beneficial[0] if beneficial else None
    target = current
    executed = requested
    blocked = False
    reason = "none"

    if requested == 1:
        selected = None if selected_target_lane is None else int(selected_target_lane)
        if best is None:
            blocked, executed, reason = True, 0, "no_safe_beneficial_adjacent_lane"
        elif selected != int(best["target_lane"]):
            blocked, executed, reason = True, 0, "target_not_highest_acceleration_candidate"
        else:
            target = int(best["target_lane"])

    if emergency:
        action[0] = min(float(action[0]), -0.8)

    action[1] = float(executed)
    request_id = None
    if requested:
        request_id = int(getattr(self, "_lane_request_trace_sequence", 0)) + 1
        self._lane_request_trace_sequence = request_id

    info = {
        "current_lane_id": current,
        "requested_target_lane_id": int(selected_target_lane if selected_target_lane is not None else current),
        "selected_target_lane_id": int(selected_target_lane if selected_target_lane is not None else current),
        "executed_target_lane_id": target,
        "policy_requested_lane_action": "CHANGE" if requested else "KEEP",
        "executed_lane_action": "CHANGE" if executed else "KEEP",
        "lane_change_blocked": bool(blocked),
        "lane_change_rejection_reason": reason,
        "lane_request_trace_id": request_id,
        "target_selector_mode": "highest_safe_predicted_acceleration_potential",
        "current_lane_acceleration_potential": float(current_potential),
        "ranked_safe_lane_candidates": all_safe_candidates,
        "selected_candidate": best,
        "emergency_braking": bool(emergency),
        "safety_filter_applied": bool(blocked or emergency),
        "overridden": bool(blocked or emergency),
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
base.HighwaySimulator._spawn_traffic_vehicle = _spawn_traffic_vehicle
base.HighwaySimulator._initialize_six_vehicle_traffic = _initialize_continuous_traffic
base.HighwaySimulator._maintain_six_vehicle_traffic = _maintain_continuous_traffic
base.HighwaySimulator._assert_traffic_invariant = _assert_continuous_traffic_invariant
base.HighwaySimulator._sanitize_traffic_slot_speeds = _sanitize_all_traffic_speeds
base.HighwaySimulator._snapshot_managed_traffic_slots = _snapshot_all_traffic
base.HighwaySimulator._record_first_negative_speed_transition = _no_negative_transition_record
base.HighwaySimulator._encode_state = _six_slot_state
base.HighwaySimulator._lane_neighbors = _lane_neighbors
base.HighwaySimulator._lane_is_safe = _lane_is_safe
base.HighwaySimulator._predict_lane_acceleration = _predict_lane_acceleration
base.HighwaySimulator._rank_safe_lanes = _rank_safe_lanes
base.HighwaySimulator.get_valid_lane_action_mask = _valid_lane_action_mask
base.HighwaySimulator.get_valid_target_lane_mask = _valid_target_lane_mask
base.HighwaySimulator.filter_ttc_action = _filter_acceleration_and_lane_action

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
        simulator._trace_phase = "V2_acceleration_oriented"
        simulator._trace_episode = episode + 1
        state = np.asarray(simulator.reset(seed=SEED + episode), dtype=np.float32)
        base.record_episode_start(
            simulator, "V2_acceleration_oriented", episode + 1, state
        )

        episode_return = 0.0
        collisions = 0
        lane_changes = 0
        speeds = []
        acceleration_rewards = []
        lane_costs = []
        safe_distance = 0.0
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

            filtered_action, safety_info = simulator.filter_ttc_action(
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
            safe_distance += float(np.linalg.norm(new_position - previous_position))
            previous_position = new_position
            speed = float(simulator.env.unwrapped.vehicle.speed)
            speeds.append(speed)

            base.record_episode_transition(
                simulator=simulator,
                phase="V2_acceleration_oriented",
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
            "mean_acceleration_reward": (
                float(np.mean(acceleration_rewards)) if acceleration_rewards else 0.0
            ),
            "mean_lane_change_cost": (
                float(np.mean(lane_costs)) if lane_costs else 0.0
            ),
            "mean_actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "traffic_population": len(simulator.env.unwrapped.road.vehicles) - 1,
            "traffic_spawned_total": int(getattr(simulator, "_traffic_spawn_count", 0)),
            "traffic_removed_total": int(getattr(simulator, "_traffic_removal_count", 0)),
        }
        metrics_history.append(row)
        episode_returns.append(float(episode_return))

        with open(METRICS_PATH, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[V2 EPISODE {episode + 1:03d}/{NUM_EPISODES}] "
            f"steps={steps} return={episode_return:+.3f} "
            f"collisions={collisions} collision_rate={collision_rate:.0f} "
            f"mean_speed={mean_speed:.2f}m/s "
            f"speed>=90%limit={speed_target_fraction:.1%} "
            f"lane_changes={lane_changes} "
            f"change_rejection={rejection_count}/{change_requests} "
            f"safe_distance={safe_distance:.1f}m "
            f"traffic={row['traffic_population']} "
            f"spawned={row['traffic_spawned_total']} "
            f"removed={row['traffic_removed_total']}"
        )

    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"[V2 POLICY SAVED] {MODEL_PATH}")

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
    plt.title("V2 — Acceleration-oriented RL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    return {"policy": policy, "episode_returns": episode_returns,
            "metrics": metrics_history}


def main():
    print("[V2] Acceleration-oriented RL with continuous background traffic")
    print(f"Reference range: ±{REFERENCE_RANGE_M:.0f}m")
    print(f"Recycling boundary: ±{RECYCLING_BOUNDARY_M:.0f}m")
    print(f"Target background population: {TRAFFIC_TARGET_COUNT}")
    print(
        f"Temporal resolution: simulation={SIMULATION_FREQUENCY_HZ}Hz, "
        f"policy={POLICY_FREQUENCY_HZ}Hz"
    )

    # V.py reads these module-level values when constructing HighwaySimulator.
    # Set them before create_highway_simulator() so the top-level V2 settings
    # are the actual frequencies used by the environment.
    base.simulation_frequency = SIMULATION_FREQUENCY_HZ
    base.policy_frequency = POLICY_FREQUENCY_HZ

    print("Ego observation: fixed six slots (nearest front/rear in each lane)")
    print("Lane selection: hard safety filter -> predicted acceleration ranking")
    print("Reward: normalized measured acceleration - executed lane-change cost - collision penalty")
    print(f"PyTorch={torch.__version__} device={base.DEVICE}")

    simulator = base.create_highway_simulator()
    simulator.max_steps = MAX_STEPS_PER_EPISODE
    policy = base.create_initial_policy(
        state_dim=STATE_DIM,
        action_dim=POLICY_ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        scale_of_steering=0.5,
    )
    base.check_model_device(policy, "V2 policy")
    train(simulator, policy)
    simulator.close()


if __name__ == "__main__":
    main()

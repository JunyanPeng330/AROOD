# BINARY KEEP/CHANGE POLICY + LEARNED TARGET-LANE RANKING HEAD
# raw_prob contains only [P(KEEP), P(CHANGE)]; target lane is selected by a separate learned head.
# Execution-time lane safety is rechecked; acceleration is not a learned policy dimension.
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

import gymnasium as gym
import highway_env
from highway_env.vehicle.kinematics import Vehicle

# ============================================================
# 1. Configuration — instantaneous lane-change diagnostic mode
# ============================================================
INSTANT_LANE_CHANGE_MODE = True
steering_scale = 0.5
offroad_penalty = 0.5

# The policy acts at 5 Hz; the simulator advances physics at 15 Hz (3 substeps/action).
policy_frequency = 5
simulation_frequency = 5

lane_keeping_weight = 0.20

# Stable long-term reward objectives. TTC is reserved for the independent
# safety shield; it is not a continuous reward term.
# Semantic/embedding reward is disabled. Progress is the primary positive signal.
INITIAL_TRACKING_WEIGHT = 0.0
INITIAL_PROGRESS_WEIGHT = 4.0
INITIAL_HEADWAY_WEIGHT = 0.75
INITIAL_COLLISION_PENALTY = 5.0
INITIAL_OFFROAD_PENALTY = 5.0
INITIAL_LANE_CHANGE_COST = 0.05

# TTC thresholds are used by the safety shield, not reward shaping.
ttc_critical_threshold = 2.0
ttc_danger_threshold = 3.0
ttc_caution_threshold = 4.0

SPEED_MATCH_TTC = 3.0
SPEED_MATCH_GAP_TOLERANCE = 5.0  # metres; avoid needless overrides for tiny gap deviations
# Execution-time safety shield (TTC is measured in seconds).
SAFETY_OVERRIDE_TTC = 2.0       # Emergency braking threshold
# Candidate-lane TTC thresholds. Keep front TTC at least as conservative as rear TTC.
LANE_CHANGE_MIN_FRONT_TTC = 4.0
LANE_CHANGE_MIN_REAR_TTC = 3.0
assert LANE_CHANGE_MIN_FRONT_TTC >= LANE_CHANGE_MIN_REAR_TTC, (
    "Design constraint violated: minimum front TTC must be >= minimum rear TTC."
)
LANE_CHANGE_PREDICTION_HORIZON = 1.5  # Retained for transition/diagnostic predictions (seconds)
LANE_CHANGE_TTC_MARGIN = 0.5     # Extra rear-TTC margin during the transition
LANE_CHANGE_MIN_GAP = 5.0        # Independent minimum bumper-gap safeguard (metres)

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

REFERENCE_GAP_MIN = 10.0
REFERENCE_GAP_MAX = 60.0
RANDOMIZED_FRONT_GAP_MIN = 10.0
RANDOMIZED_FRONT_GAP_MAX = 60.0
RANDOMIZED_REAR_GAP_MIN = 10.0
RANDOMIZED_REAR_GAP_MAX = 40.0
TRAFFIC_SPEED_VARIATION = 8.0

# Lane changes are evaluated through stable task rewards and safety filtering.
# Do not grant an instantaneous bonus based on TTC.
#LANE_CHANGE_COST = 0.0

# Diagnostic mode: a LEFT/RIGHT policy decision sets a persistent lane target;
# the steering controller continues tracking it over subsequent physics steps.
# Disable the post-completion cooldown so the next policy decision is available
# as soon as the previous maneuver physically finishes. This does NOT teleport
# the vehicle or make lane changes instantaneous.
LANE_CHANGE_COOLDOWN_STEPS = 0


# Speed-matching controller
DESIRED_TIME_HEADWAY = 1.2   # seconds
MIN_FOLLOWING_GAP = 5.0      # metres
SPEED_MATCH_KV = 0.5         # 1/second
SPEED_MATCH_KD = 0.08        # 1/second^2

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
ACTION_DIM = 2  # execution envelope [controller_acceleration, binary_action]
POLICY_ACTION_DIM = 2  # KEEP, CHANGE

HIDDEN_DIM = 128


NUM_REFERENCE_EPISODES = 50
REFERENCE_TRAINING_EPOCHS = 5
REFERENCE_BATCH_SIZE = 256
NUM_ONLINE_EPISODES = 100

MAX_STEPS = 200


TTC_PROXY_THRESHOLD = 0.2


POLICY_LR = 3e-4




DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Structured diagnostics: one JSON object per line.

DIAGNOSTIC_LOG_PATH = "training_diagnostics_binary_lane_ranked.jsonl"

# Episode-level evaluator settings. The placeholder is deterministic and does not
# call a network or pretend to be a real LLM.
EVALUATOR_BATCH_SIZE = 10
EVALUATOR_MIN_EPISODES = 10
EVALUATOR_MAX_SAFETY_WEIGHT = 3.0
EVALUATOR_MIN_SAFETY_WEIGHT = 0.5
EVALUATOR_MAX_LANE_CHANGE_COST = 0.05
EVALUATOR_MIN_LANE_CHANGE_COST = 0.0
EVALUATOR_LANE_CHANGE_RATE_LOW = 0.005


reward_path = 'figures/' + 'Re_B_1.png'
reward_per_step_path = 'figures/' + 'Re_B_1.png'

plot_path = 'figures/' + 'plot_B_1.txt'
MODEL_PATH = 'policy/' + 'B_binary_lane_ranked.pt'



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


def write_diagnostic_event(event, simulator=None, **fields):
    """Append one structured diagnostic event; logging failures must not stop RL."""
    record = {
        "event": str(event),
        "phase": getattr(simulator, "_trace_phase", None) if simulator is not None else None,
        "episode": getattr(simulator, "_trace_episode", None) if simulator is not None else None,
        "simulator_step": getattr(simulator, "step_count", None) if simulator is not None else None,
    }
    record.update(fields)
    try:
        with open(DIAGNOSTIC_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False, default=str, allow_nan=True) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        print("[DIAGNOSTIC LOG WARNING]", f"event={event}", f"error={exc}")


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
    """Binary KEEP/CHANGE actor with a learned lane-ranking head."""
    def __init__(self, state_dim, action_dim, hidden_dim, steering_scale=None):
        super().__init__()
        if action_dim != 2: raise ValueError("Expected binary KEEP/CHANGE policy")
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.action_logits = nn.Linear(hidden_dim, 2)
        self.lane_rank_logits = nn.Linear(hidden_dim, NUM_LANES)
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

    def act(self, state, deterministic=False, valid_action_mask=None,
            valid_target_lane_mask=None, return_aux=False):
        dist = self.action_distributions(state, valid_action_mask)
        a = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        target = -1
        if valid_target_lane_mask is not None:
            tm = torch.as_tensor(valid_target_lane_mask, dtype=torch.bool, device=dist.logits.device)
            if tm.ndim == 1: tm = tm.unsqueeze(0)
            if tm.shape[0] == 1 and dist.logits.shape[0] > 1: tm = tm.expand(dist.logits.shape[0], -1)
            if tm.any(dim=-1).all():
                ld = self.lane_distributions(state, tm)
                li = torch.argmax(ld.logits, dim=-1) if deterministic else ld.sample()
                target = int(li[0].detach().cpu().item())
        action = torch.stack([torch.zeros_like(a, dtype=torch.float32), a.float()], dim=-1)
        out = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return (out, {"pre_tanh_accel": 0.0, "selected_target_lane": target}) if return_aux else out

    def __call__(self, state, valid_action_mask=None, valid_target_lane_mask=None):
        return self.act(state, valid_action_mask=valid_action_mask,
                        valid_target_lane_mask=valid_target_lane_mask)

    def value(self, state):
        return self.critic(self.features(state)).squeeze(-1)


def lane_policy_diagnostic(policy, state, valid_action_mask, sampled_action, phase, episode, step,
                           selected_target_lane=-1, valid_target_lane_mask=None):
    with torch.no_grad():
        x = to_tensor(state)
        if x.ndim == 1: x = x.unsqueeze(0)
        h = policy.shared(x)
        logits = policy.action_logits(h)
        raw = torch.softmax(logits, dim=-1)
        mask = torch.as_tensor(valid_action_mask, dtype=torch.bool, device=logits.device)
        if mask.ndim == 1: mask = mask.unsqueeze(0)
        masked = torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1)
        lane_logits = policy.lane_rank_logits(h)
        if valid_target_lane_mask is not None:
            lm = torch.as_tensor(valid_target_lane_mask, dtype=torch.bool, device=lane_logits.device)
            if lm.ndim == 1: lm = lm.unsqueeze(0)
            if lm.shape[0] == 1 and lane_logits.shape[0] > 1: lm = lm.expand(lane_logits.shape[0], -1)
            lane_probs = torch.softmax(lane_logits.masked_fill(~lm, -1e9), dim=-1) if lm.any(dim=-1).all() else torch.zeros_like(lane_logits)
        else: lane_probs = torch.softmax(lane_logits, dim=-1)
    i = int(np.clip(np.rint(float(sampled_action[1])), 0, 1))
    print("[BINARY POLICY]", f"phase={phase}", f"episode={episode}", f"step={step}",
          f"raw_prob_KEEP_CHANGE={np.round(raw.squeeze(0).cpu().numpy(),4).tolist()}",
          f"masked_prob_KEEP_CHANGE={np.round(masked.squeeze(0).cpu().numpy(),4).tolist()}",
          f"sampled_action={('KEEP','CHANGE')[i]}",
          f"valid_target_lane_mask={None if valid_target_lane_mask is None else np.asarray(valid_target_lane_mask,dtype=bool).tolist()}",
          f"target_lane_probs={np.round(lane_probs.squeeze(0).cpu().numpy(),4).tolist()}",
          f"selected_target_lane={selected_target_lane}")

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
        self.traffic_speed_variation = TRAFFIC_SPEED_VARIATION
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

                # Add or change these two settings:
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
              f"speed_variation=±{self.traffic_speed_variation:.1f}m/s",
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
    TRAFFIC_RELEVANT_RANGE = 120.0  # metres ahead/behind; farther vehicles are recycled
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

    def _spawn_slot_vehicle(self, lane_id, side):
        """Spawn a replacement at the configured gap in the requested lane."""
        base_env = self.env.unwrapped
        ego = base_env.vehicle
        lane_index = self._lane_index_for_id(lane_id)
        # Compute s in the target lane's own parameterization for placement;
        # classification and distance comparisons use the shared road frame.
        ego_s_target_lane = self._lane_longitudinal_position(ego, lane_index)
        # Randomized reference traffic gets a distinct independently sampled
        # gap for every lane/side pair. The same rule is used for replacements,
        # so traffic remains varied as vehicles pass or leave the relevant range.
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
        speed_offset = random.uniform(-self.traffic_speed_variation, self.traffic_speed_variation)
        speed = float(np.clip(ego.speed + speed_offset, 8.0, 30.0))
        traffic_vehicle = Vehicle.make_on_lane(base_env.road, lane_index,
                                                longitudinal=longitudinal, speed=speed)
        base_env.road.vehicles.append(traffic_vehicle)
        print("[TRAFFIC SPAWN]", f"lane={lane_id}", f"side={side}",
              f"gap={gap:.1f}m", f"range={self.TRAFFIC_RELEVANT_RANGE:.1f}m",
              f"speed={speed:.2f}m/s", f"id={id(traffic_vehicle)}")
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

        print(
            "[MASK DEBUG][LANE CHANGE TTC PRINCIPLES]",
            f"current_lane={current_lane}",
            f"candidate_lane={candidate_lane}",
            f"ego_speed_mps={ego_speed:.3f}",
            f"candidate_front_speed_mps={front_speed}",
            f"candidate_rear_speed_mps={rear_speed}",
            f"front_gap_m={front_gap:.3f}",
            f"rear_gap_m={rear_gap:.3f}",
            f"front_ttc_s={front_ttc:.3f}",
            f"required_front_ttc_s={LANE_CHANGE_MIN_FRONT_TTC:.3f}",
            f"front_ttc_pass={front_ttc >= LANE_CHANGE_MIN_FRONT_TTC}",
            f"rear_ttc_s={rear_ttc:.3f}",
            f"required_rear_ttc_s={LANE_CHANGE_MIN_REAR_TTC:.3f}",
            f"rear_ttc_pass={rear_ttc >= LANE_CHANGE_MIN_REAR_TTC}",
            f"threshold_ordering_pass={LANE_CHANGE_MIN_FRONT_TTC >= LANE_CHANGE_MIN_REAR_TTC}",
            f"minimum_gap_m={LANE_CHANGE_MIN_GAP:.3f}",
            f"reasons={reasons}",
        )
        return reasons

    def _safe_lane_change_candidates(self, lane_ttc=None):
        """Return safe adjacent-lane candidates with a comparable utility score."""
        base_env = self.env.unwrapped
        vehicle = base_env.vehicle
        if lane_ttc is None:
            lane_ttc = self._min_ttc_by_lane()
        start_node, end_node, current_lane = vehicle.lane_index
        available_lanes = base_env.road.network.graph[start_node][end_node]
        valid_ids = set(range(len(available_lanes)))
        ego_speed = float(vehicle.speed)
        candidates = []
        # Lane IDs are geometry only. The selector checks both adjacent IDs;
        # LEFT/RIGHT are not separate policy actions.
        for candidate_lane in (int(current_lane) - 1, int(current_lane) + 1):
            if candidate_lane not in valid_ids or candidate_lane not in lane_ttc:
                continue
            reasons = self._lane_change_principle_reasons(
                candidate_lane, lane_ttc, current_lane=int(current_lane),
                ego_speed=ego_speed,
            )
            if reasons:
                continue
            values = lane_ttc[candidate_lane]
            front_gap = float(values.get("front_gap", float("inf")))
            rear_gap = float(values.get("rear_gap", float("inf")))
            front_speed = values.get("front_speed")
            # Hard TTC/gap constraints are applied before this utility score.
            front_space_score = min(front_gap, 100.0) if np.isfinite(front_gap) else 100.0
            rear_space_score = min(rear_gap, 100.0) if np.isfinite(rear_gap) else 100.0
            lead_speed_score = float(front_speed) if front_speed is not None else ego_speed
            utility = front_space_score + 0.5 * lead_speed_score + 0.25 * rear_space_score
            candidates.append({
                "target_lane": int(candidate_lane),
                "utility": float(utility),
                "front_gap": front_gap,
                "rear_gap": rear_gap,
                "front_ttc": float(values.get("front_ttc", float("inf"))),
                "rear_ttc": float(values.get("rear_ttc", float("inf"))),
                "front_speed": front_speed,
                "rear_speed": values.get("rear_speed"),
            })
        return candidates

    def _select_best_safe_lane_candidate(self, lane_ttc=None):
        """Choose the highest-utility safe target lane, or None if none is safe."""
        candidates = self._safe_lane_change_candidates(lane_ttc)
        if not candidates:
            return None
        selected = max(candidates, key=lambda item: (item["utility"], -item["target_lane"]))
        print("[TARGET LANE SELECTOR]",
              f"candidates={candidates}",
              f"selected_lane={selected['target_lane']}",
              f"selected_utility={selected['utility']:.3f}")
        return selected

    def get_valid_lane_action_mask(self):
        """Two-action mask: [KEEP, CHANGE]; CHANGE is valid if any safe target exists."""
        vehicle = self.env.unwrapped.vehicle
        if self._lane_change_active or self._lane_change_cooldown > 0:
            mask = np.array([True, False], dtype=bool)
            print("[MASK DEBUG][FORCED KEEP]",
                  f"current_lane={int(vehicle.lane_index[2])}",
                  f"lane_change_active={self._lane_change_active}",
                  f"cooldown={self._lane_change_cooldown}",
                  f"mask={mask.tolist()}",
                  "reason=active_maneuver_or_cooldown")
            return mask
        lane_ttc = self._min_ttc_by_lane()
        candidates = self._safe_lane_change_candidates(lane_ttc)
        mask = np.array([True, bool(candidates)], dtype=bool)
        print("[LANE ACTION MASK]",
              f"current_lane={int(vehicle.lane_index[2])}",
              f"mask={mask.tolist()}",
              f"safe_target_lanes={[item['target_lane'] for item in candidates]}",
              f"principles=front_ttc>={LANE_CHANGE_MIN_FRONT_TTC:.1f}s_and_rear_ttc>={LANE_CHANGE_MIN_REAR_TTC:.1f}s",
              "action_order=[KEEP, CHANGE]")
        return mask

    def _min_ttc_by_lane(self):
        """
        Calculate minimum front TTC and rear TTC for every lane.
        Returns:
            {lane_id: {
                    "front_ttc": float,
                    "rear_ttc": float,
                    "min_ttc": float,}}

        TTC is measured in seconds.
        Assumptions:
            - The road is a straight, multi-lane highway.
            - Vehicles are assigned to lanes using lane_index[2].
            - Longitudinal positions are comparable across lanes.
            - Vehicle lengths are used to estimate bumper clearance.
        A TTC of 0.0 indicates an overlapping or non-positive
        longitudinal bumper gap in the relevant direction.
        """


        base_env = self.env.unwrapped
        road = base_env.road
        ego = base_env.vehicle

        ego_lane = int(ego.lane_index[2])
        ego_speed = float(ego.speed)
        ego_s_global = float(ego.position[0])

        lanes_count = int(base_env.config["lanes_count"])

        # Initialize all lanes, including lanes with no nearby vehicles.
        lane_ttc = {
            lane_id: {
                "front_ttc": float("inf"),
                "rear_ttc": float("inf"),
                "min_ttc": float("inf"),

                # Data for the nearest front vehicle in this lane
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

        # Vehicle length is usually 5 m in highway-env.
        # Read the actual attribute when available.
        ego_length = float(getattr(ego, "LENGTH", getattr(ego, "length", 5.0),))

        for other in list(road.vehicles):

            # Ignore the ego vehicle itself and crashed traffic.
            if other is ego:
                continue
            if bool(getattr(other, "crashed", False)):
                continue

            # Safely obtain the other vehicle's lane.
            try:
                other_lane_index = other.lane_index
                other_lane = int(other_lane_index[2])
            except (AttributeError, TypeError, ValueError, IndexError):
                continue

            if other_lane not in lane_ttc:
                continue



            other_speed = float(other.speed)
            other_s_global = float(other.position[0])

            other_length = float(getattr(other, "LENGTH", getattr(other, "length", 5.0),))

            # Center-to-center longitudinal distance.
            center_gap = other_s_global - ego_s_global

            # Estimated bumper-to-bumper clearance.
            half_length_sum = 0.5 * (ego_length + other_length)

            closing_speed = None
            bumper_gap = None
            ttc = float("inf")
            vehicle_direction = "same_position"

            # -----------------------------------------------------
            # A. Vehicle ahead of the ego vehicle.
            # -----------------------------------------------------
            # A. Vehicle ahead of the ego vehicle.
            if center_gap > 0.0:
                vehicle_direction = "front"
                bumper_gap = center_gap - half_length_sum
                closing_speed = ego_speed - other_speed

                if bumper_gap <= 0.0:
                    ttc = 0.0
                elif closing_speed > 0.1:
                    ttc = bumper_gap / closing_speed

                # Preserve the minimum TTC among all front vehicles.
                lane_ttc[other_lane]["front_ttc"] = min(lane_ttc[other_lane]["front_ttc"], ttc,)

                # Select the nearest front vehicle by center distance, even
                # when its bumper gap is zero/negative. Otherwise an overlapping
                # same-lane vehicle is accidentally reported as "no front vehicle".
                if center_gap < lane_ttc[other_lane]["front_center_gap"]:
                    lane_ttc[other_lane]["front_center_gap"] = center_gap
                    lane_ttc[other_lane]["front_gap"] = bumper_gap
                    lane_ttc[other_lane]["front_speed"] = other_speed
                    lane_ttc[other_lane]["nearest_front_ttc"] = ttc
                    lane_ttc[other_lane]["front_vehicle_id"] = id(other)


            elif center_gap < 0.0:
                vehicle_direction = "rear"
                bumper_gap = -center_gap - half_length_sum
                closing_speed = other_speed - ego_speed

                if bumper_gap <= 0.0:
                    # Longitudinal projections overlap. TTC=0 is a
                    # conservative lane-safety signal, not proof of a
                    # physical collision.
                    ttc = 0.0
                elif closing_speed > 0.1:
                    ttc = bumper_gap / closing_speed

                lane_ttc[other_lane]["rear_ttc"] = min(
                    lane_ttc[other_lane]["rear_ttc"], ttc
                )
                if abs(center_gap) < lane_ttc[other_lane]["rear_center_gap"]:
                    lane_ttc[other_lane]["rear_center_gap"] = abs(center_gap)
                    lane_ttc[other_lane]["rear_gap"] = bumper_gap
                    lane_ttc[other_lane]["rear_speed"] = other_speed
                    lane_ttc[other_lane]["rear_vehicle_id"] = id(other)

            else:
                # Same longitudinal center: direction is ambiguous, but
                # this lane is unsafe to enter due to longitudinal overlap.
                vehicle_direction = "same_position"
                bumper_gap = -half_length_sum
                closing_speed = None
                ttc = 0.0
                lane_ttc[other_lane]["front_ttc"] = 0.0
                lane_ttc[other_lane]["rear_ttc"] = 0.0

            longitudinal_overlap = (
                bumper_gap is not None and bumper_gap <= 0.0
            )
            same_lane_as_ego = (other_lane == ego_lane)





        # ---------------------------------------------------------
        # C. Calculate the minimum of front and rear TTC per lane.
        # ---------------------------------------------------------
        for lane_id, values in lane_ttc.items():
            values["min_ttc"] = min(values["front_ttc"], values["rear_ttc"])

        return lane_ttc






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
                          f"utility={selected_candidate['utility']:.3f}")

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
            f"physical_accel_before={accel_before_speed_floor * MAX_PHYSICAL_ACCELERATION:.3f}m/s^2",
            f"physical_accel_after={accel_after_speed_floor * MAX_PHYSICAL_ACCELERATION:.3f}m/s^2",
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



        print(
            "[TTC FILTER]",
            f"current_lane={current_lane}",
            f"front_ttc={current_front_ttc:.3f}",
            f"rear_ttc={current_rear_ttc:.3f}",
            f"emergency={emergency_braking}",
        )

        if policy_requested_lane_action == 0:
            # KEEP does not evaluate a candidate lane; report measured current
            # lane values instead of misleading default inf candidate values.
            current_lane_values = lane_ttc_current.get(current_lane, {})
            print(
                "[LANE SAFETY]",
                "requested=KEEP",
                f"current_lane={current_lane}",
                f"current_front_ttc={float(current_lane_values.get('front_ttc', float('inf'))):.3f}",
                f"current_rear_ttc={float(current_lane_values.get('rear_ttc', float('inf'))):.3f}",
                f"current_front_gap={float(current_lane_values.get('front_gap', float('inf'))):.3f}",
                f"current_rear_gap={float(current_lane_values.get('rear_gap', float('inf'))):.3f}",
                "lane_change_evaluation=not_applicable",
                f"lane_change_active={self._lane_change_active}",
                f"cooldown={self._lane_change_cooldown}",
                "blocked=False",
                "reason=none",
                "filtered=KEEP",
            )
        else:
            print(
                "[LANE SAFETY]",
                f"requested={requested_lane_name}",
                f"candidate_lane={candidate_lane}",
                f"front_ttc={candidate_front_ttc:.3f}",
                f"rear_ttc={candidate_rear_ttc:.3f}",
                f"front_gap={candidate_front_gap:.3f}",
                f"rear_gap={candidate_rear_gap:.3f}",
                f"front_gap_limit={candidate_front_gap_limit:.3f}",
                f"rear_gap_limit={candidate_rear_gap_limit:.3f}",
                f"lane_available={candidate_lane_available}",
                f"speed_match_active_but_not_a_veto={speed_match_active}",
                f"lane_change_active={self._lane_change_active}",
                f"cooldown={self._lane_change_cooldown}",
                f"blocked={lane_change_blocked}",
                f"reason={lane_change_rejection_reason}",
                f"filtered={filtered_lane_name}",
            )


        # Correlate final output with this exact filter invocation.
        safety_info["ttc_filter_call_id"] = ttc_filter_call_id
        safety_info["safety_filter_applied"] = True
        print(
            "[FILTER EXIT]",
            f"call_id={ttc_filter_call_id}",
            f"sampled={sampled_action.tolist()}",
            f"filtered={filtered_action.tolist()}",
        )




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




    def step(self, action, apply_safety_filter=False, sampled_action=None, safety_info=None):
        """Execute the proposed hybrid action with instantaneous lane changes.

        The policy chooses KEEP or CHANGE. For CHANGE, the separate target-lane
        selector supplies an adjacent lane. No per-step TTC shield overrides the
        policy action; the simulator applies its own vehicle dynamics.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Expected hybrid action [acceleration, lane_action], got {action.shape}")
        incoming_action = action.copy()

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
                    # Preserve longitudinal position and snap to the selected
                    # safe target lane centre (instantaneous diagnostic mode).
                    ego.position = np.asarray(target_lane.position(longitudinal, 0.0), dtype=float)
                    ego.heading = float(target_lane.heading_at(longitudinal))
                    ego.lane_index = target_lane_index
                    self.target_lane_index = target_lane_index
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
        policy_dt = 1.0 / float(base_env.config["policy_frequency"])
        physical_action = np.asarray([acceleration, 0.0], dtype=np.float32)
        obs, env_reward, terminated, truncated, env_info = self.env.step(physical_action)
        ego_after = base_env.vehicle
        speed_after = float(ego_after.speed)
        collision = bool(getattr(ego_after, "crashed", False))

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
            "safety_violation": collision,
            "collision": collision,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "env_reward": float(env_reward),
            "lane_action_id": lane_action_id,
            "lane_action_index": lane_action_id,
            "lane_action_delta": (int(requested_target_lane_id - source_lane_id) if lane_changed else 0),
            "lane_action_name": lane_name,
            "current_lane_id": source_lane_id,
            "requested_target_lane_id": requested_target_lane_id,
            "accepted_target_lane_id": int(ego_after.lane_index[2]),
            "lane_change_completed_instantly": bool(lane_changed),
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
        })
        self.state = next_state
        print("[INSTANT STEP]",
              f"source_lane={source_lane_id}",
              f"requested={lane_name}",
              f"target_lane={requested_target_lane_id}",
              f"lane_changed={lane_changed}",
              f"executed_lane={int(ego_after.lane_index[2])}",
              f"accel={acceleration:.3f}",
              f"collision={collision}")
        return next_state.copy(), float(env_reward), bool(terminated or truncated), info

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
# Non-intervening lane selector and prompt-ready episode evaluator
# ============================================================

def _safety_aware_lane_action_mask(self):
    """Binary action mask [KEEP, CHANGE]."""
    if getattr(self, "_lane_change_active", False) or getattr(self, "_lane_change_cooldown", 0) > 0:
        self._last_safe_lane_candidates = []
        return np.asarray([True, False], dtype=bool)
    candidates = self._safe_lane_change_candidates(self._min_ttc_by_lane())
    self._last_safe_lane_candidates = candidates
    mask = np.asarray([True, bool(candidates)], dtype=bool)
    print("[BINARY SAFETY MASK]", f"mask={mask.tolist()}",
          f"valid_target_lanes={[int(c['target_lane']) for c in candidates]}")
    return mask


def _get_valid_target_lane_mask(self):
    mask = np.zeros(NUM_LANES, dtype=bool)
    if getattr(self, "_lane_change_active", False) or getattr(self, "_lane_change_cooldown", 0) > 0:
        self._last_safe_lane_candidates = []
        return mask
    candidates = self._safe_lane_change_candidates(self._min_ttc_by_lane())
    self._last_safe_lane_candidates = candidates
    for c in candidates:
        lane = int(c["target_lane"])
        if 0 <= lane < NUM_LANES: mask[lane] = True
    return mask


def _safety_aware_lane_selector_and_controller(self, sampled_action, selected_target_lane=None):
    """Recheck the learned target lane immediately before execution."""
    action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
    if action.size != 2: raise ValueError("Expected [acceleration placeholder, binary action]")
    requested = int(np.clip(np.rint(float(action[1])), 0, 1))
    base, ego = self.env.unwrapped, self.env.unwrapped.vehicle
    current = int(ego.lane_index[2])
    metrics = self._min_ttc_by_lane()
    safe = {int(c["target_lane"]) for c in self._safe_lane_change_candidates(metrics)}
    selected = None if selected_target_lane is None else int(selected_target_lane)
    target, executed, blocked, reason = current, requested, False, "none"
    if requested == 1:
        if selected is None or selected < 0 or selected >= min(int(base.config["lanes_count"]), NUM_LANES):
            blocked, reason, executed = True, "no_valid_target_lane_selected", 0
        elif selected not in safe:
            blocked, reason, executed = True, "target_lane_failed_execution_time_safety_recheck", 0
        else: target = selected
    vals = metrics.get(target, {})
    front_speed, front_gap = vals.get("front_speed"), float(vals.get("front_gap", 100.0))
    accel = (float(np.clip(0.5 * (30.0 - float(ego.speed)) / MAX_PHYSICAL_ACCELERATION, -1.0, 1.0)
             ) if front_speed is None else self._speed_matching_acceleration(float(front_speed), front_gap))
    cur = metrics.get(current, {})
    emergency = (float(cur.get("front_ttc", float("inf"))) <= SAFETY_OVERRIDE_TTC
                 or float(cur.get("front_gap", float("inf"))) <= MIN_FOLLOWING_GAP)
    if emergency: accel = min(accel, -0.5)
    req_id = None
    if requested == 1:
        req_id = int(getattr(self, "_lane_request_trace_sequence", 0)) + 1
        self._lane_request_trace_sequence = req_id
    info = {
        "current_lane_id": current, "requested_target_lane_id": int(selected if selected is not None else current),
        "selected_target_lane_id": selected, "executed_target_lane_id": int(target),
        "policy_requested_lane_action": ("KEEP", "CHANGE")[requested],
        "executed_lane_action": ("KEEP", "CHANGE")[executed],
        "policy_requested_binary_action": ("KEEP", "CHANGE")[requested],
        "executed_binary_action": ("KEEP", "CHANGE")[executed],
        "lane_change_blocked": bool(blocked), "lane_change_rejection_reason": reason,
        "lane_request_trace_id": req_id,
        "lane_request_direction": ("LEFT" if selected is not None and selected < current else "RIGHT" if selected is not None and selected > current else "NONE"),
        "lane_request_source_lane": current, "lane_request_target_lane": int(selected if selected is not None else current),
        "lane_request_decision": "TARGET_SELECTED" if requested == 1 and not blocked else ("REJECTED" if blocked else "KEEP"),
        "target_selector_mode": "learned_lane_ranking_head_plus_execution_recheck",
        "safety_filter_applied": bool(blocked or emergency), "overridden": bool(blocked or emergency),
        "emergency_braking": bool(emergency),
        "intervention_reason": "current_lane_emergency" if emergency else ("unsafe_target_rejected" if blocked else "none"),
        "sampled_action": action.copy(), "filtered_action": np.asarray([accel, executed], dtype=np.float32),
        "controller_acceleration": float(accel),
        "lane_change_candidate_front_gap": float(vals.get("front_gap", float("inf"))),
        "lane_change_candidate_rear_gap": float(vals.get("rear_gap", float("inf"))),
        "lane_change_candidate_front_ttc": float(vals.get("front_ttc", float("inf"))),
        "lane_change_candidate_rear_ttc": float(vals.get("rear_ttc", float("inf"))),
    }
    print("[LEARNED TARGET CHECK]", f"requested={('KEEP','CHANGE')[requested]}",
          f"selected={selected}", f"safe={sorted(safe)}", f"executed={('KEEP','CHANGE')[executed]}",
          f"blocked={blocked}", f"reason={reason}")
    return np.asarray([accel, executed], dtype=np.float32), info


HighwaySimulator.get_valid_lane_action_mask = _safety_aware_lane_action_mask
HighwaySimulator.get_valid_target_lane_mask = _get_valid_target_lane_mask
HighwaySimulator.filter_ttc_action = _safety_aware_lane_selector_and_controller




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
            "Make useful forward progress when traffic permits.",
            "Maintain reasonable distance from the vehicle ahead.",
            "Change lanes when a meaningful benefit justifies the associated risk.",
            "Avoid unnecessary lane changes and excessive braking.",
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
        current_lane_cost = float(reward_specification.get("lane_change_cost", INITIAL_LANE_CHANGE_COST))

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
        elif (
            int(metrics.get("episode_count", 0)) >= EVALUATOR_MIN_EPISODES
            and collision_rate == 0.0
            and lane_change_rate < EVALUATOR_LANE_CHANGE_RATE_LOW
            and safe_steps > 0.0
            and current_lane_cost > EVALUATOR_MIN_LANE_CHANGE_COST
        ):
            new_value = max(EVALUATOR_MIN_LANE_CHANGE_COST, current_lane_cost - 0.002)
            if new_value < current_lane_cost:
                updates["lane_change_cost"] = round(float(new_value), 6)
                diagnosis = (
                    f"No collisions were observed, while completed lane changes per step "
                    f"were low ({lane_change_rate:.4f}). This is a signal to inspect possible "
                    "over-penalization, not proof that more lane changes are always desirable."
                )
                recommendation = "Reduce the lane-change cost slightly and test whether useful changes increase."
                expected_tradeoff = "May encourage beneficial lane changes, but could also increase unnecessary changes."

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
    direction = "LEFT" if int(target) < int(current) else "RIGHT"
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
    """Minimal reward: progress minus collision/safety penalty minus accepted lane-change cost.
    TTC is enforced by the independent shield, not by reward shaping.
    """
    def reward_function(
        behavior_description, safety_violation, action, lane_deviation, offroad,
        min_ttc, lane_change_allowed=True, ego_speed=0.0, safety_info=None,
        policy_sampled_action=None, reward_diagnostics=None,
    ):
        safety_info = dict(safety_info or {})
        sampled_action = np.asarray(
            action if policy_sampled_action is None else policy_sampled_action,
            dtype=np.float32,
        ).reshape(-1)

        # 1. Forward progress, capped at the target speed.
        target_speed = max(1e-6, float(reward_specification.get("target_speed", 25.0)))
        progress_weight = max(0.0, float(reward_specification.get("progress_weight", INITIAL_PROGRESS_WEIGHT)))
        ego_speed = max(0.0, float(ego_speed))
        progress_ratio = float(np.clip(ego_speed / target_speed, 0.0, 1.0))
        progress_reward = progress_weight * progress_ratio

        # 2. Collision/safety penalty only. Routine shield interventions do not
        # receive a reward penalty.
        safety_penalty = (
            max(0.0, float(reward_specification.get("safety_weight", INITIAL_COLLISION_PENALTY)))
            if bool(safety_violation) else 0.0
        )

        # 3. Charge a small cost only when a lane change was accepted/executed.
        lane_change_cost_weight = max(
            0.0, float(reward_specification.get("lane_change_cost", INITIAL_LANE_CHANGE_COST))
        )
        lane_change_accepted = bool(
            safety_info.get("lane_change_accepted",
                            safety_info.get("lane_change_completed_instantly", False))
        )
        sampled_lane_action = (
            int(np.clip(np.rint(sampled_action[1]), 0, 2))
            if sampled_action.size > 1 else 0
        )
        lane_change_cost = (
            lane_change_cost_weight
            if lane_change_accepted and sampled_lane_action in (1, 2) else 0.0
        )

        reward = float(progress_reward - safety_penalty - lane_change_cost)
        if reward_diagnostics is not None:
            reward_diagnostics.update({
                "progress_reward": float(progress_reward),
                "safety_penalty": float(safety_penalty),
                "lane_change_cost": float(lane_change_cost),
                "lane_change_accepted": lane_change_accepted,
                "total_reward": reward,
            })
        print(
            f"[REWARD DEBUG] progress={progress_reward:.3f}, "
            f"speed={ego_speed:.2f}m/s, target_speed={target_speed:.2f}m/s, "
            f"safety_penalty={safety_penalty:.3f}, "
            f"lane_change_cost={lane_change_cost:.3f}, "
            f"lane_change_accepted={lane_change_accepted}, total={reward:.3f}"
        )
        print("[TTC REWARD] disabled; TTC is recorded for episode-level evaluation, not used as a per-step action override.")
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
        safety_info=safety_info,
        policy_sampled_action=policy_sampled_action,
        reward_diagnostics=reward_diagnostics,
    )



# ============================================================
# 21. RL Policy Update
# ============================================================
def update_policy(policy, state, action, reward, next_state, done,
                  valid_action_mask=None, terminated=None, pre_tanh_accel=None,
                  selected_target_lane=-1, valid_target_lane_mask=None):
    """Update the binary policy and conditional target-lane ranking head."""
    st, nst = to_tensor(state).reshape(1, -1), to_tensor(next_state).reshape(1, -1)
    act, rew = to_tensor(action).reshape(1, -1), to_tensor([reward])
    value = policy.value(st)
    with torch.no_grad():
        nv = policy.value(nst)
        terminal = bool(done) if terminated is None else bool(terminated)
        target = rew if terminal else rew + 0.99 * nv
    adv = target - value
    critic_loss = F.mse_loss(value, target)
    binary = act[:, 1].round().long().clamp(0, 1)
    action_log_prob = policy.action_distributions(st, valid_action_mask).log_prob(binary)
    target_log_prob = torch.zeros_like(action_log_prob)
    if int(binary.item()) == 1 and int(selected_target_lane) >= 0 and valid_target_lane_mask is not None:
        lane_dist = policy.lane_distributions(
            st, torch.as_tensor(valid_target_lane_mask, dtype=torch.bool, device=DEVICE).reshape(1, -1)
        )
        target_log_prob = lane_dist.log_prob(torch.as_tensor([int(selected_target_lane)], dtype=torch.long, device=DEVICE))
    actor_loss = -((action_log_prob + target_log_prob) * adv.detach()).mean()
    loss = actor_loss + critic_loss
    policy.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    policy.optimizer.step()
    return {"actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "advantage": float(adv.detach().mean().item()),
            "binary_log_prob": float(action_log_prob.detach().mean().item()),
            "target_lane_log_prob": float(target_log_prob.detach().mean().item())}





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
        "name": "pure_rl_episode_adaptive_reward",
        "target_behavior": (
            "Make useful forward progress, avoid collisions, and change lanes "
            "when beneficial without unnecessary lane changes or excessive braking."
        ),
        "safety_weight": INITIAL_COLLISION_PENALTY,
        "progress_weight": INITIAL_PROGRESS_WEIGHT,
        "target_speed": 40.0,
        "lane_change_cost": INITIAL_LANE_CHANGE_COST,
    }
    reward_function = construct_reward(reward_specification)
    # The closure and evaluator share this mutable specification. Only the
    # episode-level adaptation module may update it, between episodes.
    reward_function.reward_specification = reward_specification
    return reward_function


def _hybrid_action_log_probability(policy, states, actions, pre_tanh_accels, masks,
                                  selected_target_lanes=None, target_lane_masks=None):
    """Joint likelihood of binary KEEP/CHANGE and conditional target lane."""
    binary = actions[:, 1].round().long().clamp(0, 1)
    log_prob = policy.action_distributions(states, valid_action_mask=masks).log_prob(binary)
    if selected_target_lanes is not None and target_lane_masks is not None:
        lanes = torch.as_tensor(selected_target_lanes, dtype=torch.long, device=DEVICE).reshape(-1)
        lane_masks = torch.as_tensor(target_lane_masks, dtype=torch.bool, device=DEVICE)
        lane_log_prob = policy.lane_distributions(states, lane_masks).log_prob(lanes.clamp(0, NUM_LANES - 1))
        active = (binary == 1) & (lanes >= 0) & lane_masks.any(dim=-1)
        log_prob = log_prob + torch.where(active, lane_log_prob, torch.zeros_like(lane_log_prob))
    return log_prob




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
        trajectory = {
            "states": [], "actions": [], "pre_tanh_accels": [], "masks": [],
            "selected_target_lanes": [], "target_lane_masks": [],
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
            sampled_action, policy_aux = policy.act(
                state, deterministic=False, valid_action_mask=valid_mask,
                valid_target_lane_mask=target_lane_mask, return_aux=True,
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
            trajectory["rewards"].append(float(reward))
            trajectory["next_states"].append(next_state.copy())
            is_terminated = (
                bool(step_info.get("terminated", False) or offroad)
                if isinstance(step_info, dict) else bool(done or offroad)
            )
            trajectory["terminated"].append(is_terminated)
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
            "selected_target_lanes", "target_lane_masks",
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
            batch_returns = returns[idx]
            batch_weights = weights[idx]

            log_prob = _hybrid_action_log_probability(
                policy, batch_states, batch_actions, batch_pre_tanh, batch_masks,
                batch_target_lanes, batch_target_masks
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
    # collision_rate: 1 if any collision occurs in the episode, otherwise 0.
    # lane_change_rate: completed lane changes divided by episode steps.
    # safe_travel_dis: collision-free distance accumulated before the first collision.
    # safe_step_count: 1-based collision step, or episode length if no collision occurs.
    collision_rate_history = []
    lane_change_rate_history = []
    safe_travel_dis_history = []
    safe_step_count_history = []
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
    try:
        with open(DIAGNOSTIC_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")
    except OSError as exc:
        print("[DIAGNOSTIC LOG WARNING]", exc)

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
        episode_return = 0.0
        episode_steps = 0
        collisions = 0
        first_collision_step = None
        safe_travel_dis = 0.0
        previous_ego_position = np.asarray(simulator.env.unwrapped.vehicle.position, dtype=np.float64).copy()
        episode_actor_losses = []
        episode_critic_losses = []
        lane_counters = make_lane_request_counters()
        seen_completed = {
            "LEFT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_LEFT", 0)),
            "RIGHT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_RIGHT", 0)),
        }

        for t in range(MAX_STEPS):
            valid_mask = simulator.get_valid_lane_action_mask()
            simulator._trace_valid_action_mask = np.asarray(valid_mask, dtype=bool).tolist()

            target_lane_mask = np.asarray(simulator.get_valid_target_lane_mask(), dtype=bool)
            sampled_action, policy_aux = policy.act(
                state, deterministic=False, valid_action_mask=valid_mask,
                valid_target_lane_mask=target_lane_mask, return_aux=True,
            )
            sampled_action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
            pre_tanh_accel = float(policy_aux.get("pre_tanh_accel", 0.0))
            selected_target_lane = int(policy_aux.get("selected_target_lane", -1))
            lane_policy_diagnostic(
                policy, state, valid_mask, sampled_action, phase="pure_rl",
                episode=episode + 1, step=t + 1, selected_target_lane=selected_target_lane,
                valid_target_lane_mask=target_lane_mask,
            )
            filtered_action, safety_info = simulator.filter_ttc_action(
                sampled_action, selected_target_lane=selected_target_lane
            )
            filtered_action = np.asarray(filtered_action, dtype=np.float32).reshape(-1)
            record_lane_request_decision(lane_counters, sampled_action, valid_mask, safety_info)

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

            min_ttc, closing_vehicle_count = calculate_min_ttc(simulator)
            collision_occurred = bool(
                step_info.get("collision", step_info.get("safety_violation", False))
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

            reward = compute_reward(
                reward_function=reward_function,
                behavior_description="",
                safety_violation=safety_violation,
                action=executed_action,
                lane_deviation=lane_deviation,
                offroad=offroad,
                min_ttc=min_ttc,
                lane_change_allowed=True,
                ego_speed=float(simulator.env.unwrapped.vehicle.speed),
                safety_info=reward_safety_info,
                policy_sampled_action=sampled_action,
                reward_diagnostics=reward_components,
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
        reward_history.append(episode_return)
        mean_step_reward_history.append(mean_step_reward)
        episode_times.append(elapsed)
        collision_history.append(collisions)
        lane_stats = dict(getattr(simulator, "_lane_change_stats", {}))
        lane_change_history.append(lane_stats)

        # A collision-free episode contributes 0 to collision_rate. For an episode
        # without collision, safe_step_count is the full number of completed steps.
        collision_rate = float(first_collision_step is not None)
        lane_change_rate = float(lane_stats.get("completed", 0)) / max(episode_steps, 1)
        safe_step_count = int(
            first_collision_step if first_collision_step is not None else episode_steps
        )
        collision_rate_history.append(collision_rate)
        lane_change_rate_history.append(lane_change_rate)
        safe_travel_dis_history.append(float(safe_travel_dis))
        safe_step_count_history.append(safe_step_count)

        actor_loss_history.append(float(np.mean(episode_actor_losses)) if episode_actor_losses else 0.0)
        critic_loss_history.append(float(np.mean(episode_critic_losses)) if episode_critic_losses else 0.0)

        print_lane_request_counters("PURE RL EPISODE", lane_counters)
        print(
            f"Episode {episode + 1:03d}/{NUM_ONLINE_EPISODES} | "
            f"steps={episode_steps} | return={episode_return:.3f} | "
            f"mean_step_reward={mean_step_reward:+.4f} | collisions={collisions} | "
            f"collision_rate={collision_rate:.0f} | "
            f"lane_change_rate={lane_change_rate:.4f} | "
            f"safe_travel_dis={safe_travel_dis:.2f}m | "
            f"safe_step_count={safe_step_count} | "
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
            "steps": int(episode_steps),
            "collisions": int(collisions),
            "collision_rate": float(collision_rate),
            "lane_change_rate": float(lane_change_rate),
            "safe_travel_dis": float(safe_travel_dis),
            "safe_step_count": int(safe_step_count),
        }
        episode_metric_history.append(episode_metrics)

        write_diagnostic_event(
            "episode_summary", simulator,
            episode_return=float(episode_return),
            mean_reward_per_step=float(mean_step_reward),
            steps=int(episode_steps),
            collisions=int(collisions),
            collision_rate=float(collision_rate),
            lane_change_rate=float(lane_change_rate),
            safe_travel_dis=float(safe_travel_dis),
            safe_step_count=int(safe_step_count),
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
        if batch_boundary:
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
                    elif key == "lane_change_cost":
                        value = float(np.clip(
                            value, EVALUATOR_MIN_LANE_CHANGE_COST,
                            EVALUATOR_MAX_LANE_CHANGE_COST
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
            write_diagnostic_event(
                "episode_level_evaluator", simulator, **evaluator_event
            )
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

    with open(plot_path, "w", encoding="utf-8") as f:
        f.write(str(reward_history) + "\n")
        f.write(str(collision_history) + "\n")
        f.write(str(collision_rate_history) + "\n")
        f.write(str(safe_travel_dis_history) + "\n")
        f.write(str(safe_step_count_history) + "\n")
        f.write(str(lane_change_history) + "\n")
        f.write(str(episode_times) + "\n")
        f.write(str(lane_change_rate_history) + "\n")
        f.write(str(mean_step_reward_history) + "\n")


    print("\\n[FINAL EPISODE METRICS]")
    print(f"collision_rate={float(np.mean(collision_rate_history)) if collision_rate_history else 0.0:.4f}")
    print(f"lane_change_rate={float(np.mean(lane_change_rate_history)) if lane_change_rate_history else 0.0:.4f}")
    print(f"safe_travel_dis={float(np.mean(safe_travel_dis_history)) if safe_travel_dis_history else 0.0:.2f} m/episode (mean)")
    print(f"safe_step_count={float(np.mean(safe_step_count_history)) if safe_step_count_history else 0.0:.2f} steps/episode (mean)")

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
    print("Safety-aware lane-only RL + episode-level prompt-ready evaluator; no real LLM call.")
    print(f"State dimension: {STATE_DIM}; policy actions: {POLICY_ACTION_DIM}; execution envelope: {ACTION_DIM}")
    print("Architecture: safety-augmented observation, pre-policy mask, KEEP/LEFT/RIGHT policy, deterministic acceleration controller.")
    print(f"Episode evaluator batch size: {EVALUATOR_BATCH_SIZE}; backend: {PromptBasedEvaluator.backend_name}")
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


if __name__ == "__main__":
    main()








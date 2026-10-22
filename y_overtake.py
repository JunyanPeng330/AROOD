# SIMPLIFIED / SINGLE-POLICY-DECISION VERSION
# New standalone copy of y_action_consistent_lane_safety.py.
# The original file is intentionally left unchanged.
# A LEFT/RIGHT policy choice sets a persistent target lane; the low-level
# steering controller executes the maneuver over multiple simulator steps.
# This is not an instantaneous lane teleport. TTC/gap checks and the safety
# shield remain active.
# Changes: explicit same-longitudinal-position handling, clearer overlap-vs-collision
# diagnostics, and TTC filter invocation IDs. Original y_discrete.py is unchanged.
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
# 1. Configuration
# ============================================================
steering_scale = 0.5
offroad_penalty = 0.5

# The policy acts at 5 Hz; the simulator advances physics at 15 Hz (3 substeps/action).
policy_frequency = 5
simulation_frequency = 15

lane_keeping_weight = 0.20

# Stable long-term reward objectives. TTC is reserved for the independent
# safety shield; it is not a continuous reward term.
INITIAL_TRACKING_WEIGHT = 0.20
INITIAL_PROGRESS_WEIGHT = 0.35
INITIAL_HEADWAY_WEIGHT = 1.50
INITIAL_COLLISION_PENALTY = 10.0
INITIAL_OFFROAD_PENALTY = 5.0
INITIAL_LANE_CHANGE_COST = 0.01

# TTC thresholds are used by the safety shield, not reward shaping.
ttc_critical_threshold = 2.0
ttc_danger_threshold = 3.0
ttc_caution_threshold = 4.0

SPEED_MATCH_TTC = 6.0
SPEED_MATCH_GAP_TOLERANCE = 3.0  # metres; avoid needless overrides for tiny gap deviations
# Execution-time safety shield (TTC is measured in seconds).
SAFETY_OVERRIDE_TTC = 2.0       # Emergency braking threshold
LANE_CHANGE_MIN_TTC = 3.0       # Minimum current front TTC for entering a target lane
LANE_CHANGE_PREDICTION_HORIZON = 1.5  # Approximate lateral-transition safety horizon (seconds)
LANE_CHANGE_TTC_MARGIN = 0.5     # Extra rear-TTC margin during the transition
LANE_CHANGE_MIN_GAP = 8.0        # Shared static/predicted bumper-gap requirement (metres)

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

# Lane changes are evaluated through stable task rewards and safety filtering.
# Do not grant an instantaneous bonus based on TTC.
LANE_CHANGE_COST = 0.0

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

MAX_PHYSICAL_ACCELERATION = 5.0  # m/s^2




NUM_LANES = 3
NUM_SURROUNDING_VEHICLES = 6
VEHICLE_FEATURES = 7
NUM_VEHICLE_ROWS = NUM_SURROUNDING_VEHICLES + 1
VEHICLE_STATE_DIM = NUM_VEHICLE_ROWS * VEHICLE_FEATURES
CONTEXT_DIM = 4
STATE_DIM = VEHICLE_STATE_DIM + CONTEXT_DIM  # 53
ACTION_DIM = 2

HIDDEN_DIM = 128

NUM_REFERENCE_EPISODES = 40
NUM_ONLINE_EPISODES = 100

MAX_STEPS = 150

RISK_HORIZON = 5

TAU_MMD = 0.50
TAU_RISK = 0.60

TTC_PROXY_THRESHOLD = 0.2

DISTRIBUTION_WINDOW = 35

DYNAMICS_LR = 1e-3
RISK_LR = 1e-3
POLICY_LR = 3e-4

DYNAMICS_EPOCHS = 5
RISK_EPOCHS = 20

MMD_SIGMA = 1.0

# Overtaking motivation: only consider a pass when the current-lane leader
# is within this center-to-center distance and is not >=20% faster than ego.
OVERTAKE_FRONT_DISTANCE_MAX = 30.0  # metres; center-to-center distance
OVERTAKE_FRONT_SPEED_RATIO = 1.20    # no pass if leader speed >= 1.2 * ego speed

# Reference-collection scenario diversity.
# Keep the existing distribution-shift baseline range separate from the
# wider per-vehicle ranges used to create varied overtaking situations.
REFERENCE_GAP_MIN = 35.0
REFERENCE_GAP_MAX = 60.0
RANDOMIZED_FRONT_GAP_MIN = 20.0
RANDOMIZED_FRONT_GAP_MAX = 70.0
RANDOMIZED_REAR_GAP_MIN = 20.0
RANDOMIZED_REAR_GAP_MAX = 65.0
TRAFFIC_SPEED_VARIATION = 8.0


EMBEDDING_DIM = 384
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Structured diagnostics: one JSON object per line. These logs are designed to
# separate policy requests, shield decisions, physical maneuvers, and reward trends.
DIAGNOSTIC_LOG_PATH = "training_diagnostics.jsonl"


reward_path = 'figures/' + 'Re__1' + '.png'
adaptation_path = 'figures/' + 'Ad__1' + '.png'
plot_path = 'figures/' + 'plot__1' + '.txt'
MODEL_PATH = 'policy/' + 'adaptive_policy_highway__1.pt'
# reference model without Shift Detection
checkpoint_path = 'checkpoint/' + 'reference_checkpoint_highway__1.pt'


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



def flatten_transition(
    state,
    action,
    next_state
):
    """
    Construct the feature vector used by MMD.

    z_t = [s_t, a_t, s_{t+1}]
    """
    return np.concatenate(
        [
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32)
        ]
    )


# ============================================================
# 3. Dynamics Model
# ============================================================

class DynamicsModel(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):

        state = to_tensor(state)
        action = to_tensor(action)

        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=-1)

        return self.network(x)

    def predict(self, state, action):

        with torch.no_grad():
            state_t = to_tensor(state).unsqueeze(0)
            action_t = to_tensor(action).unsqueeze(0)
            prediction = self.forward(
                state_t,
                action_t
            ).squeeze(0)

        return prediction.cpu().numpy()


# ============================================================
# 4. Risk Predictor
# ============================================================
# this RiskPredictor is trained to predict future collision events, not general driving risk or distribution shift.
class RiskPredictor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        input_dim = (2 * state_dim + action_dim)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action, next_state_pred):

        state = to_tensor(state)
        action = to_tensor(action)
        next_state_pred = to_tensor(next_state_pred)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        if next_state_pred.ndim == 1:
            next_state_pred = next_state_pred.unsqueeze(0)

        x = torch.cat([state, action, next_state_pred], dim=-1)

        return self.network(x)

    def predict(self, state, action, next_state_pred):

        with torch.no_grad():
            state_t = to_tensor(state).unsqueeze(0)
            action_t = to_tensor(action).unsqueeze(0)
            next_state_t = to_tensor(next_state_pred).unsqueeze(0)

            risk = self.forward(state_t, action_t, next_state_t)

        return risk.squeeze()


# ============================================================
# 5. Actor-Critic Policy
# ============================================================
class ActorCriticPolicy(nn.Module):
    """
    Hybrid actor-critic:
      action[0] = continuous acceleration in [-1, 1]
      action[1] = discrete lane intention:
                 0=left, 1=keep lane, 2=right
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        steering_scale=None,  # Kept for compatibility with existing caller
    ):
        super().__init__()

        if action_dim != 2:
            raise ValueError(
                "Hybrid policy expects ACTION_DIM=2: "
                "[acceleration, lane_action_id]"
            )

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Continuous acceleration head: one Gaussian mean
        self.accel_mean = nn.Linear(hidden_dim, 1)

        # Learnable standard deviation for acceleration only
        self.accel_log_std = nn.Parameter(torch.zeros(1))

        # Discrete lane-intention head: left / keep / right
        self.lane_logits = nn.Linear(hidden_dim, 3)

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.optimizer = Adam(
            self.parameters(),
            lr=POLICY_LR,
        )



    def action_distributions(self, state, valid_action_mask=None):
        state = to_tensor(state)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        features = self.shared(state)

        accel_mean = self.accel_mean(features)
        accel_std = torch.exp(self.accel_log_std).expand_as(accel_mean)

        accel_distribution = torch.distributions.Normal(accel_mean, accel_std)

        lane_logits = self.lane_logits(features)

        if valid_action_mask is not None:
            mask = torch.as_tensor(
                valid_action_mask,
                dtype=torch.bool,
                device=lane_logits.device,
            )

            if mask.ndim == 1:
                mask = mask.unsqueeze(0)

            # Support a single mask for a batch of states.
            if mask.shape[0] == 1 and lane_logits.shape[0] > 1:
                mask = mask.expand(lane_logits.shape[0], -1)

            if mask.shape != lane_logits.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match "
                    f"lane logits {lane_logits.shape}"
                )

            if not mask.any(dim=-1).all():
                raise ValueError("Every state needs at least one valid lane action.")

            lane_logits = lane_logits.masked_fill(~mask, -1e9)

        lane_distribution = torch.distributions.Categorical(logits=lane_logits)

        return accel_distribution, lane_distribution



    def act(self, state, deterministic=False, valid_action_mask=None, return_aux=False,):
        accel_dist, lane_dist = self.action_distributions(
            state,
            valid_action_mask=valid_action_mask,
        )

        if deterministic:
            raw_accel = accel_dist.mean
            lane_action = torch.argmax(lane_dist.logits, dim=-1)
        else:
            raw_accel = accel_dist.sample()
            lane_action = lane_dist.sample()

        acceleration = torch.tanh(raw_accel)

        action = torch.cat([acceleration, lane_action.float().unsqueeze(-1)], dim=-1,)

        action_np = (
            action.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
        )

        if not return_aux:
            return action_np

        # Preserve the original Gaussian sample before tanh.
        pre_tanh_np = (
            raw_accel.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
        )

        return action_np, pre_tanh_np



    def __call__(self, state, valid_action_mask=None):
        return self.act(
            state,
            deterministic=False,
            valid_action_mask=valid_action_mask,
        )




    def value(self, state):
        state = to_tensor(state)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        features = self.shared(state)
        return self.critic(features).squeeze(-1)






def lane_policy_diagnostic(policy, state, valid_action_mask, sampled_action, phase, episode, step):
    """Compactly report raw/masked lane logits, probabilities, and sampled intent."""
    with torch.no_grad():
        state_tensor = to_tensor(state)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        features = policy.shared(state_tensor)
        raw_logits = policy.lane_logits(features)
        raw_probs = torch.softmax(raw_logits, dim=-1)

        mask = torch.as_tensor(valid_action_mask, dtype=torch.bool, device=raw_logits.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        masked_logits = raw_logits.masked_fill(~mask, -1e9)
        masked_probs = torch.softmax(masked_logits, dim=-1)

        raw_logits_np = raw_logits.squeeze(0).cpu().numpy()
        raw_probs_np = raw_probs.squeeze(0).cpu().numpy()
        masked_logits_np = masked_logits.squeeze(0).cpu().numpy()
        masked_probs_np = masked_probs.squeeze(0).cpu().numpy()

    lane_index = int(np.clip(np.rint(float(sampled_action[1])), 0, 2))
    lane_name = ("LEFT", "KEEP", "RIGHT")[lane_index]
    print(
        "[LANE POLICY DISTRIBUTION]",
        f"phase={phase}",
        f"episode={episode}",
        f"step={step}",
        f"valid_action_mask={np.asarray(valid_action_mask, dtype=bool).tolist()}",
        f"raw_lane_logits={np.round(raw_logits_np, 4).tolist()}",
        f"raw_lane_probs={np.round(raw_probs_np, 4).tolist()}",
        f"masked_lane_logits={np.round(masked_logits_np, 4).tolist()}",
        f"masked_lane_probs={np.round(masked_probs_np, 4).tolist()}",
        f"sampled_lane_action_index={lane_index}",
        f"sampled_lane_action_name={lane_name}",
    )


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
                "pip install highway-env gymnasium sentence-transformers"
            )
        if state_dim != STATE_DIM or action_dim != ACTION_DIM:
            raise ValueError(f"HighwaySimulator expects STATE_DIM={STATE_DIM} and ACTION_DIM={ACTION_DIM}")
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

        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)



    def set_distribution_shift(self, level):
        """Vary traffic gaps within the configured [35, 60] m range."""
        level = float(np.clip(level, 0.0, 1.0))
        gap = REFERENCE_GAP_MAX - (REFERENCE_GAP_MAX - REFERENCE_GAP_MIN) * level
        self.traffic_front_gap = float(np.clip(gap, REFERENCE_GAP_MIN, REFERENCE_GAP_MAX))
        self.traffic_rear_gap = float(np.clip(gap, REFERENCE_GAP_MIN, REFERENCE_GAP_MAX))
        print("[TRAFFIC SETTING]", f"vehicles={NUM_SURROUNDING_VEHICLES}",
              f"front_gap={self.traffic_front_gap:.1f}m",
              f"rear_gap={self.traffic_rear_gap:.1f}m",
              f"speed_variation=±{self.traffic_speed_variation:.1f}m/s")



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
              f"gap_sampling={'independent_per_lane_and_side' if self.randomize_traffic_gaps else 'configured_shift_setting'}",
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

    def _overtaking_motivation(self, current_lane_values=None):
        """Return whether passing is worthwhile, independent of target-lane safety.

        A pass is motivated only when a current-lane front vehicle exists,
        its center distance is <= 30 m, and its speed is less than 120% of
        ego speed. This is motivation only: target-lane safety is checked
        separately by the TTC/gap shield.
        """
        ego = self.env.unwrapped.vehicle
        if current_lane_values is None:
            current_lane_values = self._min_ttc_by_lane().get(
                int(ego.lane_index[2]), {}
            )

        front_vehicle_id = current_lane_values.get("front_vehicle_id")
        front_distance = float(current_lane_values.get("front_center_gap", float("inf")))
        front_speed = current_lane_values.get("front_speed")
        ego_speed = max(0.0, float(ego.speed))

        if front_vehicle_id is None or front_speed is None or not np.isfinite(front_distance):
            return False, {
                "reason": "no_current_lane_front_vehicle",
                "front_distance_m": front_distance,
                "front_speed_mps": front_speed,
                "ego_speed_mps": ego_speed,
            }
        if front_distance > OVERTAKE_FRONT_DISTANCE_MAX:
            reason = "front_vehicle_beyond_overtaking_distance"
            should_overtake = False
        elif float(front_speed) >= OVERTAKE_FRONT_SPEED_RATIO * ego_speed:
            reason = "front_vehicle_not_slow_enough"
            should_overtake = False
        else:
            reason = "front_vehicle_constrains_progress"
            should_overtake = True

        return should_overtake, {
            "reason": reason,
            "front_distance_m": front_distance,
            "front_speed_mps": float(front_speed),
            "ego_speed_mps": ego_speed,
        }

    def _lane_change_safety_reasons(self, values, ego_speed, min_gap=LANE_CHANGE_MIN_GAP):
        """Return the canonical target-lane safety rejection reasons.

        The same checks are used by the policy action mask and execution shield,
        preventing the mask from using stricter/different thresholds than the shield.
        Front safety uses current TTC + bumper gap. Rear safety uses bumper gap
        and a short-horizon prediction; a separate static rear-TTC check would
        duplicate the predictive rear-TTC check.
        """
        reasons = []
        front_ttc = float(values.get("front_ttc", float("inf")))
        front_gap = float(values.get("front_gap", float("inf")))
        rear_gap = float(values.get("rear_gap", float("inf")))
        rear_speed = values.get("rear_speed", None)
        ego_speed = max(0.0, float(ego_speed))

        if front_ttc <= LANE_CHANGE_MIN_TTC:
            reasons.append("target_lane_front_ttc_below_threshold")
        if front_gap < min_gap:
            reasons.append("target_lane_front_gap_too_small")
        if rear_gap < min_gap:
            reasons.append("target_lane_rear_gap_too_small")

        if rear_speed is not None and np.isfinite(rear_gap):
            rear_closing_speed = max(0.0, float(rear_speed) - ego_speed)
            predicted_rear_gap = (
                rear_gap - rear_closing_speed * LANE_CHANGE_PREDICTION_HORIZON
            )
            predicted_rear_ttc = (
                rear_gap / rear_closing_speed
                if rear_closing_speed > 0.1 else float("inf")
            )
            if predicted_rear_gap < min_gap:
                reasons.append("target_lane_rear_gap_closes_during_maneuver")
            if predicted_rear_ttc <= (LANE_CHANGE_MIN_TTC + LANE_CHANGE_TTC_MARGIN):
                reasons.append("target_lane_rear_ttc_unsafe_during_maneuver")
        return reasons

    def get_valid_lane_action_mask(self):
        """Mask lane changes unless overtaking is motivated and the target is safe."""
        base_env = self.env.unwrapped
        road = base_env.road
        vehicle = base_env.vehicle
        if self._lane_change_active or self._lane_change_cooldown > 0:
            mask = np.array([False, True, False], dtype=bool)
            return mask

        current_lane = int(vehicle.lane_index[2])
        start_node, end_node, _ = vehicle.lane_index
        available_lanes = road.network.graph[start_node][end_node]
        valid_ids = set(range(len(available_lanes)))
        lane_ttc = self._min_ttc_by_lane()
        min_gap = LANE_CHANGE_MIN_GAP
        mask = [False, True, False]
        current_values = lane_ttc.get(current_lane, {})
        current_front_ttc = float(current_values.get("front_ttc", float("inf")))
        should_overtake, overtaking_info = self._overtaking_motivation(current_values)
        print(
            "[OVERTAKING MOTIVATION]",
            f"should_overtake={should_overtake}",
            f"reason={overtaking_info['reason']}",
            f"front_distance_m={overtaking_info['front_distance_m']:.3f}",
            f"front_speed_mps={overtaking_info['front_speed_mps']}",
            f"ego_speed_mps={overtaking_info['ego_speed_mps']:.3f}",
            f"distance_limit_m={OVERTAKE_FRONT_DISTANCE_MAX:.1f}",
            f"front_speed_ratio_limit={OVERTAKE_FRONT_SPEED_RATIO:.2f}",
        )
        # A critical TTC on the current lane is a longitudinal warning, not
        # an unconditional lateral-action veto. If passing is motivated, still
        # evaluate adjacent lanes; only a target lane that passes the safety
        # checks below may be selected. The execution shield retains emergency
        # braking while the lateral maneuver is considered.
        if not should_overtake:
            return np.asarray(mask, dtype=bool)

        # Policy index 0 means relative LEFT (-1); index 2 means
        # relative RIGHT (+1). Convert each semantic delta to a geometry ID.
        for action_id, lane_delta in ((0, -1), (2, +1)):
            candidate_lane = relative_lane_delta_to_absolute_lane(
                current_lane, lane_delta
            )
            if candidate_lane not in valid_ids:
                continue
            values = lane_ttc.get(candidate_lane)
            if values is None:
                continue
            mask[action_id] = not self._lane_change_safety_reasons(
                values, float(vehicle.speed), min_gap=min_gap
            )
        print(
            "[LANE ACTION MASK]",
            f"current_lane={current_lane}",
            f"target_lane={getattr(self, 'target_lane_index', None)}",
            f"mask={mask}",
            f"lane_change_active={self._lane_change_active}",
            f"cooldown={self._lane_change_cooldown}",
            f"min_ttc={LANE_CHANGE_MIN_TTC:.2f}",
            f"required_ttc={LANE_CHANGE_MIN_TTC + LANE_CHANGE_TTC_MARGIN:.2f}",
            f"min_gap={min_gap:.2f}",
            f"candidate_absolute_lane_ids={{0: {relative_lane_delta_to_absolute_lane(current_lane, -1)}, 2: {relative_lane_delta_to_absolute_lane(current_lane, +1)}}}",
        )
        return np.asarray(mask, dtype=bool)


#?????????????????????????????????????????????????????????????????????????????????????????????????????
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





    def filter_ttc_action(self, sampled_action):
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
        lane_change_threshold = float(LANE_CHANGE_MIN_TTC)

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
        # If the sampled policy action requests a lane change into a lane
        # that passes the same target-lane checks used by the execution shield,
        # do not let routine headway regulation cancel the overtaking acceleration.
        # This exception never disables critical-TTC emergency braking below.
        requested_lane_value = int(np.clip(np.rint(float(sampled_action[1])), 0, 2))
        requested_lane_delta_for_speed_match = {0: -1, 1: 0, 2: 1}[requested_lane_value]
        safe_overtake_request = False
        if requested_lane_delta_for_speed_match != 0 and not self._lane_change_active:
            current_lane_for_speed_match = int(vehicle.lane_index[2])
            target_lane_for_speed_match = relative_lane_delta_to_absolute_lane(
                current_lane_for_speed_match, requested_lane_delta_for_speed_match
            )
            should_overtake_for_speed_match, _ = self._overtaking_motivation(
                self._min_ttc_by_lane().get(current_lane_for_speed_match, {})
            )
            candidate_values_for_speed_match = self._min_ttc_by_lane().get(
                target_lane_for_speed_match
            )
            safe_overtake_request = (
                should_overtake_for_speed_match
                and candidate_values_for_speed_match is not None
                and not self._lane_change_safety_reasons(
                    candidate_values_for_speed_match,
                    float(vehicle.speed),
                    min_gap=LANE_CHANGE_MIN_GAP,
                )
            )

        speed_match_active = (
            front_vehicle_present
            and (front_ttc_trigger or gap_and_closing_trigger)
            and not rear_pressure_suppresses_routine_matching
            and not safe_overtake_request
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
        # 2. Decode the discrete lane action correctly.
        # ---------------------------------------------------------
        policy_requested_lane_action = int(np.clip(np.rint(sampled_action[1]), 0, 2))
        requested_lane_action = policy_requested_lane_action
        # Prioritize speed matching over initiating a new lane change.
        # Emergency braking is handled separately above.
        lane_action_names = {
            0: "LEFT",
            1: "KEEP",
            2: "RIGHT",
        }

        # Prioritize speed matching over initiating a new lane change.
        # Only report a deferred lane change if LEFT or RIGHT was requested.


        requested_lane_delta = lane_action_index_to_delta(policy_requested_lane_action)
        candidate_lane = relative_lane_delta_to_absolute_lane(
            current_lane, requested_lane_delta
        )
        requested_lane_name = {-1: "LEFT", 0: "KEEP", 1: "RIGHT"}[requested_lane_delta]

        filtered_lane_action = requested_lane_action
        filtered_lane_name = requested_lane_name
        if requested_lane_delta != 0 and getattr(self, "_lane_change_stats", None) is not None:
            self._lane_change_stats["requested"] += 1

        # Each rejection must have an explicit cause. Routine speed matching
        # is deliberately NOT a lane-change veto; only target-lane safety,
        # lane availability, an active lane change, or cooldown can reject it.
        lane_change_rejection_reasons = []
        lane_change_blocked = False
        candidate_front_ttc = float("inf")
        candidate_rear_ttc = float("inf")
        candidate_front_gap = float("inf")
        candidate_rear_gap = float("inf")
        candidate_front_gap_limit = LANE_CHANGE_MIN_GAP
        candidate_rear_gap_limit = LANE_CHANGE_MIN_GAP
        candidate_lane_available = False

        # One authoritative pre-decision snapshot. The selected neighbor IDs/gaps
        # and the raw target-lane vehicles below are all derived from this same
        # live simulator state and the same lane_ttc_current calculation.
        if requested_lane_delta != 0:
            ego_s_snapshot = float(vehicle.position[0])
            ego_speed_snapshot = float(vehicle.speed)
            ego_length_snapshot = float(getattr(vehicle, "LENGTH", getattr(vehicle, "length", 5.0)))
            target_lane_vehicles = []
            for other in list(base_env.road.vehicles):
                if other is vehicle or bool(getattr(other, "crashed", False)):
                    continue
                try:
                    other_lane_snapshot = int(other.lane_index[2])
                except (AttributeError, TypeError, ValueError, IndexError):
                    continue
                if other_lane_snapshot != candidate_lane:
                    continue

                center_gap_snapshot = float(other.position[0]) - ego_s_snapshot
                other_length_snapshot = float(getattr(other, "LENGTH", getattr(other, "length", 5.0)))
                bumper_gap_snapshot = abs(center_gap_snapshot) - 0.5 * (
                    ego_length_snapshot + other_length_snapshot
                )
                other_speed_snapshot = float(other.speed)
                if center_gap_snapshot > 0.0:
                    side_snapshot = "front"
                    closing_speed_snapshot = ego_speed_snapshot - other_speed_snapshot
                elif center_gap_snapshot < 0.0:
                    side_snapshot = "rear"
                    closing_speed_snapshot = other_speed_snapshot - ego_speed_snapshot
                else:
                    side_snapshot = "overlap"
                    closing_speed_snapshot = float("nan")

                if bumper_gap_snapshot <= 0.0:
                    ttc_snapshot = 0.0
                elif np.isfinite(closing_speed_snapshot) and closing_speed_snapshot > 0.1:
                    ttc_snapshot = bumper_gap_snapshot / closing_speed_snapshot
                else:
                    ttc_snapshot = float("inf")

                target_lane_vehicles.append({
                    "id": id(other),
                    "lane": other_lane_snapshot,
                    "side": side_snapshot,
                    "center_gap_m": round(center_gap_snapshot, 3),
                    "bumper_gap_m": round(bumper_gap_snapshot, 3),
                    "speed_mps": round(other_speed_snapshot, 3),
                    "closing_speed_mps": (
                        round(float(closing_speed_snapshot), 3)
                        if np.isfinite(closing_speed_snapshot) else None
                    ),
                    "ttc_s": (
                        round(float(ttc_snapshot), 3)
                        if np.isfinite(ttc_snapshot) else "inf"
                    ),
                })
            target_lane_vehicles.sort(key=lambda item: abs(item["center_gap_m"]))
            selected_target_values = lane_ttc_current.get(candidate_lane, {})
            print(
                "[LANE-CHANGE SAFETY SNAPSHOT]",
                f"filter_call={ttc_filter_call_id}",
                f"sim_time={getattr(base_env, 'time', None)}",
                f"physics_steps={getattr(base_env, 'steps', None)}",
                f"policy_step={self.step_count}",
                f"ego_lane={current_lane}",
                f"target_lane={candidate_lane}",
                f"requested={requested_lane_name}",
                f"ego_speed_mps={ego_speed_snapshot:.3f}",
                f"source_rear_ttc_s={current_values.get('rear_ttc', float('inf')):.3f}",
                f"source_rear_pressure={float(current_values.get('rear_ttc', float('inf'))) <= ttc_danger_threshold}",
                f"selected_front_id={selected_target_values.get('front_vehicle_id')}",
                f"selected_front_gap_m={selected_target_values.get('front_gap', float('inf')):.3f}",
                f"selected_front_ttc_s={selected_target_values.get('front_ttc', float('inf')):.3f}",
                f"selected_rear_id={selected_target_values.get('rear_vehicle_id')}",
                f"selected_rear_gap_m={selected_target_values.get('rear_gap', float('inf')):.3f}",
                f"selected_rear_ttc_s={selected_target_values.get('rear_ttc', float('inf')):.3f}",
                f"target_lane_vehicles={target_lane_vehicles}",
            )

        overtaking_motivated, overtaking_info = self._overtaking_motivation(current_values)
        if requested_lane_delta != 0 and not overtaking_motivated:
            lane_change_rejection_reasons.append(
                "overtaking_not_motivated:" + str(overtaking_info["reason"])
            )

        if requested_lane_delta != 0:
            # Check the actual road graph, not only a hard-coded lane count.
            try:
                start_node, end_node, _ = vehicle.lane_index
                available_lanes = base_env.road.network.graph[start_node][end_node]
                available_lane_ids = set(range(len(available_lanes)))
                candidate_lane_available = candidate_lane in available_lane_ids
            except (AttributeError, KeyError, TypeError, IndexError):
                candidate_lane_available = 0 <= candidate_lane < lanes_count

            if not candidate_lane_available:
                lane_change_rejection_reasons.append("target_lane_unavailable")

            if self._lane_change_active:
                lane_change_rejection_reasons.append("lane_change_already_in_progress")

            if self._lane_change_cooldown > 0:
                lane_change_rejection_reasons.append("lane_change_cooldown")

            candidate_values = lane_ttc_current.get(candidate_lane)
            if candidate_lane_available and candidate_values is not None:
                candidate_front_ttc = float(candidate_values["front_ttc"])
                candidate_rear_ttc = float(candidate_values["rear_ttc"])
                candidate_front_gap = float(candidate_values.get("front_gap", float("inf")))
                candidate_rear_gap = float(candidate_values.get("rear_gap", float("inf")))

                # Use the same canonical target-lane rule as the policy mask.
                # This avoids duplicate static rear-TTC checks that overlap with
                # the rear-TTC prediction over the lane-transition horizon.
                lane_change_rejection_reasons.extend(
                    self._lane_change_safety_reasons(
                        candidate_values,
                        float(vehicle.speed),
                        min_gap=LANE_CHANGE_MIN_GAP,
                    )
                )

                # Source-lane rear pressure is logged separately, not treated as
                # an automatic veto. A fast-approaching source-lane vehicle can
                # make leaving the lane desirable; target-lane/transition safety
                # remains the deciding constraint.
                source_rear_ttc = float(current_values.get("rear_ttc", float("inf")))
                source_rear_pressure = source_rear_ttc <= ttc_danger_threshold
            elif candidate_lane_available:
                lane_change_rejection_reasons.append("target_lane_traffic_data_unavailable")

            # Emergency braking is a longitudinal override, not an automatic
            # lane-change veto. Target-lane and maneuver safety checks above
            # still reject unsafe lateral moves. This allows an escape lane to
            # be selected while braking for a critical current-lane leader.

            lane_change_blocked = bool(lane_change_rejection_reasons)
            if lane_change_blocked:
                filtered_action[1] = 1.0
                filtered_lane_action = 1
                filtered_lane_name = "KEEP"

        lane_change_rejection_reason = (
            ",".join(lane_change_rejection_reasons)
            if lane_change_rejection_reasons else "none"
        )
        if requested_lane_delta != 0 and getattr(self, "_lane_change_stats", None) is not None:
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

        filtered_action[1] = float(np.clip(np.rint(filtered_action[1]), 0, 2))




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
            "filtered_lane_delta": lane_action_index_to_delta(filtered_lane_action),
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
            "overtaking_motivated": bool(overtaking_motivated),
            "overtaking_motivation": dict(overtaking_info),
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
            "emergency_braking": bool(emergency_braking),
            "lane_change_blocked": bool(lane_change_blocked),
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

        if requested_lane_delta == 0:
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




        # Structured request-level trace: one ID per sampled LEFT/RIGHT decision.
        # This intentionally does not equate a request with a unique physical maneuver.
        if requested_lane_delta != 0:
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
        Convert a categorical policy index into a persistent absolute lane target.

        Policy index 0/1/2 maps to semantic relative delta -1/0/+1
        (LEFT/KEEP/RIGHT). Absolute Highway lane IDs are retained for geometry;
        in this road graph, absolute IDs increase toward the left.
        """

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




    def step(self, action, apply_safety_filter=True, sampled_action=None, safety_info=None,):
        """
        Input:
            action[0] = continuous normalized acceleration
            action[1] = categorical lane-action index:
                        0 -> delta -1 (LEFT), 1 -> delta 0 (KEEP),
                        2 -> delta +1 (RIGHT)
        The lane decision is distinct from low-level steering.
        Highway receives:
            [continuous acceleration, continuous normalized steering]
        """


        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size != 2:
            raise ValueError(
                f"Expected hybrid action of size 2, got {action.shape}"
            )

        # The action received by the adapter is the source of truth.
        incoming_action = np.asarray(action, dtype=np.float32).copy()

        if apply_safety_filter:
            # Filter the action that was actually received.
            executed_action, safety_info = self.filter_ttc_action(incoming_action.copy())
            executed_action = np.asarray(executed_action, dtype=np.float32).copy()
            safety_filter_applied = True

        else:
            # Bypass filtering: preserve the incoming action.
            executed_action = incoming_action.copy()
            safety_filter_applied = False

            if safety_info is not None:
                safety_info = dict(safety_info)
            else:
                # Prefer a diagnostic-only method here if available.
                safety_info = {
                    "ttc": float("inf"),
                    "safety_filter_applied": False,
                }

        # Keep action provenance explicit. In the normal training path the
        # filter runs once before step(), then step() receives that output with
        # apply_safety_filter=False to prevent a second filter pass.
        prior_filtered = safety_info.get("filtered_action")
        if not safety_filter_applied and prior_filtered is not None:
            prior_filtered = np.asarray(prior_filtered, dtype=np.float32).reshape(-1)
            if prior_filtered.size == 2 and not np.allclose(prior_filtered, incoming_action, atol=1e-5):
                print(
                    "[ACTION FLOW WARNING] step input differs from prior filter output:",
                    f"prior_filtered={prior_filtered.tolist()}",
                    f"step_input={incoming_action.tolist()}",
                )
        safety_info["simulator_input_action"] = incoming_action.copy()
        safety_info["executed_high_level_action"] = executed_action.copy()
        safety_info["safety_filter_applied_in_step"] = safety_filter_applied
        safety_info["safety_filter_applied"] = bool(
            safety_info.get("safety_filter_applied", False) or safety_filter_applied
        )

        print(
            f"[ACTION FLOW FIXED] "
            f"filter_call_id={getattr(self, '_ttc_filter_call_count', 'not_called')} "
            f"incoming={incoming_action.tolist()} "
            f"executed={executed_action.tolist()} "
            f"filter_applied={safety_filter_applied}"
        )

        # Use executed_action in the subsequent action-mapping code.
        action = executed_action
        action = np.asarray(action, dtype=np.float32).copy()

        print(
            "[ACTION FLOW]",
            f"policy_sampled={(np.asarray(sampled_action, dtype=np.float32).reshape(-1).tolist() if sampled_action is not None else None)}",
            f"step_input={incoming_action.tolist()}",
            f"simulator_action={action.tolist()}",
            f"filter_applied_in_step={safety_filter_applied}",
            f"filter_applied_before_step={bool(safety_info.get('safety_filter_applied', False))}",
            f"filter_call_id={safety_info.get('ttc_filter_call_id', 'N/A')}",
            f"lane_rejection_reason={safety_info.get('lane_change_rejection_reason', 'N/A')}",
        )
        print(
            "[ACTION AGREEMENT]",
            f"incoming={incoming_action.tolist()}",
            f"adapter_filtered={action.tolist()}",
            f"same={np.allclose(incoming_action, action)}",)


        acceleration = float(np.clip(action[0], -1.0, 1.0))
        lane_action_id = int(np.clip(np.rint(action[1]), 0, 2))
        current_lane_before = int(self.env.unwrapped.vehicle.lane_index[2])

        lane_action_delta = lane_action_index_to_delta(lane_action_id)
        lane_action_names = {-1: "LEFT", 0: "KEEP", 1: "RIGHT"}
        lane_action_name = lane_action_names[lane_action_delta]
        requested_target_lane_id = relative_lane_delta_to_absolute_lane(
            current_lane_before, lane_action_delta
        )

        sampled_for_log = (
            np.asarray(sampled_action, dtype=np.float32).reshape(-1).tolist()
            if sampled_action is not None else None
        )
        print(
            "[ACTION MAPPING]",
            f"sampled_action={sampled_for_log}",
            f"incoming_action={incoming_action.tolist()}",
            f"filtered_action={action.tolist()}",
            f"current_lane={current_lane_before}",
            f"lane_action_index={lane_action_id}",
            f"lane_action_delta={lane_action_delta:+d}",
            f"lane_action={lane_action_name}",
            f"current_absolute_lane_id={current_lane_before}",
            f"requested_target_lane_id={requested_target_lane_id}",
        )

        # One high-level LEFT/RIGHT decision sets a persistent target lane.
        # The low-level steering controller keeps tracking that target across
        # subsequent simulator steps; the policy does not need to repeat LEFT/RIGHT.
        active_before_update = bool(self._lane_change_active)
        cooldown_before_update = int(self._lane_change_cooldown)
        previous_target_lane = tuple(self.target_lane_index) if self.target_lane_index is not None else None
        request_trace_id = safety_info.get("lane_request_trace_id")
        self._current_lane_request_trace_id = request_trace_id
        self._update_lane_target(lane_action_id)
        active_after_update = bool(self._lane_change_active)
        if request_trace_id is not None:
            if safety_info.get("lane_change_blocked", False):
                adapter_outcome = "FILTER_REJECTED"
            elif not active_before_update and active_after_update and self._active_lane_request_trace_id == request_trace_id:
                adapter_outcome = "MANEUVER_STARTED"
            elif active_before_update:
                adapter_outcome = "IGNORED_WHILE_MANEUVER_ACTIVE"
            elif cooldown_before_update > 0:
                adapter_outcome = "REJECTED_COOLDOWN"
            elif previous_target_lane == tuple(self.target_lane_index):
                adapter_outcome = "NO_TARGET_CHANGE"
            else:
                adapter_outcome = "TARGET_UPDATED_NOT_STARTED"
            write_diagnostic_event(
                "lane_request_adapter_outcome", self,
                request_id=int(request_trace_id),
                direction=safety_info.get("lane_request_direction"),
                safety_decision=safety_info.get("lane_request_decision"),
                adapter_outcome=adapter_outcome,
                rejection_reasons=safety_info.get("lane_request_rejection_reasons", []),
                source_lane=safety_info.get("lane_request_source_lane"),
                requested_target_lane=safety_info.get("lane_request_target_lane"),
                previous_target_lane=previous_target_lane,
                current_target_lane=self.target_lane_index,
                active_maneuver_id=self._active_lane_maneuver_id,
                active_before=active_before_update,
                active_after=active_after_update,
                cooldown_before=cooldown_before_update,
                filtered_action=action.tolist(),
            )
        self._current_lane_request_trace_id = None
        if safety_info.get("lane_request_trace_id") is not None:
            print(
                "[LANE REQUEST OUTCOME]",
                f"id={safety_info['lane_request_trace_id']}",
                f"decision={safety_info.get('lane_request_decision')}",
                f"source_lane={safety_info.get('lane_request_source_lane')}",
                f"target_lane={safety_info.get('lane_request_target_lane')}",
                f"active_before={active_before_update}",
                f"active_after={active_after_update}",
                f"previous_target={previous_target_lane}",
                f"current_target={self.target_lane_index}",
                f"physical_lane_after_update={int(self.env.unwrapped.vehicle.lane_index[2])}",
            )

        target_lane = getattr(self, "target_lane", None)
        print(
            "[LANE TARGET]",
            f"current_lane={current_lane_before}",
            f"lane_action_delta={lane_action_delta:+d}",
            f"lane_action={lane_action_name}",
            f"target_lane_index={self.target_lane_index}",
            f"target_absolute_lane_id={int(self.target_lane_index[2])}",
            f"lane_change_active={self._lane_change_active}",
            f"source_lane_index={self._lane_change_source_index}",
            f"cooldown={self._lane_change_cooldown}",
        )

        current_speed = float(self.env.unwrapped.vehicle.speed)
        if current_speed >= self.env.unwrapped.vehicle.MAX_SPEED:
            if acceleration > 0.0:
                acceleration = 0.0
                action[0] = acceleration
                safety_info["filtered_action"] = action.copy()
                safety_info["speed_limit_override"] = True
                safety_info["overridden"] = True



        # Continuous low-level steering controller.
        steering, lateral, heading_error = (
            self.get_lane_keeping_steering(
                k_lateral=0.08,
                k_heading=0.8,
                max_steering=steering_scale,
            )
        )


        #???????????????????????????????????????????????????????????????????????????????????????????????
        # Final longitudinal speed guard, immediately before env.step().
        base_env = self.env.unwrapped
        ego = base_env.vehicle

        ego_before = base_env.vehicle
        ego_position_before = np.asarray(ego_before.position, dtype=float).copy()

        current_speed = float(ego.speed)
        max_physical_acceleration = 5.0
        policy_dt = 1.0 / float(base_env.config["policy_frequency"])

        # Allow a small positive speed margin to avoid crossing zero
        # due to numerical/discretization differences.
        minimum_target_speed = 0.05

        if current_speed < -0.1:
            raise RuntimeError(
                f"Ego speed was already negative before execution: "
                f"{current_speed:.4f} m/s. "
                "Inspect the previous [SPEED BEFORE]/[SPEED AFTER] pair "
                "and restart the episode."
            )

        if acceleration < 0.0 and current_speed >= 0.0:
            min_safe_acceleration = (minimum_target_speed - current_speed) / (max_physical_acceleration * policy_dt)
            min_safe_acceleration = float(np.clip(min_safe_acceleration, -1.0, 0.0))

            acceleration_before_guard = acceleration
            acceleration = max(acceleration, min_safe_acceleration,)

            if acceleration != acceleration_before_guard:
                print(
                    "[ZERO-SPEED GUARD]",
                    f"speed={current_speed:.4f}",
                    f"accel_before={acceleration_before_guard:.4f}",
                    f"accel_after={acceleration:.4f}",
                    f"predicted_speed={current_speed + acceleration * max_physical_acceleration * policy_dt:.4f}",
                )

        # Construct the actual physical action only after the guard.

        # Physical action sent to Highway
        executed_action = np.asarray([acceleration, steering], dtype=np.float32)



        speed_before = float(ego_before.speed)
        crashed_before = bool(getattr(ego_before, "crashed", False))
        max_speed = getattr(ego_before, "MAX_SPEED", None)


        obs, env_reward, terminated, truncated, info = (self.env.step(executed_action))

        ego_after = self.env.unwrapped.vehicle
        speed_after = float(ego_after.speed)
        crashed_after = bool(getattr(ego_after, "crashed", False))

        # Trace the target geometry on every physical step. Completion is
        # finalized at the next high-level decision in _update_lane_target().
        try:
            target_lane_geometry = self.env.unwrapped.road.network.get_lane(self.target_lane_index)
            target_s, target_lateral = target_lane_geometry.local_coordinates(ego_after.position)
            target_heading = target_lane_geometry.heading_at(target_s)
            target_heading_error = (float(ego_after.heading) - float(target_heading) + np.pi) % (2.0 * np.pi) - np.pi
        except (AttributeError, KeyError, TypeError, ValueError):
            target_lateral = float("nan")
            target_heading_error = float("nan")

        if crashed_after:
            ego_pos = np.asarray(ego_after.position, dtype=float)
            print(
                "[COLLISION EGO]",
                f"id={id(ego_after)}",
                f"position={ego_pos.tolist()}",
                f"lane_index={getattr(ego_after, 'lane_index', None)}",
                f"speed={float(ego_after.speed):.3f}",
                f"target_lane={self.target_lane_index}",
                f"source_lane={self._lane_change_source_index}",
                f"lane_change_active={self._lane_change_active}",
                f"steering={steering:.4f}",
                f"acceleration={acceleration:.4f}",)
            for other in list(base_env.road.vehicles):
                if other is ego_after:
                    continue
                other_pos = np.asarray(other.position, dtype=float)
                delta = other_pos - ego_pos
                print(
                    "[COLLISION CANDIDATE]",
                    f"id={id(other)}",
                    f"crashed={bool(getattr(other, 'crashed', False))}",
                    f"position={other_pos.tolist()}",
                    f"lane_index={getattr(other, 'lane_index', None)}",
                    f"speed={float(other.speed):.3f}",
                    f"relative_dx={float(delta[0]):.3f}",
                    f"relative_dy={float(delta[1]):.3f}",
                    f"center_distance={float(np.linalg.norm(delta)):.3f}",)



        dt = 1.0 / float(self.env.unwrapped.config["policy_frequency"])

        observed_acceleration = (speed_after - speed_before) / dt

        print(
            f"[SPEED AFTER] "
            f"speed={speed_after:.4f} m/s, "
            f"delta_v={speed_after - speed_before:+.4f} m/s, "
            f"observed_accel={observed_acceleration:+.4f} m/s^2, "
            f"crashed={crashed_after}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )
        #??????????????????????????????????????????????????????????????????????????????????????

        self.step_count += 1
        self._maintain_six_vehicle_traffic()
        obs = self.env.unwrapped.observation_type.observe()
        next_state = self._encode_state(obs)
        self._assert_traffic_invariant("step")

        ego = self.env.unwrapped.vehicle
        collision = bool(getattr(ego, "crashed", False))

        info = dict(info or {})
        info["safety_violation"] = collision
        info["collision"] = collision
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["env_reward"] = float(env_reward)

        # Keep both representations distinct.
        info["lane_action_id"] = lane_action_id  # categorical index for policy compatibility
        info["lane_action_index"] = lane_action_id
        info["lane_action_delta"] = lane_action_delta
        info["lane_action_name"] = lane_action_name
        info["current_lane_id"] = current_lane_before
        info["requested_target_lane_id"] = requested_target_lane_id
        info["accepted_target_lane_id"] = int(self.target_lane_index[2])
        info["sampled_policy_action"] = sampled_action.copy()
        info["policy_action"] = np.asarray([acceleration, lane_action_id], dtype=np.float32,)
        info["filtered_policy_action"] = np.asarray([acceleration, lane_action_id], dtype=np.float32)
        info["executed_action"] = executed_action.copy()  # [normalized acceleration, normalized steering]
        info["executed_acceleration"] = float(acceleration)
        info["executed_steering"] = float(steering)
        info["ttc_safety_filter"] = safety_info

        self.state = next_state

        return (
            next_state.copy(),
            float(env_reward),
            bool(terminated or truncated),
            info,
        )





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
        direction: {"requested": 0, "mask_allowed": 0, "safety_accepted": 0, "completed": 0}
        for direction in ("LEFT", "RIGHT")
    }


def make_action_mask_counters():
    """Measure available lane actions independently of policy choices."""
    return {"steps": 0, "left_valid": 0, "right_valid": 0,
            "both_valid": 0, "only_keep": 0}


def record_action_mask(counters, valid_action_mask):
    mask = np.asarray(valid_action_mask, dtype=bool).reshape(-1)
    left_valid = bool(mask[0]) if mask.size > 0 else False
    right_valid = bool(mask[2]) if mask.size > 2 else False
    counters["steps"] += 1
    counters["left_valid"] += int(left_valid)
    counters["right_valid"] += int(right_valid)
    counters["both_valid"] += int(left_valid and right_valid)
    counters["only_keep"] += int(mask.size > 1 and bool(mask[1]) and not left_valid and not right_valid)


def print_action_mask_counters(tag, counters):
    steps = max(1, counters["steps"])
    print(f"[{tag} ACTION OPPORTUNITIES]",
          f"steps={counters['steps']}",
          f"left_valid={counters['left_valid']} ({100*counters['left_valid']/steps:.1f}%)",
          f"right_valid={counters['right_valid']} ({100*counters['right_valid']/steps:.1f}%)",
          f"both_valid={counters['both_valid']} ({100*counters['both_valid']/steps:.1f}%)",
          f"only_keep={counters['only_keep']} ({100*counters['only_keep']/steps:.1f}%)")


def record_lane_request_decision(counters, sampled_action, valid_action_mask, safety_info):
    """Count request, mask permission, and safety-filter acceptance for one step."""
    lane_index = int(np.clip(np.rint(float(sampled_action[1])), 0, 2))
    if lane_index == 1:
        return
    direction = "LEFT" if lane_index == 0 else "RIGHT"
    counters[direction]["requested"] += 1
    mask = np.asarray(valid_action_mask, dtype=bool).reshape(-1)
    if lane_index < len(mask) and bool(mask[lane_index]):
        counters[direction]["mask_allowed"] += 1
    if not bool(safety_info.get("lane_change_blocked", False)):
        counters[direction]["safety_accepted"] += 1


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
            f"safety_accepted={values['safety_accepted']}",
            f"completed={values['completed']}",
        )


def collect_rollouts(
    simulator,
    policy,
    num_episodes,
    max_steps
):
    rollouts = []
    reference_counters = make_lane_request_counters()
    reference_mask_counters = make_action_mask_counters()
    reference_seen_completed = {"LEFT": 0, "RIGHT": 0}

    for episode in range(num_episodes):
        simulator._trace_phase = "reference"
        simulator._trace_episode = episode + 1
        state = simulator.reset(randomize_traffic_gaps=True)
        rollout = []
        episode_counters = make_lane_request_counters()
        episode_mask_counters = make_action_mask_counters()
        episode_seen_completed = {
            "LEFT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_LEFT", 0)),
            "RIGHT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_RIGHT", 0)),
        }

        for t in range(max_steps):    #10

            # Policy chooses acceleration and a discrete lane intention.
            valid_mask = simulator.get_valid_lane_action_mask()
            simulator._trace_valid_action_mask = np.asarray(valid_mask, dtype=bool).tolist()
            record_action_mask(episode_mask_counters, valid_mask)
            record_action_mask(reference_mask_counters, valid_mask)
            sampled_action = np.asarray(
                policy(state, valid_action_mask=valid_mask),
                dtype=np.float32,
            ).reshape(-1)
            lane_policy_diagnostic(
                policy, state, valid_mask, sampled_action,
                phase="reference", episode=episode + 1, step=t + 1,
            )
            action, safety_info = simulator.filter_ttc_action(sampled_action)
            record_lane_request_decision(episode_counters, sampled_action, valid_mask, safety_info)
            record_lane_request_decision(reference_counters, sampled_action, valid_mask, safety_info)
            action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
            safety_info["policy_sampled_action"] = np.asarray(sampled_action, dtype=np.float32).copy()
            safety_info["filter_output_action"] = action.copy()
            print(
                "[ACTION FLOW PRE-STEP]",
                f"filter_call_id={safety_info.get('ttc_filter_call_id')}",
                f"policy_sampled={np.asarray(sampled_action).tolist()}",
                f"filter_output={action.tolist()}",
                f"lane_rejection_reason={safety_info.get('lane_change_rejection_reason', 'none')}",
            )
            # Measure physical TTC in seconds before the transition.
            min_ttc_before_step, _ = calculate_min_ttc(simulator)

            # Predict/learn from the filtered high-level action that will be executed.
            next_state, reward, done, info = simulator.step(action, apply_safety_filter=False, sampled_action=sampled_action, safety_info=safety_info,)
            sync_lane_completion_counters(episode_counters, simulator, episode_seen_completed)
            sync_lane_completion_counters(reference_counters, simulator, reference_seen_completed)

            # Store the high-level action selected by the policy.
            transition = {
                "state": np.asarray(state, dtype=np.float32),
                "action": action.copy(),
                "sampled_action": sampled_action.copy(),
                "safety_filter": safety_info,
                "valid_action_mask": valid_mask.copy(),
                "reward": float(reward),
                "next_state": np.asarray(next_state, dtype=np.float32),
                "safety_violation": bool(info.get("safety_violation", False)),
                "terminated": bool(info.get("terminated", False)),
                "truncated": bool(info.get("truncated", False)),
                "min_ttc": float(min_ttc_before_step),
            }

            rollout.append(transition)


            state = next_state
            if done:
                '''
                base_env = simulator.env.unwrapped
                print("\n=== EPISODE END ===")
                print("episode:", episode)
                print("collector t:", t)
                print("adapter step_count:", simulator.step_count)
                print("Highway time:", base_env.time)
                print("Highway steps:", base_env.steps)
                print("terminated:", info.get("terminated"))
                print("truncated:", info.get("truncated"))
                print("crashed:", info.get("collision"))
                print("===================\n")
                '''
                break

        sync_lane_completion_counters(episode_counters, simulator, episode_seen_completed)
        sync_lane_completion_counters(reference_counters, simulator, reference_seen_completed)
        rollouts.append(rollout)

        print(f"[REFERENCE LANE-REQUEST COUNTER] episode={episode + 1}/{num_episodes}")
        print_lane_request_counters("REFERENCE EPISODE", episode_counters)
        print_action_mask_counters("REFERENCE EPISODE", episode_mask_counters)
        print(
            f"Reference episode "
            f"{episode + 1}/"
            f"{num_episodes}"
        )

    print(f"[REFERENCE LANE-REQUEST TOTALS] episodes={num_episodes}")
    print_lane_request_counters("REFERENCE TOTAL", reference_counters)
    print_action_mask_counters("REFERENCE TOTAL", reference_mask_counters)
    return rollouts


# ============================================================
# 10. Train Dynamics Model
# ============================================================

def train_dynamics_model(
    rollouts,
    state_dim,
    action_dim
):

    dynamics_model = DynamicsModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = Adam(dynamics_model.parameters(), lr=DYNAMICS_LR)
    criterion = nn.MSELoss()

    dataset = []

    for rollout in rollouts:
        for transition in rollout:
            dataset.append(transition)

    if not dataset:
        raise RuntimeError("No transitions available for DynamicsModel training.")

    print(f"Training DynamicsModel " f"on {len(dataset)} transitions...")

    for epoch in range(DYNAMICS_EPOCHS):

        total_loss = 0.0
        random.shuffle(dataset)

        for transition in dataset:

            state = to_tensor(transition["state"])
            action = to_tensor(transition["action"])
            next_state = to_tensor(transition["next_state"])
            prediction = dynamics_model(state, action).squeeze(0)
            loss = criterion(prediction, next_state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = (total_loss / len(dataset))

        print(
            f"  Dynamics epoch "
            f"{epoch + 1}/"
            f"{DYNAMICS_EPOCHS}: "
            f"loss={mean_loss:.6f}"
        )

    return dynamics_model



#+++++++++++++++++++++++++++++++++++++++++++++++++++
# add a TTC (time to collision) into the risk computation
#+++++++++++++++++++++++++++++++++++++++++++++++++++
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




def calculate_ttc_penalty(
    min_ttc,
    critical_ttc=ttc_critical_threshold,
    safe_ttc=ttc_caution_threshold,
):
    if min_ttc <= critical_ttc:
        return 1.0
    if min_ttc >= safe_ttc or np.isinf(min_ttc):
        return 0.0
    return (safe_ttc - min_ttc) / (safe_ttc - critical_ttc)


def calculate_ttc_risk_indicator(state, debug=False):
    state = np.asarray(state, dtype=np.float32).reshape(-1)

    if state.size < VEHICLE_STATE_DIM:
        raise ValueError(f"Expected at least {VEHICLE_STATE_DIM} vehicle features, got {state.size}")

    vehicle_rows = state[:VEHICLE_STATE_DIM].reshape(NUM_VEHICLE_ROWS, VEHICLE_FEATURES)
    #print("Vehicle rows : ")
    #print(np.asarray(vehicle_rows))
    ttc_proxies = []

    for i, row in enumerate(vehicle_rows[1:], start=1):
        presence, x_rel, y_rel, vx_rel = row[:4]

        if presence < 0.5:
            reason = "no observed vehicle"
        elif x_rel <= 0.0:
            reason = "vehicle not ahead (x_rel <= 0)"
        elif vx_rel >= 0.0:
            reason = "not closing (vx_rel >= 0)"
        else:
            proxy = -float(x_rel) / float(vx_rel)
            ttc_proxies.append(proxy)

        if debug:
            print(
                f"Vehicle row {i}: "
                f"presence={presence:.2f}, "
                f"x_rel={x_rel:.3f}, "
                f"vx_rel={vx_rel:.3f} ,"
                f"proxy={proxy:.4f}"
            )
    result = min(ttc_proxies, default=float("inf"))
    if debug:
        print("Minimum TTC proxy:", result)
    return result



# ============================================================
# 11. Train Risk Predictor
# ============================================================


def train_risk_predictor(
    rollouts,
    dynamics_model,
    state_dim,
    action_dim,
    horizon_H
):

    risk_model = RiskPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = Adam(risk_model.parameters(), lr=RISK_LR)
    criterion = nn.BCELoss()

    samples = []

    for rollout in rollouts:
        if len(rollout) <= horizon_H:
            continue
        for t in range(len(rollout) - horizon_H):
            state = rollout[t]["state"]
            action = rollout[t]["action"]
            next_state_pred = dynamics_model.predict(state, action)

            # A positive label means a future collision or a TTC-like risk indicator crossing the threshold.
            risk_label = 0.0


            # Define this near the other TTC constants:
            TTC_RISK_LABEL_THRESHOLD = 2.0  # seconds; initial conservative label

            # Inside train_risk_predictor(), replace the existing future loop:
            for k in range(1, horizon_H + 1):
                future_transition = rollout[t + k]
                # A collision within the prediction horizon is positive risk.
                if future_transition["safety_violation"]:
                    risk_label = 1.0
                    break
                # Use the physical TTC stored during rollout collection.
                future_ttc = float(future_transition.get("min_ttc", float("inf")))
                if future_ttc <= TTC_RISK_LABEL_THRESHOLD:
                    risk_label = 1.0
                    break

            samples.append((state, action, next_state_pred, risk_label))


    if not samples:
        raise RuntimeError(
            "No samples available "
            "for RiskPredictor training."
        )





    for epoch in range(RISK_EPOCHS):

        random.shuffle(samples)
        total_loss = 0.0

        for (state, action, next_state_pred, risk_label) in samples:

            state = to_tensor(state)
            action = to_tensor(action)
            next_state_pred = to_tensor(next_state_pred)
            target = to_tensor([risk_label])
            prediction = risk_model(state, action, next_state_pred).view(-1)
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(samples)


        print(
            f"  Risk epoch "
            f"{epoch + 1}/"
            f"{RISK_EPOCHS}: "
            f"loss={mean_loss:.10e}"       # .6f
        )


# test the risk_model
    risk_model.eval()

    diagnostic_predictions = []
    diagnostic_labels = []
    with torch.no_grad():

        for state, action, next_state_pred, risk_label in samples:
            prediction = risk_model(
                to_tensor(state),
                to_tensor(action),
                to_tensor(next_state_pred)
            ).view(-1)

            diagnostic_predictions.append((float(prediction.item())))
            if torch.is_tensor(risk_label):
                label = float(risk_label.detach().cpu().item())
            else:
                label = float(np.asarray(risk_label).item())

            diagnostic_labels.append(label)


    # diagnoise                                                           这里positive label是0 个
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^diagnoise^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    preds = np.asarray(diagnostic_predictions, dtype=float)
    labels = np.asarray(diagnostic_labels, dtype=float)

    print("\n=== Same-Samples Consistency Check ===")
    print("Number of samples:", len(samples))
    print("Prediction count:", len(preds))
    print("Label count:", len(labels))
    print("Unique labels and counts:", np.unique(labels, return_counts=True))
    print("Positive labels:", int((labels == 1).sum()))
    print("Negative labels:", int((labels == 0).sum()))

#############################################

    return risk_model


# ============================================================
# 12. Build Reference Distribution
# ============================================================

def construct_reference_distribution(
    rollouts
):

    reference_distribution = []

    for rollout in rollouts:
        for transition in rollout:
            feature = flatten_transition(
                transition["state"],
                transition["action"],
                transition["next_state"]
            )

            reference_distribution.append(feature)

    if not reference_distribution:

        raise RuntimeError("Reference distribution is empty.")

    return np.asarray(reference_distribution, dtype=np.float32)


# ============================================================
# 13. Update Current Distribution
# ============================================================

def update_current_distribution(
    current_distribution,
    state,
    action,
    next_state
):

    feature = flatten_transition(
        state,
        action,
        next_state
    )

    current_distribution.append(feature)

    return current_distribution


# ============================================================
# 14. RBF Kernel
# ============================================================

def rbf_kernel(x, y, sigma=1.0):

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T

    squared_distance = (x_norm + y_norm - 2.0 * np.matmul(x, y.T))

    squared_distance = np.maximum(squared_distance, 0.0)

    return np.exp(-squared_distance / (2.0 * sigma * sigma))


# ============================================================
# 15. MMD Shift Detector
# ============================================================

def compute_MMD(
    reference_distribution,
    current_distribution,
    sigma=MMD_SIGMA
):

    reference = np.asarray(reference_distribution, dtype=np.float32)
    current = np.asarray(current_distribution, dtype=np.float32)

    if len(reference) == 0:
        return 0.0

    if len(current) == 0:
        return 0.0

    # --------------------------------------------------------
    # Limit reference samples for computational efficiency
    # --------------------------------------------------------

    reference_sample_size = min(len(reference), 200)

    current_sample_size = min(len(current), 100)

    reference_indices = np.random.choice(len(reference), reference_sample_size, replace=False)

    current_indices = np.random.choice(len(current), current_sample_size, replace=False)

    x = reference[reference_indices]
    y = current[current_indices]

    K_xx = rbf_kernel(x, x, sigma)

    K_yy = rbf_kernel(y, y, sigma)

    K_xy = rbf_kernel(x, y, sigma)

    # --------------------------------------------------------
    # Biased empirical MMD^2 estimator
    # --------------------------------------------------------

    mmd_squared = (K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())
    mmd_squared = max(float(mmd_squared), 0.0)

    return float(np.sqrt(mmd_squared))


def detect_distribution_shift(
    reference_distribution,
    current_distribution,
    tau_MMD
):

    shift_score = compute_MMD(reference_distribution, current_distribution)

    shift_flag = (shift_score > tau_MMD)

    return (shift_score, shift_flag)


# ============================================================
# 16. Shift-Risk Trigger
# ============================================================
'''
This preserves the two signals and returns separate trigger reasons rather than only a Boolean.
Now the controller can distinguish three situations:
'''
def evaluate_adaptation_signals(
    shift_score,
    risk_score,
    off_road,
    tau_MMD=TAU_MMD,
    tau_risk=TAU_RISK,
):
    # Signal A: distribution shift
    shift_flag = bool(shift_score > tau_MMD)

    # Signal B: predicted safety risk
    risk_flag = bool(risk_score > tau_risk)
    off_road_flag = bool(off_road)

    return {
        "shift_score": float(shift_score),
        "risk_score": float(risk_score),
        "shift_flag": shift_flag,
        "risk_flag": risk_flag,
        "off_road": off_road_flag,
        # Keep the causes separate.
        "revision_reasons": [
            reason
            for reason, active in (
                ("distribution_shift_flag", shift_flag),
                ("safety_risk_flag", risk_flag),
                ("off_road_flag", off_road)
            )
            if active
        ],
    }



# ============================================================
# 17. Environment Information             State_Encode
# ============================================================

def construct_environment_information(state, action, next_state_pred, shift_score, risk_score, min_ttc):


    """Build numeric and human-readable Highway driving context."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    next_state_pred = np.asarray(next_state_pred, dtype=np.float32).reshape(-1)

    vehicle_rows = state[:VEHICLE_STATE_DIM].reshape(NUM_VEHICLE_ROWS, VEHICLE_FEATURES)

    ego = vehicle_rows[0]
    neighbors = []
    for i, row in enumerate(vehicle_rows[1:], start=1):
        if row[0] > 0.5:
            neighbors.append({
                "slot": i,
                "relative_x_normalized": float(row[1]),
                "relative_y_normalized": float(row[2]),
                "relative_vx_normalized": float(row[3]),
                "relative_vy_normalized": float(row[4]),
            })

    risk = float(np.clip(risk_score, 0.0, 1.0))
    shift = float(max(0.0, shift_score))
    if risk >= 0.8:
        risk_text = "high predicted risk"
    elif risk >= 0.4:
        risk_text = "moderate predicted risk"
    else:
        risk_text = "low predicted risk"


    if min_ttc <= ttc_critical_threshold:
        ttc_risk_level = "critical"
    elif min_ttc <= ttc_danger_threshold:
        ttc_risk_level = "danger"
    elif min_ttc <= ttc_caution_threshold:
        ttc_risk_level = "caution"
    else:
        ttc_risk_level = "no_immediate_ttc_threat"

    ttc_information = {
        "minimum_ttc_seconds": float(min_ttc),
        "ttc_risk_level": ttc_risk_level,
        "critical_threshold_seconds": ttc_critical_threshold,
        "danger_threshold_seconds": ttc_danger_threshold,
        "caution_threshold_seconds": ttc_caution_threshold,
    }



    forward = [v for v in neighbors if v["relative_x_normalized"] > 0]
    if forward:
        closest = min(forward, key=lambda v: v["relative_x_normalized"])
        traffic_text = "a detected vehicle is ahead"
        if closest["relative_vx_normalized"] < float(ego[3]):
            traffic_text = "a detected vehicle ahead may be slower than the ego vehicle"
    elif neighbors:
        traffic_text = "nearby traffic is detected, with no observed vehicle ahead in the selected slots"
    else:
        traffic_text = "no nearby vehicle is present in the selected observation slots"

    context_text = (
        f"Highway driving scene: {traffic_text}; {risk_text}; "
        f"MMD shift score {shift:.3f}; "
        f"normalized acceleration command {float(action[0]):.2f}; "
        f"discrete lane intention {int(np.rint(action[1]))} "
        f"(0=left, 1=keep, 2=right)."
    )
    return {
        "environment": "highway-env",
        "ego_features": ego.tolist(),
        "speed_feature_normalized": float(state[VEHICLE_STATE_DIM]),
        "lane_feature_normalized": float(state[VEHICLE_STATE_DIM + 1]),
        "collision_indicator": float(state[VEHICLE_STATE_DIM + 2]),
        "nearby_vehicles": neighbors,
        "action": action.tolist(),
        "predicted_next_state": next_state_pred.tolist(),
        "shift_score": shift,
        "shift_detected": bool(shift > TAU_MMD),
        "risk_score": risk,
        "context_text": context_text,
        "ttc_information": ttc_information,
        "minimum_ttc_seconds": float(min_ttc),
        "ttc_risk_level": ttc_risk_level,
    }


# ============================================================
# 18. Executable AutoReward / LLM Interface
# ============================================================
# The AutoReward_LLM function is still a local, rule-based placeholder.
# It is not yet connected to a real generative LLM API
def AutoReward_LLM(driving_context, performance_feedback=None,):
    shift = bool(driving_context["shift_detected"])
    risk = float(driving_context["risk_score"])
    # TTC is intentionally not used to shape the reward. The independent
    # safety shield uses it to filter or modify unsafe actions.

    # Separate reward-design decisions.

    if risk >= 0.80:
        target_behavior = (
            "Make steady highway progress while maintaining "
            "safe spacing and smooth control. Prefer following "
            "a slower front vehicle by matching its speed when "
            "the current lane remains acceptable. Consider a "
            "lane change when the current lane materially limits "
            "progress or becomes unsafe, but change lanes only "
            "when the target lane is demonstrably safer and "
            "offers a useful improvement. Avoid unnecessary "
            "lane changes and never sacrifice safety for speed."
        )
        tracking_weight = 0.15
        safety_weight = 12.0
        action_weight = 0.01
        reward_mode = "safety"

    elif risk >= TAU_RISK:
        target_behavior = (
            "Maintain safe following distance and smooth "
            "control while making cautious progress."
        )
        tracking_weight = 0.18
        safety_weight = 10.0
        action_weight = 0.01
        reward_mode = "safety"

    elif shift:
        # Shift-specific response, even when risk is low.
        target_behavior = (
            "Adapt to changed traffic conditions, monitor "
            "nearby vehicles, maintain safe spacing, "
            "and make stable progress."
        )
        tracking_weight = 0.20
        safety_weight = 8.0
        action_weight = 0.01
        reward_mode = "shift_adaptation"

    else:
        target_behavior = (
            "Make steady highway progress while maintaining "
            "safe spacing and smooth control."
        )
        tracking_weight = INITIAL_TRACKING_WEIGHT
        safety_weight = INITIAL_COLLISION_PENALTY
        action_weight = 0.01
        reward_mode = "nominal"



    # Separate RL-performance feedback from shift and risk.
    if performance_feedback is not None:
        collision_rate = performance_feedback.get("collision_rate", 0.0)
        performance_degraded = performance_feedback.get("performance_degraded", False)

        if collision_rate >= 0.10:
            safety_weight = max(safety_weight, 3.0)
            reward_mode += "_collision_correction"

        if performance_degraded:
            # Ask for reward refinement; do not assume that
            # all poor performance is caused by distribution shift.
            target_behavior += (
                " Refine the reward design to improve recent "
                "task performance without sacrificing safety."
            )
            reward_mode += "_performance_refinement"

    return {
        "name": reward_mode,
        "target_behavior": target_behavior,
        "tracking_weight": tracking_weight,
        "safety_weight": safety_weight,
        "action_weight": action_weight,
        # Explicit, modest incentive for forward progress; safety penalties remain separate.
        "progress_weight": INITIAL_PROGRESS_WEIGHT,
        "target_speed": 25.0,  # m/s; tune to the scenario/speed limit
        "headway_weight": INITIAL_HEADWAY_WEIGHT,
        "headway_time_headway": DESIRED_TIME_HEADWAY,
        "headway_min_gap": MIN_FOLLOWING_GAP,
        "lane_change_cost": INITIAL_LANE_CHANGE_COST,
        "shift_score": float(driving_context["shift_score"]),
        "risk_score": risk,
        # New road-keeping constraints
        "lane_keeping_weight": lane_keeping_weight,
        "max_lateral_deviation": 2.0,
        "offroad_penalty": INITIAL_OFFROAD_PENALTY,
    }



# ============================================================
# 19. Construct Adaptive Reward
# ============================================================
def construct_reward(reward_specification, embedding_model):
    """
    Construct the semantic adaptive reward with explicit
    lane-keeping and off-road penalties.
    Semantic reward:
        r_semantic = (cos(e_behavior, e_target) + 1) / 2
    Total reward:
        r = tracking_weight * r_semantic
            + progress_weight * clip(ego_speed / target_speed, 0, 1)
            - headway_penalty
            - safety / intervention / control / lane / off-road penalties

    TTC is excluded from reward construction and remains in the independent
    execution-time safety shield.
    """

    target_text = reward_specification.get("target_behavior", reward_specification["name"],)
    target_embedding = embedding_model.encode(target_text)

    # Configurable road-keeping parameters.
    lane_keeping_weight = float(reward_specification.get("lane_keeping_weight"))
    max_lateral_deviation = max(float(reward_specification.get("max_lateral_deviation")), 1e-6,)
    offroad_penalty_magnitude = float(reward_specification.get("offroad_penalty"))



    def reward_function(
        behavior_description,
        safety_violation,
        action,
        lane_deviation,
        offroad,
        min_ttc,
        lane_change_allowed=True,
        ego_speed=0.0,
        safety_info=None,
        policy_sampled_action=None,
        reward_diagnostics=None,
    ):
        behavior_embedding = embedding_model.encode(behavior_description)

        semantic_reward = cosine_similarity(behavior_embedding, target_embedding,)

        # Map cosine similarity [-1, 1] to [0, 1].
        semantic_reward = (semantic_reward + 1.0) / 2.0

        # Penalize an observed collision/safety violation separately from shield use.
        # Routine speed matching is a controller action, not a safety violation.
        safety_penalty = (
            float(reward_specification.get("safety_weight", 1.0))
            if safety_violation
            else 0.0
        )
        safety_info = dict(safety_info or {})
        shield_penalty = float(
            safety_info.get(
                "reward_safety_penalty",
                SAFETY_OVERRIDE_REWARD_PENALTY
                if safety_info.get("emergency_braking", False)
                else LANE_REJECTION_REWARD_PENALTY
                if safety_info.get("lane_change_blocked", False)
                else 0.0,
            )
        )

        # Existing action/control penalty.
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        acceleration = float(np.clip(action[0], -1.0, 1.0))
        lane_action_id = int(np.clip(np.rint(action[1]), 0, 2))
        # Keep acceleration regularization very small: a large action penalty
        # would teach the policy to coast rather than make safe forward progress.
        control_penalty = float(reward_specification.get("action_weight", 0.01)) * abs(acceleration)
        lane_change_cost = max(0.0, float(reward_specification.get("lane_change_cost", INITIAL_LANE_CHANGE_COST)))
        control_penalty += lane_change_cost if lane_action_id != 1 else 0.0

        # Explicitly expose routine shield intervention to the reward. Without
        # this term, a raw +1.0 request filtered to -0.02 receives reward based
        # almost entirely on -0.02, hiding the policy's repeated over-request.
        sampled_for_reward = np.asarray(
            action if policy_sampled_action is None else policy_sampled_action,
            dtype=np.float32).reshape(-1)
        sampled_accel_for_reward = (float(np.clip(sampled_for_reward[0], -1.0, 1.0))
                                    if sampled_for_reward.size else acceleration)
        accel_intervention_magnitude = abs(sampled_accel_for_reward - acceleration)
        intervention_penalty = SHIELD_ACCEL_INTERVENTION_WEIGHT * min(accel_intervention_magnitude, 2.0)
        print(
            "[REWARD ACTION UNITS]",
            f"sampled_accel_normalized={sampled_accel_for_reward:+.4f}",
            f"executed_accel_normalized={acceleration:+.4f}",
            f"sampled_accel_physical_mps2={sampled_accel_for_reward * MAX_PHYSICAL_ACCELERATION:+.3f}",
            f"executed_accel_physical_mps2={acceleration * MAX_PHYSICAL_ACCELERATION:+.3f}",
            f"intervention_delta_normalized={accel_intervention_magnitude:.4f}",
            f"intervention_delta_physical_mps2={accel_intervention_magnitude * MAX_PHYSICAL_ACCELERATION:.3f}",
            "control_penalty_input_units=normalized_acceleration",
        )

        # No instantaneous TTC-triggered lane-change bonus. Lane changes
        # are learned through stable task objectives while the independent
        # safety shield continues to enforce TTC constraints.
        lane_change_bonus = 0.0



        # New lane-keeping penalty.
        lane_deviation = abs(float(lane_deviation))
        normalized_deviation = np.clip(lane_deviation / max_lateral_deviation, 0.0, 1.0,)
        lane_penalty = lane_keeping_weight * normalized_deviation


        # New explicit off-road penalty.
        road_penalty = offroad_penalty_magnitude if offroad else 0.0


        # TTC is not converted into a reward penalty. min_ttc remains in the
        # signature for compatibility and diagnostics; the safety shield still
        # uses TTC independently.
        min_ttc = float(min_ttc)
        ttc_penalty = 0.0

        # Reward progress explicitly, rather than relying on semantic text similarity.
        # Clip at target_speed so the policy is not encouraged to exceed it.
        ego_speed = max(0.0, float(ego_speed))
        target_speed = max(
            1e-6,
            float(reward_specification.get("target_speed", 25.0)),
        )
        progress_weight = max(
            0.0,
            float(reward_specification.get("progress_weight", 0.20)),
        )
        progress_ratio = float(np.clip(ego_speed / target_speed, 0.0, 1.0))
        progress_reward = progress_weight * progress_ratio

        # Headway shaping: penalize a short gap only while the ego is closing
        # on the lead vehicle (or the gap is approximately stable). If the lead
        # vehicle is pulling away, do not punish the ego merely for a currently
        # short gap: the gap is already recovering naturally.
        front_gap_for_reward = float(safety_info.get("front_gap", float("inf")))
        front_speed_for_reward = safety_info.get("front_speed", None)
        headway_time_headway = max(
            0.1, float(reward_specification.get("headway_time_headway", DESIRED_TIME_HEADWAY))
        )
        headway_min_gap = max(
            0.0, float(reward_specification.get("headway_min_gap", MIN_FOLLOWING_GAP))
        )
        # The TTC filter snapshot is pre-action, so use its matching ego
        # speed for gap/closing calculations. Progress still uses post-step speed.
        ego_speed_for_headway = max(
            0.0, float(safety_info.get("current_speed", ego_speed))
        )
        desired_headway_gap = headway_min_gap + headway_time_headway * ego_speed_for_headway
        front_closing_speed_for_reward = (
            ego_speed_for_headway - float(front_speed_for_reward)
            if front_speed_for_reward is not None else 0.0
        )
        headway_weight = max(0.0, float(reward_specification.get("headway_weight", INITIAL_HEADWAY_WEIGHT)))
        headway_shortfall = (
            float(np.clip(
                (desired_headway_gap - max(0.0, front_gap_for_reward))
                / max(desired_headway_gap, 1e-6),
                0.0, 1.0
            ))
            if front_speed_for_reward is not None and np.isfinite(front_gap_for_reward)
            else 0.0
        )
        lead_pulling_away_for_reward = front_speed_for_reward is not None and front_closing_speed_for_reward < -0.1
        headway_penalty = (
            headway_weight * headway_shortfall
            if not lead_pulling_away_for_reward else 0.0
        )

        # Rear TTC remains a safety-shield signal, not a reward penalty.
        rear_ttc_for_reward = float(safety_info.get("current_rear_ttc", float("inf")))
        rear_ttc_penalty = 0.0

        print(
            "[LONG-TERM OBJECTIVE]",
            f"desired_gap={desired_headway_gap:.3f}",
            f"front_gap={front_gap_for_reward:.3f}",
            f"front_closing_speed={front_closing_speed_for_reward:.3f}",
            f"lead_pulling_away={lead_pulling_away_for_reward}",
            f"headway_shortfall={headway_shortfall:.3f}",
            f"headway_penalty={headway_penalty:.3f}",
            f"rear_ttc_shield_only={rear_ttc_for_reward:.3f}",
        )

        print(
            f"[PROGRESS REWARD] speed={ego_speed:.3f} m/s, "
            f"target={target_speed:.3f} m/s, "
            f"ratio={progress_ratio:.3f}, "
            f"weight={progress_weight:.3f}, "
            f"reward={progress_reward:.4f}"
        )
        print("[TTC REWARD] disabled; TTC is handled by the independent safety shield.")
        reward = (
                float(reward_specification["tracking_weight"])
                * float(semantic_reward)
                + progress_reward
                - safety_penalty
                - shield_penalty
                - intervention_penalty
                - control_penalty
                - lane_penalty
                - road_penalty
                - headway_penalty
        )

        if reward_diagnostics is not None:
            reward_diagnostics.update({
                "semantic_reward": float(semantic_reward),
                "progress_reward": float(progress_reward),
                "safety_penalty": float(safety_penalty),
                "shield_penalty": float(shield_penalty),
                "intervention_penalty": float(intervention_penalty),
                "control_penalty": float(control_penalty),
                "lane_penalty": float(lane_penalty),
                "offroad_penalty": float(road_penalty),
                "ttc_penalty": 0.0,
                "headway_penalty": float(headway_penalty),
                "rear_ttc_penalty": 0.0,
                "lane_change_bonus": 0.0,
                "total_reward": float(reward),
            })

        print(
            f"[REWARD DEBUG] "
            f"semantic={semantic_reward:.3f}, "
            f"progress={progress_reward:.3f}, "
            f"speed={ego_speed:.2f}m/s, "
            f"safety={safety_penalty:.3f}, "
            f"shield={shield_penalty:.3f}, "
            f"action_intervention_penalty={intervention_penalty:.3f}, "
            f"accel_intervention={accel_intervention_magnitude:.3f}, "
            f"intervention={safety_info.get('intervention_reason', 'none')}, "
            f"lane={lane_penalty:.3f}, "
            f"offroad={road_penalty:.3f}, "
            f"ttc_penalty=0.000 (shield-only), "
            f"headway_penalty={headway_penalty:.3f}, "
            f"total={reward:.3f}"
        )


        return float(reward)

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
def update_policy(
    policy,
    state,
    action,
    reward,
    next_state,
    done,
    valid_action_mask=None,
    terminated=None,
    pre_tanh_accel=None,
):
    """
    One-step actor-critic update for a hybrid action:
      action[0] = tanh-squashed Gaussian acceleration
      action[1] = categorical lane ID (0, 1, or 2)
    """

    state_tensor = to_tensor(state).reshape(1, -1)
    next_state_tensor = to_tensor(next_state).reshape(1, -1)
    action_tensor = to_tensor(action).reshape(1, -1)
    reward_tensor = to_tensor([reward])

    # -------------------------------
    # Critic: one-step TD target
    # -------------------------------
    value = policy.value(state_tensor).reshape(-1)

    with torch.no_grad():
        next_value = policy.value(next_state_tensor).reshape(-1)

        # A true terminal state must not bootstrap.
        # A time-limit truncation can still bootstrap.
        is_terminal = (
            bool(done) if terminated is None
            else bool(terminated)
        )

        if is_terminal:
            target = reward_tensor
        else:
            target = reward_tensor + 0.99 * next_value

    advantage = target - value
    critic_loss = F.mse_loss(value, target)

    # -------------------------------
    # Actor: hybrid log probability
    # -------------------------------
    accel_dist, lane_dist = policy.action_distributions(state_tensor, valid_action_mask=valid_action_mask,)

    # Calculate log probability using the original Gaussian sample
    # whenever it is available.
    if pre_tanh_accel is None:
        # Compatibility fallback for existing callers.
        accel = action_tensor[:, 0].clamp(-0.999, 0.999)
        pre_tanh_tensor = (0.5 * (torch.log1p(accel) - torch.log1p(-accel))).unsqueeze(-1)
    else:
        pre_tanh_tensor = to_tensor(pre_tanh_accel).reshape(1, 1)

    accel_log_prob = accel_dist.log_prob(pre_tanh_tensor).squeeze(-1)

    # Jacobian correction for the tanh transformation.
    bounded_accel = action_tensor[:, 0].clamp(-1.0, 1.0)
    accel_log_jacobian = torch.log(1.0 - bounded_accel.pow(2) + 1e-6)
    accel_log_prob = accel_log_prob - accel_log_jacobian


    # Lane action is categorical, not continuous.
    lane_action = (
        action_tensor[:, 1]
        .round()
        .long()
        .clamp(0, 2)
    )

    lane_log_prob = lane_dist.log_prob(lane_action)

    # Joint log probability of the hybrid action.
    log_probability = accel_log_prob + lane_log_prob

    actor_loss = -(log_probability * advantage.detach()).mean()

    # -------------------------------
    # Joint actor-critic update
    # -------------------------------
    loss = actor_loss + critic_loss

    policy.optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0,)

    policy.optimizer.step()

    return {
        "loss": float(loss.item()),
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "advantage": float(advantage.detach().mean().item()),
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
# Embedding Model
# ============================================================

class RewardEmbeddingModel:
    """Semantic sentence embeddings shared by behavior and target text.

    Requires: pip install sentence-transformers
    The model returns 384-D normalized vectors, so their dot product is
    cosine similarity. No generative LLM is invoked by this encoder.
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM,
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if embedding_dim != 384:
            raise ValueError("all-MiniLM-L6-v2 produces 384-D embeddings; set EMBEDDING_DIM=384")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install the semantic encoder dependency with: "
                "pip install sentence-transformers"
            ) from exc
        self.encoder = SentenceTransformer(model_name, device=str(DEVICE))
        self.embedding_dim = 384

    def encode(self, text):
        if isinstance(text, (dict, list, tuple)):
            text = json.dumps(text, ensure_ascii=False, sort_keys=True)
        vector = self.encoder.encode(
            str(text), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vector, dtype=np.float32)


def cosine_similarity(
    embedding_a,
    embedding_b
):

    a = np.asarray(embedding_a, dtype=np.float32)
    b = np.asarray(embedding_b, dtype=np.float32)

    denominator = (np.linalg.norm(a) *np.linalg.norm(b))

    if denominator < 1e-8:
        return 0.0

    return float(np.dot(a, b) / denominator)



# ============================================================
# Behavior Description
# ============================================================

def describe_current_behavior(state, action, next_state, risk_score):
    """Create a grounded textual summary from the encoded Highway state."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    risk = float(np.clip(risk_score, 0.0, 1.0))
    vehicle_rows = state[:VEHICLE_STATE_DIM].reshape(NUM_VEHICLE_ROWS, VEHICLE_FEATURES)
    neighbors = [row for row in vehicle_rows[1:] if row[0] > 0.5]
    forward = [row for row in neighbors if row[1] > 0]
    if forward:
        closest = min(forward, key=lambda row: row[1])
        traffic = "a vehicle is observed ahead"
        if closest[3] < vehicle_rows[0, 3]:
            traffic = "a potentially slower vehicle is observed ahead"
    elif neighbors:
        traffic = "nearby vehicles are observed"
    else:
        traffic = "no nearby vehicles are observed in the encoded slots"
    risk_text = "high" if risk >= 0.8 else "moderate" if risk >= 0.5 else "low"
    accel = float(action[0]) if action.size else 0.0

    lane_action_id = (
        int(np.clip(np.rint(action[1]), 0, 2))
        if action.size > 1
        else 1
    )
    lane_action_delta = lane_action_index_to_delta(lane_action_id)
    lane_action_names = {
        -1: "request a left lane change (relative delta -1)",
         0: "keep the current lane (relative delta 0)",
         1: "request a right lane change (relative delta +1)",
    }

    lane_intention = lane_action_names[lane_action_delta]

    transition_change = float(np.linalg.norm(next_state[:VEHICLE_STATE_DIM] - state[:VEHICLE_STATE_DIM]))
    smoothness = "noticeable state change" if transition_change > 0.7 else "relatively small state change"

    return (
        f"Highway driving behavior: {traffic}; "
        f"predicted risk is {risk_text}; "
        f"normalized acceleration command {accel:.2f}; "
        f"lane intention: {lane_intention}; "
        f"the encoded traffic state shows {smoothness}."
    )





#**********************************************************************************************
# ============================================================
# 17. Save Complete Reference Checkpoint
# ============================================================

def save_checkpoint(
    checkpoint_path,
    policy,
    dynamics_model,
    risk_predictor,
    reference_distribution,

):

    checkpoint = {
        "policy": policy.state_dict(),
        "dynamics_model": dynamics_model.state_dict(),
        "risk_predictor": risk_predictor.state_dict(),
        "reference_distribution": reference_distribution
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(
        f"Reference checkpoint saved to "
        f"{checkpoint_path}"
    )



# ============================================================
# 18. Load Reference Checkpoint
# ============================================================

def load_checkpoint(
    path,
    policy,
    dynamics_model,
    risk_predictor,
):
    """
    Load the complete reference-training checkpoint.
    """

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    policy.load_state_dict(checkpoint["policy"])
    dynamics_model.load_state_dict(checkpoint["dynamics_model"])
    risk_predictor.load_state_dict(checkpoint["risk_predictor"])

    reference_distribution = checkpoint["reference_distribution"]

    # Reference models are frozen during comparison/evaluation.

    dynamics_model.eval()
    risk_predictor.eval()

    for param in dynamics_model.parameters():
        param.requires_grad = False
    for param in risk_predictor.parameters():
        param.requires_grad = False

    print(f"Checkpoint loaded from: {path}")

    return reference_distribution





def collect_checkpoint_data( env, policy, num_steps=1000 ):
    states = []
    actions = []
    next_states = []
    rewards = []
    safety_violations = []
    state = env.reset()
    policy.eval()
    for _ in range( num_steps ):
        action = policy.act( state )
        next_state, reward, done, info =  env.step(action)
        states.append( np.asarray( state, dtype=np.float32 ) )
        actions.append( np.asarray( action, dtype=np.float32 ) )
        next_states.append( np.asarray( next_state, dtype=np.float32 ) )
        rewards.append( float(reward) )
        safety_violations.append( bool( info.get( "safety_violation", False ) ) )
        state = next_state
        if done:
            state = env.reset()

    return { "states": np.asarray(states),
             "actions": np.asarray(actions),
             "next_states": np.asarray(next_states),
             "rewards": np.asarray(rewards),
             "safety_violations": np.asarray( safety_violations ) }


def construct_checkpoint_distribution( data ):
    return np.concatenate(
        [ data["states"],
          data["actions"],
          data["next_states"]
          ], axis=1 )




# ============================================================
# 23. Main AutoReward OOD Adaptive RL
# ============================================================

def AROOD(simulator, policy):

    try:
        with open(DIAGNOSTIC_LOG_PATH, "w", encoding="utf-8") as log_file:
            log_file.write("")
    except OSError as exc:
        print("[DIAGNOSTIC LOG WARNING]", f"cannot initialize {DIAGNOSTIC_LOG_PATH}: {exc}")

    reward_List = []
    reward_per_step_List = []
    adaptation_count_List = []
    time_List = []
    risk_List = []
    collision_List = []


    # ========================================================
    # PHASE I
    # Reference Data Collection
    # ========================================================

    print("\n" + "=" * 60)
    print("PHASE I: Collecting reference rollouts")
    print("=" * 60)

    rollouts = collect_rollouts(
        simulator=simulator,
        policy=policy,
        num_episodes=NUM_REFERENCE_EPISODES,
        max_steps=MAX_STEPS
    )

    print("\n=== Reference Rollout Structure ===")
    print("Number of rollouts:", len(rollouts))

    for i, rollout in enumerate(rollouts):
        print(f"Rollout {i}: {len(rollout)} transitions")

    print("Total transitions:", sum(len(r) for r in rollouts))
    print("Required risk horizon:", RISK_HORIZON)
    print("Usable rollouts:", sum( len(r) > RISK_HORIZON for r in rollouts))



    # ========================================================
    # PHASE II
    # Dynamics Model
    # ========================================================

    print("\n" + "=" * 60)
    print("PHASE II: Training DynamicsModel")
    print("=" * 60)

    dynamics_model = train_dynamics_model(
            rollouts=rollouts,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM
        )

    # ========================================================
    # PHASE III
    # Risk Predictor
    # ========================================================
    print("\n" + "=" * 60)
    print("PHASE III: Training RiskPredictor")
    print("=" * 60)

    risk_model = train_risk_predictor(
            rollouts=rollouts,
            dynamics_model=dynamics_model,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            horizon_H=RISK_HORIZON
        )

    # ========================================================
    # PHASE IV
    # Construct reference distribution
    # ========================================================

    print("\n" + "=" * 60)
    print("PHASE IV: Constructing reference distribution")
    print("=" * 60)

    reference_distribution = construct_reference_distribution(rollouts)

    print("Reference samples:", len(reference_distribution))
    print("[DIAGNOSTIC OUTPUT]", os.path.abspath(DIAGNOSTIC_LOG_PATH))

    current_buffer = deque(maxlen=DISTRIBUTION_WINDOW)

    offroad_fraction = sum(item.get("offroad", False) for item in current_buffer) / max(len(current_buffer), 1)
    print(f"[BUFFER] offroad_fraction={offroad_fraction:.3f}")


    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding_model = RewardEmbeddingModel(EMBEDDING_DIM)

    # ========================================================
    # PHASE V
    # Online Adaptive RL
    # ========================================================

    print("\n" + "=" * 60)
    print("PHASE V: Online adaptive RL")
    print("=" * 60)


    # These variables: use the episode-level outcomes to decide whether to refine the reward specification.
    recent_returns = deque(maxlen=10)
    recent_collision_rates = deque(maxlen=10)
    previous_return_baseline = None
    last_revision_step = -10 ** 9
    global_step = 0
    MIN_REVISION_INTERVAL = 20
    #--------------------------------------------------------------------------------------------------


    reward_specification = {
        "name": "nominal_reward",
        "target_behavior": (
            "Make steady highway progress while maintaining "
            "safe spacing and smooth control. Prefer following "
            "a slower front vehicle by matching its speed when "
            "the current lane remains acceptable. Consider a "
            "lane change when the current lane materially limits "
            "progress or becomes unsafe, but change lanes only "
            "when the target lane is demonstrably safer and "
            "offers a useful improvement. Avoid unnecessary "
            "lane changes and never sacrifice safety for speed."
        ),
        "tracking_weight": INITIAL_TRACKING_WEIGHT,
        "safety_weight": INITIAL_COLLISION_PENALTY,
        "action_weight": 0.01,
        "progress_weight": INITIAL_PROGRESS_WEIGHT,
        "target_speed": 25.0,
        "headway_weight": INITIAL_HEADWAY_WEIGHT,
        "headway_time_headway": DESIRED_TIME_HEADWAY,
        "headway_min_gap": MIN_FOLLOWING_GAP,
        "rear_ttc_weight": INITIAL_REAR_TTC_WEIGHT,
        "lane_change_cost": INITIAL_LANE_CHANGE_COST,
        "shift_score": 0.0,
        "risk_score": 0.0,
        # Lane-keeping and road-boundary constraints
        "lane_keeping_weight": lane_keeping_weight,
        "max_lateral_deviation": 2.0,
        "offroad_penalty": INITIAL_OFFROAD_PENALTY,
        "ttc_weight": INITIAL_TTC_WEIGHT,
        "ttc_critical_threshold": ttc_critical_threshold,
        "ttc_safe_threshold": ttc_caution_threshold,
    }

    reward_function = construct_reward(reward_specification, embedding_model,)




    for episode in range(NUM_ONLINE_EPISODES):
        simulator._trace_phase = "online"
        simulator._trace_episode = episode + 1
        print("----------------------------------------------------&&&&&&&&&&&&&&&&&&&&&&&& epi ",episode)
        start = time.time()
        episode_collision_count = 0

        #=============================================================================================
        #At the beginning of each online episode, before the step loop, construct feedback from completed episodes:
        performance_feedback = {
            "mean_return": (float(np.mean(recent_returns))
                if recent_returns else None),
            "collision_rate": (float(np.mean(recent_collision_rates))
                if recent_collision_rates else 0.0),
            "performance_degraded": (previous_return_baseline is not None
                    and bool(recent_returns)
                    and float(np.mean(recent_returns)) < previous_return_baseline)}
        # This uses previously observed RL outcomes, rather than the current step's reward, to avoid making the reward specification chase every immediate reward fluctuation.
        #===============================================================================================


        # Gradually shift traffic density for a controlled OOD experiment.
        # The reference rollouts are collected before this schedule starts.
        shift_level = episode / max(NUM_ONLINE_EPISODES - 1, 1)
        print("shift_level : ", shift_level)
        if hasattr(simulator, "set_distribution_shift"):
            simulator.set_distribution_shift(shift_level)


        state = simulator.reset()

        episode_reward = 0.0
        episode_reward_component_sums = Counter()
        adaptation_count = 0
        risk_score_count = 0

        # Diagnostics for separating policy behavior from safety-shield effects.
        policy_step_count = 0
        shield_intervention_count = 0
        strong_braking_request_count = 0
        sampled_accel_sum = 0.0
        online_lane_counters = make_lane_request_counters()
        online_seen_completed = {
            "LEFT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_LEFT", 0)),
            "RIGHT": int(getattr(simulator, "_lane_change_stats", {}).get("completed_RIGHT", 0)),
        }
        actor_loss_values = []
        advantage_values = []


        for t in range(MAX_STEPS):

            # =================================================
            # STEP 1
            # Policy
            # =================================================

            valid_mask = simulator.get_valid_lane_action_mask()
            simulator._trace_valid_action_mask = np.asarray(valid_mask, dtype=bool).tolist()
            sampled_action, pre_tanh_accel = policy.act(
                state,
                deterministic=False,
                valid_action_mask=valid_mask,
                return_aux=True,
            )
            sampled_action = np.asarray(sampled_action, dtype=np.float32,).reshape(-1)
            lane_policy_diagnostic(
                policy, state, valid_mask, sampled_action,
                phase="online", episode=episode + 1, step=t + 1,
            )


            filtered_action, safety_info = simulator.filter_ttc_action(sampled_action)
            record_lane_request_decision(online_lane_counters, sampled_action, valid_mask, safety_info)

            sampled_accel = float(sampled_action[0])
            policy_step_count += 1
            sampled_accel_sum += sampled_accel
            strong_braking = sampled_accel <= -0.80
            strong_braking_request_count += int(strong_braking)
            shield_intervened = not np.allclose(sampled_action, filtered_action, atol=1e-5)
            shield_intervention_count += int(shield_intervened)

            print(
                "[POLICY ACTION DIAGNOSTIC]",
                f"sampled_accel={sampled_accel:.4f}",
                f"strong_braking_request={strong_braking}",
                f"filtered_accel={float(filtered_action[0]):.4f}",
                f"shield_intervened={shield_intervened}",
                f"pre_tanh_accel={float(np.asarray(pre_tanh_accel).reshape(-1)[0]):.4f}",
            )
            print("[FILTER RECEIVED]", filtered_action.tolist())

            '''
            print(
                "[ACTION CONSISTENCY]",
                f"sampled={sampled_action.tolist()}",
                f"filtered={filtered_action.tolist()}",
                f"override={safety_info['overridden']}",
                f"reasons="
                f"emergency:{safety_info['emergency_braking']},"
                f"lane_blocked:{safety_info['lane_change_blocked']},"
                f"speed_limit:{safety_info.get('speed_limit_override', False)}",
            )
            '''


            # Canonical high-level action for prediction and execution.
            reward_action = filtered_action.copy()

            next_state_pred = dynamics_model.predict(state, reward_action)

            next_state, simulator_reward, done, step_info = simulator.step(
                reward_action,
                apply_safety_filter=False,
                sampled_action=sampled_action,
                safety_info=safety_info,
            )
            sync_lane_completion_counters(online_lane_counters, simulator, online_seen_completed)

            # Reconcile the requested high-level action with the action
            # actually accepted by the simulator adapter.
            actual_policy_action = np.asarray(step_info["filtered_policy_action"], dtype=np.float32,).copy()

            if not np.allclose(reward_action, actual_policy_action):
                print(
                    "[ACTION MISMATCH]",
                    f"predicted_action={reward_action.tolist()}",
                    f"adapter_action={actual_policy_action.tolist()}",
                )

                # Recompute prediction if the adapter had to adjust the action.
                reward_action = actual_policy_action
                next_state_pred = dynamics_model.predict(state, reward_action)

            # Use the reconciled action for risk prediction.
            risk = risk_model.predict(state, reward_action, next_state_pred)

            # Reward TTC is measured after the action has been executed, not
            # from the stale pre-step scene.
            min_ttc, closing_vehicle_count = calculate_min_ttc(simulator)
            print("[POST-STEP TTC FOR REWARD]", f"min_ttc={min_ttc:.3f}",
                  f"closing_vehicle_count={closing_vehicle_count}")



            safety_violation = bool(step_info.get("safety_violation", False)) if isinstance(step_info, dict) else bool(step_info)
            episode_collision_count += int(safety_violation)

            # ========================================================
            # STEP 3B: Get actual lane and road information
            # ========================================================
            lane_info = simulator.get_lane_information()
            offroad = not lane_info["on_road"]
            lane_deviation = lane_info["lane_deviation"]

            if offroad:

                # Mark this transition as terminal for policy learning.
                done = True


            # =================================================
            # STEP 4
            # Current actual transition distribution
            # =================================================

            update_current_distribution(current_buffer, state, reward_action, next_state)
            #print("len(current_buffer) : ", len(current_buffer))

            # =================================================
            # STEP 5
            # MMD OOD Detection
            # =================================================
            MIN_MMD_SAMPLES = DISTRIBUTION_WINDOW

            if len(current_buffer) < MIN_MMD_SAMPLES:
                shift_score = 0.0
                shift_flag = False
                shift_ready = False
            else:
                shift_score, shift_flag = detect_distribution_shift(
                    reference_distribution,
                    list(current_buffer),
                    TAU_MMD
                )
                shift_ready = True


            # =================================================
            # STEP 6
            # Risk Prediction
            #
            # Q_risk =
            # =================================================

            #risk = risk_model.predict(state, action, next_state_pred)
            risk_score = float(risk.item())
            if risk_score > TAU_RISK:
                risk_score_count += 1


            #=========================================================
            # STEP 7: Evaluate independent signals
            #=========================================================
            signals = evaluate_adaptation_signals(
                shift_score=shift_score,
                risk_score=risk_score,
                off_road=offroad
            )

            revision_reasons = list(signals["revision_reasons"])


            #==============================================================
            # STEP 8: Include independent RL-performance feedback
            #==============================================================

            if performance_feedback["performance_degraded"]:
                revision_reasons.append("poor_rl_performance")

            if performance_feedback["collision_rate"] >= 0.10:
                revision_reasons.append("high_observed_collision_rate")


            #================================================================
            # STEP 9: Decide whether to revise the reward.
            # The signals remain separate; this is an OR-based controller.
            #================================================================
            revision_requested = bool(revision_reasons)
            cooldown_ready = (global_step - last_revision_step >= MIN_REVISION_INTERVAL)
            adaptation_trigger = (revision_requested and cooldown_ready)

            if adaptation_trigger:
                driving_context = construct_environment_information(
                    state,
                    reward_action,
                    next_state_pred,
                    shift_score,
                    risk_score,
                    min_ttc
                )

                # Pass independent feedback channels to the reward designer.
                reward_specification = AutoReward_LLM(driving_context, performance_feedback=performance_feedback,)

                reward_function = construct_reward(reward_specification, embedding_model,)

                last_revision_step = global_step
                adaptation_count += 1

                print(
                    f"[Episode {episode + 1:03d} | Step {t:03d}] "
                    f"REVISION | reasons={revision_reasons} | "
                    f"shift={shift_score:.4f} | "
                    f"risk={risk_score:.4f} | "
                    f"mode={reward_specification['name']}"
                )

            # Otherwise, keep the previously generated reward function.
            # Do not call AutoReward_LLM() again here.

            # Continue with the existing behavior-description and
            # compute_reward() calls using reward_function.

            global_step += 1




#--------------------------------------------------------------------------
            # ------------------------------------------------
            # 10. Generate semantic behavior description
            # ------------------------------------------------

            behavior_description = describe_current_behavior(
                    state,
                    reward_action,
                    next_state,
                    risk_score
                )


            # ------------------------------------------------
            # 11. Embedding + cosine-similarity reward
            # ------------------------------------------------

            lane_change_allowed = (
                    not safety_info["lane_change_blocked"]
                    and not safety_info["emergency_braking"])

            print(
                "[POLICY LEARNING ACTION]",
                f"sampled={np.asarray(sampled_action).tolist()}",
                f"reward_action={np.asarray(reward_action).tolist()}",
                f"intervened={not np.allclose(sampled_action, reward_action)}",
            )

            # Read speed once from the live ego vehicle and pass that exact value
            # through compute_reward -> reward_function. This diagnostic catches
            # a stale/wrong speed source or argument-plumbing bug.
            ego_vehicle_for_reward = simulator.env.unwrapped.vehicle
            ego_speed_for_reward = float(ego_vehicle_for_reward.speed)
            encoded_speed = float(np.asarray(next_state, dtype=np.float32)[VEHICLE_STATE_DIM]) * 40.0

            print(
                "[REWARD SPEED SOURCE]",
                f"vehicle_speed={ego_speed_for_reward:.6f}",
                f"next_state_context_speed={encoded_speed:.6f}",
                f"vehicle_id={id(ego_vehicle_for_reward)}",
                f"state_speed_context={float(np.asarray(next_state)[VEHICLE_STATE_DIM]):.6f}",
            )

            reward_components = {}
            reward = compute_reward(
                reward_function=reward_function,
                behavior_description=behavior_description,
                safety_violation=safety_violation,
                action=reward_action,
                lane_deviation=lane_deviation,
                offroad=offroad,
                min_ttc=min_ttc,
                lane_change_allowed=lane_change_allowed,
                ego_speed=ego_speed_for_reward,
                safety_info=safety_info,
                policy_sampled_action=sampled_action,
                reward_diagnostics=reward_components,
            )

            for component_name, component_value in reward_components.items():
                try:
                    episode_reward_component_sums[component_name] += float(component_value)
                except (TypeError, ValueError):
                    pass

            write_diagnostic_event(
                "training_step_reward", simulator,
                reward=float(reward),
                reward_components={key: float(value) for key, value in reward_components.items()
                                  if isinstance(value, (int, float, np.integer, np.floating))},
                speed_mps=float(ego_speed_for_reward),
                min_ttc=float(min_ttc),
                lane_deviation=float(lane_deviation),
                offroad=bool(offroad),
                safety_violation=bool(safety_violation),
                risk_score=float(risk_score),
                shift_score=float(shift_score),
                revision_reasons=list(revision_reasons),
                adaptation_trigger=bool(adaptation_trigger),
                reward_specification_name=str(reward_specification.get("name", "unknown")),
                sampled_action=np.asarray(sampled_action).tolist(),
                executed_action=np.asarray(reward_action).tolist(),
                safety_intervention_reason=str(safety_info.get("intervention_reason", "none")),
                lane_change_request_id=safety_info.get("lane_request_trace_id"),
                active_maneuver_id=getattr(simulator, "_active_lane_maneuver_id", None),
            )
            print(
                "[REWARD COMPONENT TRACE]",
                " ".join(f"{key}={value:+.4f}" for key, value in reward_components.items()),
            )

            print(
                "[SHIELD REWARD DIAGNOSTIC]",
                f"reason={safety_info.get('intervention_reason', 'none')}",
                f"emergency={bool(safety_info.get('emergency_braking', False))}",
                f"speed_match={bool(safety_info.get('speed_match_active', False))}",
                f"lane_blocked={bool(safety_info.get('lane_change_blocked', False))}",
                f"penalty={float(safety_info.get('reward_safety_penalty', 0.0)):.3f}",
                f"reward={reward:.3f}",
            )


#--------------------------------------------------------------------------



            # =================================================
            # STEP 10
            # RL Policy Update
            # =================================================

            update_stats = update_policy(
                policy,
                state,
                sampled_action,  # Original sampled action whose probability the actor computes
                reward,
                next_state,
                done,
                valid_action_mask=valid_mask,
                terminated=bool(step_info.get("terminated", False) or offroad),
                pre_tanh_accel=pre_tanh_accel,
            )
            actor_loss_values.append(float(update_stats["actor_loss"]))
            advantage_values.append(float(update_stats["advantage"]))

            print(
                "[POLICY UPDATE DIAGNOSTIC]",
                f"reward={float(reward):.4f}",
                f"advantage={float(update_stats['advantage']):+.6f}",
                f"actor_loss={float(update_stats['actor_loss']):+.6f}",
                f"critic_loss={float(update_stats['critic_loss']):.6f}",
                f"sampled_accel={float(sampled_action[0]):+.4f}",
                f"executed_accel={float(reward_action[0]):+.4f}",
                f"intervened={shield_intervened}",
            )

            # =================================================
            # STEP 11
            # Move to actual next state
            # =================================================

            state = next_state
            episode_reward += reward          #??????????????????????????

            # =================================================
            # STEP 12
            # Termination
            # =================================================

            if done:
                break

        time_one_episode = time.time() - start
        print('time one episode: ', time_one_episode)
        time_List.append(time_one_episode)
        reward_List.append(episode_reward)
        reward_per_step = float(episode_reward / max(policy_step_count, 1))
        reward_per_step_List.append(reward_per_step)
        write_diagnostic_event(
            "episode_summary", simulator,
            episode_return=float(episode_reward),
            mean_reward_per_step=float(reward_per_step),
            steps=int(policy_step_count),
            collisions=int(episode_collision_count),
            collision_rate=float(episode_collision_count / max(t + 1, 1)),
            adaptations=int(adaptation_count),
            mean_actor_loss=float(np.mean(actor_loss_values)) if actor_loss_values else 0.0,
            mean_advantage=float(np.mean(advantage_values)) if advantage_values else 0.0,
            reward_component_sums={key: float(value) for key, value in episode_reward_component_sums.items()},
            lane_change_stats=dict(getattr(simulator, "_lane_change_stats", {})),
            lane_request_counters={
                direction: dict(values)
                for direction, values in online_lane_counters.items()
            },
            active_maneuver_at_episode_end=bool(getattr(simulator, "_lane_change_active", False)),
        )
        adaptation_count_List.append(adaptation_count)
        risk_List.append(risk_score_count)
        collision_List.append(episode_collision_count)

        episode_collision_rate = (episode_collision_count / max(t + 1, 1))
        mean_sampled_accel = sampled_accel_sum / max(policy_step_count, 1)
        strong_braking_rate = strong_braking_request_count / max(policy_step_count, 1)
        shield_intervention_rate = shield_intervention_count / max(policy_step_count, 1)
        mean_actor_loss = float(np.mean(actor_loss_values)) if actor_loss_values else 0.0
        mean_advantage = float(np.mean(advantage_values)) if advantage_values else 0.0

        sync_lane_completion_counters(online_lane_counters, simulator, online_seen_completed)
        if bool(getattr(simulator, "_lane_change_active", False)):
            write_diagnostic_event(
                "lane_maneuver_episode_end_unresolved", simulator,
                maneuver_id=getattr(simulator, "_active_lane_maneuver_id", None),
                request_id=getattr(simulator, "_active_lane_request_trace_id", None),
                direction=(
                    "LEFT" if simulator._lane_change_source_index is not None
                    and int(simulator.target_lane_index[2]) > int(simulator._lane_change_source_index[2])
                    else "RIGHT" if simulator._lane_change_source_index is not None else "UNKNOWN"
                ),
                source_lane=getattr(simulator, "_lane_change_source_index", None),
                target_lane=getattr(simulator, "target_lane_index", None),
                elapsed_steps=(
                    int(simulator.step_count - simulator._lane_change_started_step)
                    if getattr(simulator, "_lane_change_started_step", None) is not None else None
                ),
                outcome="ACTIVE_AT_EPISODE_END",
            )
        print(
            "[LANE CHANGE EPISODE SUMMARY]",
            f"requested={getattr(simulator, '_lane_change_stats', {}).get('requested', 0)}",
            f"accepted={getattr(simulator, '_lane_change_stats', {}).get('accepted', 0)}",
            f"rejected={getattr(simulator, '_lane_change_stats', {}).get('rejected', 0)}",
            f"completed={getattr(simulator, '_lane_change_stats', {}).get('completed', 0)}",
            f"active_at_episode_end={getattr(simulator, '_lane_change_active', False)}",
            f"active_maneuver_id={getattr(simulator, '_active_lane_maneuver_id', None)}",
            f"target_lane_at_episode_end={getattr(simulator, 'target_lane_index', None)}",
            f"started={getattr(simulator, '_lane_change_stats', {}).get('started', 0)}",
            f"timed_out={getattr(simulator, '_lane_change_stats', {}).get('timed_out', 0)}",
        )
        print_lane_request_counters("ONLINE EPISODE", online_lane_counters)

        print(
            "[EPISODE POLICY DIAGNOSTICS]",
            f"steps={policy_step_count}",
            f"mean_sampled_accel={mean_sampled_accel:+.4f}",
            f"strong_braking_rate={strong_braking_rate:.1%}",
            f"shield_intervention_rate={shield_intervention_rate:.1%}",
            f"mean_actor_loss={mean_actor_loss:+.6f}",
            f"mean_advantage={mean_advantage:+.6f}",
        )

        recent_returns.append(episode_reward)
        recent_collision_rates.append(episode_collision_rate)



        rolling_window = min(10, len(reward_List))
        rolling_mean_return = float(np.mean(reward_List[-rolling_window:]))
        rolling_mean_step_reward = float(np.mean(reward_per_step_List[-rolling_window:]))
        print(
            f"Episode "
            f"{episode + 1:03d}/"
            f"{NUM_ONLINE_EPISODES} | "
            f"return={episode_reward:.3f} | "
            f"mean_step_reward={reward_per_step:+.4f} | "
            f"rolling{rolling_window}_return={rolling_mean_return:.3f} | "
            f"rolling{rolling_window}_step_reward={rolling_mean_step_reward:+.4f} | "
            f"adaptations={adaptation_count}"
        )

    # Reward-component sums, actor/critic losses, and lane-maneuver outcomes
    # are recorded in DIAGNOSTIC_LOG_PATH for post-run convergence analysis.
    episode_reward_List = list(range(len(reward_List)))
    fig_path = reward_path
    plt.plot(episode_reward_List, reward_List, alpha=0.35, label="raw episode return")
    if len(reward_List) >= 3:
        window = min(10, len(reward_List))
        smoothed_returns = np.convolve(
            np.asarray(reward_List, dtype=np.float64),
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )
        plt.plot(
            np.arange(window - 1, len(reward_List)),
            smoothed_returns,
            linewidth=2.0,
            label=f"{window}-episode moving average",
        )
    plt.xlabel('episode')
    plt.ylabel('total episode return')
    plt.title('Training return (raw and smoothed)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    # Also plot mean reward per environment step. This separates learning
    # trends from variation caused by episodes ending at different lengths.
    plt.figure()
    plt.plot(episode_reward_List, reward_per_step_List, alpha=0.35, label="raw mean reward/step")
    if len(reward_per_step_List) >= 3:
        window = min(10, len(reward_per_step_List))
        smoothed_step_rewards = np.convolve(
            np.asarray(reward_per_step_List, dtype=np.float64),
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )
        plt.plot(
            np.arange(window - 1, len(reward_per_step_List)),
            smoothed_step_rewards,
            linewidth=2.0,
            label=f"{window}-episode moving average",
        )
    plt.xlabel('episode')
    plt.ylabel('mean reward per step')
    plt.title('Training reward per step (raw and smoothed)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_path.replace(".png", "_per_step.png"))
    plt.close()

    episode_adapt_List = list(range(len(adaptation_count_List)))
    fig_path = adaptation_path
    plt.plot(episode_adapt_List, adaptation_count_List)
    plt.xlabel('episode')
    plt.ylabel('adaptation count')
    plt.title('adaptation count in episodes')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()


    plotfile = open(plot_path, "w")
    plotfile.writelines(str(reward_List))
    plotfile.write('\n')
    plotfile.writelines(str(time_List))
    plotfile.write('\n')
    plotfile.writelines(str(adaptation_count_List))
    plotfile.write('\n')
    plotfile.writelines(str(risk_List))
    plotfile.write('\n')
    plotfile.writelines(str(collision_List))
    plotfile.write('\n')
    plotfile.writelines(str(recent_collision_rates))


    return {
        "policy": policy,
        "dynamics_model": dynamics_model,
        "risk_model": risk_model,
        "reference_distribution": reference_distribution
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






# ============================================================
# 24. Main Entry Point
# ============================================================

def main():
    tot_time = 0
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Selected DEVICE:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\n" + "=" * 100)

    print("AROOD:")

    print("=" * 30)
    print(f"State dimension : {STATE_DIM}")
    print(f"Action dimension: {ACTION_DIM} " "(normalized acceleration, discrete lane ID)")
    print(f"MMD threshold   : {TAU_MMD}")
    print(f"Risk threshold  : {TAU_RISK}")


    # --------------------------------------------------------
    # Initialize simulator
    # --------------------------------------------------------

    simulator = create_highway_simulator()
    #inspect_observation_encoding(simulator)
    simulator.reset(seed=SEED)


    # --------------------------------------------------------
    # Initialize policy
    # --------------------------------------------------------

    policy = create_initial_policy(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            scale_of_steering=steering_scale
        )
    check_model_device(policy, "Policy")

    # --------------------------------------------------------
    # Run complete algorithm
    # --------------------------------------------------------


    results = AROOD(
            simulator=simulator,
            policy=policy
        )

    trained_policy = results["policy"]

    # --------------------------------------------------------
    # Save final policy
    # --------------------------------------------------------

    save_policy(
        trained_policy,
        MODEL_PATH
    )

    dynamics_model = results["dynamics_model"]
    risk_predictor = results["risk_model"]
    reference_distribution = results["reference_distribution"]
    save_checkpoint(checkpoint_path, trained_policy, dynamics_model, risk_predictor, reference_distribution)

    check_model_device(policy, "Policy")
    check_model_device(dynamics_model, "DynamicsModel")
    check_model_device(risk_predictor, "RiskPredictor")

    print("\n" + "=" * 30)

    print(
        "Training completed successfully."
    )

    print("=" * 100)


# ============================================================
# 25. Python Entry Point
# ============================================================

if __name__ == "__main__":

    main()

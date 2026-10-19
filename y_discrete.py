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

import gymnasium as gym
import highway_env

# ============================================================
# 1. Configuration
# ============================================================
steering_scale = 0.01
offroad_penalty = 0.5

policy_frequency = 5
simulation_frequency = 5

lane_keeping_weight = 0.1

ttc_weight = 2.0
ttc_critical_threshold = 2.0
ttc_danger_threshold = 3.0
ttc_caution_threshold = 4.0
# Execution-time safety shield (TTC is measured in seconds).
SAFETY_OVERRIDE_TTC = 4.0       # Emergency braking threshold
LANE_CHANGE_MIN_TTC = 4.0       # Reject a new lane change below this TTC
SAFETY_OVERRIDE_REWARD_PENALTY = 1.0  # Extra learning signal when the shield intervenes TTC safety constants




STATE_DIM = 32
ACTION_DIM = 2

HIDDEN_DIM = 128

NUM_REFERENCE_EPISODES = 40
NUM_ONLINE_EPISODES = 200

MAX_STEPS = 100

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

MODEL_PATH = "adaptive_policy.pt"


EMBEDDING_DIM = 384
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


reward_path = 'figures/' + 'Re_7' + '.png'
adaptation_path = 'figures/' + 'Ad_7' + '.png'
plot_path = 'figures/' + 'plot_7' + '.txt'



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

    def act(self, state, deterministic=False, valid_action_mask=None):
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

        return (
            action.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
        )

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





# ============================================================
# 6. Highway-env Simulator Adapter
# ============================================================

class HighwaySimulator:
    """Highway-env wrapper with the legacy four-value step API.

    State is a fixed 32-D vector: four 7-feature vehicle rows plus
    four context values.
    Policy action is [normalized acceleration, discrete lane ID]:
    0=left, 1=keep, 2=right.

    Highway receives [physical acceleration, normalized steering].
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 max_steps=MAX_STEPS, render_mode=None, vehicles_count=4):
        if gym is None or highway_env is None:
            raise ImportError(
                "Highway-env dependencies are missing. Install with: "
                "pip install highway-env gymnasium sentence-transformers"
            )
        if state_dim != 32 or action_dim != 2:
            raise ValueError("HighwaySimulator expects STATE_DIM=32 and ACTION_DIM=2")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0
        self.render_mode = render_mode
        self.vehicles_count = vehicles_count
        self.env = gym.make(
            "highway-v0",
            render_mode=render_mode,
            config={
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 4,
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
                "vehicles_count": vehicles_count,
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


        rows = np.zeros((4, 7), dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] < 7:
            raise ValueError(f"Unexpected Highway observation shape: {obs.shape}")
        n = min(4, obs.shape[0])
        rows[:n] = obs[:n, :7]

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
        """Change traffic density between episodes for controlled OOD tests.

        level=0 uses the reference density; level=1 increases the vehicle
        count. The new setting takes effect on the next reset.
        """
        level = float(np.clip(level, 0.0, 1.0))
        target_count = int(round(15 + 10 * level))
        self.env.unwrapped.config["vehicles_count"] = target_count



    def reset(self, seed=None):

        print("\n=== RESET BEFORE ===")
        print("time:", getattr(self.env.unwrapped, "time", None))
        print("steps:", getattr(self.env.unwrapped, "steps", None))

        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)

        print("=== RESET AFTER ===")
        print("time:", getattr(self.env.unwrapped, "time", None))
        print("steps:", getattr(self.env.unwrapped, "steps", None))
        print("info:", info)


        self.step_count = 0

        # Store the target lane once at the beginning of this episode.
        # Do NOT recalculate it at every time step.
        base_env = self.env.unwrapped
        ego_vehicle = base_env.vehicle

        self.target_lane_index = ego_vehicle.lane_index
        self._lane_change_source_index = None
        self._lane_change_active = False
        self._lane_change_cooldown = 0

        print(
            "[TARGET LANE]",
            self.target_lane_index,
            "initial_position=",
            np.asarray(ego_vehicle.position).round(3),
        )

        self.state = self._encode_state(obs)
        return self.state.copy()




    def get_valid_lane_action_mask(self):
        """
        Action IDs:
            0 = left  -> lane index + 1
            1 = keep  -> no lane change
            2 = right -> lane index - 1
        Returns a Boolean mask in action-ID order.
        """
        road = self.env.unwrapped.road

        # During a lane change or cooldown, only "keep" is meaningful.
        if self._lane_change_active or self._lane_change_cooldown > 0:
            return np.array([False, True, False], dtype=bool)

        start_node, end_node, target_lane_id = self.target_lane_index
        available_lanes = road.network.graph[start_node][end_node]
        valid_ids = set(range(len(available_lanes)))

        left_lane = int(target_lane_id) + 1
        right_lane = int(target_lane_id) - 1
        return np.array([
            left_lane in valid_ids,  # action 0
            True,  # action 1
            right_lane in valid_ids,  # action 2
        ], dtype=bool)


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

            # -----------------------------------------------------
            # A. Vehicle ahead of the ego vehicle.
            # -----------------------------------------------------
            if center_gap > 0.0:

                bumper_gap = center_gap - half_length_sum
                closing_speed = ego_speed - other_speed

                # A non-positive clearance is an immediate hazard.
                if bumper_gap <= 0.0:
                    front_ttc = 0.0

                # Only calculate future TTC if the ego is closing
                # on the vehicle ahead.
                elif closing_speed > 0.1:
                    front_ttc = bumper_gap / closing_speed

                else:
                    front_ttc = float("inf")

                lane_ttc[other_lane]["front_ttc"] = min(lane_ttc[other_lane]["front_ttc"], front_ttc,)

            # -----------------------------------------------------
            # B. Vehicle behind the ego vehicle.
            # -----------------------------------------------------
            elif center_gap < 0.0:

                rear_center_gap = -center_gap
                bumper_gap = rear_center_gap - half_length_sum

                # Positive means the rear vehicle is catching up.
                rear_closing_speed = other_speed - ego_speed

                # A non-positive clearance is an immediate hazard.
                if bumper_gap <= 0.0:
                    rear_ttc = 0.0

                elif rear_closing_speed > 0.1:
                    rear_ttc = bumper_gap / rear_closing_speed

                else:
                    rear_ttc = float("inf")

                lane_ttc[other_lane]["rear_ttc"] = min(lane_ttc[other_lane]["rear_ttc"], rear_ttc,)

            # A vehicle with essentially identical longitudinal
            # position is not assigned a front/rear direction here.
            # If such a case occurs, inspect it as a possible overlap.

            # Optional diagnostic output for TTC calculations.
            if center_gap != 0.0:
                print(
                    "[TTC VEHICLE CHECK]",
                    f"ego_lane={ego_lane}",
                    f"other_lane={other_lane}",
                    f"center_gap={center_gap:.2f}",
                    f"bumper_gap={abs(center_gap) - half_length_sum:.2f}",
                    f"ego_speed={ego_speed:.2f}",
                    f"other_speed={other_speed:.2f}",
                    f"front_ttc={lane_ttc[other_lane]['front_ttc']:.3f}",
                    f"rear_ttc={lane_ttc[other_lane]['rear_ttc']:.3f}",
                )

        # ---------------------------------------------------------
        # C. Calculate the minimum of front and rear TTC per lane.
        # ---------------------------------------------------------
        for lane_id, values in lane_ttc.items():
            values["min_ttc"] = min(values["front_ttc"], values["rear_ttc"],)

        return lane_ttc




    def filter_ttc_action(self, sampled_action):
        """
        Apply TTC-based emergency braking and lane-change safety checks.
        Action representation:
            action[0] = normalized acceleration in [-1, 1]
            action[1] = discrete lane action:
                        0 = LEFT
                        1 = KEEP
                        2 = RIGHT
        Requires _min_ttc_by_lane() to return:
            {lane_id: {
                    "front_ttc": float,
                    "rear_ttc": float,
                    "min_ttc": float,}}
        Returns:
            filtered_action: np.ndarray
            safety_info: dict
        """

        sampled_action = np.asarray(sampled_action, dtype=np.float32).reshape(-1)
        print(
            "[POLICY ACTION DEBUG]",
            f"raw_acceleration={float(sampled_action[0]):.4f}",
            f"raw_lane_action={float(sampled_action[1]):.4f}",
        )


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

        current_front_ttc = float(current_values["front_ttc"])
        current_rear_ttc = float(current_values["rear_ttc"])

        current_lane_ttc = min(current_front_ttc, current_rear_ttc,)

        # The current lane is always relevant.
        front_ttc_values = [current_front_ttc]
        # During a lane change, monitor both the source and target lanes.
        if self._lane_change_active:
            for lane_index in (self._lane_change_source_index, self.target_lane_index,):
                if lane_index is None:
                    continue
                lane_id = int(lane_index[2])
                lane_values = lane_ttc_current.get(lane_id)
                if lane_values is not None:
                    front_ttc_values.append(float(lane_values["front_ttc"]))

        min_relevant_front_ttc = min(front_ttc_values)
        emergency_braking = min_relevant_front_ttc < emergency_threshold

        print(
            "[TTC FILTER]",
            f"current_lane={current_lane}",
            f"front_ttc={current_front_ttc:.3f}",
            f"critical_threshold={ttc_critical_threshold:.3f}",
            f"emergency_threshold={emergency_threshold:.3f}",
            f"emergency={emergency_braking}",
        )

        if emergency_braking:
            filtered_action[0] = -1.0

        # ---------------------------------------------------------
        # 2. Decode the discrete lane action correctly.
        # ---------------------------------------------------------
        requested_lane_action = int(np.clip(np.rint(sampled_action[1]), 0, 2))

        if requested_lane_action == 0:
            candidate_lane = current_lane + 1
            requested_lane_name = "LEFT"

        elif requested_lane_action == 2:
            candidate_lane = current_lane - 1
            requested_lane_name = "RIGHT"

        else:
            candidate_lane = current_lane
            requested_lane_name = "KEEP"

        filtered_lane_action = requested_lane_action
        filtered_lane_name = requested_lane_name

        lane_change_blocked = False

        candidate_front_ttc = current_front_ttc
        candidate_rear_ttc = current_rear_ttc

        # ---------------------------------------------------------
        # 3. Assess the candidate lane only if changing lanes.
        # ---------------------------------------------------------
        if requested_lane_action in (0, 2):

            # Reject an invalid lane at the road boundary.
            if not (0 <= candidate_lane < lanes_count):
                lane_change_blocked = True

            else:
                candidate_values = lane_ttc_current.get(candidate_lane, default_ttc)

                candidate_front_ttc = float(candidate_values["front_ttc"])
                candidate_rear_ttc = float(candidate_values["rear_ttc"])

                # Reject the lane change if a vehicle ahead or behind
                # has a TTC below the safety threshold.
                lane_change_blocked = (
                        candidate_front_ttc < lane_change_threshold
                        or candidate_rear_ttc < lane_change_threshold
                )

            if lane_change_blocked:
                # IMPORTANT: 1 means KEEP in your step() interface.
                filtered_action[1] = 1.0
                filtered_lane_action = 1
                filtered_lane_name = "KEEP"

        # If KEEP was requested originally, no candidate-lane TTC
        # evaluation is needed.

        # ---------------------------------------------------------
        # 3B. Apply the speed limit to the filtered action.
        # This must happen before dynamics prediction and execution.
        # ---------------------------------------------------------
        # Apply the speed limit before dynamics prediction and execution.
        speed_limit_override = False

        current_speed = float(vehicle.speed)
        max_speed = float(getattr(vehicle, "MAX_SPEED", 40.0))

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
            "candidate_lane": candidate_lane,

            "requested_lane_action": requested_lane_action,
            "requested_lane_name": requested_lane_name,

            "filtered_lane_action": filtered_lane_action,
            "filtered_lane_name": filtered_lane_name,

            "current_front_ttc": current_front_ttc,
            "current_rear_ttc": current_rear_ttc,
            "current_lane_ttc": float(current_lane_ttc),

            "candidate_front_ttc": float(candidate_front_ttc),
            "candidate_rear_ttc": float(candidate_rear_ttc),

            "emergency_braking": bool(emergency_braking),
            "lane_change_blocked": bool(lane_change_blocked),

            "emergency_threshold": emergency_threshold,
            "lane_change_threshold": lane_change_threshold,

            "sampled_action": sampled_action.copy(),

            "speed_limit_override": bool(speed_limit_override),
            "filtered_action": filtered_action.copy(),
            "overridden": bool(emergency_braking or lane_change_blocked or speed_limit_override),
        }

        print(
            "[TTC FILTER]",
            f"current_lane={current_lane}",
            f"front_ttc={current_front_ttc:.3f}",
            f"rear_ttc={current_rear_ttc:.3f}",
            f"emergency={emergency_braking}",
        )

        print(
            "[LANE SAFETY]",
            f"requested={requested_lane_name}",
            f"candidate_lane={candidate_lane}",
            f"front_ttc={candidate_front_ttc:.3f}",
            f"rear_ttc={candidate_rear_ttc:.3f}",
            f"blocked={lane_change_blocked}",
            f"filtered={filtered_lane_name}",
        )

        print(
            "[TTC FILTER RESULT]",
            f"sampled={sampled_action.tolist()}",
            f"filtered={filtered_action.tolist()}",
        )

        return filtered_action, safety_info





#???????????????????????????????????????????????????????????????????????????????????????????????????????????
    def _update_lane_target(self, lane_action_id):
        """
        Convert a discrete lane intention into a persistent target lane.
        IDs:
          0 = left
          1 = keep lane
          2 = right
        A new lane change is not started while another is in progress.
        """

        base_env = self.env.unwrapped
        ego = base_env.vehicle
        road = base_env.road

        lane_action_id = int(lane_action_id)

        # Check whether the vehicle has reached its target lane.
        target_lane = road.network.get_lane(self.target_lane_index)


        longitudinal, lateral = target_lane.local_coordinates(ego.position)

        target_reached = (
                abs(float(lateral)) < 0.25
                and abs(float(ego.heading - target_lane.heading_at(longitudinal))) < 0.05
        )

        if self._lane_change_active:
            if target_reached:
                self._lane_change_active = False
                self._lane_change_cooldown = 10
                self._lane_change_source_index = None
            return

        if self._lane_change_cooldown > 0:
            self._lane_change_cooldown -= 1
            return

        # Keep-lane action does not change the target lane.
        if lane_action_id == 1:
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
        candidate_lane_id = (
            current_lane_id + 1
            if lane_action_id == 0
            else current_lane_id - 1
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

        print(
            "[LANE CHANGE REQUEST] "
            f"action={lane_action_id}, "
            f"target_lane={self.target_lane_index}"
        )




    def step(self, action, apply_safety_filter=True, sampled_action=None, safety_info=None,):
        """
        Input:
            action[0] = continuous normalized acceleration
            action[1] = discrete lane ID: 0=left, 1=keep, 2=right
        Highway receives:
            [continuous acceleration, continuous normalized steering]
        """
        print(
            "[STEP INPUT]",
            f"apply_safety_filter={apply_safety_filter}",
            f"action={np.asarray(action).tolist()}",
        )


        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size != 2:
            raise ValueError(
                f"Expected hybrid action of size 2, got {action.shape}"
            )



        # Keep the sampled policy action distinct from the filtered action.
        incoming_action = action.copy()
        if sampled_action is None:
            sampled_action = incoming_action.copy()
        else:
            sampled_action = np.asarray(sampled_action, dtype=np.float32).reshape(-1).copy()

        if apply_safety_filter:
            action, safety_info = self.filter_ttc_action(sampled_action)
        else:
            action = incoming_action.copy()
            if safety_info is None:
                # Backward compatibility for callers that do not supply it.
                _, safety_info = self.filter_ttc_action(sampled_action)
            else:
                safety_info = dict(safety_info)
            safety_info["filtered_action"] = action.copy()

        print(
            "[ACTION AGREEMENT]",
            f"incoming={incoming_action.tolist()}",
            f"adapter_filtered={action.tolist()}",
            f"same={np.allclose(incoming_action, action)}",
        )

        acceleration = float(np.clip(action[0], -1.0, 1.0))
        lane_action_id = int(np.clip(np.rint(action[1]), 0, 2))
        current_lane_before = int(self.env.unwrapped.vehicle.lane_index[2])

        lane_action_names = {
            0: "LEFT",
            1: "KEEP",
            2: "RIGHT",
        }

        print(
            "[ACTION MAPPING]",
            f"sampled_action={sampled_action.tolist()}",
            f"incoming_action={incoming_action.tolist()}",
            f"filtered_action={action.tolist()}",
            f"current_lane={current_lane_before}",
            f"lane_action_id={lane_action_id}",
            f"lane_action={lane_action_names[lane_action_id]}",
        )

        # Convert the filtered discrete intention into a persistent lane target.
        self._update_lane_target(lane_action_id)

        target_lane = getattr(self, "target_lane", None)
        print(
            "[LANE TARGET]",
            f"current_lane={current_lane_before}",
            f"lane_action={lane_action_names[lane_action_id]}",
            f"target_lane_index={self.target_lane_index}",
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
                max_steering=1.0,
            )
        )


        #???????????????????????????????????????????????????????????????????????????????????????????????
        # Physical action sent to Highway
        executed_action = np.asarray([acceleration, steering], dtype=np.float32)
        print(
            "[PHYSICAL ACTION]",
            f"acceleration={acceleration:.4f}",
            f"steering={steering:.4f}",
            f"lateral_error={lateral:.4f}",
            f"heading_error={heading_error:.4f}",
            f"executed_action={executed_action.tolist()}",
        )

        base_env = self.env.unwrapped
        ego_before = base_env.vehicle

        speed_before = float(ego_before.speed)
        crashed_before = bool(getattr(ego_before, "crashed", False))
        max_speed = getattr(ego_before, "MAX_SPEED", None)

        print(
            f"\n[SPEED BEFORE] "
            f"speed={speed_before:.4f} m/s, "
            f"max_speed={max_speed}, "
            f"crashed={crashed_before}, "
            f"normalized_accel={acceleration:.4f}, "
            f"executed_action={executed_action}"
        )


        obs, env_reward, terminated, truncated, info = (self.env.step(executed_action))

        ego_after = self.env.unwrapped.vehicle
        speed_after = float(ego_after.speed)
        crashed_after = bool(getattr(ego_after, "crashed", False))

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
        next_state = self._encode_state(obs)

        ego = self.env.unwrapped.vehicle
        collision = bool(getattr(ego, "crashed", False))

        info = dict(info or {})
        info["safety_violation"] = collision
        info["collision"] = collision
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["env_reward"] = float(env_reward)

        # Keep both representations distinct.
        info["lane_action_id"] = lane_action_id
        info["sampled_policy_action"] = sampled_action.copy()
        info["policy_action"] = np.asarray([acceleration, lane_action_id], dtype=np.float32,)
        info["filtered_policy_action"] = np.asarray([acceleration, lane_action_id], dtype=np.float32)
        info["executed_action"] = executed_action.copy()
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
            max_steering=1.0,
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
        steering = -k_lateral * lateral - k_heading * heading_error
        steering = float(np.clip(steering, -max_steering, max_steering))

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

def collect_rollouts(
    simulator,
    policy,
    num_episodes,
    max_steps
):
    rollouts = []


    for episode in range(num_episodes):
        state = simulator.reset()
        rollout = []

        for t in range(max_steps):    #10

            # Policy chooses acceleration and a discrete lane intention.
            valid_mask = simulator.get_valid_lane_action_mask()

            sampled_action = np.asarray(policy(state, valid_action_mask=valid_mask), dtype=np.float32).reshape(-1)
            action, safety_info = simulator.filter_ttc_action(sampled_action)
            # Measure physical TTC in seconds before the transition.
            min_ttc_before_step, _ = calculate_min_ttc(simulator)

            # Predict/learn from the filtered high-level action that will be executed.
            next_state, reward, done, info = simulator.step(action, apply_safety_filter=False, sampled_action=sampled_action, safety_info=safety_info,)

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

        rollouts.append(rollout)

        print(
            f"Reference episode "
            f"{episode + 1}/"
            f"{num_episodes}"
        )

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

    # Use the physical lane occupied by the ego vehicle.
    physical_lane_id = int(ego.lane_index[2])
    relevant_lane_ids = {physical_lane_id}

    # During a lane change, monitor both the source and target.
    if getattr(simulator, "_lane_change_active", False):
        target = getattr(simulator, "target_lane_index", None)
        source = getattr(simulator, "_lane_change_source_index", None)

        if target is not None:
            relevant_lane_ids.add(int(target[2]))

        if source is not None:
            relevant_lane_ids.add(int(source[2]))

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

    if state.size < 28:
        raise ValueError(
            f"Expected at least 28 vehicle features, got {state.size}"
        )

    vehicle_rows = state[:28].reshape(4, 7)
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

    vehicle_rows = state[:28].reshape(4, 7)

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
    if risk >= 0.7:
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
        "speed_feature_normalized": float(state[28]),
        "lane_feature_normalized": float(state[29]),
        "collision_indicator": float(state[30]),
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
    ttc = float(driving_context["minimum_ttc_seconds"])


    # Separate reward-design decisions.

    if risk >= 0.80:
        target_behavior = (
            "Prioritize collision avoidance, safe spacing, "
            "and smooth control, even at the cost of progress."
        )
        tracking_weight = 0.5
        safety_weight = 4.0
        action_weight = 0.10
        reward_mode = "safety"

    elif risk >= TAU_RISK:
        target_behavior = (
            "Maintain safe following distance and smooth "
            "control while making cautious progress."
        )
        tracking_weight = 0.8
        safety_weight = 2.5
        action_weight = 0.1
        reward_mode = "safety"

    elif shift:
        # Shift-specific response, even when risk is low.
        target_behavior = (
            "Adapt to changed traffic conditions, monitor "
            "nearby vehicles, maintain safe spacing, "
            "and make stable progress."
        )
        tracking_weight = 1.0
        safety_weight = 1.5
        action_weight = 0.1
        reward_mode = "shift_adaptation"

    else:
        target_behavior = (
            "Make steady highway progress while maintaining "
            "safe spacing and smooth control."
        )
        tracking_weight = 1.0
        safety_weight = 1.0
        action_weight = 0.1
        reward_mode = "nominal"



    if ttc <= ttc_critical_threshold:
        target_behavior += (
            "Prioritize reducing collision risk and increasing the following gap when TTC is low, while maintaining stable lane control."
        )
        ttc_weight = 5.0
    elif ttc <= ttc_danger_threshold:
        ttc_weight = 3.0
    elif ttc <= ttc_caution_threshold:
        ttc_weight = 2.0
    else:
        ttc_weight = 1.0


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
        "shift_score": float(driving_context["shift_score"]),
        "risk_score": risk,
        # New road-keeping constraints
        "lane_keeping_weight": lane_keeping_weight,
        "max_lateral_deviation": 2.0,
        "offroad_penalty": offroad_penalty,
        "ttc_weight": ttc_weight,
        "ttc_critical_threshold": ttc_critical_threshold,
        "ttc_danger_threshold": ttc_danger_threshold,
        "ttc_caution_threshold": ttc_caution_threshold,

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
            - safety_penalty
            - control_penalty
            - lane_penalty
            - offroad_penalty
            - ttc_penalty
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
    ):
        behavior_embedding = embedding_model.encode(behavior_description)

        semantic_reward = cosine_similarity(behavior_embedding, target_embedding,)

        # Map cosine similarity [-1, 1] to [0, 1].
        semantic_reward = (semantic_reward + 1.0) / 2.0

        # Existing safety penalty.
        safety_penalty = (
            float(reward_specification["safety_weight"])
            if safety_violation
            else 0.0
        )

        # Existing action/control penalty.
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        acceleration = float(np.clip(action[0], -1.0, 1.0))
        lane_action_id = int(np.clip(np.rint(action[1]), 0, 2))

        # Penalize the magnitude of acceleration/braking.
        control_penalty = (float(reward_specification["action_weight"]) * abs(acceleration))

        # Optional small cost to discourage unnecessary lane changes.
        lane_change_cost = (0.01 if lane_action_id != 1 else 0.0)

        control_penalty += lane_change_cost




        # New lane-keeping penalty.
        lane_deviation = abs(float(lane_deviation))
        normalized_deviation = np.clip(lane_deviation / max_lateral_deviation, 0.0, 1.0,)
        lane_penalty = lane_keeping_weight * normalized_deviation


        # New explicit off-road penalty.
        road_penalty = offroad_penalty_magnitude if offroad else 0.0


        # 6. New TTC penalty: TTC is measured in seconds
        min_ttc = float(min_ttc)

        # Read the actual settings selected by the reward designer.
        ttc_weight_value = float(reward_specification.get("ttc_weight", ttc_weight))

        critical_ttc_value = float(reward_specification.get("ttc_critical_threshold", ttc_critical_threshold,))

        safe_ttc_value = float(reward_specification.get("ttc_safe_threshold", reward_specification.get("ttc_caution_threshold", ttc_caution_threshold,),))


        print(
            f"[TTC PENALTY DEBUG] "
            f"min_ttc={min_ttc:.3f}, "
            f"critical={critical_ttc_value:.3f}, "
            f"safe={safe_ttc_value:.3f}, "
            f"weight={ttc_weight_value:.3f}"
        )


        ttc_penalty_value = calculate_ttc_penalty(
            min_ttc,
            critical_ttc=critical_ttc_value,
            safe_ttc=safe_ttc_value,
        )

        ttc_penalty = ttc_weight_value * ttc_penalty_value

        print(
            f"[TTC PENALTY RESULT] "
            f"raw_ttc_penalty={ttc_penalty_value:.4f}, "
            f"weighted={ttc_penalty:.4f}"
        )


        reward = (
            float(reward_specification["tracking_weight"])
            * float(semantic_reward)
            - safety_penalty
            - control_penalty
            - lane_penalty
            - road_penalty
            - ttc_penalty
        )

        print(
            f"[REWARD DEBUG] semantic={semantic_reward:.3f}, "
            f"safety={safety_penalty:.3f}, "
            f"lane={lane_penalty:.3f}, "
            f"offroad={road_penalty:.3f}, "
            f"min_ttc={min_ttc:.2f}s, "
            f"ttc_penalty={ttc_penalty:.3f}, "
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
):
    return reward_function(
        behavior_description=behavior_description,
        safety_violation=safety_violation,
        action=action,
        lane_deviation=lane_deviation,
        offroad=offroad,
        min_ttc=min_ttc
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


    # Acceleration is tanh-transformed from a Gaussian sample.
    accel = action_tensor[:, 0].clamp(-0.999, 0.999)

    pre_tanh_accel = 0.5 * (torch.log1p(accel) - torch.log1p(-accel))

    accel_log_prob = accel_dist.log_prob(pre_tanh_accel.unsqueeze(-1)).squeeze(-1)

    # Jacobian correction for acceleration = tanh(raw_accel)
    accel_log_jacobian = torch.log(1.0 - accel.pow(2) + 1e-6)

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
    vehicle_rows = state[:28].reshape(4, 7)
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
    lane_action_names = {
        0: "request a left lane change",
        1: "keep the current lane",
        2: "request a right lane change",
    }

    lane_intention = lane_action_names[lane_action_id]

    transition_change = float(np.linalg.norm(next_state[:28] - state[:28]))
    smoothness = "noticeable state change" if transition_change > 0.7 else "relatively small state change"

    return (
        f"Highway driving behavior: {traffic}; "
        f"predicted risk is {risk_text}; "
        f"normalized acceleration command {accel:.2f}; "
        f"lane intention: {lane_intention}; "
        f"the encoded traffic state shows {smoothness}."
    )





#**********************************************************************************************
# reference model without Shift Detection
checkpoint_path = "reference_checkpoint_highway.pt"

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

    reward_List = []
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
            "safe spacing and smooth control."
        ),
        "tracking_weight": 1.0,
        "safety_weight": 1.0,
        "action_weight": 0.02,
        "shift_score": 0.0,
        "risk_score": 0.0,
        # Lane-keeping and road-boundary constraints
        "lane_keeping_weight": lane_keeping_weight,
        "max_lateral_deviation": 2.0,
        "offroad_penalty": offroad_penalty,
        "ttc_weight": 2.0,
        "ttc_critical_threshold": ttc_critical_threshold,
        "ttc_safe_threshold": ttc_caution_threshold,
    }

    reward_function = construct_reward(reward_specification, embedding_model,)




    for episode in range(NUM_ONLINE_EPISODES):
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
        adaptation_count = 0
        risk_score_count = 0


        for t in range(MAX_STEPS):

            # =================================================
            # STEP 1
            # Policy
            # =================================================

            valid_mask = simulator.get_valid_lane_action_mask()

            #action = np.asarray(policy(state, valid_action_mask=valid_mask), dtype=np.float32,).reshape(-1)
            sampled_action = np.asarray(policy(state, valid_action_mask=valid_mask), dtype=np.float32).reshape(-1)
            filtered_action, safety_info = simulator.filter_ttc_action(sampled_action)
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


            # Canonical high-level action for prediction and execution.
            reward_action = filtered_action.copy()

            next_state_pred = dynamics_model.predict(state, reward_action)
            min_ttc, closing_vehicle_count = calculate_min_ttc(simulator)

            next_state, simulator_reward, done, step_info = simulator.step(
                reward_action,
                apply_safety_filter=False,
                sampled_action=sampled_action,
                safety_info=safety_info,
            )

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

            # Use the reconciled action for risk prediction and all
            # subsequent transition/reward context calculations.
            risk = risk_model.predict(state, reward_action, next_state_pred)



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

            reward = compute_reward(
                reward_function=reward_function,
                behavior_description=behavior_description,
                safety_violation=safety_violation,
                action=reward_action,
                lane_deviation=lane_deviation,
                offroad=offroad,
                min_ttc=min_ttc,
            )

            if safety_info["overridden"]:
                reward -= SAFETY_OVERRIDE_REWARD_PENALTY
                print(
                    "[TTC SAFETY REWARD] intervention penalty="
                    f"{SAFETY_OVERRIDE_REWARD_PENALTY:.2f}, "
                    f"adjusted_reward={reward:.3f}"
                )


#--------------------------------------------------------------------------



            # =================================================
            # STEP 10
            # RL Policy Update
            # =================================================

            update_policy(
                policy,
                state,
                sampled_action,  # Original action whose probability the actor computes
                reward,
                next_state,
                done,
                valid_action_mask=valid_mask,
                terminated=bool(step_info.get("terminated", False) or offroad),
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
        adaptation_count_List.append(adaptation_count)
        risk_List.append(risk_score_count)
        collision_List.append(episode_collision_count)

        episode_collision_rate = (episode_collision_count / max(t + 1, 1))
        recent_returns.append(episode_reward)
        recent_collision_rates.append(episode_collision_rate)



        print(
            f"Episode "
            f"{episode + 1:03d}/"
            f"{NUM_ONLINE_EPISODES} | "
            f"reward={episode_reward:.3f} | "
            f"adaptations={adaptation_count}"
        )

    episode_reward_List = list(range(len(reward_List)))
    fig_path = reward_path
    plt.plot(episode_reward_List, reward_List)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('reward in episodes')
    # plt.show()
    plt.savefig(fig_path)
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
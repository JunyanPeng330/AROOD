
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# ============================================================
# 1. Configuration
# ============================================================

STATE_DIM = 32
ACTION_DIM = 3

HIDDEN_DIM = 128

NUM_REFERENCE_EPISODES = 40     # short reference window = 20
NUM_ONLINE_EPISODES = 150

MAX_STEPS = 200

RISK_HORIZON = 10

TAU_MMD = 0.60
TAU_RISK = 0.60

DISTRIBUTION_WINDOW = 35

DYNAMICS_LR = 1e-3
RISK_LR = 1e-3
POLICY_LR = 3e-4

DYNAMICS_EPOCHS = 5
RISK_EPOCHS = 20

MMD_SIGMA = 1.0

MODEL_PATH = "adaptive_policy.pt"


EMBEDDING_DIM = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# when CUDA is enabled, model parameters commonly report their device as cuda:0,
# while torch.device("cuda") may have no explicit device index.


# ============================================================
# 1. Reproducibility from x_main
# ============================================================

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

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                state_dim + action_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                state_dim
            )
        )

    def forward(
        self,
        state,
        action
    ):

        state = to_tensor(state)
        action = to_tensor(action)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        x = torch.cat(
            [state, action],
            dim=-1
        )

        return self.network(x)

    def predict(
        self,
        state,
        action
    ):

        with torch.no_grad():
            state_t = to_tensor(state).unsqueeze(0)
            action_t = to_tensor(action).unsqueeze(0)
            prediction = self.forward(
                state_t,
                action_t
            ).squeeze(0)

        return prediction.cpu().numpy() ###########################################################################################################


# ============================================================
# 4. Risk Predictor
# ============================================================

class RiskPredictor(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim
    ):
        super().__init__()

        input_dim = (2 * state_dim + action_dim)

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            ),

            nn.Sigmoid()                         # Sigmoid      #################################################################################
        )

    def forward(
        self,
        state,
        action,
        next_state_pred
    ):

        state = to_tensor(state)
        action = to_tensor(action)
        next_state_pred = to_tensor(next_state_pred)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        if next_state_pred.ndim == 1:
            next_state_pred = next_state_pred.unsqueeze(0)

        x = torch.cat(
            [
                state,
                action,
                next_state_pred
            ],
            dim=-1
        )

        return self.network(x)

    def predict(
        self,
        state,
        action,
        next_state_pred
    ):

        with torch.no_grad():
            state_t = to_tensor(state).unsqueeze(0)
            action_t = to_tensor(action).unsqueeze(0)
            next_state_t = to_tensor(next_state_pred).unsqueeze(0)

            risk = self.forward(
                state_t,
                action_t,
                next_state_t
            )

        return risk.squeeze()


# ============================================================
# 5. Actor-Critic Policy
# ============================================================

class ActorCriticPolicy(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim
    ):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                action_dim
            )
        )

        self.critic = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

        self.optimizer = Adam(
            self.parameters(),
            lr=POLICY_LR
        )

    def action_distribution(
        self,
        state
    ):

        state = to_tensor(state)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        mean = self.actor(state)

        std = torch.exp(self.log_std).expand_as(mean)

        return torch.distributions.Normal(mean, std)

    def act(
        self,
        state,
        deterministic=False
    ):

        distribution = self.action_distribution(state)

        if deterministic:
            action = distribution.mean

        else:
            action = distribution.sample()

        action = torch.tanh(action)

        return action.squeeze(0).detach().cpu().numpy()

    def __call__(
        self,
        state
    ):

        return self.act(
            state,
            deterministic=False
        )

    def value(
        self,
        state
    ):

        state = to_tensor(state)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        return self.critic(state).squeeze(-1)


# ============================================================
# 6. Executable Simulator
#
# This simulator is deliberately lightweight so the entire
# script can run without CARLA.
#
# Replace this class with the CARLA wrapper later.
# ============================================================

class SimpleDrivingSimulator:

    def __init__(
        self,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_steps=MAX_STEPS
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps

        self.step_count = 0

        self.state = None

        # Environment condition.
        # Reference environment starts at 0.
        # Online episodes gradually move toward an OOD
        # condition.
        self.condition = 0.0

    def reset(self):

        self.step_count = 0

        self.state = np.random.normal(
            0.0,
            0.1,
            self.state_dim
        ).astype(
            np.float32
        )

        return self.state.copy()

    def step(
        self,
        action
    ):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(action, -1.0, 1.0)

        self.step_count += 1

        # ----------------------------------------------------
        # Simulated vehicle dynamics
        # ----------------------------------------------------

        action_effect = np.zeros(
            self.state_dim,
            dtype=np.float32
        )

        action_effect[:self.action_dim] = (0.20 * action)

        noise = np.random.normal(0.0, 0.03, self.state_dim).astype(np.float32)

        # OOD condition changes the dynamics.
        shift_effect = np.zeros(
            self.state_dim,
            dtype=np.float32
        )

        shift_effect[:5] = (self.condition * 0.25)

        next_state = (0.95 * self.state + action_effect + shift_effect + noise)

        # ----------------------------------------------------
        # Driving-style safety condition
        # ----------------------------------------------------

        state_magnitude = np.linalg.norm(next_state[:5])

        action_magnitude = np.linalg.norm(action)

        safety_violation = (
            state_magnitude > 1.20
            or
            action_magnitude > 1.55
        )

        # ----------------------------------------------------
        # Simulator reward
        # ----------------------------------------------------

        reward = (1.0 - 0.05 * action_magnitude - 0.10 * state_magnitude)

        if safety_violation:
            reward -= 5.0

        done = (self.step_count >= self.max_steps)

        info = {"safety_violation": bool(safety_violation)}

        self.state = next_state

        return (
            next_state.copy(),
            float(reward),
            done,
            info
        )


# ============================================================
# 7. Simulator Factory
# ============================================================

def create_carla_simulator():
    """
    Factory used by main().

    The current implementation returns the executable
    simulator above.

    To connect CARLA, replace the return statement with
    your CARLA environment wrapper implementing:

        reset()
        step(action)

    and returning:

        next_state, reward, done, info
    """

    return SimpleDrivingSimulator(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_steps=MAX_STEPS
    )


# ============================================================
# 8. Create Initial Policy
# ============================================================

def create_initial_policy(
    state_dim,
    action_dim
):

    return ActorCriticPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM
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

    for episode in range(
        num_episodes
    ):

        state = simulator.reset()

        rollout = []

        for t in range(
            max_steps
        ):

            action = policy(
                state
            )

            (
                next_state,
                reward,
                done,
                info
            ) = simulator.step(
                action
            )

            transition = {

                "state":
                    np.asarray(
                        state,
                        dtype=np.float32
                    ),

                "action":
                    np.asarray(
                        action,
                        dtype=np.float32
                    ),

                "reward":
                    float(reward),

                "next_state":
                    np.asarray(
                        next_state,
                        dtype=np.float32
                    ),

                "safety_violation":
                    bool(
                        info.get(
                            "safety_violation",
                            False
                        )
                    )
            }

            rollout.append(
                transition
            )

            state = next_state

            if done:
                break

        rollouts.append(
            rollout
        )

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

    optimizer = Adam(
        dynamics_model.parameters(),
        lr=DYNAMICS_LR
    )

    criterion = nn.MSELoss()

    dataset = []

    for rollout in rollouts:

        for transition in rollout:

            dataset.append(
                transition
            )

    if not dataset:
        raise RuntimeError(
            "No transitions available for DynamicsModel training."
        )

    print(
        f"Training DynamicsModel "
        f"on {len(dataset)} transitions..."
    )

    for epoch in range(
        DYNAMICS_EPOCHS
    ):

        total_loss = 0.0

        random.shuffle(
            dataset
        )

        for transition in dataset:

            state = to_tensor(
                transition["state"]
            )

            action = to_tensor(
                transition["action"]
            )

            next_state = to_tensor(
                transition["next_state"]
            )

            prediction = \
                dynamics_model(
                    state,
                    action
                ).squeeze(0)

            loss = criterion(
                prediction,
                next_state
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += \
                loss.item()

        mean_loss = (
            total_loss
            / len(dataset)
        )

        print(
            f"  Dynamics epoch "
            f"{epoch + 1}/"
            f"{DYNAMICS_EPOCHS}: "
            f"loss={mean_loss:.6f}"
        )

    return dynamics_model


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

    optimizer = Adam(
        risk_model.parameters(),
        lr=RISK_LR
    )

    criterion = nn.BCELoss()

    samples = []

    for rollout in rollouts:

        if len(rollout) <= horizon_H:
            continue

        for t in range(
            len(rollout) - horizon_H
        ):

            state = \
                rollout[t]["state"]

            action = \
                rollout[t]["action"]

            next_state_pred = \
                dynamics_model.predict(
                    state,
                    action
                )

            risk_label = 0.0

            for k in range(
                1,
                horizon_H + 1
            ):

                if rollout[
                    t + k
                ]["safety_violation"]:

                    risk_label = 1.0

                    break

            samples.append(
                (
                    state,
                    action,
                    next_state_pred,
                    risk_label
                )
            )

    if not samples:
        raise RuntimeError(
            "No samples available "
            "for RiskPredictor training."
        )

    print(
        f"Training RiskPredictor "
        f"on {len(samples)} samples..."
    )

    for epoch in range(
        RISK_EPOCHS
    ):

        random.shuffle(
            samples
        )

        total_loss = 0.0

        for (
            state,
            action,
            next_state_pred,
            risk_label
        ) in samples:

            state = to_tensor(
                state
            )

            action = to_tensor(
                action
            )

            next_state_pred = \
                to_tensor(
                    next_state_pred
                )

            target = to_tensor(
                [risk_label])

            prediction = \
                risk_model(
                    state,
                    action,
                    next_state_pred
                ).view(-1)

            loss = criterion(
                prediction,
                target
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += \
                loss.item()

        mean_loss = (
            total_loss
            / len(samples)
        )

        print(
            f"  Risk epoch "
            f"{epoch + 1}/"
            f"{RISK_EPOCHS}: "
            f"loss={mean_loss:.6f}"
        )

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

            reference_distribution.append(
                feature
            )

    if not reference_distribution:

        raise RuntimeError(
            "Reference distribution is empty."
        )

    return np.asarray(
        reference_distribution,
        dtype=np.float32
    )


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

    current_distribution.append(
        feature
    )

    return current_distribution


# ============================================================
# 14. RBF Kernel
# ============================================================

def rbf_kernel(
    x,
    y,
    sigma=1.0
):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    x_norm = np.sum(
        x * x,
        axis=1,
        keepdims=True
    )

    y_norm = np.sum(
        y * y,
        axis=1,
        keepdims=True
    ).T

    squared_distance = (
        x_norm
        + y_norm
        - 2.0 * np.matmul(
            x,
            y.T
        )
    )

    squared_distance = np.maximum(
        squared_distance,
        0.0
    )

    return np.exp(
        -squared_distance
        / (
            2.0 * sigma * sigma
        )
    )


# ============================================================
# 15. MMD Shift Detector
# ============================================================

def compute_MMD(
    reference_distribution,
    current_distribution,
    sigma=MMD_SIGMA
):

    reference = np.asarray(
        reference_distribution,
        dtype=np.float32
    )

    current = np.asarray(
        current_distribution,
        dtype=np.float32
    )

    if len(reference) == 0:
        return 0.0

    if len(current) == 0:
        return 0.0

    # --------------------------------------------------------
    # Limit reference samples for computational efficiency
    # --------------------------------------------------------

    reference_sample_size = min(
        len(reference),
        200
    )

    current_sample_size = min(
        len(current),
        100
    )

    reference_indices = np.random.choice(
        len(reference),
        reference_sample_size,
        replace=False
    )

    current_indices = np.random.choice(
        len(current),
        current_sample_size,
        replace=False
    )

    x = reference[
        reference_indices
    ]

    y = current[
        current_indices
    ]

    K_xx = rbf_kernel(
        x,
        x,
        sigma
    )

    K_yy = rbf_kernel(
        y,
        y,
        sigma
    )

    K_xy = rbf_kernel(
        x,
        y,
        sigma
    )

    # --------------------------------------------------------
    # Biased empirical MMD^2 estimator
    # --------------------------------------------------------

    mmd_squared = (

        K_xx.mean()

        + K_yy.mean()

        - 2.0 * K_xy.mean()
    )

    mmd_squared = max(
        float(mmd_squared),
        0.0
    )

    return float(
        np.sqrt(
            mmd_squared
        )
    )


def detect_distribution_shift(
    reference_distribution,
    current_distribution,
    tau_MMD
):

    shift_score = compute_MMD(

        reference_distribution,

        current_distribution
    )

    shift_flag = (
        shift_score > tau_MMD
    )

    return (
        shift_score,
        shift_flag
    )


# ============================================================
# 16. Shift-Risk Trigger
# ============================================================

def shift_risk_trigger(
    shift_score,
    risk,
    tau_MMD,
    tau_risk
):

    shift_flag = (
        shift_score > tau_MMD
    )

    risk_flag = (
        risk > tau_risk
    )

    # --------------------------------------------------------
    # Adaptation requires:
    #
    #     OOD shift
    #          AND
    #     high safety risk
    # --------------------------------------------------------

    adaptation_trigger = (

        shift_flag
        and
        risk_flag

    )

    return adaptation_trigger


# ============================================================
# 17. Environment Information
# ============================================================

def construct_environment_information(
    state,
    action,
    next_state_pred,
    shift_score,
    risk_score
):

    driving_context = {

        "state":
            np.asarray(
                state
            ),

        "action":
            np.asarray(
                action
            ),

        "predicted_next_state":
            np.asarray(
                next_state_pred
            ),

        "shift_score":
            float(
                shift_score
            ),

        "risk_score":
            float(
                risk_score
            )
    }

    return driving_context


# ============================================================
# 18. Executable AutoReward / LLM Interface
# ============================================================

def AutoReward_LLM(
    driving_context
):

    """
    Executable stand-in for the LLM-based AutoReward module.
    Conditional reward-specification generator
    In the final research system this function can be replaced
    with an actual LLM call.

    The current implementation generates a reward specification
    from the detected driving context.
    """

    risk = driving_context["risk_score"]

    shift_score = driving_context["shift_score"]

    if risk >= 0.80:

        reward_specification = {

            "name":
                "high_risk_safety_reward",

            "tracking_weight":
                0.5,

            "safety_weight":
                5.0,

            "action_weight":
                0.10,

            "shift_score":
                shift_score,

            "risk_score":
                risk
        }

    elif risk >= 0.50:

        reward_specification = {

            "name":
                "moderate_risk_safety_reward",

            "tracking_weight":
                0.8,

            "safety_weight":
                2.5,

            "action_weight":
                0.05,

            "shift_score":
                shift_score,

            "risk_score":
                risk
        }

    else:

        reward_specification = {

            "name":
                "nominal_reward",

            "tracking_weight":
                1.0,

            "safety_weight":
                1.0,

            "action_weight":
                0.02,

            "shift_score":
                shift_score,

            "risk_score":
                risk
        }

    return reward_specification


# ============================================================
# 19. Construct Adaptive Reward
# ============================================================

def construct_reward(
    reward_specification,
    embedding_model
):
    """
    Constructs the semantic reward from the LLM-generated
    reward specification.

    The central reward mechanism is:

        e_b = Embedding(behavior_description)

        e_r = Embedding(reward_specification)

        r_semantic =
            cos(e_b, e_r)

    The previous Euclidean-norm tracking reward is no longer
    used as the main reward signal.
    """

    target_text = reward_specification[
        "name"
    ]

    target_embedding = embedding_model.encode(
        target_text
    )

    def reward_function(
        behavior_description,
        safety_violation,
        action
    ):

        behavior_embedding = (
            embedding_model.encode(
                behavior_description
            )
        )

        semantic_reward = cosine_similarity(
            behavior_embedding,
            target_embedding
        )

        # Map cosine similarity [-1, 1]
        # into a normalized reward [0, 1].
        semantic_reward = (
            semantic_reward + 1.0
        ) / 2.0

        safety_penalty = (
            reward_specification[
                "safety_weight"
            ]
            if safety_violation
            else 0.0
        )

        control_penalty = (
            reward_specification[
                "action_weight"
            ]
            * np.linalg.norm(action)
        )

        reward = (
            reward_specification[
                "tracking_weight"
            ]
            * semantic_reward
            -
            safety_penalty
            -
            control_penalty
        )

        return float(reward)

    return reward_function


def compute_reward(
    reward_function,
    behavior_description,
    safety_violation,
    action
):

    return reward_function(
        behavior_description,
        safety_violation,
        action
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
    done
):

    """
    One-step actor-critic update.

    """

    state_tensor = to_tensor(
        state
    ).unsqueeze(0)

    next_state_tensor = to_tensor(
        next_state
    ).unsqueeze(0)

    action_tensor = to_tensor(
        action
    ).unsqueeze(0)

    reward_tensor = to_tensor(
        [reward])

    # --------------------------------------------------------
    # Critic
    # --------------------------------------------------------

    value = policy.value(
        state_tensor
    )

    with torch.no_grad():

        next_value = policy.value(
            next_state_tensor
        )

        target = reward_tensor

        if not done:
            target = (
                reward_tensor
                + 0.99 * next_value
            )

    advantage = (
        target
        - value
    )

    critic_loss = (
        advantage.pow(2).mean()
    )

    # --------------------------------------------------------
    # Actor
    # --------------------------------------------------------

    distribution = \
        policy.action_distribution(
            state_tensor
        )

    # The environment receives tanh(action).
    # Reconstruct the pre-tanh value approximately.
    clipped_action = torch.clamp(
        action_tensor,
        -0.999,
        0.999
    )

    pre_tanh_action = (
        0.5
        * torch.log(
            (1.0 + clipped_action)
            /
            (1.0 - clipped_action)
        )
    )

    log_probability = \
        distribution.log_prob(
            pre_tanh_action
        ).sum(
            dim=-1
        )

    actor_loss = -(
        log_probability
        * advantage.detach()
    ).mean()

    loss = (
        actor_loss
        + critic_loss
    )

    policy.optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        max_norm=1.0
    )

    policy.optimizer.step()

    return float(
        loss.item()
    )


# ============================================================
# 22. Save Policy
# ============================================================

def save_policy(
    policy,
    path
):
    if not isinstance(policy, nn.Module):
        raise TypeError("save_policy() expected a PyTorch policy model, " f"but received {type(policy).__name__}.")

    torch.save(policy.state_dict(), path)

    print(
        f"Policy saved to: {path}"
    )





# ============================================================
# Embedding Model
# ============================================================

class RewardEmbeddingModel:
    """
    Converts textual behavior descriptions and reward
    specifications into fixed-dimensional embeddings.

    The implementation is deliberately self-contained so that
    the complete program does not require an external API.

    In a full experimental implementation this class can be
    replaced by a pretrained sentence-embedding model such as
    Sentence-BERT.

    The interface remains:

        text -> embedding vector
    """

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM
    ):
        self.embedding_dim = embedding_dim

    def _tokenize(self, text):

        text = text.lower()

        tokens = []
        current = ""

        for character in text:

            if character.isalnum():

                current += character

            else:

                if current:
                    tokens.append(current)
                    current = ""

        if current:
            tokens.append(current)

        return tokens

    def encode(self, text):

        """
        Deterministic hashed text embedding.

        This provides a completely executable embedding
        mechanism without requiring an external model.
        """

        vector = np.zeros(
            self.embedding_dim,
            dtype=np.float32
        )

        tokens = self._tokenize(text)

        if not tokens:
            return vector

        for token in tokens:

            index = (
                hash(token)
                % self.embedding_dim
            )

            sign = (
                1.0
                if hash(token + "_sign") % 2 == 0
                else -1.0
            )

            vector[index] += sign

        # Add simple bigram information
        for i in range(len(tokens) - 1):

            bigram = (
                tokens[i] +
                "_" +
                tokens[i + 1]
            )

            index = (
                hash(bigram)
                % self.embedding_dim
            )

            vector[index] += 0.5

        norm = np.linalg.norm(vector)

        if norm > 1e-8:
            vector /= norm

        return vector.astype(np.float32)


def cosine_similarity(
    embedding_a,
    embedding_b
):
    """
    cos(e_a, e_b)
    """

    a = np.asarray(
        embedding_a,
        dtype=np.float32
    )

    b = np.asarray(
        embedding_b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator < 1e-8:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )



# ============================================================
# Behavior Description
# ============================================================

def describe_current_behavior(
    state,
    action,
    next_state,
    risk_score
):
    """
    Converts the current driving situation into a textual
    semantic representation.

    In the complete AutoReward system this description is
    generated/selected through the LLM reasoning stage.

    This function provides the executable local representation
    used by the embedding stage.
    """

    state_magnitude = float(
        np.linalg.norm(state[:5])
    )

    next_state_magnitude = float(
        np.linalg.norm(next_state[:5])
    )

    action_magnitude = float(
        np.linalg.norm(action)
    )

    if risk_score > 0.8:

        risk_description = (
            "high risk driving situation"
        )

    elif risk_score > 0.5:

        risk_description = (
            "moderate risk driving situation"
        )

    else:

        risk_description = (
            "low risk driving situation"
        )

    if (
        next_state_magnitude
        > state_magnitude
    ):

        motion_description = (
            "vehicle state is becoming less stable"
        )

    else:

        motion_description = (
            "vehicle state is relatively stable"
        )

    return (
        f"{risk_description}; "
        f"{motion_description}; "
        f"control magnitude {action_magnitude:.3f}"
    )









#**********************************************************************************************
# reference model without Shift Detection
checkpoint_path = "reference_checkpoint.pt"
GAMMA = 0.99
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
        "reference_distribution": reference_distribution}

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

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

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




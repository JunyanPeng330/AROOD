# ============================================================
# AutoReward + OOD Detection + Risk-Decision Adaptive RL

# For CARLA:
#     Replace create_simulator() with a CARLA environment
#     implementing the same reset()/step() interface.
# ============================================================
reward_path = 'figures/' + 'x_Re_main_0' + '.png'
adaptation_path = 'figures/' + 'x_Ad_main_0' + '.png'
plot_path = 'figures/' + 'x_plot_main_0' + '.txt'

import time
import matplotlib.pyplot as plt
import random
from collections import deque
import numpy as np
import torch


from x_block_functions import SimpleDrivingSimulator, create_initial_policy, collect_rollouts, train_dynamics_model, train_risk_predictor, RewardEmbeddingModel
from x_block_functions import construct_reference_distribution, update_current_distribution, detect_distribution_shift, shift_risk_trigger, describe_current_behavior
from x_block_functions import construct_environment_information, AutoReward_LLM, construct_reward, compute_reward, update_policy, save_policy
from x_block_functions import save_checkpoint, checkpoint_path
from x_block_functions import STATE_DIM, ACTION_DIM, NUM_REFERENCE_EPISODES, NUM_ONLINE_EPISODES, MAX_STEPS, RISK_HORIZON, TAU_MMD, TAU_RISK, DISTRIBUTION_WINDOW, EMBEDDING_DIM, MODEL_PATH, DEVICE



# ============================================================
# 0. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 1. Configuration from x_block_function.py
# ============================================================


# ============================================================
# 23. Main AutoReward OOD Adaptive RL
# ============================================================

def AROOD(
    simulator,
    policy
):

    reward_List = []
    adaptation_count_List = []
    time_List = []
    risk_List = []


    # ========================================================
    # PHASE I
    # Reference Data Collection
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "PHASE I: Collecting reference rollouts"
    )

    print("=" * 60)

    rollouts = collect_rollouts(

        simulator=simulator,

        policy=policy,

        num_episodes= NUM_REFERENCE_EPISODES,

        max_steps= MAX_STEPS
    )


    # ========================================================
    # PHASE II
    # Dynamics Model
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "PHASE II: Training DynamicsModel"
    )

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

    print(
        "PHASE III: Training RiskPredictor"
    )

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

    print(
        "PHASE IV: Constructing reference distribution"
    )

    print("=" * 60)

    reference_distribution = construct_reference_distribution(
            rollouts
        )

    print(
        "Reference samples:",
        len(reference_distribution)
    )

    current_buffer = deque(
        maxlen=DISTRIBUTION_WINDOW
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding_model = RewardEmbeddingModel(
        EMBEDDING_DIM
    )




    # ========================================================
    # PHASE V
    # Online Adaptive RL
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "PHASE V: Online adaptive RL"
    )

    print("=" * 60)



    for episode in range(
        NUM_ONLINE_EPISODES
    ):

        start = time.time()




        # ----------------------------------------------------
        # Gradually introduce an OOD condition.
        #
        # This exists only for the executable demonstration.
        # A real CARLA environment should control its own
        # scenario distribution.
        # ----------------------------------------------------

        if hasattr(
            simulator,
            "condition"
        ):

            simulator.condition = min(1.0, episode/ max(1, NUM_ONLINE_EPISODES // 2))

        state = simulator.reset()

        # Gradually introduce changed
        # environmental conditions for demonstration.
        simulator.condition = (episode/ max(NUM_ONLINE_EPISODES - 1, 1))




        current_distribution = deque(
            maxlen=DISTRIBUTION_WINDOW
        )

        episode_reward = 0.0

        adaptation_count = 0
        risk_score_count = 0

        for t in range(MAX_STEPS):

            # =================================================
            # STEP 1
            # Policy
            # =================================================

            action = policy(
                state
            )


            # =================================================
            # STEP 2
            # DynamicsModel
            #
            # Predict:
            #
            #     s_hat_{t+1}
            #     = f_phi(s_t, a_t)
            # =================================================

            next_state_pred = dynamics_model.predict(state, action)


            # =================================================
            # STEP 3
            # ACTUAL simulator transition
            #
            # This is deliberately performed before MMD
            # because MMD compares actual transition
            # distributions.
            # =================================================

            (next_state, simulator_reward, done, safety_violation) = simulator.step(action)


            # =================================================
            # STEP 4
            # Current actual transition distribution
            # =================================================

            update_current_distribution(current_distribution, state, action, next_state)


            # =================================================
            # STEP 5
            # MMD OOD Detection
            # =================================================

            (shift_score, shift_flag) = detect_distribution_shift(reference_distribution, list(current_distribution), TAU_MMD)


            # =================================================
            # STEP 6
            # Risk Prediction
            #
            # Q_risk =
            # q_omega(
            #     s_t,
            #     a_t,
            #     s_hat_{t+1}
            # )
            # =================================================

            risk = risk_model.predict(state, action, next_state_pred)

            risk_score = float(risk.item())

            if risk_score > TAU_RISK:
                risk_score_count += 1


            # =================================================
            # STEP 7
            # Shift-Risk Trigger
            # =================================================

            adaptation_trigger = shift_risk_trigger(
                    shift_score=shift_score,
                    risk=risk_score,
                    tau_MMD=TAU_MMD,
                    tau_risk=TAU_RISK
                )





            # =================================================
            # STEP 8
            # AutoReward adaptation
            # =================================================



            if adaptation_trigger:

                adaptation_count += 1

                driving_context = construct_environment_information(
                        state,
                        action,
                        next_state_pred,
                        shift_score,
                        risk_score
                    )

                reward_specification = AutoReward_LLM(
                        driving_context
                    )

                reward_function = construct_reward(
                        reward_specification,
                        embedding_model
                    )

                print(
                    f"[Episode {episode + 1:03d} | "
                    f"Step {t:03d}] "
                    f"shift_score={shift_score:.4f}, "
                    f"risk_score={risk_score:.4f} "
                    f"-> Adapt"
                )
            else:
                driving_context = construct_environment_information(
                    state,
                    action,
                    next_state_pred,
                    shift_score,
                    risk_score
                )
                reward_specification = AutoReward_LLM(driving_context)
                reward_function = construct_reward(reward_specification, embedding_model)

                print(
                    f"[Episode {episode + 1:03d} | "
                    f"Step {t:03d}] "
                    f"shift_score={shift_score:.4f}, "
                    f"risk_score={risk_score:.4f} "
                    f"-> Not Adapt")




#--------------------------------------------------------------------------
            # ------------------------------------------------
            # 10. Generate semantic behavior description
            # ------------------------------------------------

            behavior_description = (
                describe_current_behavior(
                    state,
                    action,
                    next_state,
                    risk_score
                )
            )

            # ------------------------------------------------
            # 11. Embedding + cosine-similarity reward
            # ------------------------------------------------

            reward = compute_reward(
                reward_function,
                behavior_description,
                safety_violation,
                action
            )
#--------------------------------------------------------------------------



            # =================================================
            # STEP 10
            # RL Policy Update
            # =================================================

            update_policy(
                policy,
                state,
                action,
                reward,    # 加入 cosine_similarity 前是 rl_reward
                next_state,
                done
            )


            # =================================================
            # STEP 11
            # Move to actual next state
            # =================================================

            state = next_state

            episode_reward += reward     # 加入 cosine_similarity 前是 rl_reward


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



        print(
            f"Online episode "
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

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Selected DEVICE:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


    print("\n" + "=" * 100)

    print(
        "AROOD:"
    )

    print("=" * 30)

    print(
        f"State dimension : {STATE_DIM}"
    )

    print(
        f"Action dimension: {ACTION_DIM}"
    )

    print(
        f"MMD threshold   : {TAU_MMD}"
    )

    print(
        f"Risk threshold  : {TAU_RISK}"
    )


    # --------------------------------------------------------
    # Initialize simulator
    # --------------------------------------------------------

    simulator = SimpleDrivingSimulator()           # create_carla_simulator()
    simulator.reset()


    # --------------------------------------------------------
    # Initialize policy
    # --------------------------------------------------------

    policy = create_initial_policy(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM
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
    save_checkpoint(checkpoint_path, policy, dynamics_model, risk_predictor, reference_distribution)

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

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from model import create_dense_model, setup_optimizer
from cube_agent import CubeAgent

SAVE_DIR = "/content/drive/My Drive/Spring 2025/DS3001 Reinforcement Learning/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_rubiks_agent(iterations=50, lr=1e-3, debug=False):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(SAVE_DIR, f"rubiks_model_{timestamp}.keras")

    state_dim = 20 * 24
    model = create_dense_model(state_dim)
    model = setup_optimizer(model, lr)

    depth = 5 if debug else 10
    batch_size = 3 if debug else 5

    all_losses = []
    avg_rewards = []

    for it in range(iterations):
        print(f"Iteration {it + 1}/{iterations}...")

        cube_agent = CubeAgent(max_depth=depth, batch_size=batch_size)
        cube_agent.scramble_cubes_for_data()
        cubes = np.array(cube_agent.env).flatten()

        encoded_states = np.empty((batch_size * depth, state_dim))
        values = np.empty((batch_size * depth, 1))
        policies = np.empty(batch_size * depth)
        rewards_per_state = np.empty(batch_size * depth)

        i = 0
        for state in cubes:
            values_for_state = np.zeros(len(state.action_space))
            immediate_rewards = []
            encoded_states[i] = state.get_one_hot_state().flatten()
            actions = state.action_space
            start_state = state.cube.copy()

            for j, action in enumerate(actions):
                _, reward = state.step(j)
                immediate_rewards.append(reward)
                child_state_encoded = state.get_one_hot_state().flatten()
                state.set_state(start_state)

                value, policy = model.predict(child_state_encoded[None, :], verbose=0)
                values_for_state[j] = value[0][0] + reward

            values[i] = values_for_state.max()
            policies[i] = values_for_state.argmax()
            rewards_per_state[i] = np.max(immediate_rewards)
            i += 1

        history = model.fit(
            encoded_states,
            {"output_policy": policies, "output_value": values},
            epochs=3,
            verbose=1
        )

        all_losses.extend(history.history['loss'])
        avg_rewards.append(np.mean(rewards_per_state))

        model.save(model_path)
        print(f"Model saved to: {model_path}")

    # Combined loss and reward plot
    plt.figure(figsize=(8, 6))
    plt.plot(all_losses, label='Loss per epoch')
    xs = np.linspace(0, len(all_losses), len(avg_rewards))
    plt.plot(xs, avg_rewards, label='Avg reward per iteration')
    plt.legend()
    plt.title("Training Metrics")

    metrics_path = os.path.join(SAVE_DIR, f"training_metrics_{timestamp}.png")
    plt.savefig(metrics_path)
    plt.close()

    print(f"Training plot saved to: {metrics_path}")

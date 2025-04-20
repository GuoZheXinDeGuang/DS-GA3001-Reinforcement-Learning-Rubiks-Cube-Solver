import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from model      import create_dense_model, setup_optimizer
from cube_agent import CubeAgent

# Directory to save models and plots
SAVE_DIR = "/content/drive/My Drive/Spring 2025/DS3001 Reinforcement Learning/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_rubiks_agent(iterations: int = 50,
                       lr: float = 1e-3,
                       debug: bool = False) -> None:
    """
    Train a Rubik's Cube policy-value network, save model checkpoints,
    and output a combined loss-and-reward plot.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    state_dim  = 20 * 24  # flattened one-hot size

    # Build and compile the network
    net = create_dense_model(state_dim)
    net = setup_optimizer(net, lr)

    # Debug mode reduces depth and batch size
    depth      = 5 if debug else 10
    batch_size = 3 if debug else 5

    all_losses  = []
    avg_rewards = []

    for it in range(1, iterations + 1):
        print(f"Iteration {it}/{iterations} ...")
        agent = CubeAgent(max_depth=depth, batch_size=batch_size)
        envs  = agent.collect()  # List of RubiksEnv instances

        # Collect state vectors and rewards
        states  = np.stack([env.reset(0) for env in envs])
        rewards = np.array([env.reward()    for env in envs])
        actions = np.zeros(len(envs), int)  # dummy policy labels

        history = net.fit(
            states,
            {'value': rewards, 'policy': actions},
            epochs=3,
            verbose=1
        )

        all_losses.extend(history.history['loss'])
        avg_rewards.append(rewards.mean())

        # Save model checkpoint each iteration
        model_file = os.path.join(SAVE_DIR, f"model_{timestamp}_it{it}.keras")
        net.save(model_file)
        print(f"Saved model to: {model_file}")

    # Create and save combined loss/reward plot
    plt.figure(figsize=(8, 6))
    plt.plot(all_losses, label='Loss per epoch')
    xs = np.linspace(0, len(all_losses), len(avg_rewards))
    plt.plot(xs, avg_rewards, label='Avg reward per iteration')
    plt.legend()
    plt.title("Training Metrics")
    out = os.path.join(SAVE_DIR, f"training_metrics_{timestamp}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved training plot to: {out}")

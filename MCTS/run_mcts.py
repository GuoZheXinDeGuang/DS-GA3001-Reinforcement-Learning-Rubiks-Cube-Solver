
import time
from tensorflow import keras

from cube_env import RubiksEnv, Move
from mcts     import MCTS

if __name__ == "__main__":
    # 1) Load Keras and your pretrained model
    print("Keras version:", keras.__version__)
    MODEL_PATH = "/content/drive/My Drive/Spring 2025/DS3001 Reinforcement Learning/saved_models/your_model_name.keras"
    model = keras.models.load_model(MODEL_PATH)

    # 2) Wrap it in MCTS
    mcts = MCTS(model)

    # 3) Build a RubiksEnv and apply your custom scramble
    SCRAMBLE = "R U R'"
    cube = RubiksEnv()              # starts solved
    cube.cube(SCRAMBLE)             # apply the exact formula
    cube.last_formula = SCRAMBLE

    cube.render()
    print("Scramble formula:", cube.last_formula)

    # 4) Run MCTS until it returns a non‑None action sequence
    start = time.time()
    solution = None
    while solution is None:
        solution = mcts.train(cube)
    duration = time.time() - start
    print(f"{duration:.2f}s – MCTS solution:", [Move(i).value for i in solution])

    # 5) As a sanity‑check, do a BFS over the explored tree
    start = time.time()
    bfs_sol = mcts.bfs(cube)
    duration = time.time() - start
    print(f"{duration:.2f}s – BFS fallback :", [Move(i).value for i in bfs_sol])

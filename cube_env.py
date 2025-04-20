# cube_env.py

import numpy as np
import pycuber as pc
from enum import Enum

class Move(Enum):
    F  = "F"
    F_ = "F'"
    B  = "B"
    B_ = "B'"
    L  = "L"
    L_ = "L'"
    R  = "R"
    R_ = "R'"
    U  = "U"
    U_ = "U'"
    D  = "D"
    D_ = "D'"

def one_hot_encode(cube: pc.Cube) -> np.ndarray:
    """
    Encode corners (20×3) and edges (12×2) into a flat 480-dimensional vector.
    """
    corner_sets = [frozenset(s) for s in [
        {'white','orange','blue'},{'white','orange','green'},{'white','red','green'},{'white','red','blue'},
        {'yellow','orange','blue'},{'yellow','orange','green'},{'yellow','red','green'},{'yellow','red','blue'}
    ]]
    edge_sets = [frozenset(s) for s in [
        {'white','blue'},{'white','orange'},{'white','green'},{'white','red'},{'yellow','blue'},{'yellow','orange'},
        {'yellow','green'},{'yellow','red'},{'orange','blue'},{'blue','red'},{'green','red'},{'orange','green'}
    ]]
    colour_map = {'white':0,'yellow':0,'blue':1,'green':1,'orange':2,'red':2}

    arr = np.zeros((20,24), dtype=int)
    idx = 0

    # corners
    corners = sorted(cube.select_type('corner'), key=lambda p: sorted(p.facings.keys()))
    for c in corners:
        faces = sorted(c.facings.keys())
        colours = frozenset(c.facings[f].colour for f in faces)
        base = 3 * corner_sets.index(colours)
        ori = colour_map[c.facings[faces[0]].colour]
        arr[idx, base+ori] = 1
        idx += 1

    # edges
    edges = sorted(cube.select_type('edge'), key=lambda p: sorted(p.facings.keys()))
    for e in edges:
        faces = sorted(e.facings.keys())
        colours = frozenset(e.facings[f].colour for f in faces)
        base = 2 * edge_sets.index(colours)
        ori = colour_map[e.facings[faces[0]].colour]
        arr[idx, base+ori] = 1
        idx += 1

    return arr.flatten()


class RubiksEnv:
    """
    Rubik's Cube environment with configurable scramble and reward.
    """
    def __init__(self, scramble_moves: int=0, use_oll_pll: bool=False, seed: int=None):
        if seed is not None:
            np.random.seed(seed)
        self._solved = pc.Cube()
        self.cube = self._solved.copy()
        self.last_formula = ''
        if scramble_moves > 0:
            self.scramble(scramble_moves)
        if use_oll_pll:
            self.scramble_oll_pll()

    def scramble(self, moves: int) -> None:
        seq = [np.random.choice(list(Move)).value for _ in range(moves)]
        self.last_formula = ' '.join(seq)
        self.cube(self.last_formula)

    def scramble_oll_pll(self) -> None:
        formula = "R U R' U R U2 R'"
        self.last_formula = formula
        self.cube(formula)

    def reset(self, moves: int=0) -> np.ndarray:
        self.cube = self._solved.copy()
        if moves > 0:
            self.scramble(moves)
        return one_hot_encode(self.cube)

    def step(self, move_idx: int) -> (np.ndarray, int):
        move = list(Move)[move_idx].value
        self.cube.perform_step(move)
        return one_hot_encode(self.cube), self.reward()

    def reward(self) -> int:
        return 1 if self.cube == self._solved else -1

    def is_solved(self) -> bool:
        return self.cube == self._solved

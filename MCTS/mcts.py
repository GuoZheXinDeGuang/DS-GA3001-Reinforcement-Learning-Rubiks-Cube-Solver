

import math
import random
from collections import deque
from copy import deepcopy

import numpy as np

from cube_env import RubiksEnv, one_hot_encode, Move


class MCTS:
    def __init__(self, model, loss_constant=150, exploration_constant=4):
        """
        model: a Keras Model that takes shape (1,480) inputs
               and returns [value, policy_logits]
        """
        self.model = model

        # Map RubiksEnv -> ([children], policy, visits, losses, best_reward)
        self.children_and_data = {}

        self.LOSS_C = loss_constant
        self.EXPLORE_C = exploration_constant

    def train(self, root: RubiksEnv):
        path, actions, leaf = self.traverse(root)
        reward = self.expand(leaf)
        self.backpropagate(path, actions, reward)

        # if any child is solved, return the action sequence
        for idx, child in enumerate(self.children_and_data[leaf][0]):
            if child.is_solved():
                return actions + [idx]
        return None

    def traverse(self, state: RubiksEnv):
        path = []
        actions = []
        current = state

        while True:
            data = self.children_and_data.get(current)
            if not data or not data[0]:
                # no children yet
                return path, actions, current

            children, _, visits, _, best_rewards = data
            total_visits = sum(visits)

            # if nobody has been visited, pick random
            if total_visits == 0:
                a = random.randrange(len(children))
            else:
                # UCT-style score
                scores = []
                for i in range(len(children)):
                    U = (self.EXPLORE_C
                         * data[1][i]
                         * math.sqrt(total_visits) 
                         / (1 + visits[i]))
                    W = best_rewards[i]
                    P = - data[3][i]         # penalize by accumulated loss
                    scores.append(U + W + P)
                a = max(range(len(scores)), key=scores.__getitem__)

            # descend
            path.append(current)
            actions.append(a)
            # apply a small loss to discourage loops
            data[3][a] += self.LOSS_C  
            current = children[a]

    def expand(self, state: RubiksEnv):
        # get prediction on the one-hot encoding
        vec = one_hot_encode(state.cube)[None, :]   # shape (1,480)
        value, policy_logits = self.model.predict(vec, verbose=0)

        # generate all one-step children
        kids = []
        for i in range(len(Move)):
            child = deepcopy(state)
            child.step(i)     # modifies child.cube
            kids.append(child)

        # initialize visit counts, loss penalties, best_reward caches
        n_actions = len(kids)
        self.children_and_data[state] = (
            kids,
            policy_logits[0],          # length‑12
            [0]*n_actions,             # visits
            [0]*n_actions,             # loss penalties
            [float('-inf')]*n_actions  # best observed reward
        )
        return float(value[0][0])

    def backpropagate(self, path, actions, reward):
        # update best_reward and visits up the tree
        for state, a in zip(path, actions):
            kids, policy, visits, losses, best_rewards = self.children_and_data[state]
            visits[a] += 1
            best_rewards[a] = max(best_rewards[a], reward)

    def bfs(self, root: RubiksEnv):
        """
        Optional breadth-first search to verify reachability.
        """
        if root not in self.children_and_data:
            return None

        parent = {root: (None, None)}
        q = deque([root])
        solved = None

        while q:
            cur = q.popleft()
            if cur.is_solved():
                solved = cur
                break
            children = self.children_and_data[cur][0]
            for i, child in enumerate(children):
                if child not in parent:
                    parent[child] = (cur, i)
                    q.append(child)

        if solved is None:
            return None

        # reconstruct action path
        acts = []
        node = solved
        while parent[node][0] is not None:
            node, act = parent[node]
            acts.append(act)
        return list(reversed(acts))

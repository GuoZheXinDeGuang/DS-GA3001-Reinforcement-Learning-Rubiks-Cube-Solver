
from cube_env import RubiksEnv
class CubeAgent:
    """
    Maintains a grid of RubiksEnv instances for data collection.
    """
    def __init__(self, number_of_cubes=3, batch_size=10, seed=None):
        self.number_of_cubes = number_of_cubes
        self.batch_size      = batch_size

        # Initialize a matrix of environments
        self.env = [
            [RubiksEnv(seed=seed) for _ in range(batch_size)]
            for _ in range(number_of_cubes)
        ]

    def scramble_cubes_for_data(self, scramble_depth: int = 1):
        """
        For each 'layer' of cubes, copy the previous cube's state
        then apply `scramble_depth` random moves to produce a new sample.
        """
        for i in range(self.number_of_cubes):
            for j in range(1, self.batch_size):
                # copy prior cube state
                self.env[i][j].cube = self.env[i][j-1].cube.copy()
                # apply additional random moves
                self.env[i][j].scramble(scramble_depth)

    def reset_envs(self, scramble_moves: Sequence[int] = None):
        """
        Reset each environment to solved state and then scramble it
        by the specified number of moves. If no list is provided,
        all are reset with zero moves.
        """
        if scramble_moves is None:
            scramble_moves = [0] * self.number_of_cubes

        for i, moves in enumerate(scramble_moves):
            # reset to solved, then scramble `moves` times
            self.env[i][0].reset(moves)
            # re‐apply to entire batch in row i
            for j in range(1, self.batch_size):
                # copy the freshly scrambled reference
                self.env[i][j].cube = self.env[i][0].cube.copy()

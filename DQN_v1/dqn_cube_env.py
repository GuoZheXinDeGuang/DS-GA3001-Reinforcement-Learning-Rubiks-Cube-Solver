# dqn_cube_env.py
# Fixed version of the original cube_env.py environment, specifically designed for DQN training

import numpy as np
import pycuber as pc
from enum import Enum
from typing import Tuple, List, Dict, Any

# Keep the same action definitions as the original environment
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

# Define color mapping, ensuring indices are always within valid range
CORNER_COLORS = [
    {'white','orange','blue'}, {'white','orange','green'}, 
    {'white','red','green'}, {'white','red','blue'},
    {'yellow','orange','blue'}, {'yellow','orange','green'}, 
    {'yellow','red','green'}, {'yellow','red','blue'}
]

EDGE_COLORS = [
    {'white','blue'}, {'white','orange'}, {'white','green'}, {'white','red'},
    {'yellow','blue'}, {'yellow','orange'}, {'yellow','green'}, {'yellow','red'},
    {'orange','blue'}, {'blue','red'}, {'green','red'}, {'orange','green'}
]

COLOR_MAP = {
    'white': 0, 'yellow': 0,
    'blue': 1, 'green': 1,
    'orange': 2, 'red': 2
}

def enhanced_one_hot_encode(cube: pc.Cube) -> np.ndarray:
    """
    Enhanced version of the cube state encoding function, fixed index out of bounds issues
    Encodes cube's corner and edge pieces into a one-dimensional vector
    """
    # Use a larger feature vector space to ensure no out of bounds
    feature_size = 480  # 20 x 24
    state_vector = np.zeros(feature_size)
    
    # Convert frozenset to list for easier lookup
    corner_sets = [frozenset(colors) for colors in CORNER_COLORS]
    edge_sets = [frozenset(colors) for colors in EDGE_COLORS]
    
    # Process corner pieces
    try:
        corners = sorted(cube.select_type('corner'), key=lambda p: sorted(p.facings.keys()))
        for i, corner in enumerate(corners):
            if i >= 8:  # Cube only has 8 corner pieces
                break
                
            # Get corner piece color set
            faces = sorted(corner.facings.keys())
            colours = frozenset(corner.facings[f].colour for f in faces)
            
            # Find the index of this corner piece in predefined sets
            if colours in corner_sets:
                # Calculate position in feature vector
                base_idx = 3 * corner_sets.index(colours)
                # Get orientation information
                first_color = corner.facings[faces[0]].colour
                ori = COLOR_MAP.get(first_color, 0)
                
                # Ensure index is within valid range
                feature_idx = i * 24 + base_idx + ori
                if 0 <= feature_idx < feature_size:
                    state_vector[feature_idx] = 1
    except Exception as e:
        print(f"Error processing corner pieces: {e}")
    
    # Process edge pieces
    try:
        edges = sorted(cube.select_type('edge'), key=lambda p: sorted(p.facings.keys()))
        for i, edge in enumerate(edges):
            if i >= 12:  # Cube only has 12 edge pieces
                break
                
            # Get edge piece color set
            faces = sorted(edge.facings.keys())
            colours = frozenset(edge.facings[f].colour for f in faces)
            
            # Find the index of this edge piece in predefined sets
            if colours in edge_sets:
                # Calculate position in feature vector
                base_idx = 2 * edge_sets.index(colours)
                # Get orientation information
                first_color = edge.facings[faces[0]].colour
                ori = COLOR_MAP.get(first_color, 0)
                
                # Ensure index is within valid range
                feature_idx = (i + 8) * 24 + base_idx + ori  # After 8 corner pieces
                if 0 <= feature_idx < feature_size:
                    state_vector[feature_idx] = 1
    except Exception as e:
        print(f"Error processing edge pieces: {e}")
    
    return state_vector

class DQNRubiksEnv:
    """
    Rubik's Cube environment specifically designed for DQN training, with enhanced stability and error handling
    """
    def __init__(self, scramble_moves: int=0, seed: int=None):
        """
        Initialize the Rubik's Cube environment
        
        Parameters:
            scramble_moves: Number of initial scramble moves
            seed: Random number seed
        """
        if seed is not None:
            np.random.seed(seed)
            
        self._solved = pc.Cube()  # Save solved state
        self.cube = self._solved.copy()  # Current state
        self.last_moves = []  # Record last move sequence
        self.move_count = 0  # Record total move count
        
        # If initial scramble moves are specified, perform scramble
        if scramble_moves > 0:
            self.scramble(scramble_moves)
    
    def scramble(self, moves: int) -> None:
        """
        Randomly scramble the cube
        
        Parameters:
            moves: Number of scramble moves
        """
        if moves <= 0:
            return
            
        self.last_moves = []
        try:
            # Generate random move sequence
            move_sequence = []
            for _ in range(moves):
                move = np.random.choice(list(Move))
                move_sequence.append(move.value)
                self.last_moves.append(move)
            
            # Execute moves
            formula = ' '.join(move_sequence)
            self.cube(formula)
            self.move_count += moves
        except Exception as e:
            print(f"Error scrambling cube: {e}")
            # Reset to solved state on error
            self.cube = self._solved.copy()
            self.last_moves = []
    
    def reset(self, scramble_moves: int=0) -> np.ndarray:
        """
        Reset environment to initial state
        
        Parameters:
            scramble_moves: Number of scramble moves after reset
            
        Returns:
            Cube state vector
        """
        self.cube = self._solved.copy()
        self.last_moves = []
        self.move_count = 0
        
        if scramble_moves > 0:
            self.scramble(scramble_moves)
            
        return enhanced_one_hot_encode(self.cube)
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one action step
        
        Parameters:
            action_idx: Action index (0-11)
            
        Returns:
            (next_state, reward, done, info)
        """
        # Validate if action is legal
        if not 0 <= action_idx < len(Move):
            print(f"Warning: Invalid action index {action_idx}")
            action_idx = 0  # Use default action
        
        done = False
        info = {"success": False, "move_count": self.move_count + 1}
        
        try:
            # Get corresponding move
            move = list(Move)[action_idx]
            self.last_moves.append(move)
            
            # Execute move
            self.cube.perform_step(move.value)
            self.move_count += 1
            
            # Check if solved
            done = self.is_solved()
            if done:
                info["success"] = True
                
            # Calculate reward
            reward = self._compute_reward(done)
            
            # Get next state
            next_state = enhanced_one_hot_encode(self.cube)
            
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"Error executing action: {e}")
            # On error, return current state without changes
            return enhanced_one_hot_encode(self.cube), -1.0, False, info
    
    def _compute_reward(self, solved: bool) -> float:
        """
        Calculate reward function
        
        Parameters:
            solved: Whether the cube is solved
            
        Returns:
            Reward value
        """
        if solved:
            # High reward for solving the cube
            return 100.0
        else:
            # Small penalty for regular moves to encourage faster solving
            return -0.1
    
    def is_solved(self) -> bool:
        """
        Check if cube is solved
        
        Returns:
            Whether in solved state
        """
        return self.cube == self._solved
    
    def get_action_space_size(self) -> int:
        """
        Get action space size
        
        Returns:
            Size of action space
        """
        return len(Move)
    
    def get_state_size(self) -> int:
        """
        Get state space size
        
        Returns:
            Dimension of state vector
        """
        return 480  # 20 x 24

# Add compatibility layer with original environment for easy code migration
class RubiksEnv(DQNRubiksEnv):
    """
    Interface compatible with original RubiksEnv in cube_env.py
    """
    def step(self, move_idx: int) -> Tuple[np.ndarray, float]:
        """
        Execute move and return state and reward
        """
        next_state, reward, _, _ = super().step(move_idx)
        return next_state, reward
    
    def reward(self) -> float:
        """
        Get current state reward
        """
        return 1.0 if self.is_solved() else -0.1

# For compatibility, export same symbols
one_hot_encode = enhanced_one_hot_encode 

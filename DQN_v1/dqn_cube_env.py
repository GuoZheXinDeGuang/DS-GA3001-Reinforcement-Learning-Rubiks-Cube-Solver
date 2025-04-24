# dqn_cube_env.py
# 基于原始cube_env.py的修复版环境，专为DQN训练设计

import numpy as np
import pycuber as pc
from enum import Enum
from typing import Tuple, List, Dict, Any

# 保持与原始环境相同的动作定义
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

# 定义颜色映射，确保索引始终在有效范围内
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
    增强版的魔方状态编码函数，修复了索引越界问题
    将魔方的角块和棱块编码为一维向量
    """
    # 使用更大的特征向量空间，确保不会越界
    feature_size = 480  # 20 x 24
    state_vector = np.zeros(feature_size)
    
    # 将frozenset转换为列表，以便更容易进行查找
    corner_sets = [frozenset(colors) for colors in CORNER_COLORS]
    edge_sets = [frozenset(colors) for colors in EDGE_COLORS]
    
    # 处理角块
    try:
        corners = sorted(cube.select_type('corner'), key=lambda p: sorted(p.facings.keys()))
        for i, corner in enumerate(corners):
            if i >= 8:  # 魔方只有8个角块
                break
                
            # 获取角块的颜色集合
            faces = sorted(corner.facings.keys())
            colours = frozenset(corner.facings[f].colour for f in faces)
            
            # 查找该角块在预定义集合中的索引
            if colours in corner_sets:
                # 计算特征向量中的位置
                base_idx = 3 * corner_sets.index(colours)
                # 获取方向信息
                first_color = corner.facings[faces[0]].colour
                ori = COLOR_MAP.get(first_color, 0)
                
                # 确保索引在有效范围内
                feature_idx = i * 24 + base_idx + ori
                if 0 <= feature_idx < feature_size:
                    state_vector[feature_idx] = 1
    except Exception as e:
        print(f"处理角块时出错: {e}")
    
    # 处理棱块
    try:
        edges = sorted(cube.select_type('edge'), key=lambda p: sorted(p.facings.keys()))
        for i, edge in enumerate(edges):
            if i >= 12:  # 魔方只有12个棱块
                break
                
            # 获取棱块的颜色集合
            faces = sorted(edge.facings.keys())
            colours = frozenset(edge.facings[f].colour for f in faces)
            
            # 查找该棱块在预定义集合中的索引
            if colours in edge_sets:
                # 计算特征向量中的位置
                base_idx = 2 * edge_sets.index(colours)
                # 获取方向信息
                first_color = edge.facings[faces[0]].colour
                ori = COLOR_MAP.get(first_color, 0)
                
                # 确保索引在有效范围内
                feature_idx = (i + 8) * 24 + base_idx + ori  # 8个角块之后
                if 0 <= feature_idx < feature_size:
                    state_vector[feature_idx] = 1
    except Exception as e:
        print(f"处理棱块时出错: {e}")
    
    return state_vector

class DQNRubiksEnv:
    """
    专为DQN设计的魔方环境，增强了稳定性和错误处理
    """
    def __init__(self, scramble_moves: int=0, seed: int=None):
        """
        初始化魔方环境
        
        参数:
            scramble_moves: 初始打乱的步数
            seed: 随机数种子
        """
        if seed is not None:
            np.random.seed(seed)
            
        self._solved = pc.Cube()  # 保存解决状态
        self.cube = self._solved.copy()  # 当前状态
        self.last_moves = []  # 记录最后的移动序列
        self.move_count = 0  # 记录总移动次数
        
        # 如果指定了初始打乱步数，则执行打乱
        if scramble_moves > 0:
            self.scramble(scramble_moves)
    
    def scramble(self, moves: int) -> None:
        """
        随机打乱魔方
        
        参数:
            moves: 打乱的步数
        """
        if moves <= 0:
            return
            
        self.last_moves = []
        try:
            # 生成随机移动序列
            move_sequence = []
            for _ in range(moves):
                move = np.random.choice(list(Move))
                move_sequence.append(move.value)
                self.last_moves.append(move)
            
            # 执行移动
            formula = ' '.join(move_sequence)
            self.cube(formula)
            self.move_count += moves
        except Exception as e:
            print(f"打乱魔方时出错: {e}")
            # 出错时重置为解决状态
            self.cube = self._solved.copy()
            self.last_moves = []
    
    def reset(self, scramble_moves: int=0) -> np.ndarray:
        """
        重置环境到初始状态
        
        参数:
            scramble_moves: 重置后打乱的步数
            
        返回:
            魔方的状态向量
        """
        self.cube = self._solved.copy()
        self.last_moves = []
        self.move_count = 0
        
        if scramble_moves > 0:
            self.scramble(scramble_moves)
            
        return enhanced_one_hot_encode(self.cube)
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        参数:
            action_idx: 动作的索引 (0-11)
            
        返回:
            (next_state, reward, done, info)
        """
        # 验证动作是否合法
        if not 0 <= action_idx < len(Move):
            print(f"警告: 无效的动作索引 {action_idx}")
            action_idx = 0  # 使用默认动作
        
        done = False
        info = {"success": False, "move_count": self.move_count + 1}
        
        try:
            # 获取对应的移动
            move = list(Move)[action_idx]
            self.last_moves.append(move)
            
            # 执行移动
            self.cube.perform_step(move.value)
            self.move_count += 1
            
            # 检查是否解决
            done = self.is_solved()
            if done:
                info["success"] = True
                
            # 计算奖励
            reward = self._compute_reward(done)
            
            # 获取下一个状态
            next_state = enhanced_one_hot_encode(self.cube)
            
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"执行动作时出错: {e}")
            # 出错时不改变状态，返回当前状态
            return enhanced_one_hot_encode(self.cube), -1.0, False, info
    
    def _compute_reward(self, solved: bool) -> float:
        """
        计算奖励函数
        
        参数:
            solved: 魔方是否已解决
            
        返回:
            奖励值
        """
        if solved:
            # 成功解决魔方，给予高奖励
            return 100.0
        else:
            # 普通步骤，给予小惩罚以鼓励尽快解决
            return -0.1
    
    def is_solved(self) -> bool:
        """
        检查魔方是否已解决
        
        返回:
            是否处于解决状态
        """
        return self.cube == self._solved
    
    def get_action_space_size(self) -> int:
        """
        获取动作空间大小
        
        返回:
            动作空间的大小
        """
        return len(Move)
    
    def get_state_size(self) -> int:
        """
        获取状态空间大小
        
        返回:
            状态向量的维度
        """
        return 480  # 20 x 24

# 添加与原始环境的兼容层，方便现有代码迁移
class RubiksEnv(DQNRubiksEnv):
    """
    与原始cube_env.py中的RubiksEnv兼容的接口
    """
    def step(self, move_idx: int) -> Tuple[np.ndarray, float]:
        """
        执行移动并返回状态和奖励
        """
        next_state, reward, _, _ = super().step(move_idx)
        return next_state, reward
    
    def reward(self) -> float:
        """
        获取当前状态的奖励
        """
        return 1.0 if self.is_solved() else -0.1

# 为了兼容性，导出相同的符号
one_hot_encode = enhanced_one_hot_encode 

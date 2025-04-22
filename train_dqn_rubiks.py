import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import time  # 添加时间模块用于跟踪执行时间

# 导入新的环境文件
from dqn_cube_env import DQNRubiksEnv, Move

# 修改保存目录，使用当前目录下的saved_models文件夹
# SAVE_DIR = "/content/drive/My Drive/Spring 2025/DS3001 Reinforcement Learning/saved_models"
SAVE_DIR = "./saved_models"  # 使用当前目录下的saved_models文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

class DQNAgent:
    """
    使用DQN算法的魔方求解智能体
    """
    def __init__(self, state_dim, action_dim, 
                 learning_rate=1e-3, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_min=0.1, 
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=64,
                 update_target_freq=20):
        # 算法超参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        
        # 创建Q网络和目标网络
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        
        # 预编译网络，避免第一次预测时的延迟
        print("预编译网络...")
        dummy_state = np.zeros((1, self.state_dim))
        self.q_network.predict(dummy_state, verbose=0)
        self.target_network.predict(dummy_state, verbose=0)
        
        self.update_target_network()  # 初始化目标网络权重
        
        # 训练计数器
        self.train_step_counter = 0
    
    def _build_model(self):
        """构建深度Q网络"""
        # 使用函数式API创建模型，避免Sequential模型的警告
        inputs = keras.Input(shape=(self.state_dim,))
        x = keras.layers.Dense(1024, activation='relu')(inputs)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        outputs = keras.layers.Dense(self.action_dim, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def update_target_network(self):
        """更新目标网络的权重为当前Q网络的权重"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """将经验存储到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """使用epsilon-greedy策略选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay(self):
        """从经验回放缓冲区中学习"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 从记忆中采样一个批次
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_dim))
        targets = np.zeros((self.batch_size, self.action_dim))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                t = self.target_network.predict(next_state.reshape(1, -1), verbose=0)[0]
                target[action] = reward + self.gamma * np.max(t)
            
            states[i] = state
            targets[i] = target
        
        # 训练网络
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        
        # 增加训练计数器并更新目标网络
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()
            
        return history.history['loss'][0]

def train_rubiks_agent(iterations=50, 
                       episodes_per_iteration=100,
                       max_steps=100, 
                       lr=1e-3, 
                       gamma=0.99,
                       initial_scramble=10,  # 使用正常的打乱深度
                       debug=False):
    """
    使用DQN算法训练魔方求解智能体，保存模型检查点并输出性能图表
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    state_dim = 480  # 使用新环境中的状态维度
    action_dim = len(Move)  # 动作空间大小
    
    # 如果是调试模式，减少训练规模
    if debug:
        episodes_per_iteration = 10
        max_steps = 20
    
    print(f"创建DQN智能体，状态维度: {state_dim}, 动作维度: {action_dim}")
    # 创建DQN智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=lr,
        gamma=gamma,
        buffer_size=50000 if not debug else 1000,
        batch_size=64 if not debug else 8
    )
    
    print(f"创建魔方环境...")
    # 创建环境 - 使用新的DQNRubiksEnv环境
    env = DQNRubiksEnv()
    
    # 用于记录训练过程的指标
    all_rewards = []
    all_losses = []
    success_rates = []
    
    print(f"开始训练，总共{iterations}次迭代...")
    
    for iteration in range(1, iterations + 1):
        start_time = time.time()
        print(f"迭代 {iteration}/{iterations} ...")
        
        iteration_rewards = []
        iteration_success = 0
        iteration_losses = []
        
        for episode in range(episodes_per_iteration):
            # 重置环境并随机打乱魔方
            start_reset = time.time()
            state = env.reset(initial_scramble)
            print(f"  Episode {episode+1}: 环境重置用时 {time.time() - start_reset:.2f}秒")
            
            episode_reward = 0
            solved = False
            
            for step in range(max_steps):
                # 选择动作
                action = agent.choose_action(state)
                
                # 执行动作 - 使用新环境的step方法
                next_state, reward, done, info = env.step(action)
                
                # 将经验存储到回放缓冲区
                agent.remember(state, action, reward, next_state, done)
                
                # 从回放缓冲区中学习
                loss = agent.replay()
                if loss > 0:
                    iteration_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    solved = True
                    break
            
            # 衰减探索率
            agent.decay_epsilon()
            
            # 记录本轮的奖励和成功情况
            iteration_rewards.append(episode_reward)
            if solved:
                iteration_success += 1
                
            # 打印进度
            if (episode + 1) % 5 == 0 or episode == 0:  # 增加日志频率
                print(f"  Episode {episode + 1}/{episodes_per_iteration}, "
                      f"Epsilon: {agent.epsilon:.4f}, "
                      f"Mean Reward: {np.mean(iteration_rewards[-min(5, len(iteration_rewards)):]):.2f}, "
                      f"内存样本: {len(agent.memory)}")
        
        # 确保有数据再计算统计信息
        if iteration_rewards:
            # 计算当前迭代的平均指标
            avg_reward = np.mean(iteration_rewards)
            success_rate = iteration_success / max(1, len(iteration_rewards))
            
            all_rewards.append(avg_reward)
            success_rates.append(success_rate)
            all_losses.extend(iteration_losses)
            
            iter_time = time.time() - start_time
            print(f"  迭代 {iteration} 完成，用时 {iter_time:.2f}秒")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  成功率: {success_rate:.2%}")
            
            # 保存模型检查点
            model_file = os.path.join(SAVE_DIR, f"dqn_model_{timestamp}_it{iteration}.keras")
            agent.q_network.save(model_file)
            print(f"  模型已保存至: {model_file}")
            
            # 每5次迭代生成中间图表
            if iteration % 5 == 0 and all_rewards:
                plot_metrics(all_rewards, success_rates, all_losses, timestamp, iteration)
    
    # 生成最终的训练指标图表
    if all_rewards:
        plot_metrics(all_rewards, success_rates, all_losses, timestamp, iterations, final=True)

def plot_metrics(rewards, success_rates, losses, timestamp, iteration, final=False):
    """绘制和保存训练指标图表"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # 绘制平均奖励
    axs[0].plot(rewards, 'b-')
    axs[0].set_title('每次迭代的平均奖励')
    axs[0].set_xlabel('迭代次数')
    axs[0].set_ylabel('平均奖励')
    axs[0].grid(True)
    
    # 绘制成功率
    axs[1].plot(success_rates, 'g-')
    axs[1].set_title('每次迭代的魔方求解成功率')
    axs[1].set_xlabel('迭代次数')
    axs[1].set_ylabel('成功率')
    axs[1].set_ylim([0, 1])
    axs[1].grid(True)
    
    # 绘制损失
    if losses:
        axs[2].plot(losses, 'r-', alpha=0.5)
        axs[2].set_title('训练损失')
        axs[2].set_xlabel('训练步骤')
        axs[2].set_ylabel('损失')
        axs[2].grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plot_type = "final" if final else f"it{iteration}"
    plot_file = os.path.join(SAVE_DIR, f"dqn_training_metrics_{timestamp}_{plot_type}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"  训练图表已保存至: {plot_file}")

if __name__ == "__main__":
    # 设置调试模式参数用于更快测试
    debug_mode = True  # 修改为True可以加快测试

    # 设置默认参数训练魔方智能体
    train_rubiks_agent(
        iterations=3 if debug_mode else 50,         # 训练迭代次数
        episodes_per_iteration=5 if debug_mode else 100, # 每次迭代的训练回合数
        max_steps=20 if debug_mode else 50,         # 每个回合的最大步数
        lr=0.001,                  # 学习率
        gamma=0.99,                # 折扣因子
        initial_scramble=3 if debug_mode else 10,   # 适当减少打乱深度加快测试
        debug=debug_mode           # 是否开启调试模式
    ) 
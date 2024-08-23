import os, sys
BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEPATH)

from cma import CMAEvolutionStrategy
from Grid import Gridworld, QLearningAgent

import numpy as np
import random
from cma import CMAEvolutionStrategy

times = 0
# 定义奖励特征
def reward_function_1(state, action, hit, env, weights):
    ret = 0
    if state == env.goal:
        ret += weights[0]
    elif hit == True:
        ret += weights[1]
    else:
        ret += weights[2]
    return ret


# 奖励函数权重组合
def combined_reward(state, action, hit, env, weights):
    # return reward_function(state, action, hit, env, weights)
    return reward_function_1(state, action, hit, env, weights)

# 评估函数：以多目标函数进行适应度评估
def evaluate_reward_function(weights, env, episodes=100):
    agent = QLearningAgent(env)
    total_reward = 0
    total_steps = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            global times
            times += 1
            action = agent.choose_action(state)
            next_state, reward, done, hit = env.step(action)
            reward = combined_reward(next_state, action, hit, env, weights)
            # print(f"q table: {agent.q_table}")
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps += 1
        agent.update_epsilon()
        total_reward += reward
        total_steps += steps
        #print(f"episode {episode}, steps: {steps}")
    
    avg_steps = total_steps / episodes
    avg_reward = total_reward / episodes
    return -avg_reward, avg_steps  # 返回两个目标：负奖励和平均步骤数, CMA-ES是最小化目标

# 帕累托排序
def pareto_sort(rewards_and_efforts):
    num_solutions = len(rewards_and_efforts)
    pareto_indices = []

    for i in range(num_solutions):
        dominated = False
        for j in range(num_solutions):
            if i != j:
                # 逐个比较两个目标
                dominates = (
                    (rewards_and_efforts[j][0] <= rewards_and_efforts[i][0] and
                     rewards_and_efforts[j][1] <= rewards_and_efforts[i][1]) and
                    (rewards_and_efforts[j][0] < rewards_and_efforts[i][0] or
                     rewards_and_efforts[j][1] < rewards_and_efforts[i][1])
                )
                if dominates:
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)

    return pareto_indices

# 使用CMA-ES进行多目标优化
env = Gridworld(7, 6, [(4, 0), (4, 1), (4, 3), (4, 4), (4,5),(4,6)], (5, 0), (0, 3))

es = CMAEvolutionStrategy([1, -1, -0.5], 0.01)  # 初始权重和步长
print("Initial population size (number of candidates):", es.popsize)

# 全局保存所有代次的帕累托最优解
pareto_solutions = []

for generation in range(20):  # 运行20代
    
    solutions = es.ask()  # 生成候选解
    print(f"solution: {solutions}")
    rewards_and_efforts = [evaluate_reward_function(weights, env) for weights in solutions]
    print(f"rewards_and_efforts: {rewards_and_efforts}")
    
    # 帕累托排序，获取当前代次的帕累托最优解
    pareto_indices = pareto_sort(rewards_and_efforts)
    
    current_pareto_solutions = [(solutions[i], rewards_and_efforts[i][0], rewards_and_efforts[i][1]) for i in pareto_indices]
    print(f"generation pareto solution: {current_pareto_solutions}")
    
    # 将当前代次的帕累托最优解保存到全局帕累托解集中
    pareto_solutions.extend(current_pareto_solutions)
    
    # 使用当前代次的帕累托前沿上的解进行种群更新
    selected_solutions = [solutions[i] for i in pareto_indices]
    selected_rewards = [rewards_and_efforts[i][0] for i in pareto_indices]
    
    # 如果帕累托前沿的解数量太少，从全局帕累托解中补充种群
    if len(selected_solutions) < es.popsize:
        deficit = es.popsize - len(selected_solutions)
        if len(pareto_solutions) >= deficit:
            extra_solutions = random.sample([s[0] for s in pareto_solutions], deficit)
            extra_rewards = [evaluate_reward_function(weights, env)[0] for weights in extra_solutions]
        else:
            extra_solutions = random.sample(solutions, deficit)
            extra_rewards = [evaluate_reward_function(weights, env)[0] for weights in extra_solutions]
        
        selected_solutions.extend(extra_solutions)
        selected_rewards.extend(extra_rewards)
    
    # 使用选出的解更新CMA-ES的种群
    es.tell(selected_solutions, selected_rewards)

# 在所有代次结束后，从全局帕累托解集中选出最终的帕累托前沿解
final_pareto_indices = pareto_sort([(s[1], s[2]) for s in pareto_solutions])
final_pareto_solutions = [pareto_solutions[i] for i in final_pareto_indices]

# 打印最终的帕累托前沿解的权重、reward 和 effort
print("Final Pareto Optimal Solutions' Reward Weights, Rewards and Efforts:")
for i, (solution, reward, effort) in enumerate(final_pareto_solutions):
    print(f"Solution {i+1}: Weights: {solution}, Reward: {-reward}, Effort (Avg Steps): {effort}")

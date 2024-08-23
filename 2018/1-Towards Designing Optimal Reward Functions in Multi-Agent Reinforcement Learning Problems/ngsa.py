import os, sys
BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEPATH)

import random
import numpy as np
from deap import base, creator, tools, algorithms
from Grid import Gridworld, QLearningAgent

# 定义奖励特征函数
def reward_function_1(state, action, hit, env, weights):
    ret = 0
    if state == env.goal:
        ret += weights[0]
    elif hit == True:
        ret += weights[1]
    else:
        ret += weights[2]
    return ret

# 定义目标函数
def evaluate(individual, env):
    weights = individual  # 权重就是个体本身
    agent = QLearningAgent(env)
    total_reward = 0
    total_steps = 0

    for episode in range(100):  # 使用100个episode来评估
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        while not done and steps < 800:
            action = agent.choose_action(state)
            next_state, _, done, hit = env.step(action)
            reward = reward_function_1(next_state, action, hit, env, weights)
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps += 1
            episode_reward += reward
        agent.update_epsilon()
        total_reward += episode_reward
        total_steps += steps
    
    avg_steps = total_steps / 100
    avg_reward = total_reward / 100
    return -avg_reward, avg_steps  # 返回两个目标：负的平均奖励和平均步数

# 设置DEAP框架
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # 最小化两个目标
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)  # 个体的每个基因是0到1之间的浮点数
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)  # 个体长度为2
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

env = Gridworld(7, 6, [(4, 0), (4, 1), (4, 3), (4, 4), (4, 5), (4, 6)], (5, 0), (0, 3))
toolbox.register("evaluate", evaluate, env=env)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# 运行NSGA-II算法
population = toolbox.population(n=50)

# 打印初始种群
print("Initial Population:")
for ind in population:
    print(f"Individual: {ind}, Fitness: {ind.fitness.values}")

for gen in range(20):  # 迭代20代
    print(f"\nGeneration {gen + 1}:")
    
    # 评估种群适应度
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 选择下一代种群
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 应用交叉和突变操作
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:  # 交叉概率
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:  # 突变概率
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 评估新的候选解的适应度
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # 打印每一代的候选解
    print("Offspring:")
    for ind in offspring:
        print(f"Individual: {ind}, Fitness: {ind.fitness.values}")

    # 将后代替换为当前种群
    population[:] = offspring

# 获取帕累托前沿
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

# 打印最终的帕累托前沿解的权重、reward 和 effort
print("\nFinal Pareto Optimal Solutions' Reward Weights, Rewards and Efforts:")
for ind in pareto_front:
    print(f"Solution: {ind}, Fitness: {ind.fitness.values}")

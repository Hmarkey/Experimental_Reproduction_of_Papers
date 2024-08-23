import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return x**2

def f2(x):
    return (x - 2)**2

# 生成随机解
num_solutions = 100
x_values = np.random.uniform(-2, 2, num_solutions)

# 计算每个解的目标函数值
f1_values = f1(x_values)
f2_values = f2(x_values)

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # 只有在所有目标上都不优于解c的其他解，才标记为非有效解
            is_efficient[i] = not np.any(np.all(costs <= c, axis=1) & np.any(costs < c, axis=1))
    return is_efficient

# 将目标函数值组合为一个数组
costs = np.vstack((f1_values, f2_values)).T
print("costs: ", costs)
# 检测哪些点是帕累托最优的
pareto_mask = is_pareto_efficient(costs)
print("pareto_mask: ", pareto_mask)
pareto_front = costs[pareto_mask]
print("pareto_front: ", pareto_front)

# 绘制所有解的目标函数值
plt.scatter(f1_values, f2_values, label="Solutions", color="blue")

# 绘制帕累托前沿
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], label="Pareto Front", color="red")

plt.xlabel("Objective 1 (f1)")
plt.ylabel("Objective 2 (f2)")
plt.title("Pareto Front")
plt.legend()
plt.show()

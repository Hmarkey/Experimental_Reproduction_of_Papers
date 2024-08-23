# Paper

Grunitzki R, da Silva B C, Bazzan L C A. Towards designing optimal reward functions in multi-agent reinforcement learning problems[C]//2018 International Joint Conference on Neural Networks (IJCNN). IEEE, 2018: 1-8.

## Main Content

The paper introduces an Extended Optimal Reward Problem (EORP) aimed at optimizing reward functions in Multi-Agent Reinforcement Learning (MARL). The key contributions of the paper include:

1. **Automated Selection of Reward Features and Weights**: EORP automatically selects and identifies the optimal reward features and weights that constitute the reward function, rather than merely adjusting predefined weights. This addresses the issue where an inappropriate set of features selected by the designer may negatively impact the learning outcome.

2. **Broad Applicability**: Unlike existing methods, EORP is applicable to both single-agent and multi-agent reinforcement learning problems and does not rely on specific reinforcement learning models, making it widely applicable.

3. **Scalability**: EORP can scale with the number of agents learning simultaneously without significantly increasing the dimensionality of the optimization problem.

4. **Learning Efficiency Evaluation**: EORP introduces the concept of Learning Effort, a metric that evaluates the speed of learning. This allows for a trade-off between learning performance and learning speed during reward function design, leading to reward functions that not only enhance learning outcomes but also accelerate convergence.

## How This Method Helps Design Optimal Reward Functions in Multi-Agent Reinforcement Learning:
The primary advantage of EORP lies in its automated optimization of reward functions and its emphasis on learning efficiency. Specifically, it aids the design of optimal reward functions in the following ways:

1. **Automation and Optimization**:
   
   Traditional reward function design relies on manually selecting reward features and setting weights, where the designer may not foresee which features are most beneficial for learning. EORP automates the identification of reward features and optimizes the weights, reducing the burden on the designer and improving the quality of the reward function.

2. **Adaptation to Multi-Agent Environments**:
   
   In multi-agent settings, designing a common reward function is often more scalable than creating a separate reward function for each agent. EORP can design an optimal common reward function for multiple agents, simplifying system complexity and enhancing coordination and efficiency in learning.

3. **Consideration of Learning Efficiency**:

    Traditional Optimal Reward Problems (ORP) focus solely on maximizing final learning performance (Fitness), whereas EORP also considers learning efficiency, i.e., selecting a reward function that converges faster given the same learning outcome. This is particularly valuable in multi-agent environments where efficient use of time and resources is critical.

4. **Pareto Optimization**:

    EORP employs a multi-objective optimization strategy to balance learning performance and learning efficiency, finding a set of Pareto optimal solutions. These solutions achieve the best trade-off among different objectives, providing greater flexibility for the designer.

## Experimental Validation:

The paper validates the effectiveness of EORP through experiments in Gridworld and Traffic Assignment scenarios. The results show that reward functions designed by EORP outperform traditional methods in enhancing learning performance and accelerating learning speed.

## Conclusion:
EORP offers a systematic and efficient approach to designing optimal reward functions in multi-agent reinforcement learning. By automating the selection of reward features, optimizing weights, and considering learning efficiency, EORP assists designers in better guiding the learning process in complex multi-agent environments, achieving effective and coordinated learning.


## Description of Experimental Reproduction
GridWord

## Issues

However, the method need a long time to obtain an appropriate reward function because of multiple rounds of experiments in evolution algorithms, especially in complex multi agent situations.
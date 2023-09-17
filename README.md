# Flagged-Maze-Reinforcement-Learning

![image](https://github.com/proshir/Flagged-Maze-Reinforcement-Learning-Implementation/assets/19504971/528d8f88-c4e6-439c-a5af-fc24b8a03ae5)

This project is presented in Jupyter Notebook format, providing visibility into the implementation of class definitions and algorithm training. You can observe the program's execution by using the rl.play(True, True) command within the pygame environment. This command executes the program, irrespective of the epsilon capability, facilitating a more focused search.

## Model State Determination and Reduction

The number of model states is contingent upon the environment's size. We achieve state reduction by equating certain positions, streamlining the model's complexity.

## Concepts and Components

States: These correspond to the agent's positions within the environment.
Actions: Define agent movements, encompassing "up," "down," "left," and "right."
Rewards: Define the system of penalties and incentives governing agent behavior.
Goal State: Identified as "T," this marks the endpoint the agent must reach.

## Learning Rate (α) Impact

The learning rate (α) significantly influences the algorithm's performance:

It affects the speed of convergence and oscillation.
It strikes a balance between exploration and exploitation.
It plays a pivotal role in stabilization and solution accuracy.

## Discount Factor (γ) Impact

The discount factor (γ) holds a crucial role in reinforcement learning:

It delineates the importance of long-term versus short-term rewards.
It guides the pursuit of optimal policies and underscores the significance of achieving the goal.
It influences the convergence rate and temporal consistency of the learning process.

## Result
![image](https://github.com/proshir/Flagged-Maze-Reinforcement-Learning-Implementation/assets/19504971/3235df6e-fa53-4814-9b6a-ab5212f97603)
![image](https://github.com/proshir/Flagged-Maze-Reinforcement-Learning-Implementation/assets/19504971/b4b7c689-5cc4-4496-aeb8-a517e4cb37dc)

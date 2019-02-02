[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Udacity : Deep Reinforcement Learning Nanodegree

## Project 1: Navigation (Banana Bot)

### Introduction

For this project, we had to build an AI agent to conduct an episodic task by exploring the Unity Banana bot environment with the aim of reaching a high score of + 13 over 100 consecutive episodes. 

![Trained Agent][image1]

The environment consists of a state space with 37 dimensions with the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.


## Getting started

### Pre-requisites
You can clone the repository follow the below mentioned steps to get started:
1) Install all the requirements - pip install requirements.txt
2) Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
3) Run juypyter notebooks
4) Run the agent

3) And then to install python dependencies. 

    pip install -r requirements.txt

Then you should be able to run `jupyter notebook` and view `Navigation.ipynb`. 

The code for the Neural Network and Agent are in `brains.py` and `agent.py`, respectively.

## Instructions

Run each cell of `Navagation.ipynb`.
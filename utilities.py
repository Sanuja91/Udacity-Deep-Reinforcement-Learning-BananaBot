from unityagents import UnityEnvironment
import numpy as np

def initialize_env():
    env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", worker_id=1, seed=1)

    """Resetting environment"""
    # get the default brainhttp://localhost:8888/notebooks/Navigation.ipynb#8.-Running-simulation
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    return env, env_info, state, state_size, action_size, brain_name
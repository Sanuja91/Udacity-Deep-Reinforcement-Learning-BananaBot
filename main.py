import utilities, constants
from agents import Agent
from dqn import dqn


env, env_info, state, state_size, action_size, brain_name = utilities.initialize_env()

# Vanilla DQN
print("########################## VANILLA DQN ##########################")
agent = Agent(env_info=env_info, nn_type=constants.VANILLA_DQN, state_size=state_size, action_size=action_size, seed=0, load_agent = True)

vanilla_dqn_scores = dqn(agent, env, brain_name)

# Dueling Network
print("########################## DUELING DQN ##########################")
agent = Agent(env_info=env_info, nn_type=constants.DUELING_DQN, state_size=state_size, action_size=action_size, seed=0)

dueling_dqn_scores = dqn(agent, env, brain_name)

# Double DQN
print("########################## DOUBLE DQN ##########################")
agent = Agent(env_info=env_info, nn_type=constants.VANILLA_DQN, state_size=state_size, action_size=action_size, seed=0, doubleDQN = True)

double_dqn_scores = dqn(agent, env, brain_name)

# Double Dueling DQN
print("########################## DOUBLE DUELING DQN ##########################")
agent = Agent(env_info=env_info, nn_type=constants.DUELING_DQN, state_size=state_size, action_size=action_size, seed=0, doubleDQN = True)

double_dueling_dqn_scores = dqn(agent, env, brain_name)

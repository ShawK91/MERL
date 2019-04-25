from dm_control.locomotion import soccer as dm_soccer
import numpy as np

# Load the 2-vs-2 soccer environment with episodes of 10 seconds:
env = dm_soccer.load(team_size=2, time_limit=10.)

# Retrieves action_specs for all 4 players.
action_specs = env.action_spec()

# Step through the environment for one episode with random actions.
time_step = env.reset()
while not time_step.last():
  actions = []
  for action_spec in action_specs:
    action = np.random.uniform(
        action_spec.minimum, action_spec.maximum, size=action_spec.shape)
    actions.append(action)
  time_step = env.step(actions)
  print(time_step[1])

  # for i in range(len(action_specs)):
  #   print(
  #       "Player {}: reward = {}, discount = {}, observations = {}.".format(
  #           i, time_step.reward[i], time_step.discount,
  #           time_step.observation[i]))
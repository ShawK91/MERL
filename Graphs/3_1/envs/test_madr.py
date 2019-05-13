from madrl.walker.multi_walker import MultiWalkerEnv
import numpy as np

# Load the 2-vs-2 soccer environment with episodes of 10 seconds:
n_walkers = 5
reward_mech = 'local'
env = MultiWalkerEnv(n_walkers=n_walkers)
env.reset()
for i in range(10000000):
  env.render()
  a = np.array([env.agents[0].action_space.sample() for _ in range(n_walkers)])
  o, r, done, _ = env.step(a)
  print("\nStep:", i)
  # print "Obs:", o
  print("Rewards:", r)
  # print "Term:", done
  if done:
    break



# # Step through the environment for one episode with random actions.
# time_step = env.reset()
# while not time_step.last():
#   actions = []
#   for action_spec in action_specs:
#     action = np.random.uniform(
#         action_spec.minimum, action_spec.maximum, size=action_spec.shape)
#     actions.append(action)
#   time_step = env.step(actions)
#   print(time_step[1])
#
#   # for i in range(len(action_specs)):
#   #   print(
#   #       "Player {}: reward = {}, discount = {}, observations = {}.".format(
#   #           i, time_step.reward[i], time_step.discount,
#   #           time_step.observation[i]))
from cassie.cassie_env.cassieRLEnv import cassieRLEnv
import numpy as np

env = cassieRLEnv()
env.reset()
fit = 0; step = 0
for i in range(10000000):
  #env.render()
  a = np.random.normal(0, 1, 10)
  o, r, done, _ = env.step(a)
  fit += r; step+=1
  if done:
    print(fit, step)
    fit = 0; step=0
    env.reset()



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
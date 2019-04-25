from hyper.PowerPlant_env import PowerPlant
from scipy.special import expit
import numpy as np


class Fast_Simulator():  # TF Simulator individual (One complete simulator genome)
  def __init__(self):
    self.W = None

  def predict(self, input):
    # Feedforward operation
    h_1 = expit(np.dot(input, self.W[0]) + self.W[1])
    return np.dot(h_1, self.W[2]) + self.W[3]


class Parameters:
  def __init__(self):
    self.target_sensor = 1
    self.run_time = 300
    self.sensor_noise = 0.00
    self.reconf_shape = 0
    self.num_profiles = 3  # only applicable for some reconf_shapes


args = Parameters()

# Load the 2-vs-2 soccer environment with episodes of 10 seconds:
env = PowerPlant(args)
env.reset()
for i in range(10000000):
  env.render()
  a = np.zeros((2))
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
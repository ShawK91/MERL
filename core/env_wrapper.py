from core.rover_domain import Task_Rovers


class EnvironmentWrapper:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args):
		"""
		A base template for all environment wrappers.
		"""
		self.env = Task_Rovers(args)
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		return self.env.reset()


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""


		#action = self.action_low + action * (self.action_high - self.action_low)
		return self.env.step(action)

	def render(self):
		self.env.render()




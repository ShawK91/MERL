from core.rovers import rover_domain_w_setup as r


class EnvironmentWrapper:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args):
		"""
		A base template for all environment wrappers.
		"""
		self.env = r.RoverDomain()
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		self.env.reset()
		next_state = self.env.rover_observations.base
		next_state = next_state.reshape(next_state.shape[0], -1)
		return next_state


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
		#action = [ac*0 for ac in action]

		next_state, reward, done, info = self.env.step(action)
		next_state = next_state.base; reward = reward.base
		next_state = next_state.reshape(next_state.shape[0], -1)

		#print(self.env.rover_positions.base, self.env.poi_positions.base, self.env.rover_orientations.base, action)

		return next_state, reward, done, info

	def render(self):
		self.env.render()




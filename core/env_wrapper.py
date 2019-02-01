


class RoverDomainCython:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain_cython import rover_domain_w_setup as r
		self.env = r.RoverDomain()
		self.env.n_rovers = args.num_agents
		self.env.n_pois = args.num_poi
		self.env.interaction_dist = args.act_dist
		self.env.n_obs_sections = int(360/args.angle_res)
		self.env.n_req = args.coupling
		self.env.n_steps = args.ep_len
		self.env.setup_size = args.dim_x


		#Action Space
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

		#print(self.env.rover_positions.base, self.env.poi_positions.base, action, reward)
		# import numpy as np
		# if np.sum(reward) != 0:
		# 	k = 0

		#print(self.env.rover_positions.base, self.env.poi_positions.base, reward)
		##None

		return next_state, reward, done, info

	def render(self):

		# Visualize
		grid = [['-' for _ in range(self.args.dim_x)] for _ in range(self.args.dim_y)]


		# Draw in rover path
		for time_step, joint_pos in enumerate(self.env.rover_position_histories.base):
			for rover_id, rover_pos in enumerate(joint_pos):
				x = int(rover_pos[0]);
				y = int(rover_pos[1])
				# print x,y
				try: grid[x][y] = str(rover_id)
				except: None

		# Draw in food
		for poi_pos, poi_status in zip(self.env.poi_positions.base, self.env.poi_status.base):
			x = int(poi_pos[0]);
			y = int(poi_pos[1])
			marker = '#' if poi_status else '$'
			grid[x][y] = marker

		for row in grid:
			print(row)
		print()

		print('------------------------------------------------------------------------')



class RoverDomainPython:


	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args):
		"""
		A base template for all environment wrappers.
		"""
		from envs.rover_domain_python import Task_Rovers
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
		#action = [ac*0 for ac in action]
		return self.env.step(action)

	def render(self):
		self.env.render()

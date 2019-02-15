import numpy as np

class RoverDomainPython:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain.rover_domain_python import RoverDomain

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = RoverDomain(args.config)
			self.universe.append(env)

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
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return joint_obs


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

		joint_obs = []; joint_reward = []; joint_done = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:])
			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done)

		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, None



	def render(self):

		rand_univ = np.random.randint(0, len(self.universe))
		self.universe[rand_univ].render()


class RoverDomainCython:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain.rover_domain_cython import rover_domain_w_setup as r

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = r.RoverDomain()
			env.n_rovers = args.num_agents
			env.n_pois = args.num_poi
			env.interaction_dist = args.act_dist
			env.n_obs_sections = int(360/args.angle_res)
			env.n_req = args.coupling
			env.n_steps = args.ep_len
			env.setup_size = args.dim_x
			self.universe.append(env)


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
		joint_obs = []
		for env in self.universe:
			env.reset()
			next_state = env.rover_observations.base
			next_state = next_state.reshape(next_state.shape[0], -1)
			joint_obs.append(next_state)

		next_state = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

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


		joint_obs = []; joint_reward = []; joint_done = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:])
			next_state = next_state.base; reward = reward.base
			next_state = next_state.reshape(next_state.shape[0], -1)
			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done)

		joint_obs = np.stack(joint_obs, axis=1)
		joint_reward = np.stack(joint_reward, axis=1)



		#print(self.env.rover_positions.base, self.env.poi_positions.base, action, reward)
		# import numpy as np
		# if np.sum(reward) != 0:
		# 	k = 0

		#print(self.env.rover_positions.base, self.env.poi_positions.base, reward)
		##None

		return joint_obs, joint_reward, joint_done, None

	def render(self):

		# Visualize
		grid = [['-' for _ in range(self.args.dim_x)] for _ in range(self.args.dim_y)]

		rand_univ = np.random.randint(0, len(self.universe))

		# Draw in rover path
		for time_step, joint_pos in enumerate(self.universe[rand_univ].rover_position_histories.base):
			for rover_id, rover_pos in enumerate(joint_pos):
				x = int(rover_pos[0]);
				y = int(rover_pos[1])
				# print x,y
				try: grid[x][y] = str(rover_id)
				except: None

		# Draw in food
		for poi_pos, poi_status in zip(self.universe[rand_univ].poi_positions.base, self.universe[rand_univ].poi_status.base):
			x = int(poi_pos[0]);
			y = int(poi_pos[1])
			marker = '#' if poi_status else '$'
			grid[x][y] = marker

		for row in grid:
			print(row)
		print()

		print('------------------------------------------------------------------------')







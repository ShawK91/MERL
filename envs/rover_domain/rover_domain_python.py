import random, sys
from random import randint
import numpy as np
import math
import matplotlib.pyplot as plt



class RoverDomainVel:

	def __init__(self, args):

		self.args = args
		self.task_type = args.env_choice
		self.harvest_period = args.harvest_period

		#Gym compatible attributes
		self.observation_space = np.zeros((1, int(2*360 / self.args.angle_res)+1))
		self.action_space = np.zeros((1, 2))

		self.istep = 0 #Current Step counter
		self.done = False

		# Initialize POI containers tha track POI position and status
		self.poi_pos = [[None, None] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][x, y] coordinate
		self.poi_status = [self.harvest_period for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][status] --> [harvest_period --> 0 (observed)] is observed?
		#self.poi_value = [float(i+1) for i in range(self.args.num_poi)]  # FORMAT: [poi_id][value]?
		self.poi_value = [1.0 for _ in range(self.args.num_poi)]
		self.poi_visitor_list = [[] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][visitors]?

		# Initialize rover pose container
		self.rover_pos = [[0.0, 0.0, 0.0] for _ in range(self.args.num_agents)]  # FORMAT: [rover_id][x, y, orientation] coordinate with pose info
		self.rover_vel = [[0.0, 0.0] for _ in range(self.args.num_agents)]

		#Local Reward computing methods
		self.rover_closest_poi = [self.args.dim_x*2 for _ in range(self.args.num_agents)]
		self.cumulative_local = [0 for _ in range(self.args.num_agents)]


		#Rover path trace for trajectory-wide global reward computation and vizualization purposes
		self.rover_path = [[] for _ in range(self.args.num_agents)] # FORMAT: [rover_id][timestep][x, y]
		self.action_seq = [[] for _ in range(self.args.num_agents)] # FORMAT: [timestep][rover_id][action]


	def reset(self):
		self.done = False
		self.reset_poi_pos()
		self.reset_rover_pos()
		self.rover_vel = [[0.0, 0.0] for _ in range(self.args.num_agents)]
		#self.poi_value = [float(i+1) for i in range(self.args.num_poi)]
		self.poi_value = [1.0 for _ in range(self.args.num_poi)]

		self.rover_closest_poi = [self.args.dim_x*2 for _ in range(self.args.num_agents)]
		self.cumulative_local = [0 for _ in range(self.args.num_agents)]

		self.poi_status = [self.harvest_period for _ in range(self.args.num_poi)]
		self.poi_visitor_list = [[] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][visitors]?
		self.rover_path = [[] for _ in range(self.args.num_agents)]
		self.action_seq = [[] for _ in range(self.args.num_agents)]
		self.istep = 0
		return self.get_joint_state()


	def step(self, joint_action):

		#If done send back dummy trasnsition
		if self.done:
			dummy_state, dummy_reward, done, info = self.dummy_transition()
			return dummy_state, dummy_reward, done, info


		self.istep += 1
		joint_action = joint_action.clip(-1.0, 1.0)


		for rover_id in range(self.args.num_agents):

			magnitude = 0.5*(joint_action[rover_id][0]+1) # [-1,1] --> [0,1]
			self.rover_vel[rover_id][0] += magnitude

			joint_action[rover_id][1] /= 2.0 #Theta (bearing constrained to be within 90 degree turn from heading)
			self.rover_vel[rover_id][1] += joint_action[rover_id][1]

			#Constrain
			if self.rover_vel[rover_id][0] < 0: self.rover_vel[rover_id][0] = 0.0
			elif self.rover_vel[rover_id][0] > 1: self.rover_vel[rover_id][0] = 1.0

			if self.rover_vel[rover_id][1] < 0.5: self.rover_vel[rover_id][0] = 0.5
			elif self.rover_vel[rover_id][1] > 0.5: self.rover_vel[rover_id][0] = 0.5




			theta = self.rover_vel[rover_id][1] * 180 + self.rover_pos[rover_id][2]
			if theta > 360: theta -= 360
			elif theta < 0: theta += 360
			#self.rover_pos[rover_id][2] = theta

			#Update position
			x = self.rover_vel[rover_id][0] * math.cos(math.radians(theta))
			y = self.rover_vel[rover_id][0] * math.sin(math.radians(theta))
			self.rover_pos[rover_id][0] += x
			self.rover_pos[rover_id][1] += y

			#Log
			self.rover_path[rover_id].append((self.rover_pos[rover_id][0], self.rover_pos[rover_id][1], self.rover_pos[rover_id][2]))
			self.action_seq[rover_id].append([magnitude, joint_action[rover_id][1]*180])


		#Compute done
		self.done = int(self.istep >= self.args.ep_len or sum(self.poi_status) == 0)

		#info
		global_reward = None
		if self.done: global_reward = self.get_global_reward()

		return self.get_joint_state(), self.get_local_reward(), self.done, global_reward


	def reset_poi_pos(self):

		start = 0.0;
		end = self.args.dim_x - 1.0
		rad = int(self.args.dim_x/(2*(math.sqrt(10)))) +2 #         int(self.args.dim_x / math.sqrt(3) / 2.0)
		center = int((start + end) / 2.0)

		if self.args.poi_rand: #Random
			for i in range(self.args.num_poi):
				if i % 3 == 0:
					x = randint(start, center - rad - 1)
					y = randint(start, end)
				elif i % 3 == 1:
					x = randint(center + rad + 1, end)
					y = randint(start, end)
				elif i % 3 == 2:
					x = randint(center - rad, center + rad)
					if random.random()<0.5:
						y = randint(start, center - rad - 1)
					else:
						y = randint(center + rad + 1, end)

				self.poi_pos[i] = [x, y]

		else: #Not_random
			for i in range(self.args.num_poi):
				if i % 4 == 0:
					x = start + int(i/4) #randint(start, center - rad - 1)
					y = start + int(i/3)
				elif i % 4 == 1:
					x = end - int(i/4) #randint(center + rad + 1, end)
					y = start + int(i/4)#randint(start, end)
				elif i % 4 == 2:
					x = start+ int(i/4) #randint(center - rad, center + rad)
					y = end - int(i/4) #randint(start, center - rad - 1)
				else:
					x = end - int(i/4) #randint(center - rad, center + rad)
					y = end - int(i/4) #randint(center + rad + 1, end)
				self.poi_pos[i] = [x, y]


	def reset_rover_pos(self):
		start = 1.0; end = self.args.dim_x - 1.0
		rad = int(self.args.dim_x/(2*(math.sqrt(10)))) #10% area in the center for Rovers
		center = int((start + end) / 2.0)

		#Random Init
		lower = center - rad
		upper = center + rad
		for i in range(self.args.num_agents):
			x = randint(lower, upper)
			y = randint(lower, upper)
			self.rover_pos[i] = [x, y, 0.0]


	def get_joint_state(self):
		joint_state = []
		for rover_id in range(self.args.num_agents):
			self_x = self.rover_pos[rover_id][0]; self_y = self.rover_pos[rover_id][1]; self_orient = self.rover_pos[rover_id][2]

			rover_state = [0.0 for _ in range(int(360 / self.args.angle_res))]
			poi_state = [0.0 for _ in range(int(360 / self.args.angle_res))]
			temp_poi_dist_list = [[] for _ in range(int(360 / self.args.angle_res))]
			temp_rover_dist_list = [[] for _ in range(int(360 / self.args.angle_res))]

			# Log all distance into brackets for POIs
			for loc, status, value in zip(self.poi_pos, self.poi_status, self.poi_value):
				if status == 0: continue #If accessed ignore

				angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
				if dist > self.args.obs_radius: continue #Observability radius

				angle -= self_orient
				if angle < 0: angle += 360

				try: bracket = int(angle / self.args.angle_res)
				except: bracket = 0
				if bracket >= len(temp_poi_dist_list):
					print("ERROR: BRACKET EXCEED LIST", bracket, len(temp_poi_dist_list))
					bracket = len(temp_poi_dist_list)-1
				if dist == 0: dist = 0.001
				temp_poi_dist_list[bracket].append((value/(dist*dist)))

				#update closest POI for each rover info
				if dist < self.rover_closest_poi[rover_id]: self.rover_closest_poi[rover_id] = dist

			# Log all distance into brackets for other drones
			for id, loc, in enumerate(self.rover_pos):
				if id == rover_id: continue #Ignore self

				angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
				angle -= self_orient
				if angle < 0: angle += 360

				if dist > self.args.obs_radius: continue #Observability radius

				if dist == 0: dist = 0.001
				try: bracket = int(angle / self.args.angle_res)
				except: bracket = 0
				if bracket >= len(temp_rover_dist_list):
					print("ERROR: BRACKET EXCEED LIST", bracket, len(temp_rover_dist_list), angle)
					bracket = len(temp_rover_dist_list)-1
				temp_rover_dist_list[bracket].append((1/(dist*dist)))


			####Encode the information onto the state
			for bracket in range(int(360 / self.args.angle_res)):
				# POIs
				num_poi = len(temp_poi_dist_list[bracket])
				if num_poi > 0:
					if self.args.sensor_model == 'density': poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi #Density Sensor
					elif self.args.sensor_model == 'closest': poi_state[bracket] = max(temp_poi_dist_list[bracket])  #Closest Sensor
					else: sys.exit('Incorrect sensor model')
				else: poi_state[bracket] = -1.0

				#Rovers
				num_agents = len(temp_rover_dist_list[bracket])
				if num_agents > 0:
					if self.args.sensor_model == 'density': rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents #Density Sensor
					elif self.args.sensor_model == 'closest': rover_state[bracket] = max(temp_rover_dist_list[bracket]) #Closest Sensor
					else: sys.exit('Incorrect sensor model')
				else: rover_state[bracket] = -1.0

			state = [rover_id] + rover_state + poi_state + self.rover_vel[rover_id] #Append rover_id, rover LIDAR and poi LIDAR to form the full state

			# #Append wall info
			# state = state + [-1.0, -1.0, -1.0, -1.0]
			# if self_x <= self.args.obs_radius: state[-4] = self_x
			# if self.args.dim_x - self_x <= self.args.obs_radius: state[-3] = self.args.dim_x - self_x
			# if self_y <= self.args.obs_radius :state[-2] = self_y
			# if self.args.dim_y - self_y <= self.args.obs_radius: state[-1] = self.args.dim_y - self_y

			#state = np.array(state)
			joint_state.append(state)

		return joint_state


	def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two predators relative to (1,0) vector (x-axis)
		v1 = x2 - x1;
		v2 = y2 - y1
		angle = np.rad2deg(np.arctan2(v1, v2))
		if angle < 0: angle += 360
		if math.isnan(angle): angle = 0.0

		dist = v1 * v1 + v2 * v2
		dist = math.sqrt(dist)

		if math.isnan(angle) or math.isinf(angle): angle = 0.0

		return angle, dist


	def get_local_reward(self):
		#Update POI's visibility
		poi_visitors = [[] for _ in range(self.args.num_poi)]
		poi_visitor_dist = [[] for _ in range(self.args.num_poi)]
		for i, loc in enumerate(self.poi_pos): #For all POIs
			if self.poi_status[i]== 0:
				continue #Ignore POIs that have been harvested already

			for rover_id in range(self.args.num_agents): #For each rover
				x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
				dist = math.sqrt(x1 * x1 + y1 * y1)
				if dist <= self.args.act_dist:
					poi_visitors[i].append(rover_id) #Add rover to POI's visitor list
					poi_visitor_dist[i].append(dist)

		#Compute reward
		rewards = [0.0 for _ in range(self.args.num_agents)]
		for poi_id, rovers in enumerate(poi_visitors):
				#if self.task_type == 'rover_tight' and len(rovers) >= self.args.coupling or self.task_type == 'rover_loose' and len(rovers) >= 1:
				#Update POI status
				if self.task_type == 'rover_tight' and len(rovers) >= self.args.coupling or self.task_type == 'rover_loose' and len(rovers) >= 1 or self.task_type == 'rover_trap' and len(rovers) >= 1:
					self.poi_status[poi_id] -= 1
					self.poi_visitor_list[poi_id] = list(set(self.poi_visitor_list[poi_id]+rovers[:]))

				if self.args.is_lsg: #Local subsume Global?
					for rover_id, dist in zip(rovers, poi_visitor_dist[poi_id]):
						rewards[rover_id] += self.poi_value[poi_id]*10

		#Proximity Rewards
		if self.args.is_proxim_rew:
			for i in range(self.args.num_agents):
				proxim_rew = self.args.act_dist/self.rover_closest_poi[i]
				if proxim_rew > 1.0: proxim_rew = 1.0
				rewards[i] += proxim_rew
				#print(self.rover_closest_poi[i], proxim_rew)
				self.cumulative_local[i] += proxim_rew
		self.rover_closest_poi = [self.args.dim_x * 2 for _ in range(self.args.num_agents)] #Reset closest POI


		return rewards


	def dummy_transition(self):
		joint_state = [[0.0 for _ in range(int(720 / self.args.angle_res)+3)] for _ in range(self.args.num_agents)]
		rewards = [0.0 for _ in range(self.args.num_agents)]

		return joint_state, rewards, True, None


	def get_global_reward(self):
		global_rew = 0.0; max_reward = 0.0

		if self.task_type == 'rover_tight' or self.task_type == 'rover_loose':
			for value, status in zip(self.poi_value, self.poi_status):
				global_rew += (status == 0) * value
				max_reward += value

		elif self.task_type == 'rover_trap':  # Rover_Trap domain
			for value, visitors in zip(self.poi_value, self.poi_visitor_list):
				multiplier = len(visitors) if len(visitors) < self.args.coupling else self.args.coupling
				global_rew += value * multiplier
				max_reward += self.args.coupling * value

		else:
			sys.exit('Incorrect task type')


		global_rew = global_rew/max_reward

		return global_rew


	def render(self):
		# Visualize
		grid = [['-' for _ in range(self.args.dim_x)] for _ in range(self.args.dim_y)]


		# Draw in rover path
		for rover_id, path in enumerate(self.rover_path):
			for loc in path:
				x = int(loc[0]); y = int(loc[1])
				if x < self.args.dim_x and y < self.args.dim_y and x >=0 and y >=0:
					grid[x][y] = str(rover_id)

		# Draw in food
		for poi_pos, poi_status in zip(self.poi_pos, self.poi_status):
			x = int(poi_pos[0]);
			y = int(poi_pos[1])
			marker = '$' if poi_status == 0 else '#'
			grid[x][y] = marker

		for row in grid:
			print(row)

		for agent_id, temp in enumerate(self.action_seq):
			print()
			print('Action Sequence Rover ', str(agent_id),)
			for entry in temp:
				print(['{0: 1.1f}'.format(x) for x in entry], end =" " )
		print()


		print('------------------------------------------------------------------------')


	def viz(self, save=False, fname=''):

		padding = 70

		observed = 3 + self.args.num_agents*2
		unobserved = observed + 3

		# Empty Canvas
		matrix = np.zeros((padding*2+self.args.dim_x*10, padding*2+self.args.dim_y*10))


		#Draw in rover
		color = 3.0; rover_width = 1; rover_start_width = 4
		# Draw in rover path
		for rover_id, path in enumerate(self.rover_path):
			start_x, start_y = int(path[0][0]*10)+padding, int(path[0][1]*10)+padding

			matrix[start_x-rover_start_width:start_x+rover_start_width, start_y-rover_start_width:start_y+rover_start_width] = color
			#continue
			for loc in path[1:]:
				x = int(loc[0]*10)+padding; y = int(loc[1]*10)+padding
				if x > len(matrix) or y > len(matrix) or x < 0 or y < 0 : continue

				#Interpolate and Draw
				for i in range(int(abs(start_x-x))):
					if start_x > x:
						matrix[x+i - rover_width:x+i + rover_width, start_y - rover_width:start_y + rover_width] = color
					else:
						matrix[x - i - rover_width:x - i + rover_width, start_y - rover_width:start_y + rover_width] = color

				for i in range(int(abs(start_y-y))):
					if start_y > y:
						matrix[x - rover_width:x + rover_width, y + i- rover_width:y +i+ rover_width] = color
					else:
						matrix[x  - rover_width:x + rover_width, y-i- rover_width:y-i + rover_width] = color
				start_x, start_y = x, y

			color += 2

		#Draw in POIs
		poi_width = 8
		for poi_pos, poi_status in zip(self.poi_pos, self.poi_status):
			x = padding+int(poi_pos[0])*10; y = padding+int(poi_pos[1])*10
			if poi_status: color = unobserved #Not observed
			else: color = observed #Observed

			matrix[x-poi_width:x+poi_width, y-poi_width:y+poi_width] = color

		fig, ax = plt.subplots()
		im = ax.imshow(matrix, cmap='Accent', origin='upper')
		if save:
			plt.savefig(fname=fname, dpi=300, quality=90, format='png')
		else:
			plt.show()

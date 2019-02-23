import random, sys
from random import randint
import numpy as np
import math

class RoverDomain:

	def __init__(self, args):

		self.args = args
		self.task_type = args.env_choice

		#Gym compatible attributes
		self.observation_space = np.zeros((1, int(2*360 / self.args.angle_res)))
		self.action_space = np.zeros((1, 2))

		self.istep = 0 #Current Step counter

		# Initialize POI containers tha track POI position and status
		self.poi_pos = [[None, None] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][x, y] coordinate
		self.poi_status = [False for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][status] --> [T/F] is observed?
		self.poi_value = [float(i+1) for i in range(self.args.num_poi)]  # FORMAT: [poi_id][value]?
		self.poi_visitor_list = [[] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][visitors]?

		# Initialize rover pose container
		self.rover_pos = [[0.0, 0.0, 0.0] for _ in range(self.args.num_agents)]  # FORMAT: [rover_id][x, y, orientation] coordinate with pose info


		#Rover path trace for trajectory-wide global reward computation and vizualization purposes
		self.rover_path = [[] for _ in range(self.args.num_agents)] # FORMAT: [rover_id][timestep][x, y]
		self.action_seq = [[0.0 for _ in range(2)] for _ in range(self.args.num_agents)] # FORMAT: [timestep][rover_id][action]


	def reset(self):
		self.reset_poi_pos()
		self.reset_rover_pos()
		self.poi_value = [float(i+1) for i in range(self.args.num_poi)]
		self.poi_status = [False for _ in range(self.args.num_poi)]
		self.poi_visitor_list = [[] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][visitors]?
		self.rover_path = [[] for _ in range(self.args.num_agents)]
		self.action_seq = [[0.0 for _ in range(2)] for _ in range(self.args.num_agents)]
		self.istep = 0
		return self.get_joint_state()


	def step(self, joint_action):
		self.istep += 1

		joint_action = joint_action.clip(-1.0, 1.0)


		for rover_id in range(self.args.num_agents):

			magnitude = 0.5*(joint_action[rover_id][0]+1) # [-1,1] --> [0,1]
			theta = joint_action[rover_id][1] * 180 + self.rover_pos[rover_id][2]
			if theta > 360: theta -= 360
			if theta < 0: theta += 360
			self.rover_pos[rover_id][2] = theta

			#Update position
			x = magnitude * math.cos(math.radians(theta))
			y = magnitude * math.sin(math.radians(theta))
			self.rover_pos[rover_id][0] += x
			self.rover_pos[rover_id][1] += y


			#Log
			self.rover_path[rover_id].append((self.rover_pos[rover_id][0], self.rover_pos[rover_id][1], self.rover_pos[rover_id][2]))
			self.action_seq[rover_id].append((magnitude, joint_action[rover_id][1]*180))





		#Compute done
		done = int(self.istep >= self.args.ep_len or sum(self.poi_status) == len(self.poi_status))

		#info
		global_reward = None
		if done: global_reward = self.get_global_reward()

		return self.get_joint_state(), self.get_local_reward(), done, global_reward


	def reset_poi_pos(self):

		start = 0.0;
		end = self.args.dim_x - 1.0
		rad = int(self.args.dim_x / math.sqrt(3) / 2.0)
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
		rad = int(self.args.dim_x / math.sqrt(3) / 2.0)
		center = int((start + end) / 2.0)


		for rover_id in range(self.args.num_agents):
				quadrant = rover_id % 4
				if quadrant == 0:
					x = center - 1 - (rover_id / 4) % (center - rad)
					y = center - (rover_id / (4 * center - rad)) % (center - rad)
				if quadrant == 1:
					x = center + (rover_id / (4 * center - rad)) % (center - rad)-1
					y = center - 1 + (rover_id / 4) % (center - rad)
				if quadrant == 2:
					x = center + 1 + (rover_id / 4) % (center - rad)
					y = center + (rover_id / (4 * center - rad)) % (center - rad)
				if quadrant == 3:
					x = center - (rover_id / (4 * center - rad)) % (center - rad)
					y = center + 1 - (rover_id / 4) % (center - rad)
				self.rover_pos[rover_id] = [x, y, 0.0]


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
				if status == True: continue #If accessed ignore

				angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
				if dist > self.args.obs_radius: continue #Observability radius

				angle -= self_orient
				if angle < 0: angle += 360

				bracket = int(angle / self.args.angle_res)
				if dist == 0: dist = 0.001
				temp_poi_dist_list[bracket].append((value/(dist*dist)))

			# Log all distance into brackets for other drones
			for id, loc, in enumerate(self.rover_pos):
				if id == rover_id: continue #Ignore self

				angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
				angle -= self_orient
				if angle < 0: angle += 360

				if dist > self.args.obs_radius: continue #Observability radius

				if dist == 0: dist = 0.001
				bracket = int(angle / self.args.angle_res)
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

			state = rover_state + poi_state #Append rover and poi to form the full state

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

		dist = v1 * v1 + v2 * v2
		dist = math.sqrt(dist)

		return angle, dist



	def get_local_reward(self):
		#Update POI's visibility
		poi_visitors = [[] for _ in range(self.args.num_poi)]
		for i, loc in enumerate(self.poi_pos): #For all POIs
			if self.poi_status[i]== True: continue #Ignore POIs that have been harvested already

			for rover_id in range(self.args.num_agents): #For each rover
				x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
				dist = math.sqrt(x1 * x1 + y1 * y1)
				if dist <= self.args.act_dist: poi_visitors[i].append(rover_id) #Add rover to POI's visitor list

		#Compute reward
		rewards = [0.0 for _ in range(self.args.num_agents)]
		for poi_id, rovers in enumerate(poi_visitors):
			if len(rovers) >= self.args.coupling:
				self.poi_status[poi_id] = True
				self.poi_visitor_list[poi_id] = rovers[:]
				for rover_id in rovers:
					rewards[rover_id] += self.poi_value[poi_id]


		return rewards


	#TODO
	def get_global_reward(self):
		#use
		#self.rover_path and self.poi_pos and self.poi_value to compute TRAJECTORY_WIDE REWARD
		if self.task_type == 'rover_loose':
			global_rew = 0.0; max_reward = 0.0
			for value, visitors in zip(self.poi_value, self.poi_visitor_list):
				global_rew += value * len(visitors)
				max_reward += self.args.num_agents * value


		if self.task_type == 'rover_tight':
			global_rew = 0.0; max_reward = 0.0
			for value, status in zip(self.poi_value, self.poi_status):
				global_rew += status * value
				max_reward += value


		return global_rew/max_reward





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
			marker = '$' if poi_status else '#'
			grid[x][y] = marker

		for row in grid:
			print(row)
		#print(self.rover_path)
		for i, temp in enumerate(self.action_seq):
			print('Action Sequence Rover ', str(i), temp)
		print()


		print('------------------------------------------------------------------------')




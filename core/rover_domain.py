import random, sys
from random import randint
import numpy as np
import math

class Task_Rovers:

    def __init__(self, parameters):


        self.params = parameters; self.dim_x = parameters.dim_x; self.dim_y = parameters.dim_y
        self.observation_space = np.zeros((int(2*360 / self.params.angle_res + 4), 1))
        self.action_space = np.zeros((self.params.action_dim,1))
        self.ep_len = parameters.ep_len; self.istep = 0

        # Initialize food position container
        self.poi_pos = [[None, None] for _ in range(self.params.num_poi)]  # FORMAT: [item] = [x, y] coordinate
        self.poi_status = [False for _ in range(self.params.num_poi)]  # FORMAT: [item] = [T/F] is observed?

        # Initialize rover position container
        self.rover_pos = [[0.0, 0.0] for _ in range(self.params.num_agents)]  # Track each rover's position
        self.ledger_closest = [[0.0, 0.0] for _ in range(self.params.num_agents)]  # Track each rover's ledger call


        #Rover path trace (viz)
        self.rover_path = [[(loc[0], loc[1])] for loc in self.rover_pos]
        self.action_seq = [[0.0 for _ in range(self.params.action_dim)] for _ in range(self.params.num_agents)]

    def reset_poi_pos(self):

        if self.params.unit_test == 1: #Unit_test
            self.poi_pos[0] = [0,1]
            return

        if self.params.unit_test == 2: #Unit_test
            if random.random()<0.5: self.poi_pos[0] = [4,0]
            else: self.poi_pos[0] = [4,9]
            return

        start = 1.0;
        end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        if self.params.poi_rand: #Random
            for i in range(self.params.num_poi):
                if i % 3 == 0:
                    x = randint(start, center - rad - 1)
                    y = randint(start, end)
                elif i % 3 == 1:
                    x = randint(center + rad + 1, end)
                    y = randint(start, end)
                elif i % 3 == 2:
                    x = randint(center - rad, center + rad)
                    y = randint(start, center - rad - 1)
                else:
                    x = randint(center - rad, center + rad)
                    y = randint(center + rad + 1, end)
                self.poi_pos[i] = [x, y]

        else: #Not_random
            for i in range(self.params.num_poi):
                if i % 3 == 0:
                    x = start + i/4 #randint(start, center - rad - 1)
                    y = start + i/3
                elif i % 3 == 1:
                    x = center + i/4 #randint(center + rad + 1, end)
                    y = start + i/4#randint(start, end)
                elif i % 3 == 2:
                    x = start+i/4#randint(center - rad, center + rad)
                    y = center + i/4#randint(start, center - rad - 1)
                else:
                    x = center+i/4#randint(center - rad, center + rad)
                    y = center+i/4#randint(center + rad + 1, end)
                self.poi_pos[i] = [x, y]

    def reset_rover_pos(self):
        start = 1.0; end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        if self.params.unit_test == 1: #Unit test
            self.rover_pos[0] = [end,0];
            return

        for rover_id in range(self.params.num_agents):
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
                self.rover_pos[rover_id] = [x, y]

    def reset(self):
        self.reset_poi_pos()
        self.reset_rover_pos()
        self.poi_status = self.poi_status = [False for _ in range(self.params.num_poi)]
        self.rover_path = [[(loc[0], loc[1])] for loc in self.rover_pos]
        self.action_seq = [[0.0 for _ in range(self.params.action_dim)] for _ in range(self.params.num_agents)]
        self.istep = 0
        return self.get_joint_state()

    def get_joint_state(self):
        joint_state = []
        for rover_id in range(self.params.num_agents):
            self_x = self.rover_pos[rover_id][0]; self_y = self.rover_pos[rover_id][1]

            rover_state = [0.0 for _ in range(int(360 / self.params.angle_res))]
            poi_state = [0.0 for _ in range(int(360 / self.params.angle_res))]
            temp_poi_dist_list = [[] for _ in range(int(360 / self.params.angle_res))]
            temp_rover_dist_list = [[] for _ in range(int(360 / self.params.angle_res))]

            # Log all distance into brackets for POIs
            x2 = -1.0; y2 = 0.0
            for loc, status in zip(self.poi_pos, self.poi_status):
                if status == True: continue #If accessed ignore

                x1 = loc[0] - self_x; y1 = loc[1] - self_y
                angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                if dist > self.params.obs_radius: continue #Observability radius

                bracket = int(angle / self.params.angle_res)
                temp_poi_dist_list[bracket].append(dist)

            # Log all distance into brackets for other drones
            for id, loc, in enumerate(self.rover_pos):
                if id == rover_id: continue #Ignore self

                x1 = loc[0] - self_x; y1 = loc[1] - self_y
                angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                if dist > self.params.obs_radius: continue #Observability radius

                bracket = int(angle / self.params.angle_res)
                temp_rover_dist_list[bracket].append(dist)


            ####Encode the information onto the state
            for bracket in range(int(360 / self.params.angle_res)):
                # POIs
                num_poi = len(temp_poi_dist_list[bracket])
                if num_poi > 0:
                    if self.params.sensor_model == 'density': poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi #Density Sensor
                    elif self.params.sensor_model == 'closest': poi_state[bracket] = min(temp_poi_dist_list[bracket])  #Closest Sensor
                    else: sys.exit('Incorrect sensor model')
                else: poi_state[bracket] = -1.0

                #Rovers
                num_agents = len(temp_rover_dist_list[bracket])
                if num_agents > 0:
                    if self.params.sensor_model == 'density': rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents #Density Sensor
                    elif self.params.sensor_model == 'closest': rover_state[bracket] = min(temp_rover_dist_list[bracket]) #Closest Sensor
                    else: sys.exit('Incorrect sensor model')
                else: rover_state[bracket] = -1.0

            state = rover_state + poi_state #Append rover and poi to form the full state

            #Append wall info
            state = state + [-1.0, -1.0, -1.0, -1.0]
            if self_x <= self.params.obs_radius: state[-4] = self_x
            if self.params.dim_x - self_x <= self.params.obs_radius: state[-3] = self.params.dim_x - self_x
            if self_y <= self.params.obs_radius :state[-2] = self_y
            if self.params.dim_y - self_y <= self.params.obs_radius: state[-1] = self.params.dim_y - self_y

            #state = np.array(state)
            joint_state.append(state)

        return joint_state

    def get_angle_dist(self, x1, y1, x2,y2):  # Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        dot = x2 * x1 + y2 * y1  # dot product
        det = x2 * y1 - y2 * x1  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle) + 180.0 + 270.0
        angle = angle % 360
        dist = x1 * x1 + y1 * y1
        dist = math.sqrt(dist)
        if np.isnan(angle): angle = 0.0
        return angle, dist

    def step(self, joint_action):
        self.istep += 1

        for rover_id in range(self.params.num_agents):
            action = joint_action[rover_id]
            new_pos = [self.rover_pos[rover_id][0]+action[0], self.rover_pos[rover_id][1]+action[1]]

            #Check if action is legal
            if not(new_pos[0] >= self.dim_x or new_pos[0] < 0 or new_pos[1] >= self.dim_y or new_pos[1] < 0):  #If legal
                self.rover_pos[rover_id] = [new_pos[0], new_pos[1]] #Execute action

        #Append rover path
        for rover_id in range(self.params.num_agents):
            self.rover_path[rover_id].append((self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]))

        #Compute done
        done = int(self.istep >= self.ep_len)

        return self.get_joint_state(), self.get_reward(), done, None

    def get_reward(self):
        #Update POI's visibility
        poi_visitors = [[] for _ in range(self.params.num_poi)]
        for i, loc in enumerate(self.poi_pos): #For all POIs
            if self.poi_status[i]== True: continue #Ignore POIs that have been harvested already

            for rover_id in range(self.params.num_agents): #For each rover
                x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
                dist = math.sqrt(x1 * x1 + y1 * y1)
                if dist <= self.params.act_dist: poi_visitors[i].append(rover_id) #Add rover to POI's visitor list

        #Compute reward
        rewards = [0.0 for _ in range(self.params.num_agents)]
        for poi_id, rovers in enumerate(poi_visitors):
            if len(rovers) >= self.params.coupling:
                self.poi_status[poi_id] = True
                lucky_rovers = random.sample(rovers, self.params.coupling)
                for rover_id in lucky_rovers: rewards[rover_id] += 1.0/self.params.num_poi


        return rewards

    def visualize(self):

        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        # Draw in hive
        drone_symbol_bank = ["0", "1", '2', '3', '4', '5']
        for rover_pos, symbol in zip(self.rover_pos, drone_symbol_bank):
            x = int(rover_pos[0]); y = int(rover_pos[1])
            #print x,y
            grid[x][y] = symbol


        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0]); y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

        for row in grid:
            print(row)
        print

    def render(self):
        # Visualize
        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        drone_symbol_bank = ["0", "1", '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        # Draw in rover path
        for rover_id in range(self.params.num_agents):
            for time in range(self.ep_len):
                x = int(self.rover_path[rover_id][time][0]);
                y = int(self.rover_path[rover_id][time][1])
                # print x,y
                grid[x][y] = drone_symbol_bank[rover_id]

        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0]);
            y = int(loc[1])
            marker = '$' if status else '#'
            grid[x][y] = marker

        for row in grid:
            print(row)
        print()

        print('------------------------------------------------------------------------')




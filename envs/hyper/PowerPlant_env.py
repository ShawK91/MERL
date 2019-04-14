import numpy as np
import pickle as cPickle
from copy import deepcopy as dcopy
import math, sys
from scipy.special import expit
from random import randint
import random


class Fast_Simulator(): #TF Simulator individual (One complete simulator genome)
    def __init__(self):
        self.W = None

    def predict(self, input):
        # Feedforward operation
        h_1 = expit(np.dot(input, self.W[0]) + self.W[1])
        return np.dot(h_1, self.W[2]) + self.W[3]


class PowerPlant:
    def __init__(self, target_sensor, run_time, sensor_noise, reconf_shape, num_profiles):
        """
        :param args:
        reconf_shape: #0 flat line tracking, #1 Periodic shape, #2 Mimicking real shape
        """
        self.target_sensor = target_sensor
        self.run_time = run_time
        self.sensor_noise = sensor_noise
        self.reconf_shape=reconf_shape
        self.num_profiles = num_profiles
        self.observation_space = np.zeros(20)
        self.action_space = np.zeros(2)

        self.istep = 0
        self.setpoints = get_setpoints(reconf_shape, run_time, num_profiles)

        self.train_data = data_preprocess()  # Get simulator data
        self.sim_input = unsqueeze(np.copy(self.train_data[0][0:]), axis=0)         # Input to the simulator
        self.controller_input = unsqueeze(np.delete(np.copy(self.sim_input), -1))    # Input to the controller
        self.controller_input[-1] = self.setpoints[self.istep]

        self.simulator = unpickle('envs/hyper/Champion_Simulator')

        self.loss = None
        self.done = False

    def reset(self):
        """
        :return updated state, reward, done, info
        """
        self.istep = 0
        self.setpoints = get_setpoints(self.reconf_shape, self.run_time, self.num_profiles)
        self.controller_input = unsqueeze(np.delete(np.copy(self.sim_input), -1))    # Input to the controller
        self.controller_input[-1] = self.setpoints[self.istep]
        self.loss = None
        self.done = False
        return self.controller_input

    def step(self, action):
        """
        Takes the controller's action, concatenates into the current state of the controller,
        predicts (using the pre-trained simulator) the next state based on current state and
        the controller action input, then returns the next state
        """
        sim_input = dcopy(self.sim_input)  # current state is updated after simulator pass
        sim_input[0][19] = action[0]   # Assuming there is no batch
        sim_input[0][20] = action[1]
        sim_out = self.simulator.predict(sim_input)

        # update the current state
        self.sim_input[0][0:19] = sim_out

        # calculate fitness/loss
        loss = calculate_loss(sim_out[0][self.target_sensor], (self.setpoints[self.istep]))  # this needs the sim_out and the setpoint(target_state)

        self.istep += 1

        # populate states for the controller. Controller_input[20] is the target
        self.controller_input[0:-1, 0] = dcopy(sim_out[0])        # 0 to 19th. TODO: Fix this hacky indexing 1sN4All; the unnecessary array
        self.controller_input[-1, 0] = self.setpoints[self.istep]  # 20th

        done = (self.istep >= self.run_time-1)

        # Noise. If applicable, add noise to the state input to the controller
        if self.sensor_noise != 0:  # Add sensor noise
            noise_mul = np.random.normal(0, self.sensor_noise, (self.controller_input.shape[0]))
            self.controller_input = np.multiply(self.controller_input, noise_mul)

        return self.controller_input, loss, done, None

    def dummy_state(self):
        return np.zeros((1,20))

    def dummy_reward(self):
        return [0]



def calculate_loss(current_state, target):
    # Calculate error (weakness)
    loss = -1*(np.fabs(current_state - target))  # just the absolute error
    return loss


def get_setpoints(reconf_shape, run_time, num_profiles):
    if reconf_shape == 1:
        desired_setpoints = np.reshape(np.zeros(run_time), (run_time))#, 1))
        for profile in range(num_profiles):
            multiplier = randint(1, 5)
            #print profile, multiplier
            for i in range(int(run_time/num_profiles)):
                turbine_speed = math.sin(i * 0.2 * multiplier)
                turbine_speed *= 0.3 #Between -0.3 and 0.3
                turbine_speed += 0.5  #Between 0.2 and 0.8 centered on 0.5
                desired_setpoints[profile * int(run_time/num_profiles) + i] = turbine_speed

    elif reconf_shape == 2:
        desired_setpoints = np.zeros(run_time) + random.uniform(0.4, 0.6)
        noise = np.random.uniform(-0.01, 0.01, (run_time))
        desired_setpoints += noise

        for profile_id in range(num_profiles):
            phase_len = run_time/num_profiles
            phase_start = profile_id * phase_len; phase_end = phase_start + (phase_len)

            start = random.randint(phase_start, phase_end-35)
            end = random.randint(start+10, start + 35)
            magnitude = random.uniform(-0.25, 0.25)

            for i in range(start, end):
                desired_setpoints[i] += magnitude

    elif reconf_shape == 0:  # No reconfigurability, target is 0.5
        desired_setpoints = np.zeros(run_time) + 0.5

    return desired_setpoints


def data_preprocess(filename='envs/hyper/ColdAir.csv', downsample_rate=25, split = 1000):
    # Import training data and clear away the two top lines
    data = np.loadtxt(filename, delimiter=',', skiprows=2)

    # Splice data (downsample)
    ignore = np.copy(data)
    data = data[0::downsample_rate]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (i != data.shape[0] - 1):
                data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,
                                   j].sum() / downsample_rate
            else:
                residue = ignore.shape[0] - i * downsample_rate
                data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue, j].sum() / residue

    # Normalize between 0-0.99
    normalizer = np.zeros(data.shape[1])
    min = np.zeros(len(data[0]))
    max = np.zeros(len(data[0]))
    for i in range(len(data[0])):
        min[i] = np.amin(data[:, i])
        max[i] = np.amax(data[:, i])
        normalizer[i] = max[i] - min[i] + 0.00001
        data[:, i] = (data[:, i] - min[i]) / normalizer[i]

    return data


def unpickle(filename):
    sys.modules[__name__] = Fast_Simulator()
    with open(filename, 'rb') as f:
        u = cPickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p


def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))




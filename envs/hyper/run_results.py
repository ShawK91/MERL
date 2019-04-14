import numpy as np, os, random
import mod_hyper as mod, sys, math
from random import randint
from scipy.special import expit
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

#Changeable Macros
controller_id = 3 #1 TF FF
            #2 PT FF
            #3 PT GRU-MB

actuator_fname = 'actuator_noise'
sensor_fname = 'sensor_noise'
sensor_fail_fname = 'sensor_fail'
perfect_fname = 'perfect'

test = 'champ_controller'

#Noise
sensor_noise = 0.0
sensor_failure = 0.0
actuator_noise = 0.0

#FIXED
num_input = 20
is_random_initial_state = False
num_profiles = 3
run_time = 300


def plot_controller(individual, setpoints, sim_input, control_input, simulator):  # Controller fitness
    track_setpoint = []; track_output = []
    individual.reset(batch_size=1)
    weakness = 0.0

    # Convert numpy arrays to Tensors
    control_input = torch.Tensor(control_input)
    sim_input = torch.Tensor(sim_input)

    for example in range(run_time):  # For duration of run

        # Add noise to the state input to the controller
        if sensor_noise != 0:  # Add sensor noise
            noise_mul = np.random.normal(0, sensor_noise,
                                         (control_input.shape[0], control_input.shape[1]))
            control_input = np.multiply(control_input, noise_mul)

        # if random.random() < self.parameters.sensor_failure:
        #     control_input[0][11] = 0.0

        # Fill in the setpoint to control input
        if not (torch.is_tensor(setpoints)): setpoints = torch.Tensor(setpoints)
        control_input[-1, :] = setpoints[:, example]

        # RUN THE CONTROLLER TO GET CONTROL OUTPUT
        control_out = individual.predict(control_input)

        # Add actuator noise (controls)
        if actuator_noise != 0:  # Add actuator noise
            noise_mul = np.random.normal(0, actuator_noise,
                                         (control_out.shape[0], control_out.shape[1]))
            control_out = np.multiply(control_out, noise_mul)

        # Fill in the controls (OLD)
        # sim_input[:, 19] = control_out[0][:]
        # sim_input[:, 20] = control_out[1][:]

        # Fill in the controls (NEW)
        sim_input[:, 19] = control_out[:, 0]
        sim_input[:, 20] = control_out[:, 1]

        # Use the simulator to get the next state
        simulator_out = simulator.predict(sim_input)

        # Calculate error (weakness)
        track_output.append(simulator_out[:, 11])
        track_setpoint.append(setpoints[:, example])
        weakness += np.mean(np.fabs(simulator_out[:, 11] - np.array(setpoints[:, example])))  # Time variant simulation

        # Fill in the simulator inputs and control inputs
        sim_input[:, 0:19] = torch.Tensor(simulator_out[:, 0:19])
        control_input[0:-1, :] = torch.Tensor(np.transpose(simulator_out[:, 0:19]))

    return -weakness, track_setpoint, track_output


class Fast_Simulator(): #TF Simulator individual (One complete simulator genome)
    def __init__(self):
        self.W = None

    def predict(self, input):
        # Feedforward operation
        h_1 = expit(np.dot(input.detach().numpy(), self.W[0]) + self.W[1])
        return np.dot(h_1, self.W[2]) + self.W[3]

    def from_tf(self, tf_sess):
        self.W = tf_sess.run(tf.trainable_variables())


def get_setpoints(shape=2):
    if shape == 1:
        desired_setpoints = np.reshape(np.zeros(run_time), (run_time, 1))
        for profile in range(num_profiles):
            multiplier = randint(1, 5)
            # print profile, multiplier
            for i in range(int(run_time / num_profiles)):
                turbine_speed = math.sin(i * 0.2 * multiplier)
                turbine_speed *= 0.3  # Between -0.3 and 0.3
                turbine_speed += 0.5  # Between 0.2 and 0.8 centered on 0.5
                desired_setpoints[profile * int(run_time / num_profiles) + i][
                    0] = turbine_speed

    elif shape == 2:
        desired_setpoints = np.zeros(run_time) + random.uniform(0.4, 0.6)
        noise = np.random.uniform(-0.01, 0.01, (run_time))
        desired_setpoints += noise

        for profile_id in range(num_profiles):
            phase_len = run_time / num_profiles
            phase_start = profile_id * phase_len;
            phase_end = phase_start + (phase_len)

            start = random.randint(phase_start, phase_end - 35)
            end = random.randint(start + 10, start + 35)
            magnitude = random.uniform(-0.25, 0.25)

            for i in range(start, end):
                desired_setpoints[i] += magnitude

    elif shape == 0:  # No reconfigurability, target is 0.5
        desired_setpoints = np.zeros(run_time) + 0.5

    return desired_setpoints


def data_preprocess(filename='ColdAir.csv', downsample_rate=25, split=1000):
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

        # Train/Valid split
        train_data = data[0:split]
        valid_data = data[split:len(data)]

        return train_data, valid_data


if __name__ == "__main__":
    shape = 1
    train_data, _ = data_preprocess()
    simulator = mod.unpickle('Champion_Simulator')
    sim_input = mod.unsqueeze(np.copy(train_data[0][0:]), axis=0)
    control_input = mod.unsqueeze(np.delete(np.copy(sim_input), 1))  # Input to the controller
    setpoints = mod.unsqueeze(np.array(get_setpoints(shape)), axis=0)


    controller_list = []

    individual = mod.unpickle('R_Reconfigurable_Controller/champ_controller_shape1')
    # Reset to set batch size to 1
    individual.reset(1)
    weakness, setpoints, outputs = plot_controller(individual, setpoints, sim_input, control_input, simulator)


    plt.figure(figsize=(16, 9))
    plt.plot(setpoints, 'r', label='Desired Turbine Speed')
    plt.plot(outputs, 'b--', label='Controller trained with no noise')
    #plt.plot(track_output[1], 'g-', label='Controller trained with actuator noise')
    #plt.plot(track_output[2], 'g-', label='Controller trained with sensor noise')
    #plt.plot(track_output[3], 'g-', label='Controller trained for sensor failure')
    # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
    # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
    plt.legend(loc='upper right', prop={'size': 20})
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("ST-502 (Turbine Speed)", fontsize=20)
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    # print track_output[index]
    plt.show()

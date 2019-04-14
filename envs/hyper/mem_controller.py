import mod_hyper as mod, math
from scipy.special import expit
import numpy as np, os
from random import randint
from torch.autograd import Variable
import torch
import random
from torch.utils import data as util
import matplotlib.pyplot as plt
from neuroevolution import SSNE as SSNE


class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername + '/0000_CSV'
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'Controller.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/train_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/valid_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')


class Parameters:
    def __init__(self):
            self.pop_size = 50
            self.load_seed = False #Loads a seed population from the save_foldername
                                  # IF FALSE: Runs Backpropagation, saves it and uses that
            # Determine the neural archiecture
            self.output_activation = None

            #Controller choices
            self.target_sensor = 11 #Turbine speed the sensor to control
            self.run_time = 300 #Controller Run time

            #Controller noise
            self.sensor_noise = 0.05
            self.sensor_failure = 0.0
            self.actuator_noise = 0.0

            # Reconfigurability parameters
            self.is_random_initial_state = False  # Start state of controller
            self.num_profiles = 3
            self.reconf_shape = 1 #1 Periodic shape, #2 Mimicking real shape #0: flat line tracking

            #GD Stuff
            self.total_epochs = 50
            self.batch_size = 10

            #SSNE stuff
            self.num_input = 20
            self.num_hnodes = 30
            self.num_mem = self.num_hnodes
            self.num_output = 2
            self.elite_fraction = 0.07
            self.crossover_prob = 0.05
            self.mutation_prob = 0.9
            self.weight_magnitude_limit = 1000000
            self.extinction_prob = 0.004  # Probability of extinction event
            self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
            self.mut_distribution = 0  # 1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s
            self.total_gens = 10000
            self.num_evals = 10 #Number of independent evaluations before getting a fitness score
            self.save_foldername = 'R_Reconfigurable_Controller/'


class Fast_Simulator(): #TF Simulator individual (One complete simulator genome)
    def __init__(self):
        self.W = None

    def predict(self, input):
        # Feedforward operation
        h_1 = expit(np.dot(input.detach().numpy(), self.W[0]) + self.W[1])
        return np.dot(h_1, self.W[2]) + self.W[3]


class Task_Controller: #Reconfigurable Control Task
    def __init__(self, parameters):
        self.parameters = parameters
        self.num_input = parameters.num_input; self.num_hidden = parameters.num_hnodes; self.num_output = parameters.num_output

        self.train_data = self.data_preprocess() #Get simulator data
        self.ssne = SSNE(parameters) #Initialize SSNE engine

        # Save folder for checkpoints
        self.save_foldername = self.parameters.save_foldername
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        #Load simulator
        self.simulator = mod.unpickle('Champion_Simulator')
        #mod.simulator_results(self.simulator)

        #####Create Reconfigurable controller population
        self.pop = []
        for i in range(self.parameters.pop_size):
            # Choose architecture
            self.pop.append(mod.MMU(self.num_input, self.num_hidden, parameters.num_mem, self.num_output))

        ###Initialize Controller Population
        if self.parameters.load_seed:
            self.pop[0] = mod.unpickle('R_Controller/seed_controller')  # Load PT_GRUMB object
        # else:  # Run Backprop
        #     self.run_bprop(self.pop[0])

    def save(self, individual, filename ):
        mod.pickle_object(individual, filename)

    def predict(self, individual, input): #Runs the individual net and computes and output by feedforwarding
        return individual.predict(input)

    def run_bprop(self, model):
        #Get train_x
        sensor_target = self.train_data[1:, self.parameters.target_sensor:self.parameters.target_sensor + 1]  #Sensor target that needs to me met
        all_train_x = self.train_data[0:-1,0:-2]
        all_train_x = np.concatenate((all_train_x, sensor_target), axis=1) #Input training data

        #Get Train_y
        all_train_y = self.train_data[0:-1,-2:] #Target Controller Output

        if True: #GD optimizer choices
            # criterion = torch.nn.L1Loss(False)
            criterion = torch.nn.SmoothL1Loss(False)
            # criterion = torch.nn.KLDivLoss()
            #criterion = torch.nn.MSELoss()
            # criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum = 0.5, nesterov = True)
            # optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.005, momentum=0.1)

        #Set up training
        seq_len = 1
        all_train_x = torch.Tensor(all_train_x)#.cuda()
        all_train_y = torch.Tensor(all_train_y)#.cuda()
        train_dataset = util.TensorDataset(all_train_x, all_train_y)
        train_loader = util.DataLoader(train_dataset, batch_size=self.parameters.batch_size, shuffle=True)
        #model.cuda()

        for epoch in range(1, self.parameters.total_epochs + 1):

            epoch_loss = 0.0
            for data in train_loader:  # Each Batch
                net_inputs, targets = data
                #net_inputs = data[0]
                #targets = data[1]
                net_inputs = torch.t(net_inputs)
                #targets = torch.t(targets)

                model.reset(net_inputs.size()[1])  # Reset memory and recurrent out for the model
                for i in range(seq_len):  # For the length of the sequence
                    #net_inp = Variable(net_inputs[:, i], requires_grad=True).unsqueeze(0)
                    net_inp = Variable(net_inputs, requires_grad=True)
                    net_out = model.forward(net_inp)
                    target_T = Variable(targets)
                    loss = criterion(net_out, target_T)
                    loss.backward(retain_graph=True)
                    epoch_loss += loss.cpu().data.numpy()#[0]

                optimizer.step()  # Perform the gradient updates to weights for the entire set of collected gradients
                optimizer.zero_grad()

            print('Epoch: ', epoch, ' Loss: ', epoch_loss)

    def plot_controller(self, individual):
        setpoints = self.get_setpoint()
        sim_input = mod.unsqueeze(np.copy(self.train_data[0][0:]), axis=0)
        control_input = mod.unsqueeze(np.delete(np.copy(sim_input), -1))  # Input to the controller

        track_output = np.zeros((len(setpoints) - 1, 1))
        individual.reset(batch_size = 1)

        for example in range(len(setpoints) - 1):  # For duration of training
            # Fill in the setpoint to control input
            control_input[-1][0] = setpoints[example]

            # Add noise to the state input to the controller
            if self.parameters.sensor_noise != 0:  # Add sensor noise
                for i in range(19):
                    std = self.parameters.sensor_noise * abs(control_input[0][i])
                    if std != 0:
                        control_input[i][0] += np.random.normal(0, std)

            if self.parameters.sensor_failure != None:  # Failed sensor outputs 0 regardless
                if random.random() < self.parameters.sensor_failure:
                    control_input[11][0] = 0.0

            # RUN THE CONTROLLER TO GET CONTROL OUTPUT
            control_out = individual.predict(control_input)
            #
            # Add actuator noise (controls)
            if self.parameters.actuator_noise != 0:
                for i in range(len(control_out[0])):
                    std = self.parameters.actuator_noise * abs(control_out[0][i])
                    if std != 0:
                        control_out[i][0] += np.random.normal(0, std)


            # Fill in the controls
            sim_input[0][19] = control_out[0][0]
            sim_input[0][20] = control_out[1][0]

            # Use the simulator to get the next state
            simulator_out = self.simulator.predict(sim_input)

            # Calculate error (weakness)
            track_output[example][0] = simulator_out[0][11]

            # Fill in the simulator inputs and control inputs
            for i in range(simulator_out.shape[-1]):
                sim_input[0][i] = simulator_out[0][i]
                control_input[i][0] = simulator_out[0][i]


        plt.plot(setpoints, 'r--', label='Desired Turbine Speed')
        plt.plot(track_output, 'b-', label='Achieved Turbine Speed')
        # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
        # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
        plt.legend(loc='upper right', prop={'size': 15})
        plt.xlabel("Time", fontsize=15)
        plt.ylabel("ST-502 (Turbine Speed)", fontsize=15)
        axes = plt.gca()
        axes.set_ylim([0, 1.1])
        # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
        # print track_output[index]
        plt.show()

    def get_setpoint(self):
        if self.parameters.reconf_shape == 1:
            desired_setpoints = np.reshape(np.zeros(self.parameters.run_time), (parameters.run_time))#, 1))
            for profile in range(parameters.num_profiles):
                multiplier = randint(1, 5)
                #print profile, multiplier
                for i in range(int(self.parameters.run_time/self.parameters.num_profiles)):
                    turbine_speed = math.sin(i * 0.2 * multiplier)
                    turbine_speed *= 0.3 #Between -0.3 and 0.3
                    turbine_speed += 0.5  #Between 0.2 and 0.8 centered on 0.5
                    desired_setpoints[profile * int(self.parameters.run_time/self.parameters.num_profiles) + i] = turbine_speed

        elif self.parameters.reconf_shape == 2:
            desired_setpoints = np.zeros(self.parameters.run_time) + random.uniform(0.4, 0.6)
            noise = np.random.uniform(-0.01, 0.01, (parameters.run_time))
            desired_setpoints += noise

            for profile_id in range(self.parameters.num_profiles):
                phase_len = self.parameters.run_time/self.parameters.num_profiles
                phase_start = profile_id * phase_len; phase_end = phase_start + (phase_len)

                start = random.randint(phase_start, phase_end-35)
                end = random.randint(start+10, start + 35)
                magnitude = random.uniform(-0.25, 0.25)

                for i in range(start, end):
                    desired_setpoints[i] += magnitude

        elif self.parameters.reconf_shape == 0: # No reconfigurability, target is 0.5
            desired_setpoints = np.zeros(self.parameters.run_time) + 0.5



        # plt.plot(desired_setpoints, 'r--', label='Setpoints')
        # plt.show()
        return desired_setpoints

    def batch_copy(self, mat, batch_size, axis):
        padded_mat = np.copy(mat)
        for _ in range(batch_size-1): padded_mat = np.concatenate((padded_mat, mat), axis=axis)
        return padded_mat

    def compute_fitness(self, individual, setpoints, start_sim_input, control_input): #Controller fitness
        weakness = 0.0
        individual.reset(batch_size = self.parameters.num_evals)

        sim_input = self.batch_copy(start_sim_input, self.parameters.num_evals, axis=0) #Input to the simulator
        control_input = self.batch_copy(control_input, self.parameters.num_evals, axis=1)
        control_input = torch.Tensor(control_input)
        sim_input = torch.Tensor(sim_input)

        for example in range(self.parameters.run_time):  # For duration of run

            # Add noise to the state input to the controller
            if self.parameters.sensor_noise != 0:  # Add sensor noise, # TODO: this might break since now its Tensors
                noise_mul = np.random.normal(0, self.parameters.sensor_noise, (control_input.shape[0], control_input.shape[1]))
                control_input = np.multiply(control_input, noise_mul)

            # if random.random() < self.parameters.sensor_failure:
            #     control_input[0][11] = 0.0

            # Fill in the setpoint to control input
            #tmp=setpoints.tolist()
            if not(torch.is_tensor(setpoints)): setpoints=torch.Tensor(setpoints)
            control_input[-1, :] = setpoints[:, example]

            #RUN THE CONTROLLER TO GET CONTROL OUTPUT
            control_out = individual.predict(control_input)

            # Add actuator noise (controls)
            if self.parameters.actuator_noise != 0:  # Add actuator noise. # TODO: this might break since now its Tensors
                noise_mul = np.random.normal(0, self.parameters.actuator_noise, (control_out.shape[0], control_out.shape[1]))
                control_out = np.multiply(control_out, noise_mul)

            #Fill in the controls
            #control_out = control_out.detach().numpy()
            sim_input[:,19] = control_out[:,0]#control_out[:][0]#[0][:]
            sim_input[:,20] = control_out[:,1]#[1][:]

            # Use the simulator to get the next state
            simulator_out = self.simulator.predict(sim_input)

            # Calculate error (weakness)
            weakness += np.mean(np.fabs(simulator_out[:,self.parameters.target_sensor] - np.array(setpoints[:,example])))  # Time variant simulation

            # Fill in the simulator inputs and control inputs
            sim_input[:, 0:19] = torch.Tensor(simulator_out[:, 0:19])
            control_input[0:-1,:] = torch.Tensor(np.transpose(simulator_out[:,0:19]))

        return -weakness

    def evolve(self, gen):
        setpoints = []
        for _ in range(self.parameters.num_evals):
            setpoints.append(self.get_setpoint())
        setpoints = np.array(setpoints)

        sim_input = mod.unsqueeze(np.copy(self.train_data[0][0:]), axis=0)
        control_input = mod.unsqueeze(np.delete(np.copy(sim_input), 1))  # Input to the controller

        # Convert to tensors
        sim_input = torch.Tensor(sim_input)
        control_input = torch.Tensor(control_input)
        setpoints = torch.Tensor(setpoints)

        #Test all individuals and assign fitness
        fitness_evals = []
        for index, individual in enumerate(self.pop): #Test all genomes/individuals
            fitness_evals.append(self.compute_fitness(individual, setpoints, sim_input, control_input))
        gen_best_fitness = max(fitness_evals)

        #Validation Score
        champion_index = fitness_evals.index(max(fitness_evals))
        valid_setpoints = []
        for _ in range(self.parameters.num_evals):
            valid_setpoints.append(self.get_setpoint())
        valid_setpoints = np.array(valid_setpoints)
        valid_score = self.compute_fitness(self.pop[champion_index], valid_setpoints, sim_input, control_input)


        #Save population and Champion
        if gen % 20 == 0:
            self.save(self.pop[champion_index], self.save_foldername + 'champ_controller') #Save champion
            np.savetxt(self.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        self.ssne.epoch(self.pop, fitness_evals, [])

        return gen_best_fitness, valid_score

    def data_preprocess(self, filename='ColdAir.csv', downsample_rate=25, split = 1000):
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
        #Train/Valid split
        train_data = data[0:split]
        valid_data = data[split:len(data)]
        print("how am i here?")
        return train_data, valid_data


if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker

    control_task = Task_Controller(parameters)
    for gen in range(1, parameters.total_gens):
        gen_best_fitness, valid_score = control_task.evolve(gen)
        print('Generation:', gen, ' Epoch_reward:', "%0.2f" % gen_best_fitness, ' Valid Score:', "%0.2f" % valid_score, '  Cumul_Valid_Score:', "%0.2f" % tracker.hof_avg_fitness)
        tracker.add_fitness(gen_best_fitness, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(valid_score, gen)  # Add best global performance to tracker

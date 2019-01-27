from core.agent import Agent
from core import models
from core.mod_utils import list_mean, pprint, str2bool
import numpy as np, os, time, random, torch
from core import mod_utils as utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
import argparse
import random
import threading

parser = argparse.ArgumentParser()
parser.add_argument('-popsize', type=int,  help='#Evo Population size',  default=5)
parser.add_argument('-rollsize', type=int,  help='#Rollout size for agents',  default=5)
parser.add_argument('-render', type=str2bool,  help='#Render?',  default=False)
parser.add_argument('-savetag', help='Saved tag',  default='')
parser.add_argument('-gamma', type=float,  help='#Gamma',  default=0.99)
parser.add_argument('-seed', type=float,  help='#Seed',  default=7)

ROLLOUT_SIZE = vars(parser.parse_args())['rollsize']
SEED = vars(parser.parse_args())['seed']
POP_SIZE = vars(parser.parse_args())['popsize']
SAVE_TAG = vars(parser.parse_args())['savetag']
GAMMA = vars(parser.parse_args())['gamma']
RENDER = vars(parser.parse_args())['render']
CUDA = True


SAVE_TAG = SAVE_TAG + '_' + str(POP_SIZE) + '_' + str(ROLLOUT_SIZE)

class Parameters:
    def __init__(self):

        #Meta
        self.rollout_size = ROLLOUT_SIZE
        self.popn_size = POP_SIZE
        self.num_episodes = 100000


        #Rover domain
        self.dim_x = self.dim_y = 15; self.obs_radius = 100; self.act_dist = 2; self.angle_res = 20
        self.num_poi = 5; self.num_agents = 8; self.ep_len = 30
        self.poi_rand = True; self.coupling = 4; self.rover_speed = 1
        self.render = RENDER
        self.sensor_model = 'closest'  #Closest VS Density

        #TD3 params
        self.algo_name = 'TD3'
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        self.tau = 5e-3
        self.init_w = True
        self.gradperstep = 1.0
        self.gamma = GAMMA
        self.batch_size = 128
        self.buffer_size = 500000
        self.updates_per_step = 1
        self.action_loss = False
        self.policy_ups_freq = 2
        self.policy_noise = True
        self.policy_noise_clip = 0.2

        # NeuroEvolution stuff
        self.elite_fraction = 0.2
        self.crossover_prob = 0.15
        self.mutation_prob = 0.90
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform

        #Dependents
        self.state_dim = int(720 / self.angle_res + 4)
        self.action_dim = 2

        #Save Filenames
        self.save_foldername = 'R_MERL/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        self.metric_save = self.save_foldername + 'metrics/'
        self.model_save = self.save_foldername + 'models/'
        self.aux_save = self.save_foldername + 'auxiliary/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
        if not os.path.exists(self.model_save): os.makedirs(self.model_save)
        if not os.path.exists(self.aux_save): os.makedirs(self.aux_save)

        self.critic_fname = 'critic_' +SAVE_TAG
        self.actor_fname = 'actor_'+ SAVE_TAG
        self.log_fname = 'reward_'+  SAVE_TAG
        self.best_fname = 'best_'+ SAVE_TAG

        #Unit tests (Simply changes the rover/poi init locations)
        self.unit_test = 0 #0: None
                           #1: Single Agent
                           #2: Multiagent 2-coupled


class MERL:
    """Policy Gradient Algorithm main object which carries out off-policy learning using policy gradient
       Encodes all functionalities for 1. TD3 2. DDPG 3.Trust-region TD3/DDPG 4. Advantage TD3/DDPG

            Parameters:
                args (int): Parameter class with all the parameters

            """

    def __init__(self, args):
        self.args = args


        ######### Initialize the Multiagent Team of agents ########
        self.agents = [Agent(self.args, id) for id in range(self.args.num_agents)]


        ###### Buffer and Model Bucket as references to the corresponding agent's attributes ####
        self.buffer_bucket = [ag.buffer.tuples for ag in self.agents]
        self.popn_bucket = [ag.popn for ag in self.agents]
        self.rollout_bucket = [ag.rollout_actor for ag in self.agents]


        ######### EVOLUTIONARY WORKERS ############
        self.evo_task_pipes = [Pipe() for _ in range(args.popn_size)]
        self.evo_result_pipes = [Pipe() for _ in range(args.popn_size)]
        self.evo_workers = [Process(target=rollout_worker, args=(self.args, i, 'evo', self.evo_task_pipes[i][1], self.evo_result_pipes[i][0],
                                                                   self.buffer_bucket, self.popn_bucket)) for i in range(args.popn_size)]
        for worker in self.evo_workers: worker.start()


        ######### POLICY GRADIENT WORKERS ############
        self.pg_task_pipes = [Pipe() for _ in range(args.rollout_size)]
        self.pg_result_pipes = [Pipe() for _ in range(args.rollout_size)]
        self.pg_workers = [Process(target=rollout_worker, args=(self.args, i, 'pg', self.pg_task_pipes[i][1], self.pg_result_pipes[i][0],
                                                                   self.buffer_bucket, self.rollout_bucket)) for i in range(args.rollout_size)]
        for worker in self.pg_workers: worker.start()


        #### STATS AND TRACKING WHICH ROLLOUT IS DONE ######
        self.best_score = -999; self.total_frames = 0; self.gen_frames = 0


    def make_teams(self, num_agents, popn_size):
        all_inds = [list(range(popn_size)) for _ in range(num_agents)]
        for entry in all_inds: random.shuffle(entry)

        teams = [[entry[i] for entry in all_inds] for i in range(popn_size)]

        return teams


    def train(self):
        """Main training loop to do rollouts and run policy gradients

            Parameters:
                gen (int): Current epoch of training

            Returns:
                None
        """

        #Figure out teams for Coevolution
        teams = self.make_teams(args.num_agents, args.popn_size)

        ########## START EVO ROLLOUT ##########
        for pipe, team in zip(self.evo_task_pipes, teams):
            pipe[0].send(team)



        ########## START POLICY GRADIENT ROLLOUT ##########
        #Synch pg_actors to its corresponding rollout_bucket
        for agent in self.agents: agent.update_rollout_actor()

        #Start rollouts using the rollout actors
        for id, pipe in enumerate(self.pg_task_pipes):
            pipe[0].send([id for _ in range(self.args.num_agents)]) #Index 0 for the Rollout bucket




        ############ POLICY GRADIENT UPDATES #########

        # Spin up threads for each agent
        threads = [threading.Thread(target=agent.update_parameters, args=()) for agent in self.agents]

        # Start threads
        for thread in threads: thread.start()

        # Join threads
        for thread in threads: thread.join()


        all_fits = []
        ####### JOIN EVO ROLLOUTS ########
        for pipe in self.evo_result_pipes:
            entry = pipe[1].recv()
            team = entry[0]; fitness = entry[1][0]

            for agent_id, popn_id in enumerate(team): self.agents[agent_id].fitnesses[popn_id] = fitness ##Assign
            all_fits.append(fitness)


        ####### JOIN PG ROLLOUTS ########
        for pipe in self.pg_result_pipes:
            _ = pipe[1].recv()


        # #Save models periodically
        # if gen % 20 == 0:
        #     for rover_id in range(self.args.num_rover):
        #         torch.save(self.agents[rover_id].critic.state_dict(), self.args.model_save + self.args.critic_fname + '_'+ str(rover_id))
        #         torch.save(self.agents[rover_id].actor.state_dict(), self.args.model_save + self.args.actor_fname + '_'+ str(rover_id))
        #     print("Models Saved")

        return max(all_fits)





if __name__ == "__main__":
    args = Parameters()  # Create the Parameters class
    gen_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')  # Initiate tracker
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)    #Seeds

    # INITIALIZE THE MAIN AGENT CLASS
    ai = MERL(args)
    print(' State_dim:', args.state_dim, 'Action_dim', args.action_dim)
    time_start = time.time()

    ###### TRAINING LOOP ########
    for gen in range(1, 1000000000): #RUN VIRTUALLY FOREVER
        gen_time = time.time()

        #ONE EPOCH OF TRAINING
        best_score = ai.train()

        #PRINT PROGRESS
        print('Ep:', gen, 'Gen_best:', pprint(best_score),
              'Time:',pprint(time.time()-gen_time))

        gen_tracker.update([best_score], gen)




    


import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from core import mod_utils as utils
from core.models import Actor, Critic, GaussianPolicy, QNetwork, ValueNetwork, DeterministicPolicy


class Off_Policy_Algo(object):
    """Classes implementing TD3 and DDPG off-policy learners

         Parameters:
               args (object): Parameter class


     """
    def __init__(self, algo_name, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, init_w = True):

        self.algo_name = algo_name; self.gamma = gamma; self.tau = tau

        #Initialize actors
        self.actor = Actor(state_dim, action_dim)
        if init_w: self.actor.apply(utils.init_weights)
        self.actor_target = Actor(state_dim, action_dim)
        utils.hard_update(self.actor_target, self.actor)
        self.actor_optim = Adam(self.actor.parameters(), actor_lr)


        self.critic = Critic(state_dim, action_dim)
        if init_w: self.critic.apply(utils.init_weights)
        self.critic_target = Critic(state_dim, action_dim)
        utils.hard_update(self.critic_target, self.critic)
        self.critic_optim = Adam(self.critic.parameters(), critic_lr)

        self.loss = nn.MSELoss()

        self.actor_target.cuda(); self.critic_target.cuda(); self.actor.cuda(); self.critic.cuda()
        self.num_critic_updates = 0

        #Statistics Tracker
        self.action_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.policy_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.critic_loss = {'mean':[]}
        self.q = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.val = {'min':[], 'max': [], 'mean':[], 'std':[]}

    def compute_stats(self, tensor, tracker):
        """Computes stats from intermediate tensors

             Parameters:
                   tensor (tensor): tensor
                   tracker (object): logger

             Returns:
                   None


         """
        tracker['min'].append(torch.min(tensor).item())
        tracker['max'].append(torch.max(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())

    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, num_epoch=1, **kwargs):
        """Runs a step of Bellman upodate and policy gradient using a batch of experiences

             Parameters:
                  state_batch (tensor): Current States
                  next_state_batch (tensor): Next States
                  action_batch (tensor): Actions
                  reward_batch (tensor): Rewards
                  done_batch (tensor): Done batch
                  num_epoch (int): Number of learning iteration to run with the same data

             Returns:
                   None

         """

        if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch)

        for _ in range(num_epoch):
            ########### CRITIC UPDATE ####################

            #Compute next q-val, next_v and target
            with torch.no_grad():
                #Policy Noise
                policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1]))
                policy_noise = torch.clamp(torch.Tensor(policy_noise), -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])

                #Compute next action_bacth
                next_action_batch = self.actor_target.forward(next_state_batch) + policy_noise.cuda()
                next_action_batch = torch.clamp(next_action_batch, -1,1)

                #Compute Q-val and value of next state masking by done
                q1, q2, next_val = self.critic_target.forward(next_state_batch, next_action_batch)
                q1 = (1 - done_batch) * q1
                q2 = (1 - done_batch) * q2
                next_val = (1 - done_batch) * next_val

                #Select which q to use as next-q (depends on algo)
                if self.algo_name == 'TD3' or self.algo_name == 'TD3_actor_min': next_q = torch.min(q1, q2)
                elif self.algo_name == 'DDPG': next_q = q1
                elif self.algo_name == 'TD3_max': next_q = torch.max(q1, q2)

                #Compute target q and target val
                target_q = reward_batch + (self.gamma * next_q)
                #if self.args.use_advantage: target_val = reward_batch + (self.gamma * next_val)


            self.critic_optim.zero_grad()
            current_q1, current_q2, current_val = self.critic.forward((state_batch), (action_batch))
            self.compute_stats(current_q1, self.q)

            dt = self.loss(current_q1, target_q)
            # if self.args.use_advantage:
            #     dt = dt + self.loss(current_val, target_val)
            #     self.compute_stats(current_val, self.val)

            if self.algo_name == 'TD3' or self.algo_name == 'TD3_max': dt = dt + self.loss(current_q2, target_q)
            self.critic_loss['mean'].append(dt.item())

            # if self.args.critic_constraint:
            #     if dt.item() > self.args.critic_constraint_w:
            #         dt = dt * (abs(self.args.critic_constraint_w / dt.item()))
            dt.backward()

            self.critic_optim.step()
            self.num_critic_updates += 1


            #Delayed Actor Update
            if self.num_critic_updates % kwargs['policy_ups_freq'] == 0:

                actor_actions = self.actor.forward(state_batch)

                # # Trust Region constraint
                # if self.args.trust_region_actor:
                #     with torch.no_grad(): old_actor_actions = self.actor_target.forward(state_batch)
                #     actor_actions = action_batch - old_actor_actions


                Q1, Q2, val = self.critic.forward(state_batch, actor_actions)

                # if self.args.use_advantage: policy_loss = -(Q1 - val)
                policy_loss = -Q1

                self.compute_stats(policy_loss,self.policy_loss)
                policy_loss = policy_loss.mean()


                self.actor_optim.zero_grad()



                policy_loss.backward(retain_graph=True)
                #nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                # if self.args.action_loss:
                #     action_loss = torch.abs(actor_actions-0.5)
                #     self.compute_stats(action_loss, self.action_loss)
                #     action_loss = action_loss.mean() * self.args.action_loss_w
                #     action_loss.backward()
                #     #if self.action_loss[-1] > self.policy_loss[-1]: self.args.action_loss_w *= 0.9 #Decay action_w loss if action loss is larger than policy gradient loss
                self.actor_optim.step()


            # if self.args.hard_update:
            #     if self.num_critic_updates % self.args.hard_update_freq == 0:
            #         if self.num_critic_updates % self.args.policy_ups_freq == 0: self.hard_update(self.actor_target, self.actor)
            #         self.hard_update(self.critic_target, self.critic)


            if self.num_critic_updates % kwargs['policy_ups_freq'] == 0: utils.soft_update(self.actor_target, self.actor, self.tau)
            utils.soft_update(self.critic_target, self.critic, self.tau)




class SAC(object):
    def __init__(self, num_inputs, action_dim, gamma, wwid):

        self.num_inputs = num_inputs
        self.action_space = action_dim
        self.gamma = gamma
        self.tau = 0.005
        self.alpha = 0.2
        self.policy_type = "Gaussian"
        self.target_update_interval = 1

        self.critic = QNetwork(self.num_inputs, self.action_space, 256)
        self.critic_optim = Adam(self.critic.parameters(), lr=3e-4)
        self.soft_q_criterion = nn.MSELoss()

        if self.policy_type == "Gaussian":
            self.policy = GaussianPolicy(self.num_inputs, self.action_space, 256, wwid)
            self.policy_optim = Adam(self.policy.parameters(), lr=3e-4)

            self.value = ValueNetwork(self.num_inputs, 256)
            self.value_target = ValueNetwork(self.num_inputs, 256)
            self.value_optim = Adam(self.value.parameters(), lr=3e-4)
            utils.hard_update(self.value_target, self.value)
            self.value_criterion = nn.MSELoss()
        else:
            self.policy = DeterministicPolicy(self.num_inputs, self.action_space, 256)
            self.policy_optim = Adam(self.policy.parameters(), lr=3e-4)

            self.critic_target = QNetwork(self.num_inputs, self.action_space, 256)
            utils.hard_update(self.critic_target, self.critic)

        self.policy.cuda()
        self.value.cuda()
        self.value_target.cuda()
        self.critic.cuda()



    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if eval == False:
            self.policy.train()
            action, _, _, _, _ = self.policy.evaluate(state)
        else:
            self.policy.eval()
            _, _, _, action, _ = self.policy.evaluate(state)

        # action = torch.tanh(action)
        action = action.detach().cpu().numpy()
        return action[0]

    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, mask_batch, updates, **ignore):
        # state_batch = torch.FloatTensor(state_batch)
        # next_state_batch = torch.FloatTensor(next_state_batch)
        # action_batch = torch.FloatTensor(action_batch)
        # reward_batch = torch.FloatTensor(reward_batch)
        # mask_batch = torch.FloatTensor(np.float32(mask_batch))

        # reward_batch = reward_batch.unsqueeze(1)  # reward_batch = [batch_size, 1]
        # mask_batch = mask_batch.unsqueeze(1)  # mask_batch = [batch_size, 1]

        """
        Use two Q-functions to mitigate positive bias in the policy improvement step that is known
        to degrade performance of value based methods. Two Q-functions also significantly speed
        up training, especially on harder task.
        """
        expected_q1_value, expected_q2_value = self.critic(state_batch, action_batch)
        new_action, log_prob, _, mean, log_std = self.policy.evaluate(state_batch)

        if self.policy_type == "Gaussian":
            """
            Including a separate function approximator for the soft value can stabilize training.
            """
            expected_value = self.value(state_batch)
            target_value = self.value_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * target_value  # Reward Scale * r(st,at) - γV(target)(st+1))
        else:
            """
            There is no need in principle to include a separate function approximator for the state value.
            We use a target critic network for deterministic policy and eradicate the value value network completely.
            """
            next_state_action, _, _, _, _, = self.policy.evaluate(next_state_batch)
            target_critic_1, target_critic_2 = self.critic_target(next_state_batch, next_state_action)
            target_critic = torch.min(target_critic_1, target_critic_2)
            next_q_value = reward_batch + mask_batch * self.gamma * target_critic  # Reward Scale * r(st,at) - γQ(target)(st+1)

        """
        Soft Q-function parameters can be trained to minimize the soft Bellman residual
        JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        ∇JQ = ∇Q(st,at)(Q(st,at) - r(st,at) - γV(target)(st+1))
        """
        q1_value_loss = self.soft_q_criterion(expected_q1_value, next_q_value.detach())
        q2_value_loss = self.soft_q_criterion(expected_q2_value, next_q_value.detach())
        q1_new, q2_new = self.critic(state_batch, new_action)
        expected_new_q_value = torch.min(q1_new, q2_new)

        if self.policy_type == "Gaussian":
            """
            Including a separate function approximator for the soft value can stabilize training and is convenient to 
            train simultaneously with the other networks
            Update the V towards the min of two Q-functions in order to reduce overestimation bias from function approximation error.
            JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - log π(at|st)]))^2]
            ∇JV = ∇V(st)(V(st) - Q(st,at) + logπ(at|st))
            """
            next_value = expected_new_q_value - (self.alpha * log_prob)
            value_loss = self.value_criterion(expected_value, next_value.detach())
        else:
            pass

        """
        Reparameterization trick is used to get a low variance estimator
        f(εt;st) = action sampled from the policy
        εt is an input noise vector, sampled from some fixed distribution
        Jπ = 𝔼st∼D,εt∼N[logπ(f(εt;st)|st)−Q(st,f(εt;st))]
        ∇Jπ =∇log π + ([∇at log π(at|st) − ∇at Q(st,at)])∇f(εt;st)
        """
        policy_loss = ((self.alpha * log_prob) - expected_new_q_value).mean()

        # Regularization Loss
        mean_loss = 0.001 * mean.pow(2).mean()
        std_loss = 0.001 * log_std.pow(2).mean()

        policy_loss += mean_loss + std_loss

        self.critic_optim.zero_grad()
        q1_value_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        q2_value_loss.backward()
        self.critic_optim.step()

        if self.policy_type == "Gaussian":
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
        else:
            value_loss = torch.tensor(0.)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        """
        We update the target weights to match the current value function weights periodically
        Update target parameter after every n(args.target_update_interval) updates
        """
        if updates % self.target_update_interval == 0 and self.policy_type == "Deterministic":
            utils.soft_update(self.critic_target, self.critic, self.tau)

        elif updates % self.target_update_interval == 0 and self.policy_type == "Gaussian":
            utils.soft_update(self.value_target, self.value, self.tau)
        return value_loss.item(), q1_value_loss.item(), q2_value_loss.item(), policy_loss.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        if value_path is None:
            value_path = "models/sac_value_{}_{}".format(env_name, suffix)
        print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
        torch.save(self.value.state_dict(), value_path)
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path, value_path):
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            self.value.load_state_dict(torch.load(value_path))








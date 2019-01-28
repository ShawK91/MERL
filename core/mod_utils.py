from torch import nn
from torch.autograd import Variable
import random, pickle, copy, argparse
import numpy as np, torch, os

class Tracker(): #Tracker
    """Tracker class to log progress and save metrics periodically

    Parameters:
        save_folder (str): Folder name for saving progress
        vars_string (list): List of metric names to log
        project_string: (str): String decorator for metric filenames

    Returns:
        None
    """

    def __init__(self, save_folder, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = save_folder
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        self.counter = 0
        self.conv_size = 10
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)


    def update(self, updates, generation):
        """Add a metric observed

        Parameters:
            updates (list): List of new scoresfor each tracked metric
            generation (int): Current gen

        Returns:
            None
        """

        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update == None: continue
            var[0].append(update)

        #Constrain size of convolution
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size: var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0])/float(len(var[0]))

        if self.counter % 1 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0: continue
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def hard_update(target, source):
    """Hard update (clone) from target network to source

        Parameters:
              target (object): A pytorch model
              source (object): A pytorch model

        Returns:
            None
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

    #Signature transfer if applicable
    try:
        target.wwid[0] = source.wwid[0]
    except:
	    None


def soft_update(target, source, tau):
    """Soft update from target network to source

        Parameters:
              target (object): A pytorch model
              source (object): A pytorch model
              tau (float): Tau parameter

        Returns:
            None

    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def to_numpy(var):
    """Tensor --> numpy

    Parameters:
        var (tensor): tensor

    Returns:
        var (ndarray): ndarray
    """
    return var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False):
    """numpy --> Variable

    Parameters:
        ndarray (ndarray): ndarray
        volatile (bool): create a volatile tensor?
        requires_grad (bool): tensor requires gradients?

    Returns:
        var (variable): variable
    """

    if isinstance(ndarray, list): ndarray = np.array(ndarray)
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

def pickle_obj(filename, object):
    """Pickle object

    Parameters:
        filename (str): folder to dump pickled object
        object (object): object to pickle

    Returns:
        None
    """

    handle = open(filename, "wb")
    pickle.dump(object, handle)

def unpickle_obj(filename):
    """Unpickle object from disk

    Parameters:
        filename (str): file from which to load and unpickle object

    Returns:
        obj (object): unpickled object
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def init_weights(m):
    """Initialize weights using kaiming uniform initialization in place

    Parameters:
        m (nn.module): Linear module from torch.nn

    Returns:
        None
    """
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def list_mean(l):
    """compute avergae from a list

    Parameters:
        l (list): list

    Returns:
        mean (float): mean
    """
    if len(l) == 0: return None
    else: return sum(l)/len(l)

def pprint(l):
    """Pretty print

    Parameters:
        l (list/float/None): object to print

    Returns:
        pretty print str
    """

    if isinstance(l, list):
        if len(l) == 0: return None
    else:
        if l == None: return None
        else: return '%.2f'%l


# def process_dict(state_desc):
#
#     # Augmented environment from the L2R challenge
#     res = []
#     pelvis = None
#
#     for body_part in ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
#         if body_part in ["toes_r", "talus_r"]:
#             res += [0] * 9
#             continue
#         cur = []
#         cur += state_desc["body_pos"][body_part][0:2]
#         cur += state_desc["body_vel"][body_part][0:2]
#         cur += state_desc["body_acc"][body_part][0:2]
#         cur += state_desc["body_pos_rot"][body_part][2:]
#         cur += state_desc["body_vel_rot"][body_part][2:]
#         cur += state_desc["body_acc_rot"][body_part][2:]
#         if body_part == "pelvis":
#             pelvis = cur
#             res += cur[1:]
#         else:
#             cur_upd = cur
#             cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
#             cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6, 7)]
#             res += cur
#
#     for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
#         res += state_desc["joint_pos"][joint]
#         res += state_desc["joint_vel"][joint]
#         res += state_desc["joint_acc"][joint]
#
#     for muscle in sorted(state_desc["muscles"].keys()):
#         res += [state_desc["muscles"][muscle]["activation"]]
#         res += [state_desc["muscles"][muscle]["fiber_length"]]
#         res += [state_desc["muscles"][muscle]["fiber_velocity"]]
#
#     cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
#     res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]
#
#     return res

# Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')
#
# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__

def flatten(d):
    """Recursive method to flatten a dict -->list

        Parameters:
            d (dict): dict

        Returns:
            l (list)
    """

    res = []  # Result list
    if isinstance(d, dict):
        for key, val in sorted(d.items()):
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    else:
        res = [d]
    return res

def reverse_flatten(d, l):
    """Recursive method to unflatten a list -->dict [Reverse of flatten] in place

        Parameters:
            d (dict): dict
            l (list): l

        Returns:
            None
    """

    if isinstance(d, dict):
        for key, _ in sorted(d.items()):

            #FLoat is immutable so
            if isinstance(d[key], float):
                d[key] = l[0]
                l[:] = l[1:]
                continue

            reverse_flatten(d[key], l)
    elif isinstance(d, list):
        d[:] = l[0:len(d)]
        l[:] = l[len(d):]


def load_all_models_dir(dir, model_template):
    """Load all models from a given directory onto a template

        Parameters:
            dir (str): directory
            model_template (object): Class template to load the objects onto

        Returns:
            models (list): list of loaded objects
    """

    list_files = os.listdir(dir)
    print(list_files)
    models = []
    for i, fname in enumerate(list_files):
        try:
            model_template.load_state_dict(torch.load(dir + fname))
            model_template.eval()
            models.append(copy.deepcopy(model_template))
        except:
            print(fname, 'failed to load')
    return models



# def reverse_flatten(template, l):
#     print (len(l))
#     if isinstance(template, dict):
#         for key1, val1 in sorted(template.items()):
#
#             ########## LEVEL 2 ############
#             if isinstance(val1, dict):
#                 for key2, val2 in sorted(val1.items()):
#
#                     ########## LEVEL 3 ############
#                     if isinstance(val2, dict):
#                         for key3, val3 in sorted(val2.items()):
#
#                             ########## LEVEL 4 ############
#                             if isinstance(val3, dict):
#                                 for key4, val4 in sorted(val3.items()):
#
#                                     ########## LEVEL 5 ############
#                                     if isinstance(val4, dict):
#                                         for key5, val5 in sorted(val4.items()):
#
#                                             if isinstance(val5, list) and len(val5) != 0:
#                                                 template[key1][key2][key3][key4][key5] = l[0:len(val5)]
#                                                 l = l[len(val5):]
#
#
#                                             elif isinstance(val5, float):
#                                                 template[key1][key2][key3][key4][key5] = val5
#                                                 l = l[1:]
#
#                                             else: print(val5)
#
#                                     ########## LEVEL 5 ENDS ############
#
#                                     elif isinstance(val4, list) and len(val4) != 0:
#                                         template[key1][key2][key3][key4] = l[0:len(val4)]
#                                         l = l[len(val4):]
#
#
#                                     elif isinstance(val4, float):
#                                         template[key1][key2][key3][key4] = val4
#                                         l = l[1:]
#
#                                     else: print(val4)
#
#                             ########## LEVEL 4 ENDS ############
#
#                             elif isinstance(val3, list) and len(val3) != 0:
#                                 template[key1][key2][key3] = l[0:len(val3)]
#                                 l = l[len(val3):]
#
#
#                             elif isinstance(val3, float):
#                                 template[key1][key2][key3] = val3
#                                 l = l[1:]
#
#                             else: print(val3)
#
#                     ########## LEVEL 3 ENDS ############
#
#                     elif isinstance(val2, list)and len(val2) != 0:
#                         template[key1][key2] = l[0:len(val2)]
#                         l = l[len(val2):]
#
#
#                     elif isinstance(val2, float):
#                         template[key1][key2] = val2
#                         l = l[1:]
#
#                     else: print(val2)
#
#              ########## LEVEL 2 ENDS ############
#
#             elif isinstance(val1, list) and len(val1) != 0:
#                 template[key1] = l[0:len(val1)]
#                 l = l[len(val1):]
#
#             elif isinstance(val1, float):
#                 template[key1] = val1
#                 l = l[1:]
#
#             else: print(val1)
#
#
#
#     return template





import os
import torch
import numpy as np
from os import listdir

#TODO add support for optimizer tensor type
#TODO add types for function heads
#TODO support early stopping
def save_network(network,
                 optimizer,
                 save_dir,
                 model_name,
                 global_step,
                 use_gpu=True):
    '''
    save current neural network parameters and optimizer parameters
    currently the use_gpu flag only takes care of parameters in network,
    some optimizers has tensors in it. Be careful on optimizers when use
    the save load functions

    :param network: the pytorch neural network that should be saved
    :param optimizer: the optimizer for network training
    :param save_dir: path to the checkpoint directory
    :param model_name: name of the model
    :param global_step: An integer indicates how many steps the training
                        had run, (usually steps is the number of iterations and
                        number of epoches)
    :param use_gpu: a parameter indicates if the input neural network is on gpu
    :return: None
    '''
    if  '_' in model_name or '.' in model_name:
        raise ValueError('model name can not contain "." or "_"')
    save_filename = '%s_%s.pth' % (model_name, global_step)
    save_path = os.path.join(save_dir, save_filename)
    state = {'state_dict': network.cpu().state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, save_path)
    if use_gpu:
        network.cuda()

#TODO add support for optimizer tensor type

def load_last_checkpoint(network,
                         optimizer,
                         save_dir,
                         model_name,
                         use_gpu,
                         reset_optimizer = False):
    '''
    load the latest checkpoint from checkpoint folder
    currently the use_gpu flag only takes cares of parameters in network,
    some optimizers has tensors in it. Be careful on optimizers when use
    the save load functions
    :param network: the pytorch neural network
    :param optimizer: the optimizer for network training
    :param save_dir: where the checkpoint file sits
    :param model_name: name of the model
    :param use_gpu: a parameter indicates if the output neural network is on gpu
    :reset_optimizer: if set to true, optimizer will not load parameters in checkpoint dict
    :return: network with loaded parameters, optimizer with loaded parameter, iternumber of last checkpoint
    '''
    if  '_' in model_name or '.' in model_name:
        raise ValueError('model name can not contain "." or "_"')
    checkpoint_paths = [f for f in listdir(save_dir) if f.endswith('pth')]
    if not checkpoint_paths:
        print('first iteration, initialize model')
        return network, 0
    raw_names = [f.split('.')[0] for f in checkpoint_paths]
    iter_numbers = [int(f.split('_')[1]) for f in raw_names]
    max_global_step = np.max(np.asarray(iter_numbers))
    network, optimizer = load_network(network,
                           optimizer,
                           save_dir,
                           model_name,
                           max_global_step,
                           use_gpu,
                           reset_optimizer)
    return network, optimizer, max_global_step


def load_network(network,
                 optimizer,
                 save_dir,
                 model_name,
                 global_step,
                 use_gpu,
                 reset_optimizer=False):
    '''
    load neural network parameters
    currently the use_gpu flag only takes cares of parameters in network,
    some optimizers has tensors in it. Be careful on optimizers when use
    the save load functions
    :param network: the pytorch neural network (to be safe the model should be on cpu)
    :param optimizer: the optimizer for network training
    :param save_dir: where the checkpoint file sits
    :param model_name: name of the model
    :param global_step: An integer indicates how many steps the training
                        had run, (usually steps is the number of iterations and
                        number of epoches)
    :param reset_optimizer: if set to true, optimizer will not load parameters in checkpoint dict

    :return: network with loaded parameters, optimizer with loaded parameter,
    '''
    if  '_' in model_name or '.' in model_name:
        raise ValueError('model name can not contain "." or "_"')
    network.cpu()
    save_filename = '%s_%s.pth' % (model_name, global_step)
    save_path = os.path.join(save_dir, save_filename)
    state_dicts = torch.load(save_path)
    network.load_state_dict(state_dicts['state_dict'])
    if use_gpu:
        network.cuda()
    if not reset_optimizer:
        optimizer.load_state_dict(state_dicts['optimizer'])
    print('load checkpoint from {}'.format(save_filename))
    return network, optimizer

from pennylane import numpy as np
import torch

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)

def cross_entropy_loss(predict, label):
    '''
    Multi-class classifier
    '''
    p = softmax(predict)
    m = label.shape[0]
    log_likelihood = -np.log(p[range(m), label])
    return np.sum(log_likelihood) / m

def mse_loss(predict, label):
    return np.mean((predict - label)**2)

def cross_entropy(predict, label):
    '''
    Binary classifier
    '''
    prob_pos = np.exp(predict) / (np.exp(predict) + np.exp(1 - predict))
    prob_neg = 1 - prob_pos
    prob = np.concatenate((prob_pos[:, np.newaxis], prob_neg[:, np.newaxis]), axis=-1)
    label_onehot = ((label + 1) / 2)[:, np.newaxis]
    label_onehot = np.concatenate((label_onehot, 1 - label_onehot), axis=-1)
    prob = np.sum(prob * label_onehot, axis=-1)
    return np.mean(-prob * np.log(prob))

def accuracy(predicts, labels):
    '''
    Binary classifier
    '''
    assert len(predicts) == len(labels)
    return np.sum((np.sign(predicts)*labels+1)/2)/len(predicts)

def cost(param, circuit, feat, label, weight_decay=0, loss_name='MSE'):
    exp = [circuit(param, feat=feat[i]) for i in range(len(feat))]
    if loss_name == 'MSE':
        return mse_loss(np.array(exp), label) + weight_decay * np.linalg.norm(param)
    else:
        return cross_entropy_loss(np.array(exp), label) + weight_decay * np.linalg.norm(param)

# def accuracy(predicts, labels):
#     '''
#     Multi-class
#     '''


def build_optimizer(cfg, model):
     params = []
     for key, value in model.named_parameters():
         if not value.requires_grad:
             print('{} will not be updated'.format(key))
             continue

         lr = cfg.TRAIN.LR
         scale = 1.0
         if 'layer' in key:
             scale = 10.0
         weight_decay = cfg.TRAIN.WEIGHT_DECAY
         params += [{"name": key, "params": [value], "lr": lr*scale, "weight_decay": weight_decay}]

     if cfg.TRAIN.OPT.NAME == 'SGD':
         opt = torch.optim.SGD(params, momentum=cfg.TRAIN.OPT.MOMENTUM)
     elif cfg.TRAIN.OPT.NAME == 'ADAM':
         opt = torch.optim.Adam(params)
     return opt

def build_scheduler(cfg, opt):
    return torch.optim.lr_scheduler.StepLR(opt, cfg.TRAIN.OPT.STEPS, gamma=cfg.TRAIN.OPT.GAMMA, last_epoch=-1)

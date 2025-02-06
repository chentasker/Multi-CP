import torch.optim as optim
import torch.nn as nn
from load_config import *
config = load_config()

def ce_divergence(outputs_list,true_labels,ith_index,alpha=1):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    ce_loss=nn.CrossEntropyLoss()
    l_heads=outputs_list.copy()
    ith_head=l_heads.pop(ith_index)
    num_remining_heads=len(l_heads)
    ce=ce_loss(ith_head,true_labels)
    cs=0
    for r_head in l_heads:
       cs+=cos(ith_head, r_head).mean()
    cs /= num_remining_heads * (num_remining_heads - 1)
    l_loss=ce+alpha*cs
    return l_loss


def CE_LOSS(outputs_list, true_labels, ith_head):
    ce_loss = nn.CrossEntropyLoss()
    return ce_loss(outputs_list[ith_head], true_labels)

def set_criterion(current_config):
    if current_config['current_step']['criterion_name']=='CE':
        return CE_LOSS
    elif current_config['current_step']['criterion_name']=='CE divergence':
        return ce_divergence

def set_optimizer(model,current_config):
    weight_decay = current_config['current_step'].get('weight_decay', 0)
    momentum = current_config['current_step'].get('momentum', 0)
    if current_config['current_step']['optimizer_name']=='Adam':
        return optim.Adam([{'params': model.features.parameters(), 'lr': current_config['current_step']['lr_features']},
                {'params': model.classification_heads.parameters(), 'lr': current_config['current_step']['lr_heads']}],weight_decay=weight_decay)
    elif current_config['current_step']['optimizer_name']=='SGD':
        return optim.SGD([{'params': model.features.parameters(), 'lr': current_config['current_step']['lr_features']},
                {'params': model.classification_heads.parameters(), 'lr': current_config['current_step']['lr_heads']}],  momentum=momentum, weight_decay=weight_decay)
    elif current_config['current_step']['optimizer_name'] == 'RMSprop':
        return optim.RMSprop([{'params': model.features.parameters(), 'lr': current_config['current_step']['lr_features']},
                {'params': model.classification_heads.parameters(), 'lr': current_config['current_step']['lr_heads']}], weight_decay=weight_decay)
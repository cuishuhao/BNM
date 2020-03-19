import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def MK_MMD(source, target, kernel_mul=2.0, kernel_num=8, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels
    
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss / float(batch_size)


def L1N(source, target, WEIGHT=None, if_weight=False):
    if not if_weight:
        L1 = torch.mean(torch.sum(torch.abs(source - target), dim=1))
        L_s = torch.mean(torch.sum(torch.abs(source), dim=1)) 
        L_t = torch.mean(torch.sum(torch.abs(target), dim=1)) 
    else:
        L1 = torch.mean(torch.sum(torch.abs(source - target), dim=1) * WEIGHT.view(-1, 1)) 
        L_s = torch.mean(torch.sum(torch.abs(source), dim=1)* WEIGHT.view(-1, 1)) 
        L_t = torch.mean(torch.sum(torch.abs(target), dim=1)* WEIGHT.view(-1, 1)) 
    return L1/(L_s + L_t)

def LP(source, target, WEIGHT=None, if_weight=False, P=2):
    if not if_weight:
        return torch.mean(torch.nn.PairwiseDistance(p=P)(source, target))
    else:
        return torch.sum(torch.nn.PairwiseDistance(p=P)(source, target) * WEIGHT.view(-1))/(torch.sum(WEIGHT) + 0.0001)

def Matching_MMD(source, target, WEIGHT, if_weight=False):
    Mask = WEIGHT.view(-1, 1).repeat(1, 2048).byte()
    source_matched = torch.masked_select(source, Mask).view(-1, 2048)
    target_matched = torch.masked_select(target, Mask).view(-1, 2048)
    return MK_MMD(source_matched, target_matched)

loss_dict = {"MMD":MK_MMD, "JAN":JAN, "L1N":L1N, "LP":LP, "Matching-MMD":Matching_MMD}

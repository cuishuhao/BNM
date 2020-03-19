import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
from torch.autograd import Variable

import time
import json
import random

from data_list import ImageList
import network
import loss
import pre_process as prep
import lr_schedule

optim_dict = {"SGD": optim.SGD}


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-7
    methodpy = -input_ * torch.log(input_ + epsilon)
    methodpy = torch.sum(methodpy, dim=1)
    return methodpy 

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=1200.0): 
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def my_l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)

def image_classification_test(iter_test,len_now, base, class1, class2, gpu=True):
    start_test = True
    Total_1k = 0.
    Total_4k = 0.
    COR_1k = 0.
    COR_4k = 0.
    COR = 0.
    Total = 0.
    print('Testing ...')
    for i in range(len_now):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        output = base(inputs)
        out1 = class1(output)
        out2 = class2(output)
        outputs = torch.cat((out1,out2),dim=1)
        if start_test:
            all_output = outputs.data.float()
            all_label = labels.data.float()
            _, predict = torch.max(all_output, 1)
            ind_1K = all_label.gt(39)
            ind_4K = torch.logical_not(all_label.gt(39))
            COR = COR + torch.sum(torch.squeeze(predict).float() == all_label)
            Total = Total + all_label.size()[0]
            COR_1k = COR_1k + torch.sum(torch.squeeze(predict).float()[ind_1K] == all_label[ind_1K])
            Total_1k = Total_1k + torch.sum(ind_1K)
            COR_4k = COR_4k + torch.sum(torch.squeeze(predict).float()[ind_4K] == all_label[ind_4K])
            Total_4k = Total_4k + torch.sum(ind_4K)
    print('Unkown_acc: '+ str(float(COR_1k)/float(Total_1k)))                                                                                                    
    print('Known_acc: '+ str(float(COR_4k)/float(Total_4k)))
    accuracy = float(COR)/float(Total)
    return accuracy

def train_classification(config):
    ## set pre-process
    prep_train  = prep.image_train(resize_size=256, crop_size=224)
    prep_test = prep.image_test(resize_size=256, crop_size=224)
               
    ## set loss
    class_criterion = nn.CrossEntropyLoss()

    ## prepare data
    TRAIN_LIST = 'data/WEB_3D3_2.txt'
    TEST_LIST = 'data/new_AwA2.txt'
    BSZ = args.batch_size

    dsets_train = ImageList(open(TRAIN_LIST).readlines(), shape = (args.img_size,args.img_size), transform=prep_train)
    loaders_train = util_data.DataLoader(dsets_train, batch_size=BSZ, shuffle=True, num_workers=8, pin_memory=True)

    dsets_test = ImageList(open(TEST_LIST).readlines(), shape = (args.img_size,args.img_size),transform=prep_test)
    loaders_test = util_data.DataLoader(dsets_test, batch_size=BSZ, shuffle=True, num_workers=4, pin_memory=True)
    
    dsets_val = ImageList(open(TEST_LIST).readlines(), shape = (args.img_size,args.img_size),transform=prep_train)
    loaders_val = util_data.DataLoader(dsets_val, batch_size=BSZ, shuffle=True, num_workers=4, pin_memory=True)

    ## set base network
    class_num = 40
    all_num = 50
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]]()
    classifier_layer1 = nn.Linear(base_network.output_num(), class_num)
    classifier_layer2 = nn.Linear(base_network.output_num(), all_num-class_num)

    ## initialization
    base_network.load_state_dict(torch.load('model/base_net_pretrained_on_I2AwA2_source_only.pkl'))
    weight_bias=torch.load('model/awa_50_cls_basic')['fc50']
    classifier_layer1.weight.data = weight_bias[:class_num,:2048]
    classifier_layer2.weight.data = weight_bias[class_num:,:2048]
    classifier_layer1.bias.data = weight_bias[:class_num,-1]
    classifier_layer2.bias.data = weight_bias[class_num:,-1]

    #gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier_layer1 = classifier_layer1.cuda()
        classifier_layer2 = classifier_layer2.cuda()
        base_network = base_network.cuda()

    ## collect parameters
    parameter_list = [{"params": classifier_layer2.parameters(), "lr":2},
                      {"params": classifier_layer1.parameters(), "lr":5},
                      {"params": base_network.parameters(), "lr":1},]

 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    
    # dataloader lenth
    len_train_source = len(loaders_train) - 1
    len_test_source = len(loaders_test) - 1
    len_val_source = len(loaders_val) - 1
    optimizer.zero_grad()

    #train
    for i in range(config["num_iterations"]):
        if ((i + 0) % config["test_interval"] == 0 and i > 100) or i== config["num_iterations"]-1 or i==0:
            base_network.train(False)
            classifier_layer1.train(False)
            classifier_layer2.train(False)
            print(str(i)+' ACC:')
            iter_target = iter(loaders_test)
            print(image_classification_test(iter_target,len_test_source, base_network, classifier_layer1,classifier_layer2, gpu=use_gpu))
            iter_target = iter(loaders_test)

        #model train
        classifier_layer1.train(True)
        classifier_layer2.train(True)
        base_network.train(True)
        
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)

        #iter dataloader
        if i % len_train_source == 0:
            iter_source = iter(loaders_train)
        if i % (len_test_source ) == 0:
            iter_target = iter(loaders_test)
        if i % (len_val_source ) == 0:
            iter_val = iter(loaders_val)

        inputs_source, labels_source = iter_source.next()
        inputs_target, _ = iter_val.next()

        if use_gpu:
            inputs_source, labels_source, inputs_target = Variable(inputs_source).cuda(), Variable(labels_source).cuda(), Variable(inputs_target).cuda()
        else:
            inputs_source, labels_source, inputs_target = Variable(inputs_source), Variable(labels_source),Variable(inputs_target)
        #network   
        features_source = base_network(inputs_source)
        features_target = base_network(inputs_target)
        
        outputs_source1 = classifier_layer1(features_source)
        outputs_source2 = classifier_layer2(features_source)
        outputs_target1 = classifier_layer1(features_target)
        outputs_target2 = classifier_layer2(features_target)
        
        outputs_source = torch.cat((outputs_source1,outputs_source2),dim=1)
        outputs_target = torch.cat((outputs_target1,outputs_target2),dim=1)

        cls_loss = class_criterion(outputs_source1, labels_source)

        #method BNM: Batch Nuclear-norm Maximization
        target_softmax = F.softmax(outputs_target, dim=1)
        if args.method=='ENT':
            transfer_loss = torch.mean(Entropy(target_softmax))/torch.log(target_softmax.shape[1])
        elif args.method=='BNM':
            transfer_loss = -torch.norm(target_softmax,'nuc')/target_softmax.shape[0]
        elif args.method=='BFM':
            transfer_loss = -torch.sqrt(torch.mean(torch.svd(target_softmax)[1]**2))
        elif args.method=='balance':
            WEIGHT = torch.sum(torch.softmax(outputs_source, dim=1)[:,:40] * target_softmax[:,:40], 1)
            max_s = torch.max(target_softmax[:,:40], 1)[0]
            methodpy_loss = torch.sum(torch.sum(target_softmax[:,class_num:],1) * (1.0 - max_s.gt(0.5).float()*WEIGHT.gt(0.6).float()))/torch.sum(1.0 - max_s.gt(0.5).float()*WEIGHT.gt(0.6).float())
            transfer_loss = methodpy_loss + 0.1 * 0.1/methodpy_loss

        total_loss = cls_loss + transfer_loss * args.w_transfer
        print("Step "+str(i)+": cls_loss: "+str(cls_loss.cpu().data.numpy())+
                             " transfer_loss: "+str(transfer_loss.cpu().data.numpy()))

        total_loss.backward()
        if (i+1)% config["opt_num"] ==0:
              optimizer.step()
              optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, nargs='?', default=48, help="batch size")
    parser.add_argument('--img_size', type=int, nargs='?', default=256, help="image size")
    parser.add_argument('--method', type=str, nargs='?', default='BNM', help="loss name")
    parser.add_argument('--w_transfer', type=float, nargs='?', default=2., help="weight of BNM")
    parser.add_argument('--lr', type=float, nargs='?', default=0.0001, help="percent of unseen data")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    config = {}
    config["num_iterations"] = 6000
    config["test_interval"] = 400
    config["opt_num"] = 1
    config["network"] = {"name":"ResNet50"}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, "weight_decay":0.0001, "nesterov":True}, "lr_type":"inv", "lr_param":{"init_lr":args.lr, "gamma":0.0005, "power":0.75} }
    print(config)
    print(args)
    train_classification(config)

# -*- coding: UTF-8 -*-

import numpy as np
import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms ,models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
from dataPreProcessing import myDataSet
# from dataPreProcessingWithShuffled import myDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
state = np.random.get_state()
train_dataset = myDataSet(func='train',state=state)
test_dataset = myDataSet(func='test',state=state)
# train_dataset = myDataSet(func='train')
# test_dataset = myDataSet(func='test')
# %%
model = models.resnext101_32x8d(pretrained=False)
pthfile = r'resnext101_32x8d-8ba56ff5.pth'
model.load_state_dict(torch.load(pthfile))
NUMCLASS = 100
fc_in = model.fc.in_features
model.fc = nn.Linear(fc_in,NUMCLASS)

# model.load_state_dict(torch.load('./resultNet/fnl_net_params_100_6resnet101.pkl'))
NAME = 'resnet101'
model.cuda()
# loss = FocalLoss(class_num=NUMCLASS) 
loss = nn.CrossEntropyLoss().cuda()
LR = 1e-3
optimizer = optim.SGD(model.parameters(),lr = LR,momentum=0.9,weight_decay=0.0005,nesterov=True)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6, last_epoch=-1)
NUM_EPOCHS = 40
BATCH_SIZE = 20
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader  = Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE)

# %%
for echo in range(NUM_EPOCHS):
    train_loss = 0   
    train_acc = 0   
    model.train()    
    for i,(X,label) in enumerate(train_loader):    
        X = Variable(X).cuda()          
        label = Variable(label).cuda()
        out = model(X)           
        lossvalue = loss(out,label)         
        optimizer.zero_grad()       
        lossvalue.backward()    
        optimizer.step()          
         
        train_loss += float(lossvalue)   
        _,pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        train_acc += acc

    scheduler.step()
    print("echo:"+' ' + str(echo))
    print("loss:" + ' ' + str(train_loss / len(train_loader)))
    print("TrainACC:" + ' '+str(train_acc / len(train_loader)))

    eval_acc = 0
    model.eval()
    for i,(X,label) in enumerate(test_loader):
        X = Variable(X).cuda()
        label = Variable(label).cuda()
        testout = model(X)

        _, pred = testout.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        eval_acc += acc

    print('TestACC:' + ' ' + str(eval_acc/len(test_loader)))

    if echo >= 4 and echo != 0:
        torch.save(model.state_dict(), './resultNet/fnl_net_params_100_'+str(echo)+NAME+'.pkl')
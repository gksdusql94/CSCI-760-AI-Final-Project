#%%

from import_basics import *

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import random
import copy
from datetime import datetime

# %%
class Actor_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 12)
        self.fc1_1 = nn.Linear(12, output_dim)
        self.fc2 = nn.Linear(output_dim, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.relu1 = nn.ReLU()

    def forward(self, s):
        x1 = self.fc1(s)
        x1 = self.relu1(x1)
        x1 = self.fc1_1(x1)
        x1 = self.relu1(x1)
        x1 = self.fc2(x1)
        x1 = self.softmax(x1)
        return x1

class Critic_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 12)
        self.fc1_1 = nn.Linear(12, output_dim)
        self.fc2 = nn.Linear(output_dim, 1)
        self.relu1 = nn.ReLU()

    def forward(self, s):
        x1 = self.fc1(s)
        x1 = self.relu1(x1)
        x1 = self.fc1_1(x1)
        x1 = self.relu1(x1)
        x1 = self.fc2(x1)
        return x1
# %%
input_dim=1
actor = Actor_model(input_dim, 5).to(device)
critic = Critic_model(input_dim, 5).to(device)

LR = 3e-4
optim_actor = torch.optim.Adam(actor.parameters(), lr=LR)
optim_critic = torch.optim.Adam(critic.parameters(), lr=LR)

# %%
BUY=1
SELL=-BUY
epsilon=0.05
all_list = []
immediate_list = []
action_list = []
for i in range(100):

    # Forward pass through the actor model
    s=torch.randn(1).to(device)

    probs = actor(s)
    # v_hat = critic(s)
    # a = torch.distributions.Categorical(probs).sample()
        # Epsilon-greedy action selection
    if random.random() < epsilon:
        a = random.randint(0, 2)
    else:
        a = torch.distributions.Categorical(probs).sample().item()
        
    prob_old=probs[a].item()

    # print("prob_old: ",prob_old)

    


    if a==2: # sell
        # position-=
        immediate=SELL
    elif a==1: # buy
        # position+=1
        immediate=BUY
    else:
        immediate=0
    immediate
    Gt=0
    A=0
    ns=torch.randn(1).to(device)
    done=False
    action_list.append(a)
    immediate_list.append(immediate)
    v_hat=torch.zeros(1).to(device)

    all_list.append([i,s,ns,a,immediate,done,v_hat,prob_old,Gt,A])
    # print("immediate: ",immediate)

    s=copy.deepcopy(ns)
    # epsilon = max(final_epsilon, epsilon * decay_rate)

EPSILON=0.01
GAMMA=0.00
Gt_list=[]
a_list=[]
for i ,(index1,s,ns,a,immediate,done,v_hat,prob_old,Gt,_) in enumerate(all_list):
    Gt=0

    for ii,(index1,s2,ns2,a2,im2,done2,v_hat2,prob_old2,Gt2,_) in enumerate(all_list[i:]):
        # im2=(im2-min1)/(max1-min1+0.00001)
        Gt+=GAMMA**ii*im2
    Gt_list.append(Gt)
    a_list.append(a)
    all_list[i][-2]=Gt
    print("Gt: ",Gt,'A',A)
    Q=Gt
    V=v_hat
    
    A=Q-V
    all_list[i][-1]=A

    gmax1=max(Gt_list)
    # print("gmax1: ",gmax1)

    gmin1=min(Gt_list)


for index1, s, ns, a, immediate, done, v_hat, prob_old, Gt, A in all_list:
    # if index1==5:
        # break
# print("index1: ",index1)
        print("a: ",a)

        # s = s.detach()
        # A = A.detach()
        s = s.clone().detach()
        A = A.clone().detach()
        prob_old = torch.tensor(prob_old, dtype=torch.float32).to(device)

        probs = actor(s.to(device))
        v_hat = critic(s.to(device))
        # prob_old = torch.tensor(prob_old, dtype=torch.float32).to(device)
        # print("prob_old: ",prob_old)
        # print("a: ",a)
        print("probs: ",probs)


        s1 = (probs[a] / prob_old) * A
        s2 = torch.clamp((probs[a] / prob_old), 1 - EPSILON, 1 + EPSILON) * A
        # actor_loss = -torch.min(s1, s2)
        actor_loss=-torch.log(probs[a]) *Gt
        # print("actor_loss: ",actor_loss)

        # actor_loss = loss1
        critic_loss = (Gt - v_hat)**2
        # print("critic_loss: ",critic_loss)
        optim_actor.zero_grad()
        actor_loss.backward()
        # actor_loss.backward(retain_graph=False)

        # GRAD_CLIP = 100
        # nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)  # Gradient clipping for actor

        optim_actor.step()
        optim_critic.zero_grad()
        critic_loss.backward(retain_graph=False)
        # nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)  # Gradient clipping for critic

        optim_critic.step()
        # print(probs, "actor_loss: ", actor_loss, "critic_loss: ", critic_loss)

        # s = s.clone().detach()
        # A = A.clone().detach()

        # probs = actor(s.to(device))
        # v_hat = critic(s.to(device))
        # prob_old = torch.tensor(prob_old, dtype=torch.float32).to(device)
        # print("prob_old: ",prob_old)
        # print("probs: ",probs)

# %%

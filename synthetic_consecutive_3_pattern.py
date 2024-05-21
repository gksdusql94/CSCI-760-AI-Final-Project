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
import numpy as np
import pandas as pd
import torch

mu = 0.00  # Expected return
base_sigma = 0.0  # Base volatility
MAG = 0.1
oscillation_amplitude = MAG  # Amplitude of the oscillation
oscillation_frequency = MAG  # Frequency of the oscillation

def SDE_system(x, t):
    drift = mu  # Drift component models the expected return
    # Oscillating sigma
    sigma = base_sigma + oscillation_amplitude * np.sin(t)
    shock = sigma * np.random.normal()  # Shock component models volatility
    return drift + shock, sigma

T = 5000
dt = 0.015  # Time step size
x = torch.tensor([1.0])  # Initial stock price
stock_prices = []
times = np.linspace(0, T*dt, T)  # Generate time values
sigma_ = []

# Variables to track consecutive movements
consecutive_up = 0
consecutive_down = 0

for i, t in enumerate(times):
    dx, sigma = SDE_system(x, t)
    
    if len(stock_prices) > 0:
        if dx > 0:
            consecutive_up += 1
            consecutive_down = 0
        elif dx < 0:
            consecutive_down += 1
            consecutive_up = 0
        else:
            consecutive_up = 0
            consecutive_down = 0
        
        # Check for three consecutive down movements
        if consecutive_down >= 3:
            # x = x * 1.10  # Increase by 10%
            x=x+0.1
            consecutive_down = 0  # Reset the counter
        
        # Check for three consecutive up movements
        if consecutive_up >= 3:
            # x = x * 0.90  # Decrease by 10%
            x=x- 0.1
            consecutive_up = 0  # Reset the counter
    
    x = x + dt * dx  # Euler's method for SDE
    stock_prices.append(x.item())
    sigma_.append(sigma)

df2 = pd.DataFrame({'Close': stock_prices})

df2['Close'] = (df2['Close'] - df2['Close'].mean()) / df2['Close'].std()



# Plot the close prices
plt.figure(figsize=(10, 6))
plt.plot(df2['Close'])
plt.title('Close Prices Increasing Linearly from 0 to 1')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

# Define the Actor and Critic models
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


actor_loss_=[]
a_total=0
a_total2=0
CONSEC=3
BACKS = CONSEC
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate input dimensions
input_dim = (1 * BACKS) + 2

# Initialize models with the calculated input dimension
actor = Actor_model(input_dim, 5).to(device)
critic = Critic_model(input_dim, 5).to(device)
action_list_after_training=[]
action_list_after_training2=[]
pnl_after_training=[]
for name, param in actor.named_parameters():
    if param.requires_grad:
        print(name, param.data)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

actor.apply(init_weights)
critic.apply(init_weights)
for name, param in actor.named_parameters():
    if param.requires_grad:
        print(name, param.data)


LR = 3e-4
optim_actor = torch.optim.Adam(actor.parameters(), lr=LR)
optim_critic = torch.optim.Adam(critic.parameters(), lr=LR)

# Simulation parameters
dict1 = defaultdict(lambda: deque(maxlen=1000))
rand1 = random.randint(1, 1000)
timestamp_fixed = f'{datetime.now().timestamp()}_{rand1}'
print("timestamp_fixed: ", timestamp_fixed)

from torch.optim.lr_scheduler import LambdaLR
# Define the number of warm-up steps and total training steps
num_warmup_steps = 100
num_training_steps = 1000

# Create a lambda function to implement the warm-up
def lr_lambda(current_step: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return 1.0

# Create the learning rate scheduler
scheduler1 = LambdaLR(optim_actor, lr_lambda)
scheduler2 = LambdaLR(optim_critic, lr_lambda)
#%%
for iii in range(4):

    mu = 0.00  # Expected return
    base_sigma = 0.0  # Base volatility
    MAG = 0.10
    oscillation_amplitude = MAG  # Amplitude of the oscillation
    oscillation_frequency = MAG  # Frequency of the oscillation

    def SDE_system(x, t):
        drift = mu  # Drift component models the expected return
        # Oscillating sigma
        sigma = base_sigma + oscillation_amplitude * np.sin(t)
        shock = sigma * np.random.normal()  # Shock component models volatility
        return drift + shock, sigma

    T = 5000
    dt = 0.015  # Time step size
    x = torch.tensor([1.0])  # Initial stock price
    stock_prices = []
    times = np.linspace(0, T*dt, T)  # Generate time values
    sigma_ = []

    # Variables to track consecutive movements
    consecutive_up = 0
    consecutive_down = 0
    
    AMOUNT1=1
    for i, t in enumerate(times):
        dx, sigma = SDE_system(x, t)
        
        if len(stock_prices) > 0:
            if dx > 0:
                consecutive_up += 1
                consecutive_down = 0
            elif dx < 0:
                consecutive_down += 1
                consecutive_up = 0
            else:
                consecutive_up = 0
                consecutive_down = 0
            
            # Check for three consecutive down movements
            if consecutive_down >= CONSEC:
                # x = x * 1.10  # Increase by 10%
                x=x+AMOUNT1
                consecutive_down = 0  # Reset the counter
            
            # Check for three consecutive up movements
            if consecutive_up >= CONSEC:
                # x = x * 0.90  # Decrease by 10%
                x=x-AMOUNT1
                consecutive_up = 0  # Reset the counter
        
        x = x + dt * dx  # Euler's method for SDE
        stock_prices.append(x.item())
        sigma_.append(sigma)

    df2 = pd.DataFrame({'Close': stock_prices})

    df2['Close'] = (df2['Close'] - df2['Close'].mean()) / df2['Close'].std()


    # Plot the close prices
    plt.figure(figsize=(10, 6))
    plt.plot(df2['Close'])
    plt.title('Close Prices Increasing Linearly from 0 to 1')
    plt.xlabel('Time Steps')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.show()



    position = 0
    pnl = 0
    i = BACKS + 2
    i

    # index1 = random.randint(6000, len(df2) - 6000)
    # df1 = df2.iloc[index1 - 5000:index1]
    df1 = df2.reset_index(drop=True)
    df1


    stock_data = df1.iloc[i - BACKS:i].values.flatten().tolist()
    stock_data
    s = stock_data + [position] + [pnl]
    s = torch.tensor(s, dtype=torch.float32).to(device)
    s
    # epsilon = 0.05

    initial_epsilon = 0.1
    final_epsilon = 0.01
    decay_rate = 0.995
    epsilon = initial_epsilon

    done = False
    all_list = []
    immediate_list = []
    action_list = []

    cumulative_pnl = [0]

    while i < len(df1) - 1:
        position=0
        prev_pnl = copy.deepcopy(pnl)
        prev_position = copy.deepcopy(position)
        # print("prev_pnl: ",prev_pnl)
        # print("prev_position: ",prev_position)

        # Forward pass through the actor model
        probs = actor(s)
        v_hat = critic(s)
        # a = torch.distributions.Categorical(probs).sample()
            # Epsilon-greedy action selection
        if random.random() < epsilon:
            a = random.randint(0, 2)
        else:
            a = torch.distributions.Categorical(probs).sample().item()
            
        # print("probs: ",probs)
        # print("v_hat: ",v_hat)
        # print("a: ",a)


        prob_old=probs[a].item()

        # print("prob_old: ",prob_old)

        


        if a==2: # sell
            position-=1
        elif a==1: # buy
            position+=1
        else:
            position=0

        price_now=df1['Close'][i]
        price_now



        i+=1

        stock_data=df1.iloc[i-BACKS:i].values.flatten().tolist()
        price_next=df1['Close'][i]
        # print("price_next: ",price_next)

        stock_data = df1.iloc[i - BACKS:i].values.flatten().tolist()
        stock_data
        # ns = stock_data + [position] + [pnl]
        # s = torch.tensor(s, dtype=torch.float32).to(device)
        # s



        FEE=df2['Close'][i]*0.00

        current_pnl=position*(price_next-price_now-FEE)
        # print("current_pnl: ",current_pnl)

        # pnl=prev_pnl+current_pnl
        pnl=current_pnl
        # print(i,"pnl: ",pnl)

        ns=stock_data+[position]+[pnl]
        ns=torch.tensor(ns,dtype=torch.float32)
        ns

        probs2=actor(ns)
        nv_hat=critic(ns)
        # print("nv_hat: ",nv_hat)
        # print("probs2: ",probs2)


        ns.dtype
        print(a,pnl)
        immediate=pnl
        immediate
        Gt=0
        A=0


        action_list.append(a)
        immediate_list.append(immediate)
        all_list.append([i,s,ns,a,immediate,done,v_hat,prob_old,Gt,A])
        # print("immediate: ",immediate)

        s=copy.deepcopy(ns)
        epsilon = max(final_epsilon, epsilon * decay_rate)
    # print("s: ",s)


    max1=max(immediate_list)

    min1=min(immediate_list)
    a0=action_list.count(0)
    a1=action_list.count(1)
    a2=action_list.count(2)
    pnl=round(pnl,3)
    max1=round(max1,3)
    print("max1: ",max1)
    min1=round(min1,3)
    print("min1: ",min1)
    def print2(*args):
        print('\t'.join([str(arg) for arg in args]))
    # print2(_,pnl,max1,min1,a0,a1,a2,index1,BACKS,GAMMA,ALPHA)
    # timestamp1=datetime.now().timestamp()
    # dn=pd.DataFrame([timestamp1,_,pnl,max1,min1,a0,a1,a2,index1,BACKS,GAMMA,ALPHA,LR,mu,MAG,FEE]).T
    # dn.columns=['timestamp1','_','pnl','max1','min1','a0','a1','a2','index1','BACKS','GAMMA','ALPHA','LR','mu','MAG','FEE']
    # csv_update_insert_one('RL',f'{os.path.basename(__file__)}_{timestamp_fixed}',dn,no_duplicate_column='timestamp1')

    GAMMA=0.0




    EPSILON=0.01

    Gt_list=[]
    for i ,(index1,s,ns,a,immediate,done,v_hat,prob_old,Gt,_) in enumerate(all_list):
        Gt=0

        for ii,(index1,s2,ns2,a2,im2,done2,v_hat2,prob_old2,Gt2,_) in enumerate(all_list[i:]):
            # im2=(im2-min1)/(max1-min1+0.00001)
            Gt+=GAMMA**ii*im2
        Gt_list.append(Gt)
        all_list[i][-2]=Gt
        Q=Gt
        V=v_hat
        
        A=Q-V
        all_list[i][-1]=A

        gmax1=max(Gt_list)
        # print("gmax1: ",gmax1)

        gmin1=min(Gt_list)
        # print("gmin1: ",gmin1)



    for index1, s, ns, a, immediate, done, v_hat, prob_old, Gt, A in all_list[:-1]:
        position=0
        cumulative_pnl.append(cumulative_pnl[-1] + immediate)
        s=s.clone().detach()

        # prob_old=prob_old.detach()
        # A=A.detach()
        # Gt=Gt.detach()
        probs=actor(s.to(device))
        
        if torch.isnan(probs).any():
            break
        v_hat=critic(s.to(device))
        prob_old = torch.tensor(prob_old, dtype=torch.float32).to(device)  # Convert prob_old to tensor


        # for visualization
        a_trained_sampled = torch.distributions.Categorical(probs).sample()
        position2=0
        if a_trained_sampled==2:
            position2=-1    
        elif a_trained_sampled==1:
            position2=1
        a_trained = torch.argmax(probs)

        a_trained=a_trained.item()
        if a_trained==2: # sell
            position-=1
        elif a_trained==1: # buy
            position+=1
        price_now=df1['Close'][index1]
        stock_data=df1.iloc[index1-BACKS:index1].values.flatten().tolist()
        price_next=df1['Close'][index1+1]
        FEE=df2['Close'][index1]*0.00
        current_pnl=position*(price_next-price_now-FEE)
        pnl=current_pnl
        pnl_after_training.append(pnl)
        a_total+=position
        a_total2+=position2
        action_list_after_training.append(a_total)
        action_list_after_training2.append(a_total2)
        # action_list_after_training.append(a_trained)
        Q=Gt
        V=v_hat
        
        A=Q-V
        #$%%
        prob_old=1

        s1= (probs[a]/prob_old)*A
        # print("s1: ",s1)

        s2= torch.clamp((probs[a]/prob_old),1- EPSILON, 1+EPSILON) * A
        # print("s2: ",s2)
        # nv_hat=critic(ns.to(device))
        loss1=torch.min(s1,s2)
        actor_loss= -loss1
        # actor_loss=-torch.log(probs[a]) *Gt
        # print("loss1: ",loss1)
        # actor_loss+= loss1
        # critic_loss+=(Gt-v_hat)**2

        
        critic_loss=(Gt-v_hat)**2
        # print("Gt: ",Gt)
        optim_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_loss_.append(actor_loss.item())
        GRAD_CLIP=1
        # nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)  # Gradient clipping for actor

        optim_actor.step()
        scheduler1.step()

        optim_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)  # Gradient clipping for critic

        optim_critic.step()
        scheduler2.step()
        print(probs,"actor_loss: ",actor_loss,"critic_loss: ",critic_loss)


#%%


# Plot the close prices
plt.figure(figsize=(10, 6))
plt.plot(df2['Close'])
plt.title('Close Prices Increasing Linearly from 0 to 1')
plt.xlabel(f'Time Steps, mu={mu} , sd = {MAG}')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()
#%%
plt.figure(figsize=(10, 6))
plt.plot(cumulative_pnl)
plt.title('Cumulative PnL Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative PnL')
plt.grid(True)
plt.show()

#%%

import matplotlib.pyplot as plt

plt.plot(action_list_after_training)
plt.xlabel('training steps')  
plt.ylabel('Buy and Sell action projection')  
plt.title('Projection of Buy or Sell action')  
plt.show()
# %%
import matplotlib.pyplot as plt

plt.plot(action_list_after_training2)
plt.xlabel('training steps')  
plt.ylabel('Buy and Sell action projection')  
plt.title('Projection of Buy or Sell action')  
plt.show()

#%%
cumulative_pnl = np.cumsum(pnl_after_training)

# Calculate Sharpe Ratio
returns = np.diff(cumulative_pnl)
sharpe_ratio = np.mean(returns) / np.std(returns)

# Action Distribution
action_counts = np.bincount(action_list)

print(f"Final PnL: {cumulative_pnl[-1]:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Action Counts: {action_counts}")
# Cumulative PnL Plot
plt.figure(figsize=(14, 7))
plt.plot(cumulative_pnl, label='Cumulative PnL')
plt.xlabel('Time Step')
plt.ylabel('Cumulative PnL')
plt.title('Cumulative PnL Over Time')
plt.legend()
plt.show()

# Price with Actions
plt.figure(figsize=(14, 7))
plt.plot(df1['Close'].values, label='Price')
buy_signals = np.where(np.array(action_list) == 1)[0]
sell_signals = np.where(np.array(action_list) == 2)[0]
plt.scatter(buy_signals, df1['Close'].values[buy_signals], marker='^', color='g', label='Buy', alpha=0.6)
plt.scatter(sell_signals, df1['Close'].values[sell_signals], marker='v', color='r', label='Sell', alpha=0.6)
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.title('Price with Buy/Sell Actions')
plt.legend()
plt.show()

# Action Distribution
plt.figure(figsize=(7, 5))
plt.bar(['Hold', 'Buy', 'Sell'], action_counts)
plt.xlabel('Action')
plt.ylabel('Count')
plt.title('Action Distribution')
plt.show()

#%%
actions = [a for _,_, _, a, _, _, _, _, _, _ in all_list]

# Plot actions
plt.figure(figsize=(10, 6))
plt.hist(actions, bins=range(4), align='left', rwidth=0.8)
plt.title('Action Distribution')
plt.xlabel('Action (0: Hold, 1: Buy, 2: Sell)')
plt.ylabel('Frequency')
plt.xticks([0, 1, 2])
plt.grid(True)
plt.show()
#%%
immediate_rewards = [immediate for _,_, _, _, immediate, _, _, _, _, _ in all_list]

# Plot immediate rewards
plt.figure(figsize=(10, 6))
plt.plot(immediate_rewards)
plt.title('Immediate Rewards Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Immediate Reward')
plt.grid(True)
plt.show()

# Track cumulative rewards
cumulative_rewards = np.cumsum(immediate_rewards)

# Plot cumulative rewards
plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards)
plt.title('Cumulative Rewards Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.grid(True)
plt.show()
# %%
import numpy as np

# Calculate percentage of profitable trades
num_profitable_trades = sum([1 for _,_, _, _, immediate, _, _, _, _, _ in all_list if immediate > 0])
total_trades = len(all_list)
percentage_profitable_trades = num_profitable_trades / total_trades * 100
print(f'Percentage of Profitable Trades: {percentage_profitable_trades:.2f}%')

# Calculate Sharpe Ratio
returns = np.array(immediate_rewards)
mean_return = np.mean(returns)
std_return = np.std(returns)
sharpe_ratio = mean_return / (std_return + 1e-8)
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

# Calculate Maximum Drawdown
cumulative_rewards = np.array(cumulative_rewards)
rolling_max = np.maximum.accumulate(cumulative_rewards)
drawdown = (rolling_max - cumulative_rewards) / (rolling_max + 1e-8)
max_drawdown = np.max(drawdown)
print(f'Maximum Drawdown: {max_drawdown:.2f}')

# %%
def calculate_action_ratios(actions, window_size=100):
    buy_ratios = []
    sell_ratios = []
    hold_ratios = []
    for i in range(0, len(actions), window_size):
        window_actions = actions[i:i+window_size]
        buy_ratio = window_actions.count(1) / len(window_actions)
        sell_ratio = window_actions.count(2) / len(window_actions)
        hold_ratio = window_actions.count(0) / len(window_actions)
        buy_ratios.append(buy_ratio)
        sell_ratios.append(sell_ratio)
        hold_ratios.append(hold_ratio)
    return buy_ratios, sell_ratios, hold_ratios

# Calculate ratios
window_size = 100  # Adjust this based on your preference
buy_ratios, sell_ratios, hold_ratios = calculate_action_ratios(action_list_after_training, window_size)

# Plot the ratios
epochs = range(len(buy_ratios))

plt.figure(figsize=(12, 6))
plt.plot(epochs, buy_ratios, label='Buy Ratio', color='blue')
plt.plot(epochs, sell_ratios, label='Sell Ratio', color='red')
plt.plot(epochs, hold_ratios, label='Hold Ratio', color='green')
plt.xlabel('Epochs')
plt.ylabel('Action Ratio')
plt.title('Action Ratios Over Time')
plt.legend()
plt.grid(True)
plt.show()
# %%
a_trained
# %%

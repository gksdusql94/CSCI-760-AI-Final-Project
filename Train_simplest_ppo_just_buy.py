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

# Create a DataFrame with a close price increasing linearly from 0 to 1
steps = 5000
close_prices = np.linspace(1, 0, steps)
# close_prices = np.linspace(0, 1, steps)
df2 = pd.DataFrame({'Close': close_prices})
df2['Close'] = (df2['Close'] - df2['Close'].mean()) / df2['Close'].std()

# Plot the close prices
plt.figure(figsize=(10, 6))
plt.plot(df2['Close'])
plt.title('Close Prices Increasing Linearly from 0 to 1')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

#%%
steps = 50000
close_prices = np.linspace(0, steps/10, steps)
df = pd.DataFrame({'Close': close_prices})
# df['SMA50'] = df['Close'].rolling(window=50).mean()
# df['SMA200'] = df['Close'].rolling(window=200).mean()
# df['SMA1000'] = df['Close'].rolling(window=1000).mean()
df.dropna(inplace=True)  # Drop NaN values
df = df.reset_index(drop=True)
df
df['Close'] = (df['Close'] - df['Close'].mean()) / df['Close'].std()
# %%

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

# %%
actor_loss_=[]

#%%
BACKS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate input dimensions
input_dim = (len(df.columns) * BACKS) + 2

# Initialize models with the calculated input dimension
actor = Actor_model(input_dim, 5).to(device)
critic = Critic_model(input_dim, 5).to(device)
action_list_after_training=[]
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



for iii in range(2):
    position = 0
    pnl = 0
    i = BACKS + 2
    i

    index1 = random.randint(6000, len(df) - 6000)
    df1 = df.iloc[index1 - 5000:index1]
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



        FEE=df['Close'][i]*0.00

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
        # a_trained = torch.distributions.Categorical(probs).sample()
        
        a_trained = torch.argmax(probs)

        a_trained=a_trained.item()
        if a_trained==2: # sell
            position-=1
        elif a_trained==1: # buy
            position+=1
        price_now=df1['Close'][index1]
        stock_data=df1.iloc[index1-BACKS:index1].values.flatten().tolist()
        price_next=df1['Close'][index1+1]
        FEE=df['Close'][index1]*0.00
        current_pnl=position*(price_next-price_now-FEE)
        pnl=current_pnl
        pnl_after_training.append(pnl)
        action_list_after_training.append(a_trained)
        Q=Gt
        V=v_hat
        
        A=Q-V
        s1= (probs[a])*A
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
        optim_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)  # Gradient clipping for critic

        optim_critic.step()
        print(probs,"actor_loss: ",actor_loss,"critic_loss: ",critic_loss)
        
#%%
(price_next-price_now-FEE)
#%%

plt.figure(figsize=(10, 6))
plt.plot(cumulative_pnl)
plt.title('Cumulative PnL Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative PnL')
plt.grid(True)
plt.show()

#%%
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

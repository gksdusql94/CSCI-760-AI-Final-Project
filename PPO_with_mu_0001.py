#%%
from import_basics import *

import numpy as np
import torch
import matplotlib.pyplot as plt
'''
python3 a7_1_w*

'''
# Parameters
mu = 0.003  # Expected return
base_sigma = 0.0  # Base volatility
MAG=0.09
oscillation_amplitude = MAG  # Amplitude of the oscillation
oscillation_frequency = MAG  # Frequency of the oscillation

def SDE_system(x, t):
    drift = mu  # Drift component models the expected return
    # Oscillating sigma
    sigma = base_sigma + oscillation_amplitude * np.sin(t)
    shock = sigma * np.random.normal()  # Shock component models volatility
    return drift + shock,sigma

T = 51000-1
dt = 0.015  # Time step size
x = torch.tensor([1.0])  # Initial stock price
stock_prices = []
times = np.linspace(0, T*dt, T)  # Generate time values
sigma_=[]
for i, t in enumerate(times):
    dx,sigma = SDE_system(x, t)
    x = x + dt * dx  # Euler's method for SDE
    stock_prices.append(x.item())
    sigma_.append(sigma)

df = pd.DataFrame({'Time': times, 'Close': stock_prices})

# Compute SMAs
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()
df['SMA1000'] = df['Close'].rolling(window=1000).mean()

# Display the DataFrame with the computed SMAs
# import ace_tools as tools; tools.display_dataframe_to_user(name="Stock Prices with SMAs", dataframe=df)

# Optionally, you can plot the stock prices along with the SMAs to visualize them
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(df['Time'], df['Close'], label='Stock Price')
plt.plot(df['Time'], df['SMA50'], label='SMA 50')
plt.plot(df['Time'], df['SMA200'], label='SMA 200')
plt.plot(df['Time'], df['SMA1000'], label='SMA 1000')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price with Simple Moving Averages')
plt.legend()
plt.show()
#%%

# Drop rows with NaN values
df.dropna(inplace=True)
#%%
df=reset_index(df)
df=df[['Close','SMA50','SMA200','SMA1000']]
df
# %%

class Actor_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor_model, self).__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc1_1 = nn.Linear(12,output_dim)
        self.fc2 = nn.Linear(output_dim, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.relu1 = nn.ReLU()
    def forward(self, s):
        
        x1 = self.fc1(s)
        x1=self.relu1(x1)
        x1 = self.fc1_1(x1)
        x1=self.relu1(x1)
        x1= self.fc2(x1)
        x1=self.softmax(x1)
        
        return x1
    
class Critic_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic_model, self).__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc1_1 = nn.Linear(12,output_dim)
        self.fc2 = nn.Linear(output_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.relu1 = nn.ReLU()

    def forward(self, s):
        
        x1 = self.fc1(s)
        x1=self.relu1(x1)

        x1 = self.fc1_1(x1)
        x1=self.relu1(x1)

        x1= self.fc2(x1)
        x1=self.softmax(x1)
        
        return x1
#%%
BACKS=15
device='cpu'
GAMMA=0.999
ALPHA=0.002
EPSILON=0.2
i=BACKS+2
position=0
pnl=0

# stock_data=df1.iloc[i-BACKS:i].values.flatten().tolist()
# s=stock_data+[position]+[pnl]
# s=torch.tensor(s).to(device)
# s

# columns_to_normalize
# actor = Actor_model((2+len(columns_to_normalize))*BACKS+2, 5).to(device)
# critic = Critic_model((2+len(columns_to_normalize))*BACKS+2, 5).to(device)
dict1 = defaultdict(lambda: deque(maxlen=1000))
LR=3e-5

# Calculate input dimensions
input_dim = (len(df.columns) * BACKS) + 2

# Initialize models with the calculated input dimension
actor = Actor_model(input_dim, 5).to(device)
critic = Critic_model(input_dim, 5).to(device)

# Initialize weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

actor.apply(init_weights)
critic.apply(init_weights)

# Optimizers
LR = 3e-5
optim_actor = torch.optim.Adam(actor.parameters(), lr=LR)
optim_critic = torch.optim.Adam(critic.parameters(), lr=LR)

# Simulation parameters
dict1 = defaultdict(lambda: deque(maxlen=1000))
rand1 = random.randint(1, 1000)
timestamp_fixed = f'{datetime.now().timestamp()}_{rand1}'
print("timestamp_fixed: ", timestamp_fixed)

# Example loop to use the models
for _ in range(500):
    i = BACKS + 2
    position = 0
    pnl = 0

    index1 = random.randint(6000, len(df) - 6000)
    df1 = df.iloc[index1 - 5000:index1]
    df1 = df1.reset_index(drop=True)

    stock_data = df1.iloc[i - BACKS:i].values.flatten().tolist()
    s = stock_data + [position] + [pnl]
    s = torch.tensor(s, dtype=torch.float32).to(device)

    done = False
    all_list = []
    immediate_list = []
    action_list = []

    while i < len(df1) - 1:
        prev_pnl = copy.deepcopy(pnl)
        prev_position = copy.deepcopy(position)

        # Forward pass through the actor model
        probs = actor(s)
        v_hat = critic(s)
        a = torch.distributions.Categorical(probs).sample()

        prob_old=probs[a].item()
        a=a.item()


        if a==2: # sell
            position-=1
        elif a==1: # buy
            position+=1

        price_now=df['Close'][i]
        price_now

        

        i+=1

        stock_data=df1.iloc[i-BACKS:i].values.flatten().tolist()
        price_next=df1['Close'][i]
        price_next



        FEE=df1['Close'][i]*0.00

        current_pnl=position*(price_next-price_now-FEE)
        current_pnl

        pnl=prev_pnl+current_pnl
        # print(i,"pnl: ",pnl)

        ns=stock_data+[position]+[pnl]
        ns=torch.tensor(ns,dtype=torch.float32)
        ns

        probs2=actor(ns)
        nv_hat=critic(ns)
        # print("nv_hat: ",nv_hat)


        ns.dtype

        immediate=pnl
        immediate
        Gt=0
        A=0


        action_list.append(a)
        immediate_list.append(immediate)
        all_list.append([s,ns,a,immediate,done,v_hat,prob_old,Gt,A])


        s=copy.deepcopy(ns)
    

    max1=max(immediate_list)

    min1=min(immediate_list)
    a0=action_list.count(0)
    a1=action_list.count(1)
    a2=action_list.count(2)
    pnl=round(pnl,3)
    max1=round(max1,3)
    min1=round(min1,3)
    def print2(*args):
        print('\t'.join([str(arg) for arg in args]))
    print2(_,pnl,max1,min1,a0,a1,a2,index1,BACKS,GAMMA,ALPHA)
    timestamp1=datetime.now().timestamp()
    dn=pd.DataFrame([timestamp1,_,pnl,max1,min1,a0,a1,a2,index1,BACKS,GAMMA,ALPHA,LR,mu,MAG,FEE]).T
    dn.columns=['timestamp1','_','pnl','max1','min1','a0','a1','a2','index1','BACKS','GAMMA','ALPHA','LR','mu','MAG','FEE']
    csv_update_insert_one('RL',f'{os.path.basename(__file__)}_{timestamp_fixed}',dn,no_duplicate_column='timestamp1')



    Gt_list=[]
    for i ,(s,ns,a,immediate,done,v_hat,prob_old,Gt,_) in enumerate(all_list):
        Gt=0

        for ii,(s2,ns2,a2,im2,done2,v_hat2,prob_old2,Gt2,_) in enumerate(all_list[i:]):
            im2=(im2-min1)/(max1-min1)
            Gt+=GAMMA**ii*im2
        Gt_list.append(Gt)
        all_list[i][-2]=Gt
        Q=Gt
        V=v_hat
        
        A=Q-V
        all_list[i][-1]=A

    all_list


    gmax1=max(Gt_list)


    gmin1=min(Gt_list)

    actor_loss=0
    critic_loss=0
    for i ,(s,ns,a,immediate,done,v_hat,prob_old,Gt,A) in enumerate(all_list):
        

        probs=actor(s.to(device))
        v_hat=critic(s.to(device))

        s1= (probs[a]/prob_old)*A
        

        s2= torch.clamp((probs[a]/prob_old),1- EPSILON, 1+EPSILON) * A

        # nv_hat=critic(ns.to(device))
        actor_loss-= torch.min(s1,s2)
        critic_loss+=(Gt-v_hat)**2
        
        # loss=actor_loss+critic_loss
        # Compute the gradients


    actor_loss=actor_loss/len(all_list)
    critic_loss=critic_loss/len(all_list)
    # loss=actor_loss+critic_loss
    # loss.backward(retain_graph=True)
    optim_actor.zero_grad()
    actor_loss.backward()

    # optim_actor.zero_grad()
    # loss.backward()

    # # Perform a single optimization step
    optim_actor.step()

    # v_hat=

    # critic_loss=(Gt-v_hat)**2
    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()

# %%
'''

python3 a7_start*


'''
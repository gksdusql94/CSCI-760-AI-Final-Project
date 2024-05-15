#%%
from datetime import datetime,timedelta
import sys,os,copy,ast,socket,random,math,webbrowser,getpass,time,shutil
import numpy as np
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt

import torch,os,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# %%
import getpass,sys,socket

#from import_basics import *

df = pd.read_csv('data/DeepPrime_dataset_final_Feat8.csv')
df = df.reset_index()
max_edited_length = 10
max_length = len(df['WT74_On'][0].lower()) + max_edited_length

# %%
# for enu1,item in enumerate(df['type_del']):
#     if item==2:
#         break
for enu1 in range(len(df)):
    column1 = 'type_sub'
    edit_number = 2
    if int(df[column1][enu1]) == int(1) and int(df['Edit_len'][enu1]) == int(edit_number):
        print("df[column1][enu1]: ", df[column1][enu1])
        break
print("enu1: ", enu1)
#%%
def LtN(DNA):
    DNA = DNA.lower()
    mapping = {'a': 0, 'c': 1, 't': 2, 'g': 3, 'x': 4, 'd': 4}
    encoded = torch.zeros((len(DNA), 5), dtype=torch.long)
    for i, nucleotide in enumerate(DNA):
        encoded[i, mapping[nucleotide]] = 1
    return encoded

def NtS(del_target):
    string1 = ''
    for item in del_target.squeeze():
        string1 = string1 + str(item.item())
    return string1

#%%
i = 18935
i = 270164
total_x = []
total_y = []

from tqdm import tqdm
#%%

checkpoint = torch.load('data/data_checkpoint.pth')
#%%
x = checkpoint['x']
y = checkpoint['y']

#%%
x.shape
#%%
# 
y.shape
# #%%
#%%
#%%
#%%
#%%
#%%

from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#%%
# Revised SequenceEmbedding
class SequenceEmbedding(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super(SequenceEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
    def forward(self, x):
        # For each row, find all active indices and sum their embeddings
        # as you can see it is using like looping each row with one 1's and embed, and then add each one of them together i think?
        embeddings = []
        for row in x:
            active_indices = row.nonzero(as_tuple=True)[0]
            embedded = self.embedding(active_indices)
            embeddings.append(embedded.sum(dim=0))
            
        return torch.stack(embeddings)

# Embedding for scalar data
class ScalarEmbedding(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(ScalarEmbedding, self).__init__()
        self.fc = nn.Linear(num_features, embedding_dim)
        
    def forward(self, x):
        return self.fc(x)

class Feedforward(nn.Module):
    def __init__(self,n_emb):
        super().__init__()
        self.feedforward=nn.Sequential(

            nn.Linear(n_emb,4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb,n_emb), # this is projection layer
            nn.Dropout(DROPOUT)
        )

    def forward(self,x):
        x=self.feedforward(x)
        return x
# y_hat=model(x4)
class Head(nn.Module):
    def __init__(self,heads):
        super().__init__()    
        # self.linear1=nn.Linear(LENGTH*CHARN,out1*2,bias=False)
        # self.linear2=nn.Linear(out1*2,out1,bias=False)
        self.key_linear=nn.Linear(N_EMB,heads,bias=False)
        self.query_linear=nn.Linear(N_EMB,heads,bias=False)
        self.value_linear=nn.Linear(N_EMB,heads,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(LENGTH,LENGTH)))
        self.dropout=nn.Dropout(DROPOUT)
    def forward(self,x):
        B,T,C=x.shape
        xk=self.key_linear(x) # B,T,H
        # print("\n>> xk.shape= ", xk.shape)

        xq=self.query_linear(x) # B,T,H
        xv=self.value_linear(x) # B,T,H
        
        WW=xq @ xk.transpose(-2,-1) * C**-0.5
        WW=WW.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        WW=F.softmax(WW,dim=-1) # B,T,T
        # print("\n>> WW.shape= ", WW.shape)
        WW=self.dropout(WW)
        y_hat=WW@xv # (B,T,T @ B,T,H) > B,T,H
        # print("\n>> y_hat.shape= ", y_hat.shape)

        return y_hat
class Multihead(nn.Module):
    def __init__(self,headn,heads):
        super().__init__()
        self.head_list=nn.ModuleList([Head(heads) for _ in range(headn)])
        self.projection=nn.Linear(headn*heads,N_EMB)
        self.dropout=nn.Dropout(DROPOUT)
    def forward(self,x):
        x=torch.cat([head.forward(x) for head in self.head_list],dim=-1)
        x=self.projection(x)
        y_hat=self.dropout(x)
        return y_hat

class Block(nn.Module):
    def __init__(self,n_emb,headn):
        super().__init__()
        headsize=n_emb//headn
        self.multihead=Multihead(HEADN,headsize)
        self.feedforward=Feedforward(n_emb)
        self.layer_norm1=nn.LayerNorm(n_emb)
        self.layer_norm2=nn.LayerNorm(n_emb)

    def forward(self,x):
        # print('hi')
        # print(x)

        x= x + self.multihead(self.layer_norm1(x)) ### B,T,C
        x= x + self.feedforward(self.layer_norm2(x))
        return x
# %%

# Sample data
# batch_size = 10
# sequence_length = 15
# num_categories = 10
# num_features = 5
# embedding_dim = 8  # 
# Create embedding models
# but i'm not sure if this is the best method to do this.
# sequence_embedding_model = SequenceEmbedding(num_categories, embedding_dim)
# scalar_embedding_model = ScalarEmbedding(num_features, embedding_dim)

# Get embeddings
# embedded_sequence = sequence_embedding_model(sequence_data)
# embedded_scalar = scalar_embedding_model(scalar_data)
# 
# embedded_sequence, embedded_scalar
#%%

#%%
# x=copy.deepcopy(result)
# print("\n>> x.shape= ", x.shape)
# print("\n>> y_true.shape= ", y_true.shape)

# #%%
# x.shape
# #%%
# #%%
# embedding = nn.Embedding(15, N_EMB)

# print("\n>> x1.shape= ", x1.shape)

# #%%
# EMB2=nn.Embedding(LENGTH,N_EMB)
# EMB2

# x2=EMB2(torch.arange(LENGTH).to(DEVICE))
# x2
# #%%
# print("\n>> x2.shape= ", x2.shape)

# # %%
# x=x1+x2
# %%

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.EMB=nn.Embedding(CHARN, N_EMB)
        self.EMB2=nn.Embedding(LENGTH,N_EMB)
        
        self.blocks=nn.Sequential(
            *[Block(N_EMB,HEADN) for _ in range(BLOCKN)]
        )
        
        self.layer_norm=nn.LayerNorm(N_EMB)
        # self.language_modeling_head=nn.Linear(N_EMB,CHARN)
        self.final_layer=nn.Linear(LENGTH*N_EMB,1)
        # self.linear1=nn.Linear(LENGTH*CHARN,out1*2,bias=False)
        # self.linear2=nn.Linear(out1*2,out1,bias=False)
    def embed1(self,x):

        total_embed=[]
        for col_row in x:
            embeddings = []
            # print("\n>> col_row.shape= ", col_row.shape)
            # break

            for row in col_row:
                active_indices = row.nonzero(as_tuple=True)[0]
                active_indices.shape

                embedded = self.EMB(active_indices)
                embedded

                emb_sum=embedded.sum(dim=0)
                emb_sum

                embeddings.append(emb_sum)
            embeddings=torch.stack(embeddings)
            total_embed.append(embeddings)

        x1= torch.stack(total_embed)
        return x1
    def forward(self,x):
        # B,T=x.shape
        x ### B,T
        # x1=self.EMB(x) ### B,T,C
        x1=self.embed1(x)
        x2=self.EMB2(torch.arange(LENGTH).to(DEVICE)) ### T,C
        x=x1+x2 ### B,T,C
        x=self.blocks(x)
        x=self.layer_norm(x)
        # print("\n>> x.shape= ", x.shape)
        # time.sleep(5)
        x = x.view(x.size(0), -1)

        x=self.final_layer(x)

        return x
# %%
# p1 seed
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)


# p2 make dataloader
batch_size = 64
# Assuming x and y are your data tensors
dataset = MyDataset(x, y)


# Define the split sizes. For example, 80% train and 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# p3 make model

max_edited_length = 10
max_length = max_edited_length + 74
node1 = 3
N_EMB = int(30 * node1)
LENGTH = max_length
DEVICE = 'cpu'
CHARN = 15
BLOCKN = int(6 * 1)
HEADS = int(30 * node1)
HEADN = int(6 * 1)
DROPOUT = 0.2
model = Model().to(DEVICE)
EPOCHS = 1000
LR = 1e-4
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)


# p4 train
loss_ = []
for epoch in range(EPOCHS):
    for i, data in enumerate(train_loader, 0):
        x, y_true = data
        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)
        y_true = y_true.unsqueeze(1)
        y_true = y_true.float()
        y_hat = model(x)

        criterion_mse = nn.MSELoss()
        loss = criterion_mse(y_hat, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_.append(float(loss))
        
    print(f'epoch: {epoch}, loss: {loss.item()}')

# %%
loss
# %%
y_hat.dtype
# %%
y_true.dtype
# %%

# p5 test
total_mse = 0
count = 0
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        x, y_true = data
        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)
        y_true = y_true.unsqueeze(1)
        y_true = y_true.float()
        y_hat = model(x)
        mse = ((y_hat - y_true) ** 2).mean().item()
        total_mse += mse * len(x)
        count += len(x)

average_mse = total_mse / count
print(f"Test MSE: {average_mse}")

# %%
torch.save(model, f'a2_model_v1_{round(average_mse, 4)}.pt')
# %%

# (MSE, Pearson, Spearman 계산 및 그래프 그리기)
mse_values = [mse]  
pearson_corr = np.corrcoef(y.cpu().numpy(), y_hat.cpu().detach().numpy())[0, 1]
spearman_corr = spearmanr(y.cpu().numpy(), y_hat.cpu().detach().numpy())[0]

# MSE 그래프
plt.figure(figsize=(10, 5))
plt.plot(loss_, label='Training Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# Pearson, Spearman 그래프
plt.figure(figsize=(10, 5))
plt.bar(['Pearson', 'Spearman'], [pearson_corr, spearman_corr])
plt.ylabel('Correlation')
plt.title('Pearson and Spearman Correlation')
plt.show()

# 히트맵
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import random
import torch 
from torch import nn
import torch.nn.functional as func
from tqdm.notebook import tqdm

def relu2(X): return func.relu(X)**2
def tanh(X): return func.tanh(X)

class FCNN(nn.Module):
    def __init__(self,input_dim=2,output_dim=1,num_hidden=2,hidden_dim=10,act=func.tanh,transform=None):
        super().__init__()
         
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers  = nn.ModuleList([nn.Linear(input_dim,hidden_dim)])
        for _ in range(num_hidden-1): self.layers.append(nn.Linear(hidden_dim,hidden_dim))
        self.act     = act
        self.out     = nn.Linear(hidden_dim,output_dim)
        self.transform = transform
    def forward(self,X):
        if self.transform is not None: X = self.transform(X)
        for layer in self.layers: X = self.act(layer(X))
        Y = self.out(X)
        return Y
class SDE_model(nn.Module):
    def __init__(self,dim,model_V,model_g,unit_len=int(5e3)):
        super().__init__()
        self.dim      = dim
        self.model_V  = model_V
        self.model_g  = model_g
        self.unit_len = unit_len
        self.mu       = nn.Parameter(torch.tensor([0.]*dim).cuda(),requires_grad=False)
        self.sigma    = nn.Parameter(torch.tensor([1.]*dim).cuda(),requires_grad=False)
        self.coef_V   = nn.Parameter(torch.tensor(1.).cuda(),requires_grad=False)
        self.coef_g   = nn.Parameter(torch.tensor(1.).cuda(),requires_grad=False)
    def get_V_harmonic(self,X): return torch.sum(X**2,axis=-1)
    def get_V_dV(self,X): 
        I = int(np.ceil(len(X)/self.unit_len))
        V,dV = [],[]
        for i in range(I):
            X_sub = X[i*self.unit_len:(i+1)*self.unit_len]
            if not torch.is_tensor(X_sub): X_sub = torch.tensor(X_sub,requires_grad=True).cuda()
            X_    = (X_sub-self.mu)/self.sigma
            V_    = self.coef_V*(self.model_V(X_).view(-1) + self.get_V_harmonic(X_))
            dV_   = torch.autograd.grad(V_,X_sub,torch.ones_like(V_),create_graph=True)[0] # it is X_sub!!!
            V.append(V_)
            dV.append(dV_)
        V = torch.hstack(V)
        dV = torch.vstack(dV)
        return V,dV
    def get_V_np(self,X): 
        V,_ = self.get_V_dV(X);
        return V.cpu().data.numpy()
    def get_g(self,X): 
        I = int(np.ceil(len(X)/self.unit_len))
        g = []
        for i in range(I):
            X_sub = X[i*self.unit_len:(i+1)*self.unit_len]
            if not torch.is_tensor(X_sub): X_sub = torch.tensor(X_sub,requires_grad=True).cuda()
            X_    = (X_sub-self.mu)/self.sigma
            g_    = self.coef_g*self.model_g(X_)
            g.append(g_)
        g = torch.vstack(g)
        return g
    def get_g_np(self,X): 
        g = self.get_g(X);
        return g.cpu().data.numpy()
class Solver():
    def __init__(self,model):
        self.model=model
    def train_model(self,data_train,data_test,get_loss,optimizer,
                    n_steps,batch_size,scheduler=None,n_show_loss=100,error_model=None,use_tqdm=True):
        if use_tqdm: step_range = tqdm(range(n_steps))
        else: step_range = range(n_steps)
        loss_step = []
        for i_step in step_range:
            if i_step%n_show_loss==0:
                loss_train,loss_test = get_loss(self.model,data_train)[:-1],\
                                       get_loss(self.model,data_test)[:-1]
                
                def show_num(x): 
                    if abs(x)<100 and abs(x)>.01: return '%0.5f'%x
                    else: return '%0.2e'%x
                item1 = '%2dk'%np.int_(i_step/1000)
                item2 = 'Loss: '+' '.join([show_num(k) for k in loss_train])
                item3 = ' '.join([show_num(k) for k in loss_test])
                item4 = ''
                if error_model is not None:
                    item4 = 'E(QP): %0.4f' % (error_model(self.model))
                print(', '.join([item1,item2,item3,item4]))
                loss_step = loss_step + [i_step] + [k.cpu().data.numpy() for k in loss_train]\
                                                 + [k.cpu().data.numpy() for k in loss_train]
            data_batch = [k[random.sample(range(len(k)),min(batch_size,len(k)))] for k in data_train]
            loss = get_loss(self.model,data_batch)[-1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
        if error_model is not None: 
            print("Error: %0.5f" % (error_model(self.model)))
        return loss_step
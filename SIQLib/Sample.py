import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random

def Euler(h0,f,dt,num_steps):
    h = h0
    for step in range(num_steps): h  = h+f(h)*dt
    return h
def RK2(h0,f,dt,num_steps):
    h = h0
    for step in range(num_steps):
        K1 = f(h)*dt
        K2 = f(h+K1/2)*dt
        h  = h+K2
    return h
def RK4(h0,f,dt,num_steps):
    h = h0
    for step in range(num_steps):
        K1 = f(h)*dt
        K2 = f(h+K1/2)*dt
        K3 = f(h+K2/2)*dt
        K4 = f(h+K3)*dt
        h  = h+1./6*(K1+2*K2+2*K3+K4)
    return h
def perform_ode(x0,f,T,dt,m,integrator,mode,num_steps=1,use_tqdm=True):
    if use_tqdm: i_range=tqdm(range(round(T/dt)))
    else: i_range=range(round(T/dt))
    if mode=="X0":
        x  = np.array(x0)
        X0 = []  
        for i in i_range: 
            if i%m==0: X0.append(x)
            x = integrator(x,f,dt/num_steps,num_steps)
        if np.size(x0.shape)==1: return np.hstack(X0).reshape(round(T/dt/m),x0.shape[-1])
        else: return np.hstack(X0).reshape(-1,round(T/dt/m),x0.shape[-1])
    if mode=="X0X1":
        x  = np.array(x0)
        X0X1 = []  
        for i in i_range: 
            if i%m==0: X0X1.append(x)
            if i%m==1: X0X1.append(x)
            x = integrator(x,f,dt/num_steps,num_steps)
        if np.size(x0.shape)==1: return np.hstack(X0X1).reshape(round(T/dt/m),2,x0.shape[-1])
        else: return np.hstack(X0X1).reshape(-1,round(T/dt/m),2,x0.shape[-1])

def split_train_test(X0X1,N,dim,split_prop=.7): 
    N1 = round(len(X0X1)*split_prop)
    perm_index = np.random.permutation(len(X0X1))
    return X0X1[perm_index[:N1],:,0,:].reshape(-1,dim),X0X1[perm_index[:N1],:,1,:].reshape(-1,dim),\
           X0X1[perm_index[N1:],:,0,:].reshape(-1,dim),X0X1[perm_index[N1:],:,1,:].reshape(-1,dim)
    
    X          = np.reshape(X,(N,-1,dim))
    X          = X[np.random.permutation(X.shape[0])]
    train_size  = np.int_(X.shape[0]*split_prop)
    return X[:train_size].reshape(-1,dim),X[train_size:].reshape(-1,dim)
def split_train_valid_test(X0X1,N,dim,split_prop=[.7,.2]): 
    X0X1      = np.reshape(X0X1,(N,-1,2,dim))
    X0X1      = X0X1[np.random.permutation(X0X1.shape[0])]
    train_idx =             np.int_(X0X1.shape[0]*split_prop[0])
    valid_idx = train_idx + np.int_(X0X1.shape[0]*split_prop[1])
    return X0X1[         :train_idx,:,0,:].reshape(-1,dim),X0X1[         :train_idx,:,1,:].reshape(-1,dim),\
           X0X1[train_idx:valid_idx,:,0,:].reshape(-1,dim),X0X1[train_idx:valid_idx,:,1,:].reshape(-1,dim),\
           X0X1[valid_idx:         ,:,0,:].reshape(-1,dim),X0X1[valid_idx:         ,:,1,:].reshape(-1,dim),

def get_Xhat(X,tol,dim):
    X        = X.reshape(-1,dim)
    Xhat     = []
    Xsize0   = X.shape[0] 
    cc       = 0
    while X.shape[0]>0:
        idx  = np.random.randint(X.shape[0])
        Xhat.append(X[idx]+0.)
        mask = (np.abs(X[idx,0]-X[:,0])>tol) | (np.abs(X[idx,-1]-X[:,-1])>tol)
        X1,X2 = X[mask],X[~mask]
        X3   = X2[np.linalg.norm(X[idx]-X2,axis=-1)>tol]
        X    = np.vstack([X1,X3])
    return np.array(Xhat)
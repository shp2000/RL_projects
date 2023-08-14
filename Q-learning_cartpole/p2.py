import gym
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
def rollout(e, q, eps=0, T=200):
    traj = []

    #x = np.ndarray(e.reset())
    x = e.reset()[0]
    #print(x)
    for t in range(T):
        u = q.control(th.from_numpy(x).float().unsqueeze(0),eps=eps)
        #u = u.int().numpy().squeeze()
        #print(e.step(u))
        xp,r,d,info,xxx = e.step(u)
        #print(r)
        t = dict(x=x,xp=xp,r=r,u=u,d=d,info=info)
        x = xp
        traj.append(t)
        if d:
            break
    return traj

class q_t(nn.Module):
    def __init__(s, xdim, udim, hdim=16):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
                            nn.Linear(xdim, hdim),
                            nn.ReLU(True),
                            nn.Linear(hdim, udim),
                            )
    def forward(s, x):
        return s.m(x)

    def control(s, x, eps=0):
        # 1. get q values for all controls
        q = s.m(x)

        ### TODO: XXXXXXXXXXXX
        # eps-greedy strategy to choose control input
        # note that for eps=0
        # you should return the correct control u
        if np.random.random() < 1 - eps:
            u = q.argmax().item()
        else:
            u = np.random.randint(0, s.udim)
        
        return u

def loss(q, ds, q_target):
    ### TODO: XXXXXXXXXXXX
    # 1. sample mini-batch from datset ds
    # 2. code up dqn with double-q trick
    # 3. return the objective f
    #batch_size = 64
    
    #l = len(ds)
    i = 0
    xp = []
    x =[]
    u = []
    r = []
    while True:
        if i > 63:
            break

        i0 = np.random.randint(0, len(ds) - 1)
        
        i1 = np.random.randint(0, len(ds[i0]) - 1)
        dict = ds[i0][i1]
        if dict['d']:
            continue
        x.append(list(dict['x']))
        xp.append(list(dict['xp']))
        r.append(dict['r'])
        u.append(dict['u'])
        i = i + 1
    
    xp = th.from_numpy(np.array(xp)).float()
    x = th.from_numpy(np.array(x)).float()
    r = th.from_numpy(np.array(r)).float().view(64, 1)
    u = th.from_numpy(np.array(u).astype('int64')).view(64, 1)
    pred = q(x).gather(1, u)
    q_action=th.argmax(q(xp).detach(),dim=1).reshape(-1,1)
    double_q=q_target(xp).detach()
    double_q=double_q.gather(1,q_action)
    target = r + 0.99 * double_q.max(1)[0].view(64, 1)
    loss = nn.MSELoss()
    f = loss(pred, target)
    return f

# u* = argmax q(x', u)
# (q(x, u) - r - g*(1-indicator of terminal)*qc(x' , u*))**2

def evaluate(q, e):
    ### TODO: XXXXXXXXXXXX
    # 1. create a new environment e
    # 2. run the learnt q network for 100 trajectories on
    # this new environment to take control actions. Remember that
    # you should not perform epsilon-greedy exploration in the evaluation
    # phase
    # and report the average discounted
    # return of these 100 trajectories
    #e=gym.make('CartPole-v1')
    returns = []
    
    
    for _ in range(100):
        x=e.reset()[0]
        tr_ret = 0
        #tr_ret = 0
        done = False
        while not done:
            u = q.control(th.from_numpy(x).float().unsqueeze(0),eps=0)
            # u = u.int().numpy().squeeze()
            #re = e.reset()[0]
            #u = q(th.from_numpy(re).float().unsqueeze(0))
            #action  =th.argmax(u)
            xp, r, done, info,xx= e.step(u)
            tr_ret += r
            #dis *=0.99
            x=xp
            # if False:
            #     break
        returns.append(tr_ret)
    return(np.mean(returns))
        #t = dict(x=x, xp=xp, r=r, u=u, d=d, info=info)

        # x = xp
        # traj.append(t)
        # if d==True:
        #     print(len(traj))
        #     break
    #return r

if __name__=='__main__':
    e = gym.make('CartPole-v1')

    xdim, udim =    e.observation_space.shape[0], \
                    e.action_space.n

    q = q_t(xdim, udim, 8)
    # Adam is a variant of SGD and essentially works in the
    # same way
    optim = th.optim.Adam(q.parameters(), lr=1e-3,
                          weight_decay=1e-4)

    ds = []
    q_target = q_t(xdim, udim, 8)
    # collect few random trajectories with
    # eps=1
    l_list=[]
    train_returns = []
    eval_returns = []
    train_env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')
    for i in range(1000):
        ds.append(rollout(train_env, q, eps=1, T=200))
    eps=0.9
    for i in range(10000):
        #eps = max(0.1, eps - 0.0001)
        q.train()
        t = rollout(train_env, q, eps=eps)
        ds.append(t)

        # perform weights updates on the q network
        # need to call zero grad on q function
        # to clear the gradient buffer
        

        if i % 1000 == 0:
        # Evaluate policy on training environment
            #train_return= rollout(train_env, q, eps=1, T=200)
            train_return = np.mean([np.sum([t['r'] for t in traj]) for traj in ds[-10:]])
            train_returns.append(train_return)
            # Evaluate policy on evaluation environment
            eval_return = evaluate(q, eval_env)
            eval_returns.append(eval_return)
            #print(train_return)
            #print(eval_return)
            print(f'Weight update {i}, Train return: {train_return:.2f}, Eval return: {eval_return:.2f}')
        
        q.zero_grad()
        f = loss(q,ds,q_target)
        f.backward()
        optim.step()

# Plot the data
fig, ax = plt.subplots()
ax.plot(range(0, 10000, 1000), train_returns, label='Train')
ax.plot(range(0, 10000, 1000), eval_returns, label='Eval')
ax.set_xlabel('Weight updates')
ax.set_ylabel('Average return')
ax.set_title('Training and evaluation returns')
ax.legend()
plt.show()

        # exponential averaging for the target
        #print(ds)
    #     l_list.append(f.item())
    # plt.plot(l_list)
    # plt.show()

from .utils import *
import torch.utils as tutils
import numpy as np
from copy import deepcopy

class LinearDataset(tutils.data.Dataset):

    def __init__(self, n_tasks=100):
        self.n_tasks_ = n_tasks

        # build base 

        n, d, s0, graph_type, sem_type = 10, 10, 20, 'ER', 'gauss'
        self.B_true_ = ut.simulate_dag(d, s0, graph_type)
        self.W_true_ = ut.simulate_parameter(self.B_true_)

        #self.Xs_ = ut.simulate_linear_sem(W_true_, n, sem_type)

        self.data = [(self.W_true_, ut.simulate_linear_sem(self.W_true_, n , sem_type)), ]

        for i in range(n_tasks):
            W_perturb = self.perturb_(self.W_true_)
            X = ut.simulate_linear_sem(W_perturb, n, sem_type)
            self.data.append((W_perturb, X))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def split(self, ratio=0.8):
        n_train = int(len(self) * ratio)
        n_test = len(self) - n_train
        train = deepcopy(self)
        test = deepcopy(self)
        
        train.data = train.data[:n_train]
        test.data = test.data[n_train:]

        return train, test

    def perturb_(self, W):
        
        # select an edge
        Wnew = self.perturb_del_edge_(W)
        Wnew = self.perturb_weight_(Wnew)
        assert(ut.is_dag(Wnew))
        return Wnew

    def perturb_del_edge_(self, W):
        Wnew = W.copy()
        idxs = np.where( Wnew != 0)
        n_to_delete = np.random.randint(1, len(idxs[0])//2)
        to_delete = np.random.choice(len(idxs[0]), size=n_to_delete)
        for j in to_delete:
            Wnew[idxs[0][j]][idxs[1][j]] = 0
        return Wnew

    def perturb_weight_(self, W):
        Wnew = W.copy()
        idxs = np.where( Wnew != 0)
        n_to_change = np.random.randint(1, len(idxs[0])//2)
        to_change = np.random.choice(len(idxs[0]), size=n_to_change)
        for j in to_change:
            Wnew[idxs[0][j]][idxs[1][j]] = np.random.uniform(low = W.min(), high= W.max())
        return Wnew

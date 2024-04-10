from .utils import simulate_dag, simulate_nonlinear_sem, simulate_nonlinear_sem, is_dag
import torch.utils as tutils
import numpy as np
from copy import deepcopy

class NonlinearDataset(tutils.data.Dataset):

    def __init__(self, n_tasks=100):
        self.n_tasks_ = n_tasks

        # build base 

        n, d, s0, graph_type, sem_type = 10, 30, 60, 'ER', 'mim'
        self.B_true_ = simulate_dag(d, s0, graph_type)

        #self.Xs_ = ut.simulate_linear_sem(W_true_, n, sem_type)

        self.data = [(self.B_true_, simulate_nonlinear_sem(self.B_true_, n , sem_type)), ]

        for i in range(n_tasks):
            B_perturb = self.perturb_(self.B_true_)
            X = simulate_nonlinear_sem(B_perturb, n, sem_type)
            self.data.append((B_perturb, X))
        
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

    def perturb_(self, B):
        
        # select an edge
        Bnew = self.perturb_del_edge_(B)
        assert(is_dag(Bnew))
        return Bnew

    def perturb_del_edge_(self, B):
        Bnew = B.copy()
        idxs = np.where( Bnew != 0)
        n_to_delete = np.random.randint(1, len(idxs[0])//2)
        to_delete = np.random.choice(len(idxs[0]), size=n_to_delete)
        for j in to_delete:
            Bnew[idxs[0][j]][idxs[1][j]] = 0
        return Bnew


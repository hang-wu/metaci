from .base import Base
from .lbfgsb_scipy import LBFGSBScipy
from .locally_connected import LocallyConnected
import torch
import numpy as np
from .notears_nonlinear_utils import NotearsMLP, NotearsSobolev, squared_loss, dual_ascent_step
from copy import deepcopy

class MetaNotearsNonlinear(Base):

    def __init__(self, name='MetaNotearsNonlinear', model_type='MLP', meta=True):
        super().__init__(name+'-'+model_type)
        self.model_type_ = model_type
        self.meta_ = meta

    def solve_X(self, X, model, max_iter=100, lambda1=0.1, lambda2=0.1, rho_max=1e+9, h_tol=1e-5, w_threshold=0.3, *args, **kwargs):
        rho, alpha, h = 1.0, 0.0, np.inf
        n, d = X.shape
        #self.model = NotearsMLP(dims=[d, 10, 1], bias=True).cuda() if self.model_type_ == 'MLP' else NotearsSobolev(dims=[d, 10, 1], bias=True).cuda()
        for i in range(max_iter):
            print(i, end=',')
            rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max)
            if h <= h_tol or rho >= rho_max:
                break

        return model

    def solve(self, train, test, max_iter=1, lambda1=0.1, lambda2=0.1, rho_max=1e+6, h_tol=1e-3, w_threshold=0.3, *args, **kwargs):
        d = train[0][1].shape[1]
        self.model0 = NotearsMLP(dims=[d, 10, 1], bias=True).cuda() if self.model_type_ == 'MLP' else NotearsSobolev(dims=[d, 10, 1], bias=True).cuda()
        model = self.model0
        ans = []
        full_test_res = []

        for W, X in train:
            weights_before = deepcopy(model.state_dict())
            model = self.solve_X(X, model, max_iter=max_iter)
            weights_after = model.state_dict()
            W_est = model.fc1_to_adj()
            W_est[np.abs(W_est) < w_threshold] = 0
            ans.append(self.count_accuracy(W!=0, W_est!=0)['shd'])
            #if self.meta_:
            outerstepsize = 0.15 if self.meta_ else 0. # linear schedule
            model.load_state_dict({name : 
                weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
                for name in weights_before})

        print('Train Done')

        for W, X in test:
            weights_before = deepcopy(model.state_dict())
            model = self.solve_X(X, model, max_iter=max_iter)
            weights_after = model.state_dict()
            W_est = model.fc1_to_adj()
            W_est[np.abs(W_est) < w_threshold] = 0
            full_test_res.append(self.count_accuracy(W!=0, W_est!=0))
            ans.append(full_test_res[-1]['shd'])

            outerstepsize = 0.15 if self.meta_ else 0. # linear schedule
            model.load_state_dict({name : 
                weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
                for name in weights_before})




        print('Test Done')   
        self.model0 = model
        return np.mean(ans), full_test_res     


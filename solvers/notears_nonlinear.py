from .base import Base
from .lbfgsb_scipy import LBFGSBScipy
from .locally_connected import LocallyConnected
import torch
import numpy as np
from .notears_nonlinear_utils import NotearsMLP, NotearsSobolev, squared_loss, dual_ascent_step


class NotearsNonlinear(Base):

    def __init__(self, name='NotearsNonlinear', model_type='MLP'):
        super().__init__(name+'-'+model_type)
        self.model_type_ = model_type

    def solve_X(self, X, max_iter=1, lambda1=0.1, lambda2=0.1, rho_max=1e+6, h_tol=1e-3, w_threshold=0.3, *args, **kwargs):
        rho, alpha, h = 1.0, 0.0, np.inf
        n, d = X.shape
        self.model = NotearsMLP(dims=[d, 10, 1], bias=True).cuda() if self.model_type_ == 'MLP' else NotearsSobolev(dims=[d, 10, 1], bias=True).cuda()
        for _ in range(max_iter):
            rho, alpha, h = dual_ascent_step(self.model, X, lambda1, lambda2, rho, alpha, h, rho_max)
            if h <= h_tol or rho >= rho_max:
                break
        W_est = self.model.fc1_to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est

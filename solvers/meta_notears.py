from .base import Base

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from scipy.linalg import norm

from .notears import notears_linear

class MetaNotears(Base):

    def __init__(self, name='MetaNotears', meta=True):
        super().__init__(name)
        self.meta_ = meta

    def solve(self, train, test, w_threshold=0.3, *args, **kwargs):
        
        d = train[0][1].shape[1]

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        self.w0_ = np.zeros(2 * d * d)

        ans = []
        full_test_res = []

        for W, X in train:
            w_est = notears_linear(X, lambda1=0.1, loss_type='l2', w_est=self.w0_, w_threshold=w_threshold, *args, **kwargs)
            if self.meta_:
                self.w0_ = self.w0_ + 0.3 * (w_est - self.w0_)
            #print(self.w0_[:10])

            W_est = _adj(w_est)
            W_est[np.abs(W_est) < w_threshold] = 0

            ans.append(self.count_accuracy(W!=0, W_est!=0)['shd'])
        print('Train Done')

        for W, X in test:
            w_est = notears_linear(X, lambda1=0.1, loss_type='l2', w_est=self.w0_, w_threshold=w_threshold, *args, **kwargs)
            if self.meta_:
                self.w0_ = self.w0_ + 0.3 * (w_est - self.w0_)

            W_est = _adj(w_est)
            W_est[np.abs(W_est) < w_threshold] = 0
            full_test_res.append(self.count_accuracy(W!=0, W_est!=0))
            ans.append(full_test_res[-1]['shd'])
        print('Test Done')   

        return np.mean(ans)
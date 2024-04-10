import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random


class Base(object):

    def __init__(self, name):
        self.name_ = name

    def true_ll(self, X, W, loss_type='l2'):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss      

    def count_accuracy(self, B_true, B_est):
        """Compute various accuracy metrics for B_est.

        true positive = predicted association exists in condition in correct direction
        reverse = predicted association exists in condition in opposite direction
        false positive = predicted association does not exist in condition

        Args:
            B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
            B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

        Returns:
            fdr: (reverse + false positive) / prediction positive
            tpr: (true positive) / condition positive
            fpr: (reverse + false positive) / condition negative
            shd: undirected extra + undirected missing + reverse
            nnz: prediction positive
        """
        if (B_est == -1).any():  # cpdag
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:  # dag
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            if not self.is_dag(B_est):
                #raise ValueError('B_est should be a DAG')
                #print('B_est should be a DAG')
                pass
        d = B_true.shape[0]
        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est == -1)
        pred = np.flatnonzero(B_est == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)
        return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

    def is_dag(self, W):
        G = ig.Graph.Weighted_Adjacency(W.tolist())
        return G.is_dag()

    def solve_X(self, X, *args, **kwargs):
        raise NotImplementedError

    def solve(self, train, test, *args, **kwargs):
        ans = []

        for W, X in train:
            W_est = self.solve_X(X, *args, **kwargs)
            ans.append(self.count_accuracy(W!=0, W_est!=0)['shd'])
        print('Train Done')
        
        for W, X in test:
            W_est = self.solve_X(X, *args, **kwargs)
            ans.append(self.count_accuracy(W!=0, W_est!=0)['shd'])
        print('Test Done')

        return np.mean(ans)
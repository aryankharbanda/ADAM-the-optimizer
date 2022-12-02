import torch
from torch.optim.optimizer import Optimizer, required
import copy
import math

class Momentum(Optimizer):
    # This class implements the well-known ADAM optimizer
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        if lr < 0 or beta1 < 0 or beta1 < 0 or eps < 0:
            raise ValueError("Invalid params: (lr, beta1, beta2 and epsilon are all supposed to be >=0). Given values: {}, {}, {}".format(lr, beta1, beta2, eps))
        defaults = dict(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        super(Momentum, self).__init__(params, defaults)


    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:

            beta1 = group['beta1']
            lr = group['lr']


            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Momentum does not support sparse gradients, consider using SparseADAM instead')

                state = self.state[p] 


                if len(state) == 0: 

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p.data) 
                    state['exp_avg_sq'] = torch.zeros_like(p.data) 

                state['step'] += 1

                exp_avg = state['exp_avg']


                exp_avg.mul_(beta1).add_(other=grad, alpha=1-beta1)
                step = state['step']
                bias_correction1 = 1 - (beta1**step)
                step_size = lr/bias_correction1
                p.data.addcdiv_(exp_avg, 1, value=-1*step_size)

        return loss
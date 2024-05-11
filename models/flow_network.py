import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import itertools


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, hidden_dim=1024):
        super(MLP, self).__init__()
        model = []
        model += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers):
            model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class BaseFlow(nn.Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1 - delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
log = lambda x: torch.log(x * 1e2) - np.log(1e2)
logit = lambda x: log(x) - log(1 - x)


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size()) > 2 else sum1(x)

class Sigmoid(nn.Module):
    def forward(self, x):
        return sigmoid(x)

class SigmoidFlow(BaseFlow):

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        self.num_params = 3 * num_ds_dim
        self.act_a = lambda x: softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=2)

    def forward(self, x, dsparams, mollify=0.0, delta=delta):
        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim:3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        return xnew.squeeze(-1)

class DenseSigmoidFlow(BaseFlow):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.act_a = lambda x: softplus_(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=3)
        self.act_u = lambda x: softmax(x, dim=3)

        self.u_ = Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = Parameter(torch.Tensor(out_dim, hidden_dim))
        self.num_params = 3 * hidden_dim + in_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x, dsparams, logdet=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        inv = np.log(np.exp(1 - delta) - 1)
        ndim = self.hidden_dim
        pre_u = self.u_[None, None, :, :] + dsparams[:, :, -self.in_dim:][:, :, None, :]
        pre_w = self.w_[None, None, :, :] + dsparams[:, :, 2 * ndim:3 * ndim][:, :, None, :]
        a = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, :, None, :], 3) + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=3)

        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        logj = F.log_softmax(pre_w, dim=3) + \
            logsigmoid(pre_sigm[:,:,None,:]) + \
            logsigmoid(-pre_sigm[:,:,None,:]) + log(a[:,:,None,:])
        # n, d, d2, dh
        
        logj = logj[:,:,:,:,None] + F.log_softmax(pre_u, dim=3)[:,:,None,:,:]
        # n, d, d2, dh, d1
        
        logj = log_sum_exp(logj,3).sum(3)
        # n, d, d2, d1
        
        logdet_ = logj + np.log(1-delta) - \
            (log(x_pre_clipped) + log(-x_pre_clipped+1))[:,:,:,None]
        
        if logdet is None:
            logdet = logdet_.new_zeros(logdet_.shape[0], logdet_.shape[1], 1, 1)
        
        logdet = log_sum_exp(
            logdet_[:,:,:,:,None] + logdet[:,:,None,:,:], 3
        ).sum(3)
        # n, d, d2, d1, d0 -> n, d, d2, d0
        
        return xnew.squeeze(-1), logdet
        

    def extra_repr(self):
        return 'input_dim={in_dim}, output_dim={out_dim}'.format(**self.__dict__)

class DDSF(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, out_dim):
        super(DDSF, self).__init__()
        blocks = [DenseSigmoidFlow(in_dim, hidden_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            blocks += [DenseSigmoidFlow(hidden_dim, hidden_dim, hidden_dim)]
        blocks += [DenseSigmoidFlow(hidden_dim, hidden_dim, out_dim)]
        self.num_params = 0
        for block in blocks:
            self.num_params += block.num_params
        self.model = nn.ModuleList(blocks)

    def forward(self, x, dsparams):
        start = 0
        _logdet = None

        for block in self.model:
            block_dsparams = dsparams[:, :, start:start + block.num_params]
            x, _logdet = block(x, block_dsparams, logdet=_logdet)
            start = start + block.num_params

        logdet = _logdet[:,:,0,0].sum(1)

        return x, logdet


def oper(array,oper,axis=-1,keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper

def log_sum_exp(A, axis=-1, sum_op=torch.sum):    
    maximum = lambda x: x.max(axis)[0]    
    A_max = oper(A,maximum, axis, True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max    
    return B

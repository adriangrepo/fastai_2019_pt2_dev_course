from pathlib import Path
from IPython.core.debugger import set_trace
import operator
import random
from collections import OrderedDict
from functools import partial
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import re
from typing import *
import matplotlib.pyplot as plt
#pip install git+https://github.com/NVIDIA/apex
import apex.fp16_utils as fp16

from torch import tensor
from torch.nn import init
from torch import nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

import PIL,os,mimetypes
import math
import numpy as np

torch.set_num_threads(2)

TEST = 'test'
MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b):
    test(a,b,operator.eq,'==')

def near(a,b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)

def test_near(a,b):
    test(a,b,near)

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s):
    return (x-m)/s

def test_near_zero(a,tol=1e-3):
    assert a.abs()<tol, f"Near zero: {a}"

def mse(output, targ):
    return (output.squeeze(-1) - targ).pow(2).mean()

def accuracy(out, yb):
    return (torch.argmax(out, dim=1)==yb).float().mean()

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def listify(o):
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]

def get_model(data, lr=0.5, nh=50):
    m = data.train_ds.x.shape[1]
    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,data.c))
    return model, optim.SGD(model.parameters(), lr=lr)

def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))



# 05_anneal

def get_model_func(lr=0.5):
    return partial(get_model, lr=lr)

def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos):
    return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer
def sched_no(start, end, pos):
    return start

@annealer
def sched_exp(start, end, pos):
    return start * (end/start) ** pos

#This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

def pg_dicts(pgs):
    return [{'params':o} for o in pgs]

def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]

#06_cuda_cnn_hooks_init

def normalize_to(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

def view_tfm(*size):
    def _inner(x):
        return x.view(*((-1,)+size))
    return _inner

def children(m):
    return list(m.children())

def flatten(x):
    return x.view(x.shape[0], -1)

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'):
        hook.stats = ([],[])
    means,stds = hook.stats
    if mod.training:
        means.append(outp.data.mean())
        stds .append(outp.data.std())

def init_cnn(m, uniform=False):
    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
    for l in m:
        if isinstance(l, nn.Sequential):
            f(l[0].weight, a=0.1)
            l[0].bias.data.zero_()

#nb_07a
def get_batch(dl, run):
    run.xb,run.yb = next(iter(dl))
    for cb in run.cbs: cb.set_runner(run)
    run('begin_batch')
    return run.xb,run.yb

def find_modules(m, cond):
    if cond(m): return [m]
    return sum([find_modules(o,cond) for o in m.children()], [])

def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return isinstance(l, lin_layers)

def append_stat(hook, mod, inp, outp):
    d = outp.data
    hook.mean,hook.std = d.mean().item(),d.std().item()

### end of generic functions


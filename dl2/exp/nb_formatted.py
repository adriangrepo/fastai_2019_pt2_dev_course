#formatted nb_all.py using black
#https://github.com/python/black

######################### nb_00.py


TEST = "test"
######################### nb_01.py


import operator


def test(a, b, cmp, cname=None):
    if cname is None:
        cname = cmp.__name__
    assert cmp(a, b), f"{cname}:\n{a}\n{b}"


def test_eq(a, b):
    test(a, b, operator.eq, "==")


from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"


def near(a, b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)


def test_near(a, b):
    test(a, b, near)


######################### nb_02.py


def get_data():
    path = datasets.download_data(MNIST_URL, ext=".gz")
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    return map(tensor, (x_train, y_train, x_valid, y_valid))


def normalize(x, m, s):
    return (x - m) / s


def test_near_zero(a, tol=1e-3):
    assert a.abs() < tol, f"Near zero: {a}"


from torch.nn import init


def mse(output, targ):
    return (output.squeeze(-1) - targ).pow(2).mean()


from torch import nn

######################### nb_03.py


import torch.nn.functional as F


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


from torch import optim


class Dataset:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


def get_dls(train_ds, valid_ds, bs, **kwargs):
    #NB shuffle=True removed, need to manually add to kwargs if want shuffle
    return (
        DataLoader(train_ds, batch_size=bs, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, **kwargs),
    )


######################### nb_04.py


class DataBunch:
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl, self.valid_dl, self.c = train_dl, valid_dl, c

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


def get_model(data, lr=0.5, nh=50):
    m = data.train_ds.x.shape[1]
    model = nn.Sequential(nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, data.c))
    return model, optim.SGD(model.parameters(), lr=lr)


class Learner:
    def __init__(self, model, opt, loss_func, data):
        self.model, self.opt, self.loss_func, self.data = model, opt, loss_func, data


import re

_camel_re1 = re.compile("(.)([A-Z][a-z]+)")
_camel_re2 = re.compile("([a-z0-9])([A-Z])")


def camel2snake(name):
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()


class Callback:
    _order = 0

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs = 0.0
        self.run.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1.0 / self.iters
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


from typing import *


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


class Runner:
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop, self.cbs = False, [TrainEvalCallback()] + cbs

    @property
    def opt(self):
        return self.learn.opt

    @property
    def model(self):
        return self.learn.model

    @property
    def loss_func(self):
        return self.learn.loss_func

    @property
    def data(self):
        return self.learn.data

    def one_batch(self, xb, yb):
        self.xb, self.yb = xb, yb
        if self("begin_batch"):
            return
        self.pred = self.model(self.xb)
        if self("after_pred"):
            return
        self.loss = self.loss_func(self.pred, self.yb)
        if self("after_loss") or not self.in_train:
            return
        self.loss.backward()
        if self("after_backward"):
            return
        self.opt.step()
        if self("after_step"):
            return
        self.opt.zero_grad()

    def all_batches(self, dl):
        self.iters = len(dl)
        for xb, yb in dl:
            if self.stop:
                break
            self.one_batch(xb, yb)
            self("after_batch")
        self.stop = False

    def fit(self, epochs, learn):
        self.epochs, self.learn, self.loss = epochs, learn, tensor(0.0)

        try:
            for cb in self.cbs:
                cb.set_runner(self)
            if self("begin_fit"):
                return
            for epoch in range(epochs):
                self.epoch = epoch
                if not self("begin_epoch"):
                    self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self("begin_validate"):
                        self.all_batches(self.data.valid_dl)
                if self("after_epoch"):
                    break

        finally:
            self("after_fit")
            self.learn = None

    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f():
                return True
        return False


class AvgStats:
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0.0, 0
        self.tot_mets = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = (
            AvgStats(metrics, True),
            AvgStats(metrics, False),
        )

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


from functools import partial

######################### nb_05.py


def create_learner(model_func, loss_func, data):
    return Learner(*model_func(data), loss_func, data)


def get_model_func(lr=0.5):
    return partial(get_model, lr=lr)


def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start, end, pos):
    return start + pos * (end - start)


@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start, end, pos):
    return start


@annealer
def sched_exp(start, end, pos):
    return start * (end / start) ** pos


# This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.0
    pcts = tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner


class Recorder(Callback):
    def begin_fit(self):
        print('>>Recorder.begin_fit()')
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []
        self.val_losses = []

    def after_batch(self):
        if not self.in_train:
            #validation
            self.val_losses.append(self.loss.detach().cpu())
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self, pgid=-1):
        plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0):
        plt.plot(self.losses[: len(self.losses) - skip_last], color='b', label="train")
        plt.plot(self.val_losses[: len(self.val_losses) - skip_last], color='r', label="valid")
        plt.legend()


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.n_epochs / self.epochs)

    def begin_batch(self):
        if self.in_train:
            self.set_param()


def pg_dicts(pgs):
    return [{"params": o} for o in pgs]


def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]


######################### nb_05b.py


class Callback:
    _order = 0

    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs = 0.0
        self.run.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1.0 / self.iters
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class Runner:
    def __init__(self, cbs=None, cb_funcs=None):
        self.in_train = False
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop, self.cbs = False, [TrainEvalCallback()] + cbs

    @property
    def opt(self):
        return self.learn.opt

    @property
    def model(self):
        return self.learn.model

    @property
    def loss_func(self):
        return self.learn.loss_func

    @property
    def data(self):
        return self.learn.data

    def one_batch(self, xb, yb):
        try:
            self.xb, self.yb = xb, yb
            self("begin_batch")
            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb)
            self("after_loss")
            if not self.in_train:
                return
            self.loss.backward()
            self("after_backward")
            self.opt.step()
            self("after_step")
            self.opt.zero_grad()
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in dl:
                self.one_batch(xb, yb)
        except CancelEpochException:
            self("after_cancel_epoch")

    def fit(self, epochs, learn):
        self.epochs, self.learn, self.loss = epochs, learn, tensor(0.0)

        try:
            for cb in self.cbs:
                cb.set_runner(self)
            self("begin_fit")
            for epoch in range(epochs):
                self.epoch = epoch
                if not self("begin_epoch"):
                    self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self("begin_validate"):
                        self.all_batches(self.data.valid_dl)
                self("after_epoch")

        except CancelTrainException:
            self("after_cancel_train")
        finally:
            self("after_fit")
            self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = (
            AvgStats(metrics, True),
            AvgStats(metrics, False),
        )

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self, pgid=-1):
        plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0):
        plt.plot(self.losses[: len(self.losses) - skip_last])

    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.plot(lrs[:n], losses[:n])


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.n_epochs / self.epochs)

    def begin_batch(self):
        if self.in_train:
            self.set_param()


class LR_Find(Callback):
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.n_iter / self.max_iter
        lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss * 10:
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss


######################### nb_06.py


torch.set_num_threads(2)


def normalize_to(train, valid):
    m, s = train.mean(), train.std()
    return normalize(train, m, s), normalize(valid, m, s)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def flatten(x):
    return x.view(x.shape[0], -1)


class CudaCallback(Callback):
    def begin_fit(self):
        self.model.cuda()

    def begin_batch(self):
        self.run.xb, self.run.yb = self.xb.cuda(), self.yb.cuda()


class BatchTransformXCallback(Callback):
    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.run.xb = self.tfm(self.xb)


def view_tfm(*size):
    def _inner(x):
        return x.view(*((-1,) + size))

    return _inner


def get_runner(model, data, lr=0.6, cbs=None, opt_func=None, loss_func=F.cross_entropy):
    if opt_func is None:
        opt_func = optim.SGD
    opt = opt_func(model.parameters(), lr=lr)
    learn = Learner(model, opt, loss_func, data)
    return learn, Runner(cb_funcs=listify(cbs))


def children(m):
    return list(m.children())


class Hook:
    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, "stats"):
        hook.stats = ([], [])
    means, stds = hook.stats
    if mod.training:
        means.append(outp.data.mean())
        stds.append(outp.data.std())


class ListContainer:
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        try:
            return self.items[idx]
        except TypeError:
            if isinstance(idx[0], bool):
                assert len(idx) == len(self)  # bool mask
                return [o for m, o in zip(idx, self.items) if m]
            return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del self.items[i]

    def __repr__(self):
        res = f"{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            res = res[:-1] + "...]"
        return res


from torch.nn import init


class Hooks(ListContainer):
    def __init__(self, ms, f):
        super().__init__([Hook(m, f) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


def get_cnn_layers(data, nfs, layer, **kwargs):
    nfs = [1] + nfs
    return [
        layer(nfs[i], nfs[i + 1], 5 if i == 0 else 3, **kwargs)
        for i in range(len(nfs) - 1)
    ] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c)]


def conv_layer(ni, nf, ks=3, stride=2, **kwargs):
    return nn.Sequential(
        nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride), GeneralRelu(**kwargs)
    )


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


def init_cnn(m, uniform=False):
    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
    for l in m:
        if isinstance(l, nn.Sequential):
            f(l[0].weight, a=0.1)
            l[0].bias.data.zero_()


def get_cnn_model(data, nfs, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))


def get_learn_run(
    nfs, data, lr, layer, cbs=None, opt_func=None, uniform=False, **kwargs
):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model, uniform=uniform)
    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)


from IPython.display import display, Javascript


def nb_auto_export():
    display(
        Javascript(
            """{
const ip = IPython.notebook
if (ip) {
    ip.save_notebook()
    console.log('a')
    const s = `!python notebook2script.py ${ip.notebook_name}`
    if (ip.kernel) { ip.kernel.execute(s) }
}
}"""
        )
    )


######################### nb_07.py


def init_cnn_(m, f):
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, "bias", None) is not None:
            m.bias.data.zero_()
    for l in m.children():
        init_cnn_(l, f)


def init_cnn(m, uniform=False):
    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
    init_cnn_(m, f)


def get_learn_run(
    nfs, data, lr, layer, cbs=None, opt_func=None, uniform=False, **kwargs
):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model, uniform=uniform)
    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)


def conv_layer(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [
        nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride, bias=not bn),
        GeneralRelu(**kwargs),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)


class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(nf, 1, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1, 1))
        self.register_buffer("sums", torch.zeros(1, nf, 1, 1))
        self.register_buffer("sqrs", torch.zeros(1, nf, 1, 1))
        self.register_buffer("count", tensor(0.0))
        self.register_buffer("factor", tensor(0.0))
        self.register_buffer("offset", tensor(0.0))
        self.batch = 0

    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0, 2, 3)
        s = x.sum(dims, keepdim=True)
        ss = (x * x).sum(dims, keepdim=True)
        c = s.new_tensor(x.numel() / nc)
        mom1 = s.new_tensor(1 - (1 - self.mom) / math.sqrt(bs - 1))
        self.sums.lerp_(s, mom1)
        self.sqrs.lerp_(ss, mom1)
        self.count.lerp_(c, mom1)
        self.batch += bs
        means = self.sums / self.count
        varns = (self.sqrs / self.count).sub_(means * means)
        if bool(self.batch < 20):
            varns.clamp_min_(0.01)
        self.factor = self.mults / (varns + self.eps).sqrt()
        self.offset = self.adds - means * self.factor

    def forward(self, x):
        if self.training:
            self.update_stats(x)
        return x * self.factor + self.offset


######################### nb_07a.py


def get_batch(dl, run):
    run.xb, run.yb = next(iter(dl))
    for cb in run.cbs:
        cb.set_runner(run)
    run("begin_batch")
    return run.xb, run.yb


def find_modules(m, cond):
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])


def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return isinstance(l, lin_layers)


def lsuv_module(m, xb):
    h = Hook(m, append_mean)

    if getattr(m, "bias", None) is not None:
        while mdl(xb) is not None and abs(h.mean) > 1e-3:
            m.bias.data -= h.mean

    while mdl(xb) is not None and abs(h.std - 1) > 1e-3:
        m.weight.data /= h.std

    h.remove()
    return h.mean, h.std


######################### nb_08.py


import PIL, os, mimetypes

Path.ls = lambda x: list(x.iterdir())

image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)


def setify(o):
    return o if isinstance(o, set) else set(listify(o))


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(path, extensions=None, recurse=False, include=None):
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path)
        ):  # returns (dirpath, dirnames, filenames)
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, f, extensions)


def compose(x, funcs, *args, order_key="_order", **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


class ItemList(ListContainer):
    def __init__(self, items, path=".", tfms=None):
        super().__init__(items)
        self.path, self.tfms = Path(path), tfms

    def __repr__(self):
        return f"{super().__repr__()}\nPath: {self.path}"

    def new(self, items, cls=None):
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, tfms=self.tfms)

    def get(self, i):
        return i

    def _get(self, i):
        return compose(self.get(i), self.tfms)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res, list):
            return [self._get(o) for o in res]
        return self._get(res)


class ImageList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None:
            extensions = image_extensions
        return cls(
            get_files(path, extensions, recurse=recurse, include=include),
            path,
            **kwargs,
        )

    def get(self, fn):
        return PIL.Image.open(fn)


class Transform:
    _order = 0


class MakeRGB(Transform):
    def __call__(self, item):
        return item.convert("RGB")


def make_rgb(item):
    return item.convert("RGB")


def grandparent_splitter(fn, valid_name="valid", train_name="train"):
    gp = fn.parent.parent.name
    return True if gp == valid_name else False if gp == train_name else None


def split_by_func(items, f):
    mask = [f(o) for o in items]
    # `None` values will be filtered out
    f = [o for o, m in zip(items, mask) if m == False]
    t = [o for o, m in zip(items, mask) if m == True]
    return f, t


class SplitData:
    def __init__(self, train, valid):
        self.train, self.valid = train, valid

    def __getattr__(self, k):
        return getattr(self.train, k)

    # This is needed if we want to pickle SplitData and be able to load it back without recursion errors
    def __setstate__(self, data: Any):
        self.__dict__.update(data)

    @classmethod
    def split_by_func(cls, il, f):
        lists = map(il.new, split_by_func(il.items, f))
        return cls(*lists)

    def __repr__(self):
        return f"{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n"


from collections import OrderedDict


def uniqueify(x, sort=False):
    res = list(OrderedDict.fromkeys(x).keys())
    if sort:
        res.sort()
    return res


class Processor:
    def process(self, items):
        return items


class CategoryProcessor(Processor):
    def __init__(self):
        self.vocab = None

    def __call__(self, items):
        # The vocab is defined on the first use.
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi = {v: k for k, v in enumerate(self.vocab)}
        return [self.proc1(o) for o in items]

    def proc1(self, item):
        return self.otoi[item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]

    def deproc1(self, idx):
        return self.vocab[idx]


def parent_labeler(fn):
    return fn.parent.name


def _label_by_func(ds, f, cls=ItemList):
    return cls([f(o) for o in ds.items], path=ds.path)


# This is a slightly different from what was seen during the lesson,
#   we'll discuss the changes in lesson 11
class LabeledData:
    def process(self, il, proc):
        return il.new(compose(il.items, proc))

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x, self.y = self.process(x, proc_x), self.process(y, proc_y)
        self.proc_x, self.proc_y = proc_x, proc_y

    def __repr__(self):
        return f"{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n"

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def x_obj(self, idx):
        return self.obj(self.x, idx, self.proc_x)

    def y_obj(self, idx):
        return self.obj(self.y, idx, self.proc_y)

    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (
            isinstance(idx, torch.LongTensor) and not idx.ndim
        )
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deproc1(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)


def label_by_func(sd, f, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
    return SplitData(train, valid)


class ResizeFixed(Transform):
    _order = 10

    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, item):
        return item.resize(self.size, PIL.Image.BILINEAR)


def to_byte_tensor(item):
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
    w, h = item.size
    return res.view(h, w, -1).permute(2, 0, 1)


to_byte_tensor._order = 20


def to_float_tensor(item):
    return item.float().div_(255.0)


to_float_tensor._order = 30


def show_image(im, figsize=(3, 3)):
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(im.permute(1, 2, 0))


class DataBunch:
    def __init__(self, train_dl, valid_dl, c_in=None, c_out=None):
        self.train_dl, self.valid_dl, self.c_in, self.c_out = (
            train_dl,
            valid_dl,
            c_in,
            c_out,
        )

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


def databunchify(sd, bs, c_in=None, c_out=None, **kwargs):
    dls = get_dls(sd.train, sd.valid, bs, **kwargs)
    return DataBunch(*dls, c_in=c_in, c_out=c_out)


SplitData.to_databunch = databunchify


def normalize_chan(x, mean, std):
    return (x - mean[..., None, None]) / std[..., None, None]


_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
norm_imagenette = partial(normalize_chan, mean=_m.cuda(), std=_s.cuda())

import math


def prev_pow_2(x):
    return 2 ** math.floor(math.log2(x))


def get_cnn_layers(data, nfs, layer, **kwargs):
    def f(ni, nf, stride=2):
        return layer(ni, nf, 3, stride=stride, **kwargs)

    l1 = data.c_in
    l2 = prev_pow_2(l1 * 3 * 3)
    layers = [f(l1, l2, stride=1), f(l2, l2 * 2, stride=2), f(l2 * 2, l2 * 4, stride=2)]
    nfs = [l2 * 4] + nfs
    layers += [f(nfs[i], nfs[i + 1]) for i in range(len(nfs) - 1)]
    layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c_out)]
    return layers


def get_cnn_model(data, nfs, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))


def get_learn_run(nfs, data, lr, layer, cbs=None, opt_func=None, **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)


def model_summary(run, learn, data, find_all=False):
    xb, yb = get_batch(data.valid_dl, run)
    device = next(learn.model.parameters()).device  # Model may not be on the GPU yet
    xb, yb = xb.to(device), yb.to(device)
    mods = (
        find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    )
    f = lambda hook, mod, inp, out: print(f"{mod}\n{out.shape}\n")
    with Hooks(mods, f) as hooks:
        learn.model(xb)


######################### nb_09.py


def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)
    return p


class Recorder(Callback):
    #second veraion of recorder, here modifies to add validation losses
    def begin_fit(self):
        self.lrs, self.losses, self.val_losses = [], [], []

    def after_batch(self):
        if not self.in_train:
            # validation
            self.val_losses.append(self.loss.detach().cpu())
        self.lrs.append(self.opt.hypers[-1]["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self):
        plt.plot(self.lrs)

    def plot_loss(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.losses)
        ax1.set_ylabel('loss')

        ax2 = ax1.twiny()
        ax2.plot(self.val_losses, 'r-')
        for tl in ax2.get_xticklabels():
            tl.set_color('r')

    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.plot(self.lrs[:n], losses[:n])


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, listify(sched_funcs)

    def begin_batch(self):
        if not self.in_train:
            return
        fs = self.sched_funcs
        if len(fs) == 1:
            fs = fs * len(self.opt.param_groups)
        pos = self.n_epochs / self.epochs
        for f, h in zip(fs, self.opt.hypers):
            h[self.pname] = f(pos)


class LR_Find(Callback):
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.n_iter / self.max_iter
        lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for pg in self.opt.hypers:
            pg["lr"] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss * 10:
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss


def weight_decay(p, lr, wd, **kwargs):
    p.data.mul_(1 - lr * wd)
    return p


weight_decay._defaults = dict(wd=0.0)


def l2_reg(p, lr, wd, **kwargs):
    p.grad.data.add_(wd, p.data)
    return p


l2_reg._defaults = dict(wd=0.0)


def maybe_update(os, dest, f):
    for o in os:
        for k, v in f(o).items():
            if k not in dest:
                dest[k] = v


def get_defaults(d):
    return getattr(d, "_defaults", {})


class Optimizer:
    def __init__(self, params, steppers, **defaults):
        self.steppers = listify(steppers)
        maybe_update(self.steppers, defaults, get_defaults)
        # might be a generator
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list):
            self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        return [
            (p, hyper)
            for pg, hyper in zip(self.param_groups, self.hypers)
            for p in pg
            if p.grad is not None
        ]

    def zero_grad(self):
        for p, hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p, hyper in self.grad_params():
            compose(p, self.steppers, **hyper)


sgd_opt = partial(Optimizer, steppers=[weight_decay, sgd_step])


class StatefulOptimizer(Optimizer):
    def __init__(self, params, steppers, stats=None, **defaults):
        self.stats = listify(stats)
        maybe_update(self.stats, defaults, get_defaults)
        super().__init__(params, steppers, **defaults)
        self.state = {}

    def step(self):
        for p, hyper in self.grad_params():
            if p not in self.state:
                # Create a state for p and call all the statistics to initialize it.
                self.state[p] = {}
                maybe_update(self.stats, self.state[p], lambda o: o.init_state(p))
            state = self.state[p]
            for stat in self.stats:
                state = stat.update(p, state, **hyper)
            compose(p, self.steppers, **state, **hyper)
            self.state[p] = state


class Stat:
    _defaults = {}

    def init_state(self, p):
        raise NotImplementedError

    def update(self, p, state, **kwargs):
        raise NotImplementedError


def momentum_step(p, lr, grad_avg, **kwargs):
    p.data.add_(-lr, grad_avg)
    return p


def lin_comb(v1, v2, beta):
    return beta * v1 + (1 - beta) * v2


class AverageGrad(Stat):
    _defaults = dict(mom=0.9)

    def __init__(self, dampening: bool = False):
        self.dampening = dampening

    def init_state(self, p):
        return {"grad_avg": torch.zeros_like(p.grad.data)}

    def update(self, p, state, mom, **kwargs):
        state["mom_damp"] = 1 - mom if self.dampening else 1.0
        state["grad_avg"].mul_(mom).add_(state["mom_damp"], p.grad.data)
        return state


class AverageSqrGrad(Stat):
    _defaults = dict(sqr_mom=0.99)

    def __init__(self, dampening: bool = True):
        self.dampening = dampening

    def init_state(self, p):
        return {"sqr_avg": torch.zeros_like(p.grad.data)}

    def update(self, p, state, sqr_mom, **kwargs):
        state["sqr_damp"] = 1 - sqr_mom if self.dampening else 1.0
        state["sqr_avg"].mul_(sqr_mom).addcmul_(
            state["sqr_damp"], p.grad.data, p.grad.data
        )
        return state


class StepCount(Stat):
    def init_state(self, p):
        return {"step": 0}

    def update(self, p, state, **kwargs):
        state["step"] += 1
        return state


def debias(mom, damp, step):
    return damp * (1 - mom ** step) / (1 - mom)


def adam_step(
    p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs
):
    debias1 = debias(mom, mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg / debias2).sqrt() + eps)
    return p


adam_step._defaults = dict(eps=1e-5)


def adam_opt(xtra_step=None, **kwargs):
    return partial(
        StatefulOptimizer,
        steppers=[adam_step, weight_decay] + listify(xtra_step),
        stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()],
        **kwargs,
    )


######################### nb_09b.py


def param_getter(m):
    return m.parameters()


class Learner:
    def __init__(
        self,
        model,
        data,
        loss_func,
        opt_func=sgd_opt,
        lr=1e-2,
        splitter=param_getter,
        cbs=None,
        cb_funcs=None,
    ):
        self.model, self.data, self.loss_func, self.opt_func, self.lr, self.splitter = (
            model,
            data,
            loss_func,
            opt_func,
            lr,
            splitter,
        )
        self.in_train, self.logger, self.opt = False, print, None

        # NB: Things marked "NEW" are covered in lesson 12
        # NEW: avoid need for set_runner
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def add_cbs(self, cbs):
        for cb in listify(cbs):
            self.add_cb(cb)

    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs):
            self.cbs.remove(cb)

    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb, self.yb = xb, yb
            self("begin_batch")
            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb)
            self("after_loss")
            if not self.in_train:
                return
            self.loss.backward()
            self("after_backward")
            self.opt.step()
            self("after_step")
            self.opt.zero_grad()
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (xb, yb) in enumerate(self.dl):
                self.one_batch(i, xb, yb)
        except CancelEpochException:
            self("after_cancel_epoch")

    def do_begin_fit(self, epochs):
        self.epochs, self.loss = epochs, tensor(0.0)
        self("begin_fit")

    def do_begin_epoch(self, epoch):
        self.epoch, self.dl = epoch, self.data.train_dl
        return self("begin_epoch")

    def fit(self, epochs, cbs=None, reset_opt=False):
        # NEW: pass callbacks to fit() and have them removed when done
        self.add_cbs(cbs)
        # NEW: create optimizer on fit(), optionally replacing existing
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                self.do_begin_epoch(epoch)
                if not self("begin_epoch"):
                    self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self("begin_validate"):
                        self.all_batches()
                self("after_epoch")

        except CancelTrainException:
            self("after_cancel_train")
        finally:
            self("after_fit")
            self.remove_cbs(cbs)

    ALL_CBS = {
        "begin_batch",
        "after_pred",
        "after_loss",
        "after_backward",
        "after_step",
        "after_cancel_batch",
        "after_batch",
        "after_cancel_epoch",
        "begin_fit",
        "begin_epoch",
        "begin_epoch",
        "begin_validate",
        "after_epoch",
        "after_cancel_train",
        "after_fit",
    }

    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = (
            AvgStats(metrics, True),
            AvgStats(metrics, False),
        )

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        # We use the logger function of the `Learner` here, it can be customized to write in a file or in a progress bar
        self.logger(self.train_stats)
        self.logger(self.valid_stats)


def get_learner(
    nfs,
    data,
    lr,
    layer,
    loss_func=F.cross_entropy,
    cb_funcs=None,
    opt_func=sgd_opt,
    **kwargs,
):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs, opt_func=opt_func)


######################### nb_09c.py


import time
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = (
            AvgStats(metrics, True),
            AvgStats(metrics, False),
        )

    def begin_fit(self):
        met_names = ["loss"] + [m.__name__ for m in self.train_stats.metrics]
        names = (
            ["epoch"]
            + [f"train_{n}" for n in met_names]
            + [f"valid_{n}" for n in met_names]
            + ["time"]
        )
        self.logger(names)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)


class ProgressCallback(Callback):
    _order = -1

    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)

    def after_fit(self):
        self.mbar.on_iter_end()

    def after_batch(self):
        self.pb.update(self.iter)

    def begin_epoch(self):
        self.set_pb()

    def begin_validate(self):
        self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar, auto_update=False)
        self.mbar.update(self.epoch)


######################### nb_10.py


make_rgb._order = 0

import random


def show_image(im, ax=None, figsize=(3, 3)):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")
    ax.imshow(im.permute(1, 2, 0))


def show_batch(x, c=4, r=None, figsize=None):
    n = len(x)
    if r is None:
        r = int(math.ceil(n / c))
    if figsize is None:
        figsize = (c * 3, r * 3)
    fig, axes = plt.subplots(r, c, figsize=figsize)
    for xi, ax in zip(x, axes.flat):
        show_image(xi, ax)


class PilTransform(Transform):
    _order = 11


class PilRandomFlip(PilTransform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random() < self.p else x


class PilRandomDihedral(PilTransform):
    def __init__(self, p=0.75):
        self.p = (
            p * 7 / 8
        )  # Little hack to get the 1/8 identity dihedral transform taken into account.

    def __call__(self, x):
        if random.random() > self.p:
            return x
        return x.transpose(random.randint(0, 6))


from random import randint


def process_sz(sz):
    sz = listify(sz)
    return tuple(sz if len(sz) == 2 else [sz[0], sz[0]])


def default_crop_size(w, h):
    return [w, w] if w < h else [h, h]


class GeneralCrop(PilTransform):
    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR):
        self.resample, self.size = resample, process_sz(size)
        self.crop_size = None if crop_size is None else process_sz(crop_size)

    def default_crop_size(self, w, h):
        return default_crop_size(w, h)

    def __call__(self, x):
        csize = (
            self.default_crop_size(*x.size)
            if self.crop_size is None
            else self.crop_size
        )
        return x.transform(
            self.size,
            PIL.Image.EXTENT,
            self.get_corners(*x.size, *csize),
            resample=self.resample,
        )

    def get_corners(self, w, h):
        return (0, 0, w, h)


class CenterCrop(GeneralCrop):
    def __init__(self, size, scale=1.14, resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale = scale

    def default_crop_size(self, w, h):
        return [w / self.scale, h / self.scale]

    def get_corners(self, w, h, wc, hc):
        return ((w - wc) // 2, (h - hc) // 2, (w - wc) // 2 + wc, (h - hc) // 2 + hc)


class RandomResizedCrop(GeneralCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        resample=PIL.Image.BILINEAR,
    ):
        super().__init__(size, resample=resample)
        self.scale, self.ratio = scale, ratio

    def get_corners(self, w, h, wc, hc):
        area = w * h
        # Tries 10 times to get a proper crop inside the image.
        for attempt in range(10):
            area = random.uniform(*self.scale) * area
            ratio = math.exp(
                random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]))
            )
            new_w = int(round(math.sqrt(area * ratio)))
            new_h = int(round(math.sqrt(area / ratio)))
            if new_w <= w and new_h <= h:
                left = random.randint(0, w - new_w)
                top = random.randint(0, h - new_h)
                return (left, top, left + new_w, top + new_h)

        # Fallback to central crop
        left, top = randint(0, w - self.crop_size[0]), randint(0, h - self.crop_size[1])
        return (left, top, left + self.crop_size[0], top + self.crop_size[1])
        # Fallback to central crop


from torch import FloatTensor, LongTensor


def find_coeffs(orig_pts, targ_pts):
    matrix = []
    # The equations we'll need to solve.
    for p1, p2 in zip(targ_pts, orig_pts):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = FloatTensor(matrix)
    B = FloatTensor(orig_pts).view(8, 1)
    # The 8 scalars we seek are solution of AX = B
    return list(torch.solve(B, A)[0][:, 0])


def warp(img, size, src_coords, resample=PIL.Image.BILINEAR):
    w, h = size
    targ_coords = ((0, 0), (0, h), (w, h), (w, 0))
    c = find_coeffs(src_coords, targ_coords)
    res = img.transform(size, PIL.Image.PERSPECTIVE, list(c), resample=resample)
    return res


def uniform(a, b):
    return a + (b - a) * random.random()


class PilTiltRandomCrop(PilTransform):
    def __init__(
        self, size, crop_size=None, magnitude=0.0, resample=PIL.Image.BILINEAR
    ):
        self.resample, self.size, self.magnitude = resample, process_sz(size), magnitude
        self.crop_size = None if crop_size is None else process_sz(crop_size)

    def __call__(self, x):
        csize = default_crop_size(*x.size) if self.crop_size is None else self.crop_size
        left, top = randint(0, x.size[0] - csize[0]), randint(0, x.size[1] - csize[1])
        top_magn = min(
            self.magnitude, left / csize[0], (x.size[0] - left) / csize[0] - 1
        )
        lr_magn = min(self.magnitude, top / csize[1], (x.size[1] - top) / csize[1] - 1)
        up_t, lr_t = uniform(-top_magn, top_magn), uniform(-lr_magn, lr_magn)
        src_corners = tensor(
            [[-up_t, -lr_t], [up_t, 1 + lr_t], [1 - up_t, 1 - lr_t], [1 + up_t, lr_t]]
        )
        src_corners = src_corners * tensor(csize).float() + tensor([left, top]).float()
        src_corners = tuple([(int(o[0].item()), int(o[1].item())) for o in src_corners])
        return warp(x, self.size, src_corners, resample=self.resample)


import numpy as np


def np_to_float(x):
    return (
        torch.from_numpy(np.array(x, dtype=np.float32, copy=False))
        .permute(2, 0, 1)
        .contiguous()
        / 255.0
    )


np_to_float._order = 30
######################### nb_10b.py


class NoneReduce:
    def __init__(self, loss_func):
        self.loss_func, self.old_red = loss_func, None

    def __enter__(self):
        if hasattr(self.loss_func, "reduction"):
            self.old_red = getattr(self.loss_func, "reduction")
            setattr(self.loss_func, "reduction", "none")
            return self.loss_func
        else:
            return partial(self.loss_func, reduction="none")

    def __exit__(self, type, value, traceback):
        if self.old_red is not None:
            setattr(self.loss_func, "reduction", self.old_red)


from torch.distributions.beta import Beta


def unsqueeze(input, dims):
    for dim in listify(dims):
        input = torch.unsqueeze(input, dim)
    return input


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


class MixUp(Callback):
    _order = 90  # Runs after normalization and cuda

    def __init__(self, α: float = 0.4):
        self.distrib = Beta(tensor([α]), tensor([α]))

    def begin_fit(self):
        self.old_loss_func, self.run.loss_func = self.run.loss_func, self.loss_func

    def begin_batch(self):
        if not self.in_train:
            return  # Only mixup things during training
        λ = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)
        λ = torch.stack([λ, 1 - λ], 1)
        self.λ = unsqueeze(λ.max(1)[0], (1, 2, 3))
        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)
        xb1, self.yb1 = self.xb[shuffle], self.yb[shuffle]
        self.run.xb = lin_comb(self.xb, xb1, self.λ)

    def after_fit(self):
        self.run.loss_func = self.old_loss_func

    def loss_func(self, pred, yb):
        if not self.in_train:
            return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, self.yb1)
        loss = lin_comb(loss1, loss2, self.λ)
        return reduce_loss(loss, getattr(self.old_loss_func, "reduction", "mean"))


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε: float = 0.1, reduction="mean"):
        super().__init__()
        self.ε, self.reduction = ε, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss / c, nll, self.ε)


######################### nb_10c.py


from torch.nn.utils import parameters_to_vector
import apex.fp16_utils as fp16


def get_master(opt, flat_master=False):
    model_pgs = [
        [param for param in pg if param.requires_grad] for pg in opt.param_groups
    ]
    if flat_master:
        master_pgs = []
        for pg in model_pgs:
            mp = parameters_to_vector([param.data.float() for param in pg])
            mp = torch.nn.Parameter(mp, requires_grad=True)
            if mp.grad is None:
                mp.grad = mp.new(*mp.size())
            master_pgs.append([mp])
    else:
        master_pgs = [
            [param.clone().float().detach() for param in pg] for pg in model_pgs
        ]
        for pg in master_pgs:
            for param in pg:
                param.requires_grad_(True)
    return model_pgs, master_pgs


def to_master_grads(model_pgs, master_pgs, flat_master: bool = False) -> None:
    for (model_params, master_params) in zip(model_pgs, master_pgs):
        fp16.model_grads_to_master_grads(
            model_params, master_params, flat_master=flat_master
        )


def to_model_params(model_pgs, master_pgs, flat_master: bool = False) -> None:
    for (model_params, master_params) in zip(model_pgs, master_pgs):
        fp16.master_params_to_model_params(
            model_params, master_params, flat_master=flat_master
        )


def test_overflow(x):
    s = float(x.float().sum())
    return s == float("inf") or s == float("-inf") or s != s


def grad_overflow(param_groups):
    for group in param_groups:
        for p in group:
            if p.grad is not None:
                s = float(p.grad.data.float().sum())
                if s == float("inf") or s == float("-inf") or s != s:
                    return True
    return False


class MixedPrecision(Callback):
    _order = 99

    def __init__(
        self,
        loss_scale=512,
        flat_master=False,
        dynamic=True,
        max_loss_scale=2.0 ** 24,
        div_factor=2.0,
        scale_wait=500,
    ):
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.flat_master, self.dynamic, self.max_loss_scale = (
            flat_master,
            dynamic,
            max_loss_scale,
        )
        self.div_factor, self.scale_wait = div_factor, scale_wait
        self.loss_scale = max_loss_scale if dynamic else loss_scale

    def begin_fit(self):
        self.run.model = fp16.convert_network(self.model, dtype=torch.float16)
        self.model_pgs, self.master_pgs = get_master(self.opt, self.flat_master)
        # Changes the optimizer so that the optimization step is done in FP32.
        self.run.opt.param_groups = (
            self.master_pgs
        )  # Put those param groups inside our runner.
        if self.dynamic:
            self.count = 0

    def begin_batch(self):
        self.run.xb = self.run.xb.half()  # Put the inputs to half precision

    def after_pred(self):
        self.run.pred = self.run.pred.float()  # Compute the loss in FP32

    def after_loss(self):
        if self.in_train:
            self.run.loss *= self.loss_scale  # Loss scaling to avoid gradient underflow

    def after_backward(self):
        # First, check for an overflow
        if self.dynamic and grad_overflow(self.model_pgs):
            # Divide the loss scale by div_factor, zero the grad (after_step will be skipped)
            self.loss_scale /= self.div_factor
            self.model.zero_grad()
            return True  # skip step and zero_grad
        # Copy the gradients to master and unscale
        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params:
                if param.grad is not None:
                    param.grad.div_(self.loss_scale)
        # Check if it's been long enough without overflow
        if self.dynamic:
            self.count += 1
            if self.count == self.scale_wait:
                self.count = 0
                self.loss_scale *= self.div_factor

    def after_step(self):
        # Zero the gradients of the model since the optimizer is disconnected.
        self.model.zero_grad()
        # Update the params from master to model.
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)


######################### nb_11.py


def noop(x):
    return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=bias)


act_fn = nn.ReLU(inplace=True)


def init_cnn(m):
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)


def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act:
        layers.append(act_fn)
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride=1):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        layers = [conv_layer(ni, nh, 1)]
        layers += (
            [conv_layer(nh, nf, 3, stride=stride, zero_bn=True, act=False)]
            if expansion == 1
            else [
                conv_layer(nh, nh, 3, stride=stride),
                conv_layer(nh, nf, 1, zero_bn=True, act=False),
            ]
        )
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return act_fn(self.convs(x) + self.idconv(self.pool(x)))


class XResNet(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        nfs = [c_in, (c_in + 1) * 8, 64, 64]
        stem = [
            conv_layer(nfs[i], nfs[i + 1], stride=2 if i == 0 else 1) for i in range(3)
        ]

        nfs = [64 // expansion, 64, 128, 256, 512]
        res_layers = [
            cls._make_layer(
                expansion, nfs[i], nfs[i + 1], n_blocks=l, stride=1 if i == 0 else 2
            )
            for i, l in enumerate(layers)
        ]
        res = cls(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(nfs[-1] * expansion, c_out),
        )
        init_cnn(res)
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride):
        return nn.Sequential(
            *[
                ResBlock(expansion, ni if i == 0 else nf, nf, stride if i == 0 else 1)
                for i in range(n_blocks)
            ]
        )

class XResNetStem(nn.Sequential):
    #split XRestNet into 3 parts so can do telemetry on each part
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        nfs = [c_in, (c_in + 1) * 8, 64, 64]
        stem = [
            conv_layer(nfs[i], nfs[i + 1], stride=2 if i == 0 else 1) for i in range(3)
        ]

        res = cls(
            *stem,
        )
        init_cnn(res)
        return res

class XResNetRes(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers):


        nfs = [64 // expansion, 64, 128, 256, 512]
        res_layers = [
            cls._make_layer(
                expansion, nfs[i], nfs[i + 1], n_blocks=l, stride=1 if i == 0 else 2
            )
            for i, l in enumerate(layers)
        ]
        res = cls(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
        )
        init_cnn(res)
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride):
        return nn.Sequential(
            *[
                ResBlock(expansion, ni if i == 0 else nf, nf, stride if i == 0 else 1)
                for i in range(n_blocks)
            ]
        )

class XResNetAFL(nn.Sequential):
    @classmethod
    def create(cls, expansion, c_out=1000):

        nfs = [64 // expansion, 64, 128, 256, 512]
        res = cls(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(nfs[-1] * expansion, c_out),
        )
        init_cnn(res)
        return res

def xresnet18(**kwargs):
    return XResNet.create(1, [2, 2, 2, 2], **kwargs)


def xresnet34(**kwargs):
    return XResNet.create(1, [3, 4, 6, 3], **kwargs)


def xresnet50(**kwargs):
    return XResNet.create(4, [3, 4, 6, 3], **kwargs)


def xresnet101(**kwargs):
    return XResNet.create(4, [3, 4, 23, 3], **kwargs)


def xresnet152(**kwargs):
    return XResNet.create(4, [3, 8, 36, 3], **kwargs)


def get_batch(dl, learn):
    learn.xb, learn.yb = next(iter(dl))
    learn.do_begin_fit(0)
    learn("begin_batch")
    learn("after_fit")
    return learn.xb, learn.yb


def model_summary(model, data, find_all=False, print_mod=False):
    xb, yb = get_batch(data.valid_dl, learn)
    mods = find_modules(model, is_lin_layer) if find_all else model.children()
    f = lambda hook, mod, inp, out: print(
        f"====\n{mod}\n" if print_mod else "", out.shape
    )
    with Hooks(mods, f) as hooks:
        learn.model(xb)


def create_phases(phases):
    phases = listify(phases)
    return phases + [1 - sum(phases)]


def cnn_learner(
    arch,
    data,
    loss_func,
    opt_func,
    c_in=None,
    c_out=None,
    lr=1e-2,
    cuda=True,
    norm=None,
    progress=True,
    mixup=0,
    xtra_cb=None,
    **kwargs,
):
    cbfs = [partial(AvgStatsCallback, accuracy)] + listify(xtra_cb)
    if progress:
        cbfs.append(ProgressCallback)
    if cuda:
        cbfs.append(CudaCallback)
    if norm:
        cbfs.append(partial(BatchTransformXCallback, norm))
    if mixup:
        cbfs.append(partial(MixUp, mixup))
    arch_args = {}
    if not c_in:
        c_in = data.c_in
    if not c_out:
        c_out = data.c_out
    if c_in:
        arch_args["c_in"] = c_in
    if c_out:
        arch_args["c_out"] = c_out
    return Learner(
        arch(**arch_args),
        data,
        loss_func,
        opt_func=opt_func,
        lr=lr,
        cb_funcs=cbfs,
        **kwargs,
    )

######################### nb_11a.py


def random_splitter(fn, p_valid):
    return random.random() < p_valid


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


from types import SimpleNamespace

cb_types = SimpleNamespace(**{o: o for o in Learner.ALL_CBS})


class DebugCallback(Callback):
    _order = 999

    def __init__(self, cb_name, f=None):
        self.cb_name, self.f = cb_name, f

    def __call__(self, cb_name):
        if cb_name == self.cb_name:
            if self.f:
                self.f(self.run)
            else:
                set_trace()


def sched_1cycle(lrs, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    phases = create_phases(pct_start)
    sched_lr = [
        combine_scheds(phases, cos_1cycle_anneal(lr / 10.0, lr, lr / 1e5)) for lr in lrs
    ]
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler("lr", sched_lr), ParamScheduler("mom", sched_mom)]


######################### nb_12.py


def read_file(fn):
    with open(fn, "r", encoding="utf8") as f:
        return f.read()


class TextList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=".txt", recurse=True, include=None, **kwargs):
        return cls(
            get_files(path, extensions, recurse=recurse, include=include),
            path,
            **kwargs,
        )

    def get(self, i):
        if isinstance(i, Path):
            return read_file(i)
        return i


import spacy, html

# special tokens
UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ = (
    "xxunk xxpad xxbos xxeos xxrep xxwrep xxup xxmaj".split()
)


def sub_br(t):
    "Replaces the <br /> by \n"
    re_br = re.compile(r"<\s*br\s*/?>", re.IGNORECASE)
    return re_br.sub("\n", t)


def spec_add_spaces(t):
    "Add spaces around / and #"
    return re.sub(r"([/#])", r" \1 ", t)


def rm_useless_spaces(t):
    "Remove multiple spaces"
    return re.sub(" {2,}", " ", t)


def replace_rep(t):
    "Replace repetitions at the character level: cccc -> TK_REP 4 c"

    def _replace_rep(m: Collection[str]) -> str:
        c, cc = m.groups()
        return f" {TK_REP} {len(cc)+1} {c} "

    re_rep = re.compile(r"(\S)(\1{3,})")
    return re_rep.sub(_replace_rep, t)


def replace_wrep(t):
    "Replace word repetitions: word word word -> TK_WREP 3 word"

    def _replace_wrep(m: Collection[str]) -> str:
        c, cc = m.groups()
        return f" {TK_WREP} {len(cc.split())+1} {c} "

    re_wrep = re.compile(r"(\b\w+\W+)(\1{3,})")
    return re_wrep.sub(_replace_wrep, t)


def fixup_text(x):
    "Various messy things we've seen in documents"
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace("<unk>", UNK)
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(x))


default_pre_rules = [
    fixup_text,
    replace_rep,
    replace_wrep,
    spec_add_spaces,
    rm_useless_spaces,
    sub_br,
]
default_spec_tok = [UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ]


def replace_all_caps(x):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    res = []
    for t in x:
        if t.isupper() and len(t) > 1:
            res.append(TK_UP)
            res.append(t.lower())
        else:
            res.append(t)
    return res


def deal_caps(x):
    "Replace all Capitalized tokens in by their lower version and add `TK_MAJ` before."
    res = []
    for t in x:
        if t == "":
            continue
        if t[0].isupper() and len(t) > 1 and t[1:].islower():
            res.append(TK_MAJ)
        res.append(t.lower())
    return res


def add_eos_bos(x):
    return [BOS] + x + [EOS]


default_post_rules = [deal_caps, replace_all_caps, add_eos_bos]

from spacy.symbols import ORTH
from concurrent.futures import ProcessPoolExecutor


def parallel(func, arr, max_workers=4):
    if max_workers < 2:
        results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]):
        return results


class TokenizeProcessor(Processor):
    def __init__(
        self, lang="en", chunksize=2000, pre_rules=None, post_rules=None, max_workers=4
    ):
        self.chunksize, self.max_workers = chunksize, max_workers
        self.tokenizer = spacy.blank(lang).tokenizer
        for w in default_spec_tok:
            self.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pre_rules = default_pre_rules if pre_rules is None else pre_rules
        self.post_rules = default_post_rules if post_rules is None else post_rules

    def proc_chunk(self, args):
        i, chunk = args
        chunk = [compose(t, self.pre_rules) for t in chunk]
        docs = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]
        docs = [compose(t, self.post_rules) for t in docs]
        return docs

    def __call__(self, items):
        toks = []
        if isinstance(items[0], Path):
            items = [read_file(i) for i in items]
        chunks = [
            items[i : i + self.chunksize]
            for i in (range(0, len(items), self.chunksize))
        ]
        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)
        return sum(toks, [])

    def proc1(self, item):
        return self.proc_chunk([toks])[0]

    def deprocess(self, toks):
        return [self.deproc1(tok) for tok in toks]

    def deproc1(self, tok):
        return " ".join(tok)


import collections


class NumericalizeProcessor(Processor):
    def __init__(self, vocab=None, max_vocab=60000, min_freq=2):
        self.vocab, self.max_vocab, self.min_freq = vocab, max_vocab, min_freq

    def __call__(self, items):
        # The vocab is defined on the first use.
        if self.vocab is None:
            freq = Counter(p for o in items for p in o)
            self.vocab = [
                o for o, c in freq.most_common(self.max_vocab) if c >= self.min_freq
            ]
            for o in reversed(default_spec_tok):
                if o in self.vocab:
                    self.vocab.remove(o)
                self.vocab.insert(0, o)
        if getattr(self, "otoi", None) is None:
            self.otoi = collections.defaultdict(
                int, {v: k for k, v in enumerate(self.vocab)}
            )
        return [self.proc1(o) for o in items]

    def proc1(self, item):
        return [self.otoi[o] for o in item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]

    def deproc1(self, idx):
        return [self.vocab[i] for i in idx]


class LM_PreLoader:
    def __init__(self, data, bs=64, bptt=70, shuffle=False):
        self.data, self.bs, self.bptt, self.shuffle = data, bs, bptt, shuffle
        total_len = sum([len(t) for t in data.x])
        self.n_batch = total_len // bs
        self.batchify()

    def __len__(self):
        return ((self.n_batch - 1) // self.bptt) * self.bs

    def __getitem__(self, idx):
        source = self.batched_data[idx % self.bs]
        seq_idx = (idx // self.bs) * self.bptt
        return (
            source[seq_idx : seq_idx + self.bptt],
            source[seq_idx + 1 : seq_idx + self.bptt + 1],
        )

    def batchify(self):
        texts = self.data.x
        if self.shuffle:
            texts = texts[torch.randperm(len(texts))]
        stream = torch.cat([tensor(t) for t in texts])
        self.batched_data = stream[: self.n_batch * self.bs].view(self.bs, self.n_batch)


def get_lm_dls(train_ds, valid_ds, bs, bptt, **kwargs):
    return (
        DataLoader(
            LM_PreLoader(train_ds, bs, bptt, shuffle=True), batch_size=bs, **kwargs
        ),
        DataLoader(
            LM_PreLoader(valid_ds, bs, bptt, shuffle=False), batch_size=2 * bs, **kwargs
        ),
    )


def lm_databunchify(sd, bs, bptt, **kwargs):
    return DataBunch(*get_lm_dls(sd.train, sd.valid, bs, bptt, **kwargs))


from torch.utils.data import Sampler


class SortSampler(Sampler):
    def __init__(self, data_source, key):
        self.data_source, self.key = data_source, key

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(
            sorted(list(range(len(self.data_source))), key=self.key, reverse=True)
        )


class SortishSampler(Sampler):
    def __init__(self, data_source, key, bs):
        self.data_source, self.key, self.bs = data_source, key, bs

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [
            idxs[i : i + self.bs * 50] for i in range(0, len(idxs), self.bs * 50)
        ]
        sorted_idx = torch.cat(
            [tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches]
        )
        batches = [
            sorted_idx[i : i + self.bs] for i in range(0, len(sorted_idx), self.bs)
        ]
        max_idx = torch.argmax(
            tensor([self.key(ck[0]) for ck in batches])
        )  # find the chunk with the largest key,
        batches[0], batches[max_idx] = (
            batches[max_idx],
            batches[0],
        )  # then make sure it goes first.
        batch_idxs = torch.randperm(len(batches) - 2)
        sorted_idx = (
            torch.cat([batches[i + 1] for i in batch_idxs])
            if len(batches) > 1
            else LongTensor([])
        )
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)


def pad_collate(samples, pad_idx=1, pad_first=False):
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i, s in enumerate(samples):
        if pad_first:
            res[i, -len(s[0]) :] = LongTensor(s[0])
        else:
            res[i, : len(s[0])] = LongTensor(s[0])
    return res, tensor([s[1] for s in samples])


def get_clas_dls(train_ds, valid_ds, bs, **kwargs):
    train_sampler = SortishSampler(train_ds.x, key=lambda t: len(train_ds.x[t]), bs=bs)
    valid_sampler = SortSampler(valid_ds.x, key=lambda t: len(valid_ds.x[t]))
    return (
        DataLoader(
            train_ds,
            batch_size=bs,
            sampler=train_sampler,
            collate_fn=pad_collate,
            **kwargs,
        ),
        DataLoader(
            valid_ds,
            batch_size=bs * 2,
            sampler=valid_sampler,
            collate_fn=pad_collate,
            **kwargs,
        ),
    )


def clas_databunchify(sd, bs, **kwargs):
    return DataBunch(*get_clas_dls(sd.train, sd.valid, bs, **kwargs))


######################### nb_12a.py


def dropout_mask(x, sz, p):
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


import warnings

WEIGHT_HH = "weight_hh_l0"


class WeightDropout(nn.Module):
    def __init__(self, module, weight_p=[0.0], layer_names=[WEIGHT_HH]):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f"{layer}_raw", nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(
                w, p=self.weight_p, training=False
            )

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f"{layer}_raw")
            self.module._parameters[layer] = F.dropout(
                raw_w, p=self.weight_p, training=self.training
            )

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class EmbeddingDropout(nn.Module):
    "Applies dropout in the embedding layer by zeroing out some elements of the embedding vector."

    def __init__(self, emb, embed_p):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale:
            masked_embed.mul_(scale)
        return F.embedding(
            words,
            masked_embed,
            self.pad_idx,
            self.emb.max_norm,
            self.emb.norm_type,
            self.emb.scale_grad_by_freq,
            self.emb.sparse,
        )


def to_detach(h):
    "Detaches `h` from its history."
    return h.detach() if type(h) == torch.Tensor else tuple(to_detach(v) for v in h)


class AWD_LSTM(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange = 0.1

    def __init__(
        self,
        vocab_sz,
        emb_sz,
        n_hid,
        n_layers,
        pad_token,
        hidden_p=0.2,
        input_p=0.6,
        embed_p=0.1,
        weight_p=0.5,
    ):
        super().__init__()
        self.bs, self.emb_sz, self.n_hid, self.n_layers = 1, emb_sz, n_hid, n_layers
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.rnns = [
            nn.LSTM(
                emb_sz if l == 0 else n_hid,
                (n_hid if l != n_layers - 1 else emb_sz),
                1,
                batch_first=True,
            )
            for l in range(n_layers)
        ]
        self.rnns = nn.ModuleList([WeightDropout(rnn, weight_p) for rnn in self.rnns])
        self.emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        bs, sl = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        raw_output = self.input_dp(self.emb_dp(input))
        new_hidden, raw_outputs, outputs = [], [], []
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1:
                raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden)
        return raw_outputs, outputs

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [
            (self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)
        ]


class LinearDecoder(nn.Module):
    def __init__(self, n_out, n_hid, output_p, tie_encoder=None, bias=True):
        super().__init__()
        self.output_dp = RNNDropout(output_p)
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        if bias:
            self.decoder.bias.data.zero_()
        if tie_encoder:
            self.decoder.weight = tie_encoder.weight
        else:
            init.kaiming_uniform_(self.decoder.weight)

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1]).contiguous()
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded, raw_outputs, outputs


class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."

    def reset(self):
        for c in self.children():
            if hasattr(c, "reset"):
                c.reset()


def get_language_model(
    vocab_sz,
    emb_sz,
    n_hid,
    n_layers,
    pad_token,
    output_p=0.4,
    hidden_p=0.2,
    input_p=0.6,
    embed_p=0.1,
    weight_p=0.5,
    tie_weights=True,
    bias=True,
):
    rnn_enc = AWD_LSTM(
        vocab_sz,
        emb_sz,
        n_hid=n_hid,
        n_layers=n_layers,
        pad_token=pad_token,
        hidden_p=hidden_p,
        input_p=input_p,
        embed_p=embed_p,
        weight_p=weight_p,
    )
    enc = rnn_enc.emb if tie_weights else None
    return SequentialRNN(
        rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias)
    )


class GradientClipping(Callback):
    def __init__(self, clip=None):
        self.clip = clip

    def after_backward(self):
        if self.clip:
            nn.utils.clip_grad_norm_(self.run.model.parameters(), self.clip)


class RNNTrainer(Callback):
    def __init__(self, α, β):
        self.α, self.β = α, β

    def after_pred(self):
        # Save the extra outputs for later and only returns the true output.
        self.raw_out, self.out = self.pred[1], self.pred[2]
        self.run.pred = self.pred[0]

    def after_loss(self):
        # AR and TAR
        if self.α != 0.0:
            self.run.loss += self.α * self.out[-1].float().pow(2).mean()
        if self.β != 0.0:
            h = self.raw_out[-1]
            if len(h) > 1:
                self.run.loss += self.β * (h[:, 1:] - h[:, :-1]).float().pow(2).mean()

    def begin_epoch(self):
        # Shuffle the texts at the beginning of the epoch
        if hasattr(self.dl.dataset, "batchify"):
            self.dl.dataset.batchify()


def cross_entropy_flat(input, target):
    bs, sl = target.size()
    return F.cross_entropy(input.view(bs * sl, -1), target.view(bs * sl))


def accuracy_flat(input, target):
    bs, sl = target.size()
    return accuracy(input.view(bs * sl, -1), target.view(bs * sl))

from functools import partial
from utils.nb_functions import camel2snake, listify
import re
import torch
from torch import tensor, nn
import matplotlib.pyplot as plt
from utils.nb_classes_l8_to_10 import AvgStats

class Callback():
    _order=0
    def set_runner(self, run):
        self.run=run
    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f:
            #then callback exists and is now OK to call
            if f():
                #here is where we actually run the method
                return True
        return False

class TrainEvalCallback(Callback):
    def begin_fit(self):
        print(f'>>TrainEvalCallback.begin_fit()')
        self.run.n_epochs=0.
        self.run.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1

    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False

class CancelTrainException(Exception):
    pass
class CancelEpochException(Exception):
    pass
class CancelBatchException(Exception):
    pass

class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        self.in_train = False
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

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
            self.xb,self.yb = xb,yb
            #print(f'l10_revised.Runner.one_batch is_cuda xb: {xb.is_cuda}, yb: {yb.is_cuda}')
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train:
                return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:
            self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            i=0
            for xb,yb in dl:
                if i==0:
                    print(f'Runner.all_batches is_cuda xb: {xb.is_cuda}, yb: {yb.is_cuda}')
                self.one_batch(xb, yb)
                i+=1
        except CancelEpochException:
            print('Runner.CancelEpochException')
            self('after_cancel_epoch')

    def fit(self, epochs, learn):
        self.epochs,self.learn,self.loss = epochs,learn,tensor(0.)

        try:
            for cb in self.cbs:
                #print(f'Runner.fit() cb: {cb}')
                cb.set_runner(self)
            self('begin_fit')

            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'):
                    self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self('begin_validate'):
                        self.all_batches(self.data.valid_dl)
                self('after_epoch')


        except CancelTrainException:
            print('Runner.CancelTrainException')
            self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            #print(f'Runner.__call__() calling cb_name: {cb_name} on cb: {cb.name}')
            temp_res= cb(cb_name)
            res = temp_res and res
        return res

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)

class Recorder(Callback):
    def begin_fit(self):
        print(f'>>Recorder.begin_fit()')
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        for pg,lr in zip(self.opt.param_groups,self.lrs):
            lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr  (self, pgid=-1):
        plt.plot(self.lrs[pgid])
    def plot_loss(self, skip_last=0):
        plt.plot(self.losses[:len(self.losses)-skip_last])

    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs    = self.lrs[pgid]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])


class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs

    def begin_fit(self):
        print(f'>>ParamScheduler.begin_fit()')
        if not isinstance(self.sched_funcs, (list,tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups)==len(self.sched_funcs)
        for pg,f in zip(self.opt.param_groups,self.sched_funcs):
            pg[self.pname] = f(self.n_epochs/self.epochs)

    def begin_batch(self):
        if self.in_train:
            self.set_param()

'''
class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.param_groups: pg['lr'] = lr

    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss
'''

class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.hypers: pg['lr'] = lr

    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss
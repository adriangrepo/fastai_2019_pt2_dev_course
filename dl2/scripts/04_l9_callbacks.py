#!/usr/bin/env python
# coding: utf-8








#export
from utils.nb_functions import *
from utils.nb_classes_l8_to_10 import *


# ## DataBunch/Learner

# <pre>
# class Dataset():
#     def __init__(self, x, y): 
#         self.x,self.y = x,y
#     def __len__(self): 
#         return len(self.x)
#     def __getitem__(self, i): 
#         return self.x[i],self.y[i]
# </pre>



x_train,y_train,x_valid,y_valid = get_data()
train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)
nh,bs = 50,64
c = y_train.max().item()+1
loss_func = F.cross_entropy




c


# Factor out the connected pieces of info out of the fit() argument list
# 
# `fit(epochs, model, loss_func, opt, train_dl, valid_dl)`
# 
# Let's replace it with something that looks like this:
# 
# `fit(1, learn)`
# 
# This will allow us to tweak what's happening inside the training loop in other places of the code because the `Learner` object will be mutable, so changing any of its attribute elsewhere will be seen in our training loop.



#export
class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        print(f'type(train_dl): {type(train_dl)}, type(valid_dl):{type(valid_dl)}, c: {c}')
        self.train_dl,self.valid_dl,self.c = train_dl,valid_dl,c
        
    @property
    def train_ds(self): 
        return self.train_dl.dataset
        
    @property
    def valid_ds(self): 
        return self.valid_dl.dataset


# <pre>
# def get_dls(train_ds, valid_ds, bs, **kwargs):
#     return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
#             DataLoader(valid_ds, batch_size=bs*2, **kwargs))
# </pre>



data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)




#export
def get_model(data, lr=0.5, nh=50):
    m = data.train_ds.x.shape[1]
    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,data.c))
    return model, optim.SGD(model.parameters(), lr=lr)

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data




learn = Learner(*get_model(data), loss_func, data)




def fit(epochs, learn):
    for epoch in range(epochs):
        learn.model.train()
        for xb,yb in learn.data.train_dl:
            loss = learn.loss_func(learn.model(xb), yb)
            loss.backward()
            learn.opt.step()
            learn.opt.zero_grad()

        learn.model.eval()
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in learn.data.valid_dl:
                pred = learn.model(xb)
                tot_loss += learn.loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(learn.data.valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv




loss,acc = fit(1, learn)


# ## CallbackHandler

# This was our training loop (without validation) from the previous notebook, with the inner loop contents factored out:
# 
# ```python
# def one_batch(xb,yb):
#     pred = model(xb)
#     loss = loss_func(pred, yb)
#     loss.backward()
#     opt.step()
#     opt.zero_grad()
#     
# def fit():
#     for epoch in range(epochs):
#         for b in train_dl: one_batch(*b)
# ```

# Add callbacks so we can remove complexity from loop, and make it flexible:



def one_batch(xb, yb, cb):
    if not cb.begin_batch(xb,yb): 
        return
    loss = cb.learn.loss_func(cb.learn.model(xb), yb)
    if not cb.after_loss(loss): 
        return
    loss.backward()
    if cb.after_backward(): 
        print('cb.after_backward()=True')
        cb.learn.opt.step()
    if cb.after_step(): 
        print('cb.after_step()=True')
        cb.learn.opt.zero_grad()

def all_batches(dl, cb):
    for xb,yb in dl:
        one_batch(xb, yb, cb)
        if cb.do_stop(): 
            return

def fit(epochs, learn, cb):
    print('fit()')
    if not cb.begin_fit(learn): 
        return
    for epoch in range(epochs):
        if not cb.begin_epoch(epoch): 
            continue
        all_batches(learn.data.train_dl, cb)
        
        if cb.begin_validate():
            with torch.no_grad(): 
                all_batches(learn.data.valid_dl, cb)
        if cb.do_stop() or not cb.after_epoch(): 
            break
    cb.after_fit()




class Callback():
    def begin_fit(self, learn):
        print('Callback.begin_fit()')
        self.learn = learn
        return True
    def after_fit(self): 
        print('Callback.after_fit()')
        return True
    def begin_epoch(self, epoch):
        self.epoch=epoch
        return True
    def begin_validate(self): 
        return True
    def after_epoch(self): 
        return True
    def begin_batch(self, xb, yb):
        self.xb,self.yb = xb,yb
        return True
    def after_loss(self, loss):
        self.loss = loss
        return True
    def after_backward(self): 
        return True
    def after_step(self): 
        print('Callback.after_step()')
        return True




class CallbackHandler():
    def __init__(self,cbs=None):
        self.cbs = cbs if cbs else []

    def begin_fit(self, learn):
        print('CallbackHandler.begin_fit()')
        self.learn,self.in_train = learn,True
        learn.stop = False
        res = True
        for cb in self.cbs: 
            res = res and cb.begin_fit(learn)
        return res

    def after_fit(self):
        print('CallbackHandler.after_fit()')
        res = not self.in_train
        for cb in self.cbs: 
            res = res and cb.after_fit()
        return res
    
    def begin_epoch(self, epoch):
        learn.model.train()
        self.in_train=True
        res = True
        for cb in self.cbs: 
            res = res and cb.begin_epoch(epoch)
        return res

    def begin_validate(self):
        self.learn.model.eval()
        self.in_train=False
        res = True
        for cb in self.cbs: 
            res = res and cb.begin_validate()
        return res

    def after_epoch(self):
        res = True
        for cb in self.cbs: 
            res = res and cb.after_epoch()
        return res
    
    def begin_batch(self, xb, yb):
        res = True
        for cb in self.cbs: 
            res = res and cb.begin_batch(xb, yb)
        return res

    def after_loss(self, loss):
        res = self.in_train
        for cb in self.cbs: 
            res = res and cb.after_loss(loss)
        return res

    def after_backward(self):
        res = True
        for cb in self.cbs: 
            res = res and cb.after_backward()
        return res

    def after_step(self):
        print('CallbackHandler.after_step()')
        res = True
        for cb in self.cbs: 
            res = res and cb.after_step()
        return res
    
    def do_stop(self):
        try:     
            return learn.stop
        finally: 
            learn.stop = False




class TestCallback(Callback):
    def begin_fit(self,learn):
        print('TestCallback.begin_fit()')
        super().begin_fit(learn)
        self.n_iters = 0
        return True
        
    def after_step(self):
        print('TestCallback.after_step()')
        self.n_iters += 1
        print(self.n_iters)
        if self.n_iters>=10: learn.stop = True
        return True




fit(1, learn, cb=CallbackHandler([TestCallback()]))


# This is roughly how fastai does it now (except that the handler can also change and return `xb`, `yb`, and `loss`). But let's see if we can make things simpler and more flexible, so that a single class has access to everything and can change anything at any time. The fact that we're passing `cb` to so many functions is a strong hint they should all be in the same class!

# ## Runner



#export
import re

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

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


# This first callback is reponsible to switch the model back and forth in training or validation mode, as well as maintaining a count of the iterations, or the percentage of iterations ellapsed in the epoch.



#export
class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0
    
    def after_batch(self):
        if not self.in_train: 
            return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1
        
    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False


# We'll also re-create our TestCallback - but note this doesn't actually work right yet (can you see why?) We'll fix it in notebook 05b.



# Not working!
class TestCallback(Callback):
    _order=1
    def after_step(self):
        if self.n_iter>=10: 
            return True




cbname = 'TrainEvalCallback'
camel2snake(cbname)




TrainEvalCallback().name




#export
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




#export
class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
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
        self.xb,self.yb = xb,yb
        if self('begin_batch'): 
            return
        self.pred = self.model(self.xb)
        if self('after_pred'): 
            return
        self.loss = self.loss_func(self.pred, self.yb)
        if self('after_loss') or not self.in_train: 
            return
        self.loss.backward()
        if self('after_backward'): 
            return
        self.opt.step()
        if self('after_step'): 
            return
        self.opt.zero_grad()

    def all_batches(self, dl):
        self.iters = len(dl)
        for xb,yb in dl:
            if self.stop: 
                break
            self.one_batch(xb, yb)
            self('after_batch')
        self.stop=False

    def fit(self, epochs, learn):
        self.epochs,self.learn,self.loss = epochs,learn,tensor(0.)

        try:
            for cb in self.cbs: cb.set_runner(self)
            if self('begin_fit'): 
                return
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad(): 
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                if self('after_epoch'): 
                    break
            
        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): 
                return True
        return False


# Third callback: how to compute metrics.



#export
class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train
    
    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)
        
    @property
    def all_stats(self): 
        return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): 
        return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: 
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

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




learn = Learner(*get_model(data), loss_func, data)




stats = AvgStatsCallback([accuracy])
run = Runner(cbs=stats)




run.fit(2, learn)




loss,acc = stats.valid_stats.avg_stats
#assert acc>0.9
loss,acc




#export
from functools import partial




acc_cbf = partial(AvgStatsCallback,accuracy)




run = Runner(cb_funcs=acc_cbf)




run.fit(1, learn)


# Using Jupyter means we can get tab-completion even for dynamic code like this! :)



run.avg_stats.valid_stats.avg_stats


# ## Export



#!python notebook2script.py 04_callbacks.ipynb







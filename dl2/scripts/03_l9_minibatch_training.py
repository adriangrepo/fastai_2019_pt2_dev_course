#!/usr/bin/env python
# coding: utf-8








#export
from utils.nb_functions import *
import torch.nn.functional as F


# ## Initial setup

# ### Data



mpl.rcParams['image.cmap'] = 'gray'




x_train,y_train,x_valid,y_valid = get_data()




n,m = x_train.shape
c = y_train.max()+1
nh = 50




class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        print('__init__')
        super().__init__()
        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]
        
    def __call__(self, x):
        #print('__call__')
        for l in self.layers: 
            x = l(x)
            print(x)
        return x




model = Model(m, nh, 10)




pred = model(x_train)


# ### Cross entropy loss

# First, we will need to compute the softmax of our activations. This is defined by:
# 
# $$\hbox{softmax(x)}_{i} = \frac{e^{x_{i}}}{e^{x_{0}} + e^{x_{1}} + \cdots + e^{x_{n-1}}}$$
# 
# or more concisely:
# 
# $$\hbox{softmax(x)}_{i} = \frac{e^{x_{i}}}{\sum_{0 \leq j \leq n-1} e^{x_{j}}}$$ 
# 
# In practice, we will need the log of the softmax when we calculate the loss.



def log_softmax(x): 
    return (x.exp()/(x.exp().sum(-1,keepdim=True))).log()




sm_pred = log_softmax(pred)


# The cross entropy loss for some target $x$ and some prediction $p(x)$ is given by:
# 
# $$ -\sum x\, \log p(x) $$
# 
# But since our $x$s are 1-hot encoded, this can be rewritten as $-\log(p_{i})$ where i is the index of the desired target.

# This can be done using numpy-style [integer array indexing](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#integer-array-indexing). Note that PyTorch supports all the tricks in the advanced indexing methods discussed in that link.



y_train[:3]




sm_pred




sm_pred.shape




sm_pred[0]




sm_pred[0][5]




sm_pred[[0,1,2], [5,0,4]]




y_train.shape[0]




def nll(input, target):
    print(f'target.shape[0]: {target.shape[0]}')
    return -input[range(target.shape[0]), target].mean()




loss = nll(sm_pred, y_train)




loss


# Note that the formula 
# 
# $$\log \left ( \frac{a}{b} \right ) = \log(a) - \log(b)$$ 
# 
# gives a simplification when we compute the log softmax, which was previously defined as `(x.exp()/(x.exp().sum(-1,keepdim=True))).log()`



def log_softmax(x): 
    return x - x.exp().sum(-1,keepdim=True).log()




test_near(nll(log_softmax(pred), y_train), loss)


# Then, there is a way to compute the log of the sum of exponentials in a more stable way, called the [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp). The idea is to use the following formula:
# 
# $$\log \left ( \sum_{j=1}^{n} e^{x_{j}} \right ) = \log \left ( e^{a} \sum_{j=1}^{n} e^{x_{j}-a} \right ) = a + \log \left ( \sum_{j=1}^{n} e^{x_{j}-a} \right )$$
# 
# where a is the maximum of the $x_{j}$.



def logsumexp(x):
    print(f'x: {type(x)}')
    print(f'x.max(-1)[0]: {x.max(-1)[0]}')
    print(f'x.max(0)[0]: {x.max(0)[0]}')
    print(f'x.max(1)[0]: {x.max(1)[0]}')
    print(f'x.shape: {x.shape}')
    #I think by putting -1 here we specify last axis - being the column of 2d matrix
    m = x.max(-1)[0]
    return m + (x-m[:,None]).exp().sum(-1).log()


# This way, we will avoid an overflow when taking the exponential of a big activation. In PyTorch, this is already implemented for us. 



test_near(logsumexp(pred), pred.logsumexp(-1))


# So we can use it for our `log_softmax` function.



def log_softmax(x): 
    return x - x.logsumexp(-1,keepdim=True)




test_near(nll(log_softmax(pred), y_train), loss)


# Then use PyTorch's implementation.



test_near(F.nll_loss(F.log_softmax(pred, -1), y_train), loss)


# In PyTorch, `F.log_softmax` and `F.nll_loss` are combined in one optimized function, `F.cross_entropy`.



test_near(F.cross_entropy(pred, y_train), loss)


# ## Basic training loop

# Basically the training loop repeats over the following steps:
# - get the output of the model on a batch of inputs
# - compare the output to the labels we have and compute a loss
# - calculate the gradients of the loss with respect to every parameter of the model
# - update said parameters with those gradients to make them a little bit better



loss_func = F.cross_entropy




#export
def accuracy(out, yb): 
    return (torch.argmax(out, dim=1)==yb).float().mean()




bs=64                  # batch size

xb = x_train[0:bs]     # a mini-batch from x
preds = model(xb)      # predictions
preds[0], preds.shape




yb = y_train[0:bs]
loss_func(preds, yb)




accuracy(preds, yb)




lr = 0.5   # learning rate
epochs = 1 # how many epochs to train for


# class Tensor(torch._C._TensorBase):
#     ...
#     
#     def backward(self, gradient=None, retain_graph=None, create_graph=False):
#         r"""Computes the gradient of current tensor w.r.t. graph leaves.
# 
#         The graph is differentiated using the chain rule. If the tensor is
#         non-scalar (i.e. its data has more than one element) and requires
#         gradient, the function additionally requires specifying ``gradient``.
#         It should be a tensor of matching type and location, that contains
#         the gradient of the differentiated function w.r.t. ``self``.
# 
#         This function accumulates gradients in the leaves - you might need to
#         zero them before calling it.
# 
#         Arguments:
#             gradient (Tensor or None): Gradient w.r.t. the
#                 tensor. If it is a tensor, it will be automatically converted
#                 to a Tensor that does not require grad unless ``create_graph`` is True.
#                 None values can be specified for scalar Tensors or ones that
#                 don't require grad. If a None value would be acceptable then
#                 this argument is optional.
#             retain_graph (bool, optional): If ``False``, the graph used to compute
#                 the grads will be freed. Note that in nearly all cases setting
#                 this option to True is not needed and often can be worked around
#                 in a much more efficient way. Defaults to the value of
#                 ``create_graph``.
#             create_graph (bool, optional): If ``True``, graph of the derivative will
#                 be constructed, allowing to compute higher order derivative
#                 products. Defaults to ``False``.
#         """
#         torch.autograd.backward(self, gradient, retain_graph, create_graph)



for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
#         set_trace()
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        loss = loss_func(model(xb), yb)

        loss.backward()
        with torch.no_grad():
            for l in model.layers:
                if hasattr(l, 'weight'):
                    l.weight -= l.weight.grad * lr
                    l.bias   -= l.bias.grad   * lr
                    l.weight.grad.zero_()
                    l.bias  .grad.zero_()




loss_func(model(xb), yb), accuracy(model(xb), yb)


# ## Using parameters and optim

# ### Parameters

# Use `nn.Module.__setattr__` and move relu to functional:



class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in,nh)
        self.l2 = nn.Linear(nh,n_out)
        
    def __call__(self, x): 
        return self.l2(F.relu(self.l1(x)))




model = Model(m, nh, 10)




for name,l in model.named_children(): 
    print(f"{name}: {l}")




model




model.l1




def fit():
    for epoch in range(epochs):
        for i in range((n-1)//bs + 1):
            start_i = i*bs
            end_i = start_i+bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            loss = loss_func(model(xb), yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters(): 
                    p -= p.grad * lr
                model.zero_grad()




fit()
loss_func(model(xb), yb), accuracy(model(xb), yb)


# Behind the scenes, PyTorch overrides the `__setattr__` function in `nn.Module` so that the submodules you define are properly registered as parameters of the model.



class DummyModule():
    def __init__(self, n_in, nh, n_out):
        self._modules = {}
        self.l1 = nn.Linear(n_in,nh)
        self.l2 = nn.Linear(nh,n_out)
        
    def __setattr__(self,k,v):
        if not k.startswith("_"): 
            print(f'k: {k}, v: {v}')
            self._modules[k] = v
        else:
            print(f'not storing: k: {k}, v: {v}')
        super().__setattr__(k,v)
        
    def __repr__(self): 
        return f'{self._modules}'
    
    def parameters(self):
        for l in self._modules.values():
            for p in l.parameters(): 
                yield p




mdl = DummyModule(m,nh,10)
mdl




[o.shape for o in mdl.parameters()]


# ### Registering modules

# We can use the original `layers` approach, but we have to register the modules.



layers = [nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10)]




class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        for i,l in enumerate(self.layers): 
            self.add_module(f'layer_{i}', l)
        
    def __call__(self, x):
        for l in self.layers: 
            x = l(x)
        return x




model = Model(layers)




model


# ### nn.ModuleList

# `nn.ModuleList` does this for us.



class SequentialModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def __call__(self, x):
        for l in self.layers: 
            x = l(x)
        return x




model = SequentialModel(layers)




model




fit()
loss_func(model(xb), yb), accuracy(model(xb), yb)


# ### nn.Sequential

# `nn.Sequential` is a convenient class which does the same as the above:



model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))




fit()
loss_func(model(xb), yb), accuracy(model(xb), yb)




#nn.Sequential??




model


# ### optim

# Let's replace our previous manually coded optimization step:
# 
# ```python
# with torch.no_grad():
#     for p in model.parameters(): p -= p.grad * lr
#     model.zero_grad()
# ```
# 
# and instead use just:
# 
# ```python
# opt.step()
# opt.zero_grad()
# ```



class Optimizer():
    def __init__(self, params, lr=0.5):
        self.params,self.lr = list(params),lr
        
    def step(self):
        #see https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients
        with torch.no_grad():
            for p in self.params: 
                p -= p.grad * lr

    def zero_grad(self):
        for p in self.params: 
            p.grad.data.zero_()




model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))




opt = Optimizer(model.parameters())




for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()




loss,acc = loss_func(model(xb), yb), accuracy(model(xb), yb)
loss,acc


# PyTorch already provides this exact functionality in `optim.SGD` (it also handles stuff like momentum, which we'll look at later - except we'll be doing it in a more flexible way!)



#export
from torch import optim








def get_model():
    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
    return model, optim.SGD(model.parameters(), lr=lr)




xb.shape




yb.shape




model,opt = get_model()
loss_func(model(xb), yb)




for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()




loss,acc = loss_func(model(xb), yb), accuracy(model(xb), yb)
loss,acc


# Randomized tests can be very useful.



assert acc>0.7


# ## Dataset and DataLoader

# ### Dataset

# It's clunky to iterate through minibatches of x and y values separately:
# 
# ```python
#     xb = x_train[start_i:end_i]
#     yb = y_train[start_i:end_i]
# ```
# 
# Instead, let's do these two steps together, by introducing a `Dataset` class:
# 
# ```python
#     xb,yb = train_ds[i*bs : i*bs+bs]
# ```



#export
class Dataset():
    def __init__(self, x, y): 
        self.x,self.y = x,y
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self, i): 
        return self.x[i],self.y[i]




train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)
assert len(train_ds)==len(x_train)
assert len(valid_ds)==len(x_valid)




xb,yb = train_ds[0:5]
assert xb.shape==(5,28*28)
assert yb.shape==(5,)
xb,yb




model,opt = get_model()




for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
        xb,yb = train_ds[i*bs : i*bs+bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()




loss,acc = loss_func(model(xb), yb), accuracy(model(xb), yb)
assert acc>0.7
loss,acc


# ### DataLoader

# Previously, our loop iterated over batches (xb, yb) like this:
# 
# ```python
# for i in range((n-1)//bs + 1):
#     xb,yb = train_ds[i*bs : i*bs+bs]
#     ...
# ```
# 
# Let's make our loop much cleaner, using a data loader:
# 
# ```python
# for xb,yb in train_dl:
#     ...
# ```



class DataLoader():
    def __init__(self, ds, bs): 
        self.ds,self.bs = ds,bs
        
    def __iter__(self):
        for i in range(0, len(self.ds), self.bs): 
            yield self.ds[i:i+self.bs]




train_dl = DataLoader(train_ds, bs)
valid_dl = DataLoader(valid_ds, bs)




xb,yb = next(iter(valid_dl))
assert xb.shape==(bs,28*28)
assert yb.shape==(bs,)




plt.imshow(xb[0].view(28,28))
yb[0]




model,opt = get_model()




def fit():
    for epoch in range(epochs):
        for xb,yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()




fit()




loss,acc = loss_func(model(xb), yb), accuracy(model(xb), yb)
assert acc>0.7
loss,acc


# ### Random sampling

# We want our training set to be in a random order, and that order should differ each iteration. But the validation set shouldn't be randomized.



class Sampler():
    def __init__(self, ds, bs, shuffle=False):
        self.n,self.bs,self.shuffle = len(ds),bs,shuffle
        
    def __iter__(self):
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        
        for i in range(0, self.n, self.bs): 
            yield self.idxs[i:i+self.bs]




small_ds = Dataset(*train_ds[:10])




s = Sampler(small_ds,3,False)
[o for o in s]




s = Sampler(small_ds,3,True)
[o for o in s]




def collate(b):
    xs,ys = zip(*b)
    return torch.stack(xs),torch.stack(ys)

class DataLoader():
    def __init__(self, ds, sampler, collate_fn=collate):
        self.ds,self.sampler,self.collate_fn = ds,sampler,collate_fn
        
    def __iter__(self):
        for s in self.sampler: 
            yield self.collate_fn([self.ds[i] for i in s])




train_samp = Sampler(train_ds, bs, shuffle=True)
valid_samp = Sampler(valid_ds, bs, shuffle=False)




train_dl = DataLoader(train_ds, sampler=train_samp, collate_fn=collate)
valid_dl = DataLoader(valid_ds, sampler=valid_samp, collate_fn=collate)




xb,yb = next(iter(valid_dl))
plt.imshow(xb[0].view(28,28))
yb[0]




xb,yb = next(iter(train_dl))
plt.imshow(xb[0].view(28,28))
yb[0]




xb,yb = next(iter(train_dl))
plt.imshow(xb[0].view(28,28))
yb[0]




model,opt = get_model()
fit()

loss,acc = loss_func(model(xb), yb), accuracy(model(xb), yb)
assert acc>0.7
loss,acc


# ### PyTorch DataLoader



#export
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler




train_dl = DataLoader(train_ds, bs, sampler=RandomSampler(train_ds), collate_fn=collate)
valid_dl = DataLoader(valid_ds, bs, sampler=SequentialSampler(valid_ds), collate_fn=collate)




xb,yb = next(iter(train_dl))




model,opt = get_model()
fit()
loss_func(model(xb), yb), accuracy(model(xb), yb)


# PyTorch's defaults work fine for most things however:



train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, bs, shuffle=False)




xb,yb = next(iter(train_dl))




model,opt = get_model()
fit()

loss,acc = loss_func(model(xb), yb), accuracy(model(xb), yb)
assert acc>0.7
loss,acc


# Note that PyTorch's `DataLoader`, if you pass `num_workers`, will use multiple threads to call your `Dataset`.

# ## Validation

# You **always** should also have a [validation set](http://www.fast.ai/2017/11/13/validation-sets/), in order to identify if you are overfitting.
# 
# We will calculate and print the validation loss at the end of each epoch.
# 
# (Note that we always call `model.train()` before training, and `model.eval()` before inference, because these are used by layers such as `nn.BatchNorm2d` and `nn.Dropout` to ensure appropriate behaviour for these different phases.)



def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Handle batchnorm / dropout
        model.train()
#         print(model.training)
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
#         print(model.training)
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in valid_dl:
                pred = model(xb)
                tot_loss += loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv


# *Question*: Are these validation results correct if batch size varies?

# `get_dls` returns dataloaders for the training and validation sets:



#export
def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))


# Now, our whole process of obtaining the data loaders and fitting the model can be run in 3 lines of code:



train_dl,valid_dl = get_dls(train_ds, valid_ds, bs)
model,opt = get_model()
loss,acc = fit(5, model, loss_func, opt, train_dl, valid_dl)




assert acc>0.9


# ## Export



#!python notebook2script.py 03_l9_minibatch_training.ipynb







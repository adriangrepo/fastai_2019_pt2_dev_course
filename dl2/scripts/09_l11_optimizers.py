#!/usr/bin/env python
# coding: utf-8

# # Optimizer tweaks








#export
import time
import sys
from utils.nb_functions import *
from utils.nb_classes_l8_to_10 import *
from utils.nb_classes_l10_revised import *
from utils.nb_classes_cnn import *
from utils.nb_datablock import *
import utils.nb_classes_cnn as nb_classes_cnn
import utils.nb_datablock as nb_datablock




#set to false for setting breakpoints in debugger
#is really slow on the CPU, but useful if want to deeply follow the code
run_on_gpu=True

nb_classes_cnn.RUN_CNN_ON_GPU=True
nb_datablock.RUN_DATA_ON_GPU=True


# ## Imagenette data

# We grab the data from the previous notebook.



path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)




tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs=128

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)


# Then a model:



nfs = [32,64,128,256]




cbfs = [partial(AvgStatsCallback,accuracy), CudaCallback,
        partial(BatchTransformXCallback, norm_imagenette)]


# This is the baseline of training with vanilla SGD.



learn,run = get_learn_run(nfs, data, 0.4, conv_layer, cbs=cbfs)


# <pre>
# data is a DataBunch with properties:
#     c_in
#     c_out
#     train_dl
#     train_ds
#     valid_dl
#     valid_ds
# </pre>

# run_on_gpu: True, elapsed: 5.5087034702301025

# TODO
# 
# Debug why when run on CPU I get 
# ValueError: Expected input batch_size (128) to match target batch_size (94).
#     
# on run.fit(). Only difference afaik is .cpu() instead of .cuda()



start = time.time()
run.fit(1, learn)
end=time.time()
print(f'run_on_gpu: {run_on_gpu}, elapsed: {end-start}')


# ## Refining the optimizer

# In PyTorch, the base optimizer in `torch.optim` is just a dictionary that stores the hyper-parameters and references to the parameters of the model we want to train in parameter groups (different groups can have different learning rates/momentum/weight decay... which is what lets us do discriminative learning rates).
# 
# It contains a method `step` that will update our parameters with the gradients and a method `zero_grad` to detach and zero the gradients of all our parameters.
# 
# We build the equivalent from scratch, only ours will be more flexible. In our implementation, the step function loops over all the parameters to execute the step using stepper functions that we have to provide when initializing the optimizer.

# Lesson 11 1:09



class Optimizer():
    def __init__(self, params, steppers, **defaults):
        # might be a generator
        #parameter tensors = all weights and biases
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list): 
            self.param_groups = [self.param_groups]
            #eg lr, momentun, epsilon/beta in adam etc
        self.hypers = [{**defaults} for p in self.param_groups]
        print(f'self.hypers as defaults in params: {self.hypers}')
        self.steppers = listify(steppers)
        for i, f in enumerate(self.steppers):
            print(f'steppers {i}: {f.__name__}')

    def grad_params(self):
        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)
            for p in pg if p.grad is not None]

    def zero_grad(self):
        for p,hyper in self.grad_params():
            #remove gradient computation history
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p,hyper in self.grad_params(): compose(p, self.steppers, **hyper)


# To do basic SGD, this what a step looks like:



#export
def sgd_step(p, lr, **kwargs):
    #even though it looks like we are just adding, 
    #is actully doing p.data=p.data.add_(-lr*p.grad.data)
    #I spent an hour working this out, then Jenermy mentions this at 1:30:31 :-)
    p.data.add_(-lr, p.grad.data)
    #print(f'<<sgd_step p.data: {p.data}')
    return p


# Note within Optimizer we added a print to show what (hyper)params and steppers we pass in 
# Parameter group = Fastai layer group



opt_func = partial(Optimizer, steppers=[sgd_step])


# Now that we have changed the optimizer, we will need to adjust the callbacks that were using properties from the PyTorch optimizer: in particular the hyper-parameters are in the list of dictionaries `opt.hypers` (PyTorch has everything in the the list of param groups).



#export
class Recorder(Callback):
    def begin_fit(self): self.lrs,self.losses = [],[]

    def after_batch(self):
        if not self.in_train: 
            return
        self.lrs.append(self.opt.hypers[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())        

    def plot_lr  (self): 
        plt.plot(self.lrs)
    def plot_loss(self): 
        plt.plot(self.losses)
        
    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(self.lrs[:n], losses[:n])

class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_funcs):
        self.pname,self.sched_funcs = pname,listify(sched_funcs)

    def begin_batch(self): 
        if not self.in_train: 
            return
        fs = self.sched_funcs
        if len(fs)==1: fs = fs*len(self.opt.param_groups)
        pos = self.n_epochs/self.epochs
        for f,h in zip(fs,self.opt.hypers): h[self.pname] = f(pos)
            
class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9
        
    def begin_batch(self): 
        if not self.in_train: 
            return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.hypers: pg['lr'] = lr
            
    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss


# So let's check we didn't break anything and that recorder and param scheduler work properly.



sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)]) 




cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback, Recorder,
        partial(ParamScheduler, 'lr', sched)]




learn,run = get_learn_run(nfs, data, 0.4, conv_layer, cbs=cbfs, opt_func=opt_func)


# on cpu: elapsed: 101.67123055458069
# 
# on gpu: elapsed: 5.636744499206543



start = time.time()
run.fit(1, learn)
end = time.time()
print(f'elapsed: {end-start}')




run.recorder.plot_loss()




run.recorder.plot_lr()


# ## Weight decay

# By letting our model learn high parameters, it might fit all the data points in the training set with an over-complex function that has very sharp changes, which will lead to overfitting.
# 
# <img src="images/overfit.png" alt="Fitting vs over-fitting" width="600">
# 
# Weight decay comes from the idea of L2 regularization, which consists in adding to your loss function the sum of all the weights squared. Why do that? Because when we compute the gradients, it will add a contribution to them that will encourage the weights to be as small as possible.

# Limiting our weights from growing too much is going to hinder the training of the model, but it will yield to a state where it generalizes better. Going back to the theory a little bit, weight decay (or just `wd`) is a parameter that controls that sum of squares we add to our loss:
# ``` python
# loss_with_wd = loss + (wd/2) * (weights**2).sum()
# ```
# 
# In practice though, it would be very inefficient (and maybe numerically unstable) to compute that big sum and add it to the loss. If you remember a little bit of high school math, the derivative of `p**2` with respect to `p` is `2*p`. So adding that big sum to our loss is exactly the same as doing:
# ``` python
# weight.grad += wd * weight
# ```
# 
# for every weight in our model, which in the case of vanilla SGD is equivalent to updating the parameters with:
# ``` python
# weight = weight - lr*(weight.grad + wd*weight)
# ```
# 
# This technique is called "weight decay", as each weight is decayed by a factor `lr * wd`, as it's shown in this last formula.
# 
# This only works for standard SGD, as we have seen that with momentum, RMSProp and Adam, the update has some additional formulas around the gradient. In those cases, the formula that comes from L2 regularization:
# ``` python
# weight.grad += wd * weight
# ```
# is different than weight decay
# ``` python
# new_weight = weight - lr * weight.grad - lr * wd * weight
# ```
# 
# Most libraries use the first one, but as it was pointed out in [Decoupled Weight Regularization](https://arxiv.org/pdf/1711.05101.pdf) by Ilya Loshchilov and Frank Hutter, it is better to use the second one with the Adam optimizer, which is why fastai made it its default.

# Weight decay is subtracting `lr*wd*weight` from the weights. We need this function to have an attribute `_defaults` so that we are sure there is an hyper-parameter of the same name in our `Optimizer`.



#export
def weight_decay(p, lr, wd, **kwargs):
    p.data.mul_(1 - lr*wd)
    return p
weight_decay._defaults = dict(wd=0.)


# L2 regularization is adding `wd*weight` to the gradients.



#export
def l2_reg(p, lr, wd, **kwargs):
    p.grad.data.add_(wd, p.data)
    return p
l2_reg._defaults = dict(wd=0.)


# Let's allow steppers to add to our `defaults` (which are the default values of all the hyper-parameters). This helper function adds in `dest` the key/values it finds while going through `os` and applying `f` when they was no `key` of the same name.



#export
#only update if missing
def maybe_update(os, dest, f):
    for o in os:
        for k,v in f(o).items():
            if k not in dest: dest[k] = v

def get_defaults(d): 
    return getattr(d,'_defaults',{})


# This is the same as before, we just take the default values of the steppers when none are provided in the kwargs.



#export
class Optimizer():
    def __init__(self, params, steppers, **defaults):
        self.steppers = listify(steppers)
        maybe_update(self.steppers, defaults, get_defaults)
        # might be a generator
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)
            for p in pg if p.grad is not None]

    def zero_grad(self):
        for p,hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p,hyper in self.grad_params(): compose(p, self.steppers, **hyper)




#export 
sgd_opt = partial(Optimizer, steppers=[weight_decay, sgd_step])




cbfs




learn,run = get_learn_run(nfs, data, 0.4, conv_layer, cbs=cbfs, opt_func=sgd_opt)


# Before trying to train, let's check the behavior works as intended: when we don't provide a value for `wd`, we pull the corresponding default from `weight_decay`.



model = learn.model




opt = sgd_opt(model.parameters(), lr=0.1)
test_eq(opt.hypers[0]['wd'], 0.)
test_eq(opt.hypers[0]['lr'], 0.1)


# But if we provide a value, it overrides the default.



opt = sgd_opt(model.parameters(), lr=0.1, wd=1e-4)
test_eq(opt.hypers[0]['wd'], 1e-4)
test_eq(opt.hypers[0]['lr'], 0.1)


# Now let's fit.



cbfs = [partial(AvgStatsCallback,accuracy), CudaCallback, Recorder]




learn,run = get_learn_run(nfs, data, 0.3, conv_layer, cbs=cbfs, opt_func=partial(sgd_opt, wd=0.01))




run.fit(1, learn)




run.recorder.plot_loss()


# This is already better than the baseline!

# ## With momentum

# Momentum requires to add some state. We need to save the moving average of the gradients to be able to do the step and store this inside the optimizer state. To do this, we introduce statistics. Statistics are object with two methods:
# - `init_state`, that returns the initial state (a tensor of 0. for the moving average of gradients)
# - `update`, that updates the state with the new gradient value
# 
# We also read the `_defaults` values of those objects, to allow them to provide default values to hyper-parameters.



#export
class StatefulOptimizer(Optimizer):
    def __init__(self, params, steppers, stats=None, **defaults): 
        self.stats = listify(stats)
        maybe_update(self.stats, defaults, get_defaults)
        super().__init__(params, steppers, **defaults)
        self.state = {}
        
    def step(self):
        for p,hyper in self.grad_params():
            if p not in self.state:
                #Create a state for p and call all the statistics to initialize it.
                self.state[p] = {}
                maybe_update(self.stats, self.state[p], lambda o: o.init_state(p))
            state = self.state[p]
            for stat in self.stats: state = stat.update(p, state, **hyper)
            compose(p, self.steppers, **state, **hyper)
            self.state[p] = state
        #print(f'size of state: {sys.getsizeof(self.state)}')




#export
class Stat():
    _defaults = {}
    def init_state(self, p): 
        raise NotImplementedError
    def update(self, p, state, **kwargs): 
        raise NotImplementedError    


# Here is an example of `Stat`:



class AverageGrad(Stat):
    _defaults = dict(mom=0.9)

    def init_state(self, p): 
        return {'grad_avg': torch.zeros_like(p.grad.data)}
    def update(self, p, state, mom, **kwargs):
        state['grad_avg'].mul_(mom).add_(p.grad.data)
        return state


# Then we add the momentum step (instead of using the gradients to perform the step, we use the average).



#export
def momentum_step(p, lr, grad_avg, **kwargs):
    p.data.add_(-lr, grad_avg)
    return p




sgd_mom_opt = partial(StatefulOptimizer, steppers=[momentum_step,weight_decay],
                  stats=AverageGrad(), wd=0.01)




learn,run = get_learn_run(nfs, data, 0.3, conv_layer, cbs=cbfs, opt_func=sgd_mom_opt)




run.fit(1, learn)


# ### Momentum experiments

# What does momentum do to the gradients exactly? Let's do some plots to find out!



x = torch.linspace(-4, 4, 200)
#y is random, avg of 0.3
y = torch.randn(200) + 0.3
#momentums
betas = [0.5, 0.7, 0.9, 0.99]




def plot_mom(f):
    _,axs = plt.subplots(2,2, figsize=(12,8))
    for beta,ax in zip(betas, axs.flatten()):
        ax.plot(y, linestyle='None', marker='.')
        avg,res = None,[]
        for i,yi in enumerate(y):
            avg,p = f(avg, beta, yi, i)
            res.append(p)
        ax.plot(res, color='red')
        ax.set_title(f'beta={beta}')


# This is the regular momentum.



def mom1(avg, beta, yi, i): 
    if avg is None: 
        avg=yi
    #momentum function
    res = beta*avg + yi
    return res,res
plot_mom(mom1)


# As we can see, with a too high value for momentum, it may go way too high with no way to change its course- ie totally wrong.
# 
# Another way to smooth noisy data is to do an exponentially weighted moving average. In this case, there is a dampening of (1-beta) in front of the new value, which is less trusted than the current average. We'll define `lin_comb` (*linear combination*) to make this easier (note that in the lesson this was named `ewma`).



#export
#with dampening
def lin_comb(v1, v2, beta): 
    return beta*v1 + (1-beta)*v2




def mom2(avg, beta, yi, i):
    if avg is None: avg=yi
    avg = lin_comb(avg, yi, beta)
    return avg, avg
plot_mom(mom2)


# We can see it gets to a zero-constant when the data is purely random. If the data has a certain shape, it will get that shape (with some delay for high beta). In last image, item 2 is 0.99 * item1 + 0.01 * item2 - ie too much influence from item 1



y = 1 - (x/3) ** 2 + torch.randn(200) * 0.1




y[0]=0.5




plot_mom(mom2)


# Debiasing is here to correct the wrong information we may have in the very first batch. The debias term corresponds to the sum of the coefficient in our moving average. At the time step i, our average is:
# 
# $\begin{align*}
# avg_{i} &= \beta\ avg_{i-1} + (1-\beta)\ v_{i} = \beta\ (\beta\ avg_{i-2} + (1-\beta)\ v_{i-1}) + (1-\beta)\ v_{i} \\
# &= \beta^{2}\ avg_{i-2} + (1-\beta)\ \beta\ v_{i-1} + (1-\beta)\ v_{i} \\
# &= \beta^{3}\ avg_{i-3} + (1-\beta)\ \beta^{2}\ v_{i-2} + (1-\beta)\ \beta\ v_{i-1} + (1-\beta)\ v_{i} \\
# &\vdots \\
# &= (1-\beta)\ \beta^{i}\ v_{0} + (1-\beta)\ \beta^{i-1}\ v_{1} + \cdots + (1-\beta)\ \beta^{2}\ v_{i-2} + (1-\beta)\ \beta\  v_{i-1} + (1-\beta)\ v_{i}
# \end{align*}$
# 
# and so the sum of the coefficients is
# 
# $\begin{align*}
# S &=(1-\beta)\ \beta^{i} + (1-\beta)\ \beta^{i-1} + \cdots + (1-\beta)\ \beta^{2} + (1-\beta)\ \beta + (1-\beta) \\
# &= (\beta^{i} - \beta^{i+1}) + (\beta^{i-1} - \beta^{i}) + \cdots + (\beta^{2} - \beta^{3}) + (\beta - \beta^{2}) + (1-\beta) \\
# &= 1 - \beta^{i+1}
# \end{align*}$
# 
# since all the other terms cancel out each other.
# 
# By dividing by this term, we make our moving average a true average (in the sense that all the coefficients we used for the average sum up to 1).



def mom3(avg, beta, yi, i):
    if avg is None: avg=0
    avg = lin_comb(avg, yi, beta)
    return avg, avg/(1-beta**(i+1))
plot_mom(mom3)


# ## Adam and friends

# In Adam, we use the gradient averages but with dampening (not like in SGD with momentum), so let's add this to the `AverageGrad` class.
# 
# Adam = (Dampened debised momentum )/ (Dampened debiased root sum of squared gradients)



#export
class AverageGrad(Stat):
    _defaults = dict(mom=0.9)
    
    def __init__(self, dampening:bool=False): 
        self.dampening=dampening
    def init_state(self, p): 
        return {'grad_avg': torch.zeros_like(p.grad.data)}
    def update(self, p, state, mom, **kwargs):
        state['mom_damp'] = 1-mom if self.dampening else 1.
        state['grad_avg'].mul_(mom).add_(state['mom_damp'], p.grad.data)
        return state


# We also need to track the moving average of the gradients squared.



#export
class AverageSqrGrad(Stat):
    _defaults = dict(sqr_mom=0.99)
    
    def __init__(self, dampening:bool=True): 
        self.dampening=dampening
    def init_state(self, p): 
        return {'sqr_avg': torch.zeros_like(p.grad.data)}
    def update(self, p, state, sqr_mom, **kwargs):
        state['sqr_damp'] = 1-sqr_mom if self.dampening else 1.
        state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data, p.grad.data)
        return state


# We will also need the number of steps done during training for the debiasing.



#export
class StepCount(Stat):
    def init_state(self, p): 
        return {'step': 0}
    def update(self, p, state, **kwargs):
        state['step'] += 1
        return state


# This helper function computes the debias term. If we dampening, `damp = 1 - mom` and we get the same result as before. If we don't use dampening, (`damp = 1`) we will need to divide by `1 - mom` because that term is missing everywhere.



#export
def debias(mom, damp, step): return damp * (1 - mom**step) / (1-mom)


# Then the Adam step is just the following:



#export
def adam_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs):
    debias1 = debias(mom,     mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg/debias2).sqrt() + eps)
    return p
adam_step._defaults = dict(eps=1e-5)




#export
def adam_opt(xtra_step=None, **kwargs):
    return partial(StatefulOptimizer, steppers=[adam_step,weight_decay]+listify(xtra_step),
                   stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()], **kwargs)




learn,run = get_learn_run(nfs, data, 0.001, conv_layer, cbs=cbfs, opt_func=adam_opt())




run.fit(3, learn)


# ## LAMB

# It's then super easy to implement a new optimizer. This is LAMB from a [very recent paper](https://arxiv.org/pdf/1904.00962.pdf):
# 
# $\begin{align}
# g_{t}^{l} &= \nabla L(w_{t-1}^{l}, x_{t}) \\
# m_{t}^{l} &= \beta_{1} m_{t-1}^{l} + (1-\beta_{1}) g_{t}^{l} \\
# v_{t}^{l} &= \beta_{2} v_{t-1}^{l} + (1-\beta_{2}) g_{t}^{l} \odot g_{t}^{l} \\
# m_{t}^{l} &= m_{t}^{l} / (1 - \beta_{1}^{t}) \\
# v_{t}^{l} &= v_{t}^{l} / (1 - \beta_{2}^{t}) \\
# r_{1} &= \|w_{t-1}^{l}\|_{2} \\
# s_{t}^{l} &= \frac{m_{t}^{l}}{\sqrt{v_{t}^{l} + \epsilon}} + \lambda w_{t-1}^{l} \\ 
# r_{2} &= \| s_{t}^{l} \|_{2} \\
# \eta^{l} &= \eta * r_{1}/r_{2} \\ 
# w_{t}^{l} &= w_{t}^{l-1} - \eta_{l} * s_{t}^{l} \\
# \end{align}$



def lamb_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, wd, **kwargs):
    debias1 = debias(mom,     mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    r1 = p.data.pow(2).mean().sqrt()
    step = (grad_avg/debias1) / ((sqr_avg/debias2).sqrt()+eps) + wd*p.data
    r2 = step.pow(2).mean().sqrt()
    p.data.add_(-lr * min(r1/r2,10), step)
    return p
lamb_step._defaults = dict(eps=1e-6, wd=0.)




lamb = partial(StatefulOptimizer, steppers=lamb_step, stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()])




learn,run = get_learn_run(nfs, data, 0.003, conv_layer, cbs=cbfs, opt_func=lamb)




run.fit(3, learn)


# Other recent variants of optimizers:
# - [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888) (LARS also uses weight statistics, not just gradient statistics. Can you add that to this class?)
# - [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235) (Adafactor combines stats over multiple sets of axes)
# - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://arxiv.org/abs/1902.09843)

# ## Export



#!python notebook2script.py 09_optimizers.ipynb







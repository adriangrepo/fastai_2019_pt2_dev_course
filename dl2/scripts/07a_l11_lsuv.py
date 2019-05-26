#!/usr/bin/env python
# coding: utf-8








#export
from utils.nb_functions import *
from utils.nb_classes_l8_to_10 import *
from utils.nb_classes_l10_revised import *
from utils.nb_classes_cnn import *




#set to false for setting breakpoints in debugger
run_on_gpu=True


# ## Layerwise Sequential Unit Variance (LSUV)

# Getting the MNIST data and a CNN



x_train,y_train,x_valid,y_valid = get_data()

x_train,x_valid = normalize_to(x_train,x_valid)
train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)

nh,bs = 50,512
c = y_train.max().item()+1
loss_func = F.cross_entropy

data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)




mnist_view = view_tfm(1,28,28)
cbfs = [Recorder,
        partial(AvgStatsCallback,accuracy),
        CudaCallback,
        partial(BatchTransformXCallback, mnist_view)]




#create these layers with this number of filters
nfs = [8,16,32,64,64]


# See All you need is a good init paper
# Here we put a bias and weight method in the conv layer



class ConvLayer(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=2, sub=0., **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=True)
        self.relu = GeneralRelu(sub=sub, **kwargs)
    
    def forward(self, x): 
        return self.relu(self.conv(x))
    
    @property
    def bias(self): 
        return -self.relu.sub
    @bias.setter
    def bias(self,v): 
        self.relu.sub = -v
    @property
    def weight(self): 
        return self.conv.weight




learn,run = get_learn_run(nfs, data, 0.6, ConvLayer, cbs=cbfs)


# Now we're going to look at the paper [All You Need is a Good Init](https://arxiv.org/pdf/1511.06422.pdf), which introduces *Layer-wise Sequential Unit-Variance* (*LSUV*). We initialize our neural net with the usual technique, then we pass a batch through the model and check the outputs of the linear and convolutional layers. We can then rescale the weights according to the actual variance we observe on the activations, and subtract the mean we observe from the initial bias. That way we will have activations that stay normalized.
# 
# We repeat this process until we are satisfied with the mean/variance we observe.
# 
# Let's start by looking at a baseline:



run.fit(2, learn)


# Now we recreate our model and we'll try again with LSUV. Hopefully, we'll get better results!



learn,run = get_learn_run(nfs, data, 0.6, ConvLayer, cbs=cbfs)


# Helper function to get one batch of a given dataloader, with the callbacks called to preprocess it.



#export
def get_batch(dl, run):
    run.xb,run.yb = next(iter(dl))
    for cb in run.cbs: 
        cb.set_runner(run)
    run('begin_batch')
    return run.xb,run.yb




xb,yb = get_batch(data.train_dl, run)


# We only want the outputs of convolutional or linear layers. To find them, we need a recursive function. We can use `sum(list, [])` to concatenate the lists the function finds (`sum` applies the + operate between the elements of the list you pass it, beginning with the initial state in the second argument).



#export
def find_modules(m, cond):
    if cond(m): return [m]
    return sum([find_modules(o,cond) for o in m.children()], [])

def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU)
    return isinstance(l, lin_layers)




mods = find_modules(learn.model, lambda o: isinstance(o,ConvLayer))




print(mods)


# This is a helper function to grab the mean and std of the output of a hooked layer.



def append_stat(hook, mod, inp, outp):
    #model: ConvLayer, inp: tuple of data
    d = outp.data
    hook.mean,hook.std = d.mean().item(),d.std().item()




if run_on_gpu:
    mdl = learn.model.cuda()
else:
    mdl = learn.model


# So now we can look at the mean and std of the conv layers of our model.



with Hooks(mods, append_stat) as hooks:
    mdl(xb)
    for hook in hooks: 
        print(hook.mean,hook.std)


# Note above means and std dev are not 0 and 1
# 
# We first adjust the bias terms to make the means 0, then we adjust the standard deviations to make the stds 1 (with a threshold of 1e-3). The `mdl(xb) is not None` clause is just there to pass `xb` through `mdl` and compute all the activations so that the hooks get updated. 
# 
# where xb is our minibatch

# Also note this is run on model pre-training



#export
def lsuv_module(m, xb):
    #register_forward_hook for the append_stat function
    #h is then a Hook class
    h = Hook(m, append_stat)
    #see append_stat() function above - thats where we set the .mean and .std properties for Hook
    print(f'h.mean: {h.mean}')
    while mdl(xb) is not None and abs(h.mean)  > 1e-3: 
        m.bias -= h.mean
    while mdl(xb) is not None and abs(h.std-1) > 1e-3: 
        m.weight.data /= h.std
    #remove the hook
    h.remove()
    return h.mean,h.std


# We execute that initialization on all the conv layers in order:



for m in mods: print(lsuv_module(m, xb))


# Note that the mean doesn't exactly stay at 0. since we change the standard deviation after by scaling the weight.

# Then training is beginning on better grounds.





# LSUV is particularly useful for more complex and deeper architectures that are hard to initialize to get unit variance at the last layer.

# ## Export



#!python notebook2script.py 07a_lsuv.ipynb







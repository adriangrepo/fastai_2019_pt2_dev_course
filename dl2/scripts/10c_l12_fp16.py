#!/usr/bin/env python
# coding: utf-8

# # Training in mixed precision








from utils.nb_functions import *
from utils.nb_classes_cnn import *
from utils.nb_classes_l8_to_10 import *
from utils.nb_classes_l10_revised import *
from utils.nb_learner import *
from utils.nb_optimizer import *
from utils.nb_progress import *
from utils.nb_mixup import *
from torch.nn.utils import parameters_to_vector


# ## A little bit of theory

# Continuing the documentation on the fastai_v1 development here is a brief piece about mixed precision training. A very nice and clear introduction to it is [this video from NVIDIA](http://on-demand.gputechconf.com/gtc/2018/video/S81012/).
# 
# ### What's half precision?
# In neural nets, all the computations are usually done in single precision, which means all the floats in all the arrays that represent inputs, activations, weights... are 32-bit floats (FP32 in the rest of this post). An idea to reduce memory usage (and avoid those annoying cuda errors) has been to try and do the same thing in half-precision, which means using 16-bits floats (or FP16 in the rest of this post). By definition, they take half the space in RAM, and in theory could allow you to double the size of your model and double your batch size.
# 
# Another very nice feature is that NVIDIA developed its latest GPUs (the Volta generation) to take fully advantage of half-precision tensors. Basically, if you give half-precision tensors to those, they'll stack them so that each core can do more operations at the same time, and theoretically gives an 8x speed-up (sadly, just in theory).
# 
# So training at half precision is better for your memory usage, way faster if you have a Volta GPU (still a tiny bit faster if you don't since the computations are easiest). How do we do it? Super easily in pytorch, we just have to put .half() everywhere: on the inputs of our model and all the parameters. Problem is that you usually won't see the same accuracy in the end (so it happens sometimes) because half-precision is... well... not as precise ;).
# 
# ### Problems with half-precision:
# To understand the problems with half precision, let's look briefly at what an FP16 looks like (more information [here](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)).
# 
# ![half float](images/half.png)
# 
# The sign bit gives us +1 or -1, then we have 5 bits to code an exponent between -14 and 15, while the fraction part has the remaining 10 bits. Compared to FP32, we have a smaller range of possible values (2e-14 to 2e15 roughly, compared to 2e-126 to 2e127 for FP32) but also a smaller *offset*.
# 
# For instance, between 1 and 2, the FP16 format only represents the number 1, 1+2e-10, 1+2*2e-10... which means that 1 + 0.0001 = 1 in half precision. That's what will cause a certain numbers of problems, specifically three that can occur and mess up your training.
# 1. The weight update is imprecise: inside your optimizer, you basically do w = w - lr * w.grad for each weight of your network. The problem in performing this operation in half precision is that very often, w.grad is several orders of magnitude below w, and the learning rate is also small. The situation where w=1 and lr*w.grad is 0.0001 (or lower) is therefore very common, but the update doesn't do anything in those cases.
# 2. Your gradients can underflow. In FP16, your gradients can easily be replaced by 0 because they are too low.
# 3. Your activations or loss can overflow. The opposite problem from the gradients: it's easier to hit nan (or infinity) in FP16 precision, and your training might more easily diverge.
# 
# ### The solution: mixed precision training
# 
# To address those three problems, we don't fully train in FP16 precision. As the name mixed training implies, some of the operations will be done in FP16, others in FP32. This is mainly to take care of the first problem listed above. For the next two there are additional tricks.
# 
# The main idea is that we want to do the forward pass and the gradient computation in half precision (to go fast) but the update in single precision (to be more precise). It's okay if w and grad are both half floats, but when we do the operation w = w - lr * grad, we need to compute it in FP32. That way our 1 + 0.0001 is going to be 1.0001. 
# 
# This is why we keep a copy of the weights in FP32 (called master model). Then, our training loop will look like:
# 1. compute the output with the FP16 model, then the loss
# 2. back-propagate the gradients in half-precision.
# 3. copy the gradients in FP32 precision
# 4. do the update on the master model (in FP32 precision)
# 5. copy the master model in the FP16 model.
# 
# Note that we lose precision during step 5, and that the 1.0001 in one of the weights will go back to 1. But if the next update corresponds to add 0.0001 again, since the optimizer step is done on the master model, the 1.0001 will become 1.0002 and if we eventually go like this up to 1.0005, the FP16 model will be able to tell the difference.
# 
# That takes care of problem 1. For the second problem, we use something called gradient scaling: to avoid the gradients getting zeroed by the FP16 precision, we multiply the loss by a scale factor (scale=512 for instance). That way we can push the gradients to the right in the next figure, and have them not become zero.
# 
# ![half float representation](images/half_representation.png)
# 
# Of course we don't want those 512-scaled gradients to be in the weight update, so after converting them into FP32, we can divide them by this scale factor (once they have no risks of becoming 0). This changes the loop to:
# 1. compute the output with the FP16 model, then the loss.
# 2. multiply the loss by scale then back-propagate the gradients in half-precision.
# 3. copy the gradients in FP32 precision then divide them by scale.
# 4. do the update on the master model (in FP32 precision).
# 5. copy the master model in the FP16 model.
# 
# For the last problem, the tricks offered by NVIDIA are to leave the batchnorm layers in single precision (they don't have many weights so it's not a big memory challenge) and compute the loss in single precision (which means converting the last output of the model in single precision before passing it to the loss).
# 
# ![Mixed precision training](images/Mixed_precision.jpeg)
# 
# Implementing all of this in the new callback system is surprisingly easy, let's dig into this!

# ## Util functions

# Before going in the main `Callback` we will need some helper functions. We will refactor using the [APEX library](https://github.com/NVIDIA/apex) util functions. The python-only build is enough for what we will use here if you don't manage to do the CUDA/C++ installation.



# export 
import apex.fp16_utils as fp16


# ### Converting the model to FP16

# We will need a function to convert all the layers of the model to FP16 precision except the BatchNorm-like layers (since those need to be done in FP32 precision to be stable). We do this in two steps: first we convert the model to FP16, then we loop over all the layers and put them back to FP32 if they are a BatchNorm layer.



bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)




def bn_to_float(model):
    if isinstance(model, bn_types): model.float()
    for child in model.children():  bn_to_float(child)
    return model




def model_to_half(model):
    model = model.half()
    return bn_to_float(model)


# Let's test this:



model = nn.Sequential(nn.Linear(10,30), nn.BatchNorm1d(30), nn.Linear(30,2)).cuda()
model = model_to_half(model)




def check_weights(model):
    for i,t in enumerate([torch.float16, torch.float32, torch.float16]):
        assert model[i].weight.dtype == t
        assert model[i].bias.dtype   == t




check_weights(model)


# In Apex, the function that does this for us is `convert_network`. We can use it to put the model in FP16 or back to FP32.



model = nn.Sequential(nn.Linear(10,30), nn.BatchNorm1d(30), nn.Linear(30,2)).cuda()
model = fp16.convert_network(model, torch.float16)
check_weights(model)


# ### Creating the master copy of the parameters

# From our model parameters (mostly in FP16), we'll want to create a copy in FP32 (master parameters) that we will use for the step in the optimizer. Optionally, we concatenate all the parameters to do one flat big tensor, which can make that step a little bit faster.



from torch.nn.utils import parameters_to_vector

def get_master(model, flat_master=False):
    model_params = [param for param in model.parameters() if param.requires_grad]
    if flat_master:
        master_param = parameters_to_vector([param.data.float() for param in model_params])
        master_param = torch.nn.Parameter(master_param, requires_grad=True)
        if master_param.grad is None: master_param.grad = master_param.new(*master_param.size())
        return model_params, [master_param]
    else:
        master_params = [param.clone().float().detach() for param in model_params]
        for param in master_params: param.requires_grad_(True)
        return model_params, master_params


# The util function from Apex to do this is `prep_param_lists`.



model_p,master_p = get_master(model)
model_p1,master_p1 = fp16.prep_param_lists(model)




def same_lists(ps1, ps2):
    assert len(ps1) == len(ps2)
    for (p1,p2) in zip(ps1,ps2): 
        assert p1.requires_grad == p2.requires_grad
        assert torch.allclose(p1.data.float(), p2.data.float())




same_lists(model_p,model_p1)
same_lists(model_p,master_p)
same_lists(master_p,master_p1)
same_lists(model_p1,master_p1)


# We can't use flat_master when there is a mix of FP32 and FP16 parameters (like batchnorm here).



model1 = nn.Sequential(nn.Linear(10,30), nn.Linear(30,2)).cuda()
model1 = fp16.convert_network(model1, torch.float16)




model_p,master_p = get_master(model1, flat_master=True)
model_p1,master_p1 = fp16.prep_param_lists(model1, flat_master=True)




same_lists(model_p,model_p1)
same_lists(master_p,master_p1)




assert len(master_p[0]) == 10*30 + 30 + 30*2 + 2
assert len(master_p1[0]) == 10*30 + 30 + 30*2 + 2


# The thing is that we don't always want all the parameters of our model in the same parameter group, because we might:
# - want to do transfer learning and freeze some layers
# - apply discriminative learning rates
# - don't apply weight decay to some layers (like BatchNorm) or the bias terms
# 
# So we actually need a function that splits the parameters of an optimizer (and not a model) according to the right parameter groups.



def get_master(opt, flat_master=False):
    model_params = [[param for param in pg if param.requires_grad] for pg in opt.param_groups]
    if flat_master:
        master_params = []
        for pg in model_params:
            mp = parameters_to_vector([param.data.float() for param in pg])
            mp = torch.nn.Parameter(mp, requires_grad=True)
            if mp.grad is None: mp.grad = mp.new(*mp.size())
            master_params.append(mp)
    else:
        master_params = [[param.clone().float().detach() for param in pg] for pg in model_params]
        for pg in master_params:
            for param in pg: param.requires_grad_(True)
    return model_params, master_params


# ### Copy the gradients from model params to master params

# After the backward pass, all gradients must be copied to the master params before the optimizer step can be done in FP32. We need a function for that (with a bit of adjustement if we have flat master).



def to_master_grads(model_params, master_params, flat_master:bool=False)->None:
    if flat_master:
        if master_params[0].grad is None: master_params[0].grad = master_params[0].data.new(*master_params[0].data.size())
        master_params[0].grad.data.copy_(parameters_to_vector([p.grad.data.float() for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None: master.grad = master.data.new(*master.data.size())
                master.grad.data.copy_(model.grad.data)
            else: master.grad = None


# The corresponding function in the Apex utils is `model_grads_to_master_grads`.



x = torch.randn(20,10).half().cuda()
z = model(x)
loss = F.cross_entropy(z, torch.randint(0, 2, (20,)).cuda())
loss.backward()




to_master_grads(model_p, master_p)




def check_grads(m1, m2):
    for p1,p2 in zip(m1,m2): 
        if p1.grad is None: assert p2.grad is None
        else: assert torch.allclose(p1.grad.data, p2.grad.data) 




check_grads(model_p, master_p)




fp16.model_grads_to_master_grads(model_p, master_p)




check_grads(model_p, master_p)


# ### Copy the master params to the model params

# After the step, we need to copy back the master parameters to the model parameters for the next update.



from torch._utils import _unflatten_dense_tensors

def to_model_params(model_params, master_params, flat_master:bool=False)->None:
    if flat_master:
        for model, master in zip(model_params, _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params): model.data.copy_(master.data)


# The corresponding function in Apex is `master_params_to_model_params`.

# ### But we need to handle param groups

# The thing is that we don't always want all the parameters of our model in the same parameter group, because we might:
# - want to do transfer learning and freeze some layers
# - apply discriminative learning rates
# - don't apply weight decay to some layers (like BatchNorm) or the bias terms
# 
# So we actually need a function that splits the parameters of an optimizer (and not a model) according to the right parameter groups and the following functions need to handle lists of lists of parameters (one list of each param group in `model_pgs` and `master_pgs`)



# export 
def get_master(opt, flat_master=False):
    model_pgs = [[param for param in pg if param.requires_grad] for pg in opt.param_groups]
    if flat_master:
        master_pgs = []
        for pg in model_pgs:
            mp = parameters_to_vector([param.data.float() for param in pg])
            mp = torch.nn.Parameter(mp, requires_grad=True)
            if mp.grad is None: mp.grad = mp.new(*mp.size())
            master_pgs.append([mp])
    else:
        master_pgs = [[param.clone().float().detach() for param in pg] for pg in model_pgs]
        for pg in master_pgs:
            for param in pg: param.requires_grad_(True)
    return model_pgs, master_pgs




# export 
def to_master_grads(model_pgs, master_pgs, flat_master:bool=False)->None:
    for (model_params,master_params) in zip(model_pgs,master_pgs):
        fp16.model_grads_to_master_grads(model_params, master_params, flat_master=flat_master)




# export 
def to_model_params(model_pgs, master_pgs, flat_master:bool=False)->None:
    for (model_params,master_params) in zip(model_pgs,master_pgs):
        fp16.master_params_to_model_params(model_params, master_params, flat_master=flat_master)


# ## The main Callback



class MixedPrecision(Callback):
    _order = 99
    #use loss_scale multiplier to increate often very small loss values
    def __init__(self, loss_scale=512, flat_master=False):
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.loss_scale,self.flat_master = loss_scale,flat_master

    def begin_fit(self):
        self.run.model = fp16.convert_network(self.model, dtype=torch.float16)
        self.model_pgs, self.master_pgs = get_master(self.opt, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        self.run.opt.param_groups = self.master_pgs #Put those param groups inside our runner.
        
    def after_fit(self): 
        self.model.float()

    def begin_batch(self): 
        self.run.xb = self.run.xb.half() #Put the inputs to half precision
    def after_pred(self):  
        self.run.pred = self.run.pred.float() #Compute the loss in FP32
    def after_loss(self):  
        self.run.loss *= self.loss_scale #Loss scaling to avoid gradient underflow

    def after_backward(self):
        #Copy the gradients to master and unscale
        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params:
                if param.grad is not None: 
                    param.grad.div_(self.loss_scale)

    def after_step(self):
        #Zero the gradients of the model since the optimizer is disconnected.
        self.model.zero_grad()
        #Update the params from master to model.
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)


# Now let's test this on Imagenette



path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)




tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 64

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)




nfs = [32,64,128,256,512]


# Training without mixed precision



cbfs = [partial(AvgStatsCallback,accuracy),
        ProgressCallback,
        CudaCallback,
        partial(BatchTransformXCallback, norm_imagenette)]


# With adam_opt as a function I couldn't call .opt() on in in fit, created adm_opt variable



learn = get_learner(nfs, data, 1e-2, conv_layer, cb_funcs=cbfs, opt_func=adm_opt)




learn.fit(1)


# Training with mixed precision



cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback,
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette),
        MixedPrecision]




learn = get_learner(nfs, data, 1e-2, conv_layer, cb_funcs=cbfs)




learn.fit(1)




test_eq(next(learn.model.parameters()).type(), 'torch.cuda.FloatTensor')


# ## Dynamic loss scaling

# The only annoying thing with the previous implementation of mixed precision training is that it introduces one new hyper-parameter to tune, the value of the loss scaling. Fortunately for us, there is a way around this. We want the loss scaling to be as high as possible so that our gradients can use the whole range of representation, so let's first try a really high value. In all likelihood, this will cause our gradients or our loss to overflow, and we will try again with half that big value, and again, until we get to the largest loss scale possible that doesn't make our gradients overflow.
# 
# This value will be perfectly fitted to our model and can continue to be dynamically adjusted as the training goes, if it's still too high, by just halving it each time we overflow. After a while though, training will converge and gradients will start to get smaller, so we also need a mechanism to get this dynamic loss scale larger if it's safe to do so. The strategy used in the Apex library is to multiply the loss scale by 2 each time we had a given number of iterations without overflowing.
# 
# To check if the gradients have overflowed, we check their sum (computed in FP32). If one term is nan, the sum will be nan. Interestingly, on the GPU, it's faster than checking `torch.isnan`:



# export 
def test_overflow(x):
    s = float(x.float().sum())
    return (s == float('inf') or s == float('-inf') or s != s)




x = torch.randn(512,1024).cuda()




test_overflow(x)




x[123,145] = float('inf')
test_overflow(x)










# So we can use it in the following function that checks for gradient overflow:



# export 
def grad_overflow(param_groups):
    for group in param_groups:
        for p in group:
            if p.grad is not None:
                s = float(p.grad.data.float().sum())
                if s == float('inf') or s == float('-inf') or s != s: return True
    return False


# And now we can write a new version of the `Callback` that handles dynamic loss scaling.



# export 
class MixedPrecision(Callback):
    _order = 99
    def __init__(self, loss_scale=512, flat_master=False, dynamic=True, max_loss_scale=2.**24, div_factor=2.,
                 scale_wait=500):
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.flat_master,self.dynamic,self.max_loss_scale = flat_master,dynamic,max_loss_scale
        self.div_factor,self.scale_wait = div_factor,scale_wait
        self.loss_scale = max_loss_scale if dynamic else loss_scale

    def begin_fit(self):
        self.run.model = fp16.convert_network(self.model, dtype=torch.float16)
        self.model_pgs, self.master_pgs = get_master(self.opt, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        self.run.opt.param_groups = self.master_pgs #Put those param groups inside our runner.
        if self.dynamic: self.count = 0

    def begin_batch(self): self.run.xb = self.run.xb.half() #Put the inputs to half precision
    def after_pred(self):  self.run.pred = self.run.pred.float() #Compute the loss in FP32
    def after_loss(self):  
        if self.in_train: self.run.loss *= self.loss_scale #Loss scaling to avoid gradient underflow

    def after_backward(self):
        #First, check for an overflow
        if self.dynamic and grad_overflow(self.model_pgs):
            #Divide the loss scale by div_factor, zero the grad (after_step will be skipped)
            self.loss_scale /= self.div_factor
            self.model.zero_grad()
            return True #skip step and zero_grad
        #Copy the gradients to master and unscale
        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params:
                if param.grad is not None: param.grad.div_(self.loss_scale)
        #Check if it's been long enough without overflow
        if self.dynamic:
            self.count += 1
            if self.count == self.scale_wait:
                self.count = 0
                self.loss_scale *= self.div_factor

    def after_step(self):
        #Zero the gradients of the model since the optimizer is disconnected.
        self.model.zero_grad()
        #Update the params from master to model.
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)




cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback,
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette),
        MixedPrecision]




learn = get_learner(nfs, data, 1e-2, conv_layer, cb_funcs=cbfs)




learn.fit(1)


# The loss scale used is way higher than our previous number:



learn.cbs[-1].loss_scale


# ## Export



#!./notebook2script.py 10c_fp16.ipynb







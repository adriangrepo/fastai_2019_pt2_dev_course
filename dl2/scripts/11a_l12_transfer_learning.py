#!/usr/bin/env python
# coding: utf-8








#import os; os.environ['CUDA_VISIBLE_DEVICES']='0, 1 2'

#export
import time
from exp.nb_formatted import *




#set to true on first run
RETRAIN=False


# ## Serializing the model

# Store on ssd rather than in home folder



path = datasets.untar_data(datasets.URLs.IMAGEWOOF_160, dest='data')




size = 128
bs = 64
#bs = 512

tfms = [make_rgb, RandomResizedCrop(size, scale=(0.35,1)), np_to_float, PilRandomFlip()]
val_tfms = [make_rgb, CenterCrop(size), np_to_float]
il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
ll.valid.x.tfms = val_tfms
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=8)




len(il)




loss_func = LabelSmoothingCrossEntropy()
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)


# Using imagenette norm on imagewoof



learn = cnn_learner(arch=xresnet18, data=data, loss_func=loss_func, opt_func=opt_func, norm=norm_imagenette)




def sched_1cycle(lr, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    phases = create_phases(pct_start)
    sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler('lr', sched_lr),
            ParamScheduler('mom', sched_mom)]




lr = 3e-3
pct_start = 0.5
cbsched = sched_1cycle(lr, pct_start)


# save out model so can use with pets

# 1 x 2080ti, bs=256, epochs=10, elapsed: 115.17198395729065
# 
# 2 x 2080ti, bs=512, epochs=10, elapsed: 79.39946436882019



mdl_path = path/'models'
mdl_path.mkdir(exist_ok=True)

if RETRAIN:
    start=time.time()
    learn.model = torch.nn.DataParallel(learn.model, device_ids=[0,1, 2])
    learn.fit(500, cbsched)
    end=time.time()
    print(f'elapsed: {end-start}')
    st = learn.model.state_dict()

    print(type(st))

    #keys are names of the layers
    print(', '.join(st.keys()))
    print(path)

    #It's also possible to save the whole model, including the architecture, 
    #but it gets quite fiddly and we don't recommend it. 
    #Instead, just save the parameters, and recreate the model directly.

    #torch.save(st, mdl_path/'iw5')
else:
    print('Loading pre-trained model weights')
    learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
    st = learn.model.state_dict()
    print(st['10.bias'])


# ## Pets



pets = datasets.untar_data(datasets.URLs.PETS, dest='data')




pets.ls()




pets_path = pets/'images'




il = ImageList.from_files(pets_path, tfms=tfms)




il


# We dont have a sapratae validation directory so randomly grab val samples



#export
def random_splitter(fn, p_valid): return random.random() < p_valid




random.seed(42)




sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.1))




sd


# Now need to label - use filenames as cant use folders



n = il.items[0].name; n




re.findall(r'^(.*)_\d+.jpg$', n)[0]




def pet_labeler(fn): 
    return re.findall(r'^(.*)_\d+.jpg$', fn.name)[0]


# Use CategoryProcessor from last week



proc = CategoryProcessor()




ll = label_by_func(sd, pet_labeler, proc_y=proc)




', '.join(proc.vocab)




ll.valid.x.tfms = val_tfms




c_out = len(proc.vocab)




data = ll.to_databunch(bs, c_in=3, c_out=c_out, num_workers=8)




learn = cnn_learner(xresnet18, data, loss_func, opt_func, norm=norm_imagenette)




if RETRAIN:
    learn.fit(5, cbsched)


# <pre>
# epoch 	train_loss 	train_accuracy 	valid_loss 	valid_accuracy 	time
# 0 	3.467661 	0.083246 	3.488343 	0.095436 	00:07
# 1 	3.303485 	0.126594 	3.571582 	0.112033 	00:07
# 2 	3.123925 	0.179691 	3.204137 	0.159059 	00:07
# 3 	2.826835 	0.265037 	2.929686 	0.250346 	00:07
# 4 	2.544567 	0.360582 	2.638438 	0.355463 	00:07
# </pre>

# ## Custom head

# Poor result above, here we read in imagewoof model and customise for pets. Create 10 activations at end.
# 
# Also addpend Recorder callback to access loss



learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette, xtra_cb=Recorder)




mdl_path




st = torch.load(mdl_path/'iw5')




st.keys()




m = learn.model




mst = m.state_dict()




mst.keys()




#check that keys are same in both dicts




[k for k in st.keys() if k not in mst.keys()]




m.load_state_dict(st)




#want to remove last layer as have different ammount of categries - here 37 pet breeds. Find the AdaptiveAvgPool2d layer and use everything before this




m




cut = next(i for i,o in enumerate(m.children()) if isinstance(o,nn.AdaptiveAvgPool2d))
m_cut = m[:cut]




len(m_cut)




xb,yb = get_batch(data.valid_dl, learn)




pred = m_cut(xb)




pred.shape


# To find number of inputs, here we have 128 minibatch of input size 512, 4x4



#number of inputs to our head
ni = pred.shape[1]


# Note we use both Avg and Max pool and concatenate them together eg
# https://www.cs.cmu.edu/~tzhi/publications/ICIP2016_two_stage.pdf



#export
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)




nh = 40


# Allow hook to access internals of XResNet by unpacking arg list



m_new = nn.Sequential(
    *m_cut.children(), AdaptiveConcatPool2d(), Flatten(),
    nn.Linear(ni*2, data.c_out))




#m_new = nn.Sequential(
#    m_cut, AdaptiveConcatPool2d(), Flatten(),
#    nn.Linear(ni*2, data.c_out))




learn.model = m_new


# When we ran telemetry previously in lesson 10 (nb 06), were were using a manually built model with 7 layers:
# 
# <pre>
#     Sequential(
#   (0): Sequential(
#     (0): Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#     (1): ReLU()
#   )
#   (1): Sequential(
#     (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     (1): ReLU()
#   )
#   (2): Sequential(
#     (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     (1): ReLU()
#   )
#   (3): Sequential(
#     (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     (1): ReLU()
#   )
#   (4): AdaptiveAvgPool2d(output_size=1)
#   (5): Lambda()
#   (6): Linear(in_features=32, out_features=10, bias=True)
# )
# </pre>
# 
# Like the code below the AdaptiveAvgPool2d and Lambda (below we use Flatten) have the same output data-ie same plot

# #### stat and plot functions

# histc()
# 
# Computes the histogram of a tensor.
#         
#         The elements are sorted into equal width bins between :attr:`min` and
#         :attr:`max`. If :attr:`min` and :attr:`max` are both zero, the minimum and
#         maximum values of the data are used.



#this is run on each batch
def append_hist_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): 
        hook.stats = ([],[],[])
    means,stds,hists = hook.stats
    #mod 0 is XResNet object with properties:
    #_backend, _backward_hooks,_buffers,_forward_hooks (this is where stats is located)
    #,_load_state_dict_pre_hooks,_modules (87 sequentiol, 1 MaxPool2d)
    #,_parameters,_state_dict_hooks
    #then 1 AdaptiveConcatPool2d
    #then 2 Flatten
    #then 3 Linear (with shape [64,37])
    if mod.training:
        #outp.data shape is [64,512,4,4]
        means.append(outp.data.mean().cpu())
        stds .append(outp.data.std().cpu())
        hists.append(outp.data.cpu().histc(40,-10,10)) #histc isn't implemented on the GPU




def plot_hooks(hooks):
    fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
    print(f'plotting {len(hooks)} layers')
    for h in hooks:
        ms,ss, hists = h.stats
        ax0.plot(ms)
        ax0.set_title('mean')
        ax1.plot(ss)
        ax1.set_title('std. dev.')
    plt.legend(range(len(hooks)));




def plot_deltas(deltas):
    half_plots=int((len(deltas)/2) + (len(deltas)/2 % 1 > 0))
    fig,axes= plt.subplots(half_plots,2, figsize=(10,half_plots*2))
    for ax,h in zip(axes.flatten(), deltas):
        ax.plot(h)
        ax.set_title('delta')
    plt.legend(range(len(deltas)))
    plt.show()




#get the hist data at index 2
def get_hist(h): 
    assert len(h.stats)==3
    return torch.stack(h.stats[2]).t().float().log1p().cpu()




def plot_hooks_hist(hooks):
    #subplots(rows,colums)
    half_plots=int((len(hooks)/2) + (len(hooks)/2 % 1 > 0))
    fig,axes = plt.subplots(half_plots,2, figsize=(15,int(len(hooks)/2)*3))
    for ax,h in zip(axes.flatten(), hooks):
        ax.imshow(get_hist(h), origin='lower')
        ax.imshow(get_hist(h))
        ax.set_aspect('auto')
        ax.axis('on')
    plt.tight_layout()




def plot_hooks_hist_lines(hooks):
    half_plots=int((len(hooks)/2) + (len(hooks)/2 % 1 > 0))
    fig,axes = plt.subplots(half_plots,2, figsize=(15,int(len(hooks)/2)*5))
    for ax,h in zip(axes.flatten(), hooks):
        ax.plot(get_hist(h))
        ax.axis('on')




def get_start_mins(h):
    h1 = torch.stack(h.stats[2]).t().float().cpu()
    return h1[:2].sum(0)/h1.sum(0)




def get_mid_mins(h, max_hist):
    h1 = torch.stack(h.stats[2]).t().float().cpu()
    #get stats on middle bins of histogram data
    return h1[int((max_hist/2)-1):int((max_hist/2)+2)].sum(0)/h1.sum(0)




def plot_mins(hooks):
    half_plots=int((len(hooks)/2) + (len(hooks)/2 % 1 > 0))
    fig,axes = plt.subplots(half_plots,2, figsize=(15,int(len(hooks)/2)*2))
    for ax,h in zip(axes.flatten(), hooks):
        ax.plot(get_start_mins(h))
        ax.set_ylim(0,1)
        ax.axis('on')
    plt.tight_layout()
    plt.legend(['start'])




def plot_mid_mins(hooks):
    half_plots=int((len(hooks)/2) + (len(hooks)/2 % 1 > 0))
    fig,axes = plt.subplots(half_plots,2, figsize=(15,int(len(hooks)/2)*2))
    hist = get_hist(hooks[0])
    max_hist=hist.shape[0]
    print(max_hist)
    for ax,h in zip(axes.flatten(), hooks):
        ax.plot(get_mid_mins(h,max_hist))
        ax.set_ylim(0,1)
        ax.axis('on')
    plt.tight_layout()
    plt.legend(['mid'])




def plot_ep_vals(ep_vals):
    plt.ylabel("loss")
    plt.xlabel("epoch")
    epochs = ep_vals.keys()
    plt.xticks(np.asarray(list(epochs)))
    trn_losses = [item[0] for item in list(ep_vals.values())]
    val_losses = [item[1] for item in list(ep_vals.values())]
    plt.plot(epochs, trn_losses, c='b', label='train')
    plt.plot(epochs, val_losses, c='r', label='validation')
    plt.legend(loc='upper left')




for i,m in enumerate(learn.model):
    print(f'i: {i}, model part: {m}')


# ### Fit



with Hooks(learn.model, append_hist_stats) as hooks_naive: 
    learn.fit(5, cbsched)


# <pre>
# epoch 	train_loss 	train_accuracy 	valid_loss 	valid_accuracy 	time
# 0 	2.852017 	0.287686 	2.486142 	0.387275 	00:08
# 1 	2.179299 	0.469177 	2.641399 	0.352697 	00:07
# 2 	2.030308 	0.531273 	2.449481 	0.413555 	00:07
# 3 	1.731837 	0.638368 	1.788130 	0.615491 	00:07
# 4 	1.465440 	0.747413 	1.599652 	0.690180 	00:07
# </pre>



plot_hooks(hooks_naive)


# #### Histograms



hooks_naive




plot_hooks_hist(hooks_naive)




plot_mid_mins(hooks_naive)




naive_hists=[]
naive_ms=[]
naive_ss=[]
for h in hooks_naive:
    naive_hists.append(get_hist(h))
    ms,ss, hists = h.stats
    naive_ms.append(ms)
    naive_ss.append(ss)
naive_del_ms=np.diff(naive_ms)
naive_del_ss=np.diff(naive_ss)


# Means change between each layer



plot_deltas(naive_del_ms)




plot_deltas(naive_del_ss)


# Means change between first and last layer



first_n_last = [naive_ms[0], naive_ms[-1]]
print(len(first_n_last))
naive_del_fal=np.diff(first_n_last)
print(len(naive_del_fal))
plot_deltas(naive_del_fal)




learn.recorder.plot_loss()




len(learn.recorder.losses)/len(learn.recorder.val_losses)




#18.5 * more training data than validation data 


# ## adapt_model and gradual unfreezing

# Lines above pasted into one cell to create a function (Shift-M)



def adapt_model(learn, data):
    cut = next(i for i,o in enumerate(learn.model.children())
               if isinstance(o,nn.AdaptiveAvgPool2d))
    m_cut = learn.model[:cut]
    xb,yb = get_batch(data.valid_dl, learn)
    pred = m_cut(xb)
    ni = pred.shape[1]
    m_new = nn.Sequential(
        #replace m_cut with children to get data for each layer in XResNet
        *m_cut.children(), AdaptiveConcatPool2d(), Flatten(),
        nn.Linear(ni*2, data.c_out))
    learn.model = m_new




learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette, xtra_cb=Recorder)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))




adapt_model(learn, data)




len(learn.model)


# Grab all parameters in the body (the m_cut bit) and dont train these - just train the head

# #### Freeze everything before head



for p in learn.model[0].parameters(): p.requires_grad_(False)




with Hooks(learn.model, append_hist_stats) as hooks_freeze: 
    learn.fit(3, sched_1cycle(1e-2, 0.5))




plot_hooks(hooks_freeze)




plot_hooks_hist(hooks_freeze)




plot_mid_mins(hooks_naive)




learn.recorder.plot_loss()


# #### Unfreeze



for p in learn.model[0].parameters(): p.requires_grad_(True)




with Hooks(learn.model, append_hist_stats) as hooks_unfreeze: 
    learn.fit(5, cbsched, reset_opt=True)


# With freeze then unfreeze I'm getting slightly better than naive training.
# In frozen layer - train for particuar mean and std dev, but pets has different std dev and means inside the model.
# 
# What is really going on here? (1:26 in lesson video), and why do I get better results when JH got worse result?



plot_hooks(hooks_unfreeze)




plot_hooks_hist(hooks_unfreeze)




plot_mid_mins(hooks_unfreeze)




learn.recorder.plot_loss()


# Freeze only layer params that aren't in the batch norm layers
# 
# 1:27 in lesson 12

# ## Batch norm transfer



learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette, xtra_cb=Recorder)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)




def apply_mod(m, f):
    f(m)
    for l in m.children(): apply_mod(l, f)

def set_grad(m, b):
    #if linear layr at end of batchnorm layer in middle, dont change the gradient
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)


# #### Freeze and fit
# 
# Freeze just non batchnorm and last layer



apply_mod(learn.model, partial(set_grad, b=False))




with Hooks(learn.model, append_hist_stats) as hooks_freeze_non_bn: 
    learn.fit(3, sched_1cycle(1e-2, 0.5))




plot_hooks(hooks_freeze_non_bn)




plot_hooks_hist(hooks_freeze_non_bn)




plot_mins(hooks_freeze_non_bn)




learn.recorder.plot_loss()


# #### Unfreeze



apply_mod(learn.model, partial(set_grad, b=True))




with Hooks(learn.model, append_hist_stats) as hooks_unfreeze_non_bn: 
    learn.fit(5, cbsched, reset_opt=True)




plot_hooks(hooks_unfreeze_non_bn)




plot_hooks_hist(hooks_unfreeze_non_bn)




plot_mins(hooks_unfreeze_non_bn)




learn.recorder.plot_loss()


# Pytorch already has an `apply` method we can use:



learn.model.apply(partial(set_grad, b=False));


# Lesson 12 video: 1:29

# ## Non-Batch norm transfer



learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette, xtra_cb=Recorder)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)




def set_bn_grad(m, b):
    #if linear layr at end of batchnorm layer in middle, dont change the gradient
    if not isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)


# #### Freeze and fit
# 
# Freeze just batchnorm and last layer



apply_mod(learn.model, partial(set_bn_grad, b=False))




with Hooks(learn.model, append_hist_stats) as hooks_freeze_bn: 
    learn.fit(3, sched_1cycle(1e-2, 0.5))




plot_hooks(hooks_freeze_bn)




plot_hooks_hist(hooks_freeze_bn)




plot_mins(hooks_freeze_bn)




learn.recorder.plot_loss()


# #### Unfreeze



apply_mod(learn.model, partial(set_bn_grad, b=True))




with Hooks(learn.model, append_hist_stats) as hooks_unfreeze_bn: 
    learn.fit(5, cbsched, reset_opt=True)




plot_hooks(hooks_unfreeze_bn)




plot_hooks_hist(hooks_unfreeze_bn)




plot_mins(hooks_unfreeze_bn)




learn.recorder.plot_loss()


# ## Discriminative LR and param groups



learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)




learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)




def bn_splitter(m):
    def _bn_splitter(l, g1, g2):
        if isinstance(l, nn.BatchNorm2d): g2 += l.parameters()
        elif hasattr(l, 'weight'): g1 += l.parameters()
        for ll in l.children(): _bn_splitter(ll, g1, g2)
        
    g1,g2 = [],[]
    _bn_splitter(m[0], g1, g2)
    
    g2 += m[1:].parameters()
    return g1,g2




a,b = bn_splitter(learn.model)




test_eq(len(a)+len(b), len(list(m.parameters())))




Learner.ALL_CBS




#export
from types import SimpleNamespace
cb_types = SimpleNamespace(**{o:o for o in Learner.ALL_CBS})




cb_types.after_backward




#export
class DebugCallback(Callback):
    _order = 999
    def __init__(self, cb_name, f=None): self.cb_name,self.f = cb_name,f
    def __call__(self, cb_name):
        if cb_name==self.cb_name:
            if self.f: self.f(self.run)
            else:      set_trace()




#export
def sched_1cycle(lrs, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    phases = create_phases(pct_start)
    sched_lr  = [combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
                 for lr in lrs]
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler('lr', sched_lr),
            ParamScheduler('mom', sched_mom)]




disc_lr_sched = sched_1cycle([0,3e-2], 0.5)




learn = cnn_learner(xresnet18, data, loss_func, opt_func,
                    c_out=10, norm=norm_imagenette, splitter=bn_splitter)

learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)




def _print_det(o): 
    print (len(o.opt.param_groups), o.opt.hypers)
    raise CancelTrainException()

learn.fit(1, disc_lr_sched + [DebugCallback(cb_types.after_batch, _print_det)])




learn.fit(3, disc_lr_sched)




disc_lr_sched = sched_1cycle([1e-3,1e-2], 0.3)




learn.fit(5, disc_lr_sched)


# ## Export










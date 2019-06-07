#!/usr/bin/env python
# coding: utf-8








#export
from exp.nb_formatted import *




#set to true on first run
RETRAIN=False


# ## Serializing the model

# Store on ssd rather than in home folder



path = datasets.untar_data(datasets.URLs.IMAGEWOOF_160, dest='data')




size = 128
bs = 64

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



mdl_path = path/'models'
mdl_path.mkdir(exist_ok=True)

if RETRAIN:
    learn.fit(40, cbsched)
    st = learn.model.state_dict()

    print(type(st))

    #keys are names of the layers
    print(', '.join(st.keys()))
    print(st['0.0.weight'])
    print(st['10.bias'])
    print(path)

    #It's also possible to save the whole model, including the architecture, 
    #but it gets quite fiddly and we don't recommend it. 
    #Instead, just save the parameters, and recreate the model directly.

    torch.save(st, mdl_path/'iw5')
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




learn.fit(5, cbsched)


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

m_new = nn.Sequential(
    m_cut, AdaptiveConcatPool2d(), Flatten(),
    nn.Linear(ni*2, data.c_out))




learn.model = m_new


# #### stat and plot functions



def append_hist_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): 
        hook.stats = ([],[],[])
    means,stds,hists = hook.stats
    if mod.training:
        means.append(outp.data.mean().cpu())
        stds .append(outp.data.std().cpu())
        hists.append(outp.data.cpu().histc(40,0,10)) #histc isn't implemented on the GPU




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




#get the hist data at index 2
def get_hist(h): 
    assert len(h.stats)==3
    return torch.stack(h.stats[2]).t().float().log1p().cpu()




def plot_hooks_hist(hooks):
    fig,axes = plt.subplots(2,2, figsize=(15,6))
    for ax,h in zip(axes.flatten(), hooks):
        ax.imshow(get_hist(h), origin='lower')
        ax.imshow(get_hist(h))
        ax.set_aspect('auto')
        ax.axis('on')
    plt.tight_layout()




def plot_hooks_hist_lines(hooks):
    fig,axes = plt.subplots(2,2, figsize=(15,10))
    for ax,h in zip(axes.flatten(), hooks):
        ax.plot(get_hist(h))
        ax.axis('on')




def get_start_mins(h, total_bins):
    h1 = torch.stack(h.stats[2]).t().float().cpu()
    return h1[:2].sum(0)/h1.sum(0)




def get_mid_mins(h, total_bins):
    h1 = torch.stack(h.stats[2]).t().float().cpu()
    #get stats on middle bins of histogram data
    return h1[int((total_bins/2)-1):int((total_bins/2)+2)].sum(0)/h1.sum(0)




def plot_mins(hooks):
    fig,axes = plt.subplots(2,2, figsize=(15,6))
    hist = get_hist(hooks[0])
    total_bins=hist.shape[1]
    print(total_bins)
    for ax,h in zip(axes.flatten(), hooks[:4]):
        ax.plot(get_start_mins(h,total_bins))
        ax.plot(get_mid_mins(h,total_bins))
        ax.set_ylim(0,1)
        ax.axis('on')
    plt.tight_layout()
    plt.legend(['start','mid'])




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


# ### Fit



with Hooks(learn.model, append_hist_stats) as hooks_naive: 
    learn.fit(5, cbsched)




plot_hooks(hooks_naive)


# #### Histograms



hooks_naive




plot_hooks_hist(hooks_naive)




plot_mins(hooks_naive)




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
        m_cut, AdaptiveConcatPool2d(), Flatten(),
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




plot_mins(hooks_freeze)




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




plot_mins(hooks_unfreeze)




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










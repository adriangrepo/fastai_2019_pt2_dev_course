#!/usr/bin/env python
# coding: utf-8








import os 

GPUID='0'
os.environ['CUDA_VISIBLE_DEVICES']=GPUID




import sys
import datetime
import uuid
from itertools import chain
#export
import time
#pip3 install psutil
import psutil
from exp.nb_formatted import *




device = torch.device(f"cuda:{GPUID}" if torch.cuda.is_available() else "cpu")
print(device)




torch.cuda.empty_cache()




# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']




#set to true on first run
RETRAIN=False
#max mem use for weights in GB in RAM:
MAX_MEM=40




pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0] / 2. ** 30
print('memory use:', memoryUse)




#base, shallow, deep
HOOK_DEPTH = 'shallow'
CURRENT_DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
NAME=HOOK_DEPTH+'_'+CURRENT_DATE+'_'+UID+'_'+GPUID




NAME




#keep track of index and layer type
INDEX_DICT={}




cwd=os.getcwd()
IMG_PATH=os.path.abspath(cwd + "/images/")




IMG_PATH


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




#absolute path
HOOK_PATH=mdl_path/NAME
HOOK_PATH.mkdir(exist_ok=True)




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


# ### turn off shuffle 
# 
# Dont want shuffling of data at every epoch so can analyze parameter differences across epochs



data = ll.to_databunch(bs, c_in=3, c_out=c_out, num_workers=8, shuffle=False)




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




def module_grandchildren(model):
    grandkids=[]
    for m in m_cut.children():
        for g in m.children():
            grandkids.append(g)
    return grandkids




m_gc = module_grandchildren(m_cut)




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


# #### weights and stat functions

# histc()
# 
# Computes the histogram of a tensor.
#         
#         The elements are sorted into equal width bins between :attr:`min` and
#         :attr:`max`. If :attr:`min` and :attr:`max` are both zero, the minimum and
#         maximum values of the data are used.



#create a global in the __main__ scope so we can track iterations
stat_iter = 0




#saving weights to disk for later calcs (too much for RAM)
def append_hist_stats(hook, mod, inp, outp):
    '''Note that the hook is a different instance for each layer type
    ie XResNet has its own hook object, AdaptiveConcatPool2d has its own hook object etc
    '''
    if not hasattr(hook,'stats'): 
        hook.stats = ([],[],[])
    means,stds,hists = hook.stats
    if mod.training:
        #eg outp.data shape is [64,512,4,4]
        print(f"{str(mod).split('(')[0]} shape: {outp.data.shape}")
        means.append(outp.data.mean().cpu())
        stds.append(outp.data.std().cpu())
        if isinstance(mod, nn.Linear):
            hists.append(outp.data.cpu().histc(40,-10,10)) #no relu here
        else:
            hists.append(outp.data.cpu().histc(40,0,10)) #no negatives




def write_index_file(data, file_name):
    with open(file_name, 'w') as f:
        print(data, file=f)




#saving weights to disk for later calcs (too much for RAM)
def append_hist_stats_save(hook, mod, inp, outp):
    global stat_iter
    append_hist_stats(hook, mod, inp, outp)
    if mod.training:
        INDEX_DICT[stat_iter]=str(mod).split('(')[0]
        SHAPE_DICT[stat_iter]=str(list(outp.data.size()))
        np.save(f'{HOOK_DATA_PATH}/{stat_iter}.npy', outp.data.cpu())
        stat_iter+=1




#needs work before can use - far too much data to store in RAM without downsampling
def append_raw_hist_stats(hook, mod, inp, outp):
    global stat_iter
    memoryUse=0
    if not hasattr(hook,'stats'): 
        hook.stats = ([],[],[])
    if not hasattr(hook,'raws'): 
        hook.raws = ([])
    means,stds,hists = hook.stats
    raws = hook.raws
    if mod.training:
        #eg outp.data shape is [64,512,4,4]
        means.append(outp.data.mean().cpu())
        stds.append(outp.data.std().cpu())
        if (stat_iter % 100) == 0:
            memoryUse = py.memory_info()[0] / 2. ** 30
        if memoryUse < MAX_MEM:
            d = outp.data
            if len(outp.data.shape)==4:
                new_shape=outp.data.shape
                #want 3 channels so can plot r,g,b
                new_shape[-1]=3
                #put channel as last dimension
                d=d.permute(0, 2, 3, 1)
                d=d.cpu().numpy()
                #calc mean of the whole batch
                d=d.mean(0)
                #now coerce to 3 channels
                d=bin_ndarray(d, new_shape, operation='mean')
            #flatten and linear are dim 2 
            #now take mean over last index to reduce size
            else:
                d=d.cpu().numpy()
            raws.append(d.mean(-1))
            if stat_iter==0:
                print(f'data shape: {outp.data.cpu().numpy().shape}, size: {outp.data.cpu().numpy().size}, mean size: {outp.data.cpu().numpy().mean(-1).size}')
        else:
            print(f'oom appending: {outp.data.cpu().numpy().mean(-1).size}, {outp.data.cpu().numpy().shape}')
        if isinstance(mod, nn.Linear):
            hists.append(outp.data.cpu().histc(40,-10,10)) #no relu here
        else:
            hists.append(outp.data.cpu().histc(40,0,10)) #no negatives
        if stat_iter<100:
            print(f'iter: {stat_iter}, layer: {mod.__class__.__name__}, data shape: {outp.data.cpu().numpy().shape}')
        stat_iter+=1




def read_index_file(model_path, index_file):
    print(f'>>calc_hook_deltas {model_path}/{index_file}')
    data = None
    s = open(f'{model_path}/{index_file}', 'r').read()
    data = eval(s)
    data = collections.OrderedDict(sorted(data.items()))
    return data




def read_hist_stats(file_name):
    array_reloaded = np.load(file_name)
    array_reloaded=torch.from_numpy(array_reloaded).float().to(device)
    return array_reloaded
    




def get_k_by_v(d, val):
    keys = []
    for key, value in d.items():
        if value == val:
            keys.append(key)
    #print(f'val: {val}, len(keys): {len(keys)}')
    return keys 




#see chris albon
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]




def calc_hook_epoch_deltas(data_path, index_dict, tot_epochs, model_size):
    '''Differences between weights between epochs for corresponding layers 
    - assumes batches are ordered'''
    #layers of model
    model_layers = [v for v in list(index_dict.values())[:model_size]]

    # for shallow is 525
    runs = int(len(index_dict) / model_size)
    denom = int(runs / tot_epochs)

    epoch_chunks = list(chunks(list(index_dict.keys()), denom))
    first_epoch_data = epoch_chunks[0]

    # block of each full model iteration - want delta between corresponding layers in these
    first_epoch_batches = list(chunks(first_epoch_data, model_size))
    # unpack 2d list to 1d
    first_epoch_batches_list = list(chain.from_iterable(first_epoch_batches))
    first_epoch_batch_indexes = {key: index_dict[key] for key in first_epoch_batches_list if key in index_dict.keys()}

    model_deltas = {}
    i = 0
    model_count = 0
    for key, v in first_epoch_batch_indexes.items():
        # note the step to jump to corresponding layer in next batch
        for offset in range(0, denom*tot_epochs-denom, denom):
            d1 = read_hist_stats(f'{data_path}/{key + offset+ denom-1}.npy')
            d0 = read_hist_stats(f'{data_path}/{key + offset}.npy')
            if d1.size() == d0.size():
                delta = d1 - d0
            else:
                #presumably due to shuffle=on, should not hit here with shuffle=off, TODO test this
                print(f'i: {i}, key: {key}, v: {v} d1.size:{d1.size()}, d0.size:{d0.size()}')

                d1_shp=list(d1.size())
                d0_shp=list(d0.size())
                #total hack, and doesnt really make sense anyway as if sizes are different we 
                #shouldn't be differenceing
                d1_0=int(d1_shp[0])
                d0_0=int(d0_shp[0])
                if len(d1_shp)==4:
                    if d1_0>d0_0:
                        d2 = d1[:int(d0_shp[0]),d1_shp[1]-1,d1_shp[2]-1,d1_shp[3]-1]
                    else:
                        d2 = d0[:d1_shp[0],d0_shp[1]-1,d0_shp[2]-1,d0_shp[3]-1]
                elif len(d1_shp)==3:
                    if d1_0>d0_0:
                        d2 = d1[:d0_shp[0],d1_shp[1]-1,d1_shp[2]-1]
                    else:
                        d2 = d0[:d1_shp[0],d0_shp[1]-1,d0_shp[2]-1]
                elif len(d1_shp)==2:
                    if d1_0>d0_0:
                        d2 = d1[:d0_shp[0],d1_shp[1]-1]
                    else:
                        d2 = d0[:d1_shp[0],d0_shp[1]-1]
                delta = d2
            model_deltas.setdefault(f'{model_count}', []).append(delta)
            model_count += 1
            if model_count >= tot_epochs:
                model_count = 0
            i += 1
    print(f'{i} mem use: {py.memory_info()[0] / 2. ** 30}')
    return model_deltas




def calc_hook_batch_deltas(data_path, index_dict, tot_epochs, epoch, model_size):
    '''Differences between weights between batches for corresponding layers 
    - idicator of data variation between batches
    @Return: dict of key: layer index and value: list of deltas between batches'''
    assert isinstance(tot_epochs, int)
    assert isinstance(epoch, int)

    #for shallow is 525
    runs=int(len(index_dict)/model_size)
    denom=int(runs/tot_epochs)

    epoch_chunks = list(chunks(list(index_dict.keys()), denom))
    epoch_data = epoch_chunks[epoch]
    
    #block of each full model iteration - want delta between corresponding layers in these
    batches=list(chunks(epoch_data, model_size))
    #unpack 2d list to 1d
    batches_list=list(chain.from_iterable(batches))
    batch_indexes= {key: index_dict[key] for key in batches_list if key in index_dict.keys()}
    deltas_dict={}

    deltas=[]
    model_deltas={}
    i=0
    model_count =0
    for key,v in batch_indexes.items():
        #so when get to last batch dont try to read in next one
        if i<len(batch_indexes)-model_size:
            # note the step to jump to corresponding layer in next batch
            d1=read_hist_stats(f'{data_path}/{key+model_size}.npy')
            d0=read_hist_stats(f'{data_path}/{key}.npy')
            if d1.size() == d0.size():
                delta = d1-d0
            else:
                print(f'i: {i}, key: {key}, v: {v} d1.size:{d1.size()}, d0.size:{d0.size()}')
            model_deltas.setdefault(model_count, []).append(delta)
            model_count+=1
            if model_count>=model_size:
                model_count=0
            i+=1
    return model_deltas




#get the hist data at index 2
def get_hist(h): 
    assert len(h.stats)==3
    return torch.stack(h.stats[2]).t().float().log1p().cpu()




def get_delta_hist(h): 
    return torch.stack(h.hdeltas).t().float().log1p().cpu()




def diff_stats(hooks_data):
    histsl=[]
    msl=[]
    ssl=[]
    for h in hooks_data:
        histsl.append(get_hist(h))
        ms,ss, hists = h.stats
        msl.append(ms)
        ssl.append(ss)
    del_ms=np.diff(msl)
    del_ss=np.diff(ssl)
    return del_ms,del_ss




def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth




def get_start_mins(h):
    h1 = torch.stack(h.stats[2]).t().float().cpu()
    return h1[:2].sum(0)/h1.sum(0)




def get_mid_mins(h, max_hist):
    h1 = torch.stack(h.stats[2]).t().float().cpu()
    #get stats on middle bins of histogram data
    return h1[int((max_hist/2)-1):int((max_hist/2)+2)].sum(0)/h1.sum(0)


# #### plotting



def plot_hooks(hooks, model, fig_name=None):
    fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
    print(f'plotting {len(hooks)} layers')
    i=0
    first_len=0
    for h,m in zip(hooks,model):
        ms,ss, hists = h.stats
        if i==0:
            first_len=len(ms)
        #print(f"{i}, {str(m).split('(')[0]}: {len(ms)}")
        if str(m).split('(')[0]=='ReLU':
        #data for Relu layer is 19 times longer than other layers, not sure why
            ms=ms[:first_len] 
            ss=ss[:first_len] 
        ax0.plot(ms)
        ax0.set_title('mean')
        ax1.plot(ss)
        ax1.set_title('std. dev.')
        i+=1
    if model:   
        titles=[]
        idxs=[]
        for j,m in enumerate(learn.model):
            titles.append(str(j)+' '+str(m).split('(')[0])
        plt.legend(titles)
    else:
        plt.legend(range(len(hooks)))
    if fig_name:
        plt.savefig(IMG_PATH+'/'+NAME+'_'+fig_name)
    plt.show()




def plot_deltas(deltas, smoother, fig_name=None):
    half_plots=int((len(deltas)/2) + (len(deltas)/2 % 1 > 0))
    fig,axes= plt.subplots(half_plots,2, figsize=(10,half_plots*2))
    for ax,h in zip(axes.flatten(), deltas):
        h = smooth(h, smoother)
        ax.plot(h)
        ax.set_title('delta')
    plt.legend(range(len(deltas)))
    if fig_name:
        plt.savefig(IMG_PATH+'/'+NAME+'_'+fig_name)
    plt.show()




def plot_hooks_hist(hooks, fig_name=None, model=None):
    #subplots(rows,colums)
    half_plots=int((len(hooks)/2) + (len(hooks)/2 % 1 > 0))
    fig,axes = plt.subplots(half_plots,2, figsize=(15,int(len(hooks)/2)*3))
    i=0
    for ax,h in zip(axes.flatten(), hooks):
        ax.imshow(get_hist(h), origin='lower')
        ax.imshow(get_hist(h))
        ax.set_aspect('auto')
        ax.axis('on')
        if model:
            for j,m in enumerate(learn.model):
                if i == j:
                    title=str(m).split('(')[0]
                    ax.set_title(title)
        i+=1
    plt.tight_layout()
    if fig_name:
        plt.savefig(IMG_PATH+'/'+NAME+'_'+fig_name)
    plt.show()




def plot_hooks_delta_hist(hooks, fig_name=None,model=None):
    #subplots(rows,colums)
    half_plots=int((len(hooks)/2) + (len(hooks)/2 % 1 > 0))
    fig,axes = plt.subplots(half_plots,2, figsize=(15,int(len(hooks)/2)*3))
    i=0
    for ax,h in zip(axes.flatten(), hooks):
        ax.imshow(get_delta_hist(h), origin='lower')
        ax.imshow(get_delta_hist(h))
        ax.set_aspect('auto')
        ax.axis('on')
        if model:
            for j,m in enumerate(learn.model):
                if i == j:
                    title=str(m).split('(')[0]
                    ax.set_title(title)
        i+=1
    plt.tight_layout()
    if fig_name:
        plt.savefig(IMG_PATH+'/'+NAME+'_'+fig_name)
    plt.show()




def plot_hooks_hist_lines(hooks, fig_name=None):
    half_plots=int((len(hooks)/2) + (len(hooks)/2 % 1 > 0))
    fig,axes = plt.subplots(half_plots,2, figsize=(15,int(len(hooks)/2)*5))
    for ax,h in zip(axes.flatten(), hooks):
        ax.plot(get_hist(h))
        ax.axis('on')
    if fig_name:
        plt.savefig(IMG_PATH+'/'+NAME+'_'+fig_name)
    plt.show()




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
        ax.set_ylim(-1,1)
        ax.axis('on')
    plt.tight_layout()
    plt.legend(['mid'])




def plot_ep_vals(ep_vals, fig_name=None):
    plt.ylabel("loss")
    plt.xlabel("epoch")
    epochs = ep_vals.keys()
    plt.xticks(np.asarray(list(epochs)))
    trn_losses = [item[0] for item in list(ep_vals.values())]
    val_losses = [item[1] for item in list(ep_vals.values())]
    plt.plot(epochs, trn_losses, c='b', label='train')
    plt.plot(epochs, val_losses, c='r', label='validation')
    plt.legend(loc='upper left')
    if fig_name:
        plt.savefig(IMG_PATH+'/'+NAME+'_'+fig_name)
    plt.show()




def plot_raw_layers(hooks):
    for h in hooks:
        print(len(h.raws))
        h_raw_np=np.concatenate(h.raws, axis=0 )
        plot_kernels(h_raw_np)




#after Fchaubard https://discuss.pytorch.org/t/understanding-deep-network-visualize-weights/2060/7
def plot_kernels(tensor, num_cols=6):
    if isinstance(tensor, list):
        print(len(tensor))
        print(tensor[0].shape)
    if not tensor.ndim==4:
        #not plotting Flatten or Linear layers
        return
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()




for i,m in enumerate(learn.model):
    print(f'i: {i}, model part: {m}')


# ## Naive Model 
# 
# Allow hook to access internals of XResNet by unpacking arg list



m_new = nn.Sequential(
    *m_cut.children(), AdaptiveConcatPool2d(), Flatten(),
    nn.Linear(ni*2, data.c_out))




m_new_deep = nn.Sequential(
    *m_gc, AdaptiveConcatPool2d(), Flatten(),
    nn.Linear(ni*2, data.c_out))




m_new_shallow = nn.Sequential(
    m_cut, AdaptiveConcatPool2d(), Flatten(),
    nn.Linear(ni*2, data.c_out))




if HOOK_DEPTH=='deep':
    learn.model = m_new_deep
elif HOOK_DEPTH=='base':
    learn.model = m_new
elif HOOK_DEPTH=='shallow':
    learn.model = m_new_shallow




len(learn.model)


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

# ### Fit

# [batch_size, channels, height, width]



stat_iter = 0




INDEX_DICT={}
SHAPE_DICT={}
HOOK_DATA_PATH = HOOK_PATH/'naive'
HOOK_DATA_PATH.mkdir(exist_ok=True)




#5
with Hooks(learn.model, append_hist_stats_save) as hooks_naive: 
    learn.fit(5, cbsched)




write_index_file(data=INDEX_DICT, file_name=HOOK_DATA_PATH/'index.txt')
write_index_file(data=SHAPE_DICT, file_name=HOOK_DATA_PATH/'shape.txt')
print(f"wrote {len(INDEX_DICT)} items to : {HOOK_DATA_PATH/'index.txt'}")


# <pre>
# epoch 	train_loss 	train_accuracy 	valid_loss 	valid_accuracy 	time
# 0 	2.852017 	0.287686 	2.486142 	0.387275 	00:08
# 1 	2.179299 	0.469177 	2.641399 	0.352697 	00:07
# 2 	2.030308 	0.531273 	2.449481 	0.413555 	00:07
# 3 	1.731837 	0.638368 	1.788130 	0.615491 	00:07
# 4 	1.465440 	0.747413 	1.599652 	0.690180 	00:07
# </pre>



#temp for testing only
#HOOK_DATA_PATH=HOOK_PATH/'naive'




index_dict=read_index_file(HOOK_DATA_PATH.absolute(), 'index.txt')




model_layers = [v for v in list(index_dict.values())[:len(learn.model)]]
model_layers




batch_deltas=calc_hook_batch_deltas(HOOK_DATA_PATH.absolute(), index_dict, tot_epochs=5,epoch=0, model_size=len(learn.model))




len(batch_deltas)




len(batch_deltas[0])




epoch_deltas=calc_hook_epoch_deltas(HOOK_DATA_PATH.absolute(), index_dict, tot_epochs=5, model_size=len(learn.model))




len(epoch_deltas)




len(epoch_deltas[list(epoch_deltas.keys())[0]])


# ##### Plots



plot_hooks(hooks_naive,fig_name='naive_stats.png',model=learn.model)




type(hooks_naive)


# #### Histograms



len(hooks_naive)




ms,ss, hists = hooks_naive[0].stats
len(hists); 




hists[0].shape




ms[0]




plot_hooks_hist(hooks_naive,fig_name='naive_hists.png',model=learn.model)




#plot_mid_mins(hooks_naive)




naive_del_ms,naive_del_ss=diff_stats(hooks_naive)


# Means change between each layer



plot_deltas(naive_del_ms, 50,fig_name='naive_del_ms.png')




plot_deltas(naive_del_ss, 50,fig_name='naive_del_sds.png')


# Means change between first and last layer



first_n_last = [naive_del_ms[0], naive_del_ms[-1]]
print(len(first_n_last))
naive_del_fal=np.diff(first_n_last)
print(len(naive_del_fal))
plot_deltas(naive_del_fal, 50)




learn.recorder.plot_loss()




len(learn.recorder.losses)/len(learn.recorder.val_losses)




#18.5 * more training data than validation data 




# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)




#clear mem
naive_del_ms=None
naive_del_ss=None
hooks_naive=None
naive_del_fal=None


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




def adapt_deep_model(learn, data):
    cut = next(i for i,o in enumerate(learn.model.children())
               if isinstance(o,nn.AdaptiveAvgPool2d))
    m_cut = learn.model[:cut]
    m_gc = module_grandchildren(m_cut)
    xb,yb = get_batch(data.valid_dl, learn)
    pred = m_cut(xb)
    ni = pred.shape[1]
    m_new = nn.Sequential(
        #replace m_cut with grandchildren to get data for each layer in XResNet
        *m_gc, AdaptiveConcatPool2d(), Flatten(),
        nn.Linear(ni*2, data.c_out))
    learn.model = m_new




def adapt_simple_model(learn, data):
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




if HOOK_DEPTH=='deep':
    adapt_deep_model(learn, data)
elif HOOK_DEPTH=='base':
    adapt_model(learn, data)
elif HOOK_DEPTH=='shallow':
    adapt_simple_model(learn, data)


# basic model as per lesson len: 4, with children len: 11, with grandchildren len: 20



len(learn.model); HOOK_DEPTH


# Grab all parameters in the body (the m_cut bit) and dont train these - just train the head



INDEX_DICT={}
SHAPE_DICT={}
HOOK_DATA_PATH = HOOK_PATH/'freeze'
HOOK_DATA_PATH.mkdir(exist_ok=True)




#reset
stat_iter=0


# #### Freeze everything before head



#everything before AdaptiveConcatPool2d 
for i in range(len(learn.model)-3):
    for p in learn.model[0].parameters(): p.requires_grad_(False)




with Hooks(learn.model, append_hist_stats_save) as hooks_freeze: 
    learn.fit(3, sched_1cycle(1e-2, 0.5))




write_index_file(data=INDEX_DICT, file_name=HOOK_DATA_PATH/'index.txt')
write_index_file(data=SHAPE_DICT, file_name=HOOK_DATA_PATH/'shape.txt')
print(f"wrote {len(INDEX_DICT)} items to : {HOOK_DATA_PATH/'index.txt'}")




plot_hooks(hooks_freeze,fig_name='freeze_layer_stats.png',model=learn.model)




plot_hooks_hist(hooks_freeze,fig_name='freeze_hists.png',model=learn.model)




#plot_hooks_delta_hist(hooks_freeze,fig_name='freeze_delta_hists.png',model=learn.model)




#plot_mid_mins(hooks_freeze)




frozen_del_ms,frozen_del_ss=diff_stats(hooks_freeze)




plot_deltas(frozen_del_ms, 50,fig_name='freeze_del_ms.png')




plot_deltas(frozen_del_ss, 50,fig_name='freeze_del_sds.png')




learn.recorder.plot_loss()




#clear mem
frozen_del_ms=None
frozen_del_ss=None
hooks_freeze=None




# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# #### Unfreeze



#reset
stat_iter=0




INDEX_DICT={}
SHAPE_DICT={}
HOOK_DATA_PATH = HOOK_PATH/'unfreeze'
HOOK_DATA_PATH.mkdir(exist_ok=True)




#everything before AdaptiveConcatPool2d - note difference to lesson nb where just have layer 0
for i in range(len(learn.model)-3):
    for p in learn.model[i].parameters(): p.requires_grad_(True)




with Hooks(learn.model, append_hist_stats_save) as hooks_unfreeze: 
    learn.fit(5, cbsched, reset_opt=True)




write_index_file(data=INDEX_DICT, file_name=HOOK_DATA_PATH/'index.txt')
write_index_file(data=SHAPE_DICT, file_name=HOOK_DATA_PATH/'shape.txt')
print(f"wrote {len(INDEX_DICT)} items to : {HOOK_DATA_PATH/'index.txt'}")


# 
# In frozen layer - train for particuar mean and std dev, but pets has different std dev and means inside the model.
# 
# What is really going on here? (1:26 in lesson video), and why do I get better results when JH got worse result?



plot_hooks(hooks_unfreeze,fig_name='unfreeze_layer_stats.png',model=learn.model)




plot_hooks_hist(hooks_unfreeze,fig_name='unfreeze_hists.png',model=learn.model)




#plot_hooks_delta_hist(hooks_unfreeze,fig_name='unfreeze_delta_hists.png',model=learn.model)




#plot_mid_mins(hooks_unfreeze)




unfrozen_del_ms,unfrozen_del_ss=diff_stats(hooks_unfreeze)




plot_deltas(unfrozen_del_ms, 50,fig_name='unfreeze_del_ms.png')




plot_deltas(unfrozen_del_ss, 50,fig_name='unfreeze_del_sds.png')




learn.recorder.plot_loss()


# Freeze only layer params that aren't in the batch norm layers
# 
# 1:27 in lesson 12



#clear mem
unfrozen_del_ms=None
unfrozen_del_ss=None
hooks_unfreeze=None




# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# ## Batch norm transfer

# Freeze all params that are not in the batchnorm layer or linear layer at end



#reset
stat_iter=0




INDEX_DICT={}
SHAPE_DICT={}
HOOK_DATA_PATH = HOOK_PATH/'bn_freeze'
HOOK_DATA_PATH.mkdir(exist_ok=True)




learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette, xtra_cb=Recorder)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)




def apply_mod(m, f):
    f(m)
    for l in m.children(): apply_mod(l, f)

def set_grad(m, b):
    #if linear layer (at end) or batchnorm layer in middle, dont change the gradient
    print(type(m))
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)


# #### Freeze and fit
# 
# Freeze just non batchnorm and last layer, note using apply_mod we traverse children layers so can access Batchnorm layers inside sequential



apply_mod(learn.model, partial(set_grad, b=False))




#run out of mem with deltas
with Hooks(learn.model, append_hist_stats_save) as hooks_freeze_non_bn: 
    learn.fit(3, sched_1cycle(1e-2, 0.5))




write_index_file(data=INDEX_DICT, file_name=HOOK_DATA_PATH/'index.txt')
write_index_file(data=SHAPE_DICT, file_name=HOOK_DATA_PATH/'shape.txt')
print(f"wrote {len(INDEX_DICT)} items to : {HOOK_DATA_PATH/'index.txt'}")




plot_hooks(hooks_freeze_non_bn,fig_name='freeze_non_bn_layers.png',model=learn.model)




plot_hooks_hist(hooks_freeze_non_bn,fig_name='freeze_non_bn_hist.png',model=learn.model)




#plot_hooks_delta_hist(hooks_freeze_non_bn,fig_name='freeze_non_bn_delta_hist.png',model=learn.model)




#plot_mins(hooks_freeze_non_bn)




freeze_non_bn_del_ms,freeze_non_bn_del_sds=diff_stats(hooks_freeze_non_bn)




plot_deltas(freeze_non_bn_del_ms, 50,fig_name='freeze_non_bn_del_ms.png')




plot_deltas(freeze_non_bn_del_sds, 50,fig_name='freeze_non_bn_del_sds.png')




learn.recorder.plot_loss()


# #### Unfreeze



#reset
stat_iter=0




INDEX_DICT={}
SHAPE_DICT={}
HOOK_DATA_PATH = HOOK_PATH/'bn_unfreeze'
HOOK_DATA_PATH.mkdir(exist_ok=True)




apply_mod(learn.model, partial(set_grad, b=True))




with Hooks(learn.model, append_hist_stats_save) as hooks_unfreeze_non_bn: 
    learn.fit(5, cbsched, reset_opt=True)




write_index_file(data=INDEX_DICT, file_name=HOOK_DATA_PATH/'index.txt')
write_index_file(data=SHAPE_DICT, file_name=HOOK_DATA_PATH/'shape.txt')
print(f"wrote {len(INDEX_DICT)} items to : {HOOK_DATA_PATH/'index.txt'}")




plot_hooks(hooks_unfreeze_non_bn,fig_name='unfreeze_non_bn_layers.png',model=learn.model)




plot_hooks_hist(hooks_unfreeze_non_bn,fig_name='unfreeze_non_bn_hist.png',model=learn.model)




#plot_hooks_delta_hist(hooks_unfreeze_non_bn,fig_name='unfreeze_non_bn_delta_hist.png',model=learn.model)




#plot_mins(hooks_unfreeze_non_bn)




unfreeze_non_bn_del_ms,unfreeze_non_bn_del_sds=diff_stats(hooks_unfreeze_non_bn)




plot_deltas(unfreeze_non_bn_del_ms, 50,fig_name='unfreeze_non_bn_del_ms.png')




plot_deltas(unfreeze_non_bn_del_sds, 50,fig_name='unfreeze_non_bn_del_sds.png')




learn.recorder.plot_loss()


# Pytorch already has an `apply` method we can use:



learn.model.apply(partial(set_grad, b=False));


# Lesson 12 video: 1:29

# ## Discriminative LR and param groups



learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)




learn.model.load_state_dict(torch.load(mdl_path/'iw5'))




len(learn.model)




adapt_simple_model(learn, data)




len(learn.model)




def bn_splitter(m):
    def _bn_splitter(l, g1, g2):
        if isinstance(l, nn.BatchNorm2d): 
            g2 += l.parameters()
        elif hasattr(l, 'weight'): 
            g1 += l.parameters()
        for ll in l.children(): 
            _bn_splitter(ll, g1, g2)
        
    g1,g2 = [],[]
    _bn_splitter(m[0], g1, g2)
    
    g2 += m[1:].parameters()
    return g1,g2




m = learn.model




a,b = bn_splitter(m)




len(a)




len(list(m.parameters()))




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




learn.fit(1, disc_lr_sched)


# ## Export



#!./notebook2script.py 11a_transfer_learning.ipynb








from functools import partial
from utils.nb_functions import *
from utils.nb_classes_l8_to_10 import *
from utils.nb_classes_l10_revised import *
from utils.nb_classes_cnn import *

import re
import torch
from torch import tensor
import math
import PIL,os,mimetypes

RUN_DATA_ON_GPU=True

Path.ls = lambda x: list(x.iterdir())

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def setify(o):
    return o if isinstance(o,set) else set(listify(o))

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def get_files(path, extensions=None, recurse=False, include=None):
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
            if include is not None and i==0: d[:] = [o for o in d if o in include]
            else:                            d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, f, extensions)

def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x

class ItemList(ListContainer):
    def __init__(self, items, path='.', tfms=None):
        super().__init__(items)
        self.path,self.tfms = Path(path),tfms

    def __repr__(self):
        return f'{super().__repr__()}\nPath: {self.path}'
    def new(self, items):
        return self.__class__(items, self.path, tfms=self.tfms)

    def  get(self, i):
        return i
    def _get(self, i):
        return compose(self.get(i), self.tfms)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res,list):
            return [self._get(o) for o in res]
        return self._get(res)

class ImageList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None:
            extensions = image_extensions
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    def get(self, fn):
        return PIL.Image.open(fn)

class Transform():
    _order=0

class MakeRGB(Transform):
    def __call__(self, item):
        return item.convert('RGB')

def make_rgb(item):
    return item.convert('RGB')

def grandparent_splitter(fn, valid_name='valid', train_name='train'):
    gp = fn.parent.parent.name
    return True if gp==valid_name else False if gp==train_name else None

def split_by_func(ds, f):
    items = ds.items
    mask = [f(o) for o in items]
    # `None` values will be filtered out
    train = [o for o,m in zip(items,mask) if m==False]
    valid = [o for o,m in zip(items,mask) if m==True ]
    return train,valid

class SplitData():
    def __init__(self, train, valid):
        self.train,self.valid = train,valid

    def __getattr__(self,k):
        return getattr(self.train,k)

    @classmethod
    def split_by_func(cls, il, f):
        lists = map(il.new, split_by_func(il, f))
        return cls(*lists)

    def __repr__(self):
        return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n'

from collections import OrderedDict

def uniqueify(x, sort=False):
    res = list(OrderedDict.fromkeys(x).keys())
    if sort:
        res.sort()
    return res

class Processor():
    def process(self, items):
        return items

class CategoryProcessor(Processor):
    def __init__(self): self.vocab=None

    def process(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi  = {v:k for k,v in enumerate(self.vocab)}
        return [self.proc1(o) for o in items]
    def proc1(self, item):
        return self.otoi[item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return self.vocab[idx]

#This is a bit different from what was seen during the lesson but it's necessary for NLP
def _process(self, processors):
    self.processors = listify(processors)
    for proc in self.processors: self.items = proc.process(self.items)
    return self

def _obj(self, idx):
    res = self[idx]
    for proc in self.processors:
        res = proc.deprocess(res) if isinstance(res,(tuple,list,Generator)) else proc.deproc1(res)
    return res

ItemList.process = _process
ItemList.obj = _obj

def parent_labeler(fn):
    return fn.parent.name

def _label_by_func(ds, f):
    return [f(o) for o in ds.items]

#This is a bit different from what was seen during the lesson but it's necessary for NLP
class LabeledData():
    def __init__(self, x, y):
        self.x,self.y = x,y

    def __repr__(self):
        return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return len(self.x)

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        labels = _label_by_func(il, f)
        proc_inputs = il.process(proc_x)
        proc_labels = ItemList(labels, path=il.path).process(proc_y)
        return cls(il, proc_labels)

def label_by_func(sd, f, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
    return SplitData(train,valid)

class ResizeFixed(Transform):
    _order=10
    def __init__(self,size):
        if isinstance(size,int): size=(size,size)
        self.size = size

    def __call__(self, item):
        return item.resize(self.size, PIL.Image.BILINEAR)

def to_byte_tensor(item):
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
    w,h = item.size
    return res.view(h,w,-1).permute(2,0,1)
to_byte_tensor._order=20

def to_float_tensor(item):
    return item.float().div_(255.)
to_float_tensor._order=30

def show_image(im, figsize=(3,3)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(im.permute(1,2,0))

class DataBunch():
    def __init__(self, train_dl, valid_dl, c_in=None, c_out=None):
        self.train_dl,self.valid_dl,self.c_in,self.c_out = train_dl,valid_dl,c_in,c_out

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
    #print(f'>>normalize_chan is_cuda x: {x.is_cuda},mean: {mean.is_cuda},std: {std.is_cuda}')
    if not RUN_DATA_ON_GPU:
        mean = mean.cpu()
        std = std.cpu()
        x = x.cpu()
        return (x - mean[..., None, None]) / std[..., None, None]
    return (x-mean[...,None,None]) / std[...,None,None]

_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])

if RUN_DATA_ON_GPU:
    norm_imagenette = partial(normalize_chan, mean=_m.cuda(), std=_s.cuda())
else:
    norm_imagenette = partial(normalize_chan, mean=_m, std=_s)


def prev_pow_2(x):
    return 2**math.floor(math.log2(x))

def get_cnn_layers(data, nfs, layer, **kwargs):
    def f(ni, nf, stride=2):
        return layer(ni, nf, 3, stride=stride, **kwargs)
    l1 = data.c_in
    l2 = prev_pow_2(l1*3*3)
    layers =  [f(l1  , l2  , stride=1),
               f(l2  , l2*2, stride=2),
               f(l2*2, l2*4, stride=2)]
    nfs = [l2*4] + nfs
    layers += [f(nfs[i], nfs[i+1]) for i in range(len(nfs)-1)]
    layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten),
               nn.Linear(nfs[-1], data.c_out)]
    return layers

def get_cnn_model(data, nfs, layer, **kwargs):
    print(f'>>get_cnn_model() data.c_in: {data.c_in}')
    model= nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))
    return model

def get_learn_run(nfs, data, lr, layer, cbs=None, opt_func=None, **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)

def model_summary(run, learn, data, find_all=False):
    xb,yb = get_batch(data.valid_dl, run)
    if RUN_DATA_ON_GPU:
        device = next(learn.model.parameters()).device#Model may not be on the GPU yet
        xb,yb = xb.to(device),yb.to(device)
    mods = find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    f = lambda hook,mod,inp,out: print(f"{mod}\n{out.shape}\n")
    with Hooks(mods, f) as hooks:
        learn.model(xb)
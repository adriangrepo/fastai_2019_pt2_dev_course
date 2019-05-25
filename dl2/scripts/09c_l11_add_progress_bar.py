#!/usr/bin/env python
# coding: utf-8

# # Adding progress bars to Learner








from utils.nb_learner import *
import time
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time


# One thing has been missing all this time, and as fun as it is to stare at a blank screen waiting for the results, it's nicer to have some tool to track progress.

# ## Imagenette data



path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)




tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]
bs = 64

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)




nfs = [32]*4


# We rewrite the `AvgStatsCallback` to add a line with the names of the things measured and keep track of the time per epoch.



# export 
class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
    
    def begin_fit(self):
        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
        names = ['epoch'] + [f'train_{n}' for n in met_names] + [
            f'valid_{n}' for n in met_names] + ['time']
        self.logger(names)
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        stats = [str(self.epoch)] 
        for o in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats] 
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)


# Then we add the progress bars... with a Callback of course! `master_bar` handles the count over the epochs while its child `progress_bar` is looping over all the batches. We just create one at the beginning or each epoch/validation phase, and update it at the end of each batch. By changing the logger of the `Learner` to the `write` function of the master bar, everything is automatically written there.
# 
# Note: this requires fastprogress v0.1.21 or later. 



# export 
class ProgressCallback(Callback):
    _order=-1
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)
        
    def after_fit(self): self.mbar.on_iter_end()
    def after_batch(self): self.pb.update(self.iter)
    def begin_epoch   (self): self.set_pb()
    def begin_validate(self): self.set_pb()
        
    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar, auto_update=False)
        self.mbar.update(self.epoch)


# By making the progress bar a callback, you can easily choose if you want to have them shown or not.



cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback,
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette)]




learn = get_learner(nfs, data, 0.4, conv_layer, cb_funcs=cbfs)




learn.fit(2)


# ## Export



#!./notebook2script.py 09c_add_progress_bar.ipynb







# LT 8/17/22 - all neural stuff moved to neuralmonkey 
from neuralmonkey.classes.population import *


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def subsample_rand(X, n_rand):
#     """ get random sample (dim 0), or return all if n_rand larget than X.
#     """
#     import random
#     if n_rand>X.shape[0]:
#         return X, list(range(X.shape[0]))

#     inds = random.sample(range(X.shape[0]), n_rand)
#     return X[inds, ...], inds

# def plotNeurHeat(X, ax=None, barloc="right", robust=True, zlims = None):
#     """ plot heatmap for data X.
#     X must be (neuron, time)
#     """
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10,5))
    
#     # X = self.activations[pop][tasknum]
#     X_nonan = X[:]
#     X_nonan = X_nonan[~np.isnan(X_nonan)]
#     minmax = (np.min(X_nonan), np.max(X_nonan))

#     # plot
#     if zlims is not None: 
#         sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
#                robust=robust, vmin=zlims[0], vmax=zlims[1])
#     else:
#         sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
#                robust=robust)
#     # ax.set_title(f"{pop}|{minmax[0]:.2f}...{minmax[1]:.2f}")
#     ax.set_xlabel(f"robust={robust}|{minmax[0]:.2f}...{minmax[1]:.2f}")
#     ax.set_ylabel('neuron #')
        
# def plotNeurTimecourse(X, Xerror=None, ax=None, n_rand=None, marker="-"):
#     """ Plot overlaid timecourses. 
#     - X, (neuron/trials, time)
#     - Xerror, (neuron/trials, time), to add errorbars)
#     """

#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10,5))
    
#     # X = self.activations[pop][tasknum]
#     X_nonan = X[:]
#     X_nonan = X_nonan[~np.isnan(X_nonan)]
#     minmax = (np.min(X_nonan), np.max(X_nonan))

#     if n_rand is not None:
#         X, indsrand = subsample_rand(X, n_rand)
#     t = np.arange(X.shape[1])
#     ax.plot(t, X.T, marker)
    
#     # ax.set_xlabel(f"robust={robust}|{minmax[0]:.2f}...{minmax[1]:.2f}")
#     # ax.set_ylabel('neuron #')

# def plotStateSpace(X, dim1=None, dim2=None, plotndim=2, ax=None, color=None):
#     """ general, for plotting state space, will work
#     with trajectories or pts (with variance plotted)
#     X, shape should always be (neurons, time), if is larger, then
#     will raise error. 
#     - dim1, 2, list of inds to take. if None, then will take auto (for dim1
#     takes first N depending on plotver) and dim2 takes all. length of dim1 should
#     match the plotndim.
#     - plotndim, [2, 3] whether 2d or 3d
#     """
#     assert False, "copied over from drawnn.notebooks_analy.analy_everythinguptonow_021021 Not sure if works here."
#     import seaborn as sns
    
#     if ax is None:
#         fig, ax = plt.subplots()
        
#     # check that input X is correct shape
#     assert len(X.shape)<=2
    
#     # how many neural dimensions>?
#     if dim1 is not None:
#         assert len(dim1)==plotndim
#     else:
#         dim1 = np.arange(plotndim)
    
#     # how many time bins?
#     if dim2 is None:
#         dim2 = np.arange(X.shape[1])
            
#     # PLOT
#     if plotndim==2:
#         x1 = X[dim1[0], dim2]
#         x2 = X[dim1[1], dim2]
#         ax.scatter(x1, x2, c=color)
#         ax.plot(x1, x2, '-', color=color)
#         if len(x1)>1:
#             ax.plot(x1[0], x2[0], "ok") # mark onset
#     elif plotndim==3:
#         assert False, not coded
#         # %matplotlib notebook
#         fig, axes = plt.subplots(1,2, figsize=(12,6))
#         from mpl_toolkits.mplot3d import Axes3D
    
#         # --- 1
#         ax = fig.add_subplot(121, projection='3d')
#         ax.scatter(Xsub[:,0], Xsub[:,1], Xsub[:,2], c=[x for x in Mod.A.calcNumStrokes()])
#         # --- 2
#         tasks_as_nums = mapStrToNum(Mod.Tasks["train_categories"])[1]
#         ax = fig.add_subplot(122, projection='3d')
#         ax.scatter(Xsub[:,0], Xsub[:,1], Xsub[:,2], c=tasks_as_nums)

#     # fig, ax= plt.subplots()
#     # for b in [0,1]:
#     #     X = Xmean[:,:,b]
#     #     plotStateSpace(X, ax=ax)



# class PopAnal():
#     """ for analysis of population state spaces
#     """

#     def __init__(self, X, times, chans=None, dim_units=0, 
#         stack_trials_ver="append_nan", 
#         feature_list = None, spike_trains=None,
#         print_shape_confirmation = False):
#         """ 
#         Options for X:
#         - array, where one dimensions is nunits. by dfefualt that should 
#         be dim 0. if not, then tell me in dim_units which it is.
#         - list, length trials, each element (nunits, timebins).
#         can have diff timebine, but shoudl have same nunits. 
#         Useful if, e.g., each trial different length, and want to 
#         preprocess in standard ways to make them all one stack.
#         - pandas dataframe, where each row is a trial. must have a column called
#         "neur" which is (nunits, timebins)
#         - times,  array of time values for each time bin. same for all trials.
#         --------
#         - axislabels = list of strings
#         - chans, list of ids for each chan, same len as nunits. if None, then labels them [0, 1, ..]
#         - dim_units, dimeision holding units. if is not
#         0 then will transpose to do all analyses.
#         - feature_list = list of strings, each identifying one column for X if 
#         X is datafarme, where these are features already extracted in X, useful for 
#         downstream analysis.
#         - spike_trains, list of list of spike trains, where the outer list is 
#         trials, and inner list is one for each chan.
#         ATTRIBUTES:
#         - X, (nunits, cond1, cond2, ...). NOTE, can enter different
#         shape, but tell me in dim_units which dimension is units. will then
#         reorder. *(nunits, ntrials, time)

#         """
#         self.Xdataframe = None
#         self.Xz = None

#         self.dim_units = dim_units
#         if isinstance(X, list):
#             from ..tools.timeseriestools import stackTrials
#             self.X = stackTrials(X, ver=stack_trials_ver)
#             assert dim_units==1, "are you sure? usually output of stackTrials puts trials as dim0."
#         elif isinstance(X, pd.core.frame.DataFrame):
#             self.Xdataframe = X
#             try:
#                 self.X = np.stack(X["neur"].values) # (ntrials, nunits, time)
#                 self.dim_units = 1
#             except:
#                 print("assuming you gave me dataframe where each trial has diff length neural.")
#                 print("running 'stackTrials ...")
#                 X = [x for x in X["neur"].values] # list, where each elemnent is (nunits, tbins)
#                 self.X = stackTrials(X, ver=stack_trials_ver)
#                 assert dim_units==1, "are you sure? usually output of stackTrials puts trials as dim0."

#             self.Featurelist = feature_list
#         else:
#             if len(X.shape)==2:
#                 # assume it is (nunits, timebins). unsqueeze so is (nunits, 1, timebins)
#                 self.X = np.expand_dims(X, 1)
#             else:
#                 self.X = X
        
#         self.Saved = {}
#         self.Times = times
#         assert len(times)==self.X.shape[2], "n trials doesnt match"

#         if chans is None:
#             self.Chans = range(self.X.shape[0])
#         else:
#             assert len(chans)==self.X.shape[0]
#             self.Chans = chans

#         # Spike trains
#         self.SpikeTrains = spike_trains
#         if spike_trains is not None:    
#             assert len(spike_trains)==self.X.shape[1], "doesnt match num trials"
#             for st in spike_trains:
#                 assert len(st)==self.X.shape[0], "doesnt match number chans"

#         self.preprocess()
#         if print_shape_confirmation:
#             print("Final shape of self.X; confirm that is (nunits, ntrials, time)")
#             print(self.X.shape)



#     def preprocess(self):
#         """ preprocess X, mainly so units dimension is axis 0 
#         """

#         if self.dim_units!=0:
#             self.X = np.swapaxes(self.X, 0, self.dim_units)

#     def unpreprocess(self, Xin):
#         """ undoes preprocess, useful if return values,
#         does not chagne anythings in this object, just Xin."""
#         if self.dim_units!=0:
#             Xin = np.swapaxes(Xin, 0, self.dim_units)

#         return Xin

#     def sortPop(self, dim, ver="trial_length"):
#         """ sort self.X --> self.Xsorted.
#         - dim, which dimension to sort by. 
#         0, neurons; 1, trials
#         - ver, is shortcut for filter functions.
#         """
#         from ..tools.nptools import sortPop

#         def getFilt(ver="trial_length"):
#             """ returns lambda function that can be used as filt in "sortPop"
#             """
#             if ver=="trial_length":
#                 # sort by incresauibng trial duration. 
#                 # assumes trials are concatenated ysing "append_nan" method
#                 # in PopAnal. returns index of first occurance of nan.
#         #         return lambda x: print(x.shape)
#                 # assume (<>, <>, time), where time is 
#                 def filt(x):
#                     idx = np.where(np.isnan(x[0]))[0]
#                     if len(idx)>0:
#                         # then nans exist
#                         return idx[0]
#                     else:
#                         # then return number which is one larger than what 
#                         # wuld return if last index was nan.
#                         return len(x[0])
#                 return filt
#             else:
#                 assert False, "not coded"

#         filt = getFilt(ver)
#         self.Xsorted = sortPop(self.X, dim=dim, filt=filt)
#         print(f"shape of self.X: {self.X.shape}")
#         print(f"shape of self.Xsorted: {self.Xsorted.shape}")        


#     def centerAndStack(self, return_means=False):
#         """ convert to (nunits, -1), 
#         and center each row.
#         """
#         X = self.X.copy()
#         # - reshape to N x tiembins
#         X = np.reshape(X, (X.shape[0],-1))
#         # - demean
#         means = np.mean(X, axis=1)[:, None]
#         X = X - means
        
#         self.Xcentered = X
#         self.Saved["centerAndStack_means"] = means # if want to apply same trnasformation in future data.

#     def pca(self, ver="svd", ploton=False):
#         """ perform pca
#         - saves in cache the axes, self.Saved["pca"]
#         """
        
#         self.centerAndStack()

#         if ver=="svd":

#             if self.Xcentered.shape[1]>self.Xcentered.shape[0]:
#                 # "dim 1 is time x trials..., shoudl be larger"
#                 full_matrices=False
#             else:
#                 full_matrices=True

#             u, s, v = np.linalg.svd(self.Xcentered, full_matrices=full_matrices)
#             nunits = self.X.shape[0]
#             w = s**2/(nunits-1)
            
#         elif ver=="eig":
#             Xcov = np.cov(self.Xcentered)
#             w, u = np.linalg.eig(Xcov)

#             # - plot
#             if False:
#                 import matplotlib.pyplot as plt
#                 fig, axes = plt.subplots(1,2, figsize=(10,5))
#                 axes[0].imshow(Xcov);
#                 axes[1].hist(Xcov[:]);
#                 axes[1].set_title("elements in cov mat")

#                 # - plot 
#                 fig, axes = plt.subplots(1, 2, figsize=(15,5))

#                 axes[0].plot(w, '-ok')
#                 wsum = np.cumsum(w)/np.sum(w)
#                 axes[0].plot(wsum, '-or')
#                 axes[0].set_title('eigenvals')
#                 axes[1].imshow(v)
#                 axes[1].set_title('eigen vects')
#                 axes[1].set_xlabel('vect')
#         else:
#             assert False

#         w = w/np.sum(w)

#         if ploton:
#             import matplotlib.pyplot as plt
#             # - plot 
#             fig, axes = plt.subplots(1, 2, figsize=(15,5))

#             axes[0].plot(w, '-ok')
#             axes[1].plot(np.cumsum(w)/np.sum(w), '-or')
#             axes[1].set_title('cumulative variance explained')
#             axes[1].hlines(0.9, 0, len(w))

#             axes[0].set_title('s vals')
#             # axes[1].imshow(v)
#             # axes[1].set_title('eigen vects')
#             # axes[1].set_xlabel('vect')
#         else:
#             fig = None
        
#         # === save
#         self.Saved["pca"]={"w":w, "u":u}
        
#         return fig

#     # def reproject1(self, Ndim=3):
#     #     """ reprojects neural pop onto subspace.
#     #     uses axes defined by ver. check if saved if 
#     #     not then autoamitcalyl extracst those axes
#     #     - Ndim is num axes to take."""
        
#     #     maxdim = self.X.shape[0] # max number of neurons
#     #     # if Ndim>maxdim:
#     #     #     print(f"not enough actual neurons ({maxdim}) to match desired Ndim ({Ndim})")
#     #     #     print(f"reducing Ndim to {maxdim}")
#     #     #     Ndim = min((maxdim, Ndim))

#     #     # # - get old saved
#     #     # if "pca" not in self.Saved:
#     #     #     print(f"- running {ver} for first time")
#     #     #     self.pca(ver="eig")

#     #     if True:
#     #         w = self.Saved["pca"]["w"]
#     #         u = self.Saved["pca"]["u"]
                
#     #         # - project data onto eigen
#     #         usub = u[:,:Ndim]
#     #         Xsub = usub.T @ self.Xcentered

#     #         # - reshape back to (nunits, ..., ...)
#     #         sh = list(self.X.shape)
#     #         sh[0] = Ndim
#     #         # print(self.X.shape)
#     #         # print(Ndim)
#     #         # print(Xsub.shape)
#     #         # print(self.Xcentered.shape)
#     #         # print(usub.T.shape)
#     #         # print(u.shape)
#     #         # print(u.)
#     #         Xsub = np.reshape(Xsub, sh)
#     #         # Ysub = Ysub.transpose((1, 0, 2))
#     #     else:
#     #         Xsub = self.reprojectInput(self.X, Ndim)


#     #     # -- return with units in the correct axis
#     #     return self.unpreprocess(Xsub)


#     def reproject(self, Ndim=3):
#         """ reprojects neural pop onto subspace.
#         uses axes defined by ver. check if saved if 
#         not then autoamitcalyl extracst those axes
#         - Ndim is num axes to take."""
        
#         # maxdim = self.X.shape[0] # max number of neurons
#         # if Ndim>maxdim:
#         #     print(f"not enough actual neurons ({maxdim}) to match desired Ndim ({Ndim})")
#         #     print(f"reducing Ndim to {maxdim}")
#         #     Ndim = min((maxdim, Ndim))

#         # # - get old saved
#         # if "pca" not in self.Saved:
#         #     print(f"- running {ver} for first time")
#         #     self.pca(ver="eig")

#         Xsub = self.reprojectInput(self.X, Ndim)

#         # -- return with units in the correct axis
#         return self.unpreprocess(Xsub)

#     def reprojectInput(self, X, Ndim=3, Dimslist = None):
#         """ same as reproject, but here project activity passed in 
#         X.
#         - X, (nunits, *), as many dim as wanted, as long as first dim is nunits (see reproject
#         for how to deal if first dim is not nunits)
#         - Dimslist, insteast of Ndim, can give me list of dims. Must leave Ndim None.
#         RETURNS:
#         - Xsub, (Ndim, *) [X not modified]
#         NOTE: 
#         - This also applis centering tranformation to X using the saved mean of data
#         used to compute the PCA space.
#         """

#         nunits = X.shape[0]
#         sh = list(X.shape) # save original shape.

#         if Dimslist is not None:
#             assert Ndim is None, "choose wiether to take the top N dim (Ndim) or to take specific dims (Dimslist)"
#             numdims = len(Dimslist)
#         else:
#             assert Ndim is not None
#             numdims = Ndim

#         if numdims>nunits:
#             print(f"not enough actual neurons ({nunits}) to match desired num dims ({numdims})")
#             assert False
#             # print(f"reducing Ndim to {nunits}")
#             # Ndim = min((nunits, Ndim))

#         # - get old saved
#         if "pca" not in self.Saved:
#             print(f"- running {ver} for first time")
#             self.pca(ver="eig")

#         # 1) center and stack
#         # X = X.copy()
#         # - reshape to N x tiembins
#         X = np.reshape(X, (nunits,-1)) # (nunits, else)
#         # - demean
#         X = X - self.Saved["centerAndStack_means"] # demean

#         # 2) 
#         # w = self.Saved["pca"]["w"]
#         u = self.Saved["pca"]["u"]
#         if Ndim is None:
#             usub = u[:,Dimslist]
#         else:
#             usub = u[:,:Ndim]
#         Xsub = usub.T @ X

#         # 3) reshape back to origuinal
#         sh[0] = numdims
#         Xsub = np.reshape(Xsub, sh)

#         return Xsub


#     ### ANALYSIS
#     def dataframeByTrial(self, dim_trial = 1, columns_to_add = None):
#         """ useful preprocess before do analyses.
#         converts X to dataframe, where row is trial.
#         - columns_to_add, dict, where each item is new column.
#         entry must be same length as num trials.
#         - dim_trial, which dim is trialnum?
#         """
#         assert False, "[DEPRECATED] - isntead, pass in list of dicts to PopAnal directly, this ensures keeps all fields in the dicts"
#         ntrials = self.X.shape[dim_trial]


#         assert dim_trial==1, "below code assumes 1. indexing is not as flexible"
#         dat = []
#         for i in range(ntrials):
#             dat.append({
#                 "x":self.X[:, i, ...],
#                 "trial":i
#                 })
#         df = pd.DataFrame(dat)

#         if columns_to_add is not None:
#             for k, v in columns_to_add.items():
#                 assert len(v)==ntrials
#                 df[k] = v

#         return df

#     # Data Transformations
#     def zscoreFr(self, groupby=[]):
#         """ z-score firing rates using across trial mean and std.
#         - groupby, what mean and std to use. if [], then all trials
#         combined (so all use same mean and std). if ["circ_binned"], 
#         then asks for each trial what its bin (for circuiliatry) and then
#         uses mean and std within that bin. Must have binned data first 
#         before this (see binFeatures)
#         ==== RETURN:
#         modifies self.Xdataframe, adds column "neur_z"
#         """
#         from pythonlib.tools.pandastools import applyFunctionToAllRows

#         # 1. get mean and std.
#         _, colname_std = self.aggregate(groupby, "trial", "std", "std", return_new_col_name=True)
#         _, colname_mean = self.aggregate(groupby, "trial", "mean", "mean", return_new_col_name=True)

#         # 2. take zscore
#         def F(x):
#             # returns (neur, time) fr in z-score units
#             return (x["neur"] - x[colname_mean])/x[colname_std]

#         self.Xdataframe = applyFunctionToAllRows(self.Xdataframe, F, newcolname="neur_z")

#     def zscoreFrNotDataframe(self):
#         """ z-score across trials and time bins, separately for each chan
#         RETURNS:
#         - modifies self.Xz
#         - return self.Xz
#         """

#         X = self.X

#         # reshape to (nchans, trials*time)
#         x = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
#         xstd = np.std(x, axis=1)
#         xmean = np.mean(x, axis=1)
#         xstd = xstd.reshape(xstd.shape[0], 1, 1)
#         xmean = xmean.reshape(xmean.shape[0], 1, 1)

#         self.Xz = (X - xmean)/xstd
#         return self.Xz


#     def binFeatures(self, nbins, feature_list=None):
#         """ assign bin to each trial, based on its value for feature
#         in feature_list.
#         - nbins, int, will bin by percentile (can modify to do uniform)
#         - feature_list, if None, then uses self.Featurelist. otherwise give
#         list of strings, but these must already be columnes in self.Xdataframe
#         === RETURNS
#         self.Xdataframe modifed to have new columns, e..g, <feat>_bin
#         """
#         if feature_list is None:
#             feature_list = self.Featurelist

#         for k in feature_list:
#             kbin = self.binColumn(k, nbins)


#     def binColumn(self, col_to_bin, nbins):
#         """ assign bin value to each trial, by binning based on
#         scalar variable.
#         - modifies in place P.Xdataframe, appends new column.
#         """
#         from ..tools.pandastools import binColumn
#         new_col_name = binColumn(self.Xdataframe, col_to_bin=col_to_bin, nbins=nbins)
#         return new_col_name

#     def aggregate(self, groupby, axis, agg_method="mean", new_col_suffix="agg", 
#         force_redo=False, return_new_col_name = False, fr_use_this = "raw"):
#         """ get aggregate pop activity in flexible ways.
#         - groupby is how to group trials (rows of self.Xdataframe). 
#         e.g., if groupby = [], then combines all trials into one aggreg.
#         e.g., if groupby = "circ_binned", then groups by bin.
#         e.g., if groupby = ["circ_binned", "angle_binned"], then conjunction
#         - axis, how to take mean (after stacking all trials after grouping)
#         e.g., if string, then is shortcut.
#         e.g., if number then is axis.
#         - new_col_suffix, for new col name.
#         - agg_method, how to agg. could be string, or could be function.
#         - fr_use_this, whether to use raw or z-scored (already done).
#         RETURNS:
#         - modifies self.Xdataframe, with one new column, means repopulated.
#         - also the aggregated datagframe XdataframeAgg
#         """
#         from ..tools.pandastools import aggregThenReassignToNewColumn
#         if isinstance(axis, str):
#             # shortcuts
#             if axis=="trial":
#                 # average over trials.
#                 axis = 0
#             else:
#                 assert False, "not coded"

#         if isinstance(agg_method, str):
#             def F(x):
#                 if fr_use_this=="raw":
#                     X = np.stack(x["neur"].values)
#                 elif fr_use_this=="z":
#                     X = np.stack(x["neur_z"].values)
#                 # expect X to be (ntrials, nunits, time)
#                 # make axis accordinyl.
#                 if agg_method=="mean":
#                     Xagg = np.mean(X, axis=axis)
#                 elif agg_method=="std":
#                     Xagg = np.std(X, axis=axis)
#                 elif agg_method=="sem":
#                     Xagg = np.std(X, axis=axis)/np.sqrt(X.shape[0])
#                 else:
#                     assert False, "not coded"
#                 return Xagg
#         else:
#             F = agg_method
#             assert callable(F)

#         if len(groupby)==0:
#             new_col_name = f"alltrials_{new_col_suffix}"
#         elif isinstance(groupby, str):
#             new_col_name = f"{groupby}_{new_col_suffix}"
#         else:
#             new_col_name = f"{'_'.join(groupby)}_{new_col_suffix}"

#         # if already done, don't run
#         if new_col_name in self.Xdataframe.columns:
#             if not force_redo:
#                 print(new_col_name)
#                 assert False, "this colname already exists, force overwrite if actualyl want to run again."

#         [self.Xdataframe, XdataframeAgg] = aggregThenReassignToNewColumn(self.Xdataframe, F, 
#             groupby, new_col_name, return_grouped_df=True)

#         if return_new_col_name:
#             return XdataframeAgg, new_col_name
#         else:
#             return XdataframeAgg

#     ####################### SLICING
#     def slice_by_time_window(self, t1, t2, return_as_popanal=False):
#         """ Slice population by time window, where
#         time is based on self.Times
#         PARAMS;
#         - t1, t2, start and end time for slicing
#         RETURNS:
#         - np array, (nchans, ntrials, timesliced)
#         """
#         inds = (self.Times>=t1) & (self.Times<=t2)
#         x_windowed = self.X[:, :, inds]
#         times = self.Times[inds]

#         if return_as_popanal:
#             PA = PopAnal(x_windowed, times, chans=self.Chans, print_shape_confirmation=False)
#             return PA
#         else:
#             return x_windowed, times

#     def slice_by_trial(self, trials, version="raw", return_as_popanal=False):
#         """ Slice activity to only get these trials, returned as popanal
#         if return_as_popanal is True
#         PARAMS:
#         - trials, list of ints, indices into dim 1 of self.X
#         """

#         X = self.extract_activity(version=version)
#         X = X[:, trials, :]
#         if return_as_popanal:
#             PA = PopAnal(X, self.Times, chans=self.Chans, print_shape_confirmation=False)
#             return PA
#         else:
#             return X

#     def slice_by_chan(self, chans, version="raw", return_as_popanal=True, 
#             chan_inputed_row_index=False,):
#         """ Slice data to keep only subset of channels
#         PARAMS;
#         - chans, list of chan labels, These are NOT the row indices, but are instead
#         the chan labels in self.Chans. To use row indices (0,1,2, ...), make
#         chan_inputed_row_index=True. 
#         (NOTE: will be sorted as in chans)
#         RETURNS:
#         - EIther:
#         --- X, np array, shape
#         --- PopAnal object (if return_as_popanal)
#         """

#         # convert from channel labels to row indices
#         if chan_inputed_row_index:
#             inds = chans
#         else:
#             inds = [self.index_find_this_chan(ch) for ch in chans]

#         # Slice
#         X = self.extract_activity(version=version)
#         X = X[inds, :, :]
#         if return_as_popanal:
#             PA = PopAnal(X, self.Times, chans=chans)
#             return PA
#         else:
#             return X

#     def copy(self):
#         """ Returns a copy. 
#         Actually does this by slicing by trial, but entering all chans...
#         So will not copy all attributes...
#         """
#         trials = range(self.X.shape[1])
#         return self.slice_by_trial(trials, return_as_popanal=True)

#     def mean_over_time(self, version="raw", return_as_popanal=False):
#         """ Return X, but mean over time,
#         shape (nchans, ntrials)
#         """
#         X = self.extract_activity(version = version)
#         if return_as_popanal:
#             Xnew = np.mean(X, axis=2, keepdims=True)
#             return PopAnal(Xnew, times=np.array([0.]), chans=self.Chans, print_shape_confirmation=False)
#         else:
#             Xnew = np.mean(X, axis=2, keepdims=False)
#             return Xnew

#     def mean_over_trials(self, version="raw"):
#         """ Return X, but mean over trials,
#         out shape (nchans, 1, time)
#         """
#         X = self.extract_activity(version=version)
#         return np.mean(X, axis=1, keepdims=True)


#     def agg_by_trialgrouping(self, groupdict, version="raw", return_as_popanal=True):
#         """ aggreagate so that trials dimension is reduced,
#         by collecting multiple trials into single groups
#         by taking mean over trials. 
#         PARAMS:
#         - groupdict, {groupname:trials_list}, output from
#         pandatools...
#         RETURNS:
#         - eitehr PopAnal or np array, with data shape = (nchans, ngroups, time),
#         with ngroups replacing ntrials after agging.
#         """

#         # 1) Collect X (trial-averaged) for each group
#         list_x = []
#         for grp, inds in groupdict.items():
#             PAthis = self.slice_by_trial(inds, version=version, return_as_popanal=True)
#             x = PAthis.mean_over_trials()
#             list_x.append(x)

#         # 2) Concatenate all groups
#         X = np.concatenate(list_x, axis=1)

#         # 3) Convert to PA
#         if return_as_popanal:
#             # assert PAnew.shape[1]==len(groupdict)
#             return PopAnal(X, PAthis.Times, PAthis.Chans)
#         else:
#             return X

#     ################ INDICES
#     def index_find_this_chan(self, chan):
#         """ Returns the index (into self.X[:, index, :]) for this
#         chan, looking into self.Chans, tyhe lables.
#         PARAMS:
#         - chan, label, as in self.Chans
#         RETURNS;
#         - index, see above
#         """

#         for i, ch in enumerate(self.Chans):
#             if ch==chan:
#                 return i
#         assert False, "this chan doesnt exist in self.Chans"


#     #### EXTRACT ACTIVITY
#     def extract_activity(self, trial=None, version="raw"):
#         """ REturn activity for this trial
#         PARAMS:
#         - trial, int, if None, then returns over all trials
#         - version, string, ee.g. {'raw', 'z'}
#         RETURNS:
#         - X, shape (nchans, 1, ntime), or (nchans, ntrials, ntime, if trial is None)
#         """

#         if version=="raw":
#             if trial is None:
#                 return self.X
#             else:
#                 return self.X[:, trial, :]
#         elif version=="z":
#             assert self.Xz is not None, "need to do zscore first"
#             if trial is None:
#                 return self.Xz
#             else:
#                 return self.Xz[:, trial, :]
#         else:
#             print(version)
#             assert False


#     ### PLOTTING
#     def plotNeurHeat(self, trial, version="raw", **kwargs):
#         X = self.extract_activity(trial, version)
#         return plotNeurHeat(X, **kwargs)

#     def plotNeurTimecourse(self, trial, version="raw", **kwargs):
#         X = self.extract_activity(trial, version)
#         return plotNeurTimecourse(X, **kwargs)



# def extract_neural_snippets_aligned_to(MS, DS, 
#     align_to = "go_cue", 
#     t1_relonset = -0.4, t2_rel = 0):
#     """ Extract neural data, snippets, aligned to strokes, currently taking
#     alignment times relative to trial events, so only really makes sense for 
#     single-stroke trials, or for aligning to strokes directly
#     PARAMS:
#     - MS, MultSession
#     - DS, DatStrokes
#     - align_to, str, what to align snips to 
#     RETURNS:
#     - PAall, PopAnal for all snippets
#     (Also modifies DS, adding column: neural_pop_slice)
#     """

#     # For each stroke in DS, get its neural snippet

#     # # --- PARAMS
#     # align_to = "go_cue"
#     # # align_to = "on_stroke_1"
#     # t1_relonset = -0.4
#     # # t2_ver = "onset"
#     # t2_rel = 0

#     list_xslices = []

#     ParamsDict = {}
#     ParamsDict["align_to"] = align_to
#     ParamsDict["t1_relonset"] = t1_relonset
#     # ParamsDict["t2_ver"] = t2_ver 
#     ParamsDict["t2_rel"] = t2_rel 

#     for ind in range(len(DS.Dat)):
        
#         if ind%200==0:
#             print("index strokes: ", ind)

#         # --- BEH
#         trialcode = DS.Dat.iloc[ind]["dataset_trialcode"]
#         indstrok = DS.Dat.iloc[ind]["stroke_index"]
        
#         # --- NEURAL
#         # Find the trial in neural data
#         SNthis, trial_neural = MS.index_convert(trialcode)[:2]
#         trial_neural2 = SNthis.datasetbeh_trialcode_to_trial(trialcode)
#         assert trial_neural==trial_neural2
#         del trial_neural2
        
#         # get strokes ons and offs
#         if align_to=="stroke_onset":
#             # Then align to onset of stroke that is in DS
#             # Sanity check (confirm that timing for neural is same as timing saved in dataset)
#             ons, offs = SNthis.strokes_extract_ons_offs(trial_neural)
#             timeon_neural = ons[indstrok]
#             timeoff_neural = offs[indstrok]    
#             timeon = DS.Dat.iloc[ind]["time_onset"]
#             timeoff = DS.Dat.iloc[ind]["time_offset"]
#             assert np.isclose(timeon, timeon_neural)
#             assert np.isclose(timeoff, timeoff_neural)
#             time_align = timeon
#         else:
#             # Align to timing of things in trials
#             time_align = SNthis.events_get_time_all(trial_neural, list_events=[align_to])[align_to]
        
        
#         # --- POPANAL
#         # Extract the neural snippet
#         t1 = time_align + t1_relonset
#         t2 = time_align + t2_rel
#         PA = SNthis.popanal_generate_save_trial(trial_neural, print_shape_confirmation=False, 
#                                             clean_chans=True, overwrite=True)
#         PAslice = PA.slice_by_time_window(t1, t2, return_as_popanal=True)
        
#         # save this slice
#         list_xslices.append(PAslice)

#     # Save into DS
#     DS.Dat["neural_pop_slice"] = list_xslices    

#     ##### Combine all strokes into a single PA (consider them "trials")

#     from quantities import s
#     from pythonlib.neural.population import PopAnal

#     list_PAslice = DS.Dat["neural_pop_slice"].tolist()
#     CHANS = list_PAslice[0].Chans
#     TIMES = (list_PAslice[0].Times - list_PAslice[0].Times[0]) + t1_relonset*s # times all as [-predur, ..., postdur]

#     # get list of np arrays
#     Xall = np.concatenate([pa.X for pa in list_PAslice], axis=1)
#     PAall = PopAnal(Xall, TIMES, CHANS)

#     return PAall



# # Which dataset to use to construct PCA?
# def pca_make_space(PA, DF, trial_agg_method, trial_agg_grouping, time_agg_method=None, ploton=True):
#     """ Make a PopAnal object where the dataspoints are
#     used to construct a low-D PCA space
#     PARAMS:
#     - PA, popanal object, holds all data. 
#     - DF, dataframe, with one column for each categorical variable you care about (in DATAPLOT_GROUPING_VARS).
#     The len(DF) must equal num trials in PA (asserts this)
#     - trial_agg_grouping, list of str defining how to group trials, e.g,
#     ["shape_oriented", "gridloc"]
#     """
#     # First, decide whether to take mean over some way of grouping trials
#     if trial_agg_method==None:
#         # Then dont aggregate by trials
#         PApca = PA.copy()
#     elif trial_agg_method=="grouptrials":
#         # Then take mean over trials, after grouping, so shape
#         # output is (nchans, ngrps, time), where ngrps < ntrials
#         from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
#         groupdict = grouping_append_and_return_inner_items(DF, trial_agg_grouping)
#         # groupdict = DS.grouping_append_and_return_inner_items(trial_agg_grouping)
#         PApca = PA.agg_by_trialgrouping(groupdict)        
#     else:
#         print(trial_agg_method)
#         assert False
        
#     # second, whether to agg by time (optional). e..g, take mean over time
#     if time_agg_method=="mean":
#         PApca = PApca.mean_over_time(return_as_popanal=True)
#     else:
#         assert time_agg_method==None

#     fig = PApca.pca("svd", ploton=ploton)
    
#     return PApca, fig

# def compute_data_projections(PA, DF, MS, VERSION, REGIONS, DATAPLOT_GROUPING_VARS, 
#                             pca_trial_agg_grouping = ["gridloc"], pca_trial_agg_method = "grouptrials", 
#                             pca_time_agg_method = None, ploton=True):
#     """
#     Combines population nerual data (PA) and task/beh features (DF) and does (i) goruping of trials,
#     (ii) data processing, etc.
#     Process data for plotting, especialyl gropuping trials based on categorical
#     features (e..g, shape). A useful feature is projecting data to a new space
#     defined by PCA, where PCA computed on aggregated data, e..g, first get mean activity
#     for each location, then PCA on those locations (like demixed PCA).
    
#     PARAMS:
#     - PA, popanal object, holds all data. 
#     - DF, dataframe, with one column for each categorical variable you care about (in DATAPLOT_GROUPING_VARS).
#     The len(DF) must equal num trials in PA (asserts this)
#     - MS, MultSession object, holding the "raw" neural data, has useful metadata needed for this, e.g,,
#     extracting brain regions, only good sitese, etc.
#     - VERSION, str, how to represent data. does all transfomrations required.
#     --- if "PCA", then will need the params starting with pca_*:
#     - REGIONS, list of str, brain regions, prunes data to just this
#     - DATAPLOT_GROUPING_VARS, lsit of strings, each a variable, takes conjunction to make gorups. this 
#     controls data represtations, but doesnt not affect the pca space.
#     - pca_trial_agg_grouping, list of str each a category, takes conjunction to defines the groups that are then
#     used for PCA.
#     - pca_trial_agg_method, pca_time_agg_method str, both strings, how to aggregate (mean) data
#     before doing PCA> grouptrials' --> take mean before PC
#     """
#     from pythonlib.tools.pandastools import applyFunctionToAllRows, grouping_append_and_return_inner_items
#     import scipy.stats as stats

#     assert len(DF)==PA.X.shape[1], "num trials dont match"


#     # How to transform data
#     if VERSION=="raw":
#         YLIM = [0, 100]
#         VERSION_DAT = "raw"
#     elif VERSION=="z":
#         YLIM = [-2, 2]
#         VERSION_DAT = "raw"
#     elif VERSION=="PCA":
#         YLIM = [-50, 50]
#         VERSION_DAT = "raw"
#     else:
#         assert False

#     # Slice to desired chans
#     CHANS = MS.sitegetter_all(REGIONS)
#     assert len(CHANS)>0
#     PAallThis = PA.slice_by_chan(CHANS, VERSION_DAT, True)

#     # Construct PCA space
#     if VERSION=="PCA":
#         # Construct PCA space
#         # PApca, figs_pca = pca_make_space(PAallThis, pca_trial_agg_method, pca_trial_agg_grouping, pca_time_agg_method, ploton=ploton)
#         PApca, figs_pca = pca_make_space(PAallThis, DF, pca_trial_agg_method, pca_trial_agg_grouping, pca_time_agg_method, ploton=ploton)
#     else:
#         PApca = None
#         figs_pca = None

#     # # Get list of sites
#     # CHANS = SN.sitegetter_all(list_regions=REGIONS, clean=CLEAN)

#     ################ COLLECT DATA TO PLOT
#     # Generate grouping dict for data to plot
# #     gridloc = (-1,-1) Obsolete
#     groupdict = grouping_append_and_return_inner_items(DF, DATAPLOT_GROUPING_VARS)
#     # groupdict = generate_data_groupdict(DATAPLOT_GROUPING_VARS, GET_ONE_LOC=False, gridloc=None, PRUNE_SHAPES=False)

#     # - for each group, get a slice of PAall
#     DatGrp = []
#     for grp, inds in groupdict.items():
#         pa = PAallThis.slice_by_trial(inds, version=VERSION_DAT, return_as_popanal=True)
#         DatGrp.append({
#             "group":grp,
#             "PA":pa})

#     # For each group, get a vector represenetation    
#     for dat in DatGrp:
#         pa = dat["PA"]
#         x = pa.mean_over_time()

#         # PCA?
#         if VERSION=="PCA":
#             # project to space constructed using entire dataset
#             x = PApca.reprojectInput(x, len(PAallThis.Chans))
#         dat["X_timemean"] = x
#         dat["X_timetrialmean"] = np.mean(x, 1)
#         dat["X_timemean_trialsem"] = stats.sem(x, 1)
        
#     # Convert to dataframe and append columns indicate labels
#     DatGrpDf = pd.DataFrame(DatGrp)
#     for i, var in enumerate(DATAPLOT_GROUPING_VARS):
#         def F(x):
#             return x["group"][i]
#         DatGrpDf = applyFunctionToAllRows(DatGrpDf, F, var)

#     ################## PLOTS
#     if ploton:
#         # PLOT: distribution of FR (mean vec) for each shape
#         from pythonlib.tools.plottools import subplot_helper
#         getax, figholder, nplots = subplot_helper(2, 10, len(DatGrp), SIZE=4, ASPECTWH=2, ylim=YLIM)
#         for i, dat in enumerate(DatGrp):
#             ax = getax(i)
#             x = dat["X_timemean"]
#             ax.plot(PAallThis.Chans, x, '-', alpha=0.4);

#         # PLOT, get mean vector for each shape, and plot overlaied
#         fig, ax = plt.subplots(1,1, figsize=(15, 4))
#         for i, dat in enumerate(DatGrp):
#             x = dat["X_timetrialmean"]
#             xerr = dat["X_timemean_trialsem"]
#             ax.plot(PAallThis.Chans, x, '-', alpha=0.4);
#         ax.set_ylim(YLIM)

#         print("TODO: return figs for saving")
        
#     return DatGrp, groupdict, DatGrpDf


# def datgrp_flatten_to_dattrials(DatGrp, DATAPLOT_GROUPING_VARS):
#     """ Takes DatGrp, which is one entry per group,
#     and flattens to DfTrials, which is one entry per trial,
#     and returns as DataFrame
#     PARAMS;
#     - DatGrp, output of compute_data_projections
#     - DATAPLOT_GROUPING_VARS, used for breaking out each variable into its own column.
#     """
#     out = []
#     for Dat in DatGrp:

#         # extract group-level things
#         grp = Dat["group"]
#         X = Dat["X_timemean"] # processed X (nchans, ntrials)
#         ntrials = X.shape[1]

#         # collect one row for each trial
#         for i in range(ntrials):

#             # Add this trial's neural data
#             out.append({
#                 "x":X[:, i],
#                 "grp":grp,
#             })

#             # break out each label dimension
#             for i, varname in enumerate(DATAPLOT_GROUPING_VARS):
#                 out[-1][varname] = grp[i]

#     DfTrials = pd.DataFrame(out)
#     return DfTrials

# def dftrials_centerize_by_group_mean(DfTrials, grouping_for_mean):
#     """ 
#     For each row, subtract the group mean for mean neural activiy;
#     PARAMS:
#     - DfTrials, df, each row a trial
#     - grouping_for_mean, list of str, conjunction is a group, e..g, 
#     ["shape_oriented", "gridsize"]
#     RETURNS:
#     - novel dataframe, same size as input, but with extra column with 
#     name "x_mean"
#     """
#     from pythonlib.tools.pandastools import aggregThenReassignToNewColumn, append_col_with_grp_index, applyFunctionToAllRows

#     # 1) Get grouping, then get mean, then place back into each row.
#     def F(x):
#         """ get mean activity across trials
#         """
#         import numpy as np
#         return np.mean(x["x"])
#     NEWCOL = "x_grp_mean"
#     dfnew = aggregThenReassignToNewColumn(DfTrials, F, grouping_for_mean, NEWCOL)

#     # 2) Append group index as tuple
#     dfnew = append_col_with_grp_index(dfnew, grouping_for_mean, "grp", False)

#     # 3) For each row, subtract its group's mean.
#     def F(x):
#         return x["x"] - x[NEWCOL]
#     print("**********", dfnew.columns)
#     dfnew = applyFunctionToAllRows(dfnew, F, "x_centered")
#     return dfnew
import numpy as np
import pandas as pd


def subsample_rand(X, n_rand):
    """ get random sample (dim 0), or return all if n_rand larget than X.
    """
    import random
    if n_rand>X.shape[0]:
        return X, list(range(X.shape[0]))

    inds = random.sample(range(X.shape[0]), n_rand)
    return X[inds, ...], inds

def plotNeurHeat(X, ax=None, barloc="right", robust=True, zlims = None):
    """ plot heatmap for data X.
    X must be (neuron, time)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
    
    # X = self.activations[pop][tasknum]
    X_nonan = X[:]
    X_nonan = X_nonan[~np.isnan(X_nonan)]
    minmax = (np.min(X_nonan), np.max(X_nonan))

    # plot
    if zlims is not None: 
        sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
               robust=robust, vmin=zlims[0], vmax=zlims[1])
    else:
        sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
               robust=robust)
    # ax.set_title(f"{pop}|{minmax[0]:.2f}...{minmax[1]:.2f}")
    ax.set_xlabel(f"robust={robust}|{minmax[0]:.2f}...{minmax[1]:.2f}")
    ax.set_ylabel('neuron #')
        
def plotNeurTimecourse(X, Xerror=None, ax=None, n_rand=None, marker="-"):
    """ Plot overlaid timecourses. 
    - X, (neuron, time)
    - Xerror, (neuron, time), to add errorbars)
    """

    import seaborn as sns
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
    
    # X = self.activations[pop][tasknum]
    X_nonan = X[:]
    X_nonan = X_nonan[~np.isnan(X_nonan)]
    minmax = (np.min(X_nonan), np.max(X_nonan))

    if n_rand is not None:
        X, indsrand = subsample_rand(X, n_rand)
    t = np.arange(X.shape[1])
    ax.plot(t, X.T, marker)
    
    # ax.set_xlabel(f"robust={robust}|{minmax[0]:.2f}...{minmax[1]:.2f}")
    # ax.set_ylabel('neuron #')


class PopAnal():
    """ for analysis of population state spaces
    """

    def __init__(self, X, axislabels=None, dim_units=0, stack_trials_ver="append_nan", 
        feature_list = None):
        """ 
        Options for X:
        - array, where one dimensions is nunits. by dfefualt that should 
        be dim 0. if not, then tell me in dim_units which it is.
        - list, length trials, each element (nunits, timebins).
        can have diff timebine, but shoudl have same nunits. 
        Useful if, e.g., each trial different length, and want to 
        preprocess in standard ways to make them all one stack.
        - pandas dataframe, where each row is a trial. must have a column called
        "neur" which is (nunits, timebins)
        --------
        - axislabels = list of strings
        - dim_units, dimeision holding units. if is not
        0 then will transpose to do all analyses.
        - feature_list = list of strings, each identifying one column for X if 
        X is datafarme, where these are features already extracted in X, useful for 
        downstream analysis.
        ATTRIBUTES:
        - X, (nunits, cond1, cond2, ...). NOTE, can enter different
        shape, but tell me in dim_units which dimension is units. will then
        reorder. *(nunits, ntrials, time)

        """
        self.Xdataframe = None
        self.dim_units = dim_units
        if isinstance(X, list):
            from ..tools.timeseriestools import stackTrials
            self.X = stackTrials(X, ver=stack_trials_ver)
            assert dim_units==1, "are you sure? usually output of stackTrials puts trials as dim0."
        elif isinstance(X, pd.core.frame.DataFrame):
            self.Xdataframe = X
            try:
                self.X = np.stack(X["neur"].values) # (ntrials, nunits, time)
                self.dim_units = 1
            except:
                print("assuming you gave me dataframe where each trial has diff length neural.")
                print("running 'stackTrials ...")
                X = [x for x in X["neur"].values] # list, where each elemnent is (nunits, tbins)
                self.X = stackTrials(X, ver=stack_trials_ver)
                assert dim_units==1, "are you sure? usually output of stackTrials puts trials as dim0."

            self.Featurelist = feature_list
        else:
            self.X = X
        self.Saved = {}
        self.preprocess()
        print("Final shape of self.X; confirm that is (nunits, ntrials, time)")
        print(self.X.shape)



    def preprocess(self):
        """ preprocess X, mainly so units dimension is axis 0 
        """

        if self.dim_units!=0:
            self.X = np.swapaxes(self.X, 0, self.dim_units)

    def unpreprocess(self, Xin):
        """ undoes preprocess, useful if return values,
        does not chagne anythings in this object, just Xin."""
        if self.dim_units!=0:
            Xin = np.swapaxes(Xin, 0, self.dim_units)

        return Xin

    def sortPop(self, dim, ver="trial_length"):
        """ sort self.X --> self.Xsorted.
        - dim, which dimension to sort by. 
        0, neurons; 1, trials
        - ver, is shortcut for filter functions.
        """
        from ..tools.nptools import sortPop

        def getFilt(ver="trial_length"):
            """ returns lambda function that can be used as filt in "sortPop"
            """
            if ver=="trial_length":
                # sort by incresauibng trial duration. 
                # assumes trials are concatenated ysing "append_nan" method
                # in PopAnal. returns index of first occurance of nan.
        #         return lambda x: print(x.shape)
                # assume (<>, <>, time), where time is 
                def filt(x):
                    idx = np.where(np.isnan(x[0]))[0]
                    if len(idx)>0:
                        # then nans exist
                        return idx[0]
                    else:
                        # then return number which is one larger than what 
                        # wuld return if last index was nan.
                        return len(x[0])
                return filt
            else:
                assert False, "not coded"

        filt = getFilt(ver)
        self.Xsorted = sortPop(self.X, dim=dim, filt=filt)
        print(f"shape of self.X: {self.X.shape}")
        print(f"shape of self.Xsorted: {self.Xsorted.shape}")        


    def centerAndStack(self, return_means=False):
        """ convert to (nunits, -1), 
        and center each row.
        """
        X = self.X.copy()
        # - reshape to N x tiembins
        X = np.reshape(X, (X.shape[0],-1))
        # - demean
        means = np.mean(X, axis=1)[:, None]
        X = X - means
        
        self.Xcentered = X
        self.Saved["centerAndStack_means"] = means # if want to apply same trnasformation in future data.

    def pca(self, ver="eig", ploton=False):
        """ perform pca
        - saves in cache the axes, self.Saved["pca"]
        """
        
        self.centerAndStack()

        if ver=="svd":
            u, s, v = np.linalg.svd(self.Xcentered)
            w = s**2/(nunits-1)
            
        elif ver=="eig":
            Xcov = np.cov(self.Xcentered)
            w, u = np.linalg.eig(Xcov)

            # - plot
            if False:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1,2, figsize=(10,5))
                axes[0].imshow(Xcov);
                axes[1].hist(Xcov[:]);
                axes[1].set_title("elements in cov mat")

                # - plot 
                fig, axes = plt.subplots(1, 2, figsize=(15,5))

                axes[0].plot(w, '-ok')
                wsum = np.cumsum(w)/np.sum(w)
                axes[0].plot(wsum, '-or')
                axes[0].set_title('eigenvals')
                axes[1].imshow(v)
                axes[1].set_title('eigen vects')
                axes[1].set_xlabel('vect')

        w = w/np.sum(w)

        if ploton:
            import matplotlib.pyplot as plt
            # - plot 
            fig, axes = plt.subplots(1, 2, figsize=(15,5))

            axes[0].plot(w, '-ok')
            axes[1].plot(np.cumsum(w)/np.sum(w), '-or')
            axes[1].hlines(0.9, 0, len(w))

            axes[0].set_title('s vals')
            # axes[1].imshow(v)
            # axes[1].set_title('eigen vects')
            # axes[1].set_xlabel('vect')
        
        # === save
        self.Saved["pca"]={"w":w, "u":u}
        if ploton:
            return fig

    # def reproject1(self, Ndim=3):
    #     """ reprojects neural pop onto subspace.
    #     uses axes defined by ver. check if saved if 
    #     not then autoamitcalyl extracst those axes
    #     - Ndim is num axes to take."""
        
    #     maxdim = self.X.shape[0] # max number of neurons
    #     # if Ndim>maxdim:
    #     #     print(f"not enough actual neurons ({maxdim}) to match desired Ndim ({Ndim})")
    #     #     print(f"reducing Ndim to {maxdim}")
    #     #     Ndim = min((maxdim, Ndim))

    #     # # - get old saved
    #     # if "pca" not in self.Saved:
    #     #     print(f"- running {ver} for first time")
    #     #     self.pca(ver="eig")

    #     if True:
    #         w = self.Saved["pca"]["w"]
    #         u = self.Saved["pca"]["u"]
                
    #         # - project data onto eigen
    #         usub = u[:,:Ndim]
    #         Xsub = usub.T @ self.Xcentered

    #         # - reshape back to (nunits, ..., ...)
    #         sh = list(self.X.shape)
    #         sh[0] = Ndim
    #         # print(self.X.shape)
    #         # print(Ndim)
    #         # print(Xsub.shape)
    #         # print(self.Xcentered.shape)
    #         # print(usub.T.shape)
    #         # print(u.shape)
    #         # print(u.)
    #         Xsub = np.reshape(Xsub, sh)
    #         # Ysub = Ysub.transpose((1, 0, 2))
    #     else:
    #         Xsub = self.reprojectInput(self.X, Ndim)


    #     # -- return with units in the correct axis
    #     return self.unpreprocess(Xsub)


    def reproject(self, Ndim=3):
        """ reprojects neural pop onto subspace.
        uses axes defined by ver. check if saved if 
        not then autoamitcalyl extracst those axes
        - Ndim is num axes to take."""
        
        # maxdim = self.X.shape[0] # max number of neurons
        # if Ndim>maxdim:
        #     print(f"not enough actual neurons ({maxdim}) to match desired Ndim ({Ndim})")
        #     print(f"reducing Ndim to {maxdim}")
        #     Ndim = min((maxdim, Ndim))

        # # - get old saved
        # if "pca" not in self.Saved:
        #     print(f"- running {ver} for first time")
        #     self.pca(ver="eig")

        Xsub = self.reprojectInput(self.X, Ndim)

        # -- return with units in the correct axis
        return self.unpreprocess(Xsub)

    def reprojectInput(self, X, Ndim=3, Dimslist = None):
        """ same as reproject, but here project activity passed in 
        X.
        - X, (nunits, *), as many dim as wanted, as long as first dim is nunits (see reproject
        for how to deal if first dim is not nunits)
        - Dimslist, insteast of Ndim, can give me list of dims. Must leave Ndim None.
        ==== RETURNS
        - Xsub, (Ndim, *) [X not modified]
        """

        nunits = X.shape[0]
        sh = list(X.shape) # save original shape.

        if Dimslist is not None:
            assert Ndim is None, "choose wiether to take the top N dim (Ndim) or to take specific dims (Dimslist)"
            numdims = len(Dimslist)
        else:
            assert Ndim is not None
            numdims = Ndim

        if numdims>nunits:
            print(f"not enough actual neurons ({nunits}) to match desired num dims ({numdims})")
            assert False
            # print(f"reducing Ndim to {nunits}")
            # Ndim = min((nunits, Ndim))

        # - get old saved
        if "pca" not in self.Saved:
            print(f"- running {ver} for first time")
            self.pca(ver="eig")

        # 1) center and stack
        # X = X.copy()
        # - reshape to N x tiembins
        X = np.reshape(X, (nunits,-1)) # (nunits, else)
        # - demean
        X = X - self.Saved["centerAndStack_means"] # demean

        # 2) 
        w = self.Saved["pca"]["w"]
        u = self.Saved["pca"]["u"]
        if Ndim is None:
            usub = u[:,Dimslist]
        else:
            usub = u[:,:Ndim]
        Xsub = usub.T @ X

        # 3) reshape back to origuinal
        sh[0] = numdims
        Xsub = np.reshape(Xsub, sh)

        return Xsub


    ### ANALYSIS
    def dataframeByTrial(self, dim_trial = 1, columns_to_add = None):
        """ useful preprocess before do analyses.
        converts X to dataframe, where row is trial.
        - columns_to_add, dict, where each item is new column.
        entry must be same length as num trials.
        - dim_trial, which dim is trialnum?
        """
        assert False, "[DEPRECATED] - isntead, pass in list of dicts to PopAnal directly, this ensures keeps all fields in the dicts"
        ntrials = self.X.shape[dim_trial]


        assert dim_trial==1, "below code assumes 1. indexing is not as flexible"
        dat = []
        for i in range(ntrials):
            dat.append({
                "x":self.X[:, i, ...],
                "trial":i
                })
        df = pd.DataFrame(dat)

        if columns_to_add is not None:
            for k, v in columns_to_add.items():
                assert len(v)==ntrials
                df[k] = v

        return df

    # Data Transformations
    def zscoreFr(self, groupby=[]):
        """ z-score firing rates using across trial mean and std.
        - groupby, what mean and std to use. if [], then all trials
        combined (so all use same mean and std). if ["circ_binned"], 
        then asks for each trial what its bin (for circuiliatry) and then
        uses mean and std within that bin. Must have binned data first 
        before this (see binFeatures)
        ==== RETURN:
        modifies self.Xdataframe, adds column "neur_z"
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # 1. get mean and std.
        _, colname_std = self.aggregate(groupby, "trial", "std", "std", return_new_col_name=True)
        _, colname_mean = self.aggregate(groupby, "trial", "mean", "mean", return_new_col_name=True)

        # 2. take zscore
        def F(x):
            # returns (neur, time) fr in z-score units
            return (x["neur"] - x[colname_mean])/x[colname_std]

        self.Xdataframe = applyFunctionToAllRows(self.Xdataframe, F, newcolname="neur_z")

    def binFeatures(self, nbins, feature_list=None):
        """ assign bin to each trial, based on its value for feature
        in feature_list.
        - nbins, int, will bin by percentile (can modify to do uniform)
        - feature_list, if None, then uses self.Featurelist. otherwise give
        list of strings, but these must already be columnes in self.Xdataframe
        === RETURNS
        self.Xdataframe modifed to have new columns, e..g, <feat>_bin
        """
        if feature_list is None:
            feature_list = self.Featurelist

        for k in feature_list:
            kbin = self.binColumn(k, nbins)


    def binColumn(self, col_to_bin, nbins):
        """ assign bin value to each trial, by binning based on
        scalar variable.
        - modifies in place P.Xdataframe, appends new column.
        """
        from ..tools.pandastools import binColumn
        new_col_name = binColumn(self.Xdataframe, col_to_bin=col_to_bin, nbins=nbins)
        return new_col_name

    def aggregate(self, groupby, axis, agg_method="mean", new_col_suffix="agg", 
        force_redo=False, return_new_col_name = False, fr_use_this = "raw"):
        """ get aggregate pop activity in flexible ways.
        - groupby is how to group trials (rows of self.Xdataframe). 
        e.g., if groupby = [], then combines all trials into one aggreg.
        e.g., if groupby = "circ_binned", then groups by bin.
        e.g., if groupby = ["circ_binned", "angle_binned"], then conjunction
        - axis, how to take mean (after stacking all trials after grouping)
        e.g., if string, then is shortcut.
        e.g., if number then is axis.
        - new_col_suffix, for new col name.
        - agg_method, how to agg. could be string, or could be function.
        - fr_use_this, whether to use raw or z-scored (already done).
        RETURNS:
        - modifies self.Xdataframe, with one new column, means repopulated.
        - also the aggregated datagframe XdataframeAgg
        """
        from ..tools.pandastools import aggregThenReassignToNewColumn
        if isinstance(axis, str):
            # shortcuts
            if axis=="trial":
                # average over trials.
                axis = 0
            else:
                assert False, "not coded"

        if isinstance(agg_method, str):
            def F(x):
                if fr_use_this=="raw":
                    X = np.stack(x["neur"].values)
                elif fr_use_this=="z":
                    X = np.stack(x["neur_z"].values)
                # expect X to be (ntrials, nunits, time)
                # make axis accordinyl.
                if agg_method=="mean":
                    Xagg = np.mean(X, axis=axis)
                elif agg_method=="std":
                    Xagg = np.std(X, axis=axis)
                elif agg_method=="sem":
                    Xagg = np.std(X, axis=axis)/np.sqrt(X.shape[0])
                else:
                    assert False, "not coded"
                return Xagg
        else:
            F = agg_method
            assert callable(F)

        if len(groupby)==0:
            new_col_name = f"alltrials_{new_col_suffix}"
        elif isinstance(groupby, str):
            new_col_name = f"{groupby}_{new_col_suffix}"
        else:
            new_col_name = f"{'_'.join(groupby)}_{new_col_suffix}"

        # if already done, don't run
        if new_col_name in self.Xdataframe.columns:
            if not force_redo:
                print(new_col_name)
                assert False, "this colname already exists, force overwrite if actualyl want to run again."

        [self.Xdataframe, XdataframeAgg] = aggregThenReassignToNewColumn(self.Xdataframe, F, 
            groupby, new_col_name, return_grouped_df=True)

        if return_new_col_name:
            return XdataframeAgg, new_col_name
        else:
            return XdataframeAgg




    ### PLOTTING


    
""" Represents (N, D) data, where N is num datpts, and D is dimensionality,
and does things related to clustering, dimesionality reduction, etc.
Supposed to be generic across different kinds of analyses that work with high-d data.
Related to :
- neural.population
- tools.clustertools

First made when doing similarity matrix stuff for beh data (e.g, stroke similarity)
"""

import matplotlib.pyplot as plt
import numpy as np
from ..tools.nptools import sort_by_labels as sbl
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type
import seaborn as sns

class Clusters(object):
    """docstring for Clusters"""
    def __init__(self, X, labels_rows=None, labels_cols=None, ver=None, params=None):
        """ 
        PARAMS;
        - X, (N, D) data, where N is num datpts, and D is dimensionality, 
        - labels_rows, list of labels for each row of X. if None, then labels_rows as 
        [0,1,2, ..]. These are like "categories"
        - labels_cols, list of labels. if pass in None, then:
        --- if ncols == nrows, uses row labels
        --- otherwise uses (0, 1, 2, 3...)
        - ver, optional string, which is useful for defining applicable methods
        downstream. e..g, "rsa". This triggers checks of the input data.
        - params, dict of params, here just in case future proofings.
        """ 

        if params is None:
            params = {}

        self._Xinput = X # never changes
        self.Ndat = X.shape[0]
        self.Ndim = X.shape[1]

        if labels_rows is None:
            labels_rows = list(range(self.Ndat))
        else:
            assert len(labels_rows)==self.Ndat
        self.Labels = labels_rows

        if labels_cols is None:
            if self.Ndat==self.Ndim:
                labels_cols = labels_rows   
            else:
                labels_cols = list(range(self.Ndim))
        else:
            assert len(labels_cols)==self.Ndim
        self.LabelsCols = labels_cols

        self.Params = params

        # Track different versions of the data.
        # --- rows
        self.LabelsDict = {
            "raw":self.Labels
            } # tracks different ways of labeling, always aligned to self.Xinput

        self.XDict = {
            "raw":self.Xinput}

        self.ClusterResults = {}
        self.DistanceMatrices = {}
        self.ClusterComputeAllResults = {}

        if ver=="rsa":
            # This means each row is a tuple of levels across multiple grouping
            # vars. I.E., labels is list of tuples, each of len num vars.
            label_vars = params["label_vars"] # list of str
            # check that labels are correct format
            for lab in self.Labels:
                assert isinstance(lab, tuple)
                assert len(lab)==len(label_vars)
        elif ver=="dist":
            # Symmetrical distance matrix. Usually LabelsRows
            self.Xinput.shape[0] == self.Xinput.shape[1]
            assert self.LabelsCols == self.Labels, "some code might assume this..."
            for f in ["version_distance"]:
                assert f in self.Params.keys()
        else:
            assert ver is None
        self.Version = ver

    @property
    def Xinput(self):
        return self._Xinput
    

    ######################### sorting, filtering, slicing
    def sort_mat_helper(self, X, sortvalues, axis):
        """ Helper to sort X by values,matched to items along either
        row or col.
        PARAMS:
        - X, array, (n, m)
        - sortvalues, 1d arra, iether len n or m deepenwding on axis.
        - axis, 0, 1, which to sort
        RETURNS:
        - Xsorted, (n,m)
        - sortvalues_sorted,
        --- both are copies, then sort
        """
        # from ..tools.nptools import sort_by_labels as sbl

        return sbl(X, sortvalues, axis)

    def sort_by_labels(self, X=None, labels=None, axis=0):
        """ 
        Sorts labels, then sorts X using the sortinds for labels
        PARAMS:
        - axis, dimeision to srot byt. if 0, then sorts rows...
        RETURNS:
        - Xsorted, labelsSorted
        """
        # from ..tools.clustertools import sort_by_labels as sbl

        if labels is None:
            labels = self.extract_labels(axis=axis)
        
        if X is None:
            X = self.Xinput
        
        Xsorted, labelsSorted = sbl(X, labels, axis=axis)
        
        return Xsorted, labelsSorted

    def sort_by_labels_both_axes(self, X=None, labels_row=None, labels_col=None):
        """ sorts both axis 0 and 1.
        """
        Xsorted, labels_row = self.sort_by_labels(X, labels_row, axis=0) # rows
        Xsorted, labels_col = self.sort_by_labels(Xsorted, labels_col, axis=1) # cols
        return Xsorted, labels_row, labels_col


    # def sort_by_labels(labels=None):
    #     """ 
    #     Sort X by labels, in incresaing order.
    #     PARAMS:
    #     - labels, list of labels, length ndat. if None, then
    #     uses self.labels
    #     RETURNS:
    #     - X, labels, but sorted (copies)
    #     - overwrites self.Xsorted, self.LabelsSorted
    #     """
    #     if labels is None:
    #         labels = self.Labels
    #     self.Xsorted, self.LabelsSorted = self._sort_by_labels(self.X, labels)
    #     return self.Xsorted, self.LabelsSorted

    ######################## EXTRACT THINGS
    def index_find_dat_by_label(self, label_row, label_col):
        """ find the single cell with these labels.
        REturn the value.
        """

        ind1 = self.index_find_ind_by_label(label_row, axis=0)
        ind2 = self.index_find_ind_by_label(label_col, axis=1)
        X = self.extract_dat()
        return X[ind1, ind2]

    def index_find_ind_by_label(self, label, axis=0, assert_only_one=True):
        """
        Find the single index in row(axis=0) or column(axis=1)
        with this value of label.
        """
        labs = self.extract_labels(axis=axis)
        n = sum([l==label for l in labs])
        if assert_only_one and n!=1:
            print(labs)
            print(label)
            print(n)
            assert False, "why multiple identical lables?"
        return labs.index(label)

    def extract_dat(self, datkind="raw"):
        """ Rerutrn data matrix, but possibly processed
        """
        return self.XDict[datkind]

    def extract_labels(self, labelkind="raw", axis=0):
        """ Return labels, various kinds
        """

        if axis==0:
            labs = self.LabelsDict[labelkind]
        elif axis==1:
            labs = self.LabelsCols
        else:
            assert False
        assert isinstance(labs, list)
        return labs

    ################### data extract
    def dataextract_upper_triangular_flattened(self,
                                               exclude_diag=True,
                                               inds_rows_cols=None,
                                               dat = None):
        """ Return elemetns in upper diagnoal, not including
        diagnoal, flatteened into 1-array
        PARAMS:
        - inds_rows_cols, either None (Ignore) or array-like of indices into
        self.X[inds, inds], to slice a subset (will still be square).
        - exclude_diag, bool (True), then ignore the diangoanl
        """

        if dat is None:
            dat = self.Xinput

        assert dat.shape[1]==dat.shape[0]

        if inds_rows_cols is not None:
            dat = dat[inds_rows_cols, :][:, inds_rows_cols] # (len inds, len inds)

        n = dat.shape[0]
        if exclude_diag:
            inds = np.triu_indices(n, 1)
        else:
            inds = np.triu_indices(n, 0)
        vec = dat[inds].flatten()
        return vec

    ######################### PLOTS
#     def plot_heatmap_data(self, datkind="raw", labelkind="raw", sortver_bylabel=None, 
#             nrand = None, SIZE=12, zlims=(None, None)):
#         """ Plot heatmap of raw data, usually sorted (rows) by labelkind.
#         PARAMS:
#         - datkind, string name of data to plot. e..g, pca
#         - labelkind, string name to get label
#         - sortver_bylabel, either
#         --- None, no osrting
#         --- {0,1}, sorts on thsoie axis
#         --- 2, sorts both axes
#         - nrand, int for taking random subset. None to skip
#         RETURNS:
#         - fig, X, labels
#         """
        
#         # Extract data
#         X = self.extract_dat(datkind)
#         labels_row = self.extract_labels(labelkind, axis=0)
#         labels_col = self.extract_labels(labelkind, axis=1)

#         # Take subset?
#         if nrand is not None:
#             import random
#             inds = random.sample(X.shape[0], nrand)
#             X = X[inds, :]
#             labels_row = [labels_row[i] for i in inds]
#             labels_col = [labels_col[i] for i in inds]
        

#         if sortver_bylabel in 0:
#             # sort by row labels
#             X, labels_row = self.sort_by_labels(axis=sortver_bylabel)
#         elif sortver_bylabel==1:
#             sdafdsf
#             X, labels_col = self.sort_by_labels(axis=sortver_bylabel)
#         elif sortver_bylabel == 2:
#             # sort by both axes
#             X, labels_row, labels_col = self.sort_by_labels_both_axes(X, labels_row, labels_col)
#         else:
#             assert sortver_bylabel is None

#         # --- before sorting
#         ndim = X.shape[0]
#         ndat = X.shape[1]
#         ASPECT = ndim/ndat
#         fig, ax = plt.subplots(figsize=(SIZE, ASPECT*SIZE))
        
#         h = ax.imshow(X, vmin=zlims[0], vmax=zlims[1], cmap='viridis')
#         fig.colorbar(h, ax=ax)
        
#         print(labels_row)
        
#         ax.set_yticks(range(len(labels_row)), labels_row);
#         # ax.set_yticklabels(ylabels);

#         ax.set_xticks(range(len(labels_col)), labels_col, rotation=45);

# # ?        plt.xticks(rotation=45, ha='right')

#         return fig, X, labels_col, labels_row

    def _plot_heatmap_data(self, X, labels_row=None, labels_col=None,
        SIZE=12, zlims=(None, None), nrand=None, rotation=90, rotation_y=0,
                           diverge=False, annotate_heatmap=False,
                           ax=None, robust=False):
        """ Low-=level plotting of heatmap of X.
        """
        from pythonlib.tools.snstools import heatmap_mat

        if nrand is not None:
            import random
            inds = random.sample(range(X.shape[0]), nrand)
            X = X[inds, :]
            if labels_row:
                labels_row = [labels_row[i] for i in inds]

        fig, ax, rgba_values = heatmap_mat(X, ax, annotate_heatmap, zlims,
                   robust, diverge, labels_row, labels_col, rotation, rotation_y)

        return fig, X, labels_col, labels_row, ax

    def plot_heatmap_data(self, datkind="raw", labelkind="raw", 
        nrand = None, SIZE=12, zlims=(None, None)):
        """ Plot heatmap of raw data in self.Xinput, optionally taking
        subset of rows.
        PARAMS:
        - datkind, string name of data to plot. e..g, pca
        - labelkind, string name to get label
        - sortver_bylabel, either
        --- None, no osrting
        --- {0,1}, sorts on thsoie axis
        --- 2, sorts both axes
        - nrand, int for taking random subset of rows. None to skip
        RETURNS:
        - fig, X, labels
        """
        
        # Extract data
        X = self.extract_dat(datkind)
        labels_row = self.extract_labels(labelkind, axis=0)
        labels_col = self.extract_labels(labelkind, axis=1)

        # Take subset of rows?
        if nrand is not None:
            import random
            inds = random.sample(range(X.shape[0]), nrand)
            X = X[inds, :]
            labels_row = [labels_row[i] for i in inds]
            # labels_col = [labels_col[i] for i in inds]
        
        return self._plot_heatmap_data(X, labels_row, labels_col, SIZE, zlims)

    def plot_save_hier_clust(self, col_cluster=True):
        """ Perform hier clustering (agglomerative) and plot and 
        save results into self.ClusterResults["hier_clust_seaborn"]
        PARAMS:
        - col_cluster, not necessary, but useful for visualization. 
        """    
        import seaborn as sns

        if len(self.Xinput)==0:
            assert False

        try:
            if "hier_clust_seaborn" not in self.ClusterResults.keys():
                cg = sns.clustermap(self.Xinput, row_cluster=True, col_cluster=True)
                self.ClusterResults["hier_clust_seaborn"] = cg

            # Plot
            cg = self.ClusterResults["hier_clust_seaborn"]

            # Label axes both col and row.
            self._hier_clust_label_axis(cg, "col")
            self._hier_clust_label_axis(cg, "row")
        except Exception as err:
            cg = None

        return cg


    def _hier_clust_label_axis(self, cg, which_axis):
        """ Label axes with strings (i.e., Colnames) instead of the
        indices which is default
        """
        # cg = self.plot_save_hier_clust()
        ax = cg.ax_heatmap
        if which_axis=="col":
            inds = cg.dendrogram_col.reordered_ind
            list_labels = self.LabelsCols
            list_labels_this = [list_labels[i] for i in inds]
            locs = range(len(list_labels_this))
            locs = [l+0.5 for l in locs]
            ax.set_xticks(locs, labels=list_labels_this, rotation=80);
        elif which_axis=="row":
            inds = cg.dendrogram_row.reordered_ind
            list_labels = self.Labels
            list_labels_this = [list_labels[i] for i in inds]
            locs = range(len(list_labels_this))
            locs = [l+0.5 for l in locs]
            ax.set_yticks(locs, labels=list_labels_this, rotation=0);
        else:
            assert False


    ################### DO CLUSTERING
    def cluster_compute_feature_scores_assignment(self):
        """ for each trial, assign to a feature based simply on max
        of the similarity score
        """

        # self.Xinput is matric similarities.
        sims_max = self.Xinput.max(axis=1)
        sims_min = self.Xinput.min(axis=1)
        sims_median = np.median(self.Xinput, axis=1)
        # sims_mean = self.Xinput.mean(axis=1)
        sims_concentration = (sims_max - sims_min)/(sims_max + sims_min)
        sims_concentration_v2 = (sims_max - sims_median)/(sims_max + sims_median)
        
        # which shape does it match the best
        inds_maxsim = np.argmax(self.Xinput, axis=1)
        cols_maxsim = [self.LabelsCols[i] for i in inds_maxsim]

        # entropy
        from scipy.stats import entropy
        simmat_entropy= entropy(self.Xinput, axis=1)

        # Save it
        self.ClusterComputeAllResults["max_sim"] = {
            "colinds_maxsim":inds_maxsim,
            "collabels_maxsim":cols_maxsim,
            "sims_max":sims_max,
            "sims_concentration":sims_concentration,
            "sims_concentration_v2":sims_concentration_v2,
            "sims_entropy":simmat_entropy,
        }

    def cluster_compute_all(self, PCAdim_touse=5, gmm_n_mixtures=None,
        perplist = None,
        things_to_do = ("tsne", "gmm", "gmm_using_tsne"),
        gmm_tsne_perp_to_use=15):
        """ Helper to generate models of data of various types.
        E.g., Do PCA, tSNE, and GMM
        RETURNS:
        - stores models in ClusterComputeAllResults
        """
        from pythonlib.tools.clustertools import clusterSimMatrix
        simmat = self.Xinput
        if gmm_n_mixtures is None:
            gmm_n_mixtures = range(5, 20)
        if perplist is None:
            perplist = (8, 15, 25, 35, 45)

        OUT = clusterSimMatrix(simmat, None, PCAdim_touse,
            gmm_n_mixtures =gmm_n_mixtures,
            perplist =perplist,
            gmm_tsne_perp_to_use = gmm_tsne_perp_to_use,
            things_to_do = things_to_do)

        for k, v in OUT.items():
            self.ClusterComputeAllResults[k] = v

        # Also do the assignments
        self.cluster_compute_feature_scores_assignment()

    def cluster_pca_plot_all(self, savedir=None):
        """ Plot results of PCA, including sunbset of data
        projected, and variance explained
        """
        pcamodel = self.ClusterComputeAllResults["pca_model"]
        fig, axes = plt.subplots(2,2)
        ax = axes.flatten()[0]
        ax.plot(pcamodel.explained_variance_, "-o")
        if savedir:
            savefig(fig, f"{savedir}/pca_explained_variance.pdf")

        fig, X, labels_col, labels_row = self._plot_heatmap_data(pcamodel.components_, range(len(pcamodel.components_)), 
            self.LabelsCols, SIZE=5, rotation=90)
        if savedir:
            savefig(fig, f"{savedir}/pca_heatmap_loadings.pdf")


    def cluster_plot_scatter(self, space="pca", label="shape", perp=15,
        gmm_n = None, ax = None, dims=None):
        """ Wrapper to plot scatter in any space and labeled any way
        """

        from pythonlib.tools.plottools import plotScatterOverlay
        data = self.cluster_extract_data(space, perp=perp)
        if label is not None:
            labels = self.cluster_extract_label(label, gmm_n=gmm_n)
        else:
            labels = None

        if dims:
            data = data[:, [dims[0], dims[1]]]

        fig, axes = plotScatterOverlay(data, labels, ax=ax)   
        return fig, axes

    def cluster_extract_label(self, kind, gmm_n=None, nrand=None):
        """ 
        Helper to extract labels for each trial given different spaces.
        RETURNS:
        - 
        """
        if kind=="shape":
            return self.Labels
        elif kind=="col_max_sim":
            # The label of column that is max sim for each row (e.g, the basis shape).
            return self.ClusterComputeAllResults["max_sim"]["collabels_maxsim"]
        elif kind=="gmm":
            mod = self.cluster_extract_model("gmm", gmm_n=gmm_n)
            Xpca_input_models = self.cluster_extract_data("pca_inputted_into_models")
            labels = mod["mod"].predict(Xpca_input_models)
            return labels
        elif kind=="gmm_using_tsne":
            mod = self.cluster_extract_model("gmm_using_tsne", gmm_n=gmm_n)
            Xpca_input_models = self.cluster_extract_data("tsne_inputted_into_gmm")
            labels = mod["mod"].predict(Xpca_input_models)
            return labels
        elif kind=="random":
            # Generate random labels (ints, from 0,....)
            assert nrand is not None # range of numbers
            asds
            labels = [random.sample(range(nrand), 1)[0] for _ in range(len(SF))]
        else:
            assert False, "code it"

    def cluster_extract_data(self, kind, perp=None):
        """
        Extract raw data projected in different spaces. Requires that those
        models are already gnerated.
        """

        if kind=="pca":
            return self.ClusterComputeAllResults["Xpca"]
        elif kind=="pca_inputted_into_models":
            # pca rthat was inouted into Tsne and gmm
            return self.ClusterComputeAllResults["Xpca_input_models"]
        elif kind=="tsne_inputted_into_gmm":
            # pca rthat was inouted into Tsne and gmm
            return self.ClusterComputeAllResults["Xtsne_input_gmm"]
        elif kind=="tsne":
            assert perp is not None
            assert self.ClusterComputeAllResults["models_tsne"] is not None, "need to first extracgt tnse"
            modthis = [mod for mod in self.ClusterComputeAllResults["models_tsne"] if mod["perp"]==perp]
            assert len(modthis)==1, "didnt find this perp..."
            return modthis[0]["D_fit"]
        elif kind=="max_sim":
            return self.ClusterComputeAllResults["max_sim"]
        else:
            assert False
  
    # def cluster_pca_plot_scatter(self, ax):
    #     from pythonlib.tools.plottools import plotScatterOverlay

    #     for dim 
    #     X = self.cluster_results_extract_specific("pca")
    #     plotScatterOverlay(X, labels=self.Labels, ax=ax)   

    # def cluster_tsne_plot_scatter_thisperp(self, perp, ax=None):
    #     """ 
    #     """
    #     from pythonlib.tools.plottools import plotScatterOverlay
    #     mod = self.cluster_results_extract_specific("tsne", perp=perp)
    #     # models_tsne = self.ClusterComputeAllResults["models_tsne"]
    #     X = mod["D_fit"]
    #     plotScatterOverlay(X, labels=self.Labels, ax=ax)   

    # def cluster_tsne_plot_scatter_allperp(self):
    #     """
    #     """
    #     # TSNE PLOTS
    #     models_tsne = self.ClusterComputeAllResults["models_tsne"]
    #     for mod in models_tsne:
    #         perp = mod["perp"]
    #         self.cluster_tsne_plot_scatter_thisperp(perp)

    def cluster_extract_model(self, which_mod, perp=None, gmm_n=None):
        """
        Helpert to extract specific model
        RETURNS:
        - dict, usually with {"mod":model}, and other params as keys too.
        """
        if which_mod=="pca":
            return {"mod":self.ClusterComputeAllResults["pca_model"]}
        elif which_mod=="tsne":
            assert perp
            modthis = [mod for mod in self.ClusterComputeAllResults["models_tsne"] if mod["perp"]==perp]
            assert len(modthis)==1
            return modthis[0]
        elif which_mod=="gmm":
            assert gmm_n, "need to pick a model out"
            # self.ClusterComputeAllResults["models_gmm"]
            for mod in self.ClusterComputeAllResults["models_gmm"]:
                if mod["n"]==gmm_n:
                    return mod
            for mod in self.ClusterComputeAllResults["models_gmm"]:
                print(mod)
            assert False, "model with this nmix doesnt exist"
        elif which_mod=="gmm_using_tsne":
            assert gmm_n, "need to pick a model out"
            # self.ClusterComputeAllResults["models_gmm"]
            for mod in self.ClusterComputeAllResults["models_gmm_using_tsne"]:
                if mod["n"]==gmm_n:
                    return mod
            for mod in self.ClusterComputeAllResults["models_gmm_using_tsne"]:
                print(mod)
            assert False, "model with this nmix doesnt exist"
        else:
            assert False, "code it"

    def cluster_results_extract_all(self, which_mod):
        """ Helper to extract all model results across all params
        RETURNS:
        - list of dicts, each a model instance (with params included as keys)
        """
        if which_mod=="tsne":
            return self.ClusterComputeAllResults["models_tsne"]
        elif which_mod=="gmm":
            return self.ClusterComputeAllResults["models_gmm"]
        elif which_mod=="gmm_using_tsne":
            return self.ClusterComputeAllResults["models_gmm_using_tsne"]
        else:
            assert False, "code it"

    def cluster_gmm_extract_best_n(self, ver="gmm_using_tsne"):
        """ Get the N that has the best cross-validated score
        PARAMS;
        - var, either "gmm_using_tsne" or "gmm"[default]
        """

        list_mod = self.cluster_results_extract_all(ver)
        list_n = []
        list_crossval = []
        list_bic = []
        for mod in list_mod:
            list_n.append(mod["n"])
            list_crossval.append(mod["cross_val_score"])    
            list_bic.append(mod["bic"])

        # list_n
        gmm_n_best = list_n[np.argmax(list_crossval)]

        return gmm_n_best, list_n, list_crossval, list_bic

    def cluster_tsne_extract_list_perp(self):
        """
        RETurn list of nums, the perps that are presnet across models of tsne that 
        were saved
        """
        list_mod = self.cluster_results_extract_all("tsne")
        list_perp = [mod["perp"] for mod in list_mod]
        return list_perp    


    # Plot example trials, in grid organized by TSne
    def plot_grid_egtrials_organizedby(by, params, nbins=20):
        """ Plot strokes in a 2d grid, where x and y coord of each subplot
        correspond to binne coordinates in some low d representation
        PARAMS:
        - by, string name, what coordinate system
        - params, list, params that are for this "by"
        - nbins, int, how many bins per axis. 
        """
        assert False ,"in progress - is the grid, with strokes plotted at their cooridnates."
        from pythonlib.drawmodel.strokePlots import plotDatStrokes

        # Which coordinate system for deciding 2d grid organization
        Xfit = extract_dat(by, params)

        # === plot a grid, based on percentiles along 2 dimensions
        # 1) assign all indices to a position in grid, based on percentiles
        values1 = Xfit[:,0]
        values2 = Xfit[:,1]
        idxslist = range(Xfit.shape[0])
        p = np.linspace(0, 100, nbins)

        binedges = np.percentile(values1, p)
        inds1 = np.digitize(values1, binedges)

        binedges = np.percentile(values2, p)
        inds2 = np.digitize(values2, binedges)

        # 2) for each combo of inds, plot an example trial
        indslist = set(np.r_[inds1, inds2])
        fig, axes = plt.subplots(len(indslist), len(indslist), sharex=True, sharey=True, figsize=(len(indslist)*2, len(indslist)*2))
        for i1 in indslist:
            for ii2, i2 in enumerate(np.array([i for i in indslist])[::-1]): # so can go backwards.. with bottom left as 1,1
                print(i1, i2)
                ax = axes[ii2-1][i1-1]
                indsthis = list(np.where((inds1==i1) & (inds2==i2))[0])
                if len(indsthis)==0:
                    continue

                ind = random.sample(indsthis,1)[0]

                strokthis = SF["strok"][ind]
                plotDatStrokes([strokthis], ax, pcol="r")
                ax.axhline(0, color='k', alpha=0.3)
                ax.axvline(0, color='k', alpha=0.3)
                ax.set_title(f"{i1}-{i2}")
                M = 300
                ax.set_xlim([-M, M])
                ax.set_ylim([-M, M])    
            assert False

    #     fig.savefig(f"{SDIRFIGS}/tsne-behgrid-ll.pdf")
    # #################### PLOTS
    # self.plot_basis_strokes

    ################ DISTANCE MATRICES
    def distsimmat_convert(self, version_distance="pearson"):
        """ Convert data (ndat, ndim) to distnace matreix (ndat, ndat)
        - Distance --> more positive is always more distance, by convention. This
        means pearson corr will range between 0 and 1, for example. By convention 0 is close.
        RETURNS:
            - Cl, ClustClass object.
        """

        X = self.Xinput
        ndat = X.shape[0]
        if version_distance=="euclidian":
            # 0=idnetical, more positive more distance.
            from scipy.spatial import distance_matrix
            D = distance_matrix(X, X)
        elif version_distance=="pearson":
            # correlation matrix, but scaled between 0 (close) and 1 (far) as follows:
            # -1 --> 1
            # +1 --> 0
            D = 1-(np.corrcoef(X) + 1)/2
        else:
            assert False
        assert ndat==D.shape[0]
        params = {
            "version_distance":version_distance,
            "Clraw":self,
        }
        return Clusters(D, self.Labels, self.Labels, ver="dist", params=params)

    ################# RSA
    def _rsa_check_compatible(self):
        """ returns True if self is either rsa, or distance matrix from rsa
        """

        if self.Version=="rsa":
            return True
        # if "Clraw" not in self.Params.keys():
        #     print(self.Version)
        #     print(self.Params)
        #     assert False
        if self.Version=="dist" and isinstance(self.Labels[0], tuple):
            return True
        return False

    def rsa_plot_heatmap(self, sort_order=None, diverge=False):
        """ Plot the input data, which can be raw data or sim mat, anything that
        is 2d heatmapt, and has each row being a conjunction of levels of
        grouping vars, plots in useful way for inspecting relationships between levels,
        i.e., with options for sorting labels
        to allow visualziation of interesting patterns
        PARAMS:
        - sort_order, tuple of ints, defining order that labels will be
        sorted. e.g., (1,0,2) means first sort by var 1, then break ties
        using var 0, then 2.. Can pass (1,0) to ignore 2, for example.
        If raw dataA: Applies to rows
        If dist mat: Appkuies to row and columns.
        """
        from pythonlib.tools.listtools import argsort_list_of_tuples

        assert self._rsa_check_compatible()

        # Pull out data in correct format, and return as clusters.
        X = self.Xinput

        # Sort labels if needed
        labels_rows = self.Labels
        labels_cols = self.LabelsCols

        if sort_order is not None:
            key =lambda x:tuple([x[i] for i in sort_order])
            inds_sort = argsort_list_of_tuples(labels_rows, key)
            labels_rows = [labels_rows[i] for i in inds_sort]
            X = X[inds_sort, :]

            # try sorting the x labels too, if they are tuples
            if isinstance(self.LabelsCols[0], tuple):
                inds_sort = argsort_list_of_tuples(labels_cols, key)
                labels_cols = [labels_cols[i] for i in inds_sort]
                X = X[:, inds_sort]

        # Plot
        fig, X, labels_col, labels_row, ax = self._plot_heatmap_data(X, labels_rows,
                                                                   labels_cols, diverge=diverge)
        return fig, ax


    def _rsa_distmat_return_distfunc(self, var):
        """
        Helper to return a funtion that computes distnace between
        two items to help populate distnace matrix.
        Distance --> larger number means further apart
        :param var: string
        :return:
        - dist_func: (x,y) --> scalar.
        """
        # Construct dist funcs for each variable
        # Categorical
        def dist_func_cat(x,y):
            # Categorical
            if x==y:
                # same
                return 0.
            else:
                return 1.

        def dist_angle_hack(x,y, DEBUG=False):
            """ hacky, for binned angles between 1 and 4, returns
            the positive distnace between them, allwoing for rotation, so that
            1 vs 4 gives 1.
            - Range 0(same) to 2 (oppsite angles)
            """

            if np.isnan(x) or np.isnan(y):
                # can happen if there is no next stroke (e.g,, next reach..)
                return np.nan

            # print(x)
            # print(type(x))
            # assert isinstance(x, int)
            assert x>0 and x<5
            assert y>0 and y<5

            if DEBUG:
                for j in range(1,5):
                    for i in range(1,5):
                        print(i, j, ' -- ' , dist_angle_hack(i, j))
            y2 = y-4
            x2 = x-4

            d = np.min([np.abs(x-y), np.abs(x-y2), np.abs(y-x2)])

            return d

        # ordinal
        def dist_func_ord(x,y):
            # 0==same
            # 1,2,3. ... further apart.
            return np.abs(x-y)

        ########## FIRST, USE HAND-ENTERED
        # Construct all string names
        vars_angle_binned = []
        for a in ["gap_to_next", "gap_from_prev"]:
            for b in ["_angle"]:
                for c in ["_binned"]:
                    vars_angle_binned.append(f"{a}{b}{c}")

        tmp = []
        for a in ["gap_to_next", "gap_from_prev"]:
            for b in ["_dist"]:
                for c in ["", "_binned"]:
                    tmp.append(f"{a}{b}{c}")
        vars_categorical = ["shape_oriented", "gridloc", "stroke_index_semantic",
                   "CTXT_loc_next", "CTXT_shape_next"] + tmp

        if var in vars_categorical:
            # Categorical
            return dist_func_cat
        elif var in vars_angle_binned:
            # Hacky, angles binned
            return dist_angle_hack
        elif var in ["stroke_index", "stroke_index_fromlast", "stroke_index_fromlast_tskstks",
                     "gridloc_x", "FEAT_num_strokes_task", "FEAT_num_strokes_beh", "FEAT_num_strokes_task"]:
            # ordinal
            return dist_func_ord
        else:
            pass # Try to get it auto below.

        ########## SECOND, TRY AUTOMATICALLY, based on type.
        vals = self.rsa_index_values_this_var(var)
        list_types = list(set([type(v) for v in vals]))
        at_least_one_type_is_tuple = any([isinstance(t, tuple) for t in list_types])

        if len(list_types)>1 and at_least_one_type_is_tuple:
            # Mxing these types, usually is categorical
            return dist_func_cat
        elif len(list_types)>1:
            # Then not sure.
            print(var)
            print(vals)
            assert False, "dont know what distance functin to use"
        else:
            # Only one type
            t = list_types[0]
            if t in (str, bool):
                return dist_func_cat
            elif t in (int):
                # Not sure, just use categorical
                return dist_func_cat
            else:
                # Then not sure.
                print(var)
                print(vals)
                print(list_types)
                print(t)
                print(t==str)
                assert False, "dont know what distance functin to use"

    def _rsa_map_varstr_to_varind(self, var):
        """ Get the index into labels, which matches this var
        e.g., self.Labels[0][ind] is the value of this var for row 0.
        """

        self._rsa_check_compatible()
        label_vars = self.rsa_labels_extract_label_vars()
        assert var in label_vars
        return label_vars.index(var)

    def rsa_distmat_construct_theoretical(self, var, PLOT = False):
        """Construct theoretical dsitances matrices based on the varaibles
        for each row. Does not use the data in self.Xinput, just the labels
        cols and rows.
        RETURNS:
            - Cltheor.Labels will be identical to self.Labels. This way can sort
            labels as you would if this where the data matrix, to allow comparison
            with data matrix.
        """
        from pythonlib.tools.distfunctools import distmat_construct_wrapper

        # 1) pick out the feature dimension, and update labels
        # Recompute distance
        ind_var = self._rsa_map_varstr_to_varind(var) # convert from var to ind_var
        labels_row = [lab[ind_var] for lab in self.Labels]
        labels_col = [lab[ind_var] for lab in self.LabelsCols]
        _dist_func = self._rsa_distmat_return_distfunc(var)
        D = distmat_construct_wrapper(labels_row, labels_col, _dist_func)

        # Make the labels into tuples, so that they match other rsa stuff
        # labels_row = [tuple([lab]) for lab in labels_row]
        # labels_col = [tuple([lab]) for lab in labels_col]
        # No: instead keep the labels as in self, to allow plotting self and
        # Cltheor using similar sort indices.
        Cltheor = Clusters(D, self.Labels, self.LabelsCols, ver="dist",
                           params={"var":var, "version_distance":None})
        # plot
        if PLOT:
            fig = Cltheor.plot_heatmap_data()[0]
        else:
            fig = None

        return Cltheor, fig

    def rsa_labels_extract_var_levels(self):
        """ Extract the levels that exist across all vars.
        Reutrns dict, var:<list of levels>
        """
        dflab = self.rsa_labels_return_as_df()
        map_var_levels = {}
        for var in self.rsa_labels_extract_label_vars():
            map_var_levels[var] = sort_mixed_type(dflab[var].unique().tolist())
        return map_var_levels

    def rsa_labels_extract_label_vars(self):
        """ Return tuple of label vars (strings) in order they
        are used in self.Labels (i.e., self.Labels[0] is a tuple,
        ordered this way).
        """
        if self.Version == "rsa":
            label_vars = self.Params["label_vars"]
        elif self.Version == "dist":
            label_vars = self.Params["Clraw"].Params["label_vars"]
        else:
            assert False
        return label_vars

    def _rsa_matindex_plot_bool_mask(self, ma, ax):
        """ Plot this boolean mask
        """
        ax.imshow(ma)

    def _rsa_matindex_convert_to_mask_specific(self, inds1, inds2):
        """ Given indices defined by paired values in
        lists rows and cols, return boolean array with
        these indices True, and the rest False
        """
        ma = np.zeros_like(self.Xinput, dtype=bool)
        ma[inds1, inds2] = True
        return ma

    def _rsa_matindex_generate_upper_triangular(self):
        """ boolean mask with True in upper triangular, ecluding
        diagona, and elsewhere False.
        """
        return np.triu(np.ones_like(self.Xinput, dtype=bool), k=1)

    def _rsa_matindex_convert_to_mask_rect(self, rows, cols):
        """ Given row and column indices to slice (into rectangle), return
        boolean array with these indices True, and the rest False
        """
        # convert rows and cols to speicif inidices, since if you just
        # do sequential slices, it takes copy, and you wont change ma
        inds1 = []
        inds2 = []
        for r in rows:
            for c in cols:
                inds1.append(r)
                inds2.append(c)
        return self._rsa_matindex_convert_to_mask_specific(inds1, inds2)

    def _rsa_matindex_slice_specific_view(self, rows, cols):
        """ Helper to pull out values from self.Xinput which
        are indexed by conucntions of rows and cols. i.e,,
        (rows[0], cols[0]), (rows[1], cols[1]) etc.
        RETURNS:
            - np array, (len(rows),) shape. This is a view, which
            means modifications will affect oriingal array
        """
        assert len(rows)==len(cols)
        # return self.Xinput[rows, cols] # equivalent
        return self.Xinput[(rows, cols)]

    def _rsa_matindex_slice_rect_copy(self, rows, cols):
        """ Helper to pull out rectangle slice of self.Xinput,
            - np array, (len(rows), len(cols)) shape.  This is a copy, which
            means modifications will NOT affect oriingal array
        """
        assert False, "avoid using this, it is a copy..."
        return self.Xinput[rows,:][:,cols]

    # def rsa_index_indvar_for_this_var(self, varstr):
    #     """ map var string to its index in self.Labels[0]"""
    #     return self._rsa_map_varstr_to_varind(varstr)

    def rsa_index_values_this_var(self, var, inds_row=None):
        """
        Rturn list of values for var at these indices (rows),
        as a list of items
        :param var:
        :param inds_row: list of ints, if None, gets all rows.
        :return:
        - list of values, same len as inds_row.
        """
        if inds_row is None:
            inds_row = list(range(len(self.Labels)))
        levs = self.rsa_labels_return_as_df().iloc[inds_row][var].tolist()
        return levs

    def rsa_index_cols_with_this_level(self, var, level):
        """ Return columsn with this level of this var
        """
        assert self.LabelsCols==self.Labels
        inds = self.rsa_index_rows_with_this_level(var, level)
        return inds

    def rsa_index_rows_with_this_level(self, var, level):
        """ return list of ints in to self.Labels (rows) whcih
        have this level for this var
        e.g, self.Labels[inds[0]]==level will be True
        """
        dflab = self.rsa_labels_return_as_df()
        return dflab[dflab[var]==level]["row_index"].tolist()

    def rsa_matindex_same_diff_this_level(self, var, level):
        """ Return indices in distnace matrix corresponding
        to cases that are same and cases different level, compared to
        input level
        RETURNS:
            - ma_same, ma_diff, boolean matrices (nrows, ncols).
        """

        # same
        rows = self.rsa_index_rows_with_this_level(var, level)
        cols = self.rsa_index_cols_with_this_level(var, level)

        # diff
        ncols = len(self.LabelsCols)
        cols_diff = [i for i in range(ncols) if i not in cols]
        # print(rows)
        # print(cols)
        # print(cols_diff)

        # Return bool masks
        ma_same = self._rsa_matindex_convert_to_mask_rect(rows, cols)
        ma_diff = self._rsa_matindex_convert_to_mask_rect(rows, cols_diff)

        return ma_same, ma_diff

    def rsa_labels_return_as_df(self, include_row_index=True):
        """ Return df where rows are labels (rows in X) and
        columns are names of label vars, in order they are
        used in labels
        """
        import pandas as pd
        label_vars = self.rsa_labels_extract_label_vars()
        dflab =  pd.DataFrame(self.Labels, columns=label_vars)
        if include_row_index:
            dflab["row_index"] = list(range(len(dflab)))
        return dflab

    def rsa_distmat_quantify_same_diff_variables(self, ind_var, ignore_diagonal=True):
        """
        For this variable, get distance across same and diff parirs, across
        all lewvels.
        E.g., can compute distance between same shape (across other var), vs. diff shape (across all var).
        PARAMS:
        - ind_var = 0 # e..g, if each row is labeled with a tuple like (shape, loc), then if
        ind_var==0, then this means "same" is defined as having same shape
        """

        assert self._rsa_check_compatible()
        assert self.Version=="dist"

        # Collect mapping
        map_pair_labels_to_indices = {} # (lab1, lab2) --> (col, row)
        for i, lr in enumerate(self.Labels):
            for j, lc in enumerate(self.LabelsCols):
                # only take off diagonal
                if ignore_diagonal:
                    if i>=j:
                        continue
                else:
                    if i>j:
                        continue
                map_pair_labels_to_indices[(lr, lc)] = (i, j)

        # Find the coordinates of "same" and "diff" pairs.
        # given a var dimension, get all indices that are "same" along that dimension
        list_inds_same = []
        list_inds_diff = []
        for lab_pair, inds in map_pair_labels_to_indices.items():
            a = lab_pair[0][ind_var]
            b = lab_pair[1][ind_var]
            if a==b:
                # then is "same"
                list_inds_same.append(inds)
            else:
                list_inds_diff.append(inds)

        # Collect data
        list_i = [x[0] for x in list_inds_same]
        list_j = [x[1] for x in list_inds_same]
        vals_same = self.Xinput[(list_i, list_j)]

        list_i = [x[0] for x in list_inds_diff]
        list_j = [x[1] for x in list_inds_diff]
        vals_diff = self.Xinput[(list_i, list_j)]

        return vals_same, vals_diff
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
import pandas as pd

def generate_clustclass_from_flat_df(dfres, var_row, var_col, var_value, var_labels, PRINT=False):
    """
    Given dataframe whhere each row is a pair of conditions (i.e., a single cell in Cl), then
    generate a new Cl. Must be square, with each condition appearing once in rows and coluns
    PARAMS:
    - var_labels, list/tuple of strings, mathching length of each item in var_row and var_col.
    """
    ### Plot heatmap of all r2 scores

    # Convert to clustclass
    list_labels = sorted(dfres[var_row].unique())
    assert list_labels == sorted(dfres[var_col].unique()), "need each condition appearing once in rows and coluns"    
    assert isinstance(list_labels[0], tuple)
    # assert isinstance(list_labels[0][0], str)

    distmat = np.zeros((len(list_labels), len(list_labels))) - np.inf
    for row, labrow in enumerate(list_labels):
        for col, labcol in enumerate(list_labels):
            tmp = dfres[(dfres[var_row] == labrow) & (dfres[var_col] == labcol)]
            assert len(tmp)==1

            distmat[row, col] = tmp[var_value].values[0]
            
            if PRINT:
                print("row:", labrow, "  col:", labcol, " val:", distmat[row, col] )

    assert np.all(distmat > -np.inf)

    if PRINT:
        print(distmat.shape)
        print(len(list_labels), len(list_labels))
    
    params = {"label_vars":var_labels, "version_distance":"IGNORE"}
    Cl = Clusters(distmat, list_labels, list_labels, "rsa", params)
    
    return Cl

class Clusters(object):
    """Holds 2D matrix data, usually distnaces matrices, and has various methods for plotting and visualizing and manipulating"""
    def __init__(self, X, labels_rows=None, labels_cols=None, ver=None, params=None, trialcodes=None):
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

        if X is None:
            # then is just for methods
            return
        
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

        if trialcodes is not None:
            # Assumed to be matching the Labels
            assert len(trialcodes)==len(self.Labels)
        self.Trialcodes = trialcodes

        self.Params = params

        # Track different versions of the data.
        # --- rows
        self.LabelsDict = {
            "raw":self.Labels
            } # tracks different ways of labeling, always aligned to self.Xinput

        self.ClusterResults = {}
        self.DistanceMatrices = {}
        self.ClusterComputeAllResults = {}

        if ver=="rsa":
            # This means each row is a tuple of levels across multiple grouping
            # vars. I.E., labels is list of tuples, each of len num vars.
            label_vars = params["label_vars"] # list of str
            assert len(label_vars)==len(set(label_vars))
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
            if "label_vars" not in self.Params.keys():
                self.Params["label_vars"] = self.Params["Clraw"].Params["label_vars"]
            if False: # actually, allow diff length variable for labels  and var.. to allow theoretical..
                for lab in self.Labels:
                    assert isinstance(lab, tuple)
                    print("------")
                    print(len(lab), lab)
                    print(len(self.Params["label_vars"]), self.Params["label_vars"])
                    assert len(lab)==len(self.Params["label_vars"])
        elif ver=="pca":
            # Holes results of PCA. (NOT data tformed to PC space).
            # Square matrix of PCs (pcs, original dims)
            # pcs_explained_var: (npcs,) variance
            for f in ["pcs_explained_var"]:
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
            if n==0:
                assert False, "DId not find this label!"
            else:
                assert False, "why multiple identical lables?"
        return labs.index(label)

    def extract_dat(self, datkind="raw"):
        """ Rerutrn data matrix, but possibly processed
        """
        return self.Xinput
        # return self.XDict[datkind]

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
    def dataextract_masked_upper_triangular_flattened(self,
                                                      exclude_diag=True,
                                                      inds_rows_cols=None,
                                                      dat = None,
                                                      list_masks=None,
                                                      plot_mask=False):
        """ Return elemetns in upper diagnoal, not including
        diagnoal, flatteened into 1-array
        PARAMS:
        - inds_rows_cols, either None (Ignore) or array-like of indices into
        self.X[inds, inds], to slice a subset (will still be square).
        - exclude_diag, bool (True), then ignore the diangoanl
        - mask, boolean mask (nrow, ncol)
        """

        if isinstance(list_masks, list) and len(list_masks)==0:
            list_masks = None

        if dat is None:
            dat = self.Xinput
        assert dat.shape[1]==dat.shape[0]

        # Start with all data
        MA = self._rsa_matindex_generate_all_true()

        # Input masks, apply
        if list_masks is not None:
            for ma in list_masks:
                MA = MA & ma

        # Input rows, cols (rectangle).
        if inds_rows_cols is not None:
            ma = self._rsa_matindex_convert_to_mask_rect(inds_rows_cols, inds_rows_cols)
            MA = MA & ma

        # Take upper triangular
        ma = self._rsa_matindex_generate_upper_triangular(exclude_diag=exclude_diag)
        MA = MA & ma


        # Flatten
        vec = dat[MA].flatten()

        # Plot mask
        if plot_mask:
            fig, ax = plt.subplots()
            self.rsa_matindex_plot_bool_mask(MA, ax)
            ax.set_title("Final mask")

            return vec, fig, ax
        else:
            return vec

    # def dataextract_upper_triangular_flattened_OLD(self,
    #                                            exclude_diag=True,
    #                                            inds_rows_cols=None,
    #                                            dat = None):
    #     """ Return elemetns in upper diagnoal, not including
    #     diagnoal, flatteened into 1-array
    #     PARAMS:
    #     - inds_rows_cols, either None (Ignore) or array-like of indices into
    #     self.X[inds, inds], to slice a subset (will still be square).
    #     - exclude_diag, bool (True), then ignore the diangoanl
    #     - mask, boolean mask (nrow, ncol)
    #     """
    #
    #
    #     if dat is None:
    #         dat = self.Xinput
    #
    #     assert dat.shape[1]==dat.shape[0]
    #
    #     if inds_rows_cols is not None:
    #         dat = dat[inds_rows_cols, :][:, inds_rows_cols] # (len inds, len inds)
    #
    #     n = dat.shape[0]
    #     if exclude_diag:
    #         inds = np.triu_indices(n, 1)
    #     else:
    #         inds = np.triu_indices(n, 0)
    #     vec = dat[inds].flatten()
    #     return vec

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
                           ax=None, robust=False, ylabel=None):
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

        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return fig, X, labels_col, labels_row, ax

    def plot_heatmap_data(self, datkind="raw", labelkind="raw", 
        nrand = None, SIZE=12, zlims=(None, None),
                          sort_rows_by=None, diverge=False,
                          ylabel=None, X=None):
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
        - sort_rows_by, list-like, len nrows. will use sort indices from sorting
        this to then sort the rows before plotting.
        RETURNS:
        - fig, X, labels
        """
        
        labels_row = self.extract_labels(labelkind, axis=0)
        labels_col = self.extract_labels(labelkind, axis=1)

        # Extract data
        if X is None:
            X = self.extract_dat(datkind)
        else:
            assert X.shape[0]==len(labels_row)
            assert X.shape[1]==len(labels_col)

        if sort_rows_by is not None:
            inds = np.argsort(sort_rows_by)
            X = X[inds,:]
            labels_row = [labels_row[i] for i in inds]
            # X, labels_row = self.sort_by_labels(labels=sort_rows_by)

        # Take subset of rows?
        if nrand is not None:
            import random
            inds = random.sample(range(X.shape[0]), nrand)
            X = X[inds, :]
            labels_row = [labels_row[i] for i in inds]
            # labels_col = [labels_col[i] for i in inds]
        
        return self._plot_heatmap_data(X, labels_row, labels_col, SIZE, zlims, diverge=diverge, ylabel=ylabel)

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
            print("ERROR")
            print(err)

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
        """ for each trial, assign it a label based simply on argmax()
        of the similarity score.
        """
        from scipy.stats import entropy

        # self.Xinput is matric similarities.
        sims_max = self.Xinput.max(axis=1)
        sims_min = self.Xinput.min(axis=1)
        sims_median = np.median(self.Xinput, axis=1)
        # sims_mean = self.Xinput.mean(axis=1)
        sims_concentration = (sims_max - sims_min)/(sims_max + sims_min)
        sims_concentration_v2 = (sims_max - sims_median)/(sims_max + sims_median)
        
        # which shape does it match the best
        inds_maxsim = np.argmax(self.Xinput, axis=1) # for each row, get its best column
        cols_maxsim = [self.LabelsCols[i] for i in inds_maxsim]

        # entropy
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

        fig, X, labels_col, labels_row, ax = self._plot_heatmap_data(
            pcamodel.components_, range(len(pcamodel.components_)),
            self.LabelsCols, SIZE=5, rotation=90)
        ax.set_ylabel("PCs")
        ax.set_xlabel("original dims")
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

    def cluster_gmm_extract_best_n(self, ver="gmm"):
        """ Get the N that has the best score, using
        BIC (or cross-validated score, optional)
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
        # gmm_n_best = list_n[np.argmax(list_crossval)]
        gmm_n_best = list_n[np.argmin(list_bic)]

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
    def distsimmat_convert_distr_from_self(self, accurately_estimate_diagonal=False):
        """ Compute distmat using eucldiina dastnace between centroids, only
        works if this is holding distamt between trials, here , with trials 
        grouping based on variables in self.Params, therefore is agging, so
        that one row per level of conjunctive variables
        """
        version_distance="euclidian_unbiased"
        CldistAgg = self.Params["Clraw"].distsimmat_convert_distr(self.Params["label_vars"], version_distance, accurately_estimate_diagonal)
        return CldistAgg


    def distsimmat_convert_distr(self, agg_vars, version_distance, min_n_dat = 3,
                                 accurately_estimate_diagonal=True):
        """ Generate distance matrix, where distance is computed between
        matrices, e.g., multivariate distributions, as opposted to firsrt
        averaging into a single vector for each row, and then distance
        (as in distmat_convert)
        PARAMS:
        - min_n_at,          # at least this many datapts per conj.

        """
        # Convert from Clraw (holding all trials) to distance matrix that compares _distributions_ of data
        # - ie instead of first taking mean over label groupings (agg) keep separated before computing distance.

        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
        from pythonlib.tools.distfunctools import euclidian_unbiased, distmat_construct_wrapper, euclidian

        # Get data
        dflab = self.rsa_labels_return_as_df(True)
        X = self.Xinput

        # Get grouping of labels
        groupdict = grouping_append_and_return_inner_items(dflab, agg_vars)

        if False:
            print(agg_vars)
            for k, v in groupdict.items():
                print(k, len(v))
            assert False

        # Convert to list, which can then pass into distmat constructor
        list_X = []
        list_lab = []
        list_exclude = []
        for i, (grp, inds) in enumerate(groupdict.items()):
            list_X.append(X[inds,:])
            list_lab.append(grp)
            # print(len(inds))
            if len(inds) < min_n_dat:
                print("Min n dat:", min_n_dat)
                print("Current n", len(inds))
                assert False, "Stop allowing this to continue -- false negatives. and also, this needs c (??)"
                # list_exclude.append(i)

        if version_distance=="euclidian_unbiased":
            # Unbiased estimate of eucl distance, based on resampling
            # Also, for diagonal, gets estimate by splitting data in half, multiple
            # folds

            # Skip all cases that have
            # coords_skip = [(i,i) for i in list_exclude]
            D = distmat_construct_wrapper(list_X, list_X, euclidian_unbiased,
                                          accurately_estimate_diagonal=accurately_estimate_diagonal,
                                          inds_skip_rows_or_cols=list_exclude)
        else:
            print(version_distance)
            assert False

        params = {
            "version_distance":version_distance,
            "Clraw":self,
        }
        return Clusters(D, list_lab, list_lab, ver="dist", params=params)

    def distsimmat_convert(self, version_distance, list_X = None):
        """ Convert data (ndat, ndim) to distnace matreix (ndat, ndat)
        - Distance --> more positive is always more distance, by convention. This
        means pearson corr will range between 0 and 1, for example. By convention 0 is close.
        PARAMS:
        - list_X, if not None, then must be [X1, X2], which have same shape, and will be used to compute
        distance. Otherwuse ises [self.Xinput, self.Xinput]
        RETURNS:
            - Cl, ClustClass object.
        """
        from pythonlib.tools.distfunctools import distmat_construct_wrapper

        assert list_X is None, "not coded, decided to just write this form scratch"

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
        elif version_distance=="_pearson_raw":
            # Pearson, without any transformation.
            D = np.corrcoef(X)
        elif version_distance=="angle":
            # angle betwen vectors, ie., cos(angle) = dot(a,b)/(norm(a) * norm(b))
            # recaled so that 0 means parallel (angle 0) and 1 means ortho (angle 90).
            def angle_dot(a, b):
                """ Return angle between a and b, in degrees [0, 90]"""
                if np.all(a==b):
                    # otherwise get nan
                    return 0.
                dot_product = np.dot(a, b)
                prod_of_norms = np.linalg.norm(a) * np.linalg.norm(b)
                # print(dot_product)
                # print(prod_of_norms)
                angle = round(np.degrees(np.arccos(dot_product / prod_of_norms)), 2) # between 0 and 90
                return angle

            def func(vals1, vals2):
                """ return angle, betwen 0 and 1"""
                ang = angle_dot(vals1, vals2)
                dist = ang/90.
                return dist

            vals = [X[i,:] for i in range(X.shape[0])]
            D = distmat_construct_wrapper(vals, vals, func)
            assert ~np.any(np.isnan(D))

        else:
            print(version_distance)
            assert False
        assert ndat==D.shape[0]
        params = {
            "version_distance":version_distance,
            "Clraw":self,
        }
        return Clusters(D, self.Labels, self.Labels, ver="dist", params=params, trialcodes=self.Trialcodes)

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
        elif self.Version=="dist" and isinstance(self.Labels[0], tuple):
            return True
        elif self.Labels == self.LabelsCols and isinstance(self.Labels[0], tuple):
            # Then is symmetric.
            return True
        else:
            return False

    def rsa_plot_points_split_by_var_flex(self, var_x_axis_each_subplot, yvar,
                                          var_lines_within_subplots,
                                          savedir,
                                          vars_subplot_rows, vars_subplot_cols,
                                          vars_figures):
        """
        Very flexible helper to make grid of subplots, with fleixcble paremteres for which
        variable controls which aspect of plot.

        :param var_x_axis_each_subplot:
        :param yvar:
        :param var_lines_within_subplots:
        :param vars_subplot_rows:
        :param vars_subplot_cols:
        :param vars_figures:
        :return:
        """
        from pythonlib.tools.snstools import rotateLabel
        from pythonlib.tools.plottools import savefig

        # Get distances between all pts (long-form)
        dfdists = self.rsa_dataextract_with_labels_as_flattened_df(keep_only_one_direction_for_each_pair=False)
        list_lev_figures = self.rsa_labels_extract_var_levels()[vars_figures]

        # One figure for each level
        for lev_figures in list_lev_figures:
            dfthis = dfdists[(dfdists[f"{vars_figures}_row"] == lev_figures) & (dfdists[f"{vars_figures}_col"] == lev_figures)]

            fig = sns.catplot(data=dfthis, x=var_x_axis_each_subplot, y=yvar, row=vars_subplot_rows, col=vars_subplot_cols,
                       hue=var_lines_within_subplots, kind="point")

            rotateLabel(fig)

            path = f"{savedir}/levfigure={lev_figures}.pdf"
            print(path)
            savefig(fig, path)

            plt.close("all")

    def rsa_plot_heatmap(self, sort_order=None, diverge=False,
                         X=None, ax=None, mask=None, zlims=None,
                         SIZE=12, skip_if_nrows_more_than=200):
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

        assert X is None, "not coded"

        if zlims is None:
            zlims = (None, None)

        assert self._rsa_check_compatible()

        # Pull out data in correct format, and return as clusters.
        X = self.Xinput

        if len(X)>skip_if_nrows_more_than:
            print("rsa_plot_heatmap SKIPPING, too many rows:", len(X))
            # Return dummy fig, or else downstream fails
            fig, ax = plt.subplots()
            ax.set_title(f"rsa_plot_heatmap SKIPPING, too many rows: {len(X)}")
            return fig, ax

        # Sort labels if needed
        labels_rows = self.Labels
        labels_cols = self.LabelsCols

        # Mask (always do BEFORE sorting)
        if mask is not None:
            X = X.copy()
            X[~mask] = -1.

        if sort_order is None:
            # go (0,1, 2....)
            sort_order = tuple(range(len(labels_rows[0])))

        # Sort
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
                                                                   labels_cols, diverge=diverge,
                                                                     ax=ax, zlims=zlims,
                                                                     SIZE=SIZE)
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

        def dist_func_strokes(StrokeClass1, StrokeClass2):
            # Distance function between StrokeClass instances.
            # NOTE: calling StrokeClass1() returns np array.
            # 0 is perfect same
            # ~0.1 is great
            # ~0.3 is not bad.
            # higher is bad... (into 1's..)

            from pythonlib.drawmodel.strokedists import distStrokWrapper
            distStrokWrapper(StrokeClass1(), StrokeClass2(), ver="dtw_vels_2d", align_to_onset=True,
                             fs=None, rescale_ver="stretch_to_1_diag")

        # Categorical
        def dist_func_cat(x,y):
            # Categorical
            if x==y:
                # same
                return 0.
            else:
                return 1.

        def dist_func_for_dist_angle(x, y):
            """ x and y are dist angle, which is tuple
            (distcum bin, angle bin). For dist does ordinal, and
            angle, does circular ordinal
            """

            assert isinstance(x, tuple)
            assert isinstance(y, tuple)
            assert len(x)==len(y)
            dist = 0
            dist += np.abs(x[0] - y[0])
            dist += dist_angle_hack(x[1], y[1])
            return dist

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

        def dist_angle_hack_8(x,y, DEBUG=False):
            """
            anlge hack, for 8 bins...
            """
            if np.isnan(x) or np.isnan(y):
                # can happen if there is no next stroke (e.g,, next reach..)
                return np.nan

            # print(x)
            # print(type(x))
            # assert isinstance(x, int)
            assert x>0 and x<9
            assert y>0 and y<9

            if DEBUG:
                for j in range(1,9):
                    for i in range(1,9):
                        print(i, j, ' -- ' , dist_angle_hack(i, j))
            y2 = y-8
            x2 = x-8

            d = np.min([np.abs(x-y), np.abs(x-y2), np.abs(y-x2)])

            return d

        # ordinal
        def dist_func_ord(x,y):
            # 0==same
            # 1,2,3. ... further apart.
            return np.abs(x-y)

        def dist_func_tuple_of_ordinal(x, y):
            # compares each item in tuple, then sums up across items.
            assert isinstance(x, tuple)
            assert isinstance(y, tuple)
            assert len(x)==len(y)
            dist = 0
            for xx, yy in zip(x, y):
                dist += np.abs(xx-yy)
            return dist

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
        vars_categorical = ["shape_oriented", "shape",
                            "gridloc", "seqc_0_loc",
                            "stroke_index_semantic",
                            "CTXT_loc_next", "CTXT_shape_next"] + tmp

        # Ordinal
        vars_ordinal = ["stroke_index", "stroke_index_fromlast",
                        "stroke_index_fromlast_tskstks", "gridloc_x",
                        "FEAT_num_strokes_task", "FEAT_num_strokes_beh",
                        "FEAT_num_strokes_task"]

        vars_tuple_of_ordinal = [] # each index in the tuple, use as ordinal.

        if var=="velmean_thbin":
            print("... using distfunc: dist_angle_hack_8")
            return dist_angle_hack_8
        elif var=="dist_angle":
            return dist_func_for_dist_angle
        if var in vars_categorical:
            # Categorical
            print("... using distfunc: dist_func_cat")
            return dist_func_cat
        elif var in vars_angle_binned:
            # Hacky, angles binned
            print("... using distfunc: vars_angle_binned")
            return dist_angle_hack
        elif var in vars_ordinal:
            # ordinal
            print("... using distfunc: dist_func_ord")
            return dist_func_ord
        elif var in vars_tuple_of_ordinal:
            print("... using distfunc: dist_func_ord")
            return dist_func_tuple_of_ordinal
        else:
            pass # Try to get it auto below.
        
        ########## SECOND, TRY AUTOMATICALLY, based on type.
        vals = self.rsa_index_values_this_var(var)
        list_types = list(set([type(v) for v in vals]))
        at_least_one_type_is_tuple = any([isinstance(t, tuple) for t in list_types])
        from pythonlib.behavior.strokeclass import StrokeClass

        if len(list_types)>1 and at_least_one_type_is_tuple:
            # Mxing these types, usually is categorical
            print("... using distfunc: dist_func_cat")
            return dist_func_cat
        elif len(list_types)>1:
            # Then not sure.
            print(var)
            print(vals)
            assert False, "dont know what distance functin to use"
        else:
            # Only one type
            t = list_types[0]
            v = vals[0]
            if t in (str, bool, tuple):
                print("... using distfunc: dist_func_cat")
                return dist_func_cat
            elif t in (int,):
                # Not sure, just use categorical
                print("... using distfunc: dist_func_cat")
                return dist_func_cat
            elif isinstance(v, StrokeClass):
                print("... using distfunc: dist_func_strokes")
                return dist_func_strokes
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

    # def rsa_mask_context_split_levels_of_var(self, vars_context, PLOT=False,
    #                                               exclude_diagonal=True):
    #     """
    #     Return dict mapping between each level of vars_context (grouping var) and the
    #     mask that are distance between this same level. Useful for computing "within-context"
    #     scores (i.e., by taking & with mask for different effect).
    #     NOTE: WILL be only upper triangular
    #     :param vars_context:
    #     :return: map_grp_to_mask, dict[grp]--> ma, where grp is tuple of classea dn ma is bool.
    #     """
    #     from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
    #
    #     # Get row indices for levels of conjunction var
    #     dflab = self.rsa_labels_return_as_df()
    #     grpdict = grouping_append_and_return_inner_items(dflab, vars_context)
    #
    #     # Get each mask
    #     if exclude_diagonal:
    #         ma_ut = self._rsa_matindex_generate_upper_triangular()
    #     else:
    #         ma_ut = self._rsa_matindex_generate_all_true()
    #     map_grp_to_mask = {}
    #     for grp, indrows in grpdict.items():
    #         ma = self._rsa_matindex_convert_to_mask_rect(indrows, indrows) # same, for this grp
    #         map_grp_to_mask[grp] = ma & ma_ut
    #
    #     if PLOT:
    #         ncols = 2
    #         nrows = int(np.ceil(len(map_grp_to_mask)/ncols))
    #         SIZE = 5
    #         fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))
    #         for ax, (grp, ma) in zip(axes.flatten(), map_grp_to_mask.items()):
    #             self.rsa_matindex_plot_bool_mask(ma, ax)
    #             ax.set_title(grp)
    #
    #     return map_grp_to_mask

    def rsa_mask_context_split_levels_of_conj_var(self, vars_context, PLOT=False,
                                                  exclude_diagonal=True, contrast="same"):
        """
        [TESTED CAREFULLY - done]
        Return dict mapping between each level of vars_context (grouping var) and a mast that
        pulls out speicifc indices for each level of vars_context, which have a user-inputed "contrast",
        such as "same".
        NOTE: WILL be only upper triangular
        :param vars_context:
        :param contrast: str. What columns to get (note: the rows are always the same, i.e, speciifc
        to each level of vars_context). either:
        - "same", then columns will be identical to rows. To get mask that are distance between this same level.
        Useful for computing "within-context" scores (i.e., by taking & with mask for different effect).
        - "diff", then gets pairs such that one of the column or row is the level.
        - "any", then is sum of same and diff -- i.e, at least one of the row or col must be the level.
        :return: map_grp_to_mask, dict[grp]--> ma, where grp is tuple of classea dn ma is bool.
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items

        if contrast == "any":
            # This is just sum of diff and same.
            map_grp_to_mask_SAME = self.rsa_mask_context_split_levels_of_conj_var(vars_context, PLOT, exclude_diagonal, "same")
            map_grp_to_mask_DIFF = self.rsa_mask_context_split_levels_of_conj_var(vars_context, PLOT, exclude_diagonal, "diff")
            assert map_grp_to_mask_DIFF.keys() == map_grp_to_mask_SAME.keys()
            map_grp_to_mask = {}
            for grp in map_grp_to_mask_SAME.keys():
                assert not np.any(map_grp_to_mask_SAME[grp] & map_grp_to_mask_DIFF[grp])
                map_grp_to_mask[grp] = map_grp_to_mask_DIFF[grp] | map_grp_to_mask_SAME[grp]
        else:
            # Get row indices for levels of conjunction var
            dflab = self.rsa_labels_return_as_df()
            grpdict = grouping_append_and_return_inner_items(dflab, vars_context)

            # Get each mask
            if exclude_diagonal:
                ma_ut = self._rsa_matindex_generate_upper_triangular()
            else:
                ma_ut = self._rsa_matindex_generate_all_true()
            map_grp_to_mask = {}
            for grp, indrows in grpdict.items():
                if contrast=="same":
                    # same, for this grp
                    indcols = indrows
                    ma = self._rsa_matindex_convert_to_mask_rect(indrows, indcols) #
                elif contrast=="diff":

                    # One of the pair must be in this level, the other must not.
                    indothers = [i for i in range(len(dflab)) if i not in indrows]

                    # get cases where row is this level, but column is not
                    ma1 = self._rsa_matindex_convert_to_mask_rect(indrows, indothers) #

                    # cases where col is this level, but row is not.
                    ma2 = self._rsa_matindex_convert_to_mask_rect(indothers, indrows) #

                    # either one or the other.
                    ma = ma1 | ma2
                # elif contrast=="any":
                #
                #     # same, for this grp
                #     indcols = indrows
                #     ma_same = self._rsa_matindex_convert_to_mask_rect(indrows, indcols) #
                #
                #     # One of the pair must be in this level, the other must not.
                #     indothers = [i for i in range(len(dflab)) if i not in indrows]
                #
                #     # get cases where row is this level, but column is not
                #     ma1 = self._rsa_matindex_convert_to_mask_rect(indrows, indothers) #
                #
                #     # cases where col is this level, but row is not.
                #     ma2 = self._rsa_matindex_convert_to_mask_rect(indothers, indrows) #
                #
                #     # either one or the other.
                #     ma_diff = ma1 | ma2
                #
                #     # COmbine
                #     assert not np.any(ma_same & ma_diff)
                #     ma = ma_same | ma_diff
                #
                else:
                    print(contrast)
                    assert False
                map_grp_to_mask[grp] = ma & ma_ut

        if PLOT:
            ncols = 2
            nrows = int(np.ceil(len(map_grp_to_mask)/ncols))
            SIZE = 7
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))
            for ax, (grp, ma) in zip(axes.flatten(), map_grp_to_mask.items()):
                self.rsa_matindex_plot_bool_mask(ma, ax)
                ax.set_title(grp)

        return map_grp_to_mask


    def rsa_mask_context_helper_simple(self, diffctxt_vars_same=None, diffctxt_vars_diff=None, diff_context_ver=None,
                                       PLOT=False, mask_out_nans=True, path_for_save_print_lab_each_mask=None,
                                       only_upper_triangle=True):
        """
        Define a context, where context is a mask on self data, i.e., over pairs, where the pairs with True
        satisfy BOTH diffctxt_vars_same and diffctxt_vars_diff.

        i.e, context defined as pairs that are same for ALL var in diffctxt_vars_same, 
        and diff for ALL or AT_LEAST_ONE [depending on diff_context_ver] var in diffctxt_vars_diff
        
        TESTED CAREFULLY, by inspecting the print of the pairs of extract row-col, for each of these options:
        1. 
        diffctxt_vars_same = []
        diffctxt_vars_diff = [a, b], for both "diff_complete", and "diff_at_least_one.

        2.
        diffctxt_vars_same = [c]
        diffctxt_vars_diff = [a, b], for both "diff_specific", "diff_specific_lenient.
        
        3.
        diffctxt_vars_same = [c, d]
        diffctxt_vars_diff = []
        """

        if diffctxt_vars_diff is not None and len(diffctxt_vars_diff)==0:
            diffctxt_vars_diff = None

        if diffctxt_vars_same is not None and len(diffctxt_vars_same)==0:
            diffctxt_vars_same = None
            
        if diffctxt_vars_same is not None and diffctxt_vars_diff is not None:
            # assert diff_context_ver not in ["diff_complete", "diff_at_least_one"], "problem, rsa_mask_context_helper() will stupidly combine diffctxt_vars_same and diffctxt_vars_diff"
            if diff_context_ver is None:
                # Then use strict
                diff_context_ver = "diff_specific"
            assert diff_context_ver in ["diff_specific", "diff_specific_lenient"], "the two methods that take into accout both same and diff"
            MASKS, fig, axes = self.rsa_mask_context_helper(None, None, diff_context_ver,
                                    diffctxt_vars_same, diffctxt_vars_diff,
                                    PLOT, mask_out_nans, path_for_save_print_lab_each_mask,
                                    only_upper_triangle=only_upper_triangle)
            ma = MASKS["context_diff"]
        elif diffctxt_vars_same is None and diffctxt_vars_diff is not None:
            if diff_context_ver is None:
                # Then use strict
                diff_context_ver = "diff_complete"
            assert diff_context_ver in ["diff_complete", "diff_at_least_one"], "otherwise will fail, as will look for diffctxt_vars_same"
            # assert diff_context_ver not in ["diff_complete", "diff_at_least_one"], "problem, rsa_mask_context_helper() will stupidly combine diffctxt_vars_same and diffctxt_vars_diff"
            MASKS, fig, axes = self.rsa_mask_context_helper(None, None, diff_context_ver,
                                    None, diffctxt_vars_diff,
                                    PLOT, mask_out_nans, path_for_save_print_lab_each_mask,
                                    only_upper_triangle=only_upper_triangle)
            ma = MASKS["context_diff"]            
        elif diffctxt_vars_same is not None and diffctxt_vars_diff is None:
            # Then you want same
            # if mask_out_nans:
            #     ma_not_nan = ~np.isnan(self.Xinput)
            # else:
            #     ma_not_nan = self._rsa_matindex_generate_all_true()
            # ma_ut = self._rsa_matindex_generate_upper_triangular(exclude_diag=False)
            # ma = self.rsa_matindex_same_diff_mult_var_flex(vars_same=vars_context)
            diff_context_ver = "SKIP"
            MASKS, fig, axes = self.rsa_mask_context_helper(None, None, diff_context_ver,
                                    diffctxt_vars_same, None,
                                    PLOT, mask_out_nans, path_for_save_print_lab_each_mask,
                                    only_upper_triangle=only_upper_triangle)
            ma = MASKS["context_same"]            
        else:
            assert False

        return ma, fig, axes
    
    def rsa_mask_context_helper(self, var_effect, vars_context, diff_context_ver,
                                diffctxt_vars_same=None, diffctxt_vars_diff=None,
                                PLOT=False, mask_out_nans=True,
                                path_for_save_print_lab_each_mask=None, only_upper_triangle=True):
        """
        INTUITIVE EXPLANATION, as it is not that intuitive. First, note that context and var_other become mixed here.
        This is how:

        var_effect: same and different is obvious.
        context_var_other (i.e., the thing that combines var_other and context). Indeed, I generally
        have vars_context identical to the variables in context.
        - same: all levels must be same.
        - diff: 
        --- if diffctxt_vars_same and diffctxt_vars_diff are presented: then is entirely defined by this.
        --- if diffctxt_vars_same and diffctxt_vars_diff are both None: then is entirely defined by vars_context

        In other words, diffctxt_vars_same and diffctxt_vars_diff are useful if you want to see effect of 
        semeting (diffctxt_vars_diff), while holding other things constant (diffctxt_vars_same).

        -------------------------------------------

        Generate boolean mask to focus on "context" defined by relations for given variables.
        You can use this mask to then slice out data for analyses (restricting data).
        PARAMS:
        - var_effect,
        - diff_context_ver, str, how to define diff context (NOTE: This does NOT
        affect how "same context" is defined)
        For diff context, multiple ways.
        --- "diff_complete", for all vars in vars_context, they must be different.
        --- "diff_at_least_one", at least one var in vars_context must be diff.
        --- "diff_specific", hand-inputed, must satisfy constraint for
        vars that must be same, and vars that must be diff. This requires
        diffctxt_vars_same and diffctxt_vars_diff, both list of str.
        - mask_out_nans, bool, usualyl True, since nan is used to signal not neough data..
        NOTE: to define "same context", for all vars in vars_context, they must be same level. There is only
        one way to define a same context.
        :return:
        - MASKS, dict holding masks.
        NOTE: context masks are FINAL (even upper triangle, etc).
        NOTE: effect masks are NOT final. THey need to be AND-ed with context masks..
        """

        if mask_out_nans:
            ma_not_nan = ~np.isnan(self.Xinput)
        else:
            ma_not_nan = self._rsa_matindex_generate_all_true()

        if vars_context is None:
            # Then auto get as concat of same and diff
            vars_context = []
            if diffctxt_vars_same is not None:
                vars_context += diffctxt_vars_same
            if diffctxt_vars_diff is not None:
                vars_context += diffctxt_vars_diff
            assert len(vars_context)>0
            
        # Same context
        ma_context_same = self.rsa_matindex_same_diff_mult_var_flex(vars_same=vars_context)
        if only_upper_triangle:
            ma_ut = self._rsa_matindex_generate_upper_triangular(exclude_diag=False)
            ma_context_same = ma_context_same & ma_ut & ma_not_nan
        else:
            ma_context_same = ma_context_same & ma_not_nan

        # Diff context
        if diff_context_ver=="SKIP":
            # Instruting functoin to ignore diff
            ma_context_diff = self._rsa_matindex_generate_all_true()
        elif diff_context_ver=="diff_complete":
            # Then every var must be diff
            ma_context_diff = self.rsa_matindex_same_diff_mult_var_flex(vars_diff=vars_context)
        elif diff_context_ver=="diff_at_least_one":
            # Then at least one of var is diff
            ma_context_diff = self.rsa_matindex_mask_if_any_var_is_diff(vars_context)
        elif diff_context_ver=="diff_specific":
            # Specific combination of vars being same and diff
            # ALL vars must be same or diff individually -- i.e,. same for all vars in "same" and diff for all in "diff"
            assert isinstance(diffctxt_vars_diff, (tuple, list))
            assert isinstance(diffctxt_vars_same, (tuple, list))
            assert len(diffctxt_vars_diff)>0
            try:
                assert all([v in vars_context for v in diffctxt_vars_same])
                assert all([v in vars_context for v in diffctxt_vars_diff])
            except Exception as err:
                print(vars_context)
                print(diffctxt_vars_same)
                print(diffctxt_vars_diff)
                raise err
            # Previously this, but realized that context can be more
            # assert sorted(diffctxt_vars_same + diffctxt_vars_diff) == sorted(vars_context)
            ma_context_diff = self.rsa_matindex_same_diff_mult_var_flex(
                vars_same=diffctxt_vars_same, vars_diff=diffctxt_vars_diff)
        elif diff_context_ver=="diff_specific_lenient":
            # Specific combination of vars being same and diff
            # "lenient" means that only need one of the "diff" vars to be different for this to pass criterion
            # -- i.e,. same for all vars in "same" and diff for just one var in "diff"
            assert isinstance(diffctxt_vars_diff, (tuple, list))
            assert isinstance(diffctxt_vars_same, (tuple, list))
            assert len(diffctxt_vars_diff)>0
            try:
                assert all([v in vars_context for v in diffctxt_vars_same])
                assert all([v in vars_context for v in diffctxt_vars_diff])
            except Exception as err:
                print(vars_context)
                print(diffctxt_vars_same)
                print(diffctxt_vars_diff)
                raise err
            # Previously this, but realized that context can be more
            # assert sorted(diffctxt_vars_same + diffctxt_vars_diff) == sorted(vars_context)
            ma_context_diff = self.rsa_matindex_same_diff_mult_var_flex(
                vars_same=diffctxt_vars_same, vars_diff=diffctxt_vars_diff,
                lenient_diff = True)
        else:
            print(diff_context_ver)
            assert False

        if only_upper_triangle:
            ma_ut = self._rsa_matindex_generate_upper_triangular(exclude_diag=True)
            ma_context_diff = ma_context_diff & ma_ut & ma_not_nan
        else:
            ma_context_diff = ma_context_diff & ma_not_nan

        # Effect var only
        if var_effect is not None:
            ma_effect_same = self.rsa_matindex_same_diff_mult_var_flex(vars_same=[var_effect])
            ma_effect_diff = self.rsa_matindex_same_diff_mult_var_flex(vars_diff=[var_effect])
        else:
            ma_effect_same = None
            ma_effect_diff = None

        MASKS = {
            "context_same":ma_context_same,
            "context_diff":ma_context_diff,
            "effect_same":ma_effect_same,
            "effect_diff":ma_effect_diff
        }

        if PLOT:
            
            if ma_effect_diff is not None:
                # get each conjunction mask
                ma_effD_ctxtD = ma_effect_diff & ma_context_diff
                ma_effD_ctxtS = ma_effect_diff & ma_context_same
                ma_effS_ctxtD = ma_effect_same & ma_context_diff
                ma_effS_ctxtS = ma_effect_same & ma_context_same

                titles = ["ma_context_same", "ma_context_diff", "ma_effect_same", "ma_effect_diff",
                        "ma_effD_ctxtD", "ma_effD_ctxtS", "ma_effS_ctxtD", "ma_effS_ctxtS"]
                masks = [ma_context_same, ma_context_diff, ma_effect_same, ma_effect_diff,
                        ma_effD_ctxtD, ma_effD_ctxtS, ma_effS_ctxtD, ma_effS_ctxtS]
            else:
                titles = ["ma_context_same", "ma_context_diff"]
                masks = [ma_context_same, ma_context_diff]

            fig, axes = plt.subplots(3,3, figsize=(25, 28))
            for ax, ma, tit in zip(axes.flatten(), masks, titles):
                # Plot the actual mask
                self.rsa_matindex_plot_bool_mask(ma, ax)

                # add the mean score in this mask to title.
                if np.sum(ma)>0:
                    sc = np.mean(self.Xinput[ma])
                    ax.set_title(f"{tit}-mean_{sc:.2f}", color="r")
                else:
                    ax.set_title(f"{tit}", color="r")
                
                # also print the mask
                if path_for_save_print_lab_each_mask is not None:
                    from pythonlib.tools.expttools import fileparts, deconstruct_filename
                    fp = fileparts(path_for_save_print_lab_each_mask)
                    # deconstruct_filename(path_for_save_print_lab_each_mask)
                    _path = f"{fp[0]}{fp[1]}-{tit}{fp[2]}"                
                    self.rsa_matindex_print_mask_labels(ma, _path)

            return MASKS, fig, axes
        else:
            return MASKS, None, None

    def rsa_dfdist_to_dfproj_index_datapts(self, dfdist_pts, var_score="dist_mean", var_effect = "idx_morph_temp",
                            effect_lev_base1=0, effect_lev_base2=99):
        """
        For each datapt, get its "projection" between the two groups of datapts at the extreme ends of a ordered variable.
        I.e. compute an index which is (d2)/(d1 + d2) where d1 is distance from this datapt to the group 1...

        NOTE: Like dfdist_to_dfproj_index, but here dfdist rows to represent datapts/trials (So this would be trial vs. group of tirals), 
        as oposed to there, whihc is rows are distances between groups of trials. 

        PARAMS:
        - var_score, which scoree to use. By default, use dist_mean, which is the most raw. Simply the eucl distance.
        - var_effect, the variable whose 2 levels will form the "endpoints"
        """

        list_idx_datapt = sort_mixed_type(dfdist_pts["idx_row_datapt"].unique())

        res_dist_index = []
        for idx_datapt in list_idx_datapt:
            tmp = dfdist_pts[(dfdist_pts["idx_row_datapt"] == idx_datapt) & (dfdist_pts[f"{var_effect}_2"]==effect_lev_base1)]
            if not len(tmp)==1:
                print(idx_datapt)
                print(tmp)
                assert False, "prob need to set get_only_one_direction==False or version_datapts==True"
            d1 = tmp[var_score].values[0]

            tmp = dfdist_pts[(dfdist_pts["idx_row_datapt"] == idx_datapt) & (dfdist_pts[f"{var_effect}_2"]==effect_lev_base2)]
            if not len(tmp)==1:
                print(idx_datapt)
                print(tmp)
                assert False, "prob need to set get_only_one_direction==False or version_datapts==True"
            d2 = tmp[var_score].values[0]

            # Compute index
            dist_index = d1/(d1+d2)
            
            # Store
            res_dist_index.append({
                "idx_row_datapt":idx_datapt,
                "labels_1_datapt":tmp["labels_1_datapt"].values[0],
                f"{var_effect}":tmp[f"{var_effect}_1"].values[0],
                "dist_index":dist_index,
            })
        dfproj_index = pd.DataFrame(res_dist_index)

        # Optionally append trialcode, if each row corresponds to a trial.
        if "trialcode" in dfdist_pts.columns:
            map_idx_tc = {row["idx_row_datapt"]:row["trialcode"] for _, row in dfdist_pts.iterrows()}
            dfproj_index["trialcode"] = [map_idx_tc[idx] for idx in dfproj_index["idx_row_datapt"]]

            assert set(dfproj_index["trialcode"]) == set(dfdist_pts["trialcode"])
        return dfproj_index

    def rsa_distmat_score_all_pairs_of_label_groups_datapts_datapts():
        """
        """
        assert False, "this is done in rsa_dataextract_with_labels_as_flattened_df"

    def rsa_distmat_score_all_pairs_of_label_groups_datapts(self, label_vars=None,
            return_as_clustclass=False, list_grps_get=None, ignore_self_distance=False):
        """
        See rsa_distmat_score_all_pairs_of_label_groups, except here gets pairs of (datapt vs. group) whereas there was
        (group vs group).
        PARAMS:
        - list_grps_get, list of grp labels, if not None, then restricts analyses to just these. Will still use all trials 
        for single datapts, but this restricts which grps those datapts are compared to. This is useful for making this quiker, and
        also to make sure doesnt fail if you ahve grps with only 1 datapt (it will fail).
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good


        assert self.Labels == self.LabelsCols, "assumes so for soem of below"

        # Get dataframe of labels for each trial
        dflabels = self.rsa_labels_return_as_df()

        # Get all label groupings and their trial indices.
        if label_vars is None:
            label_vars= self.rsa_labels_extract_label_vars()
        grpdict = grouping_append_and_return_inner_items_good(dflabels, label_vars)

        # Prune grps, if desired.
        if list_grps_get is not None:
            grpdict = {grp:inds for grp, inds in grpdict.items() if grp in list_grps_get}

        if len(grpdict)==0:
            print(list_grps_get)
            grpdict = grouping_append_and_return_inner_items_good(dflabels, label_vars)
            print(grpdict.keys())
            assert False, "your input list_grps_get doesnt iunclude any items in grpdict"

        # Get distance between each pair or grps
        res = []
        for idx1 in range(len(self.Labels)):
            inds1 = [idx1]
            grp1 = self.Labels[idx1] # tuple
            for _, (grp2, inds2) in enumerate(grpdict.items()):
                
                if ignore_self_distance and grp1==grp2:
                    continue

                if inds2 == inds1:
                    # Then this is bad -- both are 1 item, so this is distance to itself
                    print(label_vars)
                    print(grp1)
                    print(grp2)
                    assert False, "how deal with this"
                
                # make sure never take distance between pt and itself
                inds2 = [_i for _i in inds2 if _i!=idx1]
                    
                # Should include below lower and upper triangle, to make sure get all pairs.
                ma = self._rsa_matindex_convert_to_mask_rect(inds1, inds2)
                X = self.Xinput[ma]
                # no need to exclude diagonal, becuase above step ensures no overlap in inds.

                res.append({
                    "idx_row_datapt":idx1,
                    "labels_1_datapt":grp1,
                    "labels_2_grp":grp2,
                    "dist_mean":np.mean(X),
                })
                
                # Expand, to get each variable in label.
                for _ivar, _var in enumerate(label_vars):

                    # Skip, this is done below.
                    # res[-1][f"{_var}_1"] = grp1[_ivar]
                    # res[-1][f"{_var}_2"] = grp2[_ivar]

                    res[-1][f"{_var}"] = grp1[_ivar] # Give a new colum which is the class of the datapt
                
                if self.Trialcodes is not None:
                    res[-1]["trialcode"] = self.Trialcodes[idx1]

        # Also get 98th percentile between pairs of pts.
        ma = self._rsa_matindex_generate_upper_triangular()
        DIST_50, DIST_98 = np.percentile(self.Xinput[ma], [50, 98])

        dfres = pd.DataFrame(res)
        dfres["DIST_50"] = DIST_50
        dfres["DIST_98"] = DIST_98

        if self.Trialcodes is not None:
            assert len(self.Labels)==len(self.Trialcodes)
            dfres["trialcode"] = [self.Trialcodes[i] for i in dfres["idx_row_datapt"]]

        # normalize the distances
        dfres["dist_norm"] = dfres["dist_mean"]/dfres["DIST_98"]

        # Convert scores to dist_yue_diff
        if ignore_self_distance == False: # Otherwise dist_yue_diff is not defined.

            cached_distances = {} # Otherwise can get very slow
            def _get_within_group_dist(grp, var_score):
                """ return scalar avearge distance wtihin this group"""
                if grp not in cached_distances:
                    tmp = dfres[(dfres["labels_1_datapt"]==grp) & (dfres["labels_2_grp"]==grp)] # len n datapts for this group.
                    cached_distances[grp] = tmp[var_score].mean()
                return cached_distances[grp]

            # def _get_within_group_dist(grp, var_score):
            #     """ return scalar avearge distance wtihin this group"""
            #     tmp = dfres[(dfres["labels_1_datapt"]==grp) & (dfres["labels_2_grp"]==grp)] # len n datapts for this group.
            #     return tmp[var_score].mean() # get the mean over all (datapt, group), which is equivalent to all pairs within group

            var_score = "dist_norm"
            list_dist_yue_diff = []
            for i, row in dfres.iterrows():
                
                # - get average within-group score for the two groups
                dist_within_group = _get_within_group_dist(row["labels_2_grp"], var_score)
                dist_across = row[var_score] # dist between this datapt and the group.

                dist_yue_diff = dist_across - dist_within_group

                list_dist_yue_diff.append(dist_yue_diff)

            dfres["dist_yue_diff"] = list_dist_yue_diff
        else:
            dfres["dist_yue_diff"] = -999 # just to avoid breaking downstream code.

        ### OTHER columns added
        dfres = self.rsa_distmat_population_columns_label_relations(dfres, label_vars)
        # from pythonlib.tools.pandastools import append_col_with_grp_index
        # for var in label_vars:
        #     dfres = append_col_with_grp_index(dfres, [f"{var}_1", f"{var}_2"], f"{var}_12")
        #     dfres[f"{var}_same"] = dfres[f"{var}_1"] == dfres[f"{var}_2"]

        ### Return as a ClustClass object
        if return_as_clustclass:
            assert False, "not coded yet -- see rsa_distmat_score_all_pairs_of_label_groups"
            # assert get_only_one_direction==False


            # labels = sorted(set(dfres["labels_1"].tolist() + dfres["labels_2"].tolist()))
            # # labels = sorted(set(self.Labels))
            # map_label_to_idx = {lab:i for i, lab in enumerate(labels)}

            # indsgotten = []
            # X = np.zeros((len(labels), len(labels)))-np.inf
            # for i, row in dfres.iterrows():
            #     indrow = map_label_to_idx[row["labels_1"]]
            #     indcol = map_label_to_idx[row["labels_2"]]
            #     X[indrow, indcol] = row[return_as_clustclass_which_var_score]

            #     assert (indrow, indcol) not in indsgotten
            #     indsgotten.append((indrow, indcol))
            # assert not np.any(X==-np.inf), "did not fill in all indices..."

            # # Alternative, mucgh slower method
            # # import numpy as np
            # # var_score = "dist_yue_diff"
            # # labels = sorted(set(self.Labels))
            # # X = np.zeros((len(labels), len(labels)))
            # # for i, labrow in enumerate(labels):
            # #     print(i)
            # #     for j, labcol in enumerate(labels):
            # #         tmp = dfres[(dfres["labels_1"] == labrow) & (dfres["labels_2"] == labcol)]
            # #         assert len(tmp)==1
            # #         X[i, j] = tmp[var_score].values[0]

            # # fig, ax = plt.subplots()
            # # ax.imshow(X)

            # params = {
            #     "label_vars":self.Params["label_vars"],
            #     "version_distance":self.Params["version_distance"],
            #     "Clraw":None,
            # }
            # Cldist = Clusters(X, labels, labels, ver="dist", params=params)

            # return dfres, Cldist
        else:
            return dfres


    def rsa_distmat_score_all_pairs_of_label_groups(self, get_only_one_direction=True, label_vars=None,
            return_as_clustclass=False, return_as_clustclass_which_var_score="dist_yue_diff",
            labels_get=None, context_dict=None):
        """
        Get mean distance between all conjucntions of labels, returning dataframe where each row is distance between a levbel of
        labels_1 and labels_2, where labels_1 is tuple, conjunction of self.Labels

        Gets only a single direction (ie assumes symmetric score)

        Gets dist_yue_diff between all pairs of different groups, which is a normalized euclidean distance.

        PARAMS:
        - return_as_clustclass, bool, if True, then returns data as clustclass. Otherwise is a dataframe where
        each pair of groups gets a row.
        RETURNS:
        - df, with one column for each variable in self.Labels, separtely for labels 1 vs 2, appending "_1" to 
        name of column

        NOTE:
        - all rows with same (labels_1, labels_2) will have idnetical values for all distances, as distances are symetric.

        MS: checked
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

        if return_as_clustclass:
            get_only_one_direction = False

        assert self.Labels == self.LabelsCols, "assumes so for soem of below"

        # Get dataframe of labels for each trial
        dflabels = self.rsa_labels_return_as_df()

        # Get all label groupings and their trial indices.
        if label_vars is None:
            label_vars= self.rsa_labels_extract_label_vars()
        grpdict = grouping_append_and_return_inner_items_good(dflabels, label_vars)

        # Context?
        if context_dict is not None:
            diffctxt_vars_same=context_dict["same"]
            diffctxt_vars_diff=context_dict["diff"]
            # if (len(diffctxt_vars_same)>0 and diffctxt_vars_same is not None) or (len(diffctxt_vars_same)>0 and diffctxt_vars_same is not None):
            if "diff_context_ver" in context_dict:
                diff_context_ver = context_dict["diff_context_ver"]
            else:
                diff_context_ver = None
            # PLOT = True
            # path_for_save_print_lab_each_mask = f"/tmp/SYNTAX/mask.txt"
            # ma_context, fig, axes = self.rsa_mask_context_helper_simple(diffctxt_vars_same, diffctxt_vars_diff, 
            #                                                             diff_context_ver, only_upper_triangle=False,
            #                                                             PLOT=PLOT, 
            #                                                             path_for_save_print_lab_each_mask=path_for_save_print_lab_each_mask)
            # savefig(fig, f"/tmp/SYNTAX/masks.pdf")
            # plt.close("all")
            PLOT = False
            path_for_save_print_lab_each_mask = None
            ma_context, fig, axes = self.rsa_mask_context_helper_simple(diffctxt_vars_same, diffctxt_vars_diff, 
                                                                        diff_context_ver, only_upper_triangle=False,
                                                                        PLOT=PLOT, 
                                                                        path_for_save_print_lab_each_mask=path_for_save_print_lab_each_mask)
        else:
            ma_context = self._rsa_matindex_generate_all_true()

        ma_ut = self._rsa_matindex_generate_upper_triangular()

        # Get distance between each pair or grps
        res = []
        nskips = 0
        for i, (grp1, inds1) in enumerate(grpdict.items()):
            if labels_get is not None and grp1 not in labels_get:
                continue
            for j, (grp2, inds2) in enumerate(grpdict.items()):
                if labels_get is not None and grp2 not in labels_get:
                    continue
                if get_only_one_direction and i<j:
                    continue
                if i==j:
                    # always collect this, as this is used for normalizing, for yue diff
                    # Therefore, dont prune with ma_context
                    assert inds1==inds2
                    ma = self._rsa_matindex_convert_to_mask_rect(inds1, inds2)
                    # - exclude diagonal
                    X = self.Xinput[ma & ma_ut]
                else:
                    # Should include below lower and upper triangle, to make sure get all pairs.
                    ma = self._rsa_matindex_convert_to_mask_rect(inds1, inds2)
                    X = self.Xinput[ma & ma_context]
                
                if len(X)==0:
                    # print("Skipped (due to context): ", grp1, " -- ", grp2)
                    nskips += 1
                    continue

                if np.isnan(np.mean(X)):
                    print(X)
                    print(i, grp1, inds1)
                    print(j, grp2, inds2)
                    assert False, "fix bug"

                res.append({
                    "labels_1":grp1,
                    "labels_2":grp2,
                    "dist_mean":np.mean(X),
                    "n_1_2":(len(inds1), len(inds2)),
                })
                
                # Expand, to get each variable in label.
                if False: # done below
                    for _ivar, _var in enumerate(label_vars):
                        res[-1][f"{_var}_1"] = grp1[_ivar]
                        res[-1][f"{_var}_2"] = grp2[_ivar]

        # Also get 98th percentile between pairs of pts.
        ma = self._rsa_matindex_generate_upper_triangular()
        DIST_50, DIST_98 = np.percentile(self.Xinput[ma], [50, 98])

        dfres = pd.DataFrame(res)
        dfres["DIST_50"] = DIST_50
        dfres["DIST_98"] = DIST_98

        # normalize the distances
        dfres["dist_norm"] = dfres["dist_mean"]/dfres["DIST_98"]

        # Convert scores to dist_yue_diff
        cached_distances = {} # Otherwise can get very slow
        # var_scores = []
        def _get_within_group_dist(grp, var_score):
            """ return scalar avearge distance wtihin this group"""
            if grp not in cached_distances:
                tmp = dfres[(dfres["labels_1"]==grp) & (dfres["labels_2"]==grp)]
                assert len(tmp)==1
                cached_distances[grp] = tmp[var_score].values[0]
            return cached_distances[grp]
            
        var_score = "dist_norm"
        list_dist_yue_diff = []
        for i, row in dfres.iterrows():
            
            # - get average within-group score for the two groups
            dist_within_1 = _get_within_group_dist(row["labels_1"], var_score)
            dist_within_2 = _get_within_group_dist(row["labels_2"], var_score)
            dist_within = 0.5 * (dist_within_1 + dist_within_2)

            # - get the mean across-group score
            dist_across = row[var_score]

            # - finally, normalize
            dist_yue_diff = dist_across - dist_within

            # - collect
            list_dist_yue_diff.append(dist_yue_diff)
        dfres["dist_yue_diff"] = list_dist_yue_diff

        # Add column names reflecting the "sameness" state of variables.
        ### OTHER columns added
        dfres = self.rsa_distmat_population_columns_label_relations(dfres, label_vars)

        if dfres.isnull().values.any():
            # Example DataFrame
            # Find and print columns with NaNs
            print(dfres)
            columns_with_nans = dfres.columns[dfres.isnull().any()]
            print("Columns with NaN values:", columns_with_nans.tolist())
            assert False, "replace Nones using replace_values_with_this"

        ### Return as a ClustClass object
        if return_as_clustclass:
            
            assert get_only_one_direction==False

            labels = sort_mixed_type(set(dfres["labels_1"].tolist() + dfres["labels_2"].tolist()))
            # labels = sorted(set(self.Labels))
            map_label_to_idx = {lab:i for i, lab in enumerate(labels)}

            indsgotten = []
            X = np.zeros((len(labels), len(labels)))-np.inf
            for i, row in dfres.iterrows():
                indrow = map_label_to_idx[row["labels_1"]]
                indcol = map_label_to_idx[row["labels_2"]]
                X[indrow, indcol] = row[return_as_clustclass_which_var_score]

                assert (indrow, indcol) not in indsgotten
                indsgotten.append((indrow, indcol))
            
            if context_dict is not None:
                # Then you are allowed to have (label1, label2) pairs that are missed. This would be
                # pairs for which all trials fail the ocntext.
                # Use nan.
                X[X==-np.inf] = np.nan
            else:
                assert not np.any(X==-np.inf), "did not fill in all indices..."

            # Alternative, mucgh slower method
            # import numpy as np
            # var_score = "dist_yue_diff"
            # labels = sorted(set(self.Labels))
            # X = np.zeros((len(labels), len(labels)))
            # for i, labrow in enumerate(labels):
            #     print(i)
            #     for j, labcol in enumerate(labels):
            #         tmp = dfres[(dfres["labels_1"] == labrow) & (dfres["labels_2"] == labcol)]
            #         assert len(tmp)==1
            #         X[i, j] = tmp[var_score].values[0]

            # fig, ax = plt.subplots()
            # ax.imshow(X)

            params = {
                "label_vars":label_vars,
                "version_distance":self.Params["version_distance"],
                "Clraw":None,
            }
            assert len(label_vars)==len(labels[0])
            Cldist = Clusters(X, labels, labels, ver="dist", params=params)

            return dfres, Cldist
        else:
            return dfres

    def rsa_distmat_population_columns_label_relations(self, dfdists, label_vars):
        """
        Automatically populates new columns reflecting the relations between the columns in 
        label_vars (which can be any length), such as same_shape
        PARAMS:
        - dfdists, output from things like rsa_distmat_score_all_pairs_of_label_groups
        - label_vars, list of n strings
        e.g., label_vars = [shape, loc]
        RETURNS:
        - copy of dfdists
        """
        from neuralmonkey.analyses.euclidian_distance import dfdist_extract_label_vars_specific
        dfdists = dfdist_extract_label_vars_specific(dfdists, label_vars)

        # from pythonlib.tools.pandastools import append_col_with_grp_index

        # if "labels_1_datapt" in dfdists.columns:
        #     var1 = "labels_1_datapt"
        #     var2 = "labels_2_grp"
        # else:
        #     var1 = "labels_1"
        #     var2 = "labels_2"

        # dfdists = dfdists.copy()
        
        # # Replace columns which are now incorrect
        # # label_vars = ["seqc_0_shape", "seqc_0_loc"]
        # assert isinstance(label_vars[0], str)

        # # e..g, seqc_0_shape_1
        # for i, var in enumerate(label_vars):
        #     dfdists[f"{var}_1"] = [x[i] for x in dfdists[var1]]
        #     dfdists[f"{var}_2"] = [x[i] for x in dfdists[var2]]
        #     dfdists = append_col_with_grp_index(dfdists, [f"{var}_1", f"{var}_2"], f"{var}_12")
        #     dfdists[f"{var}_same"] = dfdists[f"{var}_1"] == dfdists[f"{var}_2"]

        # if len(label_vars)==2:
        #     for i, var in enumerate(label_vars):
        #         dfdists = append_col_with_grp_index(dfdists, [f"{label_vars[0]}_same", f"{label_vars[1]}_same"], f"same-{label_vars[0]}|{label_vars[1]}")

        return dfdists

    def rsa_distmat_convert_from_triangular_to_full(self, dfdists, label_vars=None, PLOT=False,
                                                    repopulate_relations=True):
        """
        Given a dfdists that is triangular (inclues diagonmal usually), convert to 
        full matrix by copying and swapping labels 1 and 2, assuming that
        distances are symmetric.
        
        RETURNS:
        - copy of dfdists, but more rows.
        """
        from neuralmonkey.analyses.euclidian_distance import dfdist_expand_convert_from_triangular_to_full

        dfdists = dfdist_expand_convert_from_triangular_to_full(dfdists, label_vars, PLOT, repopulate_relations)
        
        # from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

        # if PLOT:
        #     grouping_plot_n_samples_conjunction_heatmap(dfdists, "labels_1", "labels_2");

        # dftmp = dfdists.copy()

        # # Flip labels
        # dftmp["labels_1"] = dfdists["labels_2"]
        # dftmp["labels_2"] = dfdists["labels_1"]
        
        # # Remove diagonal
        # dftmp = dftmp[dftmp["labels_1"]!=dftmp["labels_2"]]
        
        # # concat
        # dfdists = pd.concat([dfdists, dftmp]).reset_index(drop=True)

        # if repopulate_relations:
        #     # Repopulation all label columns
        #     assert label_vars is not None        
        #     # label_vars = ["seqc_0_shape", var_other]
        #     # from pythonlib.cluster.clustclass import Clusters
        #     # cl = Clusters(None)
        #     dfdists = self.rsa_distmat_population_columns_label_relations(dfdists, label_vars)

        # if PLOT:
        #     grouping_plot_n_samples_conjunction_heatmap(dfdists, "labels_1", "labels_2");

        # # Sanity check that populated all cells in distance matrix
        # if False: # I know this code works, so no need for this.
        #     assert dfdists.groupby(["labels_2"]).size().min() == dfdists.groupby(["labels_2"]).size().max()

        return dfdists

    def rsa_distmat_score_same_diff_by_context(self, var_effect, var_others, context_input,
                                               dat_level, PLOT_MASKS=False, plot_mask_path=None, 
                                               dir_to_print_lab_each_mask=None,
                                               path_for_save_print_lab_each_mask=None):
        """
        Like rsa_distmat_score_same_diff, but here returns scores for each context
        [GOOD] Score mean distances for a given effect var, both within- and
        across- contexts, getting all combinations of effect x context, and splitting into
        scores separate for each context.
        PARAMS:
        - var_effect, str, the var you want to test the pairwise distances for.
        - var_others, list of all vars, used to help define context, where "same" means
        all vars same, and diff defined in contingent manner. NOTE: this MUST have the context
        variables inlcuded
        - context_input, either None (ingore) or dict with "same" and
        "diff" keys, each list of str (vars) for manualyl defining contxt.
            # context_input = {
            #     "same":["seqc_0_loc"],
            #     "diff":["event"]
            # }
        - dat_level, either "pts" or "distr", this is saved as a column in results, and determines
        whether to compute the "yue" methods,which only appliues for pts.
        :return:
        - res, dict of scores, eg
                var  var_others effect_samediff context_samediff     dat_level      dist
            0   shape  (gridloc,)            diff             diff           pts  0.478187
            1   shape  (gridloc,)            diff             diff       pts_yue  1.315285
            2   shape  (gridloc,)            diff             diff  pts_yue_diff  0.114427
            3   shape  (gridloc,)            diff             diff   pts_yue_log  0.394547
            4   shape  (gridloc,)            diff             same           pts  0.431110
            5   shape  (gridloc,)            diff             same       pts_yue  1.181588
            6   shape  (gridloc,)            diff             same  pts_yue_diff  0.067350
            7   shape  (gridloc,)            diff             same   pts_yue_log  0.229909
            8   shape  (gridloc,)            same             diff           pts  0.455072
            9   shape  (gridloc,)            same             diff       pts_yue  1.241416
            10  shape  (gridloc,)            same             diff  pts_yue_diff  0.091285
            11  shape  (gridloc,)            same             diff   pts_yue_log  0.297046
            12  shape  (gridloc,)            same             same           pts  0.363760
        """

        assert dat_level in ["pts", "distr"], f"must indicate which it is -- {dat_level}"

        if "same" in context_input:
            for var in context_input["same"]:
                assert var in var_others, "by definition, this must be strue"

        if "diff" in context_input:
            for var in context_input["diff"]:
                assert var in var_others, "by definition, this must be strue"

        ### Get masks of context
        if context_input is not None and len(context_input)>0:
            print("Generating masks using context:", context_input)
            if "diff_context_ver" in context_input:
                diff_context_ver = context_input["diff_context_ver"]
            else:
                diff_context_ver = "diff_specific_lenient"
            # Then use inputed context
            MASKS, fig, axes = self.rsa_mask_context_helper(var_effect, var_others, diff_context_ver,
                                  context_input["same"], context_input["diff"], PLOT=PLOT_MASKS,
                                  path_for_save_print_lab_each_mask=path_for_save_print_lab_each_mask)
        else:
            # Called "diff" if ANY var in var_others is different.
            MASKS, fig, axes = self.rsa_mask_context_helper(var_effect, var_others, "diff_at_least_one", PLOT=PLOT_MASKS,
                                                              path_for_save_print_lab_each_mask=path_for_save_print_lab_each_mask)
        if plot_mask_path is not None and PLOT_MASKS:
            print("Saving context mask at: ", plot_mask_path)
            savefig(fig, plot_mask_path)


        ##################### COMPUTE SCORES.
        # 1. Within each context, average pairwise distance between levels of effect var
        map_grp_to_mask_context_same = self.rsa_mask_context_split_levels_of_conj_var(var_others, PLOT=PLOT_MASKS, exclude_diagonal=False,
                                                                                        contrast="same")
        map_grp_to_mask_context = self.rsa_mask_context_split_levels_of_conj_var(var_others, PLOT=PLOT_MASKS, exclude_diagonal=False,
                                                                                            contrast="any")
        map_grp_to_mask_vareffect = self.rsa_mask_context_split_levels_of_conj_var([var_effect], PLOT=PLOT_MASKS, exclude_diagonal=False,
                                                                                    contrast="any") # either row or col must be the given level.
        ma_ut = self._rsa_matindex_generate_upper_triangular()


        res = []
        def _append_result_to_res(effect_samediff, context_samediff, leveff, levo, dist,
                                  dat_level):
            res.append({
                "var":var_effect,
                "var_others":tuple(var_others),
                "effect_samediff":effect_samediff,
                "context_samediff":context_samediff,
                "leveff":leveff,
                "levo":levo,
                "dist":dist,
                "dat_level":dat_level,
            })


        # --- ALIGN TO EACH LEVEL OF CONTEXT VAR.
        # Difference between levels of var, computed within(separately) for each level of ovar
        # (NOTE: this does nto care about "context")

        for grp, ma in map_grp_to_mask_context.items():
            if np.sum(ma)==0:
                print(grp)
                assert False, "bug in Cl code."
            
            dist_diff_same = None
            dist_same_same = None
            dist_diff_diff = None


            ma_final = ma & MASKS["effect_diff"] & MASKS["context_same"] & ma_ut
            if np.any(ma_final): # might not have if allow for cases with 1 level of effect var.
                dist_diff_same = self.Xinput[ma_final].mean()
                _append_result_to_res("diff", "same", "ALL", grp, dist_diff_same, dat_level)
            if dir_to_print_lab_each_mask is not None:
                path = f"{dir_to_print_lab_each_mask}/cond_on_ctxt-{grp}-DIFF|SAME.txt"
                self.rsa_matindex_print_mask_labels(ma_final, path)
            
            # Also collect "same" effect (and same context, as above)
            ma_final = ma & MASKS["effect_same"] & MASKS["context_same"] & ma_ut
            if np.any(ma_final): # might not have if allow for cases with 1 level of effect var.
                dist_same_same = self.Xinput[ma_final].mean()
                _append_result_to_res("same", "same", "ALL", grp, dist_same_same, dat_level)
            if dir_to_print_lab_each_mask is not None:
                path = f"{dir_to_print_lab_each_mask}/cond_on_ctxt-{grp}-SAME|SAME.txt"
                self.rsa_matindex_print_mask_labels(ma_final, path)

            # NOTE: decided this is best place to put (diff, diff) since goal is usualyl to 
            # control for context...
            ma_final = ma & MASKS["effect_diff"] & MASKS["context_diff"] & ma_ut
            if np.any(ma_final): # might not have if allow for cases with 1 level of effect var.
                dist_diff_diff = self.Xinput[ma_final].mean()
                _append_result_to_res("diff", "diff", "ALL", grp, dist_diff_diff, dat_level)
            if dir_to_print_lab_each_mask is not None:
                path = f"{dir_to_print_lab_each_mask}/cond_on_ctxt-{grp}-DIFF|DIFF.txt"
                self.rsa_matindex_print_mask_labels(ma_final, path)

            if dat_level=="pts":
                # Normalized effect
                if (dist_diff_same is not None) and (dist_same_same is not None):
                    for d, dl in [
                        [dist_diff_same/dist_same_same, "pts_yue"],
                        [np.log2(dist_diff_same/dist_same_same), "pts_yue_log"],
                        [dist_diff_same-dist_same_same, "pts_yue_diff"],
                        ]:
                        _append_result_to_res("diff", "same", "ALL", grp, d, dl)

                # (diff, diff) --> A bit arbitrary, could have n pts matching context levels (here) or
                # effect levels (below). Choose here since this is the main analysis.
                if (dist_diff_diff is not None) and (dist_same_same is not None):
                    for d, dl in [
                        [dist_diff_diff/dist_same_same, "pts_yue"],
                        [np.log2(dist_diff_diff/dist_same_same), "pts_yue_log"],
                        [dist_diff_diff-dist_same_same, "pts_yue_diff"],
                        ]:
                        _append_result_to_res("diff", "diff", "ALL", grp, d, dl)
                
                # 
                if (dist_diff_diff is not None) and (dist_diff_same is not None) and (dist_same_same is not None):

                    for d, dl in [
                        [(dist_diff_same-dist_same_same)/(dist_diff_diff-dist_same_same), "pts_yue_rescale"],
                        ]:
                        _append_result_to_res("diff", "same", "ALL", grp, d, dl)
                

        #### ACROSS CONTEXTS (compute separately for each level of effect)
        for lev_effect, ma in map_grp_to_mask_vareffect.items():
            # Also collect (same effect, diff context)
            # For each level of var, get its distance to that same level of var across
            # all contexts.

            dist_same_same = None
            dist_same_diff = None
            dist_diff_diff = None

            # - same effect, same context - just for normalizing.
            ma_final = ma & MASKS["effect_same"] & MASKS["context_same"] & ma_ut
            if np.sum(ma_final)>0:
                dist_same_same = self.Xinput[ma_final].mean()
            if dir_to_print_lab_each_mask is not None:
                path = f"{dir_to_print_lab_each_mask}/cond_on_eff-{lev_effect}-SAME|SAME.txt"
                self.rsa_matindex_print_mask_labels(ma_final, path)

            # - same effect, same context - just for normalizing.
            ma_final = ma & MASKS["effect_diff"] & MASKS["context_diff"] & ma_ut
            if np.sum(ma_final)>0:
                dist_diff_diff = self.Xinput[ma_final].mean()
            if dir_to_print_lab_each_mask is not None:
                path = f"{dir_to_print_lab_each_mask}/cond_on_eff-{lev_effect}-DIFF|DIFF.txt"
                self.rsa_matindex_print_mask_labels(ma_final, path)

            # - same effect diff context
            ma_final = ma & MASKS["effect_same"] & MASKS["context_diff"] & ma_ut
            if np.sum(ma_final)>0:
                dist_same_diff = self.Xinput[ma_final].mean()
                _append_result_to_res("same", "diff", lev_effect, "ALL", dist_same_diff, dat_level)
            if dir_to_print_lab_each_mask is not None:
                path = f"{dir_to_print_lab_each_mask}/cond_on_eff-{lev_effect}-SAME|DIFF.txt"
                self.rsa_matindex_print_mask_labels(ma_final, path)

            # Normalized effect
            if dat_level=="pts":
                if (dist_same_same is not None) and (dist_same_diff is not None):
                    for d, dl in [
                        [dist_same_diff/dist_same_same, "pts_yue"],
                        [np.log2(dist_same_diff/dist_same_same), "pts_yue_log"],
                        [dist_same_diff-dist_same_same, "pts_yue_diff"],
                        ]:
                        _append_result_to_res("same", "diff", lev_effect, "ALL", d, dl)
                        # res.append({
                        #     "var":var,
                        #     "var_others":tuple(var_others),
                        #     "effect_samediff":"same",
                        #     "context_samediff":"diff",
                        #     "levo":"ALL",
                        #     "leveff":lev_effect,
                        #     "dist":d,
                        #     "dat_level":dl,
                        # })

                if (dist_diff_diff is not None) and (dist_same_diff is not None) and (dist_same_same is not None):
                    for d, dl in [
                        [(dist_same_diff-dist_same_same)/(dist_diff_diff-dist_same_same), "pts_yue_rescale"],
                        ]:
                        _append_result_to_res("same", "diff", "ALL", grp, d, dl)

                # Normalized effect
                if False: # Instead get these aligned to context (above).
                    # Distance for (diff effect, diff context)
                    ma_final = ma & MASKS["effect_diff"] & MASKS["context_diff"] & ma_ut
                    if np.sum(ma_final)>0:
                        dist_diff_diff = self.Xinput[ma_final].mean()
                        res.append({
                            "var":var,
                            "var_others":tuple(var_others),
                            "effect_samediff":"diff",
                            "context_samediff":"diff",
                            "levo":"ALL",
                            "leveff":lev_effect,
                            "dist":dist_diff_diff,
                            "dat_level":dat_level,
                        })

                    if (dist_same_same is not None) and (dist_diff_diff is not None):
                        res.append({
                            "var":var,
                            "var_others":tuple(var_others),
                            "effect_samediff":"diff",
                            "context_samediff":"diff",
                            "levo":"ALL",
                            "leveff":lev_effect,
                            "dist":dist_diff_diff/dist_same_same,
                            "dat_level":"pts_yue",
                            })
        
        if False: 
            # This is another way to compute (diff, diff) -- but decided instead to 
            # get a single score per context (above).

            # For (diff diff) get a datapt for each combo of leveff and levo
            for levo, ma_o in map_grp_to_mask_context.items():
                for lev_effect, ma_eff in map_grp_to_mask_vareffect.items():
                    # - same effect, same context - just for normalizing.
                    ma_final = ma_o & ma_eff & MASKS["effect_diff"] & MASKS["context_diff"] & ma_ut
                    if np.sum(ma_final)>0:
                        dist_diff_diff = self.Xinput[ma_final].mean()
                        _append_result_to_res("diff", "diff", lev_effect, levo, dist_diff_diff, dat_level)

        # Also get 95th percentile of pairwise distances, as an upper bound to normalize data
        ma = self._rsa_matindex_generate_upper_triangular()
        dist_all = self.Xinput[ma].flatten()
        DIST_NULL_50 = np.percentile(dist_all, 50)
        DIST_NULL_95 = np.percentile(dist_all, 95)
        DIST_NULL_98 = np.percentile(dist_all, 98)

        return res, DIST_NULL_50, DIST_NULL_95, DIST_NULL_98



    def rsa_distmat_score_same_diff_OBSOLETE(self, var_effect, vars_all,
                                    vars_test_invariance_over_dict,
                                    PLOT=False):
        """
        [GOOD] Score mean distances for a given effect var, both within- and
        across- contexts.
        PARAMS:
        - var_effect, str, the var you want to test the pairwise distances for.
        - vars_all, list of all vars, used to help define context, where "same" means
        all vars same, and diff defined in contingent manner.
        - vars_test_invariance_over_dict, either None (ingore) or dict with "same" and
        "diff" keys, each list of str (vars) for manualyl defining contxt.
            # vars_test_invariance_over_dict = {
            #     "same":["seqc_0_loc"],
            #     "diff":["event"]
            # }
        :return:
        - rest, dict of 4 scores. Eg:
            {'EffS_CtxS': -0.029363783953610462,
             'EffD_CtxS': 1.3263261555005426,
             'EffS_CtxD': 1.184288421326354,
             'EffD_CtxD': 1.4957573090747027}
        """

        assert False, "use rsa_distmat_score_same_diff_by_context isntead -- it breaks down into each context. I confirmed that averaging over those contexts yeild same answer as here"

        # Decide how to define "diff context"
        if vars_test_invariance_over_dict is not None:
            # then you want to hand-code diff context with specific combo of vars
            diff_context_ver="diff_specific"
            diffctxt_vars_same=vars_test_invariance_over_dict["same"]
            diffctxt_vars_diff=vars_test_invariance_over_dict["diff"]
        else:
            # Diff context means at least one of the other vars is diff
            diff_context_ver="diff_at_least_one"
            diffctxt_vars_same=None
            diffctxt_vars_diff=None

        # Generate masks.
        vars_context = [var for var in vars_all if not var==var_effect]
        MASKS, _, _ = self.rsa_mask_context_helper(var_effect, vars_context, diff_context_ver,
                                      diffctxt_vars_same, diffctxt_vars_diff, PLOT=PLOT)


        ######### Score
        res = {}

        ma = MASKS["context_same"] & MASKS["effect_same"]
        res["EffS_CtxS"] = self.Xinput[ma].mean()

        ma = MASKS["context_same"] & MASKS["effect_diff"]
        res["EffD_CtxS"] = self.Xinput[ma].mean()

        ma = MASKS["context_diff"] & MASKS["effect_same"]
        res["EffS_CtxD"] = self.Xinput[ma].mean()

        ma = MASKS["context_diff"] & MASKS["effect_diff"]
        res["EffD_CtxD"] = self.Xinput[ma].mean()

        return res


    def rsa_distmat_score_vs_theor(self, Cltheor, vars_test_invariance_over_dict,
                                   corr_ver="pearson", plot_and_save_mask_path=None):
        """
        Helper to score self.Xinput, assumed to be distance matrix,
        against theorietical matrix held in Cltheor. ALways takees upper triangular.
        Optioanlly mask data first
        PARAMS
        - Cltheor, holds Xinput with same shape as self.Xinput
        - mask_vars_same, mask_vars_diff, eitehr None (skips) or list of str, to generate mask.
        - exclude_diag, usually True, unles you are doing positive control (dist vs self).
        - help_context, str, methods to help constrain (mask) to either to
        --- same context: cases where all otehrvars are same ("othervars_all_same"), or
        --- diff conbtext: at lesat one var is diff.
        RETURNS:
        - c, scalar, pearson corr coeff.
        :param var:
        :param Cltheor:
        :return:
        """

        # Inital extraction and sanity checks
        var_effect = Cltheor.Params["var"]
        assert isinstance(var_effect, str)
        assert self.Labels == Cltheor.Labels
        assert self.LabelsCols == Cltheor.LabelsCols

        # Decide how to define "diff context"
        if vars_test_invariance_over_dict is not None:
            # then you want to hand-code diff context with specific combo of vars
            diff_context_ver="diff_specific"
            diffctxt_vars_same=vars_test_invariance_over_dict["same"]
            diffctxt_vars_diff=vars_test_invariance_over_dict["diff"]
        else:
            # Diff context means at least one of the other vars is diff
            diff_context_ver="diff_at_least_one"
            diffctxt_vars_same=None
            diffctxt_vars_diff=None

        # Generate masks.
        vars_all = self.rsa_labels_extract_label_vars()
        vars_context = [var for var in vars_all if not var==var_effect]
        if plot_and_save_mask_path is not None:
            MASKS, fig, axes = self.rsa_mask_context_helper(var_effect, vars_context, diff_context_ver,
                                          diffctxt_vars_same, diffctxt_vars_diff, PLOT=True,
                                            path_for_save_print_lab_each_mask=f"{plot_and_save_mask_path}.txt")
            savefig(fig, plot_and_save_mask_path)
            plt.close("all")
        else:
            # print(var_effect)
            # print(vars_context)
            # print(diff_context_ver)
            # print(diffctxt_vars_same)
            # print(diffctxt_vars_diff)
            MASKS, _, _ = self.rsa_mask_context_helper(var_effect, vars_context, diff_context_ver,
                              diffctxt_vars_same, diffctxt_vars_diff, PLOT=False)

        ######## COMPUTE correaltions
        def _corr(vec1, vec2):
            if corr_ver=="pearson":
                c = np.corrcoef(vec1, vec2)[0,1]
            elif corr_ver=="kendall":
                import scipy.stats as stats
                tau, p_value = stats.kendalltau(vec1, vec2)
                c = tau
            else:
                print(corr_ver)
                assert False

            if np.isnan(c):
                print(Cltheor.Params["var"])
                print("Data: ", vec1, vec2)
                print(c)
                print(var_effect, vars_context, diff_context_ver, diffctxt_vars_same, diffctxt_vars_diff)
                print("Locations of nans:")
                print(np.argwhere(np.isnan(self.Xinput)))
                # Cltheor.rsa_plot_heatmap()
                # self.rsa_matindex_plot_masks_overlay_on_distmat(var_effect, mask_vars_same, mask_vars_diff,
                #                                                 exclude_diag=exclude_diag)
                assert False, "proibably becuase only one datapt for each level of var..."
            return c

        # Diff context
        ma = MASKS["context_diff"]
        vec_data = self.Xinput[ma].flatten()
        vec_theor = Cltheor.Xinput[ma].flatten()
        c_diff_context = _corr(vec_data, vec_theor)

        # same context
        ma = MASKS["context_same"]
        vec_data = self.Xinput[ma].flatten()
        vec_theor = Cltheor.Xinput[ma].flatten()
        c_same_context = _corr(vec_data, vec_theor)

        return c_diff_context, c_same_context

    def rsa_distmat_construct_theoretical(self, var, PLOT = False,
                                          dist_mat_manual=None, sort_order=None):
        """Construct theoretical dsitances matrices based on the varaibles
        for each row. Does not use the data in self.Xinput, just the labels
        cols and rows.
        PARAMS:
        - dist_mat_manual, to manually overwrite enter, overwriting need for
        computation here.
        RETURNS:
            - Cltheor.Labels will be identical to self.Labels. This way can sort
            labels as you would if this where the data matrix, to allow comparison
            with data matrix.
        """
        from pythonlib.tools.distfunctools import distmat_construct_wrapper

        # 1) pick out the feature dimension, and update labels
        # Recompute distance
        if dist_mat_manual is None:
            ind_var = self._rsa_map_varstr_to_varind(var) # convert from var to ind_var
            labels_row = [lab[ind_var] for lab in self.Labels]
            labels_col = [lab[ind_var] for lab in self.LabelsCols]
            _dist_func = self._rsa_distmat_return_distfunc(var)
            D = distmat_construct_wrapper(labels_row, labels_col, _dist_func)
        else:
            D = dist_mat_manual
            # sanity check
            assert D.shape[0]==D.shape[1]==len(self.Labels)

        # Make the labels into tuples, so that they match other rsa stuff
        # labels_row = [tuple([lab]) for lab in labels_row]
        # labels_col = [tuple([lab]) for lab in labels_col]
        # No: instead keep the labels as in self, to allow plotting self and
        # Cltheor using similar sort indices.
        # Cltheor = Clusters(D, self.Labels, self.LabelsCols, ver="dist",
        #                    params={"var":var, "version_distance":None, "label_vars":(var,)})
        Cltheor = Clusters(D, self.Labels, self.LabelsCols, ver="dist",
                           params={"var":var, "version_distance":None, "label_vars":(var,)})
        # plot
        if PLOT:
            fig = Cltheor.rsa_plot_heatmap(sort_order=sort_order)[0]
            # fig = Cltheor.plot_heatmap_data()[0]
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
        if self.Version == "rsa" or "label_vars" in self.Params:
            label_vars = self.Params["label_vars"]
        elif self.Version == "dist":
            label_vars = self.Params["Clraw"].Params["label_vars"]
        else:
            print("self.Params: ", self.Params)
            assert False, "input the params"
        return label_vars

    def rsa_matindex_print_mask_labels(self, ma, savepath,
                                       keep_only_unique_lines=True):
        """ Print the lab (row and col) for each True in this mask
        PARAMS:
        - keep_only_unique_lines, bool, if True, then sorts and keeps only one
        instance of each pair. otherwise retains order and gets all pairs.
        """

        # Given a mask, print all the pairs
        inds = self._rsa_matindex_convert_from_mask_to_rowcol(ma)
        strings =[]
        for i, indthis in enumerate(inds):
            row = indthis[0]
            col = indthis[1]
            labrow = self.Labels[row]
            labcol = self.LabelsCols[col]

            if keep_only_unique_lines:
                li = f"{labrow} -- {labcol}"
                if li not in strings:
                    strings.append(li)
            else:
                li = f"{i} -- ({row},{col}) -- {labrow} -- {labcol}"
                strings.append(li)

        if keep_only_unique_lines:
            strings = sorted(strings)
        
        strings = [" -- ".join(self.rsa_labels_extract_label_vars())] + strings

        if savepath is not None:
            from pythonlib.tools.expttools import writeStringsToFile
            # path = "/tmp/test.txt"
            writeStringsToFile(savepath, strings)

    # def rsa_matindex_print_bool_mask_pairs(self, ma, savepathfull):
    #     """_summary_

    #     Args:
    #         ma (_type_): _description_
    #         savepath (_type_): _description_
    #     """

    #     from pythonlib.tools.expttools import writeStringsToFile
    #     from pythonlib.tools.listtools import stringify_list

    #     lines = []

    #     for icol in range(ma.shape[0]):
    #         for irow in range(ma.shape[1]):
    #             if ma[irow, icol]:
    #                 li = tuple(sorted([self.Labels[irow], "  --  ", self.LabelsCols[icol]]))
    #                 if li not in lines:
    #                     lines.append(stringify_list(li))
    #     lines = sorted(lines)
    #     writeStringsToFile(lines, savepathfull)


    def rsa_matindex_plot_bool_mask(self, ma, ax=None):
        """ Plot this boolean mask
        """

        labels_row = self.Labels
        labels_col = self.LabelsCols
        self._plot_heatmap_data(ma, labels_row, labels_col, ax=ax)

        # if ax is None:
        #     fig, ax = plt.subplots()
        # ax.imshow(ma)

    def _rsa_matindex_convert_to_mask_specific(self, inds1, inds2):
        """ Given indices defined by paired values in
        lists rows and cols, return boolean array with
        these indices True, and the rest False
        """
        ma = np.zeros_like(self.Xinput, dtype=bool)
        ma[inds1, inds2] = True
        return ma

    def _rsa_matindex_generate_all_true(self, all_false=False):
        """ Return bool mask with all values True (or false,
        if all_false==True)
        """
        assert all_false == False, "code it"
        ma = np.ones_like(self.Xinput, dtype=bool)
        return ma

    def _rsa_matindex_generate_upper_triangular(self, exclude_diag=True):
        """ boolean mask with True in upper triangular, ecluding
        diagona, and elsewhere False.
        """
        if exclude_diag:
            k=1
        else:
            k=0
        return np.triu(np.ones_like(self.Xinput, dtype=bool), k=k)

    def _rsa_matindex_convert_from_mask_to_rowcol(self, ma):
        """ Return the indices using this mask, in paired list of rows, cols,
        RETURNS:
            - array, (npts, 2), each index into row, col.
        """
        return np.argwhere(ma)

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

    # _rsa_matindex_slice_specific_view
    def rsa_index_sliceX_specific_view(self, rows, cols):
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

    # _rsa_matindex_slice_rect_copy
    def rsa_index_sliceX_rect_copy(self, rows, cols, allow_copy=False):
        """ Helper to pull out rectangle slice of self.Xinput,
            - np array, (len(rows), len(cols)) shape.  This is a copy, which
            means modifications will NOT affect oriingal array
        """
        if allow_copy==False:
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

    def rsa_index_cols_with_this_level(self, var, level, level_is_list_of_levels=False):
        """ Return columsn with this level of this var
        """
        assert self.LabelsCols==self.Labels
        inds = self.rsa_index_rows_with_this_level(var, level, level_is_list_of_levels)
        return inds

    def rsa_index_rows_with_this_level(self, var, level, level_is_list_of_levels=False):
        """ return list of ints in to self.Labels (rows) whcih
        have this level for this var
        e.g, self.Labels[inds[0]]==level will be True
        """
        dflab = self.rsa_labels_return_as_df()
        if level_is_list_of_levels:
            # passed in levels that are acceptable
            return dflab[dflab[var].isin(level)]["row_index"].tolist()
        else:
            # passed in a single level.
            return dflab[dflab[var]==level]["row_index"].tolist()

    def rsa_matindex_plot_bounding_box(self, x1, x2, y1, y2, ax,
                                       edgecolor="r", facecolor="none"):
        """
        Overlay a bounding box (rectangle) that covers these coordinates,
        inclusize, starting from top-left corner.
        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :param ax:
        :return:
        """
        import matplotlib.patches as patches
        rect = patches.Rectangle((x1, y1), x2-x1+1, y2-y1+1, linewidth=1, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)

    def rsa_matindex_plot_masks_overlay_on_distmat(self, var_effect, context_vars_same,
                                                   context_vars_diff, exclude_diag=True):
        """
        These are the masks that would be used to compute things about sim mat.

        Plot mask
        :param var_effect: str, var that will color diff depending on if is "same" or "diff"
        :param context_vars_same: list of str, along with context_vars_diff, defines the context,
        ie the cells that will not black out.
        :param context_vars_diff:
        :return:
        """

        # Generate context mask.
        ma_context = self.rsa_matindex_same_diff_mult_var_flex(context_vars_same, context_vars_diff, PLOT=False)
        ma_ut = self._rsa_matindex_generate_upper_triangular(exclude_diag=exclude_diag)
        ma_context = ma_context & ma_ut

        # Effect mask (same, diff)
        ma_effect_same, ma_effect_diff = self.rsa_matindex_same_diff_this_var([var_effect])

        # Final conj masks.
        ma_same = ma_context & ma_effect_same
        ma_diff = ma_context & ma_effect_diff

        fig, axes = plt.subplots(1,3, figsize=(20,6))

        # make all have same axis
        zlims = [np.min(self.Xinput), np.max(self.Xinput)]

        ax = axes.flatten()[0]
        self.rsa_plot_heatmap(ax=ax, mask=ma_same, zlims=zlims)
        # self._rsa_matindex_plot_bool_mask(ma_same, ax)
        ax.set_title(f"Same, for effect var {var_effect}, context same {context_vars_same}, context diff {context_vars_diff}")

        ax = axes.flatten()[1]
        self.rsa_plot_heatmap(ax=ax, mask=ma_diff, zlims=zlims)
        ax.set_title("Effect var: Diff")

        ax = axes.flatten()[2]
        self.rsa_plot_heatmap(ax=ax, mask=ma_context, zlims=zlims)
        ax.set_title("Context ")

        return fig

    def rsa_matindex_same_diff_mult_var_flex(self, vars_same=None, vars_diff=None, PLOT=False,
                                             lenient_diff=False):
        """
        Flexibly generate bool mask that satistifes the criteria for
        each var in vars_same being same level, and each var in vars_diff
        being different level. All vars are checked independently (ie not conjunction).
        Any var thats not included, ignores whether its
        same or different

        :param vars_same:
        :param vars_diff:
        :param lenient_diff: bool, if True, then only need one var in vars_diff to be different for this
        to pass criterion. Otherwise all should be diff (i.e, checks conjunction of diff vars).
        :return:
        """

        if vars_same is None:
            vars_same = []
        if vars_diff is None:
            vars_diff = []

        assert isinstance(vars_same, (tuple, list))
        assert isinstance(vars_diff, (tuple, list))
        assert len([v for v in vars_same if v in vars_diff])==0

        # Can check "same" all at once - just get mask so that is
        # same across all vars
        if len(vars_same)==0:
            MA = self._rsa_matindex_generate_all_true()
        else:
            MA, _ = self.rsa_matindex_same_diff_this_var(vars_same)

        if lenient_diff==False:
            # For diff, must check each var one by one, and take negation
            for var in vars_diff:
                _, ma = self.rsa_matindex_same_diff_this_var([var])
                MA = MA & ma
        else:
            # All you need is for one of the diff vars to be diff to call a cell "diff".
            _, ma = self.rsa_matindex_same_diff_this_var(vars_diff)
            MA = MA & ma

        if PLOT:
            self.rsa_matindex_plot_bool_mask(MA)

        return MA

    def rsa_matindex_mask_if_any_var_is_diff(self, vars_at_least_one_must_be_diff):
        """ Returns mask such that at elast one of the var in vars_at_least_one_must_be_diff
        will be different (i,j).
        PPARAMS:
        - vars_at_least_one_must_be_diff, list of str.
        """
        _, ma_at_least_one_diff = self.rsa_matindex_same_diff_this_var(vars_at_least_one_must_be_diff)
        return ma_at_least_one_diff

    def rsa_matindex_same_diff_this_var(self, list_var):
        """ For similarity matrices,
        Return masks that are same (one output) or different (other
        output) for this variable. COnsiders list_var as a conjunctive
        variable. i.e,. A cell (i,j) is considered
        same only if the tuple (var1, var2, ...) [from list_var] is
        completely identical for the row and column (i and j).

        The "diff" mask is then the complement of the "same" mask
        """
        from pythonlib.tools.distfunctools import distmat_construct_wrapper

        assert self.LabelsCols==self.Labels, "assumes this"
        assert isinstance(list_var, (list, tuple))
        assert not isinstance(list_var[0], (list, tuple))

        df_lab = self.rsa_labels_return_as_df(include_row_index=False)

        try:
            levs = df_lab.loc[:, list_var].values.tolist() # list of list, each inner list being levels for [var1, var2, ...]
        except Exception as err:
            print(df_lab.columns)
            print(list_var)
            raise err
        levs = [tuple(x) for x in levs]
        if False:
            def boolean_2d_array(vec1, vec2):
                # Convert lists of tuples into NumPy arrays with dtype=object
                vec1 = np.array(vec1, dtype=object)
                vec2 = np.array(vec2, dtype=object)
                
                # Construct the boolean 2D array correctly
                return np.array([[bool(np.all(item1 == item2)) for item2 in vec2] for item1 in vec1], dtype=bool)

            print(levs[0])
            ma_same = boolean_2d_array(levs, levs)
            print(ma_same.shape)
            assert False
        else:
            ma_same = distmat_construct_wrapper(levs, levs, lambda x,y:x==y).astype(bool)
            # np.all(ma_same==ma_same2)

        ma_diff = ma_same==False

        return ma_same, ma_diff

    def rsa_matindex_same_diff_this_level(self, var, level):
        """ Return indices in distnace matrix corresponding
        to cases that are same (one output) and cases
        that are different level (another output), compared to
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

    def rsa_dataextract_with_labels_as_flattened_df(self, keep_only_one_direction_for_each_pair=False,
        plot_heat=True, exclude_diagonal=False):
        """ Return a df such that each row is a single distance between
        a specific row and column of distance matrix, along with
        variables that were in the x and y axis.

        Labels are added by appending either "_row" or "_col" depending
        on which it reflects in initial distmat.

        PARAMS:
            - keep_only_one_direction_for_each_pair, bool, if True, then only gets diagonal + upper triangle (includes diagonal).
            so that gets only one dist per pair (wihcih make sesnse since dists are symmatric).
        RETURNS:
            - dfdists, len nrows x ncols
        """
        import pandas as pd
        dflab = self.rsa_labels_return_as_df()
        var_labels = self.rsa_labels_extract_label_vars()

        # Go thru each cell and collect data
        dats = []
        for _, row1 in dflab.iterrows():
            for _, row2 in dflab.iterrows():
                idx1 = row1["row_index"]
                idx2 = row2["row_index"]

                # if idx2 >= idx1:

                _dat = {
                    "idx_row":idx1,
                    "idx_col":idx2,
                    "dist":self.Xinput[idx1, idx2]
                }

                # Extract variables
                for lab in var_labels:
                    _dat[f"{lab}_row"] = row1[lab]
                    _dat[f"{lab}_col"] = row2[lab]

                # Save
                dats.append(_dat)

        dfdists = pd.DataFrame(dats)

        if keep_only_one_direction_for_each_pair:
            dfdists = dfdists[dfdists["idx_col"] > dfdists["idx_row"]].reset_index(drop=True)

        if exclude_diagonal:
            dfdists = dfdists[dfdists["idx_col"] != dfdists["idx_row"]].reset_index(drop=True)

        if plot_heat:
            # sanity, this matches heatmap (but y axis flipped)
            import seaborn as sns
            sns.scatterplot(data=dfdists, x="idx_col", y="idx_row", hue="dist")

        if False:
            # convert to similarity
            dfdists["sim"] = np.log2(1/(dfdists["dist"]+0.01))
            len(dfdists)
            dfdists["dist_rect"] = dfdists["dist"]
            dfdists.loc[dfdists["dist_rect"]<0, "dist_rect"] = 0
            dfdists["sim"] = np.log2(1/(dfdists["dist"]+0.01))

        return dfdists

    def rsa_labels_return_as_df(self, include_row_index=True):
        """ Return df where rows are labels (rows in order as in
        self.Xinput) and columns are names of label vars, in order
        they are used in labels
        """
        import pandas as pd
        label_vars = self.rsa_labels_extract_label_vars()
        dflab = pd.DataFrame(self.Labels, columns=label_vars)
        if include_row_index:
            dflab["row_index"] = list(range(len(dflab)))
        return dflab

    def rsa_distmat_quantify_same_diff_variables_OBS(self, ind_var, ignore_diagonal=True):
        """
        For this variable, get distance across same and diff parirs, across
        all lewvels.
        E.g., can compute distance between same shape (across other var), vs. diff shape (across all var).
        PARAMS:
        - ind_var = 0 # e..g, if each row is labeled with a tuple like (shape, loc), then if
        ind_var==0, then this means "same" is defined as having same shape
        """

        assert False, "should use either rsa_distmat_score_same_diff_by_context or rsa_distmat_score_all_pairs_of_label_groups"
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
    
    def convert_copy_to_rsa_dist_version(self, label_var, version_distance):
        """
        Returns copy of self, formatted to be RSA/distmat version.
        Reequres square ditance mat.
        Only works with single (not conjunctive) lbel vary
        """

        if isinstance(label_var, (list, tuple)):
            # then self is already rsa mode...
            assert len(label_var)==len(self.Labels[0])
            labels_var_tuple = tuple(label_var)
            labels = self.Labels
        else:
            assert isinstance(label_var, str)
            labels_var_tuple = tuple([label_var])
            labels = [tuple([x]) for x in self.Labels]

        params = {
            "version_distance":version_distance,
            "label_vars":labels_var_tuple,
        }

        Cl = Clusters(X = self.Xinput, labels_rows=labels,
                        labels_cols=labels, ver="dist", params=params)
        return Cl
    
    def copy_with_slicing(self, rows=None, cols=None):
        """
        Copy self, optionally slicing to rectangle decinde by rows and cosl (which are indices)
        """
        if rows is None:
            rows = list(range(len(self.Labels)))
        if cols is None:
            cols = list(range(len(self.LabelsCols)))
        
        labels = [self.Labels[i] for i in rows]
        labels_cols = [self.LabelsCols[i] for i in cols]
        X = self.rsa_index_sliceX_rect_copy(rows, cols, allow_copy=True)
        Cl = Clusters(X = X, labels_rows=labels,
                        labels_cols=labels_cols, ver=self.Version, params=self.Params)
        return Cl

    def transpose(self):
        """
        Returns a copy, with rows and columns switched.
        """
        Cl = self.copy_with_slicing()

        Cl._Xinput = Cl._Xinput.T

        tmp = [lab for lab in self.LabelsCols]
        Cl.LabelsCols = [lab for lab in self.Labels]
        Cl.Labels = tmp

        assert self.Trialcodes is None, "decide what to do for transposing.."

        return Cl
    
    ############# CLASSIFY
    def scalar_score_convert_to_classification_accuracy(self, dfdist, var_datapt = "idx_row_datapt", 
                                                        var_score="dist_mean",
                                                        plot_savedir=None, higher_score_better=True):
        """
        Use distance matrix to discretize and get classifier score.
        i.e, for each row (var_datapt) ask which column is best score, and label that row by the label
        of that column.

        Can them compute classification accuracy which is for each row class, fraction of cases correctly clasifed.
        PARAMS:
        - dfdist, can get this from rsa_distmat_score_all_pairs_of_label_groups_datapts
        - higher_score_better, if True, then is "similarity", else is "distance."
        RETURNS:
        - score, score_adjusted, dfclasses, dfaccuracy

        NOTE:
        - this ported from decode_moment.py --> scalar_score_convert_to_classification_accuracy
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        from sklearn.metrics import balanced_accuracy_score

        if higher_score_better==False:
            dfdist = dfdist.copy()
            dfdist[var_score] = -dfdist[var_score]

        grpdict = grouping_append_and_return_inner_items_good(dfdist, [var_datapt])

        labels_test = [] # Correct label
        labels_predicted = [] # which decoder won?
        # Collect labels across each test datapt (i.e., row in dflab in pa)
        res = []
        for grp, inds in grpdict.items():

            dfthis = dfdist.iloc[inds]
            # dfthis = dfscores[dfscores["pa_idx"] == pa_idx]

            decoder_class_max = dfthis.iloc[np.argmax(dfthis[var_score])]["labels_2_grp"]
            labels_predicted.append(decoder_class_max)
            
            tmp = dfthis["labels_1_datapt"].unique().tolist()
            if not len(tmp)==1:
                print(tmp)
                assert False, "each datapt (pa_idx) is assumed to have its only duplication be due to time windows. "
            label_actual = tmp[0]
            labels_test.append(label_actual)

            # Collect, to get scores for diff slices of data
            res.append({
                "var_datapt":grp[0],
                "label_predicted":decoder_class_max,
                "label_actual":label_actual,
            })
        
        score = balanced_accuracy_score(labels_test, labels_predicted, adjusted=False)
        score_adjusted = balanced_accuracy_score(labels_test, labels_predicted, adjusted=True)

        dfclasses = pd.DataFrame(res)

        ### Also get scores for each pa label
        # score separately for each ground truth label
        grpdict = grouping_append_and_return_inner_items_good(dfclasses, ["label_actual"])

        accuracy_each = []
        for grp, inds in grpdict.items():
            label_actual = grp[0]
            labels_predicted = dfclasses.iloc[inds]["label_predicted"]

            n_correct = sum([lab==label_actual for lab in labels_predicted])
            n_tot = len(labels_predicted)
            accuracy = n_correct/n_tot

            accuracy_each.append({
                "label_actual":label_actual,
                "accuracy":accuracy
            })

        dfaccuracy = pd.DataFrame(accuracy_each)        
        annotate_heatmap = len(dfclasses["label_actual"].unique())<16
        if plot_savedir is not None:
            # Plot summaries of accuracy
            from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
            for norm_method in [None, "row_sub", "col_sub"]:
                fig = grouping_plot_n_samples_conjunction_heatmap(dfclasses, "label_actual", 
                                                                  "label_predicted", None,
                                                                  norm_method=norm_method, annotate_heatmap=annotate_heatmap)            
                savefig(fig, f"{plot_savedir}/accuracy_heatmap-norm={norm_method}.pdf")

        return score, score_adjusted, dfclasses, dfaccuracy

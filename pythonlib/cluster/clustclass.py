""" Represents (N, D) data, where N is num datpts, and D is dimensionality,
and does things related to clustering, dimesionality reduction, etc

Related to :
- neural.population
- tools.clustertools

First made when doing similarity matrix stuff for beh data (e.g, stroke similarity)
"""

import matplotlib.pyplot as plt
import numpy as np
from ..tools.nptools import sort_by_labels as sbl
from pythonlib.tools.plottools import savefig

class Clusters(object):
    """docstring for Clusters"""
    def __init__(self, X, labels_rows=None, labels_cols=None, params=None):
        """ 
        PARAMS;
        - X, (N, D) data, where N is num datpts, and D is dimensionality, 
        - labels_rows, list of labels for each row of X. if None, then labels_rows as 
        [0,1,2, ..]. These are like "categories"
        - labels_cols, list of labels. if pass in None, then:
        --- if ncols == nrows, uses row labels
        --- otherwise uses (0, 1, 2, 3...)
        - params, dict of params

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
    def find_dat_by_label(self, label_row, label_col):
        """ extract the similairty between these two labels
        """

        ind1 = self.find_ind_by_label(label_row, axis=0)
        ind2 = self.find_ind_by_label(label_col, axis=1)
        X = self.extract_dat()
        return X[ind1, ind2]

    def find_ind_by_label(self, label, axis=0):
        """
        """
        labs = self.extract_labels(axis=axis)
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
        SIZE=12, zlims=(None, None), nrand=None, rotation=45):
        """ Low-=level plotting of heatmap
        """

        if nrand is not None:
            import random
            inds = random.sample(range(X.shape[0]), nrand)
            X = X[inds, :]
            if labels_row:
                labels_row = [labels_row[i] for i in inds]

        # --- before sorting
        ndim = X.shape[0]
        ndat = X.shape[1]
        ASPECT = ndim/ndat
        fig, ax = plt.subplots(figsize=(SIZE, ASPECT*SIZE))
        
        h = ax.imshow(X, vmin=zlims[0], vmax=zlims[1], cmap='viridis')
        fig.colorbar(h, ax=ax)
            
        if labels_row:
            ax.set_yticks(range(len(labels_row)), labels_row)
        if labels_col:
            ax.set_xticks(range(len(labels_col)), labels_col, rotation=rotation);

        return fig, X, labels_col, labels_row

    def plot_heatmap_data(self, datkind="raw", labelkind="raw", 
        nrand = None, SIZE=12, zlims=(None, None)):
        """ Plot heatmap of raw data, usually sorted (rows) by labelkind.
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
        
        if "hier_clust_seaborn" not in self.ClusterResults.keys():
            cg = sns.clustermap(self.Xinput, row_cluster=True, col_cluster=True)
            self.ClusterResults["hier_clust_seaborn"] = cg

        # Plot
        cg = self.ClusterResults["hier_clust_seaborn"]

        # Label axes both col and row.
        self._hier_clust_label_axis(cg, "col")
        self._hier_clust_label_axis(cg, "row")

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
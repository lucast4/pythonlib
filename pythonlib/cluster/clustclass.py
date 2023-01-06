""" Represents (N, D) data, where N is num datpts, and D is dimensionality,
and does things related to clustering, dimesionality reduction, etc

Related to :
- neural.population
- tools.clustertools

First made when doing similarity matrix stuff for beh data (e.g, stroke similarity)
"""

import matplotlib.pyplot as plt
import numpy as np

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

    @property
    def Xinput(self):
        return self._Xinput
    

    ######################### sorting, filtering, slicing
    def sort_by_labels(self, X=None, labels=None, axis=0):
        """ 
        Sorts labels, then sorts X using the sortinds for labels
        PARAMS:
        - axis, dimeision to srot byt. if 0, then sorts rows...
        RETURNS:
        - Xsorted, labelsSorted
        """
        from ..tools.clustertools import sort_by_labels as sbl

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
        - nrand, int for taking random subset. None to skip
        RETURNS:
        - fig, X, labels
        """
        
        # Extract data
        X = self.extract_dat(datkind)
        labels_row = self.extract_labels(labelkind, axis=0)
        labels_col = self.extract_labels(labelkind, axis=1)

        # Take subset?
        if nrand is not None:
            import random
            inds = random.sample(X.shape[0], nrand)
            X = X[inds, :]
            labels_row = [labels_row[i] for i in inds]
            labels_col = [labels_col[i] for i in inds]
        
        # --- before sorting
        ndim = X.shape[0]
        ndat = X.shape[1]
        ASPECT = ndim/ndat
        fig, ax = plt.subplots(figsize=(SIZE, ASPECT*SIZE))
        
        h = ax.imshow(X, vmin=zlims[0], vmax=zlims[1], cmap='viridis')
        fig.colorbar(h, ax=ax)
                
        ax.set_yticks(range(len(labels_row)), labels_row);
        # ax.set_yticklabels(ylabels);

        ax.set_xticks(range(len(labels_col)), labels_col, rotation=45);

        return fig, X, labels_col, labels_row

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
        return cg





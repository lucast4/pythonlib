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
    def __init__(self, X, labels=None, params=None):
        """ 
        PARAMS;
        - X, (N, D) data, where N is num datpts, and D is dimensionality, 
        - labels, list of labels for each row of X. if None, then labels as 
        [0,1,2, ..]. These are like "categories"
        - params, dict of params

        """ 

        if params is None:
            params = {}

        self._Xinput = X # never changes
        self.Ndat = X.shape[0]
        self.Ndim = X.shape[1]
        if labels is None:
            labels = list(range(self.Ndat))
        else:
            assert len(labels)==self.Ndat
        self.Labels = labels
        self.Params = params

        # Track different versions of the data.
        self.LabelsDict = {
            "raw":self.Labels
            } # tracks different ways of labeling, always aligned to self.Xinput
        self.XDict = {
            "raw":self.Xinput}

    @property
    def Xinput(self):
        return self._Xinput
    

    ######################### sorting, filtering, slicing
    def sort_by_labels(self, X=None, labels=None, axis=0):
        """ 
        Sorts labels, then sorts X using the sortinds for labels
        RETURNS:
        - Xsorted, labelsSorted
        """
        from ..tools.clustertools import sort_by_labels as sbl
        if labels is None:
            labels = self.Labels
        if X is None:
            X = self.Xinput
        Xsorted, labelsSorted = sbl(X, labels, axis=axis)
        return Xsorted, labelsSorted

    def sort_by_labels_both_axes(self, X=None, labels=None):
        """ sorts both axis 0 and 1
        """
        Xsorted, _ = self.sort_by_labels(X, labels, axis=0)
        Xsorted, labelsSorted = self.sort_by_labels(Xsorted, labels, axis=1)
        return Xsorted, labelsSorted


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
    def extract_dat(self, datkind="raw"):
        """ Rerutrn data matrix, but possibly processed
        """
        return self.XDict[datkind]

    def extract_labels(self, labelkind="raw"):
        """ Return labels, various kinds
        """
        return self.LabelsDict[labelkind].tolist()

    ######################### PLOTS
    def plot_heatmap_data(self, datkind="raw", labelkind="raw", sortver=None, 
            nrand = None, SIZE=12):
        """ Plot heatmap of raw data, usually sorted (rows) by labelkind.
        PARAMS:
        - datkind, string name of data to plot. e..g, pca
        - labelkind, string name to get label
        - sortver, either
        --- None, no osrting
        --- {0,1}, sorts on thsoie axis
        --- 2, sorts both axes
        - nrand, int for taking random subset. None to skip
        RETURNS:
        - fig, X, labels
        """
        
        # Extract data
        X = self.extract_dat(datkind)
        labels = self.extract_labels(labelkind)
        
        # Take subset?
        if nrand is not None:
            import random
            inds = random.sample(X.shape[0], nrand)
            X = X[inds, :]
            labels = [labels[i] for i in labels]

        if sortver in [0, 1]:
            # sort X by labels
            X, labels = self.sort_by_labels(X, labels, axis=sortver)
        elif sortver == 2:
            # sort by both axes
            X, labels = self.sort_by_labels_both_axes(X, labels)
        else:
            assert sortver is None

        # --- before sorting
        ndim = X.shape[0]
        ndat = X.shape[1]
        ASPECT = ndim/ndat
        fig, ax = plt.subplots(figsize=(SIZE, ASPECT*SIZE))
        
        ax.imshow(X)
        ax.set_yticks(range(len(labels)));
        ax.set_yticklabels(labels);

        return fig, X, labels




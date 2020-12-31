import numpy as np


def plotNeurHeat(X, ax=None, barloc="right", robust=True):
    """ plot heatmap for data X.
    X must be (neuron, time)
    """
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
    
    # X = self.activations[pop][tasknum]
    minmax = (np.min(X), np.max(X))

    # plot
    # X = pd.DataFrame(X)            
    sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
               robust=robust)
    # ax.set_title(f"{pop}|{minmax[0]:.2f}...{minmax[1]:.2f}")
    ax.set_xlabel(f"robust={robust}|{minmax[0]:.2f}...{minmax[1]:.2f}")
        


class PopAnal():
    """ for analysis of population state spaces
    """

    def __init__(self, X, axislabels=None, dim_units=0):
        """ X is (nunits, cond1, cond2, ...)
        axislabels = list of strings
        - dim_units, dimeision holding units. if is not
        0 then will transpose to do all analyses.
        """
        self.X = X
        self.Saved = {}
        self.dim_units = dim_units
        self.preprocess()

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



    def centerAndStack(self):
        """ convert to (nunits, -1), 
        and center each row.
        """
        X = self.X.copy()
        # - reshape to N x tiembins
        X = np.reshape(X, (X.shape[0],-1))
        # - demean
        X = X - np.mean(X, axis=1)[:, None]
        
        self.Xcentered = X

    def pca(self, ver="eig", ploton=False):
        """ perform pca
        - saves in cache the axes"""
        
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


    def reproject(self, Ndim=3):
        """ reprojects neural pop onto subspace.
        uses axes defined by ver. check if saved if 
        not then autoamitcalyl extracst those axes
        - Ndim is num axes to take."""
        
        maxdim = self.X.shape[0] # max number of neurons
        if Ndim>maxdim:
            print(f"not enough actual neurons ({maxdim}) to match desired Ndim ({Ndim})")
            print(f"reducing Ndim to {maxdim}")
            Ndim = min((maxdim, Ndim))

        # - get old saved
        if "pca" not in self.Saved:
            print(f"- running {ver} for first time")
            self.pca(ver="eig")

        w = self.Saved["pca"]["w"]
        u = self.Saved["pca"]["u"]
            
        # - project data onto eigen
        usub = u[:,:Ndim]
        Xsub = usub.T @ self.Xcentered

        # - reshape back to (nunits, ..., ...)
        sh = list(self.X.shape)
        sh[0] = Ndim
        # print(self.X.shape)
        # print(Ndim)
        # print(Xsub.shape)
        # print(self.Xcentered.shape)
        # print(usub.T.shape)
        # print(u.shape)
        # print(u.)
        Xsub = np.reshape(Xsub, sh)
        # Ysub = Ysub.transpose((1, 0, 2))

        # -- return with units in the correct axis
        return self.unpreprocess(Xsub)


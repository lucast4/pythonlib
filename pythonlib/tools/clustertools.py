import numpy as np

def clusterSimMatrix(similarity_matrix, PCAdim = 5, gmm_n_mixtures = list(range(8, 30)), 
        perplist = [15, 25, 35, 45, 55, 65]):
    """
    Given precomputed sim matrix, do various kinds of clustering, and 
    return dict with results
    - Not plotting...
    """
    from sklearn.decomposition import PCA 
    from sklearn.manifold import TSNE
    from sklearn.model_selection import train_test_split
    from sklearn.mixture import GaussianMixture as GMM

    # 1) Use output of PCA for below modeling
    pca_model = PCA(n_components=PCAdim)
    Xpca = pca_model.fit_transform(similarity_matrix)

    # -- TSNE
    out = []
    for perp in perplist:
        D_tsne = TSNE(n_components=2, perplexity=perp).fit_transform(Xpca)
        out.append({
            "perp":perp,
            "D_fit":D_tsne
        })
    models_tsne = out


    # === PRELIM, fitting GMM to PCA-transformed data
    covariance_type="full"
    nsplits = 1
    n_init = 1
    out = []
    for isplit in range(nsplits):
        Xtrain, Xtest = train_test_split(Xpca, test_size=0.1)
        for n in gmm_n_mixtures:
            gmm = GMM(n_components=n, n_init=1, covariance_type=covariance_type)
            gmm.fit(Xtrain)
            # bic.append(gmm.bic(np.array(s)))
        #     gmm.bic(Xin)

            out.append({
                "mod":gmm,
                "n":n, 
                "isplit":isplit,
                "bic":gmm.bic(Xtest),
                "cross_val_score":gmm.score(Xtest)
            })
    models_gmm = out

    # OUTPUT
    DAT = {
        "pca_model":pca_model,
        "Xpca":Xpca,
        "models_tsne":models_tsne,
        "models_gmm":models_gmm,
        "similarity_matrix":similarity_matrix,
        "gmm_n_mixtures":gmm_n_mixtures,
        "perplist":perplist,
        "gmm_covariance_type":covariance_type
    }

    return DAT
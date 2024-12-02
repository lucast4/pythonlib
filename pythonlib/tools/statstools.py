## bunch of stuff for doings stats, focussing on using pandas

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

import numpy as np

def compute_d_prime(dist1, dist2):
    """
    Compute d-prime (d') between two univariate distributions.

    EXAMPLE:
    # Example distributions (samples)
    dist1 = [2.1, 2.5, 3.3, 2.8, 3.0]
    dist2 = [4.2, 4.8, 5.1, 4.5, 4.9]

    # Compute d-prime
    d_prime_value = compute_d_prime(dist1, dist2)
    print(f"d-prime: {d_prime_value:.4f}")
    """
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(dist1), np.mean(dist2)
    std1, std2 = np.std(dist1, ddof=1), np.std(dist2, ddof=1)  # ddof=1 for sample std

    # Calculate d-prime
    d_prime = abs(mean1 - mean2) / np.sqrt(0.5 * (std1**2 + std2**2))
    return d_prime


def signrank_wilcoxon(x1, x2=None):
    return stats.wilcoxon(x1, y=x2) 

def signrank_wilcoxon_from_df(df, datapt_vars, contrast_var, contrast_levels, value_var,
                              PLOT=False, save_text_path=None, assert_no_na_rows=False):
    """
    Helper to pull out paired values and compute sign rank p value.
    
    PARAMS:
    - df, long-form, where 
    - datapt_vars, list of str, each datap will be unique conjunction of these vars 
    (i.e., each row)
    - contrast_var, str, the var whose 2 levesl you want to contrast.
    - contrast_levels, list of 2 levels of contrast var, which you want to comapre.
    - value_var, str, the var holding the numerical values.
    - assert_no_na_rows, if True, then fails if any rows (datapt_vars) are thrown out, which 
    can occur if any column (contrast_levels) are missing.
    """
    from pythonlib.tools.pandastools import pivot_table 
    from pythonlib.tools.statstools import signrank_wilcoxon, plotmod_pvalues
    assert len(contrast_levels)==2

    for lev in contrast_levels:
        assert lev in df[contrast_var].tolist(), f"Misentered contrast_levels. Maybe typo? {contrast_var} -- {contrast_levels}"

    var1 = f"{value_var}-{contrast_levels[0]}"
    var2 = f"{value_var}-{contrast_levels[1]}"

    dfpivot = pivot_table(df, datapt_vars, [contrast_var], [value_var], flatten_col_names=True)

    dfpivot = dfpivot.loc[:, list(datapt_vars)+[var1, var2]] # keep just the vars you care about, before removing na rows, or you might remove supriosuly. 
    n1 = len(dfpivot)
    dfpivot = dfpivot.dropna()
    n2 = len(dfpivot)
    if assert_no_na_rows and n1!=n2:
        dfpivot = pivot_table(df, datapt_vars, [contrast_var], [value_var], flatten_col_names=True)
        print(dfpivot)
        dfpivot = dfpivot.dropna()
        print(dfpivot)
        assert False, "probably dont have all conjucntiosn... of (datapt_vars, contrast_var)"

    if len(dfpivot)==0:
        return None
    
    # display(dfpivot)
    res = signrank_wilcoxon(dfpivot[var1], dfpivot[var2])

    if PLOT:
        fig, ax = plt.subplots()
        for i, row in dfpivot.iterrows():
            ax.plot([0, 1], [row[var1], row[var2]], "-ok", alpha=0.5)

        # Overlay means
        ax.plot([-0.05, 1.05], [dfpivot[var1].mean(), dfpivot[var2].mean()], "sr", alpha=0.5)

        ax.set_ylim(0)
        ax.set_xlim([-0.5, 1.5])
        plotmod_pvalues(ax, [0.5], [res.pvalue])
        ax.set_xlabel(f"<value_var>-{contrast_var} = {var1} vs. {var2}", fontsize=6)
        ax.set_ylabel(value_var)
    else:
        fig = None
    
    out = {
        "res":res,
        "p":res.pvalue,
        "dfpivot":dfpivot,
        "var1":var1,
        "var2":var2
    }

    if save_text_path is not None:
        from pythonlib.tools.expttools import writeDictToTxtFlattened, writeDictToTxt
        writeDictToTxt(out, save_text_path)

    return out, fig

def ttest_paired(x1, x2=None, ignore_nan=False):
    """ x1, x2 arrays, with same lenght
    if x2 is None, then assumes is comparing to 0s.
    Automatically removes any rows with nans
    RETURNS:
    - ttest result object.
    """
    if x2 is None:
        x2 = np.zeros_like(x1)

    if ignore_nan:
        nan_policy='omit'
    else:
        nan_policy = 'propagate'
    return stats.ttest_rel(x1, x2, nan_policy=nan_policy)

def ttest_unpaired(x1, x2, equal_var=True, ignore_nan=False, permutations=None):
    """ x1, x2 arrays, with same lenght
    if x2 is None, then assumes is comparing to 0s.
    Automatically removes any rows with nans
    RETURNS:
    - ttest result object.
    """
    if ignore_nan:
        nan_policy='omit'
    else:
        nan_policy = 'propagate'
    return stats.ttest_ind(x1, x2, equal_var=equal_var, nan_policy=nan_policy, permutations=permutations)

def zscore(val, vals_dist):
    """ zscore val relative to the vals_dist distribtion"""
    return (val - np.mean(vals_dist)) / np.std(vals_dist)
    
# def statsTtestPaired(df1, df2, varname, pairedon="human"):
#     """ttest between two variables, assume it is paired at level of human (i.e, eachhuman 1pt each var pts)"""
#     df12 = pd.merge(df1, df2, on=pairedon)
# #     print(df12)
#     return stats.ttest_rel(df12[f"{varname}_x"], df12[f"{varname}_y"])

# def statsTtestUnpaired(df1, df2, varname):
#     return stats.ttest_ind(df1[f"{varname}"], df2[f"{varname}"])

# def statsRanksum(df1, df2, varname):
#     return stats.ranksums(df1[f"{varname}"], df2[f"{varname}"])

def curve_fit_and_test_wrapper_compare_logistic_linear(X, Y, X_cat, nfolds = 5, doplot=False, plot_sanity = False):
    """
    Compare fit using logistic vs. linear curve
    NOTE: is hard coded for speicifc scales of X and Y, and for speciific form of logistic.
    For psychometric experiments. 
    PARAMS:
    - X, vector 
    - Y, vector
    - X_cat, categorical version of X, which is used for stratified train-test split.
    """

    ### Define fyucntions and hard-code bounds.
    # TODO: Replace this -- so that sigmoid is not hard coded.
    def sigmoid(x, L, k, x0, b):
        return L * (1 / (1 + np.exp(-k*(x-x0)))) + b

    # bounds_sigmoid=([0., 0., -6., -5.], [7., 15., 6., 5]) # decode vs. index_within (rnak)
    # bounds_sigmoid=([0.95, 0., 0.25, 0], [1., 15., 0.75, 0.05]) # forcing to (0, 1) ylims, and steep slope around x=0.5 [not good, PMv got worse]
    bounds_sigmoid=([0.95, 0., -6., -5.], [7., 15., 6., 0.05]) # forcing to (0, 1) ylims, and steep slope around x=0.5

    def linear(x, a, b):
        return (a*x) + b

    bounds_linear = ([-2, -5], [2, 5])

    # For the optimaization, this is better type.
    X = X.astype(float)
    Y = Y.astype(float)

    # Get train-test folds
    folds = balanced_stratified_kfold(None, X_cat, nfolds)

    res = []
    for inds_train, inds_test in folds:
        x_train = X[inds_train]
        y_train = Y[inds_train]
        x_test = X[inds_test]
        y_test = Y[inds_test]

        assert x_test.shape == y_test.shape

        if plot_sanity:
            fig, axes = plt.subplots(2,2, figsize=(10, 10), sharex=True, sharey=True)
            ax = axes.flatten()[0]
            ax.set_title("all data")
            ax.plot(x, y, "ob", alpha=0.5)
            # ax.plot(x, sigmoid(x,  2, 1, 0, -0.9), 'go')

            ax = axes.flatten()[1]
            ax.set_title("train data")
            ax.plot(x_train, y_train, "ob", alpha=0.5)
            
            ax = axes.flatten()[2]
            ax.set_title("test data")
            ax.plot(x_test, y_test, "ob", alpha=0.5)
            
            assert False

        y_pred, R2, residuals_pred, residuals_mean = curve_fit_and_test(sigmoid, bounds_sigmoid, x_train, y_train, 
                                                                        x_test, y_test, doplot=doplot)
        if y_pred is None:
            return None, None

        res.append({
            "model":"sigmoid",
            "y_pred":y_pred,
            "y_test":y_test,
            "R2":R2,
            "residuals_pred":residuals_pred,
            "residuals_mean":residuals_mean,
        })

        y_pred, R2, residuals_pred, residuals_mean = curve_fit_and_test(linear, bounds_linear, x_train, y_train, 
                                                                        x_test, y_test, doplot=doplot)
        if y_pred is None:
            return None, None

        res.append({
            "model":"linear",
            "y_pred":y_pred,
            "y_test":y_test,
            "R2":R2,
            "residuals_pred":residuals_pred,
            "residuals_mean":residuals_mean,
        })

    dfres = pd.DataFrame(res)

    # Collect single result for each model
    res = []
    for model in ["linear", "sigmoid"]:
        # Collect all predicted values, and compute R2
        
        dfresthis = dfres[dfres["model"] == model]
        
        # (1) Good version, use all datapts
        Y_PRED = np.concatenate(dfresthis["y_pred"].tolist())
        Y_TEST = np.concatenate(dfresthis["y_test"].tolist())
        assert Y_PRED.shape == Y_TEST.shape == Y.shape
        R2, residuals_pred, residuals_mean = coeff_determination_R2(Y_TEST, Y_PRED, False)

        # (2) Less good, this has high variance.
        R2_mean = np.mean(dfresthis["R2"])
        R2_std = np.std(dfresthis["R2"])

        # Collect output
        res.append({
            "model":model,
            "R2":R2,
            "R2_mean":R2_mean,
            "R2_std":R2_std
        })
    dfres_summary = pd.DataFrame(res)

    return dfres_summary, dfres

def curve_fit_and_test(func, bounds, x_train, y_train, x_test, y_test, doplot=False, if_fail_converge="return_none"):
    """
    Fit function to data, and then evaluate on new test data, getting cross-validated R^2.
    PARAMS:
    - x_train, y_train, univariate data    
    """
    from pythonlib.tools.statstools import coeff_determination_R2
    from scipy.optimize import curve_fit

    ### Fit model
    try:
        popt, pcov, infodict, msg, _ = curve_fit(func, x_train, y_train, 
                                                bounds=bounds, full_output=True, maxfev=10000)
    except RuntimeError as err:
        if if_fail_converge=="return_none":
            return None, None, None, None
        else:
            raise err

    # Score test data:
    y_pred = func(x_test, *popt)
    R2, residuals_pred, residuals_mean = coeff_determination_R2(y_test, y_pred)
        
    if doplot:
        # Get R2 for training data too
        R2_train, _, _ = coeff_determination_R2(y_train, func(x_train, *popt))
        fig, axes = plt.subplots(2,2, figsize=(10, 10), sharex=True, sharey=True)

        ax = axes.flatten()[0]
        ax.set_title(f"train data and fit (R2={R2_train:.3f})")
        ax.plot(x_train, y_train, "ob", alpha=0.5)
        ax.plot(x_train, func(x_train, *popt), 'go')

        ax = axes.flatten()[1]
        ax.set_title(f"test data and fit (R2={R2:.3f})")
        ax.plot(x_test, y_test, "ob", alpha=0.5)
        ax.plot(x_test, func(x_test, *popt), 'go')

        print('---- Fittiung this function:', func)
        # print("Parameters: ", popt)
        print(f"R2_train={R2_train:.3f} -- R2_test={R2:.3f}")

    return y_pred, R2, residuals_pred, residuals_mean
    

def coeff_determination_R2(yvals, yvals_pred, doplot=False):
    """
    Compute R2 manually.
    """
    residuals_pred = yvals - yvals_pred
    ss_resid = np.sum(residuals_pred**2)

    residuals_mean = yvals - np.mean(yvals)
    ss_tot = np.sum(residuals_mean**2)

    R2 = 1 - ss_resid/ss_tot

    if doplot:
        fig, axes = plt.subplots(2,2, figsize=(10,10))

        ax = axes.flatten()[0]
        ax.plot(yvals, yvals_pred, "xk")
        ax.set_xlabel("yvals")
        ax.set_ylabel("yvals_pred")

        ax = axes.flatten()[1]
        ax.plot(yvals, residuals_pred, "xk")
        ax.set_xlabel("yvals")
        ax.set_ylabel("residuals_pred")

        ax = axes.flatten()[2]
        ax.plot(yvals, residuals_mean, "xk")
        ax.set_xlabel("yvals")
        ax.set_ylabel("residuals_mean")

    return R2, residuals_pred, residuals_mean   

    
def statsmodel_ols(x, y, PRINT=False, 
                   overlay_on_this_ax=None, overlay_x=0, overlay_y=-0.1, overlay_color=None, overlay_font_size=12):
    """
    PARAMS:
    - x, (ndat, nfeat), or (ndat,)
    - y, (ndat,)
    - overlay_on_this_ax, bool, then overlays text of regression summary on axis.
    """ 
    import statsmodels.api as sm

    # print(x.shape)
    # print(y.shape)

    # print(x.shape)
    x = sm.add_constant(x)
    # print(x.shape)
    # assert False

    model = sm.OLS(y,x, hasconst=True)
    results = model.fit()

    if PRINT:
        print(results.summary())

    if len(results.params)!=2:
        print(results.summary())
        print(y.shape)
        print(x.shape)
        assert False, "why"

    if overlay_on_this_ax is not None:
        overlay_on_this_ax.text(overlay_x, overlay_y, f"r2={results.rsquared:.2f}|p={results.pvalues[1]:.3f}|int={results.params[0]:.3f}|slope={results.params[1]:.3f}", color=overlay_color, fontsize=overlay_font_size)

    return results  

if False:
    def getStatsOLS(dfthis_teststimonly, colname):
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        res_dict = {}
        
        X = dfthis_teststimonly[colname].values.reshape(-1,1)
        y = dfthis_teststimonly["x4"].values.reshape(-1,1)
        X = add_constant(X, prepend=False)

        res = OLS(y, X).fit()
        res_dict["DC_vs_planner"]=res

        X = dfthis_teststimonly[colname].values.reshape(-1,1)
        y = dfthis_teststimonly["human_cond"].values.reshape(-1,1)
        X = add_constant(X, prepend=False)

        res = OLS(y, X).fit()
        res_dict["DC_vs_humancond"]=res
        
        return res_dict

def empiricalPval(stat_actual, stats_shuff, side="two"):
    """ p value, useful for permutation tests.
    - side, if "two" then two sided (uses absolute value), 
    if "left" then hypothesize that actual is small, if 'right' then large.
    """     

    # How many shuffled cases are stronger in the "effect directio" compared to actual data?
    if side=="two":
        # n = sum(np.abs(stats_shuff)<=np.abs(stat_actual)) + 1 # THIS WAS BUG
        n = sum(np.abs(stats_shuff)>np.abs(stat_actual)) + 1
    elif side=="left":
        # n = sum(stats_shuff<=stat_actual) + 1
        n = sum(stats_shuff<=stat_actual) + 1
    elif side=="right":
        n = sum(stats_shuff>=stat_actual) + 1
    else:
        assert False, "not coded"
    
    nn = len(stats_shuff) + 1
    p = n/nn

    return p



def permutationTest(data, funstat, funshuff, N, plot=True, side="two"):
    """ generic permutation test function
    - funstat(data) returns a scalar. if funstat returns a tuple (a,b), then 
    treats a as the scalar for statistics, and treats b as values that will collect
    and output after doing shuffle, as [a1, a2, ...]
    - funshuff(data) return a copy of data, shuffled
    - N is how many permutations
    - side, if "two" then two sided (uses absolute value), 
    if "left" then hypothesize that actual is small, if 'right' then large.
    """

    X = funstat(data)
    if isinstance(X, tuple) or isinstance(X, list):
        assert len(X)==2
        # then outputing list/tuple
        collect_shuffstats=True
    else:
        collect_shuffstats=False

    if collect_shuffstats:
        stat_actual, stat_actual_collected = funstat(data)
    else:
        stat_actual = funstat(data)

    stats_shuff = []
    stats_shuff_collected = []
    for i in range(N):
        if i%50==0:
            print(f"shuffle # {i}")
        data_shuff = funshuff(data)
        if collect_shuffstats:
            X = funstat(data_shuff)
            stats_shuff.append(X[0])
            stats_shuff_collected.append(X[1])
        else:
            stats_shuff.append(funstat(data_shuff))
    
    # p value
    p = empiricalPval(stat_actual, stats_shuff, side=side)

    # plot 
    if plot:
        fig = plt.figure()
        plt.hist(stats_shuff)
        plt.axvline(x=stat_actual, color="r")
        plt.title(f"p={p:.3f} | {1-p:.3f}")
    else: 
        fig = None
    
    if collect_shuffstats:
        return p, stat_actual_collected, stats_shuff_collected, fig
    else:
        return p, fig

    if False:
        # fake dataset/experiement to sanity check permutation test and see how it works
        def funstat(data):
            return np.corrcoef(data[:,0], data[:,1])[0,1]
        #     return np.mean(data[:,1]) - np.mean(data[:,0])

        def funshuff(data):
            import copy
            data_shuff = copy.deepcopy(data)
            np.random.shuffle(data_shuff[:,0])
            return data_shuff

        pvals = []
        for _ in range(500):
            data = np.random.rand(50,2)
            pvals.append(permutationTest(data, funstat, funshuff, 100, plot=False))
            
        plt.figure()
        plt.hist(pvals)
        plt.title("this should be uniformly distributed")
        print("frac cases of p<0.05 should be around 0.05")
        print(sum(np.array(pvals)<0.05)/500)


def plotmod_pvalues(ax, xvals, pvals, pthresh=0.05, y_loc_frac=0.75, prefix=None, fontsize=8):
    """ Plot values on top of plots where each x locaiton is a 
    distribution with its own p value
    """

    assert len(xvals)==len(pvals)

    YLIM = ax.get_ylim()
    y = YLIM[0] + y_loc_frac*YLIM[1]-YLIM[0]
    # xrang = XLIM[1] - XLIM[0]



    for x, p in zip(xvals, pvals):
        if p<pthresh:
            col = "r"   
        else:
            col = "k"
        if prefix is None:
            ax.text(x, y, f"{p:.2E}", color=col, rotation=45, alpha=0.5, fontsize=fontsize)
        else:
            ax.text(x, y, f"{prefix} : {p:.2E}", color=col, rotation=45, alpha=0.5, fontsize=fontsize)

def crossval_folds_indices(n1, n2, nfold=None):
    """ Return the indices for partitioning
    two datasets with size n1 and n2, to obtain
    crossvalkidated data of nfolds.

    Each index used once exactly.
    
    Written for unbiased euclidian distnace.
    PARAMS:
    - n1, n2, int, size of two dataset
    - nfold, int, how many folds. None to use min of n1 n2
    RETURNS;
    - list_inds_1,  list of np ararys of ints, each indices into dataset 1
    - list_inds_2, same thing, for dataset 2.
    (Both lists will be len nfold).
    """
    inds_on_1, inds_on_2 = _crossval_folds_indices_onsets(n1, n2, nfold)

    def _convert_indson_to_inds(inds_on, n):
        list_inds = []
        for i, on in enumerate(inds_on):
            if i+1==len(inds_on):
                # last onset, it ends at the end of the data
                list_inds.append(np.arange(on, n))
            else:
                list_inds.append(np.arange(on, inds_on[i+1]))
        return list_inds

    list_inds_1 = _convert_indson_to_inds(inds_on_1, n1)
    list_inds_2 = _convert_indson_to_inds(inds_on_2, n2)

    assert len(list_inds_1)==len(list_inds_2)

    assert sorted(set(np.concatenate(list_inds_1))) == sorted(np.concatenate(list_inds_1))
    assert sorted(set(np.concatenate(list_inds_2))) == sorted(np.concatenate(list_inds_2))
    return list_inds_1, list_inds_2



def _crossval_folds_indices_onsets(n1, n2, nfold=None):
    """ Return the indices for subsampling
    two datasets with size n1 and n2, to obtain
    crossvalkidated data of nfolds.
    Written for unbiased euclidian distnace.
    PARAMS:
    - n1, n2, int, size of two dataset
    - nfold, int, how many folds. None to use min of n1 n2
    RETURNS;
    - inds_on_1, np arary ints, indices into dataset
    1 which are the onset boundaries of partitions.
    - inds_on_2, same thing, for dataset 2.
    """

    if nfold is None:
        nfold = min([n1, n2])

    if True:
        assert (nfold<=n1) and (nfold<=n2)
    else:
        if nfold>n1:
            nfold = n1
        if nfold>n2:
            nfold = n2

    if n1==n2==nfold:
        # sample each pt one by one.
        return np.arange(n1), np.arange(n2)
    elif n1==n2:
        # partition both datasets the same way
        inds_on = np.floor(np.linspace(0, n1+1, nfold+1)).astype(int)[:-1]
        return inds_on, inds_on
    else:
        # split into subsets
        n_larger = max([n1, n2])
        n_smaller = min([n1, n2])

        inds_on_larger = np.floor(np.linspace(0, n_larger+1, nfold+1)).astype(int)[:-1]
        inds_on_smaller = np.floor(np.linspace(0, n_smaller+1, nfold+1)).astype(int)[:-1]

        if n1>n2:
            return inds_on_larger, inds_on_smaller
        elif n1<n2:
            return inds_on_smaller, inds_on_larger
        else:
            assert False


def stratified_resample_split_kfold(y, n_splits, test_size=0.5, PRINT=False):
    """
    Split data (into two partitions that use up all data) in stratified manner, ie ensuring taking 
    same proportiuon of each class of labels (not balanced). 
    Useful for splitting data similar to train-test.
    Returns:
    - list of 2-tuples, each holding random train and test inds (arrays) covering all data, that index into y. 
    len of the list is n_splits.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    X = [0 for _ in range(len(y))] # not needed
    split_inds = list(sss.split(X, y))
    
    if PRINT:
        for i, (train_index, test_index) in enumerate(split_inds):
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

    return split_inds

def balanced_stratified_resample_kfold(X, y, n_splits):
    """
    Resample <n_splits> times, each time returning set of indices that are balnaced (semae n for each level of 
    class label y), and maximizing how mnay you inds you cans ample (which depends on the class with min n 
    cases).

    USeful for resampling X and y such that the calss label y is balnced.

    Is like balanced_stratified_kfold but not splitign in to train/test

    RETURNS:
    - inds_folds, list of list, each inner list are inds to sample y (e.g., y[inds]), such that
    len(inds) = num_classes * n_min_trials_across_classes.
    """
    from pythonlib.tools.pandastools import extract_resample_balance_by_var
    import pandas as pd

    df = pd.DataFrame({"index":range(len(y)), "y":y})
    nmin = df.groupby("y").size().min()
    nclass = len(df["y"].unique())

    inds_folds = []
    for k in range(n_splits):
        dfbalanced = extract_resample_balance_by_var(df, "y", n_samples=nmin, method_if_not_enough_samples="min")
        inds = sorted(dfbalanced["index"].tolist())
        assert len(inds) == nmin * nclass, "not balnaced..."
        assert len(inds) == len(set(inds)), "not unique inds.."
        inds_folds.append(inds)

    return inds_folds

def stratified_kfold_single_test(X, y, n_splits, shuffle=False, random_state=None,
                                 PRINT=False):
    """
    Stratified K-Fold cross-validator where each sample appears only once in the test set across folds.
    Handles cases where classes have fewer samples than n_splits.

    Train indices will be balanced across folds, wheras test indices will reflect the frequency in the
    data.

    NOTES:
    - Places more test inds in the first folds.
    
    Parameters:
    - X: Feature array.
    - y: Target array.
    - n_splits: Number of folds.
    - shuffle: Whether to shuffle the data before splitting.
    - random_state: Random state for reproducibility.
    
    Returns:
    - List of (train_indices, test_indices) for each fold.

    # Example usage
    X = np.random.rand(10, 5)  # 10 samples, 5 features
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4])  # Some classes have < n_splits samples

    n_splits = 3
    shuffle = True
    random_state = None

    folds = stratified_kfold_single_test(X, y, n_splits, shuffle, random_state)
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"Fold {i+1}:")
        print(f"  Train indices: {train_idx}")
        print(f"  Test indices: {test_idx}")

    --->
        Fold 1:
        Train indices: [0 2 4 5 7 9]
        Test indices: [ 1  3  6  8 10]
        Fold 2:
        Train indices: [ 1  2  3  5  6  8 10]
        Test indices: [0 4 7 9]
        Fold 3:
        Train indices: [ 0  1  3  4  6  7  8  9 10]
        Test indices: [2 5]
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    y = np.array(y)
    unique_classes = np.unique(y)

    # Initialize the folds
    folds = [[] for _ in range(n_splits)]

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(cls_indices)
        
        # Assign samples to folds in a round-robin fashion
        for i, idx in enumerate(cls_indices):
            fold_idx = i % n_splits
            folds[fold_idx].append(idx)

    # Build train/test indices for each fold
    all_indices = set(range(len(y)))
    fold_indices = []
    for k in range(n_splits):
        test_indices = np.array(folds[k])
        train_indices = np.array(sorted(all_indices - set(test_indices)))
        fold_indices.append((train_indices, test_indices))

    ### Sanity check
    # print(folds)
    # assert False
    train_inds = []
    test_inds = []
    for f in fold_indices:
        if len(f[1])==0:
            for f in fold_indices:
                print(f)
            assert False, "you want too many splits. it should be less than the max group size"
        train_inds.extend(f[0])
        test_inds.extend(f[1])

    if PRINT:
        for i, (train_idx, test_idx) in enumerate(fold_indices):
            print(f"Fold {i+1}:")
            print(f"  Train indices: {train_idx}")
            print(f"  Test indices: {test_idx}")

        # train_inds = []
        # test_inds = []
        # for f in folds:
        #     train_inds.extend(f[0])
        #     test_inds.extend(f[1])
        print(sorted(train_inds))
        print(sorted(test_inds))        

    return fold_indices

def balanced_stratified_kfold(X, y, n_splits=5, do_print=False, do_balancing_of_train_inds=True, shuffle=False):
    """ 
    Returns k-fold indices, startified (with label y), and then
    ensuring that the number of train indices per class are same within each fold (i.e.
    balancing the data for trainig), with test idnices being stratified, matching dataset distribution,
    i.e., not balanced.

    This means that all indices will take one and only one turn as test, but possibly multiple or zero turns as trainig.
    with higher n_splits leading to more turns in training on average

    This is useful to put diff classes on same "footing" by matching num training samples.
    
    PARAMS:
    - n_splits
    --- int, this many folds.
    --- auto, takes n that maximizes n fold s(i.e. like LOO)
    - do_balancing_of_train_inds, bool, if False, then doesnt balance the train idnices. This is useful if you
    want to maximize n samples for training (e.g., the trained model is not biased by n samples).
    
    RETURNS:
    - balanced_folds, list of 2-tuples of indices, which are arrays of ints.
    Example:
        [(array([ 5,  7, 13, 12,  6, 16, 15,  9, 17]), array([0, 1, 2, 3, 4])),
        (array([16,  0,  4, 17, 15,  1, 13, 18,  2]), array([ 5,  6,  7,  9, 12])),
        (array([12,  0, 16,  1, 17,  9, 13,  2,  7]), array([ 8, 10, 11, 14, 15])),
        (array([ 6, 14, 12,  1,  9, 15,  3,  2,  7]), array([13, 16, 17, 18, 19]))]

    EXAMPLE
    Example usage
    from sklearn.datasets import make_classification

    # Create a sample dataset
    X, y = make_classification(n_samples=30, n_classes=3, weights=[0.2, 0.3, 0.5], n_informative=3, random_state=42)

    balanced_folds = balanced_stratified_kfold(X, y, n_splits="auto", do_print=True)


    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter
    
    if X is None:
        X = np.zeros_like(y)

    if n_splits == "auto":
        # Then do max n splits
        n_splits = min(Counter(y).values())

        # Cap it
        max_n_splits = 15
        n_splits = min([n_splits, max_n_splits]) 


        print(f"[Auto] Doing {n_splits} splits")
    else:
        # dont take more splits than possible.
        n_splits = min([n_splits, min(Counter(y).values())])

    if n_splits==1:
        # Or else will fail
        n_splits = 2
        
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        folds = list(skf.split(X, y)) # folds[0], 2-tuple of arays of ints
    except Exception as err:
        # This deals with cases where a group has fewer items than n_splits.
        folds = stratified_kfold_single_test(X, y, n_splits, shuffle=shuffle)

    if do_balancing_of_train_inds:
        # Resample inds after shuffling for each class.
        balanced_folds = []
        for train_idx, test_idx in folds:
            # y_train = y[train_idx]
            y_train = [y[i] for i in train_idx]
            class_counts = Counter(y_train)
            min_class_count = min(class_counts.values())
            
            # Create a balanced train set. For each class, subsample y_train so that is balanced across calsses.
            new_train_idx = []
            class_indices = {cls: [i for i, ythis in enumerate(y_train) if ythis==cls] for cls in class_counts}

            # For each class, shuffle its current inds, then take the first n to subsample.
            for cls in class_counts:
                np.random.shuffle(class_indices[cls])
                cls_indices = class_indices[cls][:min_class_count]
                new_train_idx.extend(train_idx[cls_indices])
            
            # print("HER", type(train_idx), train_idx)
            # print(new_train_idx)
            # print("test:", test_idx)
            balanced_folds.append((np.array(new_train_idx), test_idx))
    else:
        balanced_folds = folds
    
    if do_print:
        for i, (train_idx, test_idx) in enumerate(balanced_folds):
            print(f"---- Fold {i+1}")
            print("Train class distribution:", Counter([y[i] for i in train_idx]))
            print("Test class distribution:", Counter([y[i] for i in test_idx]))
            print("train_idx:", train_idx)
            print("test_idx:", test_idx)

    # Sanity check that each datapt contributes once and only onces to testing, after combining across folds.
    test_idx_all = np.concatenate([x[1] for x in balanced_folds])
    assert len(test_idx_all) == len(y)
    assert len(test_idx_all) == len(set(test_idx_all))

    return balanced_folds

def cluster_kmeans_with_silhouette_score(X, n_clusters=None, n_clusters_min_max=None, PLOT_SIL=False,
                                         PLOT_FINAL=False, return_figs=False):
    """
    K-means clustering, with optinaly finding n clusters by maximizing silhoutte score
    over range of possible n_clsuters.

    Code taken from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    :param X: (nsamp, nfeats)
    :param n_clusters:
    :param n_clusters_min_max: only needed if n_clusters is None, in which case shoudl be 2-list of int,
    will try all n clusters in range of [min, max], and use the one that maximizes the silhoutte score.
    :param PLOT:
    :return: cluster_labels, list of strings represntations of ints
    """
    from sklearn.cluster import KMeans

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if n_clusters is None:
            assert isinstance(n_clusters_min_max, (list, tuple)) and len(n_clusters_min_max)==2
            range_n_clusters = list(range(n_clusters_min_max[0], n_clusters_min_max[1]+1))

            import matplotlib.cm as cm
            from sklearn.metrics import silhouette_samples, silhouette_score

            list_silhouette_avg = []
            list_n_clusters = []
            for n_clusters in range_n_clusters:

                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init="auto")
                cluster_labels = clusterer.fit_predict(X)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(X, cluster_labels)
                print(
                    "For n_clusters =",
                    n_clusters,
                    "The average silhouette_score is :",
                    silhouette_avg,
                )
                list_silhouette_avg.append(silhouette_avg)
                list_n_clusters.append(n_clusters)

                if PLOT_SIL:
                    # Compute the silhouette scores for each sample
                    sample_silhouette_values = silhouette_samples(X, cluster_labels)

                    # Create a subplot with 1 row and 2 columns
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.set_size_inches(18, 7)

                    # The 1st subplot is the silhouette plot
                    # The silhouette coefficient can range from -1, 1 but in this example all
                    # lie within [-0.1, 1]
                    ax1.set_xlim([-0.1, 1])
                    # The (n_clusters+1)*10 is for inserting blank space between silhouette
                    # plots of individual clusters, to demarcate them clearly.
                    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                    y_lower = 10
                    for i in range(n_clusters):
                        # Aggregate the silhouette scores for samples belonging to
                        # cluster i, and sort them
                        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                        ith_cluster_silhouette_values.sort()

                        size_cluster_i = ith_cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i

                        color = cm.nipy_spectral(float(i) / n_clusters)
                        ax1.fill_betweenx(
                            np.arange(y_lower, y_upper),
                            0,
                            ith_cluster_silhouette_values,
                            facecolor=color,
                            edgecolor=color,
                            alpha=0.7,
                        )

                        # Label the silhouette plots with their cluster numbers at the middle
                        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                        # Compute the new y_lower for next plot
                        y_lower = y_upper + 10  # 10 for the 0 samples

                    ax1.set_title("The silhouette plot for the various clusters.")
                    ax1.set_xlabel("The silhouette coefficient values")
                    ax1.set_ylabel("Cluster label")

                    # The vertical line for average silhouette score of all the values
                    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                    ax1.set_yticks([])  # Clear the yaxis labels / ticks
                    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                    # 2nd Plot showing the actual clusters formed
                    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                    ax2.scatter(
                        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
                    )

                    # Labeling the clusters
                    centers = clusterer.cluster_centers_
                    # Draw white circles at cluster centers
                    ax2.scatter(
                        centers[:, 0],
                        centers[:, 1],
                        marker="o",
                        c="white",
                        alpha=1,
                        s=200,
                        edgecolor="k",
                    )

                    for i, c in enumerate(centers):
                        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

                    ax2.set_title("The visualization of the clustered data.")
                    ax2.set_xlabel("Feature space for the 1st feature")
                    ax2.set_ylabel("Feature space for the 2nd feature")

                    plt.suptitle(
                        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                        % n_clusters,
                        fontsize=14,
                        fontweight="bold",
                    )
                else:
                    fig = None

            n_clusters = list_n_clusters[np.argmax(list_silhouette_avg)]
            print("Using n_clusters =", n_clusters)

            clusterer = KMeans(n_clusters=n_clusters, n_init="auto")
            cluster_labels = clusterer.fit_predict(X)
            if PLOT_FINAL:
                fig_final, ax = plt.subplots()
                for i in range(n_clusters):
                    ax.plot(X[cluster_labels==i, 0], X[cluster_labels==i, 1], "x", alpha=0.5, label=i)
                # ax.scatter(X[:,0], X[:,1], c=cluster_labels, marker="x", alpha=0.5)
                ax.set_title(f"FINAL CLUSTERS, n={n_clusters}")
            else:
                fig_final = None

            # List of strings, so that these are categorical
            # cluster_labels = [int(c) for c in cluster_labels]
            cluster_labels = [f"{c}" for c in cluster_labels]

        if return_figs:
            return cluster_labels, fig, fig_final
        else:
            return cluster_labels

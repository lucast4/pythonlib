## bunch of stuff for doings stats, focussing on using pandas

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

def ttest_paired(x1, x2=None, ignore_nan=False):
    """ x1, x2 arrays, with same lenght
    if x2 is None, then assumes is comparing to 0s.
    Automatically removes any rows with nans
    """
    if x2 is None:
        x2 = np.zeros_like(x1)

    if ignore_nan:
        nan_policy='omit'
    else:
        nan_policy = 'propagate'
    return stats.ttest_rel(x1, x2, nan_policy=nan_policy)


def statsTtestPaired(df1, df2, varname, pairedon="human"):
    """ttest between two variables, assume it is paired at level of human (i.e, eachhuman 1pt each var pts)"""
    df12 = pd.merge(df1, df2, on=pairedon)
#     print(df12)
    return stats.ttest_rel(df12[f"{varname}_x"], df12[f"{varname}_y"])

def statsTtestUnpaired(df1, df2, varname):
    return stats.ttest_ind(df1[f"{varname}"], df2[f"{varname}"])

def statsRanksum(df1, df2, varname):
    return stats.ranksums(df1[f"{varname}"], df2[f"{varname}"])

# NOT YET DONE!!!
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
    if side=="two":
        n = sum(np.abs(stats_shuff)<=np.abs(stat_actual)) + 1
    elif side=="left":
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
        plt.title(f"p={p:.3f}")
    else: 
        fig = None
    
    if len(stats_shuff_collected)>0:
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
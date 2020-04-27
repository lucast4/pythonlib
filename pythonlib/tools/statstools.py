## bunch of stuff for doings stats, focussing on using pandas

from scipy import stats
import pandas as pd

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

""" tools for use with pandas dataframes. also some stuff using python dicts and translating between that and dataframs

3/20/21 - confirmed that no mixing of values due to index error:
- i.e., in general, if function keeps df size unchanged, then will not modify the indices, and 
will check that the output df is same as input (for matching columns). 
- if does change size, then will reset indices.
"""

import pandas as pd
import numpy as np


def _mergeKeepLeftIndex(df1, df2, how='left',on=None):
    """ merge df1 and 2, with output being same length
    as df1, and same indices. 
    - on, what columnt to align by, need this if they are differnet
    sizes.
    RETURN:
    - dfout, without modifying in place.
    """
    df1["index_copy"] = df1.index # save old indices.
    dfout = df1.merge(df2, how=how, on=on) # 
    dfout = dfout.set_index("index_copy", drop=True)
    dfout.index.names = ["index"]
    return dfout

def _checkDataframesMatch(df1, df2, check_index_match=True):
    """ Checks that:
    - for any index that is shared (between df1 and df2), 
    the values for all columns are shared. 
    - check_index_match, then also checks that df1 and df2
    are identical length, with identical indices
    NOTE: if have same values (relative to index) but index is
    shuffled rleative to row locaiton, then will fail check - i.e.
    df1[col].values muist equal df2[col].values for all col.
    """
    
    columns_shared = [c for c in df1.columns if c in df2.columns]
    
    if check_index_match:
        assert np.all(df1.index == df2.index), "indices are not the same!! maybe you applied reset_index() inadvertantely?"
        assert df1[columns_shared].equals(df2[columns_shared]), "values dont match"
        
        # to show that index must be in the same order to pass check.
#         assert df1[columns_shared].sort_values("hausdorff").sort_index().equals(df2[columns_shared])
                
    else:
        assert False, "not coded, since if frames are different size, not sure what youd want to check"


#############################vvvv OBSOLETE - USE aggregGeneral
def aggreg(df, group, values, aggmethod=["mean","std"]):
    """
    get group means, for certain dimensions(values). 
    e.g., group = ["worker_model", "worker_dat"]
    e.g. values = ["score", "cond_is_same"]
    NOTE: will change name of balues filed, e.g. to score_mean.
    OBSOLETE - USE aggregGeneral]
    """
    assert False, "[OBSOLETE - use aggregGeneral]"
    # this version did not deal with non-numeric stuff that would liek to preset
    # but was useful in taking both mean and std.
    df = df.groupby(group)[values].agg(aggmethod).reset_index()
    # df.columns = df.columns.to_flat_index()
    df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
    return df

def aggregMean(df, group, values, nonnumercols=[]):
    """
    get group means, for certain dimensions(values). 
    e.g., group = ["worker_model", "worker_dat"]
    e.g. values = ["score", "cond_is_same"]
    e.g. nonnumercols=["sequence", "name"] i.e., will take the first item it encounters.
    [OBSOLETE - USE aggregGeneral]
    """
    assert False, "[OBSOLETE - use aggregGeneral]"
    agg = {c:"mean" for c in df.columns if c in values }
    agg.update({c:"first" for c in df.columns if c in nonnumercols})
    print(agg)
    df = df.groupby(group).agg(agg).reset_index()
    # df.columns = df.columns.to_flat_index()
    # df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
    return df
########################### ^^^^ OBSOLETE - USE aggregGeneral


def aggregGeneral(df, group, values, nonnumercols=[], aggmethod=["mean"]):
    """
    get group means, for certain dimensions(values). 
    e.g., group = ["worker_model", "worker_dat"]
    e.g. values = ["score", "cond_is_same"]
    e.g. nonnumercols=["sequence", "name"] i.e., will take the first item it encounters.
    """
    agg = {c:aggmethod for c in df.columns if c in values }
    agg.update({c:"first" for c in df.columns if c in nonnumercols})
    
    print(agg)

    df = df.groupby(group).agg(agg).reset_index()
    # df.columns = df.columns.to_flat_index()

    if len(aggmethod)==1:
        # then reanme columns so same as how they came in:
        # e.g., dist instead of dist_mean. can't do if 
        # multiple aggmethods, since will then be ambiguos.
    # df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
        df.columns = [tup[0] for tup in df.columns.values]
    else:
        df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]

    return df

def df2dict(df):
    return df.to_dict("records")


def applyFunctionToAllRows(df, F, newcolname="newcol"):
    """F is applied to each row. is appended to original dataframe. F(x) must take in x, a row object"""
    # To debug:
    # def F(x):
    #     return x["trial"]
    # # applyFunctionToAllRows(Pp, F, newcolname="test")
    # Ppsorted = Pp.sort_values("hausdorff").reset_index()
    # applyFunctionToAllRows(Ppsorted, F, newcolname="test")

    assert newcolname not in df.columns, f"{newcolname} already exists as a col name"

    # # OLD VERSION FAILS IF INDICES PASSED IN ARE NOT IN ORDER, 
    # # becuase of the reset_index()
    # dfnewcol = df.apply(lambda x: F(x), axis=1).reset_index()
    # dfout = df.merge(dfnewcol, left_index=True, right_index=True).rename(columns={0:newcolname})

    dfnewcol = df.apply(F, axis=1).rename(newcolname) # rename, since series must be named.
    # print(dfnewcol)
    # print(dfnewcol.columns)
    # print(dfnewcol.index)
    dfout = df.merge(dfnewcol, how="left", left_index=True, right_index=True)
    _checkDataframesMatch(df, dfout)

    return dfout

############3333 SCRATCH NOTES

# Apoply function to group-wise:
# Note; I was unsure whether the index pulled out is the global index or within the group.
# I think it is glocal.
# F = lambda x: x.iloc[x["distance"].idxmin()]
# F = lambda x: SF.iloc[x["distance"].idxmin()]
# # F = lambda x: SF.iloc[1]
# SF.groupby(["task", "epoch"]).apply(F)

######### take rows based on max/min of for values of some column:
# SF.loc[SF.groupby(["task", "epoch"])["distance"].idxmax()]


# pivoting (to take multiple rows and make them colums)
# Y = SFagg.pivot(index="task", columns="epoch", values="distance_median")


def filterGroupsSoNoGapsInData(df, group, colname, values_to_check):
    """ filter df so that each group has at
    least one item for each of the desired values
    for the column of interest. useful for removing
    data which don't have data for all days, for eg...
    -- e.g., this only keeps tasks that have data for
    both epochs:
        values_to_check = [1,2]
        colname = "epoch"
        group = "unique_task_name"
    # NOTE - index of output will be reset.
    """
    def F(x):
        """ True if has data for all values"""
        checks  = []
        for v in values_to_check:
            checks.append(v in x[colname].values)
        return all(checks)
    return df.groupby(group).filter(F).reset_index(drop=True)


def getCount(df, group, colname):
    """ return df grouped by group, and with one count value
    for each level in group, the name of that will be colname
    - colname must be a valid column from df [not sure why..]
    """

    return df.groupby(group)[colname].count().reset_index()


def binColumn(df, col_to_bin, nbins, bin_ver = "percentile"):
    """ bin values from a column, assign to a new column.
    - col_to_bin, string, name of col
    - nbins, 
    - bin_ver, how to spread out bin edges.
    NOTE:
    - bins will range from 0, 1, .., up to num gaps between edges.
    RETURN:
    - modified df in place, new col with name <col_to_bin>_binned
    - new_col_name, string
    """
    vals = df[col_to_bin].values
    df_orig = df.copy()
    # 2) Get bin edges
    if bin_ver=="uniform":
        # even bin edges
        binedges = np.linspace(min(vals)-0.1, max(vals)+0.1, nbins+1) 
    elif bin_ver=="percentile":
        # percentile bin edges
        binedges = np.percentile(vals, np.linspace(0, 100, nbins+1))
        binedges[0]-=1
        binedges[-1]+=1
    else:
        print(bin_ver)
        assert False, "not coded"
        
    # 3) assign bins
    vals_binned = np.digitize(vals, binedges, False)-1

    # 4) Put back into dataframe
    new_col_name = f"{col_to_bin}_binned"
    df[new_col_name] = vals_binned
    print(f"added column: {new_col_name}")

    _checkDataframesMatch(df, df_orig)
    return new_col_name

def aggregThenReassignToNewColumn(df, F, groupby, new_col_name, 
    return_grouped_df=False):
    """ groups, then applies aggreg function, then reassigns that output
    back to df, placing into each row for each item int he group.
    e.g., if all rows split into two groups (0, 1), then apply function, then 
    each 0 row will get the same new val in new_col_name, and so on.
    - F, function to apply to each group.
    - groupby, hwo to group
    - new_col_name, what to call new col.
    output will be same size as df, but with extra column.
    - If groupby is [], then will apply to all rows
    """

    if len(groupby)==0:
        # then add dummy column
        dummyname="dummy"
        while dummyname in df.columns:
            dummyname+="1"
        df[dummyname] = 0
        groupby = [dummyname]
        remove_dummy=True
    else:
        remove_dummy=False


    dfthis = df.groupby(groupby).apply(F).reset_index().rename(columns={0:new_col_name})
    # dfthis will be smaller than df. but merge will expand dfthis.

    # NOTE: sanity check, mking sure that merge does correctly repopulate:
    # from pythonlib.tools.pandastools import aggregThenReassignToNewColumn
    # def F(x):
    #     return [list(set(x[kbin])), list(set(x["stroknum"]))]
    # tmp = aggregThenReassignToNewColumn(P.Xdataframe, F, [kbin, "stroknum"], "test")
    # # LOOK THRU THIS AND MAKE SURE EACH PAIR IS IDENTICAL.
    # for _, row in tmp.iterrows():
    #     if row["test"] != [row[kbin], row["stroknum"]]:
    #         print(row["test"])
    #         print([[row[kbin]], [row["stroknum"]]])
    #         print("--")

    
    if False:
        # old version, didnt work for cases where input index were not in order.

        # df_new = pd.merge(df, dfthis, on=groupby)
        # df_new = pd.merge(df, dfthis, how="left", on=groupby)

        # df_new = df.merge(dfthis, how="left", on=groupby)
        df_new = df.reset_index().merge(dfthis, how="left", on=groupby).set_index("index") # maintains indices.
    else:
        df_new = _mergeKeepLeftIndex(df, dfthis, how='left',on=groupby)

    # remove dummy
    if remove_dummy:
        del df[dummyname]
        del dfthis[dummyname]
        del df_new[dummyname]

    _checkDataframesMatch(df_new, df)

    if return_grouped_df:
        return df_new, dfthis
    else:
        return df_new

def filterPandas(df, filtdict, return_indices=False):
    """ 
    filtdict is dict, where each value is a list of
    allowable values.
    - See filtdict for format
    NOTE - doesnt modify in place. just returns.
    NOTE - return_indices, returns the original row indices
    (as a list of ints) instead of the modified df
    NOTE - if return dataframe, automaticlaly resets indices.
    """
    for k, v in filtdict.items():
        # print('--')
        # print(len(df))
        # print(k)
        # print(v)
#         print(df[k].isin(v))
        df = df[df[k].isin(v)]
        # print(len(df))
    if return_indices:
        return list(df.index)
    else:
        return df.reset_index(drop=True)



def pivotAndMerge(d1, df2):
    """
    """
    assert False, "in progress"

    # from line2strokmodel.py
    DF2 = pd.pivot_table(DF, index=["trial", "strok_num_0"], values="finalcost", columns="nsubstrokes").reset_index()
    DFstrokdur = pd.pivot_table(DF, index=["trial", "strok_num_0"], values="strok_dur").reset_index()
    DF2 = pd.merge(DF2, DFstrokdur)

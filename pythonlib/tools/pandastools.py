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
    NOTE: Not checking whether columns have same values, since this
    led to segmentation fault issues (if there was nan)
    """
    
    columns_shared = [c for c in df1.columns if c in df2.columns]
    
    if check_index_match:
        if len(df1)!=len(df2):
            print(df1)
            print(df2)
            assert False, "lengths must be the same to do this check"
        assert np.all(df1.index == df2.index), "indices are not the same!! maybe you applied reset_index() inadvertantely?"
        # dfthis1 = df1[df1.notna()]
        # dfthis2 = df2[df2.notna()]
        # assert dfthis1[columns_shared].equals(dfthis2[columns_shared])
        # for col in columns_shared:
        #     assert df1[col].fillna(0).equals(df2[col].fillna(0)), "values dont match"
        
        # to show that index must be in the same order to pass check.
#         assert df1[columns_shared].sort_values("hausdorff").sort_index().equals(df2[columns_shared])
                
    else:
        assert False, "not coded, since if frames are different size, not sure what youd want to check"


def mergeOnIndex(df1, df2):
    """ merge, keeping indexes unchanged, 
    df1 and df2 must come in with idnetical indices.
    will use union of columns, but making sure to not have
    duplicate columns. if duplicate, then assumes that they are
    identical in df1 and df2 (will check) and will
    take the column values from df1.
    """
    _checkDataframesMatch(df1, df2)

    cols_to_use = df2.columns.difference(df1.columns)
    dfout = pd.merge(df1, df2[cols_to_use], left_index=True, 
        right_index=True, how='outer', validate='one_to_one')
    return dfout


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


def applyFunctionToAllRows(df, F, newcolname="newcol", replace=True, just_return_newcol=False):
    """F is applied to each row. is appended to original dataframe. F(x) must take in x, a row object
    - validates that the output will be identically indexed rel input.
    INPUT:
    - just_return_newcol, then returns single column without merging with df.
    """
    # To debug:
    # def F(x):
    #     return x["trial"]
    # # applyFunctionToAllRows(Pp, F, newcolname="test")
    # Ppsorted = Pp.sort_values("hausdorff").reset_index()
    # applyFunctionToAllRows(Ppsorted, F, newcolname="test")


    # # OLD VERSION FAILS IF INDICES PASSED IN ARE NOT IN ORDER, 
    # # becuase of the reset_index()
    # dfnewcol = df.apply(lambda x: F(x), axis=1).reset_index()
    # dfout = df.merge(dfnewcol, left_index=True, right_index=True).rename(columns={0:newcolname})

    # GET NEW SERIES
    assert len(df)>0, "empty dataframe"
    dfnewcol = df.apply(F, axis=1).rename(newcolname) # rename, since series must be named.

    # DO MERGE
    # Make copy, since must delete if want to replace col
    dfthis = df.copy()
    if replace:
        if newcolname in dfthis.columns:
            del dfthis[newcolname]
    else:
        assert newcolname not in dfthis.columns, f"{newcolname} already exists as a col name"

    # print(dfnewcol)
    # print(dfnewcol.columns)
    # assert False
    # print(dfnewcol.index)
    if len(dfnewcol)!=len(dfthis):
        print(dfnewcol)
        print(dfthis)
        assert False

    # dfout = df.merge(dfnewcol, how="left", left_index=True, right_index=True)
    # _checkDataframesMatch(df, dfnewcol)

    if just_return_newcol:
        return dfnewcol

    dfout = dfthis.merge(dfnewcol, how="left", left_index=True, right_index=True, validate="one_to_one")

    if len(dfnewcol)!=len(dfout):
        print(dfnewcol)
        print(dfthis)
        print(dfout)
        assert False

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

def filterPandas(df, filtdict, return_indices=False, auto_convert_tolist=True):
    """ 
    filtdict is dict, where each value is a list of
    allowable values.
    PARAMS:
    - auto_convert_tolist, then any values of filtdict that arent lsits are converted
    to lists. assumes you tried to enter a single value to filter.
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
        if not isinstance(v, list):
            if auto_convert_tolist:
                    v = [v]
            else:
                assert isinstance(v, list), "must make into lists the values of filtdict"
        if len(df)>0:
            if isinstance(v[0], np.ndarray) or isinstance(df[k].values[0], np.ndarray):
                from .nptools import isin_close
                def _F(x):
                    return isin_close(x[k], v)
                trues = applyFunctionToAllRows(df, _F, just_return_newcol=True)
            else:
                trues = df[k].isin(v)
            df = df[trues]
            # print(len(df))
    if return_indices:
        return list(df.index)
    else:
        return df.reset_index(drop=True)

def findPandas(df, colname, list_of_vals, reset_index=True):
    """ returns df with only rows matchibng list_of_vals. 
    output will be same length as list_of_vals, with order matching.
    INPUT:
    - df
    - colname, the column to check
    - list_of_vals, the values to pick out the rows
    RETURNS:
    - dfout
    NOTE
    - is doing it in a slow way, iterating over items.
    - will fail if list_of_vals are not unique, or any val is not found
    - will reset index
    """
    assert len(set(list_of_vals))==len(list_of_vals), "values not all unique"

    dfout = pd.concat([df[df[colname]==v] for v in list_of_vals])

    assert len(dfout)==len(list_of_vals), "at least one val was not found in df"
    tmp = dfout[colname].tolist()
    for a,b in zip(tmp, list_of_vals):
        assert a==b, "mistake somewher"

    if reset_index:
        dfout = dfout.reset_index(drop=True)
    return dfout



def pivotAndMerge(d1, df2):
    """
    """
    assert False, "in progress"

    # from line2strokmodel.py
    DF2 = pd.pivot_table(DF, index=["trial", "strok_num_0"], values="finalcost", columns="nsubstrokes").reset_index()
    DFstrokdur = pd.pivot_table(DF, index=["trial", "strok_num_0"], values="strok_dur").reset_index()
    DF2 = pd.merge(DF2, DFstrokdur)

def printOverview(df, MAX=50):
    """
    print columns and values counts)
    - MAX, if num values greater than this, then skip printing.
    """
    if False:
        # old version, doesnt plto frequency of values
        from .dicttools import printOverviewKeyValues
        printOverviewKeyValues(df2dict(df)) 
    else:
        for col in df.columns:
            print(' ')
            print(f"-- {col}")
            if isinstance(df[col].values[0], list):
                print(f"*Skipping print, since type of values are lists")
                continue
            if isinstance(df[col].values[0], np.ndarray):
                print(f"*Skipping print, since type of values are np.ndarray")
                continue
            if isinstance(df[col].values[0], dict):
                print(f"*Skipping print, since type of values are dict")
                continue
            nvals = len(set(df[col].values))
            if nvals>MAX:
                print(f"*Skipping print, since {nvals} vals > MAX")
            else:
                print(df[col].value_counts())


def append_col_with_grp_index(df, grp, new_col_name):
    """ for each col, gets its grp index (based on grp list),
    and appends as new column. first converts to string by str(list)
    INPUTS:
    - grp, list of strings, each a column. order matters!
    RETURNS:
    - df, but with new col.
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
 
    # add a column, which is grp index
    def F(x):
        tmp = [x[g] for g in grp]
        return str(tmp)
    
    return applyFunctionToAllRows(df, F, new_col_name)    


def append_col_after_applying_to_group(df, groupby, cols_to_use, F, newcol):
    """ F takes in a group (dframe) and outputs a series of same length.
    appends the combined output to a new col with name newcol.
    Confirmed that the locations (rows) will be correctly aligned with the original
    dataframe.
    - cols_to_use, will pull out datafram df.groupby(groupby)[cols_to_use]. is list. THis
    goes into F.
    - F, dataframe --> series (or list)
    - newcol, name of new column
    NOTE: is like applyFunctionToAllRows but here applies function taking in a group, not a row.
    """

    # groupby = "character"
    # newcol = "test"

    assert isinstance(cols_to_use, list), "so get dataframe out"

    # 1) get the series.
    df[newcol] = df.groupby(groupby)[cols_to_use].transform(F)
    # dfnew[newcol] = dfnew[groupby]
    # del dfnew[groupby]
    # pd.concat([dfthis, dfnew], axis=1)

    return df


def append_col_with_index_in_group(df, groupby, colname="trialnum_chron", randomize=False):
    """ appends a col, which holds index (0, 1, 2.) in order within its level within groupby.
    e.g, if groupby has 2 levels (A and B), then this gives all rows with level A an index.
    e.g.. like trial numbers for a given condition/task.
    - randomize, then will randomize the indices (only within trails with same level of groupby)
    """

    # OLD method - avoid since it mutates
    # dfthis = D.Dat
    # def F(x):
    #     # assign in chron order
    #     x["test"] = range(len(x))
    # #     x["test"] = 1
    #     return x
        
    # dfthis = dfthis.groupby("character").apply(F)

    def F(x):
        # assign in chron order
        out = list(range(len(x)))
        if randomize:
            from random import shuffle
            shuffle(out)
        return out

    return append_col_after_applying_to_group(df, groupby, [groupby], F, colname)    


def pivot_table(df, index, columns, values, aggfunc = "mean", flatten_col_names=False):
    """
    Take a long-form datagrame, and convert into a wide form. 
    INPUTS:
    - index, the new index. pass in a list, if want to group. the output will have each group category 
    as a separate column. (e.g., if index=[a, b], then a and b will be columns in output.)
    - columns, generally keep this length 1, easy to understand. if len >1, then will be hierarchical 
    columns
    - values, naems of values, list, is fine to input multiple. will be hierarhcial. can index into the output
    df as df[col1][col2]...
    - flatten_col_names, if output is hierarchical, will flatten to <val>-<col1>-<col2>.., if 
    where col1, 2, ... are the items in columsn (if it is a list)
    RETURNS:
    - new dataframe, where can access e.g., by df["value"]["col_level"]
    NOTES:
    - Also aggregates, by taking mean over all cases with a given combo of (index, columns, values)
    - essentially groups by the uniion of index, columns, values, then aggregates, then reshapes so that index
    is over rows, and volumns/values are over columns (hierarhcical)
    - resets index, so that index levels will make up a column
    e.g.:
    - index=["unique_task_name"], columns=["block"], values=["time_touchdone", "time_raise2firsttouch"]
    - Look into pd.melt(dftmp, id_vars = ["block", "unique_task_name"]), if want to undo this. would need to 
    not rename columns.
    - can eitehr do:
    (1) agg, then do this or
    (2) do this directly (since it aggs by taking mean.)

    """

    dftmp = pd.pivot_table(data=df, index=index, columns=columns, values=values, aggfunc=aggfunc)

    if flatten_col_names:
        dftmp.columns = ['-'.join([str(c) for c in col]).strip() for col in dftmp.columns.values]

    # to reindex and retain indices as new columns
    dftmp = dftmp.reset_index()

    # OLD VERSION WHERE DID THE SAME THING BUT BY HAND
    # column_levels = [17, 18]
    # column_dsets = "block"
    # value = "time_touchdone"
    # def F(x):
    # #     cols=[str(column_levels[0]), str(column_levels[1])]
    #     cols=column_levels
    #     def _getval(x, level):
    #         val = x[x[column_dsets]==level][value].values
    #         if len(val)==0:
    #             return np.nan
    #         elif len(val)>1:
    #             assert False
    #         else:
    #             return val[0]
    #     val1 = _getval(x, column_levels[0])
    #     val2 = _getval(x, column_levels[1])

    #     tmp = [val1, val2]
    #     tmp = pd.Series(tmp, index=cols)
    #     return tmp

    # DF.groupby(["unique_task_name"]).apply(F).reset_index()


    return dftmp



def summarize_feature(df, GROUPING, FEATURE_NAMES,
                          INDEX= ["character", "animal", "expt"], 
                          func = lambda x: np.nanmean(x), newcol_variable=None, newcol_value=None):
    """ [USEFUL] wide-form --> long form
    aggregating and summarizing features
    See summarize_featurediff for variables.
    NOTE:
    - if have N rows, and 8 columsn with scores under 8 different models, and want to flatten to long-form
    so that there is a single "score" column with Nx8 rows (useful for plotting). Can do following:
    summarize_feature(D.Dat, "epoch", model_score_name_list, ["character", "trialcode"]), where
    model_score_name_list is list of 8 column names. The outcome is, where "value" is the "score" column.

    dfthisflat =
         epoch   character   trialcode   variable    value
    0   baseline    mixture2-ss-2_1-111763  210821-1-205    behmodpost_baseline_chunks  0.074286
    1   baseline    mixture2-ss-2_1-111763  210821-1-273    behmodpost_baseline_chunks  0.020258
    2   baseline    mixture2-ss-2_1-111763  210821-1-364    behmodpost_baseline_chunks  0.053116
    3   baseline    mixture2-ss-2_1-111763  210821-1-438    behmodpost_baseline_chunks  0.020556
    4   baseline    mixture2-ss-2_1-111763  210821-1-478    behmodpost_baseline_chunks  0.063520
    ...     ...     ...     ...     ...     ...
    307     lolli   mixture2-ss-2_1-111763  210902-1-116    behmodpost_lolli_mkvsmk     0.017931
    308     lolli   mixture2-ss-2_1-111763  210902-1-213    behmodpost_lolli_mkvsmk     0.008365
    309     lolli   mixture2-ss-6_1-854929  210901-1-399    behmodpost_lolli_mkvsmk     0.013021
    310     lolli   mixture2-ss-6_1-854929  210901-1-598    behmodpost_lolli_mkvsmk     0.007421
    311     lolli   mixture2-ss-6_1-854929  210902-1-176    behmodpost_lolli_mkvsmk     0.010960

    (If want to aggregate over all trials, then use ["character"] instead. )

    dfthis is basically looks like the input shape, but pruned to the relevant columns.
    """
    if not isinstance(GROUPING, list):
        GROUPING = [GROUPING]
    dfagg = aggregGeneral(df, GROUPING + INDEX, FEATURE_NAMES, aggmethod=[func])
    dfaggflat = pd.melt(dfagg, id_vars = GROUPING + INDEX)

    # change name
    if newcol_variable is not None:
        dfaggflat = dfaggflat.rename(columns ={"variable":newcol_variable})
    if newcol_value is not None:
        dfaggflat = dfaggflat.rename(columns ={"value":newcol_value})

    return dfagg, dfaggflat




def summarize_featurediff(df, GROUPING, GROUPING_LEVELS, FEATURE_NAMES,
                          INDEX= ["character", "animal", "expt"], 
                          func = lambda x: np.nanmean(x), return_dfpivot=False, 
                          do_normalize=False, normalize_grouping = ["animal", "expt"]
                         ):
    """ High level summary, for each task, get its difference (for eg)
    across two levels for grouping (e..g, epoch 1 epoch2), with indices seaprated
    by INDEX (usually, animal/expt/character).
    INPUTS:
    - GROUPING, dictates which levels will be split into separate columns
    - FEATURE_NAMES, will only keep these features (columns)
    - INDEX, how to split up into unique columns, based on unique indices.
    - func, how to aggregate across multipel rows.
    - do_normalize, if True, then dfsummaryflat will have normalized values.
    i.e., for each (animal, expt, variable), take frac change relative to GROUPING_LEVEL[0]
    i.e., (a-b)/abs(b), where b is after averaging over all tasks. Will put this in a new column
    caleed "value_norm"
    OUTPUT:
    - dfsummary, new dataframe, with rows = unique combos of index, and columns line:
    ("total_time") [see eg below]
    - dfsummaryflat, similar but flattened, so that only columns are to identify index
    NOTES:
    - e.g, starting from D, 
    --- INDEX = ["character", "animal", "expt"], (must be rows in input dataframe)
    --- GROUPING = ["plan_time_cat"] --> levels {"short, "long"}  (must be 
    rows in input dataframe, e.g., short is a value that plantimecat can take
    --- FEATURE_NAMES = ["total_time", "distance", ...] (must be columns in 
    input datafrane)
    """
    
    # 1) Aggregate and split by grouping
    dfpivot = pivot_table(df, index=INDEX, columns=[GROUPING], values=FEATURE_NAMES, 
                          aggfunc=func)

    # ===== Compute summary statistic (e.g., difference across groupings)
    out = {}
    COLNAMES_DICT = []
    COLNAMES_NOABS = []
    COLNAMES_ABS = []
    COLNAMES_DIFF = []

    # 2) all other features, take difference
    for val2 in FEATURE_NAMES:
        if val2=="alignment":
            # 1) alignemnt, take mean
            colname = f"{val2}-MEAN"
            colvals = np.nanmean(np.c_[dfpivot[val2][GROUPING_LEVELS[0]].values, 
                                                dfpivot[val2][GROUPING_LEVELS[1]].values], axis=1)
        else:
            colname = f"{val2}-{GROUPING_LEVELS[1]}min{GROUPING_LEVELS[0]}"
            colvals = dfpivot[val2][GROUPING_LEVELS[1]] - dfpivot[val2][GROUPING_LEVELS[0]]
            COLNAMES_DIFF.append(colname)

        out[colname] = colvals

        # get the abs balue
        out[f"{colname}-ABS"] = np.abs(colvals)

        COLNAMES_NOABS.append(colname)
        COLNAMES_ABS.append(f"{colname}-ABS")

    # 3) retain character name
    for colthis in INDEX:
        out[colthis] = dfpivot[colthis]

    dfsummary = pd.DataFrame(out)
    dfsummary=dfsummary.dropna().reset_index(drop=True)

    # ==== Melt into long-form dataset
    dfsummaryflat = pd.melt(dfsummary, id_vars = INDEX)
    predicted_len = (len(dfsummary.columns)-len(INDEX)) * len(dfsummary)
    assert len(dfsummaryflat)==predicted_len

    # === Noramlize?
    if do_normalize:
        # Normalization, collect denominators (for percent change, signed)
        denom_level = GROUPING_LEVELS[0]
        normalization_denoms = {}
        for g in dfpivot.groupby(normalize_grouping):
            for var in FEATURE_NAMES:
                x = np.nanmean(g[1][var][denom_level])
                normalization_denoms[g[0] + tuple([var])] = x

        # extract "original" variabl ename
        def F(x):
            return x["variable"][:x["variable"].find("-")]
        dfsummaryflat = applyFunctionToAllRows(dfsummaryflat, F, "variable_origname")

        # compute normalization
        def F(x):
            idx = (x["animal"], x["expt"], x["variable_origname"])
            denom = normalization_denoms[idx]
            xnorm = x["value"]/np.abs(denom) # take abs to retain sign
            return xnorm
        dfsummaryflat = applyFunctionToAllRows(dfsummaryflat, F, "value_norm")

    
    if return_dfpivot:
        return dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot
    else:
        return dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF


# dfsummary, dfsummaryflat = summarize_featurediff(Dall.Dat, GROUPING,GROUPING_LEVELS,FEATURE_NAMES)
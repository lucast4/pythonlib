""" tools for use with pandas dataframes. also some stuff using python dicts and translating between that and dataframs

3/20/21 - confirmed that no mixing of values due to index error:
- i.e., in general, if function keeps df size unchanged, then will not modify the indices, and 
will check that the output df is same as input (for matching columns). 
- if does change size, then will reset indices.
"""

import pandas as pd
import numpy as np
from pythonlib.tools.listtools import sort_mixed_type


def _mergeKeepLeftIndex(df1, df2, how='left',on=None):
    """ merge df1 and 2, with output being same length
    as df1, and same indices. 
    - on, what columnt to align by, need this if they are differnet
    sizes.
    RETURN:
    - dfout, without modifying in place.
    """

    df1 = df1.copy() # becuase dont want to set value on a slice, need to copy.
    df1["index_copy"] = df1.index # save old indices.
    dfout = df1.merge(df2, how=how, on=on) # 
    # dfout = df1.merge(df2, how=how, on=on, validate="one_to_one") # 
    assert len(dfout)==len(df1)
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
# def aggreg(df, group, values, aggmethod=["mean","std"]):
#     """
#     get group means, for certain dimensions(values). 
#     e.g., group = ["worker_model", "worker_dat"]
#     e.g. values = ["score", "cond_is_same"]
#     NOTE: will change name of balues filed, e.g. to score_mean.
#     OBSOLETE - USE aggregGeneral]
#     """
#     assert False, "[OBSOLETE - use aggregGeneral]"
#     # this version did not deal with non-numeric stuff that would liek to preset
#     # but was useful in taking both mean and std.
#     df = df.groupby(group)[values].agg(aggmethod).reset_index()
#     # df.columns = df.columns.to_flat_index()
#     df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
#     return df

# def aggregMean(df, group, values, nonnumercols=[]):
#     """
#     get group means, for certain dimensions(values). 
#     e.g., group = ["worker_model", "worker_dat"]
#     e.g. values = ["score", "cond_is_same"]
#     e.g. nonnumercols=["sequence", "name"] i.e., will take the first item it encounters.
#     [OBSOLETE - USE aggregGeneral]
#     """
#     assert False, "[OBSOLETE - use aggregGeneral]"
#     agg = {c:"mean" for c in df.columns if c in values }
#     agg.update({c:"first" for c in df.columns if c in nonnumercols})
#     print(agg)
#     df = df.groupby(group).agg(agg).reset_index()
#     # df.columns = df.columns.to_flat_index()
#     # df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
#     return df
########################### ^^^^ OBSOLETE - USE aggregGeneral


def aggregGeneral(df, group, values, nonnumercols=None, aggmethod=None):
    """
    Aggregate by first grouping (across multiple dimensions) and then applyiong
    arbnitrary method.
    PARAMS;
    - group, list of str, each a column. groups using combos fo these
    - values, list of str, which values to apply aggregation. the returned df will
    have these as new columns.
    - nonnumercols, list of str. these columsn will be retained, keeping only the 
    first encountered value.
    - aggmethod, list of str, applies each of these agg methods.
    """
        
    if aggmethod is not None:
        assert isinstance(aggmethod, list)
    assert isinstance(values, list)
    assert isinstance(group, list)

    if nonnumercols is None:
        nonnumercols = []
    else:
        # Check they all exist and has a single unique item
        for col in nonnumercols: 
            groupdict = grouping_append_and_return_inner_items(df, group, 
                groupinner=col, new_col_name="dummy")
            for k, v in groupdict.items():
                assert len(v)==1, "nonnumercols takes the first. this is wrong if there are multipel..."

    if aggmethod is None:
        aggmethod = ["mean"]

    agg = {c:aggmethod for c in df.columns if c in values}
    agg.update({c:"first" for c in df.columns if c in nonnumercols})
    
    # print(agg)

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

def filter_prune_min_n_rows(df, column, min_n):
    """ Returns copy of df pruning so that each  level of column has at leasnt
    min_n rows. Resets index of output
    """
    assert False, 'code it'


def prune_min_ntrials_across_higher_levels(df, col_high, col_low, n_min):
    """ Wrapper for filterGroupsSoNoGapsInData
        e.g, only keep characters that have at least 5 trials in each epoch:
    self.prune_min_ntrials_across_higher_levels("epoch", "character", 5)
    PARAMS:
    - col_high, the higher-level column, string, e.g., "epoch"
    - col_low, the lower-level column, e.g., "character".
    RETURNS:
    - df, pruned. (Does not modify self.Dat)
    """
    return filterGroupsSoNoGapsInData(df, col_low, col_high, min_n_trials=n_min)

def filterGroupsSoNoGapsInData(df, group, colname, colname_levels_to_check=None,
        min_n_trials = 1):
    """ filter df so that each group has at
    least one item for each of the desired values
    for the column of interest. useful for removing
    data which don't have data for all days, for eg...
    -- e.g., this only keeps tasks that have data for
    both epochs:
        colname_levels_to_check = [1,2]
        colname = "epoch"
        group = "unique_task_name"
    # NOTE - index of output will be reset.
    """

    if colname_levels_to_check is None:
        colname_levels_to_check = df[colname].unique().tolist()

    def F(x):
        """ True if has data for all values"""
        checks  = []

        # count n trials for each level of colname
        list_n =[]
        for v in colname_levels_to_check:
            n = sum(x[colname]==v)
            list_n.append(n)

        # check if each n is >= than min
        list_good = [n>=min_n_trials for n in list_n]
        return all(list_good)
        # for v in colname_levels_to_check:
        #     print(x[colname].values)
        #     checks.append(v in x[colname].values)
        # return all(checks)
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
    return_grouped_df=False, overwrite_col=True):
    """ groups, then applies aggreg function, then reassigns that output
    back to df, placing into each row for each item int he group.
    e.g., if all rows split into two groups (0, 1), then apply function, then 
    each 0 row will get the same new val in new_col_name, and so on.
    - F, function to apply to each group.
    - groupby, list of str hwo to group
    - new_col_name, what to call new col.
    output will be same size as df, but with extra column.
    - If groupby is [], then will apply to all rows
    """

    assert isinstance(groupby, (list, tuple))

    if new_col_name in df.columns:
        del df[new_col_name]

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

    assert 0 not in df.columns, "the new coliumn is expected to be called 0"
    dfthis = df.groupby(groupby).apply(F).reset_index()
    assert 0 in dfthis.columns
    dfthis = dfthis.rename(columns={0:new_col_name})
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

def filterPandas(df, filtdict, return_indices=False, auto_convert_tolist=True, 
        reset_index=True):
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
        if reset_index:
            return df.reset_index(drop=True)
        else:
            return df


def filterPandasMultOrs(df, list_varnames, list_filts, return_as = "inds",
    verbose=False):
    """ Apply multiple filters in a row, and get the union of all
    the resulting indices into df. 
    Flexible way to define filters.
    PARAMS;
    - df, dataframe
    - list_varnames, list of str, the column names to consider
    - list_filts, list of lists, where each inner list has items
    which define what levels to keep fore ach var. the final output
    is the conjunction of each of these inner items. A few ways to define.
    e..g, list_filts = [list1, list2, ... (arbitrary number of lists)], where
    list1 = [list_of_levels_for_var1, list_of_levels_for_var2], where var1 and var2 are in list_varnames
    list1 = [level_for_var1,...] so like above, but a single level
    list1 = [None, ..] where None means keep all levels (i.e., ignore this variable).
    RETURNS:
    - Depends on return_as:
    --- if "inds", then indices (corresponding to df.index), 
    --- if "dataframe", then df
    --- if "dict", then list of dicts
    NOTE:
    - useful, for eg. if taking subset of data as heldout data for model testing.
    - is "or" for outer list, "and" for inner list.
    EG:
    list_varnames = DATAPLOT_GROUPING_VARS
    list_filts = [
        ["Lcentered-4-0", (1,1)],
        ["V-2-0", None],
        [["squiggle3-1-0", "circle-1-0"], None],
        [["squiggle3-1-0", "circle-1-0"], [(-1,1), (-1,-1)]]
    ]

    """
    
    list_inds = []
    for innerlist in list_filts:
        # get trials that are conjucntion of these features
        F = {}
        for varname, levels in zip(list_varnames, innerlist):
            if levels is None:
                # skip, i.e., get all levels
                continue
            if isinstance(levels, list):
                # then keep if it is any item in the list
                F[varname] = levels
    #         elif isinstance(levels, (str, tuple)):
            else:
                # this is assumed to be a single item. put in a list
                F[varname] = [levels]

        if verbose:
            print("For this filt: ", innerlist, ' -- Using this filt: ' , F)

        # Apply filter 
        inds = filterPandas(df, F, return_indices=True)
        list_inds.extend(inds)

    list_inds = sorted(set(list_inds))
    if verbose:
        print("Got this many indices: ", len(list_inds))

    if return_as=="dataframe":
        return df.iloc[list_inds]
    elif return_as=="inds":
        return list_inds
    elif return_as=="dict":
        return df.iloc[list_inds].to_dict("records")
    else:
        assert False

def filter_by_min_n(df, colname, n_min_per_level, types_to_consider=(str, int)):
    """ for each level in df[colname] prune all of its instances if
    the n rows is less than n_min_per_level
    PARAMS;
    - colname, str
    - n_min_per_level, int
    - types_to_consider, list of types, only prune levels that are type in this list.
    This is important so that doesnt throw out all numerical data.
    RETURNS:
    - df, a copy of the input, pruned
    """

    # n_min_per_level = 40 # min total n trials for level
    from pythonlib.tools.checktools import check_is_categorical
    
    levels = df[colname].unique().tolist()

    indstoremove = []

    for lev in levels:
        # Skip if this level is numerical
        if check_is_categorical(lev):
        # if isinstance(lev, types_to_consider):
            n = sum(df[colname]==lev)
            if n<n_min_per_level:
                # then remove it
                inds = df[df[colname]==lev].index.tolist()
                indstoremove.extend(inds)
                print(f'REMOVING, n={n} |', lev, ' --------  ', sum(df[colname]==lev))
            print(lev, ' --------  ', sum(df[colname]==lev))
    print('(removing this many indices): ', len(set(indstoremove)))
    df = df.copy()
    df = df.drop(list(set(indstoremove))).reset_index(drop=True)
    return df

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

# @param grp: [col1 col2 col3]
# @return: new column with index for category (based on col1/2/3 perm)
def append_col_with_grp_index(df, grp, new_col_name, use_strings=True, 
        strings_compact=False, return_col_name=False):
    """ for each col, gets its grp index (based on grp list),
    and appends as new column. first converts to string by str(list)
    INPUTS:
    - grp, list of strings, each a column. order matters!
    - use_strings, bool (True), then items are strings, otherwise are tuples.
    RETURNS:
    - df, but with new col. either values as string or tuples
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    
    for x in grp:
        assert not isinstance(x, (list, tuple)), "leads to errors. you prob want to flatten grp"

    # add a column, which is grp index
    def F(x):
        tmp = [x[g] for g in grp]
        if use_strings:
            if strings_compact:
                tmp_compact = []
                for t in tmp:
                    if isinstance(t, bool):
                        if t:
                            tmp_compact.append("1")
                        else:
                            tmp_compact.append("0")
                    else:
                        tmp_compact.append(str(t))
                return "|".join(tmp_compact)
            else:
                return str(tmp) 
        else:
            return tuple(tmp)
    df = applyFunctionToAllRows(df, F, new_col_name)  

    if return_col_name:
        return df, new_col_name 
    else:
        return df


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

def convert_to_1d_dataframe_hist(df, col1, plot_hist=True):
    """ Aggregate (usually counts for each level of col1).
    Plots a histogram, ordered by the n counts
    RETURNS:
    - labels, ncounts, list of str and ncounts, matching, sorted from high to low.
    - fig, ax
    """
    x = df[col1].value_counts()
    labels = x.index
    ncounts = x.values
    if plot_hist:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.bar(labels, ncounts)
        # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45)
        plt.xticks(rotation=90)

        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # df[col1].hist(xrot=45)
    else:
        fig, ax = None, None
    return labels, ncounts, fig, ax

def convert_to_2d_dataframe(df, col1, col2, plot_heatmap=False, 
        agg_method = "counts", val_name = "val", ax=None, 
        norm_method=None,
        annotate_heatmap=True, zlims=(None, None),
        diverge=False, dosort_colnames=True,
        list_cat_1 = None, list_cat_2 = None):
    """ Reshape dataframe (and prune) to construct a 2d dataframe useful for 
    plotting heatmap. Eech element is unique combo of item for col1 and col2, 
    with a particular aggregation function (by default is counts). 
    PARAMS:
    - col1, string name of col whose values will become row indices
    - col2, string name of col whose values wil become column indices.
    - plot_heatmap, bool
    - agg_method, str, what to put into each cell (by agging across datapts)
    - val_name, str name of column, optional if the agg function requires this (e,g. mean)
    - annotate_heatmap, bool, to put numerical values in each cell as text.
    - diverge, bool, if True, then centers the heat values.
    RETURNS:
    - 2d dataframne,
    - fig, 
    - ax,
    - rgba_values, (nrows, ncols, 4), where rgba_values[0,1], means rgba value for row 0 col 1.
    """
    from pythonlib.tools.snstools import heatmap

    # If col2 is None, then give a dummy varaible, so that this code runs
    if col2 is None:
        # then is really a 1d plot
        df["dummy"] = 0
        col2="dummy"

    if list_cat_1 is None:
        list_cat_1 = df[col1].unique()

    if list_cat_2 is None:
        list_cat_2 = df[col2].unique()

    if dosort_colnames:
        list_cat_1 = sorted(list_cat_1)
        list_cat_2 = sorted(list_cat_2)

    arr = np.zeros((len(list_cat_1), len(list_cat_2)))
    for i, val1 in enumerate(list_cat_1):
        for j, val2 in enumerate(list_cat_2):
            
            dfsub = df[(df[col1] == val1) & (df[col2] == val2)]
            if agg_method=="counts":
                # get counts
                n = len(dfsub)
                valthis = n
            elif agg_method=="mean":
                valthis = dfsub[val_name].mean()
            else:
                print(agg_method)
                assert False
            arr[i, j] = valthis
    
    dfthis = pd.DataFrame(arr, index=list_cat_1, columns=list_cat_2)

    if norm_method=="col_div":
        # normalize so that for each col, the sum across rows is 1
        assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
        dfthis = dfthis.div(dfthis.sum(axis=0), axis=1)
    elif norm_method=="row_div":
        # same, but for rows
        assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
        dfthis = dfthis.div(dfthis.sum(axis=1), axis=0)
    elif norm_method=="col_sub":
        # normalize so by subtracting from each column its mean across rows
        dfthis = dfthis.subtract(dfthis.mean(axis=0), axis=1)
        diverge = True
    elif norm_method=="row_sub":
        # normalize so by subtracting from each column its mean across rows
        dfthis = dfthis.subtract(dfthis.mean(axis=1), axis=0)
        diverge = True
    elif norm_method=="row_sub_firstcol":
        # for each item in a given row, subtract the value of the first colum in that row.
        dfthis = dfthis.subtract(dfthis.iloc[:,0], axis=0)
    elif norm_method is None:
        pass
    else:
        print(dfthis)
        print(norm_method)
        assert False

    if plot_heatmap:
        fig, ax, rgba_values = heatmap(dfthis, ax, annotate_heatmap, zlims, diverge=diverge)
        ax.set_xlabel(col2)
        ax.set_ylabel(col1)
        ax.set_title(agg_method)
    else:
        fig, ax, rgba_values = None, None, None

    if "dummy" in df.columns:
        del df["dummy"]
    return dfthis, fig, ax, rgba_values
            



def pivot_table(df, index, columns, values, aggfunc = "mean", flatten_col_names=False,
    flatten_separator="-", col_strings_ignore=None):
    """
    Take a long-form datagrame, and convert into a wide form. 
    PARAMS:
    - index, list of str, defining a grouping variable. Each level of this variable will be a
    new row in the output dataframe. 
    - columns, list of str or str, a grouping variable. each level of this variable will define a
    new column in output dataframe. (generally keep this length 1, easy to understand. if len >1, then will be hierarchical 
    columns)
    - values, list of str, the response variable, whose values will populate the cells in the output dataframe,
    will be aggregated (mean) across instance with same col and index.
    - flatten_col_names, if output is hierarchical, will flatten to <val>-<col1>-<col2>.., if 
    where col1, 2, ... are the items in columsn (if it is a list)
    RETURNS:
    - new dataframe, where can access e.g., by df["value"]["col_level"]
    NOTES:
    - Also aggregates, by taking mean (for a given value) over all cases with a given combo of (index, columns)
    - essentially groups by the uniion of index, columns, then aggregates, then reshapes so that index
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

    if col_strings_ignore is None:
        col_strings_ignore = []
    if flatten_col_names:
        dftmp.columns = [flatten_separator.join([str(c) for c in col if c not in col_strings_ignore]).strip() for col in dftmp.columns.values]

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
                          INDEX= ("character", "animal", "expt"), 
                          func = None, newcol_variable=None, newcol_value=None):
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
    if func is None:
        func = lambda x: np.nanmean(x)
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
                          INDEX= ("character", "animal", "expt"), 
                          func = None, return_dfpivot=False, 
                          do_normalize=False, normalize_grouping = ("animal", "expt")
                         ):
    """ High level summary, for each task (or grouping), get its difference 
    across two levels for grouping (e..g, epoch 1 epoch2), with indices seaprated
    by INDEX (usually, animal/expt/character).
    INPUTS:
    - GROUPING, str, will take the difference of each item in FEATURE_NAMES between the two 
    levels in GROUPING. e.g., GROUPING = "epoch", then takes differfence between two epochs.
    - GROUPING_LEVELS, list of two str, the two levels to take difference of
    - FEATURE_NAMES, list of str, column names for numerical values that you wish to take
    difference of. e..g, FEATURE_NAMES = ["score"]. if len >1 then rtake difference separtely for each item.
    - INDEX, list of str, each a column, the conjunction defines the new row levels
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
    # EXAMPLE:
    (levels for align_to are "stroke_onset" and "go_cue")
    dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF = 
                    summarize_featurediff(dfmodels, GROUPING="align_to", GROUPING_LEVELS=["go_cue", "stroke_onset"], 
                      FEATURE_NAMES=["score_test_mean"], INDEX=["regions_str", "yfeat"])
        
score_test_mean-stroke_onsetmingo_cue   score_test_mean-stroke_onsetmingo_cue-ABS   ntrain-stroke_onsetmingo_cue    ntrain-stroke_onsetmingo_cue-ABS    regions_str     yfeat
0   -0.001340   0.001340    0   0   ALL     gridloc
1   0.160566    0.160566    0   0   ALL     shape_oriented
2   -0.107563   0.107563    0   0   FP_p-FP_a   gridloc
3   0.017268    0.017268    0   0   FP_p-FP_a   shape_oriented
4   0.175209    0.175209    0   0   M1_m-M1_l   gridloc
5   0.284695    0.284695    0   0   M1_m-M1_l   shape_oriented
6   -0.094072   0.094072    0   0   PMd_a-PMd_p     gridloc
7   0.287808    0.287808    0   0   PMd_a-PMd_p     shape_oriented
8   0.069778    0.069778    0   0   PMv_m-PMv_l     gridloc
9   0.004547    0.004547    0   0   PMv_m-PMv_l     shape_oriented
10  -0.034416   0.034416    0   0   SMA_p-SMA_a     gridloc
11  0.244854    0.244854    0   0   SMA_p-SMA_a     shape_oriented
12  -0.073975   0.073975    0   0   dlPFC_p-dlPFC_a     gridloc
13  0.234116    0.234116    0   0   dlPFC_p-dlPFC_a     shape_oriented
14  -0.178012   0.178012    0   0   preSMA_p-preSMA_a   gridloc
15  0.158417    0.158417    0   0   preSMA_p-preSMA_a   shape_oriented
16  -0.094409   0.094409    0   0   vlPFC_p-vlPFC_a     gridloc
17  0.107170    0.107170    0   0   vlPFC_p-vlPFC_a     shape_oriented    
    """

    if func is None: 
        func = lambda x: np.nanmean(x)
    assert len(GROUPING_LEVELS)==2
    assert isinstance(GROUPING, str), "see docs for pivot_table"
    
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

def extract_trials_spanning_variable(df, varname, varlevels=None, n_examples=1,
                                    F = None, return_as_dict=False, 
                                    method_if_not_enough_examples="prune_subset"):
    """ To uniformly sample rows so that spans levels of a given variable (column)
    e..g, if a col is "shape" and you want to get one example random trial of each shape,
    then varname="shape" and varlevels = list of shape names, or None to get all.
    PARAMS:
    - df, dataframe
    - varname, column name
    - varlevels, list of string, levels. or None to get all.
    - n_examples, how many to get of each (random)
    - F, dict for filtering. will use this and append the varname:varlevel.
    - return_as_dict, then returns as out[level] = <list of indices>
    RETURNS:
    - list_inds, list of ints or None (for cases where can't get this many examples, 
    will fail all examples, not just the excess... (should fix this)). if n_examples >1, then
    adjacnet inds will be for the same example.
    - varlevels
    """
    import random
    if F is None:
        F = {}

    if False: # This didnt make sense to me...
        if method_if_not_enough_examples=="prune_subset":
            assert return_as_dict==True, "otherwise will lose traick of inidces."

    # Get the levels of this vars
    if varlevels is None:
        varlevels = df[varname].unique().tolist()
    
    # For each level, find n examples
    list_inds = []
    outdict = {}
    for val in varlevels:
        F[varname] = val
        list_idx = filterPandas(df, F, True)
        if len(list_idx)>=n_examples:
            inds = random.sample(list_idx, n_examples)[:n_examples]
        else:
            if method_if_not_enough_examples == "all_none":
                # option 1< return all as None
                inds = [None for _ in range(n_examples)]
            elif method_if_not_enough_examples=="prune_subset":
                # sample size changes... keep how many you have
                # n_examples = len(list_idx)
                # inds = random.sample(list_idx, n_examples)[:n_examples]
                inds = list_idx
            elif method_if_not_enough_examples== "fail":
                assert False, "not enough trials "
            else:
                assert False
        list_inds.extend(inds)
        outdict[val] = inds
    if return_as_dict:
        return outdict, varlevels
    else:
        return list_inds, varlevels


def grouping_get_inner_items(df, groupouter="task_stagecategory", 
    groupinner="index", groupouter_levels=None, nrand_each=None, sort_keys=False):
    """ Return dict of unique items (levels of groupinner), grouped
    by groupouter levels. 
    PARAMS:
    - groupouter, string, the first grouping.
    - groupinner, string, the second grouping. either a column or "index"
    - groupouter_levels, list of values to use for groupouter. if None, then 
    finds and uses all.
    - nrand_each, int,then gets max this much per levle of groupouter. gets 
    random, sorted. if None then gets all.
    RETURNS:
    - groupdict, where each key is a level of groupouter, and
    items are the unique values of groupinner that exist for that
    groupouter level.
    EXAMPLE:
    - if groupouter = date and groupinner = character, then returns
    {date1:<list of strings of unique characters for date1>, 
    date2:<list of strings ....}
    """
    if groupouter_levels is None:
        groupouter_levels = df[groupouter].unique()

    if sort_keys:
        groupouter_levels = sorted(groupouter_levels)
        
    groupdict = {}
    for lev in groupouter_levels:
        dfthisgroup = df[df[groupouter]==lev]
        if groupinner=="index":
            itemsinner = dfthisgroup.index.tolist()
        else:
            itemsinner = dfthisgroup[groupinner].unique().tolist()
        if nrand_each is not None:
            if len(itemsinner)>nrand_each:
                import random
                itemsinner = random.sample(itemsinner, nrand_each)
        groupdict[lev] = itemsinner
    return groupdict

def grouping_append_and_return_inner_items(df, list_groupouter_grouping_vars, 
    groupinner="index", groupouter_levels=None, new_col_name="grp"):
    """ Does in sequence (i) append_col_with_grp_index (ii) grouping_get_inner_items.
    Useful if e./g., you want to get all indices for each of the levels in a combo group,
    where the group is defined by conjunction of two columns.
    PARAMS:
    - list_groupouter_grouping_vars, list of strings, to define a new grouping variabe,
    will append this to df. this acts as the groupouter.
    - groupinner, see grouping_get_inner_items
    - groupouter_levels, see grouping_get_inner_items
    RETURNS:
    - groupdict, see grouping_get_inner_items
    NOTE: does NOT modify df.
    """

    assert not isinstance(groupinner, list)

    # 1) Append new grouping variable to each row
    while new_col_name in df.columns:
        new_col_name+="_"
    df = append_col_with_grp_index(df, list_groupouter_grouping_vars, new_col_name, use_strings=False)


    # 2) Get dict of eaceh group
    groupdict = grouping_get_inner_items(df, new_col_name, groupinner, groupouter_levels=groupouter_levels)
    
    return groupdict

def grouping_print_n_samples(df, list_groupouter_grouping_vars, Nmin=0, savepath=None,
        save_convert_keys_to_str = False, save_as="dict"):
    """ print the sample size for each conjunctive level (defined by grouping list: list_groupouter_grouping_vars)
    e.g., if goruping is [shape, location, size]: prints:
    ('Lcentered-3-0', (-1, -1), 'rig3_3x3_small') 58
    ('Lcentered-3-0', (-1, 1), 'rig3_3x3_small') 42
    ('Lcentered-3-0', (1, -1), 'rig3_3x3_small') 58
    ('Lcentered-3-0', (1, 1), 'rig3_3x3_small') 66
    ('Lcentered-4-0', (-1, -1), 'rig3_3x3_small') 52
    ('Lcentered-4-0', (-1, 1), 'rig3_3x3_small') 53
    ('Lcentered-4-0', (1, -1), 'rig3_3x3_small') 59
    ('Lcentered-4-0', (1, 1), 'rig3_3x3_small') 44
    ('V-2-0', (-1, -1), 'rig3_3x3_small') 73, 
    ... and so on.
    PARAMS;
    - list_groupouter_grouping_vars, list of str
    - savepath, include extension, saves as yaml
    - Nmin, int, skips any cases that have fewer than Nmin samples.
    - print_value_not_n, then etracts not n, but the actual values (list)
    RETURNS:
    - outdict, dict[level]=n
    """
    outdict = {}
    out = grouping_append_and_return_inner_items(df, list_groupouter_grouping_vars)
    # out[grp] = list_indices.
    for k in sort_mixed_type(out.keys()):
        n = len(out[k])
        if n>Nmin:
            # if print_value_not_n:
            #     print(k)
            #     print(out)
            #     assert False
            #     outdict[k] = out[k]
            # else:
            outdict[k] = n

    if savepath is not None:
        if save_as=="dict":
            from .expttools import writeDictToYaml
            if save_convert_keys_to_str:
                outdict = {str(k):v for k, v in outdict.items()}
            writeDictToYaml(outdict, savepath)
        elif save_as in ["txt", "text"]:
            # Then each is a string line, write to text
            from .expttools import writeStringsToFile
            lines = [f"{str(k)} : {v}" for k, v in outdict.items()]
            writeStringsToFile(savepath, lines)
        else:
            print(save_as)
            assert False
        print("Saved to: ", savepath)

    for k, v in outdict.items():
        print(k, ':    ', v)
        
    return outdict




def replaceNone(dfthis, column, replace_with):
    """ replace Nones in this column with... 
    modifies in place.
    Retains the original called <column>_orig
    """

    # 1) save orig
    dfthis[f"{column}_orig"] = dfthis[column]
    # 2) replace
    tmp = dfthis[column].tolist()
    for i, x in enumerate(tmp):
        if x is None:
            tmp[i] = replace_with
    dfthis[column] = tmp
    return dfthis


def slice_by_row_label(df, colname, rowvalues, reset_index=True, 
    assert_exactly_one_each=False):
    """ Return a sliced dataframe, where rows are sliced
    to match the list of labels (rowvalues) for a column
    PARAMS:
    - colname, str, the column in which to compare lables
    - rowvalues, list of items (labels) to comapre. the 
    returned df will have rows exactly matching these values.
    NOTE: error if rowvalues contains value not in df
    NOTE: if a value occurs in multipel rows, it extracts all rows.
    - assert_exactly_one_each, if True, then each val in rowvalues
    matches exactly one and only one.
    EG:
    df = pd.DataFrame({'A': [5,6,3,4, 5], 'B': [1,2,3,5, 6]})
    list_of_values = [3, 6, 5]
    df.set_index('A').loc[list_of_values].reset_index()
    """

    if assert_exactly_one_each:
        for val in rowvalues:
            assert sum(df[colname]==val)==1
    dfout = df.set_index(colname).loc[rowvalues]
    if reset_index:
        dfout = dfout.reset_index()
    return dfout


def concat(list_df):
    """ Return concatenated df
    Adds a column ("idx_df_orig") which indicates for 
    each row which item in list_df it came from
    PARAMS:
    - list_df, list of df, with matching cols
    RETURNS:
    - df_all, concated (n rows increases), with index resetted
    """

    df_all = pd.concat(list_df).reset_index(drop=True)
    from pythonlib.tools.listtools import indices_into_original_list_after_concat

    indices = indices_into_original_list_after_concat(list_df)
    df_all["idx_df_orig"] = indices

    return df_all


def extract_with_levels_of_conjunction_vars(df, var, vars_others, levels_var = None, 
    n_min = 8, PRINT=False, lenient_allow_data_if_has_n_levels=None, DEBUG=False):
    """ Helper to extract dataframe (i) appending a new column
    with ocnjucntions of desired vars, and (ii) keeping only 
    levels of this vars (vars_others) that has at least n trials for 
    each level of a var of interest (var).
    PARAMS:
    - var, str, the variabiel you wisht ot use this dataset to compute 
    moudlation for.
    - vars_others, list of str, variables that conjucntion will define a 
    new goruping var.
    - levels_var, either list of items, for each must have at least n_min.
    if None, then takes the unique levels across entire df
    - n_min, min n trials desired for each level of var. will only keep
    conucntions of (vars_others) which have at least this many for each evel of
    var.
    - lenient_allow_data_if_has_n_levels, either None (ignore) or int, in which case
    is more leneint. keeps a given level of vars_others if it has >= this many
    levels of var which has >=n_min datapts. usually requires _all_ levels of vars_others
    to have >=n_min datapts.
    EG:
    - you wish to ask about shape moudlation for each combaiton of location and 
    size. then var = shape and vars_others = [location, size]
    RETURNS:
    - dataframe, with new column "vars_others", concated across all levels of varothers
    - dict, level_of_varothers:df_sub
    EXAMPLE:
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

        dftest = {
            'a':[1,2,3,1,2,3,1,2,3,1 ,2, 3],
            'b':[1,1,1,2,2,2,3,3,3, 3, 3, 3],
            'c':['a','a','a','b','b','b','c','c','c', 'c', 'c', 'c']
        }
        import pandas as pd
        dfthis = pd.DataFrame(dftest)

        extract_with_levels_of_conjunction_vars(dfthis, var="a", vars_others=['b', 'c'], n_min=2)
    """

    assert n_min is not None

    if DEBUG:
        PRINT = True

    # make a copy, becuase will have to append to this dataframe
    df = df.copy()

    # Want to use entier data for this site? or do separately for each level of a given
    # conjunction variable.
    if vars_others is None:
        # then place a dummy variable so that entire thing is one level
        vars_others = ["dummy_var"]
        assert "dummy_var" not in df.columns
        df.loc[:, "dummy_var"] = "IGNORE"
        REMOVE_DUMMY = True
    else:
        REMOVE_DUMMY = False

    # 1) Append conjucntions
    var_conj_of_others = "vars_others"
    df = append_col_with_grp_index(df, vars_others, new_col_name=var_conj_of_others, use_strings=False)
    
    # 2_ get for each conjunction of other variables
    levels_others = df[var_conj_of_others].unique().tolist()
    levels_others = sort_mixed_type(levels_others)
    # except TypeError as err:
    #     pass
    #     # print("not sorting (TypeError): ", levels_others)

    # 3) check each sub datfarme
    list_dfthis = []
    dict_dfthis = {} # level:df
    
    for lev in levels_others:

        # get data
        dfthis = df[df[var_conj_of_others] == lev]

        good = check_data_has_all_levels(dfthis, var, levels_var, n_min, 
            lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels,
            PRINT=PRINT)
        if DEBUG:
            print(lev, var, levels_var, good)
        if good:
            # keep
            list_dfthis.append(dfthis)
            dict_dfthis[lev] = dfthis

    if REMOVE_DUMMY:
        del df["dummy_var"]

    # merge
    if len(list_dfthis)==0:
        return pd.DataFrame([]), {}
    else:
        return pd.concat(list_dfthis).reset_index(drop=True), dict_dfthis

# only keep if this has all the levels
def check_data_has_all_levels(dfthis, var, levels_to_check=None, 
    n_trials_min=5, lenient_allow_data_if_has_n_levels=None,
    PRINT=False):
    """ Retuirns True if this dataframe as at least n_trials_min for each
    level of var
    PARAMS:
    - var, str, column in dfthis
    - levels_to_check, either list of categorival levles of var, or None (to get all of them)
    - n_trials_min, int
    - lenient_allow_data_if_has_n_levels, None or int. if None, then must have at least n_trials_min for 
    each level in levels_to_check. if int, then only need to have at least n_trials_min for 
    lenient_allow_data_if_has_n_levels many levels.
    """
        
    assert len(dfthis)>0
    # print(levels_to_check)

    if levels_to_check is None:
        levels_to_check = dfthis[var].unique().tolist()
    # else:
    #     # check that each level exists in dfthis
    #     for lev in levels_to_check:
    #         assert lev in dfthis[var].tolist(), "you passed in levels_to_check that doesnt exist"
    assert isinstance(levels_to_check, list)

    if lenient_allow_data_if_has_n_levels is None:
        STRICT = True
    else:
        STRICT = False

    list_n = []
    for lev in levels_to_check:
        dfthisthis = dfthis[dfthis[var]==lev]
        n = len(dfthisthis)
        list_n.append(n)
        # assert n>0
        if PRINT:
            print(lev, ' -- ', n)
        if STRICT:
            if n<n_trials_min:
                return False

    if STRICT:
        # Then you have suceeded iof havent failed aboev.
        return True
    else:
        passes = [nthis >= n_trials_min for nthis in list_n] #  whether enogh trials for each level of var: [True, False, ...]
        npass = sum(passes)
        good = npass >= lenient_allow_data_if_has_n_levels

        if PRINT:
            print("----")
            print(f"{good}, because {npass}/{lenient_allow_data_if_has_n_levels} passed... ({list_n}) >= {n_trials_min}")

        if npass >= lenient_allow_data_if_has_n_levels:
            return True
        else:
            return False

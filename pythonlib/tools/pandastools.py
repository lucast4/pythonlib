""" tools for use with pandas dataframes. also some stuff using python dicts and translating between that and dataframs"""
import pandas as pd
import numpy as np


#############################vvvv OBSOLETE - USE aggregGeneral
def aggreg(df, group, values, aggmethod=["mean","std"]):
    """
    get group means, for certain dimensions(values). 
    e.g., group = ["worker_model", "worker_dat"]
    e.g. values = ["score", "cond_is_same"]
    NOTE: will change name of balues filed, e.g. to score_mean.
    OBSOLETE - USE aggregGeneral]
    """
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
    return df.merge(df.apply(lambda x: F(x), axis=1).reset_index(), left_index=True, right_index=True).rename(columns={0:newcolname})

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
    """
    def F(x):
        """ True if has data for all values"""
        checks  = []
        for v in values_to_check:
            checks.append(v in x[colname].values)
        return all(checks)
    return df.groupby(group).filter(F)


def getCount(df, group, colname):
    """ return df grouped by group, and with one count value
    for each level in group, the name of that will be colname
    - colname must be a valid column from df
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
    # print(dfthis)
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

    df_new = pd.merge(df, dfthis, on=groupby)

    # remove dummy
    if remove_dummy:
        del df[dummyname]
        del dfthis[dummyname]
        del df_new[dummyname]

    if return_grouped_df:
        return df_new, dfthis
    else:
        return df_new


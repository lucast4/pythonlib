""" To work with Dlist, or list of Datasets.
This is distinct from a concatenated dset, in the here can
do seprate operations on each dset before concat.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pythonlib.tools.pandastools import applyFunctionToAllRows

def extract_strokes_monkey_vs_self(Dlist, GROUPING, GROUPING_LEVELS, remove_bad_trial=True):
    """
    populate new column with list of strokes for all other trials
    IN:
    - Dlist, list of D, where each D is a single dataset holding a single epoch (i.e)
    i.e., grouping level. doesnt' have to be, but that is how I did.
    - remove_bad_trial, False: useful for dry run, shows inds that would remove from dset (becuase they are
    rows that dont have paired strokes from all levels). if satisfied, can then rerun with True.
    NOTES:
    - each D in Dlist will have one new columns per grouping level, which will be all strokes
    across all D in Dlist which are from that grouping_level (escept your own trial's stroke)
    """

    # 1) Concat all datasets of interest, so can get all cross-parses
    Dall = concatDatasets(Dlist) # holds pool of strokes
    for D in Dlist:
        for group in GROUPING_LEVELS:
#             group = "straight" # which group's strokes to keep
#             Dthis = D # which one to modify
            
            def _get_strokes(D, task, group):
                """ returns all strokes for this task and this group, given a dataset D
                """
                dfthis = D.Dat[(D.Dat["unique_task_name"]==task) & (D.Dat[GROUPING]==group)]
                inds = dfthis.index.to_list()
                trialcodes = dfthis["trialcode"]
                strokes = dfthis["strokes_beh"]
                return trialcodes, strokes

            list_list_strokes = []
            for i in range(len(D.Dat)):
                task = D.Dat.iloc[i]["unique_task_name"]
                tcthis = D.Dat.iloc[i]["trialcode"]

                trialcodes, strokes = _get_strokes(Dall, task, group)

                # only keep strokes that are not from the current trial
                list_strokes = []
                for tc, s in zip(trialcodes, strokes):
                    if tcthis!=tc:
                        list_strokes.append(s)

                list_list_strokes.append(list_strokes)

            parsesname = f"strokes_beh_group_{group}"
            D.Dat[parsesname] = list_list_strokes
    
    # to confirm concat didnt modify
    for D in Dlist:
        D._check_consistency()
        
    # Only keep rows which have strokes across all models
    list_col_names = [f"strokes_beh_group_{group}" for group in GROUPING_LEVELS]
    for D in Dlist:
        def remove(D, i):
            # True if any of cols are empty
            x = [len(D.Dat.iloc[i][col])==0 for col in list_col_names]
            if any(x):
                return True
            else:
                return False
        inds_remove = [i for i in range(len(D.Dat)) if remove(D, i)]
        if remove_bad_trial:
            D.Dat = D.Dat.drop(inds_remove).reset_index(drop=True)
        else:
            print("for this dset:", D.identifier_string())
            print("would remove these inds:")
            print(inds_remove)
            print("out of ntrials", len(D.Dat))
        


def concatDatasets(Dlist):
    """ concatenates datasets in Dlist into a single dataset.
    Main goal is to concatenate D.Dat. WIll attempt to keep track of 
    Metadats, but have not confirmed that this is reliable yet.
    NOTE: Currently only does Dat correclt.y doesnt do metadat, etc.
    """
    from .dataset import Dataset
    # from pythonlib.dataset.dataset import Dataset

    Dnew = Dataset([])

    if True:
        # New, updates metadat.
        ct = 0
        dflist = []
        metadatlist = []
        for D in Dlist:
            
            if len(D.Metadats)>1:
                print("check that this is working.. only confied for if len is 1")
                assert False

            # add to metadat index
            df = D.Dat.copy()
            df["which_metadat_idx"] = df["which_metadat_idx"]+ct
            dflist.append(df)

            # Combine metadats
            metadatlist.extend([m for m in D.Metadats.values()])

            ct = ct+len(D.Metadats)
        Dnew.Dat = pd.concat(dflist)
        Dnew.Dat = Dnew.Dat.reset_index(drop=True)
        Dnew.Metadats = {i:m for i,m in enumerate(metadatlist)}
        print("Done!, new len of dataset", len(Dnew.Dat))
    else:
        # OLD: did not update metadat.
        dflist = [D.Dat for D in Dlist]
        Dnew.Dat = pd.concat(dflist)

        del Dnew.Dat["which_metadat_idx"] # remove for now, since metadats not carried over.

        Dnew.Dat = Dnew.Dat.reset_index(drop=True)

        print("Done!, new len of dataset", len(Dnew.Dat))
        # Dnew.Metadats = copy.deepcopy(self.Metadats)

    # Check consisitency
    Dnew._check_consistency()

    # do cleanup?
    Dnew._cleanup()
    
    return Dnew



def mergeTwoDatasets(D1, D2, on_="rowid"):
    """ merge D2 into D1. indices for
    D1 will not change. merges on 
    unique rowid (animal-trialcode).
    RETURNS:
    D1.Dat will be updated with new columns
    from D2.
    NOTE:
    will only use cols from D2 that are not in D1. will
    not check to make sure that any shared columns are indeed
    idnetical (thats up to you).
    """
    
    df1 = D1.Dat
    df2 = D2.Dat

    # only uses columns in D2 that dont exist in D1
    cols = df2.columns.difference(df1.columns)
    cols = cols.append(pd.Index([on_]))
    df = pd.merge(df1, df2[cols], how="outer", on=on_, validate="one_to_one")
    

    return df





def matchTwoDatasets(D1, D2):
    """ Given two Datasets, slices D2.Dat,
    second dataset, so that it only inclues trials
    prsent in D1. size D2 will be <= size D1. Uses
    animal-trialcode, which is unique identifier for 
    rows. 
    RETURNS
    - D2 will be modified. Nothings return
    """
    from pythonlib.tools.pandastools import filterPandas

    # rows are uniquely defined by animal and trialcode (put into new column)
    def F(x):
        return (x["animal"], x["trialcode"])
    if "rowid" not in D1.Dat.columns:
        D1.Dat = applyFunctionToAllRows(D1.Dat, F, "rowid")
    if "rowid" not in D2.Dat.columns:
        D2.Dat = applyFunctionToAllRows(D2.Dat, F, "rowid")

    print("original length")
    print(len(D2.Dat))

    F = {"rowid":list(set(D1.Dat["rowid"].values))}
#     inds = filterPandas(D2.Dat, F, return_indices=True)
    D2.filterPandas(F, return_ver="modify")

    print("new length")
    print(len(D2.Dat))


def find_common_tasks(Dlist, verbose=True):
    """
    Find task that are present across all datasets
    Includes fixed and random tasks. (all)
    OUT:
    - list of str, unique tasknames, sorted
    """

    # start with lsit from first dset
    list_tasks_all = Dlist[0].Dat["unique_task_name"].unique().tolist()

    # Iterate over all dsets and tasks.
    list_tasks_keep = []
    for task in list_tasks_all:
        # check that every D has it
        keep_task = True
        for D in Dlist:
            if sum(D.Dat["unique_task_name"]==task)==0:
                keep_task = False
                break
        if keep_task:
            list_tasks_keep.append(task)

    if verbose:
        print("Starting tasks", list_tasks_all)
        print("Ending tasks", list_tasks_keep)

    return sorted(list_tasks_keep)



def subsampleTrialsByFixedTask(Dlist, Ntasks=10):
    """ Return subset of Dataset, useful for debugging stuff, 
    Hierarchical sampling, first gets randoms tasks, but for each 
    task keeps all their trials. Makes sure these tasks are present
    across all datasets in Dlist.
    IN:
    - Ntasks = 10 # take random N tasks.
    OUT:
    - Dlist, pruned. (also modifies in place)
    """
    import random

    # Prune to subset of tasks, using common tasks
    list_tasks = find_common_tasks(Dlist, False)
    for t in list_tasks:
        print(t)
        
    list_tasks_keep = random.sample(list_tasks, Ntasks)

    # prune datasets
    for i, D in enumerate(Dlist):
        Dnew = D.filterPandas({"unique_task_name":list_tasks_keep}, "dataset")
        if len(Dnew.Dat)==0:
            assert False
        else:
            Dlist[i] = Dnew

    return Dlist



""" stores and operates on datasets"""
import pandas as pd
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.pandastools import applyFunctionToAllRows
import torch
import os
from pythonlib.tools.expttools import makeTimeStamp, findPath
from .analy_dlist import mergeTwoDatasets, matchTwoDatasets

def _checkPandasIndices(df):
    """ make sure indices are monotonic incresaing by 1.
    """
    tmp =  np.unique(np.diff(df.index))
    assert len(tmp)==1
    assert np.unique(np.diff(df.index)) ==1 

class Dataset(object):
    """ 
    """
    def __init__(self, inputs, append_list=None, reloading_saved_state=False):
        """
        Load saved datasets. 
        - inputs, is either:
        --- list of strings, where each string is
        --- Leave this empyt if want to load datsaet in other way
        full path dataset.
        --- list of tuples, where each tuple is (dat, metadat), where
        dat and metadat are both pd.Dataframes, where each wouyld be the object
        that would be loaded if list of paths.
        - append_list, dict, where each key:value pair will be 
        appendeded to datset as new column, value must be list 
        same length as path_list.
        - reloading_saved_state, then doesnt try to cleanup. Note, this doesnt mean
        reloading the probedat-->dataset, but is if reload after saving using Dataset.save...
        """


        self._reloading_saved_state = reloading_saved_state

        if len(inputs)>0:
            self._main_loader(inputs, append_list)
        # else:
        #     print("Did not load data!!!")


        self._possible_data = ["Dat", "BPL", "SF"]

        # Save common coordinate systems
        # [[xmin, ymin], [xmax, ymax]]
        self._edges = {
        "bpl":np.array([[0, -104], [104, 0]])
        }

        # Initialize things
        self._ParserPathBase = None
        
    def initialize_dataset(self, ver, params):
        """ main wrapper for loading datasets of all kinds, including
        monkey, model, etc.
        - ver, {"monkey", "generic"}
        - params, dict of params
        """

        if ver=="monkey":
            animal = params["animal"]
            expt = params["expt"]
            if "ver" in params.keys():
                ver = params["ver"]
            else:
                ver = "single"
            self.load_dataset_helper(animal, expt, ver)

        elif ver=="generic":
            # then give me behaviro and tasks
            strokes_list = params["strokes_list"]
            task_list = params["task_list"]
            self.Dat = pd.DataFrame({
                "strokes_beh":strokes_list,
                "Task":task_list})

    def load_dataset_helper(self, animal, expt, ver="single", rule=""):
        """ load a single dataset. 
        - ver, str
        --- "single", endures that there is one and only one.
        --- "mult", allows multiple. if want animal or expt to 
        - rule, some datasets defined by a "rule". To skip, pass in "" or None
        NOTE: for animal, expt, or rule, can pass in lists of strings. Must use ver="mult"
        """
        if rule is None:
            rule = ""

        if ver=="single":
            pathlist = self.find_dataset(animal, expt, assert_only_one=True, rule=rule)
            self._main_loader(pathlist, None, animal_expt_rule=[(animal, expt, rule)])

        elif ver=="mult":
            pathlist = []
            aer_list =[]
            if isinstance(animal, str):
                animal = [animal]
            if isinstance(expt, str):
                expt = [expt]
            if isinstance(rule, str):
                rule = [rule]
            for a in animal:
                for e in expt:
                    for r in rule:
                        pathlist.extend(self.find_dataset(a, e, True, rule=r))
                        aer_list.append((a,e,r))
            self._main_loader(pathlist, None, animal_expt_rule=aer_list)


    def _main_loader(self, inputs, append_list, animal_expt_rule=None):
        """ loading function, use this for all loading purposes
        - animal_expt_rule = [aer1, aer2, ..] length of inputs, where aer1 is like (a, e, r)
        -- only applies if input is paths
        """
        
        if self._reloading_saved_state:
            assert len(inputs)==1, "then must reload one saved state."
            self._reloading_saved_state_inputs = inputs

        if isinstance(inputs[0], str):
            # then loading datasets (given path)
            self._load_datasets(inputs, append_list, animal_expt_rule=animal_expt_rule)
        else:
            # then you passed in pandas dataframes directly.
            self._store_dataframes(inputs, append_list)
        if not self._reloading_saved_state:
            self._cleanup()
            self._check_consistency()


    def _store_dataframes(self, inputs, append_list):
        assert append_list is None, "not coded yet!"

        dat_list = []
        metadats = {}

        for i, inthis in enumerate(inputs):
            dat = inthis[0]
            m = inthis[1]

            dat["which_metadat_idx"] = i
            dat_list.append(dat)
            print("Loaded dataset, size:")
            print(len(dat))

            metadats[i] = m
            print("Loaded metadat:")
            print(m)

        self.Dat = pd.concat(dat_list, axis=0)
        if self._reloading_saved_state:
            assert len(metadats)==1
            self.Metadats = metadats[0]
        else:
            self.Metadats = metadats

        # reset index
        print("----")
        print("Resetting index")
        self.Dat = self.Dat.reset_index(drop=True)



    def _load_datasets(self, path_list, append_list, animal_expt_rule=None):
        # INPUTS:
        # - animal_expt_rule, expect this to be list of 3-tuples: (animal, expt, rule) or None.
        # --- len(list) must be length of path_list

        assert append_list is None, "not coded yet!"
        if animal_expt_rule is not None:
            if len(animal_expt_rule) != len(path_list):
                print(path_list)
                print(animal_expt_rule)
                assert False, "should be list of (a,e,r)"
        else:
            animal_expt_rule = [([], [], []) for _ in range(len(path_list))] # just pass in empty things.

        dat_list = []
        metadats = {}

        for i, (path, aer) in enumerate(zip(path_list, animal_expt_rule)):
            
            print("----------------")
            print(f"Currently loading: {path}")
            # Open dataset

            try:            
                with open(f"{path}/dat.pkl", "rb") as f:
                    dat = pickle.load(f)
            except FileNotFoundError:
                with open(f"{path}/Dat.pkl", "rb") as f:
                    dat = pickle.load(f)
            dat["which_metadat_idx"] = i
            dat_list.append(dat)


            # Open metadat
            try:
                with open(f"{path}/metadat.pkl", "rb") as f:
                    m = pickle.load(f)
            except FileNotFoundError:
                with open(f"{path}/Metadats.pkl", "rb") as f:
                    m = pickle.load(f)
            metadats[i] = m
            print("Loaded metadat:")
            print(m)

            metadats[i]["path"] = path
            metadats[i]["animal"] = aer[0]
            metadats[i]["expt"] = aer[1]
            metadats[i]["rule"] = aer[2]

        self.Dat = pd.concat(dat_list, axis=0)
        self.Metadats = metadats

        # reset index
        print("----")
        print("Resetting index")
        self.Dat = self.Dat.reset_index(drop=True)

    def sliceDataset(self, inds):
        """ given inds, slices self.Dat.
        inds are same as row num, will ensure so
        RETURNS:
        - self.Dat, modified, index reset
        """
        assert False, "not coded, didnt think necesayr"

    def copy(self):
        """ returns a copy. does this by extracting 
        data
        NOTE: copies over all metadat, regardless of whether all
        metadats are used.
        """
        import copy

        Dnew = Dataset([])

        Dnew.Dat = self.Dat.copy()
        if hasattr(self, "Metadats"):
            Dnew.Metadats = copy.deepcopy(self.Metadats)

        if hasattr(self, "BPL"):
            Dnew.BPL = copy.deepcopy(self.BPL)
        if hasattr(self, "SF"):
            Dnew.SF = self.SF.copy()
        if hasattr(self, "Parses"):
            Dnew.Parses = copy.deepcopy(self.Parses)

        return Dnew

    def is_finalized(self):
        """ Was metadata flagged as finalized by me? Returns True or False
        - If there are multiple datasets loaded simultanesuoy, returns True
        only if all are finalized
        """
        outs = []
        for k, v in self.Metadats.items():
            if "metadat_probedat" not in v.keys():
                outs.append(False)
            else:
                outs.append(v["metadat_probedat"]["finalized"])
        return all(outs)

    def is_equal_to(self, other):
        """ return True if self is equal to other, by passing certain diagnostics.
        Is not fool-proof, but works for things that matter.
        Checks:
        - id string
        - trialcodes match
        - columns match.
        """

        if self.identifier_string() != other.identifier_string():
            return False
        if not all(self.Dat.columns == other.Dat.columns):
            return False
        if not all(self.Dat["trialcode"] == self.Dat["trialcode"]):
            return False

        return True



    def filterByTask(self, filtdict, kind="any_shape_in_range", return_ver="dataset"):
        """ uses method in TaskGeneral class. 
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # Make a new column, indicating whether passes filter
        def F(x):
            return x["Task"].filter_by_shapes(filtdict, kind)
        self.Dat = applyFunctionToAllRows(self.Dat, F, "tmp")

        # filter
        return self.filterPandas({"tmp":[True]}, return_ver=return_ver)



    def filterPandas(self, filtdict, return_ver = "indices"):
        """
        RETURNS:
        - if return_ver is:
        --- "indices", then returns inds.(self.Dat not modifed,) 
        --- "modify", then modifies self.Dat, and returns Non
        --- "dataframe", then returns new dataframe, doesnt modify self.Dat
        --- "dataset", then copies and returns new dataset, without affecitng sefl.
        """
        from pythonlib.tools.pandastools import filterPandas
        # traintest = ["test"]
        # random_task = [False]
        # filtdict = {"traintest":traintest, "random_task":random_task}

        _checkPandasIndices(self.Dat)
        # print(f"Original length: {len(self.Dat)}")
        if return_ver=="indices":
            return filterPandas(self.Dat, filtdict, return_indices=True)
        elif return_ver=="modify":
            print("self.Dat modified!!")
            self.Dat = filterPandas(self.Dat, filtdict, return_indices=False)
        elif return_ver=="dataframe":
            return filterPandas(self.Dat, filtdict, return_indices=False)
        elif return_ver=="dataset":
            Dnew = self.copy()
            Dnew.Dat = filterPandas(self.Dat, filtdict, return_indices=False)
            return Dnew
        else:
            print(return_ver)
            assert False

    def subsetDataset(self, inds):
        """ returns a copy of self, with self.Dat only keeping inds
        INdices will be reset
        """
        Dnew = self.copy()
        Dnew.Dat = self.Dat.iloc[inds].reset_index(drop=True)
        return Dnew





    def findPandas(self, col, list_of_vals, reset_index=True):
        """ returns slice of self.Dat, where rows are matched one-to-one to list_of_vals.
        INPUTS:
        - col, str name of col
        - list_of_vals, list of values that pick out the rows.
        OUTPUT:
        - dfout, dataframe same length as list_of_vals
        """
        from pythonlib.tools.pandastools import findPandas
        return findPandas(self.Dat, col, list_of_vals, reset_index=reset_index)


    ############### UTILS
    def _strokes_kinds(self):
        """ returns list with names (strings) for the kinds of strokes that
        we have data for, 
        e.g., ["strokes_beh", "strokes_task", ...]
        """
        allkinds = ["strokes_beh", "strokes_task", "strokes_parse", "strokes_beh_splines", "parses"]
        return [k for k in allkinds if k in self.Dat.columns]


    def removeTrialsExistAcrossGroupingLevels(self, GROUPING, GROUPING_LEVELS):
        """ only keeps trials that exist across all grouping levels (conditions on character)
        """
        _checkPandasIndices(self.Dat)

        tasklist = self.Dat["character"].unique().tolist()
        inds_to_remove = []
        for task in tasklist:
            dfthis = self.Dat[self.Dat["character"]==task]

            # check that all grouping levels exist
            grouping_levels_exist = dfthis[GROUPING].unique().tolist()

            if not all([lev in grouping_levels_exist for lev in GROUPING_LEVELS]):
                # Remove all trials for this characters, since not all desired grouping
                # levels exist for this task.
                inds_to_remove.extend(dfthis.index.tolist())

        print("removing these inds (lack trials in at least one grioupuibg level):")
        print(inds_to_remove, "N=", len(inds_to_remove))
        self.Dat = self.Dat.drop(inds_to_remove, axis=0).reset_index(drop=True)


    def removeNans(self, columns=None):
        """ remove rows that have nans for the given columns
        INPUTS:
        - columns, list of column names. 
        --- if None, then uses all columns.
        --- if [], then doesnt do anything
        RETURNS:
        [modifies self.Dat]
        """

        print("--- Removing nans")
        print("start len:", len(self.Dat))

        print("- num names for each col")
        if columns is None:
            tmpcol = self.Dat.columns
        else:
            if len(columns)==0:
                print("not removing nans, since columns=[]")
                return
            tmpcol = columns
        for c in tmpcol:
            print(c, ':', np.sum(self.Dat[c].isna()))

        if columns is None:
            self.Dat = self.Dat.dropna()
        else:
            self.Dat = self.Dat.dropna(subset=columns)
        self.Dat = self.Dat.reset_index(drop=True)
        print("ending len:", len(self.Dat))


    def printNansPerCol(self, columns=None):
        """ PRINT, for each column how many nans
        """
        if columns is None:
            columns = self.Dat.columns
        # inds_bad = []
        for col in columns:
            # print(f"{col} : {sum(self.Dat[col].isna())}")
            print(f"{sum(self.Dat[col].isna())}, {col}")
            # inds_bad.extend(np.where(dfthis[col].isna())[0])
        print(f"Total num rows: {len(self.Dat)}")

    def removeOutlierRows(self, columns, prctile_min_max):
        """ remove rows that are outliers, based on percentiles for 
        columns of interest.
        INPUT:
        - columns, list of columns, if row is outline for _any_ column,
        will remove it entirely.
        - prctile_min_max, (2,) array, range=[0, 100], indicating min and max
        prctiles to use for calling somthing outlier. e.g., [1 99]
        RETURNS:
        - [modifies self.Dat]
        """
        print("--- Removing outliers")
        assert len(self.Dat)>0, "empty dat.."
        inds_bad = []
        for val in columns:
            # print("--")
            # print(self.Dat)
            # print(val)
            # print(self.Dat[val])
            # print(np.min(self.Dat[val]))
            # print(np.max(self.Dat[val]))

            limits = np.percentile(self.Dat[val], prctile_min_max)
            indsthis = (self.Dat[val]<limits[0]) | (self.Dat[val]>limits[1])
            inds_bad.extend(np.where(indsthis)[0])

        inds_bad = sorted(set(inds_bad))
        inds_good = [i for i in range(len(self.Dat)) if i not in inds_bad]
        print("starting len(self.Dat)", len(self.Dat))
        self.Dat = self.Dat.iloc[inds_good] 
        self.Dat = self.Dat.reset_index(drop=True)
        print("final len: ", len(self.Dat))

    def removeOutlierRowsTukey(self, col, niqr = 2, replace_with_nan=False):
        """ remove rows with outliers, based on iqr (tukey method)
        outlers are those either < 25th percentile - 1.5*iqr, or above...
        INPUTS:
        - replace_with_nan, then doesnt remove row, but instead replaces with nan.
        """

        from scipy.stats import iqr

        # Get the upper and lower limits
        x = self.Dat[col]
        lowerlim = np.percentile(x, [25])[0] - niqr*iqr(x)
        upperlim = np.percentile(x, [75])[0] + niqr*iqr(x)
        outliers = (x<lowerlim) | (x>upperlim)

        print("Lower and upper lim for outlier detection, feature=", col)
        print(lowerlim, upperlim)
        print("this many outliers / total")
        print(sum(outliers), "/", len(outliers))

        if False:
            fig, ax = plt.subplots(1,1, figsize=(10,5))
            x.hist(ax=ax, bins=30)
            ax.plot(x, np.ones_like(x), "o")
            ax.axvline(lowerlim)
            ax.axvline(upperlim)

        # x[outliers] = np.nan
        # print(outliers)
        inds_outliers = outliers[outliers==True].index
        if replace_with_nan:
            print("Replacing outliers with nan")
            print(inds_outliers)
            self.Dat[col] = self.Dat[col].mask(outliers)
        else:
            print("Removing outliers")
            print(inds_outliers)
            self.Dat = self.Dat.drop(inds_outliers).reset_index(drop=True)

    ############### TASKS
    def load_tasks_helper(self, reinitialize_taskobjgeneral=True, 
            redo_cleanup=True, convert_coords_to_abstract=False, 
            unique_names_post_Sep17=True):
        """ To load tasks in TaskGeneral class format.
        Must have already asved them beforehand
        - Uses default path
        - reinitialize_taskobjgeneral, then reinitializes, which is uiseful if code for
        general taskclas updates.
        - redo_cleanup, useful since some cleanup items require that tasks are already loaded.
        RETURN:
        - self.Dat has new column called Task
        NOTE: fails if any row is not found.
        """
        from pythonlib.tools.expttools import findPath
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        if convert_coords_to_abstract==False:
            assert reinitialize_taskobjgeneral==True, "otherwise convert_coords_to_abstract does nothing"
        assert len(self.Metadats)==1, "only works for single datasets"
        # Load tasks
        a = self.animals()
        e = self.expts()
        r = self.Metadats[0]["rule"]
        # print(self.Metadats[0])
        # assert False
        # print(r)

        if len(a)>1 or len(e)>1:
            assert False, "currently only works if single animal/ext dataset. can modify easily"

        # Find path, load Tasks
        if len(r)>0:
            sdir = f"/data2/analyses/database/TASKS_GENERAL/{a[0]}-{e[0]}-{r}-all"
        else:
            sdir = f"/data2/analyses/database/TASKS_GENERAL/{a[0]}-{e[0]}-all"

        pathlist = findPath(sdir, [], "Tasks", "pkl")
        if len(pathlist)!=1:
            print(pathlist)
            assert False
        # assert len(pathlist)==1
        with open(pathlist[0], "rb") as f:
            Tasks = pickle.load(f)

        # Align tasks with dataset
        def _get_task(Tasks, trialcode):
            tmp = [T["Task"] for T in Tasks if T["trialcode"]==trialcode]
            if len(tmp)==0:
                assert False, "no found"
            if len(tmp)>1:
                assert False, "too many"
            return tmp[0]

        def F(x):
            trialcode = x["trialcode"]
            T = _get_task(Tasks, trialcode)

            # Reinitalize tasks
            if reinitialize_taskobjgeneral:
                from pythonlib.drawmodel.taskgeneral import TaskClass
                taskobj = T.Params["input_params"]
                Tnew = TaskClass()
                # print(taskobj.Task)
                Tnew.initialize("ml2", taskobj, 
                    convert_coords_to_abstract=convert_coords_to_abstract)
                T = Tnew
            return T

        self.Dat = applyFunctionToAllRows(self.Dat, F, "Task")      
        print("added new column self.Dat[Task]")  

        if redo_cleanup:
            self._cleanup_using_tasks(unique_names_post_Sep17=unique_names_post_Sep17)

    def _task_hash(self, Task, random_task, use_objects = False, 
        original_ver_before_sep21=True):
        if use_objects:
            assert False, "not coded"

        if original_ver_before_sep21:
            # Original (actually 2nd to original) ver, before 9/17/21. Is good, but
            # problem is random tasks have tasknum in name, which means might not compare
            # well across tasknums that are actually same task. (since randomly generated
            # tasknums can be arbitrary). Solution, increase hash length, and dont have 
            # taskstrings
            ndigs = 6
            return Task.get_number_hash(ndigs=ndigs, include_taskstrings=True)
        else:
            if random_task:
                return Task.get_number_hash(ndigs=10, include_taskstrings=False,
                    include_taskcat_only=True, compact=True)

            else:
                # Thiese two lines are identical. This is idenitcal to before Sep 17
                return Task.get_number_hash(ndigs=6, include_taskstrings=True)
                # return self.Dat.iloc[ind]["unique_task_name"]


    def task_hash(self, ind, use_objects = False, 
        original_ver_before_sep21=True):
        """ Return task hash, differeing depending in smart way on onctext
        INPUT:
        - either ind or Task
        RETURNS: 
        - hashable identifier that should apply across all datsaets.
        --- if task is Fixed, then returns the unique_task_name
        --- if task is random, then returns a 10-digit hash.
        --- if use_objects, then overwrites others, and returns a FrozenSet with
        the objects (and affine features). This only works after ML2 using objects, so like
        July 2021 +. SO: only change is for random tasks. Fixed tasks have exact same name as 
        before.
        """

        Task = self.Dat.iloc[ind]["Task"]
        random_task = self.Dat.iloc[ind]["random_task"]
        return self._task_hash(Task, random_task, use_objects = use_objects, 
            original_ver_before_sep21=original_ver_before_sep21)



    ############# ASSERTIONS
    def _check_is_single_dataset(self):
        """ True if single, False, if you 
        contcatenated multipel datasets"""

        if len(self.Metadats)>1:
            return False
        else:
            return True

    def _check_consistency(self):
        """ sanity checks, should run this every time after fully load 
        a Dataset object and enter data
        """

        # No repeated unique trials
        assert len(self.Dat["trialcode"].unique().tolist()) == len(self.Dat), "unique trials occur in multipel rows.., did you concat dset to isetlf?"


        _checkPandasIndices(self.Dat)



    ############# CLEANUP
    def _cleanup(self, remove_dot_strokes=True):
        """ automaitcalyl clean up using default params
        Removes rows from self.Dat, and resets indices.
        """
        print("=== CLEANING UP self.Dat ===== ")
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # dfthis = self.Dat[self.Dat["random_task"]==False]
        # dfthis = self.Dat
        # tasklist = dfthis["unique_task_name"]
        # print([t for t in tasklist if "set-53" in t])
        # # print(sorted(dfthis["unique_task_name"]))
        # assert False

        ####### Remove online aborts
        # print("ORIGINAL: online abort values")
        # print(self.Dat["online_abort"].value_counts())
        # idx_good = self.Dat["online_abort"].isin([None])
        # self.Dat = self.Dat[idx_good]
        # print(f"kept {sum(idx_good)} out of {len(idx_good)}")
        # print("removed all cases with online abort not None")

        # reset 
        self.Dat = self.Dat.reset_index(drop=True)


        # Make sure strokes are all in format (N,3)
        def F(x):
            strokes_beh = x["strokes_beh"]
            # strokes_task = x["strokes_task"]
            for i, strok in enumerate(strokes_beh):
                if len(strok.shape)==1:
                    assert strok.shape==(3,), "is maybe (2,),without time var?"
                    strokes_beh[i] = strok.reshape(1,3)
                assert strok.shape[1]==3
            return strokes_beh
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strokes_beh")

        # Make sure expts is the correct name, becuase in sme cases
        # the automaticlaly extracted name is from the filenmae, which may be
        # incorrect
        if "metadat_probedat" in self.Metadats[0].keys():
            # Since only starting saving this later...
            def F(x):
                idx = x["which_metadat_idx"]
                return self.Metadats[idx]["metadat_probedat"]["expt"]
            self.Dat = applyFunctionToAllRows(self.Dat, F, "expt")


        ###### remove strokes that are empty or just one dot
        if remove_dot_strokes:
            strokeslist = self.Dat["strokes_beh"].values
            for i, strokes in enumerate(strokeslist):
                strokes = [s for s in strokes if len(s)>1]
                strokeslist[i] = strokes
            self.Dat["strokes_beh"] = strokeslist



        ####### Remove columns of useless info.
        cols_to_remove = ["probe", "feedback_ver_prms", "feedback_ver",
            "constraints_to_skip", "prototype", "saved_setnum", "tasknum", 
            "resynthesized", "resynthesized_path", "resynthesized_trial",
            "resynthesized_setnum", "resynthesized_setname", "modelscore", 
            "modelcomp", "hausdorff_positive", "circleness", "kind"]
        for col in cols_to_remove:
            if col in self.Dat.columns:
                del self.Dat[col]
        print("Deleted unused columns from self.Dat")

        ####### Construct useful params
        # For convenience, add column for whether task is train or test (from
        # monkey's perspective)
        def F(x):
            if x["taskgroup"] in ["train_fixed", "train_random", "T1"]:
                return "train"
            elif x["taskgroup"] in ["G2", "G3", "G4", "test_fixed", "test_random"]:
                return "test"
            elif x["taskgroup"] in ["undefined"]:
                return "undefined"
            else:
                print(x)
                assert False, "huh"
        print("applying monkey train test names")
        self.Dat = applyFunctionToAllRows(self.Dat, F, "monkey_train_or_test")

        # reset 
        print("resetting index")
        self.Dat = self.Dat.reset_index(drop=True)

        ### confirm that trialcodes are all unique (this is assumed for subsequent stuff)
        assert len(self.Dat["trialcode"])==len(self.Dat["trialcode"].unique().tolist()), "not sure why"

        ### reset tvals to the new earliest data
        self._get_standard_time()

        # Sort so is in increasing by date
        self.Dat = self.Dat.sort_values("tvalfake", axis=0).reset_index(drop=True)

        # Remove trial day since this might not be accurate anymore (if mixing datasets)
        if "trial_day" in self.Dat.columns:
            self.Dat = self.Dat.drop("trial_day", axis=1)

        # Replace epoch with rule, if that exists
        def F(x):
            idx = x["which_metadat_idx"]
            if self.Metadats[idx]["rule"]:
                return self.Metadats[idx]["rule"]
            else:
                return idx+1
        self.Dat = applyFunctionToAllRows(self.Dat, F, "epoch")

        # Add new column where abort is now True or False (since None was hjard to wrok with)
        def F(x):
            if x["online_abort"] is None:
                return False
            else:
                return True
        self.Dat = applyFunctionToAllRows(self.Dat, F, "aborted")
        self.Dat["online_abort_ver"] = self.Dat["online_abort"] # so not confused in previous code for Non
        self.Dat = self.Dat.drop("online_abort_ver", axis=1)

        if "Task" in self.Dat.columns:
            self._cleanup_using_tasks()

        # assign a "character" name to each task.
        self._cleanup_rename_characters()

        # Remove any trials that were online abort.
        self.Dat = self.Dat[self.Dat["aborted"]==False]
        print("Removed online aborts")

        # dates for summary , make sure assigned
        # for MD in self.Metadats.values():
        #     print(MD[metadat_probedat])
        list_dates = [MD["metadat_probedat"]["dates_for_summary"] for MD in self.Metadats.values()]
        list_dates = [str(d) for dates in list_dates for d in dates]
        # update "is in summary date column"
        def F(x):
            return str(x["date"]) in list_dates
        self.Dat = applyFunctionToAllRows(self.Dat, F, "insummarydates")
        print("Updated columns: insummarydates, using Metadats")

        ####
        self.Dat = self.Dat.reset_index(drop=True)

    def _cleanup_using_tasks(self, unique_names_post_Sep17=False):
        """ cleanups that require "Tasks" in columns,
        i.e,, extract presaved tasks
        - unique_names_post_Sep17, imprioving naming for random tasks. see hash function./
        """

        assert "Task" in self.Dat.columns

        # Replace unique name with new one, if tasks have been loaded
        if unique_names_post_Sep17:
            def F(x):
                return self._task_hash(Task=x["Task"], random_task=x["random_task"], 
                    original_ver_before_sep21=False)
        else:
            def F(x):
                return self._task_hash(Task=x["Task"], random_task=x["random_task"], 
                    original_ver_before_sep21=True)
            # Equivalent to line above.
            # def F(x):
            #     # return x["Task"].Params["input_params"].info_generate_unique_name()
        self.Dat = applyFunctionToAllRows(self.Dat, F, "unique_task_name")

        # task cartegories should include setnum
        def F(x):
            return x["Task"].get_category_setnum()
        self.Dat = applyFunctionToAllRows(self.Dat, F, "task_stagecategory")

        # rename charactesrs
        self._cleanup_rename_characters()

        # replace strokes_task with strokes from Task.
        def F(x):
            if str(x["date"])==str(210828) and x["expt"]=="gridlinecircle":
                # Then lollis were chunked into single strokes. but should
                # really be multipel strokes. 
                x["Task"]._split_strokes_large_jump()
            return x["Task"].Strokes    
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strokes_task")


    def _cleanup_rename_characters(self):
        """ makes column called "character" which is the unique name for fixed tasks 
        and the category name for random tasks."""

        def F(x):
            if x["random_task"]:
                # print(x["task_stagecategory"])
                # For random tasks, character is just the task category
                return x["task_stagecategory"]
            else:
                # For fixed tasks, it is the unique task name
                # print(x["unique_task_name"])
                return x["unique_task_name"]
        self.Dat = applyFunctionToAllRows(self.Dat, F, newcolname="character")



    # i.e. tvals from probedat might not be accurate if here combining multiple datsets
    def _get_standard_time(self, first_date=None):
        """ get real-world time for each trial, but using a 
        decimalized version that is not actual time, but is
        good for plotting timecourses. relative to first_date, so that
        for example, noon on first day is 1.5
        INPUTS:
        - first_date, if None, then uses the earlisest date over trials. Otherwise
        pass in int or string (format YYMMDD)
        RETURNS:
        - modifies self.Dat, removes tval col and adds new col called "tvalfake"
        """
        from pythonlib.tools.datetools import standardizeTime
        
        # Remove old tvals.
        if "tval" in self.Dat.columns:
            self.Dat = self.Dat.drop("tval", axis=1)

        # Add new tvals.
        if first_date is None:
            first_date = min(self.Dat["date"])

        first_date = str(first_date) + "-000000"
        def F(x):
            dt = x["datetime"]
            # return standardizeTime(dt, first_date, daystart=0.417, dayend=0.792)
            return standardizeTime(dt, first_date)
        self.Dat= applyFunctionToAllRows(self.Dat, F, "tvalfake")

        # tvalday is just the day (plus 0.6) useful for plotting
        def F(x):
            return np.floor(x["tvalfake"])+0.6
        self.Dat= applyFunctionToAllRows(self.Dat, F, "tvalday")



    def find_dataset(self, animal, expt, assert_only_one=True, rule=""):
        """ helper to locate path for presaved (from Probedat) dataset
        can then use this toload  it
        - assert_only_one, then asserts that one and only one path found.
        RETURNS:
        - list of strings, paths.
        """
        from pythonlib.tools.expttools import findPath

        # Collects across these dirs
        SDIR_LIST = ["/data2/analyses/database/", "/data2/analyses/database/BEH"]

        def _find(SDIR):
            pathlist = findPath(SDIR, [[animal, expt, rule]], "dat", ".pkl", True)
            return pathlist

        pathlist = []
        for SDIR in SDIR_LIST:
            pathlist.extend(_find(SDIR))

        # pathlist = findPath(SDIR, [[animal, expt]], "dat", ".pkl", True)
        
        if assert_only_one:
            assert len(pathlist)==1
            
        return pathlist


    def preprocessGood(self, ver="modeling", params=None, apply_to_recenter="all"):
        """ save common preprocess pipelines by name
        returns a ordered list, which allows for saving preprocess pipeline.
        - ver, string. uses this, unless params given, in which case uses parasm
        - params, list, order determines what steps to take. if None, then uses
        ver, otherwise usesparmas
        """
        if params is None:
            if ver=="modeling":
                # recenter tasks (so they are all similar spatial coords)
                self.recenter(method="each_beh_center", apply_to=apply_to_recenter)

                # interpolate beh (to reduce number of pts)
                self.interpolateStrokes()

                # subsample traisl in a stratified manner to amke sure good represnetaiton
                # of all variety of tasks.
                self.subsampleTrials()

                # Recompute task edges (i..e, bounding box)
                self.recomputeSketchpadEdges()

                params = ["recenter", "interp", "subsample", "spad_edges"]
            elif ver=="strokes":
                # interpolate beh (to reduce number of pts)
                self.interpolateStrokes()

                # subsample traisl in a stratified manner to amke sure good represnetaiton
                # of all variety of tasks.
                self.subsampleTrials()

                params = ["interp", "subsample"]

            else:
                print(ver)
                assert False, "not coded"
        else:
            for p in params:
                if p=="recenter":
                    self.recenter(method="each_beh_center", apply_to = apply_to_recenter) 
                elif p=="interp":
                    self.interpolateStrokes()
                elif p=="interp_spatial_int":
                    self.interpolateStrokesSpatial(strokes_ver = "strokes_beh", 
                        pts_or_interval="int")
                    self.interpolateStrokesSpatial(strokes_ver = "strokes_task", 
                        pts_or_interval="int")
                elif p=="subsample":
                    self.subsampleTrials()
                elif p=="spad_edges":
                    self.recomputeSketchpadEdges()
                elif p=="rescale_to_1":
                    self.rescaleStrokes()
                else:
                    print(p)
                    assert False, "dotn know this"
        return params


    ####################
    def animals(self, force_single=False):
        """ returns list of animals in this datsaet.
        - force_single, checks that only one animal.
        """
        x = sorted(list(set(self.Dat["animal"])))
        if force_single:
            assert len(x)==1, " multipel aniaml"
        return x

    def expts(self, force_single=False):
        """ returns list of expts
        """
        x =sorted(list(set(self.Dat["expt"])))
        if force_single:
            assert len(x)==1, " multipel expt"
        return x

    def rules(self, force_single=False):
        x = sorted(list(set([M["rule"] for M in self.Metadats.values()])))
        if force_single:
            assert len(x)==1, " multipel rules"
        return x

    def trial_tuple(self, indtrial):        
        """ identifier unique over all data (a, e, r, trialcode)
        trialcode = self.Dat.iloc[indtrial]["trialcode"]
        """
        trialcode = self.Dat.iloc[indtrial]["trialcode"]
        tp = (
            self.animals(force_single=True)[0],
            self.expts(force_single=True)[0],
            self.rules(force_single=True)[0],
            trialcode
            )

        return tp
        
    def identifier_string(self):
        """ string, useful for saving
        """
        
        a = "_".join(self.animals())
        e = "_".join(self.expts())
        r = "_".join(self.rules())

        return f"{a}_{e}_{r}"


    def get_sample_rate(self, ind):
        """ 
        ind is index in self.Dat
        """
        ind_md = self.Dat.iloc[ind]["which_metadat_idx"]
        fs = self.Metadats[ind_md]["filedata_params"]["sample_rate"]
        return fs
            
    def get_motor_stats(self, ind):
        """ 
        Simple - returns dict with all the motortiming and motorevent stats
        """
        out = {}
        for k, v in self.Dat.iloc[ind]["motorevents"].items():
            out[k]=v
        for k, v in self.Dat.iloc[ind]["motortiming"].items():
            out[k]=v

        # get stroke and gap lengths for each one
        from pythonlib.drawmodel.features import strokeDistances, gapDistances
        strokes_beh = self.Dat.iloc[ind]["strokes_beh"]
        out["dists_stroke"] = strokeDistances(strokes_beh)
        out["dists_gap"] = gapDistances(strokes_beh)

        return out


    ################# SPATIAL OPERATIONS
    def recenter(self, method, apply_to="both"):
        """ translate all coordiantes to recenter, based on 
        user defined method.
        - method, str,
        --- "monkey_origin", then where monkey finger began at 
        task onset (i..e, origin aka fix) is new (0,0)
        --- "screen_center", this is default, center of screen
        --- "each_stim_center", finds center of each stim, then
        uses this as (0,0). can differ for each trial.
        --- "each_beh_center", similar, but uses center based on touch 
        coordinates for behavior.
        - apply_to, to apply to "monkey", "stim" or "both", or "all"
        [NOTE, only "both" is currently working, since I have to think 
        about in what scenario the others would make sense]
        --- can also pass in as list of strings, with strokes names, eg
        ["strokes_beh", "strokes_task"] is identical to "both"
        """

        from pythonlib.tools.pandastools import applyFunctionToAllRows
        from pythonlib.tools.stroketools import getCenter, translateStrokes


        def _get_F(which_strokes):
            # which_strokes, either strokes_beh or strokes_task

            def F(x):
                if method=="monkey_origin":
                    xydelt = -x["origin"]
                elif method == "each_stim_center":
                    stim_center = np.array(getCenter(x["strokes_task"]))
                    xydelt = -stim_center
                elif method == "each_beh_center":
                    beh_center = np.array(getCenter(x["strokes_beh"]))
                    xydelt = -beh_center
                elif method == "screen_center":
                    assert False, "this is default, so no need to recenter"
                else:
                    print(method)
                    assert False, "not yet coded"

                strokes = x[which_strokes]

                if which_strokes=="parses":
                    # then this is list of strokes
                    return [translateStrokes(s, xydelt) for s in strokes]
                else:
                    return translateStrokes(strokes, xydelt)

            return F

        if isinstance(apply_to, list):
            which_strokes_list = apply_to
        else:
            if apply_to=="both":
                which_strokes_list = ["strokes_beh", "strokes_task"]
            elif apply_to=="all":
                which_strokes_list = self._strokes_kinds()
            else:
                print(apply_to)
                assert False, "not coded"

        for i, which_strokes in enumerate(which_strokes_list):
            dummy_name = f"tmp{i}"
            F = _get_F(which_strokes)
            self.Dat = applyFunctionToAllRows(self.Dat, F, dummy_name)

        # replace original with dummy names
        # dont do in above, since would overwrite what I am using for compting centers.
        for i, which_strokes in enumerate(which_strokes_list):
            dummy_name = f"tmp{i}"
            self.Dat[which_strokes] = self.Dat[dummy_name]
            del self.Dat[dummy_name]


    def interpolateStrokesSpatial(self, strokes_ver ="strokes_beh", pts_or_interval = "pts",
        npts=50, interval=10):
        """ interpolate in space, to npts pts
        (uniformly sampled)
        INPUTS:
        - pts_or_interval, which method
        --- "int", then keeps uniform spatial interavl; interval, pixels
        --- "pts", this num pts.
        """
        from ..tools.stroketools import strokesInterpolate2

        strokes_list = self.Dat[strokes_ver].values
        if pts_or_interval=="pts":
            N = ["npts", npts]
        elif pts_or_interval=="int":
            N = ["interval", interval]
        else:
            assert False

        for i, strokes in enumerate(strokes_list):
            strokes_list[i] = strokesInterpolate2(strokes, 
                N=N, base="space")

        self.Dat[strokes_ver] = strokes_list


    def interpolateStrokes(self, fs_new=20):
        """ interpolate strokes, still uniform time periods, 
        but generally subsampling. Autoamtically computes
        current sampling rate from time series. Only applies to 
        strokes_beh.
        - fs_new, new sampling rate (samp/sec)
        NOTE: modifies self.Dat in place.
        """
        from ..tools.stroketools import strokesInterpolate2

        fs_old = self.getFs()

        strokes_list = self.Dat["strokes_beh"].values
        for i, strokes in enumerate(strokes_list):
            strokes_list[i] = strokesInterpolate2(strokes, 
                N = ["fsnew", fs_new, fs_old])

        self.Dat["strokes_beh"] = strokes_list

    def rescaleStrokes(self, rescale_ver="stretch_to_1"):
        """ spatial rescaling of strokes
        Autoatmicalyl applies same rescaling to strokes_beh and strokes_task 
        by first concatenating, then applying, then unconcatnating
        """
        from pythonlib.tools.stroketools import rescaleStrokes

        strokes_beh_list = self.Dat["strokes_beh"]
        strokes_task_list = self.Dat["strokes_task"]

        strokes_beh_list_out = []
        strokes_task_list_out = []
        for strokes_beh, strokes_task in zip(strokes_beh_list, strokes_task_list):

            strokes_combined = strokes_beh + strokes_task
            strokes_combined = rescaleStrokes(strokes_combined, ver="stretch_to_1")

            strokes_beh_list_out.append(strokes_combined[:len(strokes_beh)])
            strokes_task_list_out.append(strokes_combined[len(strokes_beh):])

        self.Dat["strokes_beh"] = strokes_beh_list_out
        self.Dat["strokes_task"] = strokes_task_list_out



    def recomputeSketchpadEdgesAll(self, strokes_ver="strokes_beh"):
        """ 
        Gets smallest bounding box over all tasks.
        Will be affected if there are outliers (without "All", isntead
        will exlcude outliers)
        RETURNS:
        - in format [[-x, -y], [+x, +y]]. does not save in self
        """
        from pythonlib.drawmodel.image import get_sketchpad_edges_from_strokes
        strokes_list = list(self.Dat[strokes_ver].values)
        edges = get_sketchpad_edges_from_strokes(strokes_list)
        return edges


    def recomputeSketchpadEdges(self):
        """ recompute sketchpad size, since tasks are now recentered.
        - will take bounding box of all behavior (using percentiles, to
        ignore outliers). 
        RETURNS:
        - Modifies self.Metadat, replacing all metadats to have same sketchpad.
        """
        import matplotlib.pyplot as plt
        from ..tools.stroketools import getMinMaxVals

        # 1. Find, in practice, edges of behavior
        # For each trial, get its min, max, for x and y
        def F(x):
            # out = np.array([
            #     np.min([np.min(xx[:,0]) for xx in x["strokes_beh"]]),
            #     np.max([np.max(xx[:,0]) for xx in x["strokes_beh"]]),
            #     np.min([np.min(xx[:,1]) for xx in x["strokes_beh"]]),
            #     np.max([np.max(xx[:,1]) for xx in x["strokes_beh"]])
            # ])

            # return out
            return getMinMaxVals(x["strokes_beh"])

        from pythonlib.tools.pandastools import applyFunctionToAllRows
        DAT2 = applyFunctionToAllRows(self.Dat, F, "beh_edges")

        # 2. get edges that contain all tasks (or 99% of tasks for each dim)
        beh_edges = np.stack(DAT2["beh_edges"].values)

        # sketchpad_edges = np.array([
        #     [np.min(beh_edges[:,0]), np.min(beh_edges[2,:])], 
        #     [np.max(beh_edges[:,1]), np.max(beh_edges[3,:])]]) # [-x, -y, +x +y]
        sketchpad_edges = np.array([
            [np.percentile(beh_edges[:,0], [0.5])[0], np.percentile(beh_edges[:,2], [0.5])[0]], 
            [np.percentile(beh_edges[:,1], [99.5])[0], np.percentile(beh_edges[:,3], [99.5])[0]]]) # [-x, -y, +x +y]

        # 3. Replace sketchpad in metadat (use same for all tasks)
        for k, v in self.Metadats.items():
            v["sketchpad_edges"] = sketchpad_edges
            self.Metadats[k] = v

        print("Replaced self.Metadats with updated sketchpad...")
        print("[-x, -y; +x +y]")
        print(sketchpad_edges)
        # print(self.Metadats)        

        # # ==== PLOT 
        # beh_edges = np.stack(DAT2["beh_edges"].values)

        # for i in range(beh_edges.shape[1]):
        #     print([np.min(beh_edges[:, i]), np.max(beh_edges[:, i])])


        # ## PLOT DISTRIBITUSIONS OF EDGES (BEH) ACROSS ALL TASKS.
        # plt.figure()
        # plt.hist(beh_edges[:,0], 100)
        # plt.figure()
        # plt.hist(beh_edges[:,1], 100)
        # plt.figure()
        # plt.hist(beh_edges[:,2], 100)
        # plt.figure()
        # plt.hist(beh_edges[:,3], 100)


    ############# HELPERS
    def getFs(self):
        """ compute actual sampling rate for strokes_beh
        SUccesfulyl returns an fs across tasks only if:
        - is clearly periodic (regular samples)
        - same fs across all tirals
        """


        strokes_list = self.Dat["strokes_beh"].values
        periods_all = []
        for i, strokes in enumerate(strokes_list):
            for S in strokes:
                pers  = np.diff(S[:,2])
                if np.any(pers<0):
                    print("trial", i)
                periods_all.append(np.diff(S[:,2]))

        periods_all = np.concatenate(periods_all)
        print(periods_all.shape)

        delt = np.max(periods_all) - np.min(periods_all)
        print("max minus min periods over all data: ", delt)
        if ~np.isclose(delt, 0):
            plt.figure()
            plt.hist(periods_all)
            print(np.min(periods_all))
            print(np.max(periods_all))            
            assert False, "why some periods are different from others?"
        else:
            print("good! all periods are identical (across samples and trials:")
            print(periods_all[0])
            print("returning fs (1/per)")

        return 1/np.mean(periods_all)


    def subsampleTrials(self, n_rand_to_keep=400, n_fixed_to_keep=50, random_seed=3):
        """ subsample trials to make sure get good variety.
        Can run this before doing train test spkit
        - n_rand_to_keep, num random trials to keep for each 
        task_category, for random tasks. Generally try to keep this within
        order of magnitude of fixed tasks, but higher since these random.
        - n_fixed_to_keep, for each fixed task (given by unique_task_name), 
        how many to subsample to?
        """
        import random
        random.seed(random_seed)
        _checkPandasIndices(self.Dat)


        def get_inds_good(df, task_category_kind, n_sub):
            """ take n_sub num subsamples for each level in 
            task_category_kind. is stratified, so will take
            n separately for each expt/epoch/task combination.
            This ensure will sample across exts/epochs where behavior
            can be wquite different.
            - df, must not use reset_index() before passing in here.
            - task_category_kind, whether to take task_stagecategory (usually for
            random tasks) or unique_task_name (suitable only for fixed tasks)
            - n_sub, how mnay to take? if less than this, then will just take all.
            """

            assert task_category_kind in ["task_stagecategory", "unique_task_name"]
            tasks = set(df[task_category_kind])
            expts = set(df["expt"])
            epochs = set(df["epoch"])
            inds_good = []
            for ex in expts:
                for ep in epochs:
                    for t in tasks:
                        inds = list(df[
                            (df[task_category_kind]==t) & (df["expt"]==ex) & (df["epoch"]==ep)
                            ].index)
                        if len(inds)==0:
                            continue
                        print(f"for this task, expt, epoch, task, this many trials exist: {t, ex, ep, len(inds)}")
                        if len(inds)>n_sub:
                            inds = random.sample(inds, n_sub)
                        inds_good.extend(inds)
            return list(set(inds_good))


        # === 1) For random tasks, subsample so that they do not overwhelm the other tasks.
        print(f"RANDOM TASKS, before subsampling to {n_rand_to_keep}")
        df = self.Dat[self.Dat["random_task"]==True]
        print(df["task_stagecategory"].value_counts())

        inds_good = get_inds_good(df, "task_stagecategory", n_rand_to_keep)

        # convert to inds to remove
        inds_all = list(df.index)
        inds_to_remove = [i for i in inds_all if i not in inds_good]

        print("Removing this many inds")
        print(len(inds_to_remove))
        print("Original size of self.Dat")
        print(len(self.Dat))
        self.Dat = self.Dat.drop(inds_to_remove, axis=0)
        print("New size of self.Dat")
        print(len(self.Dat))

        # reset
        self.Dat = self.Dat.reset_index(drop=True)


        # === 2) For fixed tasks, only keep subsample of number, for each unique task
        print("==== FIXED TASKS")
        df = self.Dat[self.Dat["random_task"]==False]
        print(df["unique_task_name"].value_counts())
        print(df["unique_task_name"].value_counts().values)


        inds_good = get_inds_good(df, "unique_task_name", n_fixed_to_keep)

        # convert to inds to remove
        inds_all = list(df.index)
        inds_to_remove = [i for i in inds_all if i not in inds_good]

        print("Removing this many inds")
        print(len(inds_to_remove))
        print("Original size of self.Dat")
        print(len(self.Dat))
        self.Dat = self.Dat.drop(inds_to_remove, axis=0)
        print("New size of self.Dat")
        print(len(self.Dat))

        # reset
        self.Dat = self.Dat.reset_index(drop=True)


    def splitTrainTestMonkey(self, expt, val=0., epoch=None, seed=2):
        """ returns train and test tasks based on whether for monkey they
        were train (got reinforcing feedback, might have practiced many times) or
        test (no reinforcing feedback, not many chances)
        INPUT:
        - expt, str, the experiemnt name. will restrict tasks to just this expt.
        - epoch, int, the epoch number, 1, 2, ... Leave None is fine, unless there are
        multiple epohcs, in which case will throw error.
        - val, how many subsample of train to pull out as validation set. default is none, but 
        good amount, if ytou need, is like 0.05?
        RETURNS:
        - inds_train, inds_test, indices into self.Dat, as lists of ints.
        """
        import random
        random.seed(seed)

        _checkPandasIndices(self.Dat)

        # First, get just for this epoch
        if epoch is None:
            df = self.Dat[self.Dat["expt"] == expt]
            epochlist = set(df["epoch"])
            assert len(epochlist)==1, "there are multipel epochs, you must tell me which one you want"
        else:
            df = self.Dat[(self.Dat["expt"] == expt) & (self.Dat["epoch"]==epoch)]

        # Second, split into train and test
        inds_train = df.index[df["monkey_train_or_test"]=="train"].to_list()
        inds_test = df.index[df["monkey_train_or_test"]=="test"].to_list()

        if val>0.:
            nval = int(np.ceil(val*len(inds_train)))
            random.shuffle(inds_train)
            inds_val, inds_train = np.split(inds_train, [nval])
        else:
            inds_val = []

        print("Got this many train, val, test inds")
        print(len(inds_train), len(inds_val), len(inds_test))
        return inds_train, inds_val, inds_test



    def splitTrainTest(self, train_frac = 0.9, val_frac = 0.05, test_frac = 0.05,
        grouby = ["character", "expt", "epoch"]):
        """ Extract datasets (splits). does useful things, like making sure
        there is good variety across tasks, etc.
        By default, combines tasks across difficulties, inclulding monkey 
        "extrapolation" (G2, G3, G4) with "training". In this case, splits done by 
        random splits (but stratified, see below, grouby)
        INPUT:
        - train, val, and test fracs, self explanaatory
        - grouby, cross produce of these levels will be used to group data, before taking
        fractions. This ensures that train, test and val are evenly distributed over
        all grouping levels (or combos thereof). 
        RETURNS:
        - inds_train, inds_val, inds_test
        NOTES:
        - val and test will usualyl have slihgly mroe than input fraction, because of
        stratification method (if low sample size , then can try to force getting a test or val)
        - shuffles before returning,
        """
        print("TODO: Alternatively, can split similar to how monkey experienced")
        from pythonlib.tools.pandastools import filterPandas
        import random

        _checkPandasIndices(self.Dat)

        train_frac = 0.9
        val_frac = 0.05
        test_frac = 0.05
        assert np.isclose(train_frac + val_frac + test_frac, 1.)

        # Collect indices for train, val, and test
        inds_train = []
        inds_val = []
        inds_test = []
        for g in self.Dat.groupby(grouby):
            inds = list(g[1].index)
            random.shuffle(inds)
            n = len(inds)
            if n==1:
                inds_train.append(inds[0])
            elif n==2:
                inds_train.append(inds[0]) # use one for trainig
                if random.random()>0.5: # use the other for etier val or train
                    inds_val.append(inds[1])
                else:
                    inds_test.append(inds[1])
            elif n==3:
                inds_train.append(inds[0]) # spread evenly, one each.
                inds_val.append(inds[1])
                inds_test.append(inds[2])
            else:
                n_val = int(np.round(np.max([val_frac*n, 1])))
                n_test = int(np.round(np.max([test_frac*n, 1])))
                n_train = int(n - n_val - n_test)
                a, b, c = np.split(inds, [n_train, n_train+n_val])
                inds_train.extend(a)
                inds_val.extend(b)
                inds_test.extend(c)

        print("got this many (train, val, test)")
        print(len(inds_train), len(inds_val), len(inds_test))

        # shuflfe
        random.shuffle(inds_train)
        random.shuffle(inds_val)
        random.shuffle(inds_test)

        return inds_train, inds_val, inds_test




    ############# EXTRACT THINGS
    def _sf_get_path(self, strokes_ver, idx_metadat):
        """ for single dataset. tell me which datset with idx_metadat
        """
        sdir = f"{self.Metadats[idx_metadat]['path']}/stroke_feats"
        os.makedirs(sdir, exist_ok=True)
        path_sf = f"{sdir}/sf-{strokes_ver}.pkl"
        path_params = f"{sdir}/params-{strokes_ver}.pkl"
        return path_sf, path_params


    def _sf_get_path_combined(self):
        """ for SF that has been reloaded, across multiple datasets.
        will get automatic name for this combo. If already exist, then will
        not update. otherwise comes up with new.
        will be sved in self.SFparams["sdir"] """
        # save and make save folder for this combined dataset
        from pythonlib.tools.expttools import makeTimeStamp

        if "sdir" not in self.SFparams:
            ts = makeTimeStamp()
            sdir = "/data2/analyses/database/combined_strokfeats"
            a = sorted(set(self.SF["animal"]))
            b = sorted(set(self.SF["expt"]))
            c = sorted(set([p["strokes_ver"] for p in self.SFparams["params_each_original_sf"]]))
            tmp = "_".join(a) + "-" + "_".join(b) + "-" + "_".join(c)

            self.SFparams["sdir"] = f"{sdir}/{tmp}-{ts}"
            
            os.makedirs(self.SFparams['sdir'])
        return self.SFparams["sdir"]


    def sf_extract_and_save(self, strokes_ver = "strokes_beh", saveon=True, assign_into_self=False):
        """ extract and save stroke features dataframe, in save dir
        where datsaet is saved. 
        Will by default run this for all rows in self.Dat, can post-hoc assign
        back to Datasets if want to do further preprocessing
        INPUT:
        - strokes_ver, {"strokes_beh", "strokes_parse", "strokes_beh_splines"}, 
        determines which storkles to use. will seave spearately depending on this 
        flag. Note: if strokes_parse, trhen will choose a random single parse. that means
        each run could give diff ansers.
        - assign_into_self, then also assigns into self.SF and self.SFparams
        RETURNS:
        - SF, dataframe. does not modify self.
        NOTE: 
        - by default only allows you to run if this is a single dataset.
        - will delete Dataset from column before saving.
        """
        import os
        
        assert self._check_is_single_dataset(), "not allowed to run this with multipe datasets..."

        params = {}
        params["strokes_ver"] = strokes_ver

        if params["strokes_ver"]=="strokes_parse":
            self.parsesLoadAndExtract() # extract parses, need to have done bfeore
        if params["strokes_ver"]=="strokes_beh_splines":
            self.strokesToSplines(strokes_ver='strokes_beh', make_new_col=True) # convert to splines (beh)
        SF = self.flattenToStrokdat(strokes_ver=strokes_ver)


        # save
        if saveon:
            # delete things you dont want to save
            del SF["Dataset"]

            # Dir
            path_sf, path_params = self._sf_get_path(strokes_ver, 0)
            # SF
            SF.to_pickle(path_sf)
            print("Saved SF to:")
            print(path_sf)
            # Save params
            with open(path_params, "wb") as f:
                pickle.dump(params, f)

        if assign_into_self:
            self.SF = SF
            self.SFparams = {}

        return SF

    def sf_load_preextracted(self, strokes_ver_list = ["strokes_beh"]):
        """ wrapper to load multiple SFs.
        - strokes_ver_list, list of strings. (see _sf_load_preextracted)
        RETURNS:
        self.SF, all concatenated. will keep record of both which dataset (idx_metadata)
        and which SF it came from
        """

        SF = []
        PARAMS = []
        for sver in strokes_ver_list:
            sf, prms = self._sf_load_preextracted(sver)

            SF.extend(sf)
            PARAMS.extend(prms)

        # concat
        SF = pd.concat(SF)
        SF = SF.reset_index(drop=True)

        # assign
        self.SF = SF
        self.SFparams = {
            "params_each_original_sf":PARAMS}

        print("SF put into self.SF")



    def _sf_load_preextracted(self, strokes_ver="strokes_beh"):
        """ Loads SF (and parasm) that have been presaved. 
        must have run self.sf_extract_and_save() first.
        RETURNS:
        - list of SF, list of params
        NOTE: fine to run this even if multiple Datasets.
        """

        SF  = []
        PARAMS = []
        for idx in range(len(self.Metadats)):

            # Load presaved for this subdataset
            path_sf, path_params = self._sf_get_path(strokes_ver, idx) 

            sf = pd.read_pickle(path_sf)
            with open(path_params, "rb") as f:
                params = pickle.load(f)

            sf["idx_metadat"] = idx
            sf["strokes_ver"] = strokes_ver
            params["idx_metadat"] = idx
            params["path_sf"] = path_sf
            params["path_params"] = path_params
            params["path_params"] = path_params

            # concat
            PARAMS.append(params)
            SF.append(sf)

        # concat
        # SF = pd.concat(SF)
        # SF = SF.reset_index(drop=True)

        # save into self
        return SF, PARAMS


    def sf_preprocess_stroks(self, align_to_onset = True, min_stroke_length_percentile = 2, 
        min_stroke_length = 50, max_stroke_length_percentile = 99.5, centerize=False, rescale_ver=None):
        """ preprocess, filter, storkes, after already extracted into 
        self.SF
        """
        from ..drawmodel.sf import preprocessStroks


        params = {
            "align_to_onset":align_to_onset,
            "min_stroke_length_percentile":min_stroke_length_percentile,
            "min_stroke_length":min_stroke_length,
            "max_stroke_length_percentile":max_stroke_length_percentile,
            "centerize":centerize,
            "rescale_ver":rescale_ver
        }

        self.SF = preprocessStroks(self.SF, params)

        # Note down done preprocessing in params
        self.SFparams["params_preprocessing"] = params


    def sf_save_combined_sf(self):
        """ must have already extracted SF. saves current state, inclding
        preprocessing, etc
        """
        sdir = self._sf_get_path_combined()

        path_SF = f"{sdir}/SF.pkl"
        self.SF.to_pickle(path_SF)

        path_SF = f"{sdir}/SFparams.pkl"
        with open(path_SF, "wb") as f:
            pickle.dump(self.SFparams, f)
        print("saved SF and SFparams to:")
        print(sdir)


    def sf_load_combined_sf(self, animals, expts, strokes, tstamp = "*", take_most_recent=True):
        """
        Load SF (already concatted across mult datasets, and possible prerpocessed) NOTE:
        do all preprocessing etc BEFORE use sf_save_combined_sf(), which saves objecst that iwll
        load here.
        - take_most_recent, then will allow if have multiple found paths by taking most recent.
        otherwise will raise error if get multiple.
        """
        sdir = "/data2/analyses/database/combined_strokfeats"
        a = sorted(animals)
        b = sorted(expts)
        c = sorted(strokes)
        tmp = "*".join(a) + "-" + "*".join(b) + "-" + "*".join(c) + "*".join(tstamp)

        pathlist = findPath(sdir, [[tmp]], "SF", "pkl", True)

        if len(pathlist) == 0:
            print(pathlist)
            assert False, "did not find presaved data"
        
        if len(pathlist)>1:
            if not take_most_recent:
                print(pathlist)
                assert False, "found to omany..."
            else:
                print("Taking most recent path")
                pathlist = [pathlist[-1]]


        sdir = pathlist[0]

        print("==== Loading:")
        path = f"{sdir}/SF.pkl"
        with open(path, "rb") as f:
            self.SF = pickle.load(f)
            print("** Loaded into self.SF:")
            print(path)

        path = f"{sdir}/SFparams.pkl"
        with open(path, "rb") as f:
            self.SFparams = pickle.load(f)
            print("** Loaded into self.SFparams:")
            print(path)

        # save this path
        self.SFparams["path_sf_combined"] = sdir



    def sf_embedding_bysimilarity(self, rescale_strokes_ver = "stretch_to_1", distancever = "euclidian_diffs",
        npts_space = 50, Nbasis = 300, saveon=True):
        """copmputs embedding of strokes in similarity space (ie defined by basis set of strokes). basis set
        picked randomly
        INPUT:
        - rescale_strokes_ver, str, how/whether to rescale strokes before process. will not change self.SF
        - distance_ver, str, distance to use between strok-strok.
        - npts_space, num pts to interpolate in space (spatially uniform)
        - Nbasis, num random sampled trials to use for basesi.
        - saveon, then saves in same dir as self.SF
        RETURNS:
        - similarity_matrix, Nsamp x Nbasis, range from 0 to 1
        - idxs_stroklist_basis, indices in self.SF used as basis set (len Nbasis)
        """
        from pythonlib.drawmodel.sf import computeSimMatrix

        # Get sim matrix
        similarity_matrix, idxs_stroklist_basis = computeSimMatrix(self.SF, rescale_strokes_ver, 
                                                                   distancever, npts_space, Nbasis)
        # params_embedding = {
        #     "align_to_onset":align_to_onset,
        # }

        dat = {
            "similarity_matrix":similarity_matrix,
        }

        params = {
            "rescale_strokes_ver":rescale_strokes_ver,
            "distancever":distancever,
            "npts_space":npts_space,
            "Nbasis":Nbasis,
            "idxs_stroklist_basis":idxs_stroklist_basis
        }

        if saveon:
            sdir = self._sf_get_path_combined()
            ts = makeTimeStamp()
            sdirthis = f"{sdir}/embeddings/similarity-{rescale_strokes_ver}-{distancever}-N{Nbasis}-{ts}"
            os.makedirs(sdirthis)
            params["path_embeddings_similarity"] = sdirthis

            # sim matrix
            path = f"{sdirthis}/dat.pkl"
            with open(path, "wb") as f:
                pickle.dump(dat, f)

            # params
            path = f"{sdirthis}/params.pkl"
            with open(path, "wb") as f:
                pickle.dump(params, f)
            print("Saved to:")
            print(sdirthis)


        return similarity_matrix, idxs_stroklist_basis, params



    def flattenToStrokdat(self, keep_all_cols = True, strokes_ver="strokes_beh",
        cols_to_exclude = ["strokes_beh", "strokes_task", "strokes_parse", "strokes_beh_splines", "parses", "motor_program"]):
        """ flatten to pd dataframe where each row is one
        strok (np array)
        - keep_all_cols, then keeps all dataframe columns. otherwise
        will just have a reference (link) to dataset.
        - strokes_ver, can choose strokes_beh, strokes_task, strokes_parse...
        for the latter must have first exgtractd strokes parse.
        - cols_to_exclude, overwrites keep_all_cols.
        RETURNS:
        - Strokdat, pd dataframe. Includes reference to Probedat.
        NOTE: does not modify self.

        """
        _checkPandasIndices(self.Dat)

        if strokes_ver=="strokes_parse":
            if "strokes_parse" not in self.Dat.columns:
                print("extracting a single parse, since you haevent done that")
                self.parsesChooseSingle()
            # assert "strokes_parse" in self.Dat.columns, "need to first extract parases"
        if strokes_ver=="strokes_beh_splines":
            if "strokes_beh_splines" not in self.Dat.columns:
                print("Running conversion of strokes_beh to splines, saving as new col strokes_beh_splines")
                self.strokesToSplines()

        strokeslist = self.Dat[strokes_ver].values
        Strokdat = []
        columns = [col for col in self.Dat.columns if col not in cols_to_exclude]
        for i, strokes in enumerate(strokeslist):
            if i%500==0:
                print(i)

            row = self.Dat.iloc[i]

            for ii, strok in enumerate(strokes):
                Strokdat.append({
                    "stroknum":ii,
                    "strok":strok,
                    "row_in_Dataset":i,
                    "Dataset":self})

                if keep_all_cols:
                    for col in columns:
                        Strokdat[-1][col] = row[col]

        return pd.DataFrame(Strokdat)


    ############## SPATIAL
    def convertCoord(self, edges_out, strokes_ver="strokes_beh", edges_in=None):
        """ convert coordinate system.
        - edges_out and edges_in, in format [xmin, xmax; ymin, ymax], 2 x 2 array
        - edges_in, either give me, or if None wil compute automatically based on 
        smallest bounding box covering all tasks.
        RETURNS:
        - modified self.Dat[strokesver]
        - updates self._Sketchpad
        """
        from ..drawmodel.image import convCoordGeneral
        if edges_in is None:
            # compute edges
            edges_in = self.recomputeSketchpadEdgesAll().T
            print("Computed these for edges_in:")
            print(edges_in)

        strokes_list = self.Dat[strokes_ver]

        def F(strokes):
            return [convCoordGeneral(s, edges_in, edges_out) for s in strokes]

        strokes_list = [F(strokes) for strokes in strokes_list]

        self.Dat[strokes_ver] = strokes_list

        if not hasattr(self, "_Sketchpad"):
            self._Sketchpad = {}
        self._Sketchpad["edges"] = edges_out
        print("DONE converting coords")


    ############### IMAGES
    def strokes2image(self, strokes_ver, imageWH=105, inds=None, ploton=False):
        """
        convert strokes to image. 
        Automaticlaly finds boudning box to that all tasks in dataset are scaled same.
        INPUTS:
        - strokes_ver, which data to use.
        - imageWH, x and y dimension (square), 105 is standard for BPL.
        - inds, which indices in dataset to use. leave None to do all. must be None or
        list.
        - ploton, if true, then plots each image one by one. will not let you do this if inds 
        is too many.
        RETURNS:
        - img_list, list of np arrays holding images. formated so that plt.imshow() 
        prodices correct orietnation.
        """
        from pythonlib.drawmodel.image import strokes2image

        if inds is None:
            inds = range(len(self.Dat))
        else:
            assert isinstance(inds, list)

        # == coordinates of the sketchpad used by monkey
        # max width or hiegh (whichever one greater), num pixels for
        # half of page, so that sketchpad will be square with edges in 
        # both dimensions of : (-WH, WH)
        tmp = []
        for MD in self.Metadats.values():
            canvas_max_WH = np.max(np.abs(MD["sketchpad_edges"])) # smallest square that bounds all the stimuli
            tmp.append(canvas_max_WH)
        canvas_max_WH = np.max(tmp)
        print(f"canvas_max_WH atuomaticaly determiend to be: {canvas_max_WH}")

        # Which strokes dataset?
        strokes_list = list(self.Dat.iloc[inds][strokes_ver])

        if len(inds)>10 and ploton==True:
            print("Forcing turning off plots, too many trials")
            ploton=False

        image_list = [strokes2image(strokes, canvas_max_WH, imageWH, plot=ploton) for strokes in strokes_list]

        return image_list

    ############### MOTOR PROGRAMS (BPL)
    def _loadMotorPrograms(self, ver="strokes"):
        """ 
        Helper to load pre-saved motor programs.
        INPUTS:
        - ver, {"strokes", "parses"} which set of pre-saved programs
        to load.
        """
        from pythonlib.tools.expttools import findPath

        # Go thru each dataset, search for its presaved motor programs. Collect
        # into a list of dicts, one dict for each dataset.
        ndat = len(self.Metadats)
        MPdict_by_metadat = {}
        for i in range(ndat):

            # find paths
            if ver=="strokes":
                path = findPath(self.Metadats[i]["path"], [["infer_MPs_from_strokes"]],
                    "params",".pkl",True)
            elif ver=="parses":
                path = findPath(self.Metadats[i]["path"], [["MPs_for_parses"], ["infer_MPs_from_strokes"]],
                    "params",".pkl",True)
            else:
                print(ver)
                assert False, "not coded"
            
            # should be one and only one path
            if len(path)==0:
                assert False, "did not find"
            elif len(path)>1:
                assert False, "found > 1. you should deletethings."
            path = path[0]

            # lioad things from path
            _namelist = ["indices_list", "MPlist", "out_all", "params", "score_all"]
            MPdict = {}
            for name in _namelist:
                paththis = f"{path}/{name}.pkl"
                with open(paththis, "rb") as f:
                    MPdict[name] = pickle.load(f)

            # check consistency
            assert len(MPdict["MPlist"])==len(MPdict["indices_list"])
            assert len(MPdict["score_all"])==len(MPdict["indices_list"])
            assert len(MPdict["out_all"])==len(MPdict["out_all"])

            MPdict_by_metadat[i] = MPdict

        return MPdict_by_metadat

    def _extractMP(self, MPdict_by_metadat, idx_mdat, trialcode, expect_only_one_MP_per_trial=True,
        fail_if_None=True, parsenum=None):
        """
        Helper, to extract a single motor program for a signle trial (trialcode), and single
        metadat index (idx_mdat)
        INPUTS:
        - fail_if_None, then fialks if any trial doesnt find.
        NOTES:
        - expect_only_one_MP_per_trial, then each trial only got one parse. will return that MP 
        (instead of a list of MPs)
        - parsenum, {0, 1, ...}, parses, in order as in presaved parses. Leave as None if this is for
        strokes (so only one program).
        [SAME AS (not parses), but here each trial can have multiple paress. these are flattened in 
        MPdict_by_metadat. THis code pulls out a specific trialcode and parsenum"]
        RETURNS:
        - list of MPs. Usually is one, if there is only single parse. Unless 
        expect_only_one_MP_per_trial==True, then returns MP (not list). If doesnt fine, then will return None
        """

        # Get the desired dataset
        MPdict = MPdict_by_metadat[idx_mdat]
        trialcode_list = MPdict["indices_list"]
        MP_list = MPdict["MPlist"]

        if parsenum is None:
            # Pull out the desired trial        
            tmp = [m for m, t in zip(MP_list, trialcode_list) if t==trialcode]
        else:
            def _code(trialcode, parsenum):
                return f"tc_{trialcode}--p_{parsenum}"

            # Pull out the desired trial        
            tmp = [m for m, t in zip(MP_list, trialcode_list) if t==_code(trialcode, parsenum)]
        
        # === CHECK 
        if len(tmp)==0:
            if fail_if_None:
                print(f"Did not find MP for idx_metadat {idx_mdat}, trialcode {trialcode}")
                assert False
            return None
        elif len(tmp)>1:
            print(tmp)
            assert False, "not sure why, multiple inds have same trialcode? must be bug"

        if expect_only_one_MP_per_trial:
            assert len(tmp[0])==1
            tmp = tmp[0]

        return tmp[0]


    def bpl_reload_saved_state(self, path):
        """ reload BPL object, previously saved
        """

        # load BPL
        fthis = f"{path}/BPL.pkl"
        with open(fthis, "rb") as f:
            self.BPL = pickle.load(f)

        print(f"Reloaded BPL into self.BPL, from {fthis}")
    

    def bpl_extract_and_save_motorprograms(self, params_preprocess = ["recenter", "spad_edges"],
            sketchpad_edges =np.array([[-260., 260.],[-260., 260.]]), save_checkpoints = [100, ""]):
        """ save for this dataset.
        does some auto preprocessing first too
        """
        from pythonlib.bpl.strokesToProgram import infer_MPs_from_strokes

        assert len(self.Metadats)==1, "advised to only do this if you are working wtih one dataset."
        save_checkpoints[1] = self.Metadats[0]["path"]

        # Preprocess
        self.preprocessGood(ver="", params=params_preprocess)

        if True:
            # maybe this better, keep it stable caross all epxts
            sketchpad_edges = sketchpad_edges
        else:
            sketchpad_edges = self.Metadats[0]["sketchpad_edges"].T

        ## returns params in lists.
        strokeslist = self.Dat["strokes_beh"].values
        trialcodelist = self.Dat["trialcode"].values
        MPlist, scores = infer_MPs_from_strokes(strokeslist, trialcodelist, params_preprocess,
                                                sketchpad_edges, save_checkpoints=save_checkpoints)

        return MPlist, scores

    def bpl_extract_and_save_motorprograms_parses(self, params_preprocess = ["recenter", "spad_edges"],
            sketchpad_edges =np.array([[-260., 260.],[-260., 260.]]), save_checkpoints = [100, ""], 
            parses_good=False):

        from pythonlib.bpl.strokesToProgram import infer_MPs_from_strokes

        assert len(self.Metadats)==1, "advised to only do this if you are working wtih one dataset."
        assert len(save_checkpoints[1])==0, "code wil overwrite..."
        if parses_good:
            save_checkpoints[1] = f"{self.Metadats[0]['path']}/parses_good/MPs_for_parses"
        else:
            save_checkpoints[1] = f"{self.Metadats[0]['path']}/MPs_for_parses"


        # Preprocess
        self.preprocessGood(ver="", params=params_preprocess, apply_to_recenter=["strokes_beh", "strokes_task", "parses"])

        if True:
            # maybe this better, keep it stable caross all epxts
            sketchpad_edges = sketchpad_edges
        else:
            sketchpad_edges = self.Metadats[0]["sketchpad_edges"].T

        # Strategy for extraction - flatten all MPs (so ntrials x nparses), but keep information about parse num and trialcode.
        strokeslist = []
        trialcode_parsenum_list =[]
        for row in self.Dat.iterrows():
            parselist = row[1]["parses"]
            trialcode = row[1]["trialcode"]

            for i, parse in enumerate(parselist):

                strokeslist.append(parse)
                trialcode_parsenum_list.append(f"tc_{trialcode}--p_{i}")


        print(len(strokeslist))
        print(len(trialcode_parsenum_list))
        # print(strokeslist[0])
        # print(trialcode_parsenum_list[0])
        # assert False

        MPlist, scores = infer_MPs_from_strokes(strokeslist, trialcode_parsenum_list, params_preprocess,
                                                sketchpad_edges, save_checkpoints=save_checkpoints)

        return MPlist, scores

    def bpl_load_motorprograms(self):
        """ 
        loads best parse (MP) presaved, BPL.
        (see strokesToProgram) for extraction.
        RETURNS:
        - self.Dat adds a new column "motor_program", with a single ctype (motor program)
        - self.BPL, which stores things related to BPL params.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # def _loadMotorPrograms():
        #     """ 
        #     """
        #     from pythonlib.tools.expttools import findPath
            
        #     # Go thru each dataset, search for its presaved motor programs. Collect
        #     # into a list of dicts, one dict for each dataset.
        #     ndat = len(self.Metadats)
        #     MPdict_by_metadat = {}
        #     for i in range(ndat):
                
        #         # find paths
        #         path = findPath(self.Metadats[i]["path"], [["infer_MPs_from_strokes"]],
        #             "params",".pkl",True)
                
        #         # should be one and only one path
        #         if len(path)==0:
        #             print(self.Metadats[i]["path"])
        #             assert False, "did not find"
        #         elif len(path)>1:
        #             assert False, "found > 1. you should deletethings."
        #         path = path[0]
                
        #         # lioad things from path
        #         _namelist = ["indices_list", "MPlist", "out_all", "params", "score_all"]
        #         MPdict = {}
        #         for name in _namelist:
        #             paththis = f"{path}/{name}.pkl"
        #             with open(paththis, "rb") as f:
        #                 MPdict[name] = pickle.load(f)

        #         # check consistency
        #         assert len(MPdict["MPlist"])==len(MPdict["indices_list"])
        #         assert len(MPdict["score_all"])==len(MPdict["indices_list"])
        #         assert len(MPdict["out_all"])==len(MPdict["out_all"])
                
        #         MPdict_by_metadat[i] = MPdict

        #     return MPdict_by_metadat
        
        # def _extractMP(MPdict_by_metadat, idx_mdat, trialcode, expect_only_one_MP_per_trial=True,
        #     fail_if_None=True):
        #     """
        #     Helper, to extract a single motor program for a signle trial (trialcode), and single
        #     metadat index (idx_mdat)
        #     INPUTS:
        #     - fail_if_None, then fialks if any trial doesnt find.
        #     NOTES:
        #     - expect_only_one_MP_per_trial, then each trial only got one parse. will return that MP 
        #     (instead of a list of MPs)
        #     RETURNS:
        #     - list of MPs. Usually is one, if there is only single parse. Unless 
        #     expect_only_one_MP_per_trial==True, then returns MP (not list). If doesnt fine, then will return None
        #     """

        #     # Get the desired dataset
        #     MPdict = MPdict_by_metadat[idx_mdat]
        #     trialcode_list = MPdict["indices_list"]
        #     MP_list = MPdict["MPlist"]

        #     # Pull out the desired trial        
        #     tmp = [m for m, t in zip(MP_list, trialcode_list) if t==trialcode]
        #     if len(tmp)==0:
        #         if fail_if_None:
        #             print(f"Did not find MP for idx_metadat {idx_mdat}, trialcode {trialcode}")
        #             assert False
        #         return None
        #     elif len(tmp)>1:
        #         print(tmp)
        #         assert False, "not sure why, multiple inds have same trialcode? must be bug"
            
        #     if expect_only_one_MP_per_trial:
        #         assert len(tmp[0])==1
        #         tmp = tmp[0]

        #     return tmp[0]

        # check that havent dione before
        assert not hasattr(self, "BPL"), "self.BPL exists. I woint overwrite."
        assert "motor_program" not in self.Dat.columns

        MPdict_by_metadat = self._loadMotorPrograms()

        # save things
        self.BPL = {}
        self.BPL["params_MP_extraction"] = {k:MP["params"] for k, MP in MPdict_by_metadat.items()}

        def F(x):
            idx_mdat = x["which_metadat_idx"]
            trialcode = x["trialcode"]
            MP = self._extractMP(MPdict_by_metadat, idx_mdat, trialcode)
            assert MP is not None, "did not save this MP..."
            return MP

        self.Dat = applyFunctionToAllRows(self.Dat, F, 'motor_program')


    def bpl_refit_libraries_to_MPs(self, gb = ["animal", "expt", "epoch", "monkey_train_or_test"], 
        params_to_update=['kappa', 'rel_type_mixture', 'prim_type_mixture', 'spatial_hist'],
        params_to_update_using_entire_dataset=[],
        increment_idx=True):
        """ generate libraries refit to MPs extracted for this datsets.
        Flexible, in that can slice dataset in different ways before applying as 
        training data.
        INPUT:
        - gb, groupby list, for each group will compute a new fitted library.
        - params_to_update, which params to update. either a list of strings, or a
        list of lists. latter case, will iterate over all sub-lists, and for each, recompute a model.
        - params_to_update_using_entire_dataset, similar idea, but these will not take gb levels but will
        use entire data. will check that there is no overlap between this and params_to_update
        - increment_idx, True, then doesnt overwrite. false then always overwrites. see RETURNS/.
        RETURNS:
        self.BPL["refits"][idx]["libraries"] = list of libraries (each a dict)
        self.BPL["refits"][idx]["libraries_grpby"] = gb
        self.BPL["refits"][idx]["libraries_params"] = params_to_update
        where idx incremenets at each run.
        """
        from pythonlib.bpl.refitting import libRefitted

        if isinstance(params_to_update[0], list):
            assert False, "not yet coded"

        assert len([p for p in params_to_update_using_entire_dataset if p in params_to_update])==0, "no overlaps allowed!"
        assert len([p for p in params_to_update if p in params_to_update_using_entire_dataset])==0, "no overlaps allowed!"

        # for saving
        if "refits" not in self.BPL:
            self.BPL["refits"] = []

        libraries_list = []
        params_this = params_to_update
        for g in self.Dat.groupby(gb):
            print("-- refitting libarry to this group")
            print(g[0])
            print(len(g[1]))
            print("updating these params")
            print(params_this)
            
            MPlist_this = list(g[1]["motor_program"].values)
            lib_update = libRefitted(MPlist_this, params_to_update=params_this)
            
            # any updates needed , using the entire dataset?
            if len(params_to_update_using_entire_dataset)>0:
                print("** updating using entire dataset!! these parasm:")
                print(params_to_update_using_entire_dataset)
                MPlist_entire_dataset = list(self.Dat["motor_program"].values)
                lib_update = libRefitted(MPlist_entire_dataset, params_to_update=params_to_update_using_entire_dataset,
                    lib_update=lib_update)

            # save
            tmp = {"lib":lib_update, "ntrial":len(g[1])}
            # for k, v in zip(gb, g[0]):
            #     tmp[k] = v
            tmp["index_grp"] = g[0]
            tmp["params_to_update"] = params_this
            tmp["params_to_update_using_entire_dataset"] = params_to_update_using_entire_dataset
            libraries_list.append(tmp)

        out = {}
        out["libraries"] = libraries_list
        out["libraries_grpby"] = gb
        out["libraries_params"] = params_to_update

        self.BPL["refits"].append(out)

    def bpl_index_to_col_name(self, index, ver="strokes"):
            if isinstance(index, (list, tuple)):
                tmp = [str(i) for i in index]
            else:
                tmp = [index]
            if ver=="strokes":
                return f"bpl-{'-'.join(tmp)}"
            elif ver=="parses":
                return f"bpl-parses-{'-'.join(tmp)}"
            else:
                print(ver)
                assert False


    def _bpl_extract_refitted_libraries(self, lib_refit_index=0, libraries_to_apply_inds = None):
        """ Extract list of BPL libraries, they should have already been refitted,
        and are in self.BPL["refits"]
        INPUTS:
        - lib_refit_index, a particular grouping index. i.e.., self.BPL["refits"][lib_refit_index]
        - libraries_to_apply_inds, list of either ints, (in which case this indexes into 
        libraries_list), or list of tuples, in which case each tuple is a grp index (i.e., a level)
        RETURNS:
        - LibListThis, list of libraries.
        """

        # Extract list of libraries
        libraries_list = self.BPL["refits"][lib_refit_index]["libraries"]
        # grp_by = self.BPL["refits"][lib_refit_index]["libraries_grpby"]
        # for L in libraries_list:
            # print(L["index_grp"])

        # 2) get libraries
        # LibListThis = [L for L in libraries_list if L["index_grp"] in dsets_to_keep] # old version, same for dste and libary
        if libraries_to_apply_inds is None:
            LibListThis = libraries_list
        elif isinstance(libraries_to_apply_inds[0], int):
            LibListThis = [libraries_list[i] for i in libraries_to_apply_inds]
        elif isinstance(libraries_to_apply_inds[0], tuple):
            LibListThis = [L for L in libraries_list if L["index_grp"] in libraries_to_apply_inds] # old version, same for dste and libary
            print(libraries_to_apply_inds)
            print([L["index_grp"] for L in libraries_list])
            assert len(LibListThis)==len(libraries_to_apply_inds)
        else:
            print(LibListThis)
            assert False
        return LibListThis


    def bpl_score_trials_by_libraries(self, lib_refit_index=0, libraries_to_apply_inds = None,
        dsets_to_keep=None, scores_to_use = ["type"]):
        """
        Score each trial in self.Dat (their motor program) based on libraries, previuosl;yl
        extracted ands aved in self.BPL. returns a scalar score, log ll
        - libraries_to_apply_inds, list of either ints, (in which case this indexes into 
        libraries_list), or list of tuples, in which case each tuple is a grp index (i.e., a level)
        - dats_to_keep not implmeneted, to get subset of data.
        RETURNS:
        - self.Dat modified with  noew columns, name "bpl_"<index_levels as a string>
        NOTES:
        dsets_to_keep = [
            ('Pancho', 'lines5', 1, 'test'),
            ('Pancho', 'lines5', 1, 'train'),
            ('Pancho', 'lines5', 2, 'test'),
            ('Pancho', 'lines5', 2, 'train')
        ]

        """
        from ..bpl.strokesToProgram import scoreMPs

        # Extract list of libraries
        # libraries_list = self.BPL["refits"][lib_refit_index]["libraries"]
        # for L in libraries_list:
        #     print(L["index_grp"])

        # 2) get libraries
        # # LibListThis = [L for L in libraries_list if L["index_grp"] in dsets_to_keep] # old version, same for dste and libary
        # if libraries_to_apply_inds is None:
        #     LibListThis = libraries_list
        # elif isinstance(libraries_to_apply_inds[0], int):
        #     LibListThis = [libraries_list[i] for i in libraries_to_apply_inds]
        # elif isinstance(libraries_to_apply_inds[0], tuple):
        #     LibListThis = [L for L in libraries_list if L["index_grp"] in libraries_to_apply_inds] # old version, same for dste and libary
        #     print(libraries_to_apply_inds)
        #     print([L["index_grp"] for L in libraries_list])
        #     assert len(LibListThis)==len(libraries_to_apply_inds)
        # else:
        #     print(LibListThis)
        #     assert False
        LibListThis = self._bpl_extract_refitted_libraries(lib_refit_index, libraries_to_apply_inds)

        # [OPTIONAL - prune dataset, and models, before running]
        if dsets_to_keep is not None:
            assert False, "not done - need to first assign col with grpby levels, then filter"
            # first get column grouped
            def subsetDataset(dat_index):
                """
                - e..g, dat_index = ('Pancho', 'arc2', 1, 'test')
                RETURNS:
                - df, subset of D.Dat
                """
                i1 = self.Dat[grp_by[0]]==dat_index[0]
                i2 = self.Dat[grp_by[1]]==dat_index[1]
                i3 = self.Dat[grp_by[2]]==dat_index[2]
                i4 = self.Dat[grp_by[3]]==dat_index[3]
                df = self.Dat[i1 & i2 & i3 & i4]
                return df


            # 1) get Dataset
            tmp =[]
            for idx in dsets_to_keep:
                dfthis = subsetDataset(idx)
                tmp.append(dfthis)
                print(len(dfthis))

            DatThis = pd.concat(tmp)
            print("-- Final lenght")
            print(len(DatThis))
            print("-- Orig length:")
            print(len(self.Dat))


        # Add a column to self.Dat, for index grp
        grp_by = self.BPL["refits"][lib_refit_index]["libraries_grpby"]
        self.grouping_append_col(grp_by, "index_grp")
        print("-- SCORING, using this model:")
        
        if False:
            # === for each row, get score for each model
            def F(x, libthis):
                # Score a single row, given a library
                
                # data pt
                MP = x["motor_program"]

                # do score
            #     score = torch.tensor(scoreMPs([MP], lib=libthis, scores_to_use=scores_to_use))
                return torch.tensor(scoreMPs([MP], lib=libthis, scores_to_use=scores_to_use)[0]).numpy()
                
            for L in LibListThis:
                lib_index = L["index_grp"]
                libthis = L["lib"]
                print(lib_index)
                
                Fthis = lambda x: F(x, libthis)
                # newcol = f"bpl-{str(lib_index)}"
                newcol = self.bpl_index_to_col_name(lib_index)

                self.Dat = applyFunctionToAllRows(self.Dat, Fthis, newcol)
        else:
            # new way, extract as vector.
            for L in LibListThis:
                lib_index = L["index_grp"]
                libthis = L["lib"]
                print(lib_index)
                
                MPlist = list(self.Dat["motor_program"].values)
                scores = scoreMPs(MPlist, lib=libthis, scores_to_use=scores_to_use)
                scores = [s.numpy().squeeze() for s in scores]
                # newcol = f"bpl-{str(lib_index)}"
                newcol = self.bpl_index_to_col_name(lib_index)
                self.Dat[newcol] = scores

    def bpl_score_parses_by_libraries(self, lib_refit_index=0, libraries_to_apply_inds = None,
        dsets_to_keep=None, scores_to_use = ["type"]):
        """
        Score each parse in self.Dat["parases"], each row can have multipe parese.
        Does this based on libraries, first converting the parses to motor programs,
        returns a scalar score, log ll
        - libraries_to_apply_inds, list of either ints, (in which case this indexes into 
        libraries_list), or list of tuples, in which case each tuple is a grp index (i.e., a level)
        RETURNS:
        - self.Dat with new col: parses_scores_<index_levels as a string>
        """
        from ..bpl.strokesToProgram import scoreMPs

        LibListThis = self._bpl_extract_refitted_libraries(lib_refit_index, libraries_to_apply_inds)

        # new way, extract as vector.
        for L in LibListThis:
            lib_index = L["index_grp"]
            libthis = L["lib"]
            print(lib_index)
            
            # Go thru all parses
            def F(x):
                MPlist = list(x["parses_motor_programs"])
                scores = scoreMPs(MPlist, lib=libthis, scores_to_use=scores_to_use)
                scores = [s.numpy().squeeze() for s in scores]
                return scores
            newcol = self.bpl_index_to_col_name(lib_index, ver="parses")
            self.Dat = applyFunctionToAllRows(self.Dat, F, newcol)


    def bpl_score_parses_factorized(self, lib_refit_index=0, libraries_to_apply_inds = None,
        weights = {"k":1/3, "parts":1/3, "rel":1/3}):
        """
        RETURNS:
        - scores, factorized into features.
        """
        # weights = [1/3, 1/3, 1/3]
        from ..bpl.strokesToProgram import scoreMPs_factorized

        LibListThis = self._bpl_extract_refitted_libraries(lib_refit_index, libraries_to_apply_inds)
        _checkPandasIndices(self.Dat)

        # new way, extract as vector.
        out = []
        for L in LibListThis:
            lib_index = L["index_grp"]
            libthis = L["lib"]
            print(lib_index)
            
            # Go thru all parses
            for i, row in enumerate(self.Dat.iterrows()):
                MPlist = list(row[1]["parses_motor_programs"])
                scores_dict = scoreMPs_factorized(MPlist, lib=libthis, return_as_tensor=False)

                out.append({
                    "lib_index":lib_index,
                    "trialcode":row[1]["trialcode"],
                    "row_num":i,
                    "scores_features":scores_dict
                    })
        return out


    ################ PARSES GOOD [PARSER]
    def _get_parser_sdir(self, parse_params, pathbase=None, assume_only_one_parseset=True):
        """ Return cached pathbase (or caches, if this is run for first time)"""
        if self._ParserPathBase is None:
            if pathbase is None:
                assert len(self.Metadats)==1, "only run this if one dataset"
                self._ParserPathBase = f"{self.Metadats[0]['path']}/parser_good"
            else:
                # save
                self._ParserPathBase = pathbase

        pathlist = findPath(self._ParserPathBase, 
            [[f"ver_{parse_params['ver']}", f"quick_{parse_params['quick']}", f"{parse_params['savenote']}"]], 
            return_without_fname=True)
        if assume_only_one_parseset and len(pathlist)>1:
            
            assert False, "found >1 set, need to mnaually prune"
        pathdir = pathlist[0]
        return pathdir

    def parser_extract_parses_single(self, ind, ver,
            plot=False, quick=False,
            # nwalk_det = 10, max_nstroke=200, max_nwalk=50, N=1000, skip_parsing = False
            nwalk_det = 10, max_nstroke=350, max_nwalk=75, N=1200, skip_parsing = False
            ):
        """ extract Parser, along with parses, for this ind, using input params
        """
        from pythonlib.parser.parser import Parser

        if ver=="graphmod":
            # then modify graph. but do not enter the ground truth parses, since now 
            # graph will not be match original strokes.
            graph_mods = ["strokes_ends", "merge", "splitclose", "merge", 
                "merge_close_edges", "merge", 
                "loops_floating", "loops_cleanup", "loops_add"]
            graph_mods_params = [
                {"strokes":"StrokesInterp", "thresh":5},
                {"thresh":40},
                {"thresh":30}, # was 25 before 9/17/21
                {"thresh":40}, 
                {"thresh":25},
                {},
                {},
                {},
                {"THRESH":40}]
            do_input_by_hand=True
            do_input_by_hand_list = ["strokes"]
        elif ver=="nographmod":
            # then add input by hand.
            # Still enter stroke ends as new nodes.
            graph_mods = ["strokes_ends", "loops_floating", "loops_cleanup"]
            graph_mods_params = [
                {"strokes":"StrokesInterp", "thresh":5},
                {},
                {}]
            do_input_by_hand=True
            do_input_by_hand_list = ["strokes"]
        else:
            assert False, "not coded"

        Task = self.Dat.iloc[ind]["Task"]
        strokes_task = self.Dat.iloc[ind]["strokes_task"]

        # Initialize
        P = Parser()
        P.input_data(strokes_task)

        # Make skeleton
        didmod_all = P.make_graph_pipeline(graph_mods=graph_mods, 
            graph_mods_params=graph_mods_params)
        if skip_parsing and plot:
            P.plot_graph()

        if skip_parsing is False:
            ############ PARSES
            if do_input_by_hand:
                # if didmod_all is False:
                #     # Then still origianl pts in graph, can enter by hand.
                    
                for handver in do_input_by_hand_list:
                    if handver=="strokes":
                        # First, add by hand all the base parses.   
                        # Pass in parse based on strokes
                        strokes = P.StrokesInterp
                        # P.plot_graph()
                        # self.plotMultStrokes([P.StrokesInterp])
                        if ver=="nographmod":
                            # then looser
                            P.wrapper_input_parse(strokes, "strokes", 
                                apply_transform=False, note="strokes")
                        elif ver=="graphmod":
                            P.wrapper_input_parse(strokes, "strokes", 
                                apply_transform=False, note="strokes",
                                require_stroke_ends_on_nodes=False)
                        else:
                            assert False, "how to do?"

                    elif handver=="shapes":

                        ### PASS IN PARSE BASED ON OBJECTS
                        if all([x[0] is not None for x in Task.Shapes]):
                            # then you have shape info
                            assert False, "extract shapes, not yet coded"
                            # P.manually_input_parse_from_strokes(strokes, note="objects")
                            P.wrapper_input_parse(strokes, "strokes", note="objects")

                    elif handver=="chunks":
                        ### CHUNKS
                        Tml2 = self.Dat.iloc[ind]["Task"].Params["input_params"]
                        strokes_models = {}
                        chunk_models = Tml2.chunks_extract_models()
                        for mod, loc in chunk_models.items():
                            assert Tml2.chunks_extract(mod)==loc
                            los = Tml2.chunks_convert_to_strokes(loc, reorder=True)
                            strokes_models[mod] = los
                            for strokes in los:
                                # P.manually_input_parse_from_strokes(strokes, note="chunks")
                                P.wrapper_input_parse(strokes, "strokes", note="chunks")

                    else:
                        assert False
                # print(do_input_by_hand)
                # print(didmod_all)
                # P.summarize_parses()
                # assert False


            ## Walker to get parses.
            P.parse_pipeline(quick=quick, stroke_order_doesnt_matter=False, 
                direction_within_stroke_doesnt_matter=True,
                nwalk_det = nwalk_det, max_nstroke=max_nstroke, max_nwalk=max_nwalk, N=N)

            ############# FILTER PARSES

            # 1) Find parses where there are paths where repeat an edge (in the same path)
            # could repeat in same or diff direction
            P.filter_all_parses("any_path_has_redundant_edge")
            P.filter_all_parses("any_two_paths_have_common_edge")

            if plot:
                P.parses_fill_in_all_strokes()
                P.summarize_parses()

        return P

    def parser_extract_bestperms_wrapper(self, saveon = True, force_redo=False):
        """ RUN THIS wrapper. for each parser, for each base parse, extract the best-fitting permutation to beh.
        Is smart about checking if already done, and not overwriting old stuff. will
        not waste time urrnning if already done. 
        - goes thru all trials and runs all, along with appropriate logging, etc.
        REQUIRES:
        - extracted and loaded all parses into this dataset.
        NOTE: 
        This finsds the single best perm, works for
        parses that don't use chunks.
        """

        from pythonlib.tools.expttools import update_yaml_dict, load_yaml_config
        import os

        list_parse_params = self._Parser_list_parseparams # get this from loading.

        for parse_params in list_parse_params:
            graphmod = parse_params["ver"]

            # the actual cached save dir
            sdir = self._get_parser_sdir(parse_params)
            pathdict = f"{sdir}/dict_beh_aligned_permutations_log.yaml" # for logging
            
            for indtrial in range(len(self.Dat)):
                # Behavioral details for this trial.
                trialcode = self.Dat.iloc[indtrial]["trialcode"]
                behid = (self.identifier_string(), self.animals(force_single=True)[0], trialcode)
                taskname=self.Dat.iloc[indtrial]["unique_task_name"]
                fname = f"{sdir}/{taskname}-behalignedperms.pkl"

                # First check whether this already done
                if force_redo is False:
                    if True:
                        # New version, look for the file directly
                        if os.path.isfile(fname):
                            print("Skipping parser_extract_bestperms_wrapper, since alreayd done", fname)
                            continue
                    else:
                        # Old version, look for this file in the log. Not most accurate.
                        x = load_yaml_config(pathdict, make_if_no_exist=True)
                        if taskname in x.keys():
                            if behid in x[taskname]:
                                print("Skipping, since alreayd done", parse_params, indtrial, taskname)
                                continue

                # Run, get one best permutation for each base parse.
                P = self._parser_extract_bestperms(indtrial, graphmod)

                if saveon:
                    print("Saving parse with added beh-aligned perms", parse_params, indtrial, taskname)

                    # parser to save
                    # P = self.parser_list_of_parsers(indtrial, [f"parser_{graphmod}"])[0]

                    ## resave these parses...
                    with open(fname, "wb") as f:
                        pickle.dump(P, f)

                    ## log 
                    update_yaml_dict(pathdict, taskname, behid, allow_duplicates=False)

    def parser_get_parser_helper(self, indtrial, parse_params=None, load_from_disk=False):
        """ [GOOD] flexible getting of a single Parser instance
        IN:
        - parse_params
        --- if None, then there must only be one parser. otherwise fails
        OUT:
        - 
        NOTE:
        - fails if doesnt find only one.
        """
        
        if parse_params is None:
            parse_cols = [col for col in self.Dat.columns if "parser_" in col]
            assert len(parse_cols)==1
            col = parse_cols[0]
        else:
            graphmod = parse_params["ver"]
            col = f"parser_{graphmod}"
        list_P = self.parser_list_of_parsers(indtrial, [col])
        assert len(list_P)==1
        return list_P[0]

    def parser_prunedataset_bychunkrules(self, list_rule):
        """ Removes trials that dont have at least one self.ParsesBase that 
        for each rule in list_rule. Useful for model comparison, if a trial
        just is not parsable by a given model (e.g.,lolli model, but no possible
        lollis)
        INPUT:
        - list_rule, list of str, each trial must have all of these rules
        e.g.,         list_rule = ["lolli", "linetocircle", "circletoline"]
        RETURN:
        - D, a copy of self. Does not modify self.
        """

        # Get list of good trials
        list_inds_good = []
        for i in range(len(self.Dat)):
            P = self.parser_get_parser_helper(i)
            list_baseparse_rules = [p["rule"] for p in P.ParsesBase]
            x = [r in list_baseparse_rules for r in list_rule]
            
            if all(x):
                list_inds_good.append(i)

        # Return pruned dataset
        print("original length:", print(len(self.Dat)))
        Dcopy = self.subsetDataset(list_inds_good)
        print("new lenghth:", print(len(Dcopy.Dat)))
        return Dcopy
                    
            
    


    def parser_extract_chunkparses(self, indtrial, parse_params, saveon=True, 
        force_redo_best_fit=False, how_to_store="dict_in_baseparse", 
        reload_parser_each_time=False, DEBUG=False):
        """
        # Get chunked parses baed on rules, and get all permtuations, and get best-fitting 
        permutation to this trial's data.
        [MANYT HINGS, wrapper]. Must have already extracted parses, but otherwise dont have to have
        extracted the best-fit parses using the other code. Will not delete any previous parses. new parses
        are entered by updating old parses, not by appending. 
        - INPUT:
        - saveon, then overwrites file.
        - reload_parser_each_time, then reloads from disk this parser. Useful if doing in parallel multiple
        extractions over datasets, so that can make sure is loading the latest Parser.
        #TODO: split stuff operating on Tasks vs on Beh trial (top vs. bottom, see comments)
        """
        ################# MODIFY THESE
        list_rule = ["baseline", "linetocircle", "circletoline", "lolli"] # should not be ony the one for this expt
        # since want to score this beh across all models.
        expt = "gridlinecircle"
        # list_rule = self.rules(force_single=True) # otherwise does all rules multipel times...

        if DEBUG:
            # only run if it is this task
            taskname = self.Dat.iloc[indtrial]["unique_task_name"]
            if not taskname=="mixture2-ss-5_1-064566":
                print("[DEBUG] SKIPPED, becuase is: ", taskname)
                print(taskname=="mixture2-ss-5_1-064566")
                return

        ############################## STUFF THAT DOESNT DEPENDS ON THE BEHAVIOR
        # 1) Extract things for this trial
        Task = self.Dat.iloc[indtrial]["Task"]
        objects = Task.Shapes
        if reload_parser_each_time:
            taskname = self.Dat.iloc[indtrial]["unique_task_name"]
            sdir = self._get_parser_sdir(parse_params)
            fname = f"{sdir}/{taskname}-behalignedperms.pkl"
            print("** LOADING parser from disk", parse_params, indtrial, taskname, "from", fname)
            with open(fname, "rb") as f:
                P = pickle.load(f)
        else:
            P = self.parser_get_parser_helper(indtrial, parse_params)

        strokes_task = self.Dat.iloc[indtrial]["strokes_task"] # original, ignore chunking

        # 2) enter base parses for each rule into P.ParsesBase
        from pythonlib.drawmodel.chunks import find_chunks_wrapper
        # is_base_parse = True
        for rule in list_rule:
            tmp = P.findparses_bycommand("rule", {"list_rule":[rule]}, is_base_parse=True)
            if len(tmp)>0:
                print("SKIPPING finding base parse, since already exists: ", rule)
                continue

            # Get chunk info for this rule
            list_chunks, list_hier, list_fixed_order = find_chunks_wrapper(Task, expt, 
                rule, strokes_task)
            
            # Enter each one as a new base parse
            for chunks, hier, fixed_order in zip(list_chunks, list_hier, list_fixed_order):
                print("ENTERING NEW BASE PARSE:, ", chunks, hier, fixed_order, rule)
                params = {
                    "chunks":chunks,
                    "hier":hier,
                    "fixed_order":fixed_order,
                    "objects_before_chunking":objects,
                    "rule":rule
                }

                P.wrapper_input_parse(strokes_task, "strokes_plus_chunkinfo", params=params, is_base_parse=True)

        # 3) Get all permutations for each baseparse. overlap them by appending to a list keeping track of link between perm and baseparse
        list_indparse = range(len(P.ParsesBase))
        P._parses_reset_perm_of_list()
        for indparse in list_indparse:
            # Permutations already gotten?
            parsedict = P.extract_parses_wrapper(indparse, "dict", is_base_parse=True)
            if "done_permutations" in parsedict.keys():
                if parsedict["done_permutations"]==True:
                    print("Skipping permutation extraction for base parse: ", indparse)
                    continue    
            if False:
                # dont do this, doesnt inset, but appends. need to remove reduntant, then will lsoe some maybe
                list_list_p = P.get_all_permutations(indparse, is_base_parse=True, direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)
                print(len(list_list_p))
                # dont need, since did updating method, not appending
                P.parses_remove_redundant(stroke_order_doesnt_matter=False, direction_within_stroke_doesnt_matter=True)
            else:
                P.get_hier_permutations(indparse, update_not_append=True)
            parsedict["done_permutations"]=True


        ############################## THIS DEPENDS ON THE BEHAVIOR
        # 4) Out of all perms, get the best-fitting to behvaior, and not that down
        from pythonlib.drawmodel.strokedists import distscalarStrokes

        def _dist(strokes1, strokes2):
            #TODO: this distnace should be identical to the one used in likelihood. 
            #TODO: Deal with issue of repeated strokes.
            return distscalarStrokes(strokes1, strokes2, "dtw_segments")

        strokes_beh = self.Dat.iloc[indtrial]["strokes_beh"]
        trial_tuple = self.trial_tuple(indtrial) 
        # 3) for each, find best per`m
        # Method: search thru all perms
        # Cant use the lsa method since it assumes one to one map between strokes beh and task, but here there is 
        # complex hierarchy
        for rule in list_rule:
            list_indparse = P.findparses_bycommand("rule", {"list_rule":[rule]}, is_base_parse=True)
            for indparse in list_indparse:

                # If best fit already gotten, skip
                if force_redo_best_fit is False:
                    if "best_fit_perms" in P.ParsesBase[indparse].keys():
                        if trial_tuple in P.ParsesBase[indparse]["best_fit_perms"].keys():
                            print(f"Skipping best-fit-parse-getter, (since already gotten) for {rule, indparse, trial_tuple}")
                            continue
                    
                print("finding best parse for rule ", rule , "ind parse", indparse)
                # find all perms that are of this 
                inds = P.findparses_bycommand("permutation_of_v2", {"parsekind": "base", "ind":indparse}, is_base_parse=False)
                if len(inds)==0:
                    print("need to first get perms for baseparsr", (rule, indparse))
                    assert False
                # across these inds, find the one that best fits the behavior.
                if False:
                    # Crashes - memoyry?
                    list_strokes_parse = [P.extract_parses_wrapper(i, "strokes", is_base_parse=False) for i in inds]
                    list_d = [_dist(strokes_beh, strokes_p) for strokes_p in list_strokes_parse]
                else:
                    list_d = [_dist(strokes_beh, P.extract_parses_wrapper(i, "strokes", is_base_parse=False)) for i in inds]

                
        #         D.plotMultStrokes([strokes_beh, list_strokes_parse[0]])
        #         assert False, "check units"
                
                if len(list_d)==0:
                    print(inds)
                    print(indparse)
                    assert False
                ind_parse_best = inds[list_d.index(min(list_d))]
                
        #         print(ind_parse_best, inds, list_d)
        #         D.plotMultStrokes([strokes_beh])
        #         P.plot_parses([ind_parse_best])
                
        #         assert False
                
                # save note into this parse
                # trialcode = self.Dat.iloc[indtrial]["trialcode"]
                # note_suffix_tuple = (
                #     self.animals(force_single=True)[0],
                #     self.expts(force_single=True)[0],
                #     self.rules(force_single=True)[0],
                #     trialcode
                #     )
                ind_par = ("base", indparse)
                keyvals_update = {
                    "bestperm_beh_list":trial_tuple,
                    "bestperm_of_list":ind_par,
                    "bestperm_rule_list":rule
                }
                print("Found best parse at perm: ", ind_parse_best, " trial_tuple: ", trial_tuple)
                if how_to_store=="dict_in_baseparse":
                    # better? save directly as a dict item
                    if "best_fit_perms" not in P.ParsesBase[indparse].keys():
                        P.ParsesBase[indparse]["best_fit_perms"] = {}
                    print(f"placing into dict: self.BaseParses[indtrial][{trial_tuple}]")
                    P.ParsesBase[indparse]["best_fit_perms"][trial_tuple] = P.extract_parses_wrapper(ind_parse_best, "dict")
                elif how_to_store=="insert_into_Parses":
                    # add metadat to self.Parses
                    pardict = P.update_existing_parse(ind_parse_best, keyvals_update, append_keyvals=True,
                                            is_base_parse=False)
                else:
                    False

        if saveon:
            taskname = self.Dat.iloc[indtrial]["unique_task_name"]

            # parser to save
            # P = self.parser_list_of_parsers(indtrial, [f"parser_{graphmod}"])[0]
            sdir = self._get_parser_sdir(parse_params)

            ## resave these parses...
            fname = f"{sdir}/{taskname}-behalignedperms.pkl"
            print("** SAVING chunked parses", parse_params, indtrial, taskname, "to", fname)
            # if fname=="/data2/analyses/database/PARSES_GENERAL/gridlinecircle/ver_graphmod-quick_True/mixture2-ss-3_1-085968-behalignedperms.pkl":
            #     assert False
            with open(fname, "wb") as f:
                pickle.dump(P, f)



    def _parser_extract_bestperms(self, indtrial, ver, note_suffix=None, 
        how_to_store="insert_into_Parses", which_base_ver="Parses"):
        """
        For each baseparse, find the top permtuation that matches behavior on 
        this trial.
        INPUT
        - ind_trial, 0, 1, ...
        - ver, {'graphmod', 'nographmod'}
        - note_suffix, useful if want to tag this permutation as best based on a specific trial beh.
        NOTE:
        - can run this
        """
        from pythonlib.tools.stroketools import assignStrokenumFromTask

        # Get parser
        list_P = self.parser_list_of_parsers(indtrial, parser_names=[f"parser_{ver}"])
        assert len(list_P)==1
        P = list_P[0]

        assert P.Finalized==True, "easy fix if not - just finalize each parse one by one.."

        # Get all base parses
        if which_base_ver=="ParsesBase":
            assert False, "dont use this"
            # New version, seaprate atribute holding base parses
            inds_parses = list(range(len(P.ParsesBase)))
        elif which_base_ver=="Parses":
            inds_parses = P.findparses_bycommand("base_parses")
        else:
            assert False, ""
        
        # Get actual behavior    
        strokes_beh = self.Dat.iloc[indtrial]["strokes_beh"]
        trialcode = self.Dat.iloc[indtrial]["trialcode"]
        note_suffix = f"{self.animals(force_single=True)[0]}-{trialcode}" # uniquely id this beh
        note_suffix_tuple = (
            self.animals(force_single=True)[0],
            self.expts(force_single=True)[0],
            self.rules(force_single=True)[0],
            trialcode
            )
        for ind_par in inds_parses:

            # get parse in strokes
            strokes_baseparse = P.extract_parses_wrapper(ind_par, "strokes")
            list_p = P.extract_parses_wrapper(ind_par, "parser_stroke_class")

            if False:
                self.plotMultStrokes([strokes_beh, strokes_baseparse]);
                assert False, "confirm they overlap"

            # get task stroke inds in assigned order
            inds_baseparse_assigned = assignStrokenumFromTask(strokes_beh, 
                strokes_baseparse, ver="stroke_stroke_lsa_usealltask")

            # Get the permutation, to store as a new parse
            list_p_this = [list_p[i] for i in inds_baseparse_assigned]

            parsenew = {
                "strokes":None,
                "list_ps": list_p_this,
                "permutation_of": ind_par,
                "note":"bestperm" if note_suffix is None else f"bestperm-{note_suffix}",
                "bestperm_beh_list":[note_suffix_tuple],
                "bestperm_of_list":[ind_par]
            }
            
            if how_to_store=="insert_into_Parses":
                # Place into self.Parses. First tries to update. if cant find, then appends.
                # First, see if already exists
                keyvals_update = {
                    "bestperm_beh_list":note_suffix_tuple,
                    "bestperm_of_list":ind_par
                }
                success, ind = P.update_parse_with_input_parse(parsenew, keyvals_update, 
                    stroke_order_doesnt_matter=False,
                    direction_within_stroke_doesnt_matter=True,
                    append_keyvals=True
                    )
                if success:
                    print('Updated at ind', ind, ' with new best-fitting parse, for baseparse: ', ind_par)
                if not success:
                    # Then add it to parses
                    P.wrapper_input_parse(parsenew, ver="dict")
                    print('Added new best-fitting parse, for baseparse: ', ind_par)
            elif how_to_store=="dict_in_baseparse":
                assert False, "dont use this"
                # self.ParsesBase[this]["best_fit_perms"][trial_tuple]
                if "best_fit_perms" not in P.ParsesBase[ind_par].keys():
                    P.ParsesBase[ind_par]["best_fit_perms"] = {}
                P.ParsesBase[ind_par]["best_fit_perms"][note_suffix_tuple] = parsenew
            else:
                assert False
        P.parses_remove_redundant(stroke_order_doesnt_matter=False,
            direction_within_stroke_doesnt_matter=True)
        return P


    def parser_extract_and_save_parses(self, ver, quick=False, 
        saveon=True, savenote="", SDIR=None, save_using_trialcode=True):
        """ 
        INPUTS:
        - SDIR, if pass, then uses that. otherwise saves in this datasets folder
        - save_using_trialcode, if True, filename is trialcode. if false, filename is unqiue task id
        """
        # from pythonlib.tools.stroketools import getStrokePermutationsWrapper
        from pythonlib.tools.expttools import makeTimeStamp
        from pythonlib.parser.parser import Parser

        assert len(self.Metadats)==1, "only run this if one dataset"

        if SDIR is None:
            SDIR = f"{self._get_parser_sdir()}/parses/{makeTimeStamp()}"
        sdir = f"{SDIR}/ver_{ver}-quick_{quick}"
        if len(savenote)>0:
            sdir += savenote

        os.makedirs(sdir, exist_ok=True)
        print("Saving parses to : ", sdir) 
        
        # 1) Quickly save dict mapping trialcode --> unique task name
        if saveon:
            dict_trialcode_taskname = {}
            for i in range(len(self.Dat)):
                trialcode = self.Dat.iloc[i]["trialcode"]
                taskname = self.Dat.iloc[i]["unique_task_name"]
                dict_trialcode_taskname[trialcode]=taskname
            # save other things
            path = f"{sdir}/dict_trialcode_taskname-{self.identifier_string()}.pkl"
            with open(path, "wb") as f:
                pickle.dump(dict_trialcode_taskname, f)
            from pythonlib.tools.expttools import writeDictToYaml
            writeDictToYaml(dict_trialcode_taskname, 
                f"{sdir}/dict_trialcode_taskname-{self.identifier_string()}.yaml")

        # 2) Get parses.
        taskname_list = []
        for i in range(len(self.Dat)):
            trialcode = self.Dat.iloc[i]["trialcode"]
            taskname = self.Dat.iloc[i]["unique_task_name"]

            if save_using_trialcode:
                path = f"{sdir}/trialcode_{trialcode}.pkl"
            else:
                path = f"{sdir}/{taskname}.pkl"

            if taskname in taskname_list:
                print("** Skipping: ", self.identifier_string(), taskname, path, "[ALREADY DONE 1]")
                continue
            
            taskname_list.append(taskname)

            if os.path.isfile(path):
                print("** Skipping: ", self.identifier_string(), taskname, path, "[ALREADY DONE 2]")
                continue

            print("** Making graph and parsing: ", self.identifier_string(), taskname, path)

            try:
                P = self.parser_extract_parses_single(i, ver,
                    plot=False, quick=quick)
            except AssertionError as err:
                print(self.identifier_string())
                print(i)
                print(self.Dat.iloc[i]["trialcode"])
                print(taskname)
                print(err)
                print("[EXCEPTION] ")
                raise err
        # for row in self.Dat.iterrows():
        #     trialcode = row[1]["trialcode"]
        #     strokes_task = row[1]["strokes_task"]

        #     P = Parser()
        #     P.input_data(strokes_task)
        #     P.parse_pipeline(quick=quick)

            # save
            if saveon:
                try:
                    # out = {"Planner":}
                    with open(path, "wb") as f:
                        pickle.dump(P, f)
                except:
                    import time
                    time.sleep(5) # sleep in case this error is becuase file accessed by other instance of phythong
                    # out = {"Planner":}
                    with open(path, "wb") as f:
                        pickle.dump(P, f)
        print("length of dataset", len(self.Dat))
        print("GOT THESE TASKS:", taskname_list)
        print("** DONE!! parser_extract_and_save_parses", self.identifier_string())



    def parser_load_presaved_parses(self, list_parseparams, list_suffixes=None,
        finalize=True, pathbase=None, name_ver = "unique_task_name", 
        ensure_extracted_beh_aligned_parses=False):
        """ warpper to load multiple parses, each into a diffefenrt coilumn in self.Dat, with names based on 
        list_suffixes.
        INPUT:
        -list_parseparams, list of dicts, one for each set of parses to load, into individula Parser
        objects
            e.g.,:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":"fixed_True"},
                {"quick":True, "ver":"nographmod", "savenote":"fixed_True"}]
        - list_suffixes, names= ["graphmod", "nographmod"], expected in the name of the directory holding parses
        - pathbase, useful if there is common dir across expts, since they share tasks.
        - name_ver, convention that was used for naming the parses
        - ensure_extracted_beh_aligned_parses, check that all trials and parsers have already
        extracted the permutations aligned to behavior. if have not, then will do that extraction
        [NOTE: this might take time the first time!]
        """

        if list_suffixes is not None:
            assert len(list_suffixes)==len(list_parseparams)
        else:
            list_suffixes = [None for _ in range(len(list_parseparams))]

        for parse_params, suffix in zip(list_parseparams, list_suffixes):
            self._parser_load_presaved_parses(parse_params, suffix=suffix, finalize=finalize, 
                pathbase=pathbase, name_ver = name_ver)

        self._Parser_list_parseparams = list_parseparams

        if ensure_extracted_beh_aligned_parses:
            self.parser_extract_bestperms_wrapper()


    def _parser_load_presaved_parses(self, parse_params = None, 
        assume_only_one_parseset=True, suffix=None, finalize=True, pathbase=None,
        name_ver = "unique_task_name"):
        """ helper, to load a single set of parses, which are defined by their parse_params, which 
        indexes into a saved set of ata.
        - Fails if any tiral is faield to nbe found
        INPUT:
        - suffix, new columns will be called parser_{suffix}, useful if exrtracting multiple parsers, then merging
        """

        if parse_params is None:
            parse_params = {"quick":False, "ver":"graphmod", "savenote":""}

        pathdir = self._get_parser_sdir(parse_params, pathbase=pathbase)

        # pathlist = findPath(pathbase, [[f"ver_{parse_params['ver']}-quick_{parse_params['quick']}-{parse_params['savenote']}"]], return_without_fname=True)
        # if assume_only_one_parseset and len(pathlist)>1:
        #     assert False, "found >1 set, need to mnaually prune"
        # pathdir = pathlist[0]

        def _findrow(x):
            # first, try to find beh-aligned perms
            P = self.load_trial_data_wrapper(pathdir, x['trialcode'], 
                x['unique_task_name'],name_ver=name_ver, suffix="behalignedperms", return_none_if_nofile=True)
            if P is None:
                # if cant find, then load origianl parses
                P = self.load_trial_data_wrapper(pathdir, x['trialcode'], 
                    x['unique_task_name'],name_ver=name_ver)
            if finalize:
                P.finalize()

            if not hasattr(P, "ParsesBase"):
                # recent coded
                P.ParsesBase = []

            return P
            # paththis = f"{pathdir}/trialcode_{x['trialcode']}.pkl"
            # with open(paththis, "rb") as f:
            #     tmp = pickle.load(f)
            # return tmp

        colname = f"parser_{suffix}"
        print(f"*Loaded parses into {colname}")
        self.Dat = applyFunctionToAllRows(self.Dat, _findrow, colname)

        return pathdir


    def parser_list_of_parses(self, indtrial, kind="summary", 
        parser_names = ["parser_graphmod", "parser_nographmod"]):
        """ return list of parses for this trial, in summary format, 
        i.e., each parse is list of dicts.
        IN:
        - parser_names, col in self.Dat
        OUT:
        - list (nparse) of list (ntraj) of dicts.
        (is concatneated, if there are multiple parser_names)
        """
        list_of_p = []
        for name in parser_names:
            P = self.Dat.iloc[indtrial][name]
            plist = P.extract_all_parses_as_list(kind=kind)
            # print(plist[0])
            # assert False
            if kind=="summary":
                for i, p in enumerate(plist):
                    for pp in p:
                        pp["parser_name"] = name
                        pp["ind_in_parser"] = i
            list_of_p.extend(plist)
        return list_of_p


    def parser_list_of_parsers(self, indtrial, parser_names = ["parser_graphmod", "parser_nographmod"]):
        """ returns list of parsers
        e.g., [P1, P2]
        """
        list_of_parsers = [self.Dat.iloc[indtrial][pname] for pname in parser_names]
        return list_of_parsers


    def parser_load_precomputed_posteriors(self, model_id):
        """ Load list of posetrtior scores (scalars, one for each row of self.Dat), precomputed
        using the model (model_id)
        IN:
        - model_id, [expt, mod], where expt and mod are strings, e.g., ["pilot", "bent"]
        OUT:
        - assigns scores to a new column, called "parser_post_{expt}_{mod}"
        """
        SDIR = "/data2/analyses/main/model_comp/planner"
        sdir = f"{SDIR}/{model_id[0]}/dset_{self.identifier_string()}-vs-mod_{model_id[1]}"
        path = f"{sdir}/posterior_scores.pkl"
        with open(path, "rb") as f:
            scores = pickle.load(f)
        assert len(scores)==len(self.Dat)
        colname = f"parser_postscores_{model_id[0]}_{model_id[1]}"
        self.Dat[colname] = scores
        print("added scores to columns in self.Dat:", colname)


    def parser_flatten(self, parser_names = ["parser_graphmod", "parser_nographmod"]):
        """ take parsers (each a single object) from multiple columns (each a parser_name)
        and combines all and flattens into a new column called "parser_all_flat"
        - each parse represented as a list of p, where p is a dict holding all representations.
        - also holds reference back to origianl Parser, in case want to call it.
        NOTE:
        - also save location of origin in the first p for each list of p (i.e., each parse)
        """
        
        for pname in parser_names:
            assert pname in self.Dat.columns, "this parser no exist"

        list_list_parses = []
        for indtrial in range(len(self.Dat)):
            list_list_parses.append(self.parser_list_of_parses(indtrial, parser_names=parser_names))

            # Put origin in the first p for each parse.
            origin = self.Dat.iloc[indtrial]["origin"]
            for list_p in list_list_parses[-1]: # this trial
                list_p[0]["origin"] = origin

        self.Dat["parses_behmod"] = list_list_parses
        print("flattend parsers and moved to new col: ", "parses_behmod")
        print("Parsers:", parser_names)

    def parserflat_extract_strokes(self, ind):
        """ extract list of parses, where each parse is a strokes (i.e. a list of Nx2 array)
        """
        
        list_list_p = self.Dat.iloc[ind]["parses_behmod"]
        return [[p["traj"] for p in list_p] for list_p in list_list_p]

    def parser_names(self):
        """ gets names of all columns that start with 'parser_'
        OUT:
        - list of str.
        """
        return [col for col in self.Dat.columns if "parser_"==col[:7]]

    ############### GENERAL PURPOSE, FOR LOADING
    def load_trial_data_wrapper(self, pathdir, trialcode, unique_task_name, 
        return_none_if_nopkl=False, name_ver="trialcode", suffix="",
        return_none_if_nofile=False):
        """ looks for file. if not find, then looks in dict to find other trial that has 
        same taskname
        INPUT:
        - name_ver, str, which version to use, 
        --- {"trialcode", "unique_task_name"}
        - return_none_if_nofile, if True, then returns None if this file doesnt eixst. useful for 
        searching for files.
        NOTE:
        - Assumes that saved datapoint is shared across trials that have same task
        - Assumes that if find file iwth this trialcode/unique_task_name, then is correct
        """
        import os

        if name_ver=="trialcode":
            paththis = f"{pathdir}/trialcode_{trialcode}"
        elif name_ver=="unique_task_name":
            paththis = f"{pathdir}/{unique_task_name}"
        else:
            print(name_ver)
            assert False

        if len(suffix)>0:
            paththis += f"-{suffix}"

        paththis+=".pkl"

        if os.path.isfile(paththis):
            # then load it
            try:
                with open(paththis, "rb") as f:
                    tmp = pickle.load(f)
            except Exception as err:
                print(paththis)
                raise err
            return tmp
        else:
            # Failed - return none?
            if return_none_if_nofile:
                return None

            # Not allowed to fail..
            print("Cant find " , paththis)
            if return_none_if_nopkl:
                # if cant find file, then just return None
                return None
            else:
                # try to find it looking for same task, different trial. if fail, then raise error.
                # Read dict
                pathdict = f"{pathdir}/dict_trialcode_taskname.pkl"
                with open(pathdict, "rb") as f:
                    dict_trialcode_taskname = pickle.load(f)
                if dict_trialcode_taskname[trialcode]==unique_task_name:
                    list_tc_same_task = [k for k,v in dict_trialcode_taskname.items() if v==unique_task_name]
                    # try to load each of these trialcodes
                    for tc in list_tc_same_task:
                        tmp = self.load_trial_data_wrapper(pathdir, tc, unique_task_name, return_none_if_nopkl=True)
                        if tmp is not None:
                            return tmp
                    print(trialcode, unique_task_name)
                    assert False, "did not find task, both pkl, or thru other tasks with same taskname."
                else:
                    assert False, "task name changed since you saved this data?"
        assert False, "shouldnt have gotten here.."



    ################ PLANNER MODEL, like a simpler version of BPL
    def _get_planner_sdir(self):
        assert len(self.Metadats)==1, "only run this if one dataset"
        return f"{self.Metadats[0]['path']}/planner_model"

    def planner_extract_save_all_parses(self, permver = "all_orders_directions",
        num_max=1000):
        """ extract and save all permtuations for each task, using the input ver. 
        will save separately depending on ver (and run)
        """
        assert len(self.Metadats)==1, "only run this if one dataset"

        from pythonlib.tools.stroketools import getStrokePermutationsWrapper
        from pythonlib.tools.expttools import makeTimeStamp
        sdir = f"{self._get_planner_sdir()}/parses/{permver}-{makeTimeStamp()}"
        os.makedirs(sdir, exist_ok=True)
        print("Saving parses to : ", sdir) 

        for row in self.Dat.iterrows():
            trialcode = row[1]["trialcode"]
            strokes_task = row[1]["strokes_task"]

            strokes_task_perms = getStrokePermutationsWrapper(strokes_task, ver=permver, 
                num_max=num_max)

            # save
            path = f"{sdir}/trialcode_{trialcode}.pkl"
            with open(path, "wb") as f:
                pickle.dump(strokes_task_perms, f)

    def planner_load_presaved_parses(self, permver = "all_orders_directions",
        assume_only_one_parseset=True):
        """ Extract for each trial, parses, permtuations of task, pre-extracted using 
        planner_extract_save_all_parses
        INPUTS:
        - assume_only_one_parseset, then fails if finds >1 presaved ste.
        NOTES:
        - will fail unless can find one and only one for each trial in self
        """
        pathbase = f"{self._get_planner_sdir()}/parses"
        pathlist = findPath(pathbase, [[permver]], return_without_fname=True)
        if assume_only_one_parseset and len(pathlist)>1:
            assert False, "found >1 set, need to mnaually prune"
        pathdir = pathlist[0]

        def _findrow(x):
            paththis = f"{pathdir}/trialcode_{x['trialcode']}.pkl"
            with open(paththis, "rb") as f:
                tmp = pickle.load(f)
            return tmp

        self.Dat = applyFunctionToAllRows(self.Dat, _findrow, "parses_planner")

        return pathdir

    def planner_score_beh_against_all_parses(self, permver = "all_orders_directions",
        distfunc = "COMBO-euclidian-euclidian_diffs",
        confidence_ver = None):
        """ for each trial the beh is scored against all parses
        parses must have been presaved (indexed by permver)
        Here, saves these scores.
        INPUTS:
        - distfunc, passes into scoreAgainstBatch, could be string or func
        - confidence_ver, passes into scoreAgainstBatch (leave conf ver as none since easy to compute, and currently requires sorting.)

        """
        from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
        from pythonlib.drawmodel.strokedists import scoreAgainstBatch

        pathdir = self.planner_load_presaved_parses(permver=permver)

        sdir = f"{pathdir}/beh_parse_distances"
        os.makedirs(sdir, exist_ok=True)
        print("Saving distances to : ", sdir) 

        # save params
        params = {
        "permver":permver,
        "distfunc":distfunc,
        "confidence_ver":confidence_ver,
        }
        writeDictToYaml(params, f"{sdir}/params.yaml")

        for i, row in enumerate(self.Dat.iterrows()):
            trialcode = row[1]["trialcode"]
            strokes_beh = row[1]["strokes_beh"]
            # strokes_task = row[1]["strokes_task"]
            strokes_task_perms = row[1]["parses_planner"]

            print(len(strokes_beh))
            print(len(strokes_task_perms))
            print(len(strokes_task_perms[0]))

            distances_unsorted = scoreAgainstBatch(strokes_beh, strokes_task_perms, 
                    distfunc = distfunc, confidence_ver=confidence_ver, sort=False)[0]
            # leave conf ver as none since easy to compute, and currently requires sorting.
            # leave sort False, so that distances are always corresponding to presaved parses.

            # save
            path = f"{sdir}/trialcode_{trialcode}.pkl"
            with open(path, "wb") as f:
                pickle.dump(distances_unsorted, f)

            if i%200==0:
                print(i)

    def planner_score_parses(self, permver = "all_orders_directions",
        scorever="dist_traveled"):
        """ scores parses , eg by efficiency
        INPUTS
        - permver, is to index, to extract the correct parses.
        """
        from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
        from pythonlib.drawmodel.strokedists import scoreAgainstBatch

        pathdir = self.planner_load_presaved_parses(permver=permver)

        sdir = f"{pathdir}/parse_scores"
        os.makedirs(sdir, exist_ok=True)
        print("Saving parse scores to : ", sdir) 

        # save params
        params = {
        "scorever":scorever,
        }
        writeDictToYaml(params, f"{sdir}/params.yaml")

        # get function, given scorever
        if isinstance(scorever, str):
            if scorever=="dist_traveled":
                from pythonlib.drawmodel.features import computeDistTraveled
                scorefun = lambda strokes: computeDistTraveled(strokes, include_origin_to_first_stroke=False, 
                                include_transition_to_done=False)
            else:
                print(scorever)
                assert False, "not coded"
        else:
            # asusme is a function
            scorefun = scorever

        # compute
        for i, row in enumerate(self.Dat.iterrows()):
            trialcode = row[1]["trialcode"]
            strokes_task_perms = row[1]["parses_planner"]
            distances_unsorted = [scorefun(strokes) for strokes in strokes_task_perms]

            # save
            path = f"{sdir}/trialcode_{trialcode}.pkl"
            with open(path, "wb") as f:
                pickle.dump(distances_unsorted, f)

            if i%200==0:
                print(i)

    def planner_load_everything(self, permver = "all_orders_directions"):
        """ Loads (1) parses; (2) beh-parse distances; (3) parses scores
        """

        # Load parses
        pathdir = self.planner_load_presaved_parses(permver=permver)
        print("Loaded parses to column: parses_planner")

        # Load beh task scores
        sdir = f"{pathdir}/beh_parse_distances"
        def _findrow(x):
            paththis = f"{sdir}/trialcode_{x['trialcode']}.pkl"
            with open(paththis, "rb") as f:
                tmp = pickle.load(f)
            return tmp
        self.Dat = applyFunctionToAllRows(self.Dat, _findrow, "parses_planner_behtaskdist")
        print("Loaded beh-task dsitrances to column: parses_planner_behtaskdist")

        # Load task scores
        sdir = f"{pathdir}/parse_scores"
        def _findrow(x):
            paththis = f"{sdir}/trialcode_{x['trialcode']}.pkl"
            with open(paththis, "rb") as f:
                tmp = pickle.load(f)
            return tmp
        self.Dat = applyFunctionToAllRows(self.Dat, _findrow, "parses_planner_taskscore")
        print("Loaded parse scores to col: parses_planner_taskscore")


    def planner_assign_sequence_id(self):
        """ for a given task, number sequences as 0, 1, 2, ..., where sequence number is
        based on order in saved planner pareses. 
        e.g., all trials with 0 have the same discrete parse, nbased on closest match to planner parses.
        RETURNS:
        - new col in self.Dat, "sequence_id"
        NOTE:
        - if ties, then will asign to sequence with lower index.
        """

        self.planner_load_everything()
        def F(x):
            parses_planner_behtaskdist = x["parses_planner_behtaskdist"]
            # find index if minimum dist
            return np.argmin(parses_planner_behtaskdist)
        self.Dat = applyFunctionToAllRows(self.Dat, F, "sequence_id")


    def planner_plot_summary(self, ind_list, n_task_perms_plot="all"):
        """ plots all task perms, along with their distance to beh, and their efficneciy scores
        INPUTS:
        - ind, index into self.Dat
        """
        from pythonlib.drawmodel.efficiencycost import rank_beh_out_of_all_possible_sequences_quick

        ranklist = []
        conflist =[]
        sumscorelist =[]
        for ind in ind_list:
            strokes_beh = self.Dat["strokes_beh"].values[ind]
            strokes_task_perms = self.Dat["parses_planner"].values[ind]
            beh_task_distances = self.Dat["parses_planner_behtaskdist"].values[ind]
            task_inefficiency = self.Dat["parses_planner_taskscore"].values[ind]

            rank, conf, sumscore, strokest = rank_beh_out_of_all_possible_sequences_quick(
                strokes_beh, strokes_task_perms, beh_task_distances, 
                task_inefficiency, plot_strokes=True, 
                plot_rank_distribution=True, plot_n_strokes=n_task_perms_plot)

            ranklist.append(rank)
            conflist.append(conf)
            sumscorelist.append(sumscore)

        fig, axes = plt.subplots(1,3, figsize=(9,3))
        x = range(len(ind_list))

        ax = axes.flatten()[0]
        ax.plot(x, ranklist, "-ok", label="rank")
        ax.axhline(0)
        ax.set_title("rank")

        ax = axes.flatten()[1]
        ax.plot(x, conflist, "-ok")
        ax.axhline(0)
        ax.set_title("confidence")

        ax = axes.flatten()[2]
        ax.plot(x, sumscorelist, "-ok")
        ax.axhline(1)
        ax.set_title("summary score")


    


    ################ PARSES
    # Working with parses, which are pre-extracted and saved int he same directocry as datasets.
    # See ..drawmodel.parsing

    def _loadParses(self, parses_good=False):
        """ load all parases in bulk.
        Assumes that they are arleady preprocessed.
        - Loads one parse set for each dataset, and then combines them.
        - parses_good, then finds the /parses_good dolfer.
        """
        PARSES = {}
        for ind in self.Metadats.keys():
            
            path = self.Metadats[ind]["path"]
            if parses_good:
                path = path + "/" + "parses_good"
            path_parses = f"{path}/parses.pkl"
            path_parses_params =  f"{path}/parses_params.pkl"

            with open(path_parses, "rb") as f:
                parses = pickle.load(f)
            with open(path_parses_params, "rb") as f:
                parses_params = pickle.load(f)
                
            PARSES[ind] = parses
        return PARSES

    def parsesLoadAndExtract(self, fail_if_empty=True, print_summary=True, parses_good = False):
        """ does heavy lifting of extracting parses, assigning back into each dataset row.
        does this by matching by unique task name.
        - if any parse is not found, then it will be empty.
        - fail_if_empty, then asserts that each row has found parses.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        _checkPandasIndices(self.Dat)

        def _extractParses(PARSES, ind_metadat, taskname):
            """ helper, to get a single list of parses for this task"""
            df = PARSES[ind_metadat]
            dfthis = df[df["unique_task_name"]==taskname]
            if len(dfthis)==0:
                print("did not find parses, for the following, returning []!!")
                print(taskname)
                print(ind_metadat)
                return []
            elif len(dfthis)>1:
                print(f"found >1 {len(dfthis)} parse! returning the first one, for the follpowing")
                print(taskname)
                print(ind_metadat)

            return dfthis["parses"].values[0]

        PARSES = self._loadParses(parses_good=parses_good)

        def F(x):
            """ pull out list of parses"""
            ind = x["which_metadat_idx"]
            task = x["unique_task_name"]
            return _extractParses(PARSES, ind, task)

        self.Dat = applyFunctionToAllRows(self.Dat, F, newcolname="parses")

        if print_summary:
            # -- sanity check, did we miss any parses?
            print("num parses -- num cases")
            print(self.Dat["parses"].apply(lambda x:len(x)).value_counts())

        if fail_if_empty:
            counts = set(self.Dat["parses"].apply(lambda x:len(x)).values)
            assert 0 not in counts, "at least one row failed to extract parses..."

        print("-- parses gotten!")

    def strokesToSplines(self, strokes_ver="strokes_beh", make_new_col=True,
        add_fake_timesteps=True):
        print("converting to splines, might take a minute...")
        from ..drawmodel.splines import strokes2splines
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        from pythonlib.tools.stroketools import fakeTimesteps      

        def F(x):
            strokes = x[strokes_ver]
            return strokes2splines(strokes)

        if make_new_col:
            newcolname = strokes_ver + "_splines"
        else:
            newcolname = strokes_ver

        self.Dat = applyFunctionToAllRows(self.Dat, F, "tmp")
        self.Dat[newcolname] = self.Dat["tmp"]
        del self.Dat["tmp"]

        if add_fake_timesteps:
            print("adding fake timesteps")
            for row in self.Dat.iterrows():
                fakeTimesteps(row[1][newcolname])



    def parsesChooseSingle(self, convert_to_splines=True, add_fake_timesteps=True, 
        replace=False):
        """ 
        pick out a random single parse and assign to a new column
        - e..g, if want to do shuffle analyses, can run this each time to pick out a random
        parse
        - replace, then overwrites, otherwise fails.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        from pythonlib.tools.stroketools import fakeTimesteps      
        import random

        assert "parses" in self.Dat.columns, "need to first extract parses"
        _checkPandasIndices(self.Dat)

        def F(x):
            """ pick out a isngle random parse"""
            strokes = random.choice(x["parses"])
            return strokes
    
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strokes_parse", replace=replace)

        if convert_to_splines:
            print("converting to splines, might take a minute...")
            self.strokesToSplines("strokes_parse", make_new_col=False)

            # from ..drawmodel.splines import strokes2splines
            # def F(x):
            #     strokes = x["strokes_parse"]
            #     return strokes2splines(strokes)

            # self.Dat = applyFunctionToAllRows(self.Dat, F, "tmp")
            # self.Dat["strokes_parse"] = self.Dat["tmp"]
            # del self.Dat["tmp"]

        if add_fake_timesteps:
            print("adding fake timesteps")
            for row in self.Dat.iterrows():
                fakeTimesteps(row[1]["strokes_parse"])

        print("done choosing a single parse, it is in self.Dat['strokes_parse']!")


    def parses_load_motor_programs(self):
        """ load pre-saved programs
        loads best parse (MP) presaved, BPL.
        (see strokesToProgram) for extraction.
        RETURNS:
        - self.Dat adds a new column "parses_motor_programs", which is a list, same len as parses,
        holding programs. 
        - self.Parses, which stores things related to params.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # check that havent dione before
        # assert not hasattr(self, "Parses"), "self.Parses exists. I woint overwrite."
        assert "parses_motor_programs" not in self.Dat.columns

        MPdict_by_metadat = self._loadMotorPrograms(ver="parses")

        # save things
        self.Parses = {}
        self.Parses["params_MP_extraction"] = {k:MP["params"] for k, MP in MPdict_by_metadat.items()}

        count=0
        def F(x):
            nparses = len(x["parses"])
            idx_mdat = x["which_metadat_idx"]
            trialcode = x["trialcode"]
            MPlist = []
            if count%100==0:
                print(trialcode)    
            count+=1
            for i in range(nparses):
                MP = self._extractMP(MPdict_by_metadat, idx_mdat, trialcode, parsenum=i)
                assert MP is not None, "did not save this MP..."
                MPlist.append(MP)
            return MPlist

        self.Dat = applyFunctionToAllRows(self.Dat, F, 'parses_motor_programs')        


    ############## BEHAVIOR ANALYSIS
    def score_visual_distance(self, DVER = "position_hd_soft", return_vals=False):
        """
        extract scalar score for each trial, based on position-wise distance
        - return_vals, then returns list of vals, without modifiying self.Dat
        """
        from pythonlib.drawmodel.strokedists import distscalarStrokes

        # Distance function
        def F(x):
            strokes_beh = x["strokes_beh"]
            strokes_task = x["strokes_task"]
            return distscalarStrokes(strokes_beh, strokes_task, DVER, 
                                     do_spatial_interpolate=True, do_spatial_interpolate_interval = 10)
        if return_vals:
            df = applyFunctionToAllRows(self.Dat, F, "hdoffline")
            return df["hdoffline"].to_list()
        else:
            self.Dat = applyFunctionToAllRows(self.Dat, F, "hdoffline")


    def extract_beh_features(self, feature_list = ["angle_overall", "num_strokes", "circ", "dist"]):
        """ extract features, one val per row, 
        INPUT:
        - feature_list, list of strings. instead of string, if pass in function, then will use that.
        func: strokes --> scalar. 
        RETURNS:
        - (modifies self.Dat in place)
        - feature_list_names, which are the col names (FEAT_...)
        NOTE:
        - for features which operate at single traj level, takes mean over all trajs for a single strokes.
        """
        from pythonlib.drawmodel.features import strokesAngleOverall, strokeCircularity, strokeDistances
        feature_list_names = []

        # get overall angle for each task
        for f in feature_list:
            if not isinstance(f, str):
                # Then should be function handle
                x = [f(strokes) for strokes in self.Dat["strokes_beh"].values]
            else:
                if f=="angle_overall":
                    x = [strokesAngleOverall(strokes) for strokes in self.Dat["strokes_beh"].values]
                elif f=="num_strokes":
                    x = [len(strokes) for strokes in self.Dat["strokes_beh"].values]
                elif f=="circ":
                    x= [np.mean(strokeCircularity(strokes)) for strokes in self.Dat["strokes_beh"].values]
                elif f=="dist":
                    x = [np.mean(strokeDistances(strokes)) for strokes in self.Dat["strokes_beh"].values]
                elif f=="hdoffline":
                    x = self.score_visual_distance(return_vals=True)
                else:
                    print(f)
                    assert False

            print(f"Num nan/total, for {f}")
            print(sum(np.isnan(x)), "/", len(x))
            # self.Dat[f"FEAT_{f}"] = x
            self.Dat[f] = x
            feature_list_names.append(f"FEAT_{f}")

        print("Added these features:")
        print(feature_list_names)
        
        return feature_list_names




    ############# EXTRACT PREPROCESSED STROKES - for subsequent analyses
    def extractStrokeLists(self, filtdict=None, recenter=False, rescale=False):
        """ extract preprocessed strokes (task and beh) useful for 
        computing visual distances
        INPUT: 
        - D, Dataset
        - filtdict
        --- e.g., F = {"epoch":[1], "taskgroup":["G3"]}
        OUT:
        - makes copy of D (note, will not have the interpolated strokes in it)
        """
        from pythonlib.drawmodel.strokedists import distscalarStrokes

        # -- preprocess strokes
        if filtdict:
            Dthis = self.filterPandas(filtdict, "dataset")
        else:
            Dthis = self.copy()

        # ---- recenter
        params =[]
        if recenter:
            params.append("recenter")
        if rescale:
            params.append("rescale_to_1")
        if len(params)>0:
            Dthis.preprocessGood(ver=None, params=params)

        # Pull out dataframe
        # -- get strokes
        strokes_beh_list = Dthis.Dat["strokes_beh"].tolist()
        strokes_task_list = Dthis.Dat["strokes_task"].tolist()

        # ---- interpolate once
        def dfunc(strokes_beh, strokes_task):
            return distscalarStrokes(strokes_beh, strokes_task, "position_hd_soft", 
                                     do_spatial_interpolate=True, do_spatial_interpolate_interval = 10,
                                        return_strokes_only=True)
        S = [dfunc(sb, st) for sb, st in zip(strokes_beh_list, strokes_task_list)]
        
        # break out
        strokes_beh_list = [SS[0] for SS in S]
        strokes_task_list = [SS[1] for SS in S]
        
        return strokes_beh_list, strokes_task_list, Dthis    

    ############# DAT dataframe manipualtions
    def grouping_append_col(self, grp_by, new_col_name):
        """ append column with index after applying grp_by, 
        as in df.groupby, where the new val is  string, from
        str(list), where list is the grp_by levels for that
        row.
        RETURNS:
        - modified self.Dat
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index
        self.Dat = append_col_with_grp_index(self.Dat, grp_by, new_col_name)
        print("appended col to self.Dat:")
        print(new_col_name)


    def dat_append_col_by_grp(self, grp_by, new_col_name):
        assert False, "moved to grouping_append_col"


    ################# EXTRACT DATA AS OTHER CLASSES
    def behclass_generate(self, indtrial):
        """ Generate the BehaviorClass object for this trial
        RETURNS:
        - Beh, class instance
        - NOTE: doesn't modify self
        """
        from pythonlib.behavior.behaviorclass import BehaviorClass

        # TODO: shape_dist_thresholds, for only keeping good matches.
        # params = {
        # "D":self,
        # "ind":ind,
        # "shape_dist_thresholds":shape_dist_thresholds
        # }
        params = {
            "D":self,
            "ind":indtrial,
        }
        Beh = BehaviorClass(params, "dataset")
        return Beh

    def behclass_generate_alltrials(self):
        """ Generate list of behClass objects, one for each trial,
        and stores as part of self.
        RETURNS:
        - self.Dat["BehClass"], list of beh class iunstance.
        """
        ListBeh = [self.behclass_generate(i) for i in range(len(self.Dat))]
        self.Dat["BehClass"] = ListBeh
        
        print("stored in self.Dat[BehClass]")

    def behclass_extract(self, inds_trials = None):
        """ Get list of behclass for these trials
        - Gets precomputed
        """
        if inds_trials is None:
            inds_trials = range(len(self.Dat))
        return [self.Dat.iloc[i]["BehClass"] for i in inds_trials]



    ################ SAVE
    def make_fig_savedir_suffix(self, filterDict):
        """ wrapper to help make suffix for saving, encoding
        animal, expt, and filter params in filterDict (which assumed
        to be used in self.filterPandas to filkter datraset)
        """

        suff = f"{'_'.join(self.animals())}-{'_'.join(self.expts())}"

        for k, v in filterDict.items():
            if isinstance(v[0], bool):
                v = [str(vv) for vv in v]
            if k=="insummarydates":
                suff += f"-insumd_{'_'.join(v)}"
            elif k=="random_task":
                suff += f"-rndmt_{'_'.join(v)}"
            elif k=="monkey_train_or_test":
                suff += f"-mktt_{'_'.join(v)}"
            elif k=="taskgroup":
                suff += f"-tskgrp_{'_'.join(v)}"
            else:
                print("dont know this: ", k)

        return suff

    def save_state(self, SDIR_MAIN, SDIR_SUB):
        """
        RETURNS:
        - SDIR_THIS, dir where this saved.
        """
        import os

        ts = makeTimeStamp()
        # os.makedirs(SDIR_MAIN, exist_ok=True)
        SDIR_THIS = f"{SDIR_MAIN}/{SDIR_SUB}-{ts}"
        os.makedirs(SDIR_THIS, exist_ok=True)
        print(SDIR_THIS)

        # Dat
        self.Dat.to_pickle(f"{SDIR_THIS}/Dat.pkl")

        # Metadats
        with open(f"{SDIR_THIS}/Metadats.pkl", "wb") as f:
            pickle.dump(self.Metadats, f)
        
        # BPL
        if hasattr(self, "BPL"):    
            with open(f"{SDIR_THIS}/BPL.pkl", "wb") as f:
                pickle.dump(self.BPL, f)

        return SDIR_THIS


    ############# PLOTS
    def plotMP(self, MP, ax):
        """ helper for plotting motor program
        """
        from ..bpl.strokesToProgram import plotMP
        plotMP(MP, ax=ax)


    def plotSingleTrial(self, idx, things_to_plot = ["beh", "task"],
        sharex=False, sharey=False, params=None, task_add_num=False):
        """ 
        idx, index into Dat, 0, 1, ...
        - params, only matters for some things.
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse, plotDatWaterfall
        dat = self.Dat

        # === Plot a single trial
        ncols=4
        nplots = len(things_to_plot)
        nrows = int(np.ceil(nplots/ncols))

        fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, figsize=(ncols*3, nrows*3))

        for thing, ax in zip(things_to_plot, axes.flatten()):
            if thing=="beh":
                # Plot behavior for this trial
                # - the ball marks stroke onset. stroke orders are color coded, and also indicated by numbers.
                strokes = dat["strokes_beh"].values[idx]
                plotDatStrokes(strokes, ax, each_stroke_separate=True)
            elif thing=="task":
                # overlay the stimulus
                stim = dat["strokes_task"].values[idx]
                if task_add_num:
                    add_stroke_number=True
                    mark_stroke_onset=True
                else:
                    add_stroke_number=False
                    mark_stroke_onset=False

                plotDatStrokes(stim, ax, each_stroke_separate=True, 
                    plotver="onecolor", add_stroke_number=add_stroke_number, 
                    mark_stroke_onset=mark_stroke_onset, pcol="k")
            elif thing=="beh_tc":
                # === Plot motor timecourse for a single trial (smoothed velocity vs. time)
                # fig, axes = plt.subplots(1,1, sharex=True, sharey=True)
                strokes = dat["strokes_beh"].values[idx]
                plotDatStrokesTimecourse(strokes, ax, plotver="vel", label="vel (pix/sec)")
            elif thing=="parse":
                if "parses" in dat.columns:
                    if "strokes_parse" not in dat.columns:
                        assert False, "need to tell me which pares to plot, run self.parsesChooseSingle()"
                    strokes = dat["strokes_parse"].values[idx]
                    plotDatStrokes(strokes, ax, each_stroke_separate=True)
            elif thing=="bpl_mp":
                if "motor_program" in dat.columns:
                    # from ..bpl.strokesToProgram import plotMP
                    MP = dat["motor_program"].values[idx]
                    # plotMP(MP, ax=ax)
                    self.plotMP(MP, ax=ax)
            elif thing=="parse_frompool_mp":
                # Then you need to tell me which mp (0, 1, ...)
                ind = params["parse_ind"]
                MP = dat["parses_motor_programs"].values[idx][ind]
                self.plotMP(MP, ax=ax)

            elif thing=="beh_splines":
                assert "strokes_beh_splines" in dat.columns, "need to run convertToSplines first!"
                strokes = dat["strokes_beh_splines"].values[idx]
                plotDatStrokes(strokes, ax, each_stroke_separate=True)
            else:
                print(thing)
                assert False

            ax.set_title(thing)
        return fig


    def plotMultStrokesByOrder(self, strokes_list, ncols = 5, titles=None, 
        naked_axes=False, mark_stroke_onset=True, add_stroke_number=True, 
        titles_on_y=False, SIZE=2.5, plotkwargs={}):
        """ Helper to call plotMultStrokesColorMap so that plots by order
        """
        colors = [np.arange(0, len(s)) for s in strokes_list]
        return self.plotMultStrokesColorMap(strokes_list, colors, 
            ncols, titles, naked_axes, mark_stroke_onset,
            add_stroke_number, titles_on_y, SIZE, plotkwargs)



    def plotMultStrokesColorMap(self, strokes_list, strokes_vals_list, 
        ncols = 5, titles=None, naked_axes=False, mark_stroke_onset=False,
        add_stroke_number=True, titles_on_y=False, SIZE=2.5, plotkwargs={}):
        """ helper to plot multiplie trials when already have strokes extracted)
        Assumes want to plot this like behavior.
        """
        from ..drawmodel.strokePlots import plotDatStrokesMapColor
        import random
        from pythonlib.tools.plottools import plotGridWrapper

        plotfunc = lambda strokes_strokesvals, ax: plotDatStrokesMapColor(
            strokes_strokesvals[0], ax, strokes_strokesvals[1],
            naked_axes=naked_axes, mark_stroke_onset=mark_stroke_onset, 
            add_stroke_number=add_stroke_number, **plotkwargs)

        list_data = [(strokes, strokes_vals) for strokes, strokes_vals in 
            zip(strokes_list, strokes_vals_list)]
        
        fig, axes= plotGridWrapper(list_data, plotfunc, ncols=ncols, titles=titles,
            naked_axes=naked_axes, origin="top_left", titles_on_y=titles_on_y, 
            SIZE=SIZE, return_axes=True)

        return fig, axes


    def plotMultStrokes(self, strokes_list, ncols = 5, titles=None, naked_axes=False, 
        add_stroke_number=True, centerize=False, jitter_each_stroke=False, 
        titles_on_y=False, SIZE=2.5, is_task=False, number_from_zero=False):
        """ helper to plot multiplie trials when already have strokes extracted)
        Assumes want to plot this like behavior.
        """
        from ..drawmodel.strokePlots import plotDatStrokes
        import random
        from pythonlib.tools.plottools import plotGridWrapper

        if is_task:
            plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_task=True, 
                add_stroke_number=add_stroke_number, centerize=centerize, 
                jitter_each_stroke=jitter_each_stroke, number_from_zero=number_from_zero)
        else:
            plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered=True, 
                add_stroke_number=add_stroke_number, centerize=centerize, 
                jitter_each_stroke=jitter_each_stroke, number_from_zero=number_from_zero)
        fig, axes= plotGridWrapper(strokes_list, plotfunc, ncols=ncols, titles=titles,naked_axes=naked_axes, origin="top_left",
            titles_on_y=titles_on_y, SIZE=SIZE, return_axes=True)

        return fig, axes

    def extractStrokeVels(self, list_inds):
        """ extract stroke as instantaneous velocities
        INPUT:
        - list_inds, list of ints, for which strokes to use. If "all", then gets all.

        RETURN:
        - list of strokes_vels, which are each list of Nx1, so is
        actually speed, not vel.
        (None, if strok to oshort to get vel.)
        """
        from pythonlib.tools.stroketools import strokesVelocity
  
        if list_inds == "all":
            list_inds = self.Dat.index.tolist()

        list_strokes_vel = []
        for ind in list_inds:
            strokes = self.Dat.iloc[ind]["strokes_beh"]
            fs = self.get_sample_rate(ind)
            _, strokes_vel = strokesVelocity(strokes, fs, clean=True)
            strokes_vel = [s[:,0] for s in strokes_vel] # remove time
            for i in range(len(strokes_vel)):
                if any([np.isnan(sv) for sv in strokes_vel[i]]):
                    list_strokes_vel.append(None)   
                    # print(strokes_vel)
                    # print(list_inds)
                    # print(strokes[i])
                    # assert False
                else:
                    list_strokes_vel.append(strokes_vel)
        return list_strokes_vel


    def _plot_prepare_strokes(self, which_strokes, idxs, nrand=None, titles=None):
        """
        Helper to extract strokes
        """
        import random

        if len(idxs)==0:
            return

        if isinstance(idxs, int):
            N = len(self.Dat)
            k = idxs
            idxs = random.sample(range(N), k=k)

        if nrand is not None:
            if nrand < len(idxs):
                idxs = sorted(random.sample(idxs, nrand))

        if which_strokes=="parses":
            # then pull out a random parse for each case
            assert "parses" in self.Dat.columns, "need to extract parses first..."
            strokes_list = [[a.copy() for a in strokes] for strokes in self.Dat["parses"].values] # copy, or else will mutate
            # for the indices you want to plot, prune to just a single parse
            for i in idxs:
                ithis = random.randrange(len(strokes_list[i]))
                strokes_list[i] = strokes_list[i][ithis]
        else:
            strokes_list = self.Dat[which_strokes].values

        # Keep only data to plot.
        strokes_list = [strokes_list[i] for i in idxs]

        if titles is None:
            # use trialcodes
            titles = self.Dat.iloc[idxs]["trialcode"].tolist()

        return strokes_list, idxs, titles

    def plotMultTrials(self, idxs, which_strokes="strokes_beh", return_idxs=False, 
        ncols = 5, titles=None, naked_axes=False, add_stroke_number=True, centerize=False, 
        nrand=None, color_by=None):
        """ plot multiple trials in a grid.
        - idxs, if list of indices, then plots those.
        --- if an integer, then plots this many random trials.
        - which_strokes, either "strokes_beh" (monkey) or "strokes_task" (stim)
        - nrand, sample random N
        - color_by, if None, ignores. otherwise different ways to color strokes.
        --- "speed", instantenous speed.
        NOTE:
        - returns None if fails to get any data.
        """
        from ..drawmodel.strokePlots import plotDatStrokes
        from pythonlib.tools.plottools import plotGridWrapper

        strokes_list, idxs, titles = self._plot_prepare_strokes(which_strokes, idxs, 
            nrand=nrand, titles=titles)
        if len(idxs)==0:
            if return_idxs:
                return None, None
            else:
                return None

        # Extract the final data
        if color_by is None:
            # Which plotting function?
            if which_strokes in ["strokes_beh", "parses"]:
                plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered=True, 
                    add_stroke_number=add_stroke_number, centerize=centerize)
            elif which_strokes == "strokes_task":
                # plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_unordered=True, 
                #     add_stroke_number=add_stroke_number)
                plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_task=True, centerize=centerize)
            else:
                assert False

            # Plot
            fig= plotGridWrapper(strokes_list, plotfunc, ncols=ncols, titles=titles,naked_axes=naked_axes, origin="top_left")

        elif color_by=="speed":
            # instantaneuos speed
            
            # -- function
            assert which_strokes=="strokes_beh", "only coded for this currently"
            from pythonlib.drawmodel.strokePlots import plotDatStrokesMapColor
            # 0: strokes... 1: stroke_vel... 2: vmin... 3: vmax
            plotfunc = lambda data, ax: plotDatStrokesMapColor(
                data[0], ax, data[1], data[2], data[3])

            # - get speed
            strokes_vels_list = self.extractStrokeVels(idxs)
            # dont keep if too short.
            strokes_list = [strokes for strokes, sv in zip(strokes_list, strokes_vels_list) if sv is not None]
            titles = [t for t, sv in zip(titles, strokes_vels_list) if sv is not None]
            idxs = [i for i, sv in zip(idxs, strokes_vels_list) if sv is not None]
            strokes_vels_list = [sv for sv in strokes_vels_list if sv is not None]

            # -- data
            data = []
            for strokes, strokes_vel in zip(strokes_list, strokes_vels_list):
                # Get the min and max vel
                # vmin is min of mins across strokes.
                x = [np.percentile(s, [1, 99]) for s in strokes_vel]
                vmin = np.min([xx[0] for xx in x])
                vmax = np.max([xx[1] for xx in x])
                if np.isnan(vmin):
                    print(x)
                    # print([len(s) for s in strokes_vel])
                    print(strokes_vel)
                    assert False
                data.append([strokes, strokes_vel, vmin, vmax])

            # use the same speed across all trials.
            if len(data)>0:
                vmin = np.min([d[2] for d in data])
                vmax = np.max([d[3] for d in data])
                if np.isnan(vmin):
                    print([d[2] for d in data])
                    assert False
                if np.isnan(vmax):
                    print([d[3] for d in data])
                    assert False
                for d in data:
                    d[2] = vmin
                    d[3] = vmax
            else:
                if return_idxs:
                    return None, None
                else:
                    return None

            # -- Plot
            # for d in data:
            #     print([len(dd) for dd in d[0]])
            #     print([dd[0] for dd in d[0]])

            fig= plotGridWrapper(data, plotfunc, ncols=ncols, titles=titles,naked_axes=naked_axes, origin="top_left")
        else:
            print(color_by)
            assert False, "not coded"

        if return_idxs:
            return fig, idxs
        else:
            return fig
        # return idxs


    def plotMultStrokesTimecourse(self, strokes_list, idxs=None, plotver="speed",
        return_idxs=False, ncols = 5, titles=None, naked_axes=False, aspect=0.8,
        align_to=None, overlay_stroke_periods=False):
        """ Helper to plot timecourse
        PARAMS:
        - plotver, str, {'speed', 'raw'}
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse, plotDatStrokesVelSpeed
        from ..drawmodel.strokePlots import plotDatStrokes
        from pythonlib.tools.plottools import plotGridWrapper

        # get sampling rate
        if idxs is not None:
            list_fs = [self.get_sample_rate(i) for i in idxs]
            assert len(np.unique(list_fs))==1, "why have diff sample rate? easy fix, make plotfunc take in fs in data tuple"
            fs = list_fs[0]
        else:
            # just pick the first trial
            fs = self.get_sample_rate(0)

        if align_to=="first_touch":
            # make the first touch 0
            strokes_list = [[s.copy() for s in strokes] for strokes in strokes_list] 
            for strokes in strokes_list:
                t0 = strokes[0][0,2]
                for s in strokes:
                    s[:,2] -= t0
        else:
            assert align_to is None
        # whether to modify the time column for strokes
        # by default, is aligned to first touch in strokes.

        # elif align_to=="first_stroke_onset":
        #     # do nothing - this is default.
        #     pass
        # else:
        #     print(align_to)
        #     assert False, "not coded"

        # Which plotting function?
        plotfunc = lambda strokes, ax: plotDatStrokesVelSpeed(strokes, ax, fs, plotver,
            overlay_stroke_periods=overlay_stroke_periods)

        # Plot
        fig= plotGridWrapper(strokes_list, plotfunc, ncols=ncols, 
            titles=titles,naked_axes=naked_axes, origin="top_left", aspect=aspect)

        if return_idxs:
            return fig, idxs
        else:
            return fig


    def plotMultTrialsTimecourse(self, idxs, plotver="speed", which_strokes="strokes_beh", return_idxs=False, 
        ncols = 5, titles=None, naked_axes=False, nrand=None, align_to="go_cue"):
        """ Plot a grid of trials timecourses.
        """ 
 
        strokes_list, idxs, titles = self._plot_prepare_strokes(which_strokes, idxs, 
            nrand=nrand, titles=titles)
        if len(idxs)==0:
            return


        for strokes in strokes_list:
            try:
                assert strokes[0][0,2]==0., "I made mistake, not actually aligning by 0 by default"
                # assert strokes_list[0][0,2]==0., "I made mistake, not actually aligning by 0 by default"
            except Exception as err:
                print(strokes)
                raise err

        if align_to=="go_cue":
            # copy strokes.
            strokes_list = [[s.copy() for s in strokes] for strokes in strokes_list] 
            for ind, strokes in zip(idxs, strokes_list):

                # get time from gocue to first touch
                motor = self.get_motor_stats(ind)
                x = motor["time_raise2firsttouch"] * 1000 # convert from s to ms.

                if np.isnan(x):
                    print("Here")
                    print(x)
                    print(ind)
                    print(self.Dat.iloc[ind])
                    assert False
                # add to all times
                # print(strokes)
                for s in strokes:
                    s[:,2] += x
                # print(strokes)
                # assert False

                # ME = Dthis.Dat.iloc[1]["motorevents"]
                # a = ME["ons"] - ME["raise"]

                # MT = Dthis.Dat.iloc[1]["motortiming"]
                # b = MT["time_raise2firsttouch"]

                # assert a==b, "I forgot what my entries mean"

        self.plotMultStrokesTimecourse(strokes_list, idxs, plotver, 
            return_idxs=return_idxs, ncols=ncols, 
            titles=titles, naked_axes=naked_axes)



    def plotOverview(self):
        """ quick plot of kinds of tasks present across days
        # To visualize the experimental structure:
        # notes:
        # - tvalday is time of trial. e..g, the first digit is the day, the value after decimal is fraction of day (not
        # exactly from midnight to midnight, but within the "experimental session"). so 2.5 means day 1, halfway through the 
        # day (from start of expt to end of expt)
        # - to get the actual time, see "tval" qwhich is fraction of day from 0:00 to 24:00
        # RETURNS: 
        # - list of figs.
        """
        import seaborn as sns
        from pythonlib.tools.snstools import rotateLabel

        figlist = []

        fig = sns.catplot(data=self.Dat, x="tvalday", y="taskgroup", hue="task_stagecategory", 
            row="expt", col="epoch")
        figlist.append(fig)

        fig = sns.catplot(data=self.Dat, x="task_stagecategory", y="taskgroup", hue="monkey_train_or_test", 
            row="date", col="epoch")
        rotateLabel(fig)
        figlist.append(fig)

        fig = sns.catplot(data=self.Dat, x="block", y="taskgroup", hue="monkey_train_or_test", 
            row="date", col="epoch", aspect=2)
        figlist.append(fig)

        fig = sns.catplot(data=self.Dat, x="tvalday", y="taskgroup", hue="monkey_train_or_test", 
            row="task_stagecategory", col="epoch", aspect=2)
        figlist.append(fig)

        # fig = sns.catplot(data=self.Dat, x="task_stagecategory", y="taskgroup", hue="online_abort", 
        #     row="date", col="epoch")
        # rotateLabel(fig)

        # fig = sns.scatter(data=self.Dat, x="task_stagecategory", y="taskgroup", hue="online_abort", 
        #     row="date", col="epoch")
        # rotateLabel(fig)

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        nrow = 1
        fig, axes = plt.subplots(nrow,1, sharex=True, figsize=(15,nrow*6), squeeze=False)
        ax=axes.flatten()[0]
        sns.lineplot(data=self.Dat, x="trial", y="block", estimator=None, ax=ax)
        sns.scatterplot(data=self.Dat, x="trial", y="block",ax=ax)
        ax.grid()
        figlist.append(fig)

        # ax=axes.flatten()[1]
        # sns.lineplot(data=D.Dat, x="trial", y="blokk", estimator=None, ax=ax)
        # ax=axes.flatten()[2]
        # sns.lineplot(data=D.Dat, x="trial", y="bloque", estimator=None, ax=ax)

        return figlist

    ############### PRINT INFO
    def printOverview(self):
        from pythonlib.tools.pandastools import printOverview
        printOverview(self.Dat)


    ################ ANALYSIS HELPERS

    def analy_get_all_inds_with_same_task(self):
        """
        returns indices for other trials
        
        """
        assert False, "havent figured out how to remove the ind for each own trial."
        def F(x):
            print(dir(x))
            task = x["unique_task_name"]
            inds = self.Dat[self.Dat["unique_task_name"]==task].index.to_list()
            print(inds)
            assert False

        self.Dat = applyFunctionToAllRows(self.Dat, F, "inds_same_task")

    def analy_subsample_trials_for_example(self, scores, inds, method="uniform",
        score_range=None, n=20):
        """ Extract subset of trials based on some filter for some score.
        e.g. get uniform distribution based on percentiles for score, or
        get top 20, or bottom 100, etc.
        Doesn't use self.Dat, you must enter the data here.
        Useful for generating/plotting examples across range of dataset
        PARAMS:
        - scores, np.array, scalar values used for sorting
        - inds, list or np.array, same length as scores. list of ids, usually ints, but could be tuples.
        - method, str, {'top', 'bottom', 'uniform'}
        - score_range, [min, max], will restrict to within this range. applies before taking
        top or uniform, etc. None means use entire range.
        - n, how many to take? will return in incresaing order of scores.
        RETURNS:
        - scores_sub, inds_sub, same length subsamples.
        """

        if isinstance(scores, list):
            scores = np.asarray(scores)
        assert isinstance(inds, list)

        # sort scores
        indsort = np.argsort(scores)
        scores = scores[indsort]
        inds = [inds[i] for i in indsort]

        # Keep only within range
        if score_range is None:
            score_range = [min(scores), max(scores)] 
        indrange = np.where((scores>=score_range[0]) & (scores<=score_range[1]))[0]
        scores = scores[indrange]
        inds = [inds[i] for i in indrange]

        # you can't request for than available
        if n>len(scores):
            n = len(scores)

        if method=="uniform":
            # take uniform sample 
            indstake = np.floor(np.linspace(0, len(scores)-1, n))
            indstake = [int(i) for i in indstake]
            scores_sub = scores[indstake]
            inds_sub = [inds[i] for i in indstake]
        elif method=="top":
            scores_sub = scores[-n:]
            inds_sub = inds[-n:]
        elif method=="bottom":
            scores_sub = scores[:n]
            inds_sub = inds[:n]
        elif method=="random":
            assert False, "get random, then sort."
        else:
            assert False
        return scores_sub, inds_sub


    def analy_get_tasks_strongesteffect(self, grouping, grouping_levels, feature,
        dfunc = lambda x,y: y-x, df_in=None, return_vals_sorted=False):
        """ Extract tasks which have greatest difference along some feature (column),
        after aggregating over tasks, separating by some grouping variable.
        Currently only works for goruping using 2 levels (takes difference)
        e.g., can ask: which task shows largest difference for feature "score", across
        epochs 1 and 2?
        INPUTS:
        - dfunc, function for scoring difference, func: x,y --> scalar, where y is feature for
        group level 2.
        - df, if None, then uses self.Dat. otherwises uses df.
        RETURNS:
        - tasklist, which is sorted. first item is that with greatest psoitive difference (level 2 minus level 1)
        """
        from pythonlib.tools.pandastools import pivot_table

        if df_in is None:
            print("Using self.Dat")
            dfthis = self.Dat
        else:
            print("Using inputted df_in")
            dfthis = df_in

        # Remove nan

        # Pivot
        df = pivot_table(dfthis, index=["character"], columns=[grouping], 
                         values = [feature])

        df = df.dropna()
        df = df.reset_index(drop=True)

        # get diff
        x1 = df[feature][grouping_levels[0]] 
        x2 = df[feature][grouping_levels[1]]
        tasklist = df["character"].to_list()
        value_diffs = [dfunc(xx1, xx2) for xx1, xx2 in zip(x1, x2)]

        tmp = [(task, v) for task, v in zip(tasklist, value_diffs)]
        tmp_sorted = sorted(tmp, key=lambda x: x[1])[::-1]
        tasklist_sorted = [t[0] for t in tmp_sorted]
        vals_sorted = [t[1] for t in tmp_sorted]

        # # tmp = [(i, v) for i, v in enumerate(value_diffs)]
        # # inds_sorted = sorted(tmp, key= lambda x: x[1])[::-1]
        # # vals_sorted = [i[1] for i in inds_sorted]
        # # inds_sorted = [i[0] for i in inds_sorted]
        # if True:
        #     print(vals_sorted)
        # # nplot = np.min([10, len(inds_sorted)])
        # # inds_to_plot = inds_sorted[:nplot]

        # tasklist_sorted = [tasklist[i] for i in inds_sorted]
        if return_vals_sorted:
            return tasklist_sorted, vals_sorted
        else:
            return tasklist_sorted


    def analy_singletask_df(self, task, row_variable, row_levels=None, return_row_levels=False):
        """ extract datafarme of just one task, with appended columns named
        "row" and "col" where col is trial num (in order seen in self.Dat) and row
        is level for row_variable. indices will be reset within each row level, but not in
        in the final df.
        RETURNS:
        - df
        - return_row_levels (if flagged on)
        """
        df = self.Dat[self.Dat["character"] == task]

        # what are rows
        if row_levels is None:
            row_levels = sorted(df[row_variable].unique().tolist())

        # To number from 0, 1, ... for within each row level, iterate over them
        # and each time reset their index, then save the index.
        out = []
        for i, lev in enumerate(row_levels):
            dfthis = df[df[row_variable]==lev]
            dfthis = dfthis.reset_index(drop=True)
            dfthis["col"] = dfthis.index.tolist()
            dfthis["row"] = i
            out.append(dfthis)

        if return_row_levels:
            return pd.concat(out), row_levels
        else:
            return pd.concat(out)

    def analy_assign_trialnums_within_task(self, groupby=None):
        """ in chron order 0, 1, 2, assigned to each row depending 
        onw hich trial for its unique task. groupby allows to have
        0, 1, 2, for a grouping variable too.
        RETURNS:
        - modifies self.Dat to have new column "trialnum_this_task"
        """
        from pythonlib.tools.nptools import rankItems
        def F(x):
            assert np.all(np.diff(x)>0), "shouldnt matter, but should confirm that now rankItems outputs correct order. then remove this assertion?"
            return rankItems(x)

        assert isinstance(groupby, list)
        gb = ["character"] + groupby
        self.Dat["trialnum_this_task"] = self.Dat.groupby(gb)["tval"].transform(F)


    def analy_match_numtrials_per_task(self, groupby):
        """ returns dataset where make sure each task has same num trials across
        all levels for group. prunes by deleting trials not matched.
        RETURNS:
        - new dataset, copied and pruned.
        """
        _checkPandasIndices(self.Dat)
        # D.Dat = D.Dat.reset_index(drop=True)

        # first Make sure assigned trialnums
        self.analy_assign_trialnums_within_task(groupby)

        tasklist = self.Dat["character"].unique().tolist()

        # Collect inds that are extraneous.
        inds_bad_all = []
        for task in tasklist:
            df = self.Dat[self.Dat["character"]==task]
            maxnum = min(df.groupby(groupby)["trialnum_this_task"].max())
            inds_bad = df.index[df["trialnum_this_task"]>maxnum].tolist()
            inds_bad_all.extend(inds_bad)

        Dtrialsmatch = self.copy()
        print(len(Dtrialsmatch.Dat))
        Dtrialsmatch.Dat = Dtrialsmatch.Dat.drop(inds_bad_all)
        Dtrialsmatch.Dat = Dtrialsmatch.Dat.reset_index(drop=True)
        print(len(Dtrialsmatch.Dat))
        return Dtrialsmatch


    def analy_match_sequence_discrete_per_task(self, groupby, grouping_levels = None,
        ver="same", print_summary=False):
        """ For each task, make sure trials use same ("same") or different ("diff")
        sequence across levels for groupby. Sequences are defined by discrete parse of
        task sequence.
        INPUT:
        - groupby, how to group trials
        - grouping_levels, leave None to use all. otherwise give me.
        - ver, {"same", "diff"}
        --- same, then will only keep a trial if every other grouping level has at least
        one trial with this trial's sequence. Note that can end up with a task having multiple
        sequences.
        e..g, --- mixture2_11-savedset-34-91957
                {'short': [21], 'long': [21, 43, 0]} [start, showing seq for each trial for 2 group levels.]
                {'short': [21], 'long': [21]} [end]
        --- diff, then will only keep trials for which no trial in other grouping levels have the
        same sequence. Note that can end up with a task having multiple
        sequences.
        e..g, --- mixture2_11-savedset-34-91957
                {'short': [21], 'long': [21, 43, 0]} [start, showing seq for each trial for 2 group levels.]
                {'short': [], 'long': [43, 0]} [end]


        """

        if grouping_levels is None:
            grouping_levels = self.Dat[groupby].unique().tolist()

        # Make sure sequence ids already asisgned.
        self.planner_assign_sequence_id()
        Dcopy = self.copy()

        # Go thru each task
        inds_to_remove = []
        tasklist = self.Dat["character"].unique().tolist()
        for task in tasklist:
            # print(task)
            dfthis = self.Dat[self.Dat["character"]==task]
            
            # get mapoing between grouping level and sequences that exist
            levels2sequences = {}
            for lev in grouping_levels:
                levels2sequences[lev] = dfthis[dfthis[groupby]==lev]["sequence_id"].unique().tolist()
            
            # get mapping between grouping level and sequences to keep
            # flatten to all indices that exist
            indices_all = set([vv for v in levels2sequences.values() for vv in v])
            # print(indices_all)
            # assert False

            if ver=="same":
                # find the indices that exist across all groups
                def _isgood(ind):
                    # Returns True if keep this ind, false otherwise
                    return all([ind in v for v in levels2sequences.values()])
            elif ver=="diff":
                def _isgood(ind):
                    tmp = [ind in v for v in levels2sequences.values()]
                    # if task=="mixture2_55-savedset-34-09627":
                    #     print("---")
                    #     print(levels2sequences)
                    #     print(ind)
                    #     print(sum(tmp))
                    if sum(tmp)==0:
                        assert False, "bug. this ind must be in at least one grouping level"
                    elif sum(tmp)==1:
                        # good, then is only in the one from which we extracted it to enter here
                        return True
                    elif sum(tmp)>1:
                        # then is present in at least one other grouping level.
                        return False
            else:
                print(ver)
                assert False, "noit coded"

            # update levels2sequences to only good sequences
            levels2sequences_good = {}
            for k, v in levels2sequences.items():
                levels2sequences_good[k] = [vv for vv in v if _isgood(vv)]

            # For each level, only keep trials if its sequence is in the good sequences.
            for row in dfthis.iterrows():
                if row[1]["sequence_id"] in levels2sequences_good[row[1][groupby]]:
                    # then good, keep this.
                    pass
                else:
                    inds_to_remove.append(row[0])
        inds_to_remove = sorted(list(set(inds_to_remove)))
        print("Remoiving these inds")
        print(inds_to_remove)
        self.Dat = self.Dat.drop(inds_to_remove, axis=0).reset_index(drop=True)


        if print_summary:
            def _check_task_sequences(D, task, unique=True):
                # print(task)
                dfthis = D.Dat[D.Dat["character"]==task]

                # get mapoing between grouping level and sequences that exist
                levels2sequences = {}
                for lev in grouping_levels:
                    if unique:
                        levels2sequences[lev] = dfthis[dfthis[groupby]==lev]["sequence_id"].unique().tolist()
                    else:
                        levels2sequences[lev] = dfthis[dfthis[groupby]==lev]["sequence_id"].tolist()
                print(levels2sequences)

            tasklist = self.Dat["character"].unique().tolist()
            print("SUMMARY - task, before, after, [sequences per trial]")
            for task in tasklist:
                print("---", task)
                _check_task_sequences(Dcopy, task, unique=False)
                _check_task_sequences(self, task, unique=False)

            
    def analy_reassign_monkeytraintest(self, key, list_train, list_test, list_other=[]):
        """ redefine monkey train test based on value in column "key".
        if a row[key] is in list_train, then is train, etc.
        - enforces that list_train and list_test no overlap.
        - enforces that each row will have an assignment.
        """

        # Make sure no common items
        assert len([x for x in list_train if x in list_test])==0
        assert len([x for x in list_train if x in list_other])==0
        
        def func(x):
            if x[key] in list_train:
                return "train"
            elif x[key] in list_test:
                return "test"
            elif x[key] in list_other:
                return "other"
            else:
                print(x)
                assert False

        self.Dat = applyFunctionToAllRows(self.Dat, func, "monkey_train_or_test")
        print(self.Dat["monkey_train_or_test"].value_counts())






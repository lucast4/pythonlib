""" stores and operates on datasets"""
import pandas as pd
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.pandastools import applyFunctionToAllRows
import torch
import os
from pythonlib.tools.expttools import makeTimeStamp, findPath

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
        else:
            print("Did not load data!!!")


        self._possible_data = ["Dat", "BPL", "SF"]

        # Save common coordinate systems
        # [[xmin, ymin], [xmax, ymax]]
        self._edges = {
        "bpl":np.array([[0, -104], [104, 0]])
        }

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
        print(f"Original length: {len(self.Dat)}")
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
        print(inds_to_remove)
        self.Dat = self.Dat.drop(inds_to_remove, axis=0).reset_index(drop=True)


    def removeNans(self, columns=None):
        """ remove rows that have nans for the given columns
        INPUTS:
        - columns, list of column names. if None, then uses all columns.
        RETURNS:
        [modifies self.Dat]
        """

        print("--- Removing nans")
        print("start len:", len(self.Dat))

        print("- num names for each col")
        if columns is None:
            tmpcol = self.Dat.columns
        else:
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
    def load_tasks_helper(self, reinitialize_taskobjgeneral=True):
        """ To load tasks in TaskGeneral class format.
        Must have already asved them beforehand
        - Uses default path
        - reinitialize_taskobjgeneral, then reinitializes, which is uiseful if code for
        general taskclas updates.
        RETURN:
        - self.Dat has new column called Task
        NOTE: fails if any row is not found.
        """
        from pythonlib.tools.expttools import findPath
        from pythonlib.tools.pandastools import applyFunctionToAllRows

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
                Tnew.initialize("ml2", taskobj)
                T = Tnew
            return T

        self.Dat = applyFunctionToAllRows(self.Dat, F, "Task")      
        print("added new column self.Dat[Task]")  





    ############# ASSERTIONS
    def _check_is_single_dataset(self):
        """ True if single, False, if you 
        contcatenated multipel datasets"""

        if len(self.Metadats)>1:
            return False
        else:
            return True

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
            if x["taskgroup"] in ["train_fixed", "train_random"]:
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
            # Replace unique name with new one, if tasks have been loaded
            def F(x):
                return x["Task"].Params["input_params"].info_generate_unique_name()
            self.Dat = applyFunctionToAllRows(self.Dat, F, "unique_task_name")

            # task cartegories should include setnum
            def F(x):
                return x["Task"].get_category_setnum()
            from pythonlib.tools.pandastools import applyFunctionToAllRows
            self.Dat = applyFunctionToAllRows(self.Dat, F, "task_stagecategory")
        
        # assign a "character" name to each task.
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


        # Remove any trials that were online abort.
        self.Dat = self.Dat[self.Dat["aborted"]==False]
        print("Removed online aborts")

        ####
        self.Dat = self.Dat.reset_index(drop=True)



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
    def animals(self):
        """ returns list of animals in this datsaet.
        """
        return sorted(list(set(self.Dat["animal"])))

    def expts(self):
        """ returns list of expts
        """
        return sorted(list(set(self.Dat["expt"])))


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
        sharex=False, sharey=False, params=None):
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
                plotDatStrokes(stim, ax, each_stroke_separate=True, plotver="onecolor", add_stroke_number=False, mark_stroke_onset=False, pcol="k")
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



            # 


    def plotMultTrials(self, idxs, which_strokes="strokes_beh", return_idxs=False, 
        ncols = 5, titles=None, naked_axes=False, add_stroke_number=True, centerize=False):
        """ plot multiple trials in a grid.
        - idxs, if list of indices, then plots those.
        --- if an integer, then plots this many random trials.
        - which_strokes, either "strokes_beh" (monkey) or "strokes_task" (stim)
        """
        from ..drawmodel.strokePlots import plotDatStrokes
        import random

        if isinstance(idxs, int):
            N = len(self.Dat)
            k = idxs
            idxs = random.sample(range(N), k=k)

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

        if False:
            # Old version, obsolete...
            nrows = int(np.ceil(len(idxs)/ncols))
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*2, nrows*2))
            
            for ind, (i, ax) in enumerate(zip(idxs, axes.flatten())):
                if which_strokes in ["strokes_beh", "parses"]:
                    plotDatStrokes(strokes_list[i], ax, clean_ordered=True)
                elif which_strokes == "strokes_task":
                    plotDatStrokes(strokes_list[i], ax, clean_unordered=True)
                else:
                    assert False
                if not titles:
                    ax.set_title(i)
                elif isinstance(titles[ind], str):
                    ax.set_title(f"{titles[ind]}")
                else:
                    ax.set_title(f"{titles[ind]:.2f}")
        else:
            # New version, uses grid wrapper
            from pythonlib.tools.plottools import plotGridWrapper
            if which_strokes in ["strokes_beh", "parses"]:
                plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered=True, 
                    add_stroke_number=add_stroke_number, centerize=centerize)
            elif which_strokes == "strokes_task":
                # plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_unordered=True, 
                #     add_stroke_number=add_stroke_number)
                plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_task=True, centerize=centerize)
            else:
                assert False
            data = [strokes_list[i] for i in idxs]
            fig= plotGridWrapper(data, plotfunc, ncols=ncols, titles=titles,naked_axes=naked_axes, origin="top_left")

        if return_idxs:
            return fig, idxs
        else:
            return fig
        # return idxs

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
        - tasklist, which is sorted.
        NOTE:
        - sorts so that first item is that with greatest psoitive difference (level 2 minus level 1)
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


def concatDatasets(Dlist):
    """ concatenates datasets in Dlist into a single dataset.
    Main goal is to concatenate D.Dat. WIll attempt to keep track of 
    Metadats, but have not confirmed that this is reliable yet.
    NOTE: Currently only does Dat correclt.y doesnt do metadat, etc.

    """

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
    return Dnew


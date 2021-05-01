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
        NOTE: currently doesnt copy over Metadats. only copies data
        """
        import copy

        Dnew = Dataset([])

        Dnew.Dat = self.Dat.copy()
        if hasattr(self, "BPL"):
            Dnew.BPL = copy.deepcopy(self.BPL)
        if hasattr(self, "SF"):
            Dnew.SF = self.SF.copy()

        return Dnew


    def filterPandas(self, filtdict, return_ver = "indices"):
        """
        RETURNS:
        - if return_ver is:
        --- "indices", then returns (self.Dat not modifed,) and returns inds.
        --- "modify", then modifies self.Dat, and returns Non
        --- "dataframe", then returns new dataframe, doesnt modify self.Dat
        --- "dataset", then copies and returns new dataset, without affecitng sefl.
        """
        from pythonlib.tools.pandastools import filterPandas
        # traintest = ["test"]
        # random_task = [False]
        # filtdict = {"traintest":traintest, "random_task":random_task}

        _checkPandasIndices(self.Dat)

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


    def load_dataset_helper(self, animal, expt, ver="single"):
        """ load a single dataset. 
        - ver, str
        --- "single", endures that there is one and only one.
        --- "mult", allows multiple. if want animal or expt to 
        iterate over pass in lists of strings.
        """
        if ver=="single":
            pathlist = self.find_dataset(animal, expt, assert_only_one=True)
            self._main_loader(pathlist, None)
        elif ver=="mult":
            pathlist = []
            if isinstance(animal, str):
                animal = [animal]
            if isinstance(expt, str):
                expt = [expt]
            for a in animal:
                for e in expt:
                    pathlist.extend(self.find_dataset(a, e, True))
            self._main_loader(pathlist, None)



    def _main_loader(self, inputs, append_list):
        """ loading function, use this for all loading purposes"""
        
        if self._reloading_saved_state:
            assert len(inputs)==1, "then must reload one saved state."
            self._reloading_saved_state_inputs = inputs

        if isinstance(inputs[0], str):
            self._load_datasets(inputs, append_list)
        else:
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



    def _load_datasets(self, path_list, append_list):

        assert append_list is None, "not coded yet!"

        dat_list = []
        metadats = {}

        for i, path in enumerate(path_list):
            
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



        self.Dat = pd.concat(dat_list, axis=0)
        self.Metadats = metadats

        # reset index
        print("----")
        print("Resetting index")
        self.Dat = self.Dat.reset_index(drop=True)


    ############### TASKS
    def load_tasks_helper(self):
        """ To load tasks in TaskGeneral class format.
        Must have already asved them beforehand
        - Uses default path
        RETURN:
        - self.Dat has new column called Task
        NOTE: fails if any row is not found.
        """
        from pythonlib.tools.expttools import findPath
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # Load tasks
        a = self.animals()
        e = self.expts()
        if len(a)>1 or len(e)>1:
            assert False, "currently only works if single animal/ext dataset. can modify easily"

        # Find path, load Tasks
        sdir = f"/data2/analyses/database/TASKS_GENERAL/{a[0]}-{e[0]}-all"
        pathlist = findPath(sdir, [], "Tasks", "pkl")
        assert len(pathlist)==1
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


        ####### Remove online aborts
        print("ORIGINAL: online abort values")
        print(self.Dat["online_abort"].value_counts())
        idx_good = self.Dat["online_abort"].isin([None])
        self.Dat = self.Dat[idx_good]
        print(f"kept {sum(idx_good)} out of {len(idx_good)}")
        print("removed all cases with online abort not None")
        # reset 
        self.Dat = self.Dat.reset_index(drop=True)


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
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x):
            if x["taskgroup"] in ["train_fixed", "train_random"]:
                return "train"
            elif x["taskgroup"] in ["G2", "G3", "G4", "test_fixed", "test_random"]:
                return "test"
            else:
                print(x)
                assert False, "huh"
        self.Dat = applyFunctionToAllRows(self.Dat, F, "monkey_train_or_test")

        # reset 
        self.Dat = self.Dat.reset_index(drop=True)


        ######## assign a "character" name to each task.
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x):
            if x["random_task"]:
                # For random tasks, character is just the task category
                return x["task_stagecategory"]
            else:
                # For fixed tasks, it is the unique task name
                return x["unique_task_name"]
        self.Dat = applyFunctionToAllRows(self.Dat, F, newcolname="character")

        ####
        self.Dat = self.Dat.reset_index(drop=True)

        ###### remove empty strokes
        # for 


    def find_dataset(self, animal, expt, assert_only_one=True):
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
            pathlist = findPath(SDIR, [[animal, expt]], "dat", ".pkl", True)
            return pathlist

        pathlist = []
        for SDIR in SDIR_LIST:
            pathlist.extend(_find(SDIR))

        # pathlist = findPath(SDIR, [[animal, expt]], "dat", ".pkl", True)
        
        if assert_only_one:
            assert len(pathlist)==1
            
        return pathlist


    def preprocessGood(self, ver="modeling", params=None):
        """ save common preprocess pipelines by name
        returns a ordered list, which allows for saving preprocess pipeline.
        - ver, string. uses this, unless params given, in which case uses parasm
        - params, list, order determines what steps to take. if None, then uses
        ver, otherwise usesparmas
        """
        if params is None:
            if ver=="modeling":
                # recenter tasks (so they are all similar spatial coords)
                self.recenter(method="each_beh_center")

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
                    self.recenter(method="each_beh_center")
                elif p=="interp":
                    self.interpolateStrokes()
                elif p=="subsample":
                    self.subsampleTrials()
                elif p=="spad_edges":
                    self.recomputeSketchpadEdges()
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
        - apply_to, to apply to "monkey", "stim" or "both".
        [NOTE, only "both" is currently working, since I have to think 
        about in what scenario the others would make sense]
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

                return translateStrokes(strokes, xydelt)

            return F

        if apply_to=="both":
            which_strokes_list = ["strokes_beh", "strokes_task"]
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


    def interpolateStrokesSpatial(self, strokes_ver ="strokes_beh", npts=50):
        """ interpolate in space, to npts pts
        (uniformly sampled)
        """
        from ..tools.stroketools import strokesInterpolate2

        strokes_list = self.Dat[strokes_ver].values
        for i, strokes in enumerate(strokes_list):
            strokes_list[i] = strokesInterpolate2(strokes, 
                N=["npts", npts], base="space")

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
        import pickle
        
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
    def bpl_reload_saved_state(self, path):
        """ reload BPL object, previously saved
        """

        # load BPL
        import pickle
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
        import pickle

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


    def bpl_load_motorprograms(self):
        """ 
        loads best parse (MP) presaved, BPL.
        (see strokesToProgram) for extraction.
        RETURNS:
        - self.Dat adds a new column "motor_program", with a single ctype (motor program)
        - self.BPL, which stores things related to BPL params.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        def _loadMotorPrograms():
            """ 
            """
            from pythonlib.tools.expttools import findPath
            
            # Go thru each dataset, search for its presaved motor programs. Collect
            # into a list of dicts, one dict for each dataset.
            ndat = len(self.Metadats)
            MPdict_by_metadat = {}
            for i in range(ndat):
                
                # find paths
                path = findPath(self.Metadats[i]["path"], [["infer_MPs_from_strokes"]],
                    "params",".pkl",True)
                
                # should be one and only one path
                if len(path)==0:
                    print(self.Metadats[i]["path"])
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
        
        def _extractMP(MPdict_by_metadat, idx_mdat, trialcode, expect_only_one_MP_per_trial=True,
            fail_if_None=True):
            """
            Helper, to extract a single motor program for a signle trial (trialcode), and single
            metadat index (idx_mdat)
            INPUTS:
            - fail_if_None, then fialks if any trial doesnt find.
            NOTES:
            - expect_only_one_MP_per_trial, then each trial only got one parse. will return that MP 
            (instead of a list of MPs)
            RETURNS:
            - list of MPs. Usually is one, if there is only single parse. Unless 
            expect_only_one_MP_per_trial==True, then returns MP (not list). If doesnt fine, then will return None
            """

            # Get the desired dataset
            MPdict = MPdict_by_metadat[idx_mdat]
            trialcode_list = MPdict["indices_list"]
            MP_list = MPdict["MPlist"]

            # Pull out the desired trial        
            tmp = [m for m, t in zip(MP_list, trialcode_list) if t==trialcode]
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

        # check that havent dione before
        assert not hasattr(self, "BPL"), "self.BPL exists. I woint overwrite."
        assert "motor_program" not in self.Dat.columns

        MPdict_by_metadat = _loadMotorPrograms()

        # save things
        self.BPL = {}
        self.BPL["params_MP_extraction"] = {k:MP["params"] for k, MP in MPdict_by_metadat.items()}

        def F(x):
            idx_mdat = x["which_metadat_idx"]
            trialcode = x["trialcode"]
            MP = _extractMP(MPdict_by_metadat, idx_mdat, trialcode)
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

    def bpl_index_to_col_name(self, index):
            tmp = [str(i) for i in index]
            return f"bpl-{'-'.join(tmp)}"

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
        libraries_list = self.BPL["refits"][lib_refit_index]["libraries"]
        grp_by = self.BPL["refits"][lib_refit_index]["libraries_grpby"]
        for L in libraries_list:
            print(L["index_grp"])

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


    ################ PARSES
    # Working with parses, which are pre-extracted and saved int he same directocry as datasets.
    # See ..drawmodel.parsing

    def _loadParses(self):
        """ load all parases in bulk.
        Assumes that they are arleady preprocessed.
        - Loads one parse set for each dataset, and then combines them.
        """
        import pickle
        PARSES = {}
        for ind in self.Metadats.keys():
            
            path = self.Metadats[ind]["path"]
            path_parses = f"{path}/parses.pkl"
            path_parses_params =  f"{path}/parses_params.pkl"

            with open(path_parses, "rb") as f:
                parses = pickle.load(f)
            with open(path_parses_params, "rb") as f:
                parses_params = pickle.load(f)
                
            PARSES[ind] = parses
        return PARSES

    def parsesLoadAndExtract(self, fail_if_empty=True, print_summary=True):
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
            
        PARSES = self._loadParses()

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



    def parsesChooseSingle(self, convert_to_splines=True, add_fake_timesteps=True):
        """ 
        pick out a random single parse and assign to a new column
        - e..g, if want to do shuffle analyses, can run this each time to pick out a random
        parse
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
    
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strokes_parse")

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
    def save_state(self, SDIR_MAIN, SDIR_SUB):
        """
        RETURNS:
        - SDIR_THIS, dir where this saved.
        """
        import os
        import pickle

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
    def plotSingleTrial(self, idx, things_to_plot = ["beh", "beh_tc", "task", "parse", "bpl_mp", "beh_splines"]):
        """ 
        idx, index into Dat, 0, 1, ...
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse, plotDatWaterfall
        dat = self.Dat

        # === Plot a single trial
        ncols=4
        nplots = len(things_to_plot)
        nrows = int(np.ceil(nplots/ncols))

        fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(ncols*3, nrows*3))

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
                if "parse" in dat.columns:
                    strokes = dat["strokes_parse"].values[idx]
                    plotDatStrokes(strokes, ax, each_stroke_separate=True)
            elif thing=="bpl_mp":
                if "motor_program" in dat.columns:
                    from ..bpl.strokesToProgram import plotMP
                    MP = dat["motor_program"].values[idx]
                    plotMP(MP, ax=ax)
            elif thing=="beh_splines":
                assert "strokes_beh_splines" in dat.columns, "need to run convertToSplines first!"
                strokes = dat["strokes_beh_splines"].values[idx]
                plotDatStrokes(strokes, ax, each_stroke_separate=True)
            else:
                assert False

            ax.set_title(thing)
        return fig



            # 

    def plotMultTrials(self, idxs, which_strokes="strokes_beh", return_idxs=False, 
        ncols = 5, titles=None):
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
            else:
                ax.set_title(f"{titles[ind]:.2f}")

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
        """
        import seaborn as sns
        sns.catplot(data=self.Dat, x="tvalday", y="taskgroup", hue="task_stagecategory", 
            row="expt", col="epoch")

    ############### PRINT INFO
    def printOverview(self):
        from pythonlib.tools.pandastools import printOverview
        printOverview(self.Dat)





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

""" stores and operates on datasets"""
import pandas as pd
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt


def _checkPandasIndices(df):
    """ make sure indices are monotonic incresaing by 1.
    """
    assert np.unique(np.diff(df.index)) ==1 

class Dataset(object):
    """ 
    """
    def __init__(self, inputs, append_list=None):
        """
        Load saved datasets. 
        - inputs, is either:
        --- list of strings, where each string is
        full path dataset.
        --- list of tuples, where each tuple is (dat, metadat), where
        dat and metadat are both pd.Dataframes, where each wouyld be the object
        that would be loaded if list of paths.
        - append_list, dict, where each key:value pair will be 
        appendeded to datset as new column, value must be list 
        same length as path_list.
        """

        if isinstance(inputs[0], str):
            self._load_datasets(inputs, append_list)
        else:
            self._store_dataframes(inputs, append_list)
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
            with open(f"{path}/dat.pkl", "rb") as f:
                dat = pickle.load(f)
                dat["which_metadat_idx"] = i
                dat_list.append(dat)
                print("Loaded dataset, size:")
                print(len(dat))


            # Open metadat
            with open(f"{path}/metadat.pkl", "rb") as f:
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



    def preprocessGood(self, ver="modeling"):
        """ save common preprocess pipelines by name
        """
        if ver=="modeling":
            # recenter tasks (so they are all similar spatial coords)
            D.recenter(method="each_beh_center")

            # interpolate beh (to reduce number of pts)
            D.interpolateStrokes()

            # subsample traisl in a stratified manner to amke sure good represnetaiton
            # of all variety of tasks.
            D.subsampleTrials()

            # Recompute task edges (i..e, bounding box)
            D.recomputeSketchpadEdges()
        elif ver=="strokes":
            # interpolate beh (to reduce number of pts)
            D.interpolateStrokes()

            # subsample traisl in a stratified manner to amke sure good represnetaiton
            # of all variety of tasks.
            D.subsampleTrials()
        else:
            print(ver)
            assert False, "not coded"


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


    def subsampleTrials(self, n_rand_to_keep=250, n_fixed_to_keep=50):
        """ subsample trials to make sure get good variety.
        Can run this before doing train test spkit
        - n_rand_to_keep, num random trials to keep for each 
        task_category, for random tasks. Generally try to keep this within
        order of magnitude of fixed tasks, but higher since these random.
        - n_fixed_to_keep, for each fixed task (given by unique_task_name), 
        how many to subsample to?
        """
        import random
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
    def flattenToStrokdat(self, keep_all_cols = True, strokes_ver="strokes_beh",
        cols_to_exclude = ["strokes_beh", "strokes_task", "strokes_parse", "strokes_beh_splines", "parses"]):
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
            self.strokes2splines("strokes_parse", make_new_col=False)

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


    ############# PLOTS
    def plotSingleTrial(self, idx):
        """ 
        idx, index into Dat, 0, 1, ...
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse, plotDatWaterfall
        dat = self.Dat

        # === Plot a single trial
        fig, axes = plt.subplots(1,2, sharex=True, sharey=True)

        # Plot behavior for this trial
        # - the ball marks stroke onset. stroke orders are color coded, and also indicated by numbers.
        strokes = dat["strokes_beh"][idx]
        plotDatStrokes(strokes, axes[0], each_stroke_separate=True)

        # overlay the stimulus
        stim = dat["strokes_task"][idx]
        plotDatStrokes(stim, axes[1], each_stroke_separate=True, plotver="onecolor", add_stroke_number=False, mark_stroke_onset=False, pcol="k")


        # === Plot motor timecourse for a single trial (smoothed velocity vs. time)
        fig, axes = plt.subplots(1,1, sharex=True, sharey=True)
        plotDatStrokesTimecourse(strokes, axes, plotver="vel", label="vel (pix/sec)")

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












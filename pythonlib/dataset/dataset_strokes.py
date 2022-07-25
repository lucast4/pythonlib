""" A bag of strokes, where beh are represented as StrokeClass, and task are
represented as PrimitiveClass and/or datseg tokens. 
Also see 
drawmodel.sf
dataset.sf...

And associated methods for:
- ploting distributions of task and beh strokes
- clustering strokes.

This supercedes all SF (strokefeats) things.

See notebook: analy_spatial_220113 for development.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DatStrokes(object):
    """docstring for DatStrokes"""
    def __init__(self, Dataset, version="beh"):
        """
        PARAMS:
        - version, string, whether each datapoint is "beh" or "task"
        """

        self.Dataset = Dataset
        self.Dat = None
        self.Params = {}
        self.Version = None

        self._prepare_dataset()
        self._extract_strokes_from_dataset(version=version)

    def _prepare_dataset(self):
        """ Prepare dataset before doing strokes extraction
        TODO:
        - check if already ran... if so, skip
        """

        D = self.Dataset
        # Generate all beh calss
        D.behclass_generate_alltrials()
        # For each compute datsegs
        D.behclass_alignsim_compute()
        # Prune cases where beh did not match any task strokes.
        D.behclass_clean()

    def _extract_strokes_from_dataset(self, version="beh"):
        """ Flatten all trials into bag of strokes, and for each stroke
        storing its associated task stroke, and params for that taskstroke
        PARAMS:
        - version, string. if
        --- "beh", then each datapoint is a beh stroke
        --- "task", then is task primitive
        RETURNS:
        - modifies self.Dat to hold dataframe where each row is stroke.
        """

        D = self.Dataset

        # Collect all beh strokes
        list_features = ["circularity", "distcum", "displacement", "angle"]

        DAT_BEHPRIMS = []
        for ind in range(len(D.Dat)):
        #     Beh = D.Dat.iloc[ind]["BehClass"]
            if ind%100==0:
                print(ind)
            T = D.Dat.iloc[ind]["Task"]
            
            # 1) get each beh stroke, both continuous and discrete represntations.
            primlist, datsegs_behlength, datsegs_tasklength, out_combined = D.behclass_extract_beh_and_task(ind)
            
            if version=="beh":
                strokes = primlist
                datsegs = datsegs_behlength
                combined_tuple = (None for _ in range(len(strokes)))
            elif version=="task":
                strokes = [dat[2]["Prim"].Stroke for dat in out_combined] # list of PrimitiveClass
                datsegs = [dat[2] for dat in out_combined] # task version
                combined_tuple = out_combined
                assert datsegs == datsegs_tasklength, "bug?"
            else:
                print(version)
                assert False

            # 2) Information about task (e..g, grid size)
            
            # 2) For each beh stroke, get its infor
            for i, (stroke, dseg, comb) in enumerate(zip(strokes, datsegs, combined_tuple)):
                DAT_BEHPRIMS.append({
                    'Stroke':stroke,
                    'datseg':dseg})
                
                # get features for this stroke
                for f in list_features:
                    DAT_BEHPRIMS[-1][f] = stroke.extract_single_feature(f)
                    
                # Which task kind?
                DAT_BEHPRIMS[-1]["task_kind"] =  T.get_task_kind()
                
                ### Task information
                DAT_BEHPRIMS[-1]["gridsize"] = T.PlanDat["TaskGridClass"]["Gridname"]

                # Info linking back to dataset
                DAT_BEHPRIMS[-1]["dataset_trialcode"] = D.Dat.iloc[ind]["trialcode"]
                DAT_BEHPRIMS[-1]["stroke_index"] = i

                # Specific things for Task
                if version=="task":
                    DAT_BEHPRIMS[-1]["aligned_beh_inds"] = comb[0]
                    DAT_BEHPRIMS[-1]["aligned_beh_strokes"] = comb[1]

        # Expand out datseg keys each into its own column (for easy filtering/plotting later)
        for DAT in DAT_BEHPRIMS:
            for k, v in DAT["datseg"].items():
                DAT[k] = v
                if k=="gridloc":
                    DAT["gridloc_x"] = v[0]
                    DAT["gridloc_y"] = v[1]
                    

        # generate a table with features
        self.Dat = pd.DataFrame(DAT_BEHPRIMS)

        # make a new column with the strok eexposed, for legacy code
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x):
            strok = x["Stroke"]() 
            return strok
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strok")

        print("This many beh strokes extracted: ", len(DAT_BEHPRIMS))       

        self.Version = version

        # DEBUGGING
        if False:
            # Plot a random task
            import random
            ind = random.choice(range(len(D.Dat)))

            D.plotSingleTrial(ind)
            Task = D.Dat.iloc[ind]["Task"]
            print("Task kind: ", Task.get_task_kind())
            Beh = D.Dat.iloc[ind]["BehClass"]
            T.plotStrokes()

            T.tokens_generate()

            Beh = D.Dat.iloc[ind]["BehClass"]
            Beh.Alignsim_taskstrokeinds_sorted

            Beh.alignsim_plot_summary()

            Beh.alignsim_extract_datsegs()

            T = D.Dat.iloc[ind]["Task"]
            len(T.Strokes)

    def _process_strokes(self, align_to_onset = True, min_stroke_length_percentile = 2, 
        min_stroke_length = 50, max_stroke_length_percentile = 99.5, centerize=False, 
        rescale_ver=None):
        """ To do processing of strokes, e.g,, centerizing, etc.
        - Only affects the "strok" key in self.Dat
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

        self.Dat = preprocessStroks(self.Dat, params)

        # Note down done preprocessing in params
        self.Params["params_preprocessing"] = params



    ########################## EXTRACTIONS
    def extract_strokes(self, version="list_arrays", inds=None, ver_behtask=None):
        """ Methods to extract strokes across all trials
        PARAMS:
        - version, string, what format to output
        - inds, indices, if None then gets all
        - ver_behtask, None means get whatever is active version. otherwies:
        --- task
        --- beh (gets the best matching one)
        RETURNS:
        - strokes
        """

        if inds is None:
            inds = range(len(self.Dat))

        if ver_behtask is None:
            ver_behtask = self.Version
            
        def _extractsinglestroke(i):
            """ extracts strok if exists, otherwise gets Stroke()
            """
            if ver_behtask==self.Version:
                if "strok" in self.Dat.columns:
                    return self.Dat.iloc[i]["strok"]
                else:
                    return self.Dat.iloc[i]["Stroke"]()
            elif ver_behtask=="task":
                assert False, "code it"
            elif ver_behtask=="beh":
                # Then pull out the beh taht matches this task stroek the best
                assert "aligned_beh_strokes_disttotask" in self.Dat.columns, "need to extract this first"
                strokes_beh = [S() for S in self.Dat.iloc[i]["aligned_beh_strokes"]]
                distances = self.Dat.iloc[i]["aligned_beh_strokes_disttotask"]
                return strokes_beh[np.argmin(distances)]
            else:
                assert False

        assert isinstance(inds, list)

        # -----
        if version=="list_arrays":
            # List of np arrays (i.e., "strokes" type)
            strokes = [_extractsinglestroke(i) for i in inds]
        elif version=="list_list_arrays":
            # i.e., like multiple strokes...
            strokes = [[_extractsinglestroke(i)] for i in inds]
        else:
            print(version)
            assert False
        return strokes

    ###################### RELATED to original dataset
    def _dataset_index(self, ind, return_row=False):
        """ For this index (in self), what is its index into self.Dataset?
        PARAMS:
        - ind, int index into self.Dat
        - return_row, bool (FAlse) if True return the dataframe row. if False,
        just the index (df.index)
        """
        tc = self.Dat.iloc[ind]["dataset_trialcode"]
        row = self.Dataset.Dat[self.Dataset.Dat["trialcode"] == tc]
        assert len(row)==1

        if return_row:
            return row
        else:
            return row.index.tolist()[0]

    def dataset_extract(self, colname, ind):
        """ Extract value for this colname in original datset,
        for ind indexing into the strokes (self.Dat)
        """

        return self._dataset_index(ind, True)[colname].tolist()[0]

    def dataset_append_column(self, colname):
        """ Append a column to self.Dat from original datsaet.
        Overwrites if it already exists
        """
        valsthis = [self.dataset_extract(colname, i) for i in range(len(self.Dat))]
        self.Dat[colname] = valsthis
        print(f"Appended {colname} to self.Dat")


    def dataset_slice_by(self, key, list_vals, return_index=False):
        """ Extract slice of dataset using key-val pairs that may not yet 
        exist in self.Dat, but may be extracted by mapping back from orignal
        Dataset. First checks if this key exitss, if not then appends it as a
        new column.
        PARAMS:
        - key, string name
        - list_vals, list of vals, where keeps rows in dataset that are in this list
        - return_index, bool (False), ifTrue then returns just the index. otherwise the dataframe
        RETURNS:
        - df, sliced dataset or index/
        """

        if key not in self.Dat.columns:
            # Then pull it from dstaset
            self.dataset_append_column(key)

        dfthis = self.Dat[self.Dat[key].isin(list_vals)]

        if return_index:
            return dfthis.index.tolist()
        else:
            return dfthis

    def dataset_slice_by_mult(self, filtdict, return_indices=False, reset_index=True):
        """ Get subset of dataset, filtering in multiple intersecting ways
        Gets the key-val from original datset, if theyu are not arleady present
        PARAMS:
        - filtdict, keys are columns, and vals are lists of items to keep.
        - return_indices, bool (False), then returns indices. otherwise dataframe
        RETURNS:
        - pandas dataframe (indices reset) or indices
        """
        from pythonlib.tools.pandastools import filterPandas

        # 1) If a key doesnt already exist in self.Dat, then extract it from Dataset
        list_keys = filtdict.keys()
        for k in list_keys:
            if k not in self.Dat.columns:
                self.dataset_append_column(k)

        # 2) Filter
        return filterPandas(self.Dat, filtdict, return_indices=return_indices, reset_index=reset_index)


    ######################### SUMMARIZE
    def print_summary(self):
        assert False, "not coded"
        print(DF_PRIMS["task_kind"].value_counts())
        DF_PRIMS.iloc[0]


    ####################### PLOTS
    def plot_single(self, ind, ax=None):
        """ Plot a single stroke
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes
        if ax==None:
            fig, ax = plt.subplots(1,1)
        S = self.Dat.iloc[ind]["Stroke"]
        plotDatStrokes([S()], ax, clean_ordered_ordinal=True)
        return ax

    def plot_multiple(self, list_inds, ver_behtask=None, titles=None):
        """ Plot mulitple strokes, each on a subplot
        PARAMS:
        - ver_behtask, "task", "beh", or None(default).
        """
        strokes = self.extract_strokes("list_list_arrays", 
            inds = list_inds, ver_behtask=ver_behtask)
        self.Dataset.plotMultStrokes(strokes, titles=titles)

    def plot_strokes_overlaid(self, inds, ax=None, nmax = 50, color_by="order", ver_behtask=None):
        """ Plot strokes all overlaid on same plot
        Colors strokes by order
        PARAMS;
        - inds, indices into self.Dat (must be less than nmax)
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes
        if ax is None:
            fig, ax = plt.subplots(1,1)

        assert len(inds)<=nmax, "too many strokes to oveerlay..."
        
        strokes = self.extract_strokes("list_arrays", inds, ver_behtask=ver_behtask)
        # strokes = [S() for S in self.Dat.iloc[inds]["Stroke"]]
        if color_by=="order":
            plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, alpha=0.5)
        else:
            plotDatStrokes(strokes, ax, clean_ordered=True, alpha=0.5)

        return ax

    def plot_egstrokes_grouped_in_subplots(self, task_kind="prims_on_grid", 
        key_subplots="shape_oriented",
        key_to_extract_stroke_variations_in_single_subplot = "gridloc", 
        n_examples = 2, color_by="order", ver_behtask=None):
        """
        PARAMS:
        - task_kind, string, indexes into the task_kind column
        --- e..g, {"character", "prims_on_grid"}
        - key_subplots, string, each level of this grouping variable will have its
        own subplot.
        - key_to_extract_stroke_variations_in_single_subplot, string, for each subplot, 
        how to extract exabple strokes. e..g, if "gridloc", then strokes will be at
        variable locations in the subplot. This is ignored if task_kind=="character", because
        that doesnt have structured variation in these params.
        """
        from pythonlib.tools.plottools import subplot_helper
        from pythonlib.tools.pandastools import extract_trials_spanning_variable
        import random

                
        if task_kind=="character":
            key_to_extract_stroke_variations_in_single_subplot = None

        # 0) Static params:
        F = {
            "task_kind":[task_kind],
        }

        # One plot for each level
        subplot_levels = sorted(self.Dat[key_subplots].unique().tolist())
        getax, figholder, nplots = subplot_helper(4, 10, len(subplot_levels), SIZE=4)
        for i, level in enumerate(subplot_levels):
            
            # - update the shape
            F[key_subplots] = level

            # 2) For each combo of features, select a random case of it.
            if key_to_extract_stroke_variations_in_single_subplot is None:
                # Use entire pool, not specific variation
                list_idx = self.dataset_slice_by_mult(F, return_indices=True)
                if len(list_idx)>=n_examples:
                    inds = random.sample(list_idx, n_examples)[:n_examples]
                else:
                    inds = [None for _ in range(n_examples)]
                list_inds = inds
            else:
                # 1) get all unique values for a given key
                def bad(x):
                    # Return True is this value is a nan
                    if isinstance(x, (tuple, list)):
                        return False
                    else:
                        return np.isnan(x)
                vals_vary = self.Dat[key_to_extract_stroke_variations_in_single_subplot].unique().tolist()
                vals_vary = [v for v in vals_vary if not bad(v)]
                list_inds, _ = extract_trials_spanning_variable(self.Dat, 
                    key_to_extract_stroke_variations_in_single_subplot, vals_vary, n_examples, F)

            # 3) pull out these examples and plot
            list_inds = [i for i in list_inds if i is not None]
            if len(list_inds)>0:
                ax = getax(i)[1]
                self.plot_strokes_overlaid(list_inds, ax=ax, color_by=color_by, ver_behtask=ver_behtask)   
                ax.set_title(level)

    ############################### DISTANCES (scoring)
    def _dist_strok_pair(self, strok1, strok2):
        """ compute doistance between strok1 and 2
        uses by defaiult hausdorff mean
        """
        from pythonlib.drawmodel.strokedists import distMatrixStrok
        return distMatrixStrok([strok1], [strok2], convert_to_similarity=False).squeeze()

    def dist_alignedbeh_to_task(self):
        """ Get distance from each beh stroke to their matched task stroke (i.e,, N to 1).
        Only applies for self.Version=="task"
        RETURNS:
        - new columnin self.Dat, aligned_beh_strokes_disttotask, np array, len num beh strokes assginged
        to that task stroke, distance in hd
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        assert self.Version=="task"

        def F(x):
            
            # task stroke
            strok_task = x["strok"]

            # get distances against all beh strokes
            list_dist = np.array([self._dist_strok_pair(strok_task, strok_beh()) for strok_beh in x["aligned_beh_strokes"]])
            return list_dist

        self.Dat = applyFunctionToAllRows(self.Dat, F, "aligned_beh_strokes_disttotask")


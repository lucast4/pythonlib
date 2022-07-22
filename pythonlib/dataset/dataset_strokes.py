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

    def _process_strokes_inputed(self, strokes, align_to_onset = True, min_stroke_length_percentile = 2, 
        min_stroke_length = 50, max_stroke_length_percentile = 99.5, centerize=False, 
        rescale_ver=None):
        """
        Process the inpute strokes 
        RETURNS:
        - strokes, without modifying anything in  strol
        """
        from ..drawmodel.sf import preprocessStroksList 

        params = {
            "align_to_onset":align_to_onset,
            "min_stroke_length_percentile":min_stroke_length_percentile,
            "min_stroke_length":min_stroke_length,
            "max_stroke_length_percentile":max_stroke_length_percentile,
            "centerize":centerize,
            "rescale_ver":rescale_ver
        }

        return preprocessStroks(strokes, params)


    def _process_strokes(self, align_to_onset = True, min_stroke_length_percentile = 2, 
        min_stroke_length = 50, max_stroke_length_percentile = 99.5, centerize=False, 
        rescale_ver=None):
        """ To do processing of strokes, e.g,, centerizing, etc.
        - Only affects the "strok" key in self.Dat
        RETURNS:
        - Modifies self.Dat["strok"]
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

    def clean_data(self, methods=[]):
        """ Combine methods for pruning dataset to get clean data in specific ways
        PARAMS;
        - methods, list of str, each a method to run in order
        RETURNS:
        - Modifies self.Dat
        """

        for meth in methods:
            if meth=="remove_if_multiple_behstrokes_per_taskstroke":
                # Only keep trials with good behavior
                # - takes at most one beh strok to get this task strok.
                list_inds_bad = []
                for i in range(len(self.Dat)):
                    indtrial = self._dataset_index(i)
                    indstroke = self.Dat.iloc[i]["stroke_index"]
                    tok = self.dataset_call_method("behclass_find_behtask_token", [indtrial, indstroke], {"version":"beh"})
                    n_beh_strokes = len(tok[0])
                    if n_beh_strokes>1:
                        # Then this beh stroke is part of multiple beh strokes targeting same task strokes.
                        list_inds_bad.append(i)
                print("This many cases with >1 beh stroke needed to completed a task stroke: ", len(list_inds_bad))
                self.Dat = self.Dat.drop(list_inds_bad).reset_index(drop=True)
            elif meth=="visual_accuracy":
                # Remove if innacurate 
                assert False, "in progress"
                # Visal accuracy (stroke beh vs. stroke task)
                i = 104
                indtrial = DS._dataset_index(i)
                indstroke = DS.Dat.iloc[i]["stroke_index"]
                tok = D.behclass_find_behtask_token(indtrial, indstroke, version="beh")

                beh_stroke = tok[1][0]
                task_prim = tok[2]["Prim"]

                D.plotMultStrokes([[beh_stroke(), task_prim.Stroke()]]) 

                # Compute distance
                DS._dist_strok_pair(beh_stroke(), task_prim.Stroke())

                # TODO: filter based on this score.
            else:
                print(meth)
                assert False, "code it"



    ######################### MOTOR TIMING
    def timing_extract_basic(self):
        """Extract basis stats for timing, such as time of onset and offset,
        - Looks into Dataset to find there
        RETURNS:
        - Appends columns to self.Dat, including "time_onset", "time_offset" 
        (of strokes, relative to start of trial.)
        """

        # 1) Collect onset and offset of each stroke by refereing back to original dataset.
        list_ons = []
        list_offs = []
        for ind in range(len(self.Dat)):
            me = self.dataset_extract("motorevents", ind)
            indstrok = self.Dat.iloc[ind]["stroke_index"]
            
            # onset and offset
            on = me["ons"][indstrok]
            off = me["offs"][indstrok]
            # Note: I have checked that these match exactly what would get if use
            # neural "Session" object to get timings (which goes thru getTrials ...)
            
            # save it
            list_ons.append(on)
            list_offs.append(off)
            
        self.Dat["time_onset"] = list_ons
        self.Dat["time_offset"] = list_offs
        print("DONE!")
            

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
            inds = list(range(len(self.Dat)))

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
                strokes_task = self.dataset_extract("strokes_task", i)
                print("HACKY (extract_strokes) for task, taking entire task")
                return np.concatenate(strokes_task, axis=0)
                # return strokes_task[0]
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
        for ind indexing into the strokes (DatsetStrokes.Dat)
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

    def dataset_call_method(self, method, args, kwargs):
        """
        return D.<method>(*args, **kwargs)
        """
        return getattr(self.Dataset, method)(*args, **kwargs)

    ######################### SUMMARIZE
    def print_summary(self):
        assert False, "not coded"
        print(DF_PRIMS["task_kind"].value_counts())
        DF_PRIMS.iloc[0]


    ####################### PLOTS
    def plot_single_strok(self, strok, ax=None):
        """ plot a single inputed strok on axis.
        INPUT:
        - strok, np array,
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes
        plotDatStrokes([strok], ax, clean_ordered_ordinal=True)

    def plot_single(self, ind, ax=None):
        """ Plot a single stroke
        """
        if ax==None:
            fig, ax = plt.subplots(1,1)
        S = self.Dat.iloc[ind]["Stroke"]
        self.plot_single_strok(S(), ax)
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
        - task_kind, string, indexes into the task_kind column. Leave None to keep any
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
        if task_kind is None:
            F = {}  
        else:
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
                ax = getax(i)
                self.plot_strokes_overlaid(list_inds, ax=ax, color_by=color_by, ver_behtask=ver_behtask)
                ax.set_title(level)

    def plot_examples_grid(self, col_grp="shape_oriented", col_levels=None, nrows=2,
            flip_plot=False):
        """ 
        Plot grid of strokes, where cols are (e.g.) shapes and rows are
        example trials
        PARAMS:
        - flip_plot, bool, if True, then cols are actually plotted as rows
        """
        from pythonlib.tools.pandastools import extract_trials_spanning_variable        
        from pythonlib.tools.plottools import plotGridWrapper
        from pythonlib.drawmodel.strokePlots import plotDatStrokes

        outdict = extract_trials_spanning_variable(self.Dat, "shape_oriented", 
            col_levels, n_examples=2, return_as_dict=True)[0]
        list_row = []
        list_col = []
        list_inds = []
        for col, (shape, inds) in enumerate(outdict.items()):
            for row, index in enumerate(inds):
                list_inds.append(index)
                list_row.append(row)
                list_col.append(col)

        strokes = self.Dat.iloc[list_inds]["strok"].values
        def plotfunc(strok, ax):
            plotDatStrokes([strok], ax)
        if flip_plot:
            plotGridWrapper(strokes, plotfunc, list_row, list_col[::-1])
        else:
            plotGridWrapper(strokes, plotfunc, list_col, list_row)


    def plotStrokOrderedByLabel(labels, SF, labels_in_order=None):
        """ plot example (rows) of each label(cols), ordred as in 
        labels_in_order.
        INPUTS:
        - labels_in_order, if None, then will use sorted(set(labels))
        - labels, list, same len as SF
        """

        from pythonlib.drawmodel.strokePlots import plotStroksInGrid
        # === for each cluster, plot examples
        if labels_in_order is None:
            labels_in_order = sorted(list(set(labels)))

        indsplot =[]
        titles=[]
        for ii in range(3):
            # collect inds
            for lab in labels_in_order:
                inds = [i for i, l in enumerate(labels) if l==lab]
                indsplot.append(random.sample(inds, 1)[0])
                if ii==0:
                    titles.append(lab)
                else:
                    titles.append('')

        # plot    
        stroklist = [SF["strok"].values[i] for i in indsplot]
        fig = plotStroksInGrid(stroklist, ncols=len(labels_in_order), titlelist=titles);

    ############################## GROUPING
    def grouping_get_inner_items(self, groupouter="shape_oriented", groupinner="index"):
        from pythonlib.tools.pandastools import grouping_get_inner_items
        groupdict = grouping_get_inner_items(self.Dat, groupouter, groupinner)
        return groupdict


    def grouping_append_and_return_inner_items(self, list_grouping_vars=["shape_oriented", "gridloc"]):
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
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items

        groupdict = grouping_append_and_return_inner_items(self.Dat, list_grouping_vars,
            "index")

        return groupdict


    ################################ SIMILARITY/CLUSTERING
    def cluster_compute_sim_matrix(self, inds, do_preprocess=False, label_by="shape_oriented"):
        """ Compute similarity matrix between each pair of trials in inds
        PARAMS:
        - label_by, either string (col name) or list of string (will make a new grouping 
        that conjunction of these)
        """
        from ..drawmodel.sf import computeSimMatrixGivenBasis
        from ..cluster.clustclass import Clusters

        strokes = self.extract_strokes("list_arrays", inds)


        # Preprocess storkes if descired
        if do_preprocess:
            strokes = self._process_strokes_inputed(strokes, min_stroke_length_percentile = None, 
                min_stroke_length = None, max_stroke_length_percentile = None)

        simmat = computeSimMatrixGivenBasis(strokes, strokes, 
            rescale_strokes_ver="stretch_to_1", distancever="euclidian_diffs", npts_space=50) 

        # labels
        if isinstance(label_by, list):
            assert False, "to do, take conjuctins, see grouping_append_and_return_inner_items"

        label_by = "shape_oriented"
        labels = self.Dat.iloc[inds][label_by]

        # Make cluster class
        Cl = Clusters(X = simmat, labels=labels)

        # Plot
        Cl.plot_heatmap_data()
        Cl.plot_heatmap_data(sortver=0)
        Cl.plot_heatmap_data(sortver=2)

        return Cl


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


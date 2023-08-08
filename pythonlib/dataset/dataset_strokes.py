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
from pythonlib.globals import PATH_DATASET_BEH
from pythonlib.tools.plottools import savefig
import seaborn as sns
from pythonlib.tools.pandastools import append_col_with_grp_index

class DatStrokes(object):
    """docstring for DatStrokes"""
    def __init__(self, Dataset, version="beh"):
        """
        PARAMS:
        - version, string, whether each datapoint is "beh" or "task"
        """

        self.Dataset = Dataset.copy()
        self.Dat = None
        self.Params = {}
        self.Version = None

        self._prepare_dataset()
        self._extract_strokes_from_dataset(version=version)
        self._clean_preprocess()


    def _clean_preprocess(self):
        """ All things that generally want to run for preprocessing
        """

        ############ GIVE SHAPES A HASH 
        # Useful for shapes with non-informative name (e..g, novel prims), 
        # append a suffix to name
        # And replace shape oriented.
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # Method 1: Use task hash
        self.dataset_append_column("character")

        # give each character an index
        list_char = sorted(self.Dat["character"].unique().tolist())
        map_char_to_index = {}
        for i, char in enumerate(list_char):
            map_char_to_index[char] = i

        def F(x):
            idx_char = map_char_to_index[x['character']]
            return f"{x['shape_oriented']}|{idx_char}"
        self.Dat = applyFunctionToAllRows(self.Dat, F, "shape_char")


    def _prepare_dataset(self):
        """ Prepare dataset before doing strokes extraction
        TODO:
        - check if already ran... if so, skip
        """

        D = self.Dataset
        if D.behclass_check_if_tokens_extracted() ==False:
            D.behclass_preprocess_wrapper()
            # # Generate all beh calss
            # D.behclass_generate_alltrials()
            # # For each compute datsegs
            # D.behclass_alignsim_compute()

        # Prune cases where beh did not match any task strokes.
        D.behclass_clean()

        # Sanity check that all gridloc are relative the same grid (across trials).
        D.taskclass_tokens_sanitycheck_gridloc_identical()

    def _extract_strokes_from_dataset(self, version="beh", include_scale=True, 
            tokens_extract_keys=None, tokens_get_relations=True):
        """ Flatten all trials into bag of strokes, and for each stroke
        storing its associated task stroke, and params for that taskstroke
        PARAMS:
        - version, string. if
        --- "beh", then each datapoint is a beh stroke. each row is a single beh stroke, includes
        all strokes, ordered correctly.
        --- "task", then is task primitive. each row is a single task prim, based on order
        that each prim gotten (touched for first time) by beh
        - tokens_extract_keys, either None (does nthign) or str, in whcih case extracts this
        key from tokens and appends as a new column in self.Dat
        - tokens_get_relations, bool, if true, does preproess of toekns so they have relatioms
        and sequence context.
        RETURNS:
        - modifies self.Dat to hold dataframe where each row is stroke.
        """
        from pythonlib.drawmodel.tokens import Tokens

        assert tokens_extract_keys is None, "not coded. keeps datsegs anyway, so dont need it."
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
            primlist, datsegs_behlength, datsegs_tasklength, out_combined = D.behclass_extract_beh_and_task(ind, include_scale=include_scale)
                
            if version=="beh":
                strokes = primlist
                datsegs = datsegs_behlength
                out_combined = (None for _ in range(len(strokes)))
            elif version=="task":
                strokes = [dat[2]["Prim"].Stroke for dat in out_combined] # list of PrimitiveClass
                datsegs = [dat[2] for dat in out_combined] # task version
                assert datsegs == datsegs_tasklength, "bug?"
            else:
                print(version)
                assert False

            # PREPROCESS THE TOKENS
            try:
                if tokens_get_relations:
                    # get this ind's sequence context
                    Tk = Tokens(datsegs)
                    Tk.sequence_context_relations_calc()
                    datsegs = Tk.Tokens
            except Exception as err:
                # fig, ax = plt.subplots()
                # self.plot_single_overlay_entire_trial(ind, ax=ax, overlay_beh_or_task="beh", SIZE=2)
                # fig, ax = plt.subplots()
                # self.plot_single_overlay_entire_trial(ind, ax=ax, overlay_beh_or_task="task", SIZE=2)
                for tok in datsegs:
                    print(tok)
                print(ind)
                print(D.Dat.iloc[ind])
                raise err


            # 2) For each beh stroke, get its infor
            len_strokes = len(strokes)
            for i, (stroke, dseg, comb) in enumerate(zip(strokes, datsegs, out_combined)):
                DAT_BEHPRIMS.append({
                    'Stroke':stroke,
                    'datseg':dseg})
                
                # get features for this stroke
                for f in list_features:
                    DAT_BEHPRIMS[-1][f] = stroke.extract_single_feature(f)

                # Which task kind?
                DAT_BEHPRIMS[-1]["task_kind"] =  T.get_task_kind()
                DAT_BEHPRIMS[-1]["grid_ver"] =  T.get_grid_ver()
                
                ### Task information
                DAT_BEHPRIMS[-1]["gridsize"] = T.PlanDat["TaskGridClass"]["Gridname"]

                # Info linking back to dataset
                DAT_BEHPRIMS[-1]["dataset_trialcode"] = D.Dat.iloc[ind]["trialcode"]
                DAT_BEHPRIMS[-1]["stroke_index"] = i
                DAT_BEHPRIMS[-1]["stroke_index_fromlast"] = i - len(strokes) # counting back from last stroke: -1, -2, ...

                # Specific things for Task
                if version=="task":
                    DAT_BEHPRIMS[-1]["aligned_beh_inds"] = comb[0]
                    DAT_BEHPRIMS[-1]["aligned_beh_strokes"] = comb[1]

                # Get gap infrmation 
                # (preceding gap)

        # Expand out datseg keys each into its own column (for easy filtering/plotting later)
        EXCLUDE = ["width", "height", "diag", "max_wh", "Prim", "rel_from_prev", "start", "h_v_move_from_prev", "start", "ind_behstrokes"]
        for DAT in DAT_BEHPRIMS:
            for k, v in DAT["datseg"].items():
                if k not in EXCLUDE:
                    DAT[k] = v
                    if k=="gridloc":
                        if DAT["grid_ver"] == "on_grid":
                            DAT["gridloc_x"] = v[0]
                            DAT["gridloc_y"] = v[1]
                        else:
                            assert v is None, "for not on grid, how are there gridlocs? this assumption is wrong?"
                            DAT["gridloc_x"] = None
                            DAT["gridloc_y"] = None

        # generate a table with features
        self.Dat = pd.DataFrame(DAT_BEHPRIMS)

        # make a new column with the strok eexposed, for legacy code
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x):
            strok = x["Stroke"]() 
            return strok
        self.Dat = applyFunctionToAllRows(self.Dat, F, "strok")

        print("This many strokes extracted: ", len(DAT_BEHPRIMS))       

        self.Version = version

        # Other preprocessing
        self.strokerank_extract_semantic()

        # seq context
        self.Dat = append_col_with_grp_index(self.Dat, ["CTXT_loc_next", "CTXT_shape_next"], "CTXT_locshape_next")
        self.Dat = append_col_with_grp_index(self.Dat, ["CTXT_loc_prev", "CTXT_shape_prev"], "CTXT_locshape_prev")
        
        self.dataset_append_column("epoch")
        
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

    def clean_data(self, methods, params=None):
        """wrapper, legacy"""
        return self.clean_preprocess_data(methods, params)

    def clean_preprocess_data(self, methods=None, params = None, shape_key = "shape"):
        """ Combine methods for pruning dataset to get clean data in specific ways
        Also redefines the shape name.
        PARAMS;
        - methods, list of str, each a method to run in order
        - params, optional, dict with key the method string
        - shape_key, which field to copy to "shape_oriented"
        RETURNS:
        - Modifies self.Dat
        """
        
        print("clean_preprocess_data...")
        if methods is None:
            methods = []
        for meth in methods:
            print(f"len of DS.Dat = {len(self.Dat)}, before running... {meth}")
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
            elif meth=="prune_if_shape_has_low_n_trials":
                # Remove rows for shapes that have not neough trials.
                # nmin = 5
                self.shape_switch_to_this_kind(shape_key)

                nmin = params["prune_if_shape_has_low_n_trials"][0]

                list_shapes = self.Dat["shape_oriented"].unique().tolist()
                list_shapes_keep = []
                for shape in list_shapes:
                    n = sum(self.Dat["shape_oriented"]==shape)
                    if n>=nmin:
                        print("Keeping shape:", shape, "n=", n)
                        list_shapes_keep.append(shape)
                    else:
                        print("Excluding shape:", shape, "n=", n)

                self.filter_dataframe({"shape_oriented":list_shapes_keep}, True)
            elif meth=="stroke_too_short":
                # run this to evaluate what is good length as thershold
                # self.plot_multiple_after_slicing_within_range_values
                print("Doing...:", meth)
                assert "min_stroke_length" in params.keys()
                self.Dat = self.Dat[self.Dat["distcum"]>=params["min_stroke_length"]].reset_index(drop=True)
            elif meth=="beh_task_dist_too_large":
                # prune strokes that are outliers.
                # inds = DS.Dat[DS.Dat["dist_beh_task_strok"]>37].index.tolist()
                # DS.plot_multiple(inds);
                print("Doing...:", meth)
                assert "min_beh_task_dist" in params.keys()
                self.Dat = self.Dat[self.Dat["dist_beh_task_strok"]<=params["min_beh_task_dist"]].reset_index(drop=True)
            else:
                print(meth)
                assert False, "code it"

            print("New len: ", len(self.Dat))
        self.Dat = self.Dat.reset_index(drop=True)


    ######################### PREP THE DATASET
    def distgood_compute_beh_task_strok_distances(self):
        """ For each beh stroke, get its distance to matching single task strok
        RETURNS:
        - adds to dataframe, column: "dist_beh_task_strok"
        """

        list_inds = list(range(len(self.Dat)))

        list_strok_beh = self.extract_strokes(inds=list_inds, ver_behtask="beh")
        list_strok_task = self.extract_strokes(inds=list_inds, ver_behtask="task_aligned_single_strok")
        
        # compute distances    
        dists = self._dist_alignedtask_to_beh(list_inds)   

        # Save it
        self.Dat["dist_beh_task_strok"] = dists
        print("Added column: dist_beh_task_strok")


    ######################### MOTOR TIMING
    def timing_extract_basic(self):
        """Extract basis stats for timing, such as time of onset and offset,
        - Looks into Dataset to find there. 
        Also gets gap duration and distance information. i.e. for each stroke collects
        its preceding gap.
        RETURNS:
        - Appends columns to self.Dat, including "time_onset", "time_offset" 
        (of strokes, relative to start of trial.)
        """

        # 1) Collect onset and offset of each stroke by refereing back to original dataset.
        list_ons = []
        list_offs = []
        list_gap_dur = []
        list_gap_dist = []

        for ind in range(len(self.Dat)): 
            me = self.dataset_extract("motorevents", ind)
            mt = self.dataset_extract("motortiming", ind)
            indstrok = self.Dat.iloc[ind]["stroke_index"]
            strokthis = self.Dat.iloc[ind]["strok"]

            ################## STROKE INFORMATION
            # onset and offset
            on = me["ons"][indstrok]
            off = me["offs"][indstrok]
            # Note: I have checked that these match exactly what would get if use
            # neural "Session" object to get timings (which goes thru getTrials ...)
            
            # save it
            list_ons.append(on)
            list_offs.append(off)

            ############### GAP INFORMATION
            ## Preceding gap
            if indstrok==0:
                # the first strok is time from raise
                gap_from_prev_dur = me["ons"][indstrok] - me["raise"]
                assert gap_from_prev_dur==mt["time_raise2firsttouch"], "just sanity"
            else:
                gap_from_prev_dur = me["ons"][indstrok] - me["offs"][indstrok-1]
            list_gap_dur.append(gap_from_prev_dur)

            ## Distance of current strok onset from prewvious strok offset
            def _eucl_dist(pt1, pt2):
                return np.sum((pt1 - pt2)**2)**0.5

            if indstrok==0:
                # distance from fixation to first touch
                off_pt_prev_stroke = self.dataset_extract("origin", ind)
                on_pt_next_stroke = strokthis[0, :2]
                gap_from_prev_dist = _eucl_dist(off_pt_prev_stroke, on_pt_next_stroke)
                assert np.isclose(gap_from_prev_dist, mt["dist_raise2firsttouch"]), "just sanity"
            else:
                strokes_beh = self.dataset_extract("strokes_beh", ind)
                assert np.all(strokes_beh[indstrok] == strokthis), "just sanity"
                off_pt_prev_stroke = strokes_beh[indstrok-1][-1, :2]
                on_pt_next_stroke = strokes_beh[indstrok][0, :2]
                gap_from_prev_dist = _eucl_dist(off_pt_prev_stroke, on_pt_next_stroke)
            list_gap_dist.append(gap_from_prev_dist)
            
        self.Dat["time_onset"] = list_ons
        self.Dat["time_offset"] = list_offs
        self.Dat["time_duration"] = self.Dat["time_offset"] - self.Dat["time_onset"]
        self.Dat["velocity"] = self.Dat["distcum"]/self.Dat["time_duration"]
        self.Dat["gap_from_prev_dur"] = list_gap_dur
        self.Dat["gap_from_prev_dist"] = list_gap_dist

        print("DONE!")
            

    ########################## EXTRACTIONS
    def extract_strokes_example_for_this_shape(self, shape, 
        ver_behtask="task_aligned_single_strok"):
        """ return either beh or task, a single exmaple for this shape.
        Always takes the first index it finds.
        PARAMS:
        - shape, str, value for column shape_oriented
        RETURNS:
        - np array
        """
        inds = self.find_indices_this_shape(shape, return_first_index=True)
        assert len(inds)==1
        return self.extract_strokes(inds=inds, ver_behtask=ver_behtask)[0]



    def extract_strokes_as_velocity(self, inds):
        """ Returns teh default stroeks (in self.Version) as velocity profiles
        RETURNS:
        - list of strok (np array), each (N,3) where columns are (xvel, yvel, time)
        (or strok can be None, if too short to get velocity)
        """
        list_strokes = self.extract_strokes(version="list_list_arrays", inds=inds)
        list_strokes_vel = self.Dataset._extractStrokeVels(list_strokes, remove_time_column=False, version="vel")
        if False:
            DS.Dataset.plotMultStrokesTimecourse(list_strokes, plotver=version, 
                align_to="first_touch")
        for strokes in list_strokes_vel:
            assert len(strokes)==1
        list_strokes_vel = [strokes[0] for strokes in list_strokes_vel] # convert to list of strok
        return list_strokes_vel

    def extract_datseg_tokens(self, ind):
        """ return the datseg tokens for the entire trial this ind is in.
        In order of beh.
        RETURNS:
        - tokens, tuple, in beh order.
        """
        tokens = tuple(self.context_extract_strokeslength_list(ind, "datseg")[0])
        return tokens

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
            elif ver_behtask=="task_entire":
                strokes_task = self.dataset_extract("strokes_task", i)
                print("HACKY (extract_strokes) for task, taking entire task")
                return np.concatenate(strokes_task, axis=0)
                # return strokes_task[0]
            elif ver_behtask=="task_aligned_single_strok":
                # Return the best matching single taskstroke
                # This is using the datseg alignment (so considers alginment of
                # entire set of task and beh strokes)
                ind_strok_task = self.Dat.iloc[i]["ind_taskstroke_orig"]
                strokes_task = self.dataset_extract("strokes_task", i) 
                return strokes_task[ind_strok_task]
            elif ver_behtask=="task_aligned_single_primitiveclass":
                # A sinngle PrimitiveClass() object, represnting this task stroke
                ind_strok_task = self.Dat.iloc[i]["ind_taskstroke_orig"]
                Task = self.dataset_extract("Task", i)
                return Task.Primitives[ind_strok_task]

            elif ver_behtask=="beh_aligned_single_strok":
                # Then pull out the beh taht matches this task stroek the best
                assert "aligned_beh_strokes_disttotask" in self.Dat.columns, "need to extract this first"
                strokes_beh = [S() for S in self.Dat.iloc[i]["aligned_beh_strokes"]]
                distances = self.Dat.iloc[i]["aligned_beh_strokes_disttotask"]
                return strokes_beh[np.argmin(distances)]
            else:
                print(self.Version)
                print(ver_behtask)
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
    def _dataset_index_here_given_trialcode(self, trialcode, return_rows=False):
        """ Find data with this trialcode
        PARAMS:
        - return_rows, bool, if True, then returns copy of the rows with this trialcode
        RETURNS:
        - either df with indices _NOT_ reset (see "return_rows") or list of ints (indices into self.Dat)
        """
        rows = self.Dat[self.Dat["dataset_trialcode"] == trialcode]
        if return_rows:
            return rows.copy()
        else:
            return rows.index.tolist()

    # def _dataset_index_here_given_dataset(self, ind_dataset, return_rows=False):
    #     """ Find the indices in self.Dat matching datsaet index
    #     ind_dataset
    #     RETUNS:
    #     - either df rows, or list of ints (can be empty if doesnt exist)
    #     """
    #     assert False, "never use ind_dataset instead use trialcode"
    #     tc = self.Dataset.Dat.iloc[ind_dataset]["trialcode"]
    #     rows = self.Dat[self.Dat["dataset_trialcode"] == tc]
    #     if return_rows:
    #         return rows
    #     else:
    #         return rows.index.tolist()

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

    def dataset_extract_strokeslength_list_ind_here(self, ind, column):
        """ Same as dataset_extract_strokeslength_list(), but input
        the index into self.Dat (ind), i.e., stroke level.
        """
        trialcode = self.Dat.iloc[ind]["dataset_trialcode"]
        # ind_dataset = self._dataset_index(ind)
        # assert False, "use trialcode"
        return self.dataset_extract_strokeslength_list(trialcode, column)


    def dataset_extract_strokeslength_list(self, trialcode, column):
        """ Extract a list of items matching the beh strokes in this dataset(trial level)
        in the order of the beh strokes and not missing any items.
        Does sanity checks to confirm correct extraction. 
        e..g, if column=="shape", then gets the list of shapes matching the beh strokes
        e.g., if column = "datseg" then gets the beh tokens for the trial holding this stroke.
        PARAMS;
        - ind_dataset, index into Dataset (trial level)
        - column, str, name of column to extract
        RETURNS:
        - list_vals, list of values, or None, if this dataset has no strokes in self
        """
        

        inds = self._dataset_index_here_given_trialcode(trialcode)
        ind_dataset = self.Dataset.index_by_trialcode(trialcode)

        # inds = self._dataset_index_here_given_dataset(ind_dataset)
        if len(inds)==0:
            return None
        else:
            # get all rows matching this datsate index, sorted by beh stroke index
            df = self.Dat.iloc[inds].sort_values("stroke_index")
            
            # sanity check that got all beh strokes, in order from 0.
            assert np.all(np.diff(df["stroke_index"])==1)
            assert df.iloc[0]["stroke_index"]==0
            assert len(df)==len(self.Dataset.Dat.iloc[ind_dataset]["strokes_beh"])
            
            # confirm trialcode
            tc = self.Dat.iloc[inds]["dataset_trialcode"].unique().tolist()
            assert len(tc)==1
            assert tc[0] == self.Dataset.Dat.iloc[ind_dataset]["trialcode"]

            return df[column].values.tolist()

    def dataset_replace_dataset(self, DatNew):
        """Replace entirely the beh dataset
        NOTE: confirmed that all indexing should still work, since it 
        uses trialcodes 
        """
        self.Dataset = DatNew
        
    def dataset_extract(self, colname, ind):
        """ Extract value for this colname in original datset,
        for ind indexing into the strokes (DatsetStrokes.Dat)
        PARAMS:
        - ind, index in to self (NOT into Dataset)
        """
        return self._dataset_index(ind, True)[colname].tolist()[0]
 
    def dataset_append_column(self, colname):
        """ Append a column to self.Dat from original datsaet.
        Overwrites if it already exists
        """
        valsthis = [self.dataset_extract(colname, i) for i in range(len(self.Dat))]
        self.Dat[colname] = valsthis
        print(f"Appended {colname} to self.Dat")


    def dataset_slice_by_trialcode_strokeindex(self, list_trialcode, list_stroke_index,
            df=None, assert_exactly_one_each=True):
        """ Returns self.Dat, sliced to subset of rows, matching exactly the inputed
        list_trialcode and list_strokeindex.
        PARAMS;
        - list_trialcode, list of str,
        - list_stroke_index, list of int, matching len of list_trialcode
        - df, optional, if None, uses self.Dat
        RETURNS:
        - dfslice, maatching exaxflty one to one the input, (if assert_exactly_one_each True)
        NOTE: also appends column called trialcode_strokeidx to df if doesnt already exist
        """
        from pythonlib.tools.pandastools import slice_by_row_label

        assert len(list_trialcode)==len(list_stroke_index)
        assert isinstance(list_trialcode[0], str)
        assert isinstance(list_stroke_index[0], int)

        if df is None:
            df = self.Dat

        # sp.DS.grouping_append_and_return_inner_items(["trialcode", "stroke_index"], new_col_name="trialcode_strokeidx")
        if "trialcode_strokeidx" not in df.columns:
            # - first, append a new colum that is the conjunction of trialcode and stroke index
            df = append_col_with_grp_index(df, ["trialcode", "stroke_index"], new_col_name="trialcode_strokeidx",
                                                 use_strings=False)

        # - slice DS.Dat using desired (trialcode, strokeindex)
        list_keys = [(tc, si) for tc, si in zip(list_trialcode, list_stroke_index)]

        # get slice
        dfslice = slice_by_row_label(df, "trialcode_strokeidx", list_keys, assert_exactly_one_each=assert_exactly_one_each)

        return dfslice



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

        len_in = len(self.Dat)

        # 1) If a key doesnt already exist in self.Dat, then extract it from Dataset
        list_keys = filtdict.keys()
        for k in list_keys:
            if k not in self.Dat.columns:
                self.dataset_append_column(k)

        # 2) Filter
        tmp = filterPandas(self.Dat, filtdict, return_indices=return_indices, reset_index=reset_index)
        assert len(self.Dat)==len_in, "why mutated?"
        return tmp

    def dataset_call_method(self, method, args, kwargs):
        """
        return D.<method>(*args, **kwargs)
        """
        return getattr(self.Dataset, method)(*args, **kwargs)

    ######################### SUMMARIZE
    def print_summary(self, Nmin=0):

        list_keys = ["task_kind", "gridsize", "shape_oriented", "gridloc"]
        for key in list_keys:
            print(f"\n--- Value counts for {key}")
            print(self.Dat[key].value_counts())

        print("\n ==== n samples per combo of params")
        self.print_n_samples_per_combo_grouping(Nmin=Nmin)


    def print_n_samples_per_combo_grouping(self, 
        list_grouping = None, 
        Nmin=0, savepath=None):
        """ 
        print n samples for combo of goruping vars
        """
        if list_grouping is None:
            list_grouping = ["shape_oriented", "gridloc", "gridsize"]

        from pythonlib.tools.pandastools import grouping_print_n_samples
        outdict = grouping_print_n_samples(self.Dat, list_grouping, Nmin, savepath, save_as="txt")

    def print_n_samples_per_combo(self, list_grouping):
        """ Wrapper, legacy name"""
        return self.print_n_samples_per_combo_grouping(list_grouping)

    ####################### PLOTS
    def plot_multiple_strok(self, list_strok, ver="beh", ax=None,
        overlay=True, titles=None, ncols=5, size_per_sublot=2):
        """
        PARAMS;
        - ax, either None, single ax (if overlay==True) or list of axes
        (if overlay==False).
        """

        if overlay:
            # Then all use the same axis.
            if ax is None:
                fig, ax = plt.subplots(1,1)
            axes = [ax for _ in range(len(list_strok))]
        else:
            # Then all a separate axis.
            if ax is None:
                nrows = int(np.ceil(len(list_strok)/ncols))
                assert nrows<=10
                fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, 
                    figsize=(ncols*size_per_sublot, nrows*size_per_sublot))
                axes = axes.flatten()
            else:
                assert len(ax)==len(list_strok)

        for strok, ax in zip(list_strok, axes):
            self.plot_single_strok(strok, ver=ver, ax=ax)

        if titles is not None:
            for ax, tit in zip(axes, titles):
                ax.set_title(tit)

        return fig, axes


    def plot_single_strok(self, strok, ver="beh", ax=None, 
            color=None):
        """ plot a single inputed strok on axis.
        INPUT:
        - strok, np array,
        """
        assert ax is not None
        from pythonlib.drawmodel.strokePlots import plotDatStrokesWrapper, plotDatStrokes
        if ver=="beh":
            plotDatStrokesWrapper([strok], ax, color=color, add_stroke_number=False, 
                mark_stroke_onset=True)
        elif ver=="task_dots":
            plotDatStrokes([strok], ax, clean_unordered=True, alpha=0.2)
        elif ver=="task":
            plotDatStrokes([strok], ax, clean_task=True, alpha=0.2)
        else:
            assert False


    def plot_single_overlay_entire_trial(self, ind, ax=None, 
            overlay_beh_or_task="beh", SIZE=2):
        """
        Plot a single stroke, overlaying it on all the strokes from this trial, colored
        and numbered.
        PARAMS
        - overlay_beh_or_task, str, whether to overlay the entire trial;s' beh or task.
        """

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(SIZE, SIZE))

        # plot trial
        inddat = self._dataset_index(ind)
        if overlay_beh_or_task=="beh":
            self.Dataset.plot_single_trial(inddat, ax, single_color="k")
        elif overlay_beh_or_task=="task":
            self.Dataset.plot_single_trial(inddat, ax, single_color="k", ver="task")
        else:
            assert False

        # overlay the stroke
        self.plot_single(ind, ax, color="r")

        return inddat, fig, ax


    def plot_multiple_overlay_entire_trial(self, list_inds, ncols=5, 
            overlay_beh_or_task="beh", titles=None):
        """ Plot Multipel strokes on multiple supblots, each one plot the stroke overlaying it on the etnire
        trial for that stroke. 
        """ 
        n = len(list_inds)
        nrows = int(np.ceil(n/ncols))
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(2*ncols, 2*nrows))


        inds_trials_dataset = []
        for ind, ax in zip(list_inds, axes.flatten()):
            inddat, _, _ = self.plot_single_overlay_entire_trial(ind, ax, overlay_beh_or_task=overlay_beh_or_task)
            inds_trials_dataset.append(inddat)

        if titles is not None:
            assert len(titles)==n
            for ax, tit in zip(axes.flatten(), titles):
                ax.set_title(tit)

        return fig, axes, inds_trials_dataset


    def plot_single(self, ind, ax=None, color=None):
        """ Plot a single stroke, based on index into self.Dat
        """
        if ax==None:
            fig, ax = plt.subplots(1,1)
        S = self.Dat.iloc[ind]["Stroke"]
        self.plot_single_strok(S(), ver="beh", ax=ax, color=color)
        return ax

    def plot_multiple_after_slicing_within_range_values(self, colname, minval, 
        maxval, plot_hist=True):
        """ Plot example trials that are wihitn this range of values for
        a given column, e.g,., frac_touched
        """

        if plot_hist:
            self.Dat[colname].hist()

        # d1 = 0.6
        # d2 = 0.7
        inds = self.Dat[(self.Dat[colname]>minval) & (self.Dat[colname]<maxval)].index.tolist()
        print("This many trials found:", len(inds))
        self.plot_multiple_overlay_entire_trial(inds, overlay_beh_or_task="task")
        self.plot_multiple_overlay_entire_trial(inds, overlay_beh_or_task="beh")

    def plot_multiple(self, list_inds, ver_behtask=None, titles=None, ncols=5,
            titles_by_dfcolumn=None, nrand=20):
        """ Plot mulitple strokes, each on a subplot, based on index into self.Dat
        PARAMS:
        - ver_behtask, "task", "beh", or None(default).
        - title_by_dfcolumn, if not None, then uses values from this column in self.Dat.
        titles must be None
        """

        if list_inds is None:
            # get random n
            import random
            list_inds = random.sample(range(len(self.Dat)), nrand)

        if titles_by_dfcolumn is not None:
            assert titles is None
            titles = self.Dat.iloc[list_inds][titles_by_dfcolumn].values.tolist()

        strokes = self.extract_strokes("list_list_arrays", 
            inds = list_inds, ver_behtask=ver_behtask)
        fig, axes = self.Dataset.plotMultStrokes(strokes, titles=titles, ncols=ncols)

        return fig, axes, list_inds

    def plot_strokes_overlaid(self, inds, ax=None, nmax = 50, color_by="order", ver_behtask=None):
        """ Plot strokes all overlaid on same plot
        Colors strokes by order
        PARAMS;
        - inds, indices into self.Dat (must be less than nmax)
        - nmax, if over, then takes random subset
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes
        if ax is None:
            fig, ax = plt.subplots(1,1)

        if len(inds)>nmax:
            import random
            inds = sorted(random.sample(inds, nmax))
        
        strokes = self.extract_strokes("list_arrays", inds, ver_behtask=ver_behtask)
        # strokes = [S() for S in self.Dat.iloc[inds]["Stroke"]]
        if color_by=="order":
            plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, alpha=0.5)
        else:
            plotDatStrokes(strokes, ax, clean_ordered=True, alpha=0.5)

        return ax

    def plotshape_egtrials_sorted_by_behtaskdist(self, shape):
        """ 
        Plot eample trials, sorted by the distance between the beh and task stroke it 
        is assigned to

        """

        inds, dists = self.shape_extract_indices_sorted_by_behtaskdist(shape)

        inds = inds[::5]
        dists = dists[::5]
    #     DS.plot_multiple(inds, titles=dists, nrand=100);




    #     for shape in list_shape:
    #         inds = 
        fig, axes, _ = DS.plot_multiple_overlay_entire_trial(inds, titles=dists, overlay_beh_or_task="task");
            
        fig.savefig(f"{sdir}/{shape}.pdf")
        plt.close("all")
        

    def plotshape_singleshape_egstrokes_overlaid(self, shape=None, filtdict=None, nplot=40, 
        ver_behtask="beh"):
    # def plot_egstrokes_overlaid(self, shape=None, filtdict=None, nplot=40, 
    #     ver_behtask="beh"):
        """Plot example strokes overlaid on a single plot.
        PARAMS:
        - shape, if not None, then str, filters to onlyu plot htis hape.
        - filtdict, dict, more flexible filter, only works if shape is None.
        RETURNS:
        - fig, ax
        """

        if shape is not None:
            assert filtdict is None
            F = {"shape_oriented":[shape]}
            inds = self.dataset_slice_by_mult(F, return_indices=True)
        elif filtdict is not None:
            inds = self.dataset_slice_by_mult(filtdict, return_indices=True)
        else:
            inds = range(len(self.Dat))

        # Extract and plot
        fig, ax = plt.subplots(1,1)
        if len(inds)>nplot:
            import random
            inds = random.sample(inds, nplot)
        self.plot_strokes_overlaid(inds, ver_behtask=ver_behtask, ax=ax)
        return fig, ax

    def plotwrap_timecourse_vels_grouped_by_shape(self, n_each, version="vel",
        also_plot_example_strokes=False, savedir=None):
        """ Plot velocity timecourse for n_each exmaples for each shape, each a 
        separate plot. 
        """
        list_shape = sorted(self.Dat["shape"].unique().tolist())
        groupdict = self.find_indices_these_shapes(list_shape, nrand_each=n_each)
        for shape, inds in groupdict.items():
            print("here:", shape)
            list_strokes = self.extract_strokes(version="list_list_arrays", inds=inds)
            fig = self.Dataset.plotMultStrokesTimecourse(list_strokes, plotver=version, 
                align_to="first_touch")

            if savedir is not None:
                fig.savefig(f"{savedir}/{shape}.pdf")
                plt.close("all")

        if also_plot_example_strokes:
            figholder = self.plotshape_multshapes_egstrokes(n_examples_total_per_shape=2, color_by=None,
                list_shape=list_shape)

            for i, X in enumerate(figholder):
                fig = X[0]
                fig.savefig(f"{savedir}/egstrokes-{i}.pdf")
                plt.close("all")
            

    def plotshape_multshapes_egstrokes(self, key_subplots = "shape_oriented",
            n_examples_total_per_shape = 4, color_by=None, list_shape=None):
    # def plot_egstrokes_grouped_by_shape(self, key_subplots = "shape_oriented",
    #         n_examples_total_per_shape = 4, color_by=None, list_shape=None):
        """ Wrapper to make one subplot per shape, either plotting mean stroke (not 
        yet coded) or 
        individual trials
        PARAMS;
        """

        key_to_extract_stroke_variations_in_single_subplot = None
        ver_behtask = "beh"

        return self.plotshape_multshapes_egstrokes_grouped_in_subplots(None, key_subplots,
                                             key_to_extract_stroke_variations_in_single_subplot,
                                             ver_behtask=ver_behtask, ncols=6, SIZE=3, 
                                              n_examples=n_examples_total_per_shape, color_by=color_by,
                                             levels_subplots=list_shape)

    def _plotshape_row_col_vs_othervar(self, df, rowvar, colvar="shape", n_examples_per_sublot=1,
        plot_task=False):
        """ Plot shapes on columns and othervar as rows, each slot an example stroke 
        conjucntion those levels
        PARAMS:
        - othervar, string, column in self.Dat (caregorical)
        """
        from pythonlib.dataset.plots import plotwrapper_draw_grid_rows_cols
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        if "strokes_beh" not in df.columns:
            def F(x):
                return [x["strok"]]
            df = applyFunctionToAllRows(df, F, "strokes_beh").copy()        
        else:
            df = df.copy()
        figbeh, figtask = plotwrapper_draw_grid_rows_cols(df, rowvar, colvar, 
            n_examples_per_sublot=n_examples_per_sublot, sort_colvar=True, plot_task=plot_task)
        return figbeh, figtask

    def plotshape_row_col_vs_othervar(self, rowvar, colvar="shape", n_examples_per_sublot=1,
        plot_task=False):
        """ Plot shapes on columns and othervar as rows, each slot an example stroke 
        conjucntion those levels
        PARAMS:
        - othervar, string, column in self.Dat (caregorical)
        """
        df = self.Dat
        figbeh, figtask = self._plotshape_row_col_vs_othervar(df, rowvar, colvar, n_examples_per_sublot,
            plot_task)
        return figbeh, figtask

    def plotshape_row_col_size_loc(self):
        """ Plot grid, where cols are shapes, rows and sizes, and within each subplot plots 
        one stroke for each location
        """
        from pythonlib.tools.pandastools import grouping_get_inner_items

        # NEed to extract strokes beh and task
        list_strokes_task = self.extract_strokes("list_list_arrays", ver_behtask="task_aligned_single_strok")
        self.Dat["strokes_task"] = list_strokes_task

        # keep one trial for each location/size/size combo
        self.Dat = append_col_with_grp_index(self.Dat, ["shape", "gridsize", "gridloc"], "shape_size_loc")
        groupdict = grouping_get_inner_items(self.Dat, "shape_size_loc", nrand_each=1)

        inds_keep = sorted([x[0] for x in groupdict.values()])
        DF = self.Dat.iloc[inds_keep].copy().reset_index(drop=True)

        n = len(self.Dat["gridloc"].unique()) # exhaust the number of locations
        fig_beh, fig_task = self._plotshape_row_col_vs_othervar(DF, "gridsize", n_examples_per_sublot=n, plot_task=True)        

        return fig_beh, fig_task

    def plotshape_multshapes_egstrokes_grouped_in_subplots(self, task_kind=None, 
        key_subplots="shape_oriented",
        key_to_extract_stroke_variations_in_single_subplot = "gridloc", 
        n_examples = 2, color_by=None, ver_behtask=None,
        filtdict = None, ncols=5, SIZE=4,
        levels_subplots=None):
    # def plot_egstrokes_grouped_in_subplots(self, task_kind=None, 
    #     key_subplots="shape_oriented",
    #     key_to_extract_stroke_variations_in_single_subplot = "gridloc", 
    #     n_examples = 2, color_by=None, ver_behtask=None,
    #     filtdict = None, ncols=5, SIZE=4,
    #     levels_subplots=None):
        """
        Plot all sahpes, each shape a subplot. Can choose to sample across specific levels
        fora  var (e.g, gridloc), n_examples, for each subplot.
        PARAMS:
        - task_kind, string, indexes into the task_kind column. Leave None to keep any
        --- e..g, {"character", "prims_on_grid"}
        - key_subplots, string, each level of this grouping variable will have its
        own subplot.
        - key_to_extract_stroke_variations_in_single_subplot, string, for each subplot, 
        how to extract exabple strokes. e..g, if "gridloc", then strokes will be at
        variable locations in the subplot. This is ignored if task_kind=="character", because
        that doesnt have structured variation in these params.
        - color_by, {None, 'order'}
        """
        from pythonlib.tools.plottools import subplot_helper
        from pythonlib.tools.pandastools import extract_trials_spanning_variable
        import random

                
        if task_kind=="character":
            key_to_extract_stroke_variations_in_single_subplot = None

        # 0) Static params:
        # filtdict = {}
        if filtdict is None:
            filtdict = {}
        if task_kind is not None:
            # assert "task_kind" not in F.keys()
            filtdict["task_kind"] = [task_kind]

        # One plot for each level
        if levels_subplots is None:
            levels_subplots = sorted(self.Dat[key_subplots].unique().tolist())

        getax, figholder, nplots = subplot_helper(ncols, 10, len(levels_subplots), SIZE=SIZE)
        for i, level in enumerate(levels_subplots):
            
            # - update the shape
            filtdict[key_subplots] = level

            # 2) For each combo of features, select a random case of it.
            if key_to_extract_stroke_variations_in_single_subplot is None:
                # Use entire pool, not specific variation
                list_idx = self.dataset_slice_by_mult(filtdict, return_indices=True)
                if len(list_idx)>=n_examples:
                    inds = random.sample(list_idx, n_examples)[:n_examples]
                else:
                    inds = list_idx
                    # inds = [None for _ in range(n_examples)] # dont plot
                list_inds = inds
            else:
                # 1) get all unique values for a given key
                def bad(x):
                    # Return True is this value is a nan
                    if isinstance(x, (tuple, list, str)):
                        return False
                    else:
                        return np.isnan(x)
                vals_vary = self.Dat[key_to_extract_stroke_variations_in_single_subplot].unique().tolist()
                vals_vary = [v for v in vals_vary if not bad(v)]
                list_inds, _ = extract_trials_spanning_variable(self.Dat, 
                    key_to_extract_stroke_variations_in_single_subplot, vals_vary, 
                    n_examples, filtdict)

            # 3) pull out these examples and plot
            list_inds = [i for i in list_inds if i is not None]
            if len(list_inds)>0:
                ax = getax(i)
                self.plot_strokes_overlaid(list_inds, ax=ax, color_by=color_by, ver_behtask=ver_behtask)
                ax.set_title(level)
        return figholder

    def plot_beh_and_aligned_task_strokes(self, list_inds, title_with_dists=False):
        """ One subplot for each beh strokes, each a single subplot, 
        and for each overlay its best-aligned task signle stroke
        """
        
        assert len(list_inds)<50, "too many plots..."
        
        # collect beh and task strokes
        list_strok_beh = self.extract_strokes(inds=list_inds, ver_behtask="beh")
        list_strok_task = self.extract_strokes(inds=list_inds, ver_behtask="task_aligned_single_strok")
            
        # Title with distances?
        if title_with_dists:
            dists = self._dist_alignedtask_to_beh(list_inds)   
            titles = [f"{d:.2f}" for d in dists]
        else:
            titles = list_inds 

        from pythonlib.tools.plottools import subplot_helper
        ncols = 5
        nrows = 10
        n = len(list_inds)
        getax, figholder, nplots = subplot_helper(ncols, nrows, n)

        for i, (sbeh, stask, tit) in enumerate(zip(list_strok_beh, list_strok_task, titles)):
            ax = getax(i)
            self.plot_single_strok(stask, "task", ax=ax)
            self.plot_single_strok(sbeh, "beh", ax=ax)
            ax.set_title(tit)
        

    def plotshape_multshapes_trials_grid(self, col_grp="shape_oriented", col_levels=None, 
            nrows=2, flip_plot=False):
    # def plot_examples_grid(self, col_grp="shape_oriented", col_levels=None, nrows=2,
    #         flip_plot=False):
        """ 
        Plot grid of strokes (sublots), where cols are (e.g.) shapes and rows are
        example trials
        PARAMS:
        - flip_plot, bool, if True, then cols are actually plotted as rows
        - nrows, num examples for each shape, each will be a row.
        """
        from pythonlib.tools.pandastools import extract_trials_spanning_variable        
        from pythonlib.tools.plottools import plotGridWrapper
        from pythonlib.drawmodel.strokePlots import plotDatStrokes

        outdict = extract_trials_spanning_variable(self.Dat, col_grp,
            col_levels, n_examples=nrows, return_as_dict=True)[0]
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

        # print(len(strokes))
        # assert False
        if flip_plot:
            fig = plotGridWrapper(strokes, plotfunc, list_row, list_col[::-1])
        else:
            fig = plotGridWrapper(strokes, plotfunc, list_col, list_row)
        return fig




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
    def grouping_get_inner_items(self, groupouter="shape_oriented", groupinner="index",
        nrand_each=None):
        from pythonlib.tools.pandastools import grouping_get_inner_items
        groupdict = grouping_get_inner_items(self.Dat, groupouter, groupinner, nrand_each=nrand_each)
        return groupdict


    def grouping_append_and_return_inner_items(self, list_grouping_vars=None,
        new_col_name="grp"):
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

        if list_grouping_vars is None:
            list_grouping_vars = ["shape_oriented", "gridloc"]

        groupdict = grouping_append_and_return_inner_items(self.Dat, list_grouping_vars,
            "index", new_col_name=new_col_name)

        return groupdict

    # def generate_data_groupdict(self, list_grouping_vars, GET_ONE_LOC, gridloc, PRUNE_SHAPES, bad_shapes = ["Lcentered-3-0"],
    #         verbose=False):
        
    #     # 1) generate new groups and return groupdict.
    #     groupdict = self.grouping_append_and_return_inner_items(list_grouping_vars)
    #     list_cats = groupdict.keys()
        
    #     # Prune list to only one specific grid location
    #     if GET_ONE_LOC:
    #         # gridloc = (1,1)
    #         list_cats = [cat for cat in list_cats if cat[1]==gridloc]

    #     if PRUNE_SHAPES:
    #         # PRUNE certain shapes (ad hoc)

    #         # Remove, since does in diff orders on diff trials: Lcentered-3-0
    #         list_cats = [cat for cat in list_cats if cat[0] not in bad_shapes]

    #     groupdict = {k:v for k,v in groupdict.items() if k in list_cats}
    #     if verbose:
    #         print("FINAL GROUPDICT")
    #         for k in groupdict.keys():
    #             print(k)

    #     return groupdict


    ############################### SEQUENTIAL CONTEXT
    def context_extract_strokeslength_list(self, ind, column="datseg"):
        """ A wrapper for extracting the entire trial's data that this stroke (ind) is in
        PARAMS:
        - ind, index into self.Dat (stroke-level)
        RETURNS:
        - list of data objects, aligned to the strokes for the trial that has this ind
        - index_stroke, int, index into the list, whcih is the data for the inputed ind
        """
        list_dat = self.dataset_extract_strokeslength_list_ind_here(ind, column)
        strokind = self.Dat.iloc[ind]["stroke_index"]
        # print(self.Dat.iloc[strokind][column])
        # print(list_dat[strokind])
        # assert len(list_dat)>strokind, "sanity"
        assert list_dat[strokind] == self.Dat.iloc[ind][column], "sanity"
        return list_dat, strokind

    def context_extract_tokens_pre_post(self, ind):
        """
        RETURNS:
        - previous token
        - this token
        - next token
        (for pre and next, if doesnt exist, reuturns None)
        """

        tokens, indstrok = self.context_extract_strokeslength_list(ind, "datseg")

        if indstrok==0:
            tok_prev = None
        else:
            tok_prev = tokens[indstrok-1]

        tok_this = tokens[indstrok]

        if indstrok+1==len(tokens):
            tok_next = None
        else:
            tok_next = tokens[indstrok+1]

        return tok_prev, tok_this, tok_next

    def context_chunks_diff(self, ind, first_stroke_diff_to_zero=False):
        """ Return the diffefence of this strokes chunk from that of the
        previous stroke.
        PARAMS:
        - first_stroke_diff_to_zero, bool, if true, then the first stroke defined to 
        have diff of 0.
        RETURNS:
        - either None (if predecing stroke is niot defined/exiost)
        - or (chunk_diff, rank_within_diff)
        """
        tok_prev, tok_this, tok_next = self.context_extract_tokens_pre_post(ind)

        if first_stroke_diff_to_zero and self.Dat.iloc[ind]["stroke_index"]==0:
            assert tok_prev==None
            chunk_diff=0
            rank_within_diff = 0
        elif tok_prev is None:
            chunk_diff=None
            rank_within_diff = None
        else:
            chunk_diff = tok_this["chunk_rank"] - tok_prev["chunk_rank"]
            rank_within_diff = tok_this["chunk_within_rank"] - tok_prev["chunk_within_rank"]
        return chunk_diff, rank_within_diff


    ################################ SIMILARITY/CLUSTERING
    def cluster_compute_mean_stroke(self, inds, center_at_onset=True,
        check_same_direction=True):
        """Compute the mean stroke across these inds, after linear time warping to align them
        PARAMS:
        - inds, indices into self.Dat
        - check_same_direction, bool, only continues if strokes are same direction
        movements.
        RETURNS:
        - strokmean, (npts, 3), the mean stroke, where npts is by default 50
        - strokstacked, (n_inds, npts, 3), all trials.
        """

        from pythonlib.tools.stroketools import strokes_average
        
        strokes = self.extract_strokes("list_arrays", inds)
        if check_same_direction:
            assert self._strokes_check_all_aligned_direction(strokes, plot_failure=True), "same shape done in multipel directions?"

        strokmean, strokstacked = strokes_average(strokes, 
            center_at_onset=center_at_onset)
        return strokmean, strokstacked    

    def cluster_compute_mean_stroke_thisshape(self, shape, best_n_by_dist=None):
        """ Compute mean stroke by averaging all trials for this shape
        """
        
        inds, dists = self.shape_extract_indices_sorted_by_behtaskdist(shape)

        if best_n_by_dist is not None:
            # Take the top n datapts, by beh-task dist.
            inds = inds[:best_n_by_dist]

        # inds = self.Dat[self.Dat["shape_oriented"]==shape].index.tolist()
        assert len(inds)>0, f"why empty?, {shape}"
        return self.cluster_compute_mean_stroke(inds, center_at_onset=True)


    def _cluster_compute_sim_matrix(self, strokes_data, strokes_basis, 
        rescale_strokes_ver=None, distancever="euclidian_diffs", 
        return_as_Clusters=False, labels_for_Clusters=None, labels_for_basis=None):
        """ Low-level code to compute the similarity matrix between list of strokes
        and basis set. Autoatmically recenters strokes to onset.
        PARAMS:
        - strokes_data, list of strok
        - strokes_basis, list of strok, (columns)
        """
        from ..drawmodel.sf import computeSimMatrixGivenBasis
        from ..cluster.clustclass import Clusters

        if labels_for_basis is not None:
            assert len(labels_for_basis)==len(strokes_basis)

        simmat = computeSimMatrixGivenBasis(strokes_data, strokes_basis, 
            rescale_strokes_ver=rescale_strokes_ver, distancever=distancever) 

        if return_as_Clusters:
            if labels_for_Clusters is None:
                labels_for_Clusters = [i for i in range(len(strokes_data))]

            Cl = Clusters(X = simmat, labels_rows=labels_for_Clusters, 
                labels_cols=labels_for_basis)

            # Plot
            if False:
                Cl.plot_heatmap_data()
                Cl.plot_heatmap_data(sortver=0)
                Cl.plot_heatmap_data(sortver=2)
            return Cl
        else:
            return simmat

    def _cluster_compute_sim_matrix_multiplever(self, strokes_data, strokes_basis, 
        list_ver, labels_for_Clusters=None, labels_for_basis=None,
        return_as_Clusters=True):
        """Low-level code to iterate over multiple distancever's, each time computing 
        a similatiy matrix
        PARAMS:
        - list_ver, list of string, distance metrics.
        RETUNRS:
        - list_simmat, list of np array (ndat, nbas) similarity matrices
        """
        list_simmat = []
        for ver in list_ver:
            simmat = self._cluster_compute_sim_matrix(strokes_data, strokes_basis, 
                distancever=ver, return_as_Clusters=return_as_Clusters, 
                labels_for_Clusters=labels_for_Clusters, labels_for_basis=labels_for_basis) 
            list_simmat.append(simmat)

        return list_simmat

    def _cluster_compute_sim_matrix_aggver(self, strokes_data, strokes_basis, 
            list_ver=["euclidian_diffs", "euclidian", "hausdorff_alignedonset", "hausdorff_centered"], 
            labels_for_Clusters=None, labels_for_basis=None):
        from ..cluster.clustclass import Clusters
        """ Low-level code to compute multiple similarity matrices (diff distance vers)
        and average them, to return a single sim mat
        """

        # collect
        list_simmat = self._cluster_compute_sim_matrix_multiplever(strokes_data, 
            strokes_basis, list_ver, return_as_Clusters=False)

        # Average
        x = np.stack(list_simmat)
        simmat = np.mean(x, axis=0)

        if labels_for_Clusters is None:
            labels_for_Clusters = [i for i in range(len(strokes_data))]

        Cl = Clusters(X = simmat, labels_rows=labels_for_Clusters, 
            labels_cols=labels_for_basis)

        return Cl


    def clustergood_featurespace_project(self, df, which_space,
            list_strok_basis=None, which_basis_set = None,
            which_shapes = None,
            list_distance_ver=None):
        """
        Given dataset, projects data to a variety of possible feature spaces
        
        """

        # def _load_basis_strokes():
        #     """ Helper to load pre-saved basis set, when needed"""
        #     if which_basis_set is None:
        #         if self.Dataset.animals()==["Pancho"]:
        #             which_basis_set = "standard_17"
        #         elif self.Dataset.animals()==["Diego"]:
        #             which_basis_set = "diego_all_minus_open_to_left"
        #         else:
        #             print(D.animals())
        #             assert False

        #     dfbasis, _, _ = self.stroke_shape_cluster_database_load_helper(
        #         which_basis_set=which_basis_set, which_shapes=which_shapes)

        #     return dfbasis


        params = {}
        if which_space=="strok_sim_motor":
            """ Similarity between strokes, at motor level"""

            # Extract dataset
            list_strok = df["strok"].tolist()
            list_shape = df["shape"].tolist()

            # Extract the basis set
            if list_strok_basis is None:
                dfbasis, _, _ = self.stroke_shape_cluster_database_load_helper(
                    which_basis_set=which_basis_set,
                    which_shapes=which_shapes)
                list_strok_basis = dfbasis["strok"].tolist()
                list_shape_basis = dfbasis["shape"].tolist()

            # Which distance score
            if list_distance_ver is None:
                list_distance_ver  =("euclidian_diffs", "euclidian", "hausdorff_alignedonset")

            # Compute similarity
            Cl = self._cluster_compute_sim_matrix_aggver(list_strok, list_strok_basis, list_distance_ver,
                                                                labels_for_Clusters = list_shape, 
                                                                labels_for_basis = list_shape_basis)

            params["list_strok_basis"] = list_strok_basis
            params["list_shape_basis"] = list_shape_basis
            params["list_distance_ver"] = list_distance_ver

        elif which_space=="strok_sim_task":
            """ Each datapt's task image"""
            
            # 2) Visual similarity to basis set
            # - for each shape, return a single example task image
            list_strok = df["strok_task"].values.tolist()
            list_shape = df["shape"].values.tolist()

            if list_strok_basis is None:
                dfbasis, _, _ = self.stroke_shape_cluster_database_load_helper(
                    which_basis_set=which_basis_set,
                    which_shapes=which_shapes)
                list_strok_basis = dfbasis["strok_task"].tolist()
                list_shape_basis = dfbasis["shape"].tolist()

            # Cluster
            Cl = self._cluster_compute_sim_matrix(list_strok, list_strok_basis,
                                                rescale_strokes_ver=None, distancever="hausdorff_max", return_as_Clusters=True,
                                               labels_for_Clusters=list_shape)

            params["list_strok_basis"] = list_strok_basis
            params["list_shape_basis"] = list_shape

        elif which_space=="shape_cat_abstract":
            """One-hot enocding of the shape category"""
            from sklearn.preprocessing import OneHotEncoder
            from pythonlib.cluster.clustclass import Clusters

            list_shape = df["shape"].tolist()

            # convert to one-hot
            encoder = OneHotEncoder(handle_unknown='ignore')
            X = encoder.fit_transform(df[["shape_cat_abstract"]]).toarray()

            labels_rows = list_shape
            labels_cols = encoder.categories_[0].tolist()
            Cl = Clusters(X, labels_rows=labels_rows, labels_cols=labels_cols)
            # ClustDict["task_shape_cat_abstract"] = 

        else:
            print(which_space)
            assert False


        return Cl, params

    def clustergood_assign_data_to_cluster(self, ClustDict, ParamsDict, 
            ParamsGeneral, dfdat,
            which_features = "beh_motor_sim",
            trial_summary_score_ver="clust_sim_max"):

        # Given a Cl, use it to assign each beh to a cluster.
        

        ##########################################
        ### [FOR MOTOR] Extract scalar values summarizing the simialrity scores (e.g,, clustering)
        # For each beh stroke, get (i) match and (ii) uniqueness.
        Cl = ClustDict[which_features]
        Cl.cluster_compute_feature_scores_assignment()
        dat = Cl.cluster_extract_data("max_sim")

        # sims_max = Cl.Xinput.max(axis=1)
        # sims_min = Cl.Xinput.min(axis=1)
        # sims_median = np.median(Cl.Xinput, axis=1)
        # # sims_mean = Cl.Xinput.mean(axis=1)
        # sims_concentration = (sims_max - sims_min)/(sims_max + sims_min)
        # sims_concentration_v2 = (sims_max - sims_median)/(sims_max + sims_median)
        # # which shape does it match the best
        # inds_maxsim = np.argmax(Cl.Xinput, axis=1)
        # cols_maxsim = [Cl.LabelsCols[i] for i in inds_maxsim]

        ### Slide back into DS
        # dfdat["clust_sim_max"] = dat["sims_max"]
        # dfdat["clust_sim_concentration"] = dat["sims_concentration"]
        # dfdat["clust_sim_concentration_v2"] = dat["sims_concentration_v2"]
        # dfdat["clust_sim_max_ind"] = dat["colinds_maxsim"]
        # dfdat["clust_sim_max_colname"] = dat["collabels_maxsim"]
        # dfdat["clust_sim_vec"] = [vec for vec in Cl.Xinput]

        if ParamsGeneral["version_trial_or_shapemean"]=="trial":
            #, "you want to assign back to self.Dat..."

            print("Adding columns to self.Dat")
            ### Slide back into DS
            self.Dat["clust_sim_max"] = dat["sims_max"]
            self.Dat["clust_sim_concentration"] = dat["sims_concentration"]
            self.Dat["clust_sim_max_ind"] = dat["colinds_maxsim"]
            self.Dat["clust_sim_max_colname"] = dat["collabels_maxsim"]
            self.Dat["clust_sim_vec"] = [vec for vec in Cl.Xinput]

            ### Slide back in to D: Collect scores (avg over strokes for a trial) and put back into D
            print("Adding columns to self.Dataset.Dat")
            list_scores = []
            for i in range(len(self.Dataset.Dat)):
                # get all rows in DS
                tc = self.Dataset.Dat.iloc[i]["trialcode"]
                inds = self._dataset_index_here_given_trialcode(tc)
                # inds = DS._dataset_index_here_given_dataset(i)
                if len(inds)>0:
                    score = np.mean(self.Dat.iloc[inds][trial_summary_score_ver])
                else:
                    # assert False, "not sure why this D trial has no strokes..."
                    score = np.nan
                list_scores.append(score)
            self.Dataset.Dat["strokes_clust_score"] = list_scores

    def clustergood_plot_raw_results(self, ClustDict, ParamsDict, ParamsGeneral, dfdat, 
            SDIR, N_EXAMPLES_BEH=15):
        """
        Plot results including
        -- heatmap of features (data vs. feature vectors)
        -- example strokes based on assignment by maximum sim.
        PARAMS:
        - ClustDict, ParamsDict, dfdat, outputs of features_wrapper_generate_all_features
        """
        from pythonlib.tools.snstools import heatmap
        from pythonlib.tools.listtools import random_inds_uniformly_distributed
        sdir = f"{SDIR}/hier_clustering"
        n_max_drawings = 60

        # Plot result of clustering
        n_examples = 5
        for PLOT in ClustDict.keys():
        #     PLOT = "task_shape_cat_abstract"
            import os
            os.makedirs(sdir, exist_ok=True)
            
            # 1) Plot heatmap
            Cl = ClustDict[PLOT]
            cg = Cl.plot_save_hier_clust()
            Cl._hier_clust_label_axis(cg, "col")
            Cl._hier_clust_label_axis(cg, "row")
            savefig(cg.fig, f"{sdir}/{PLOT}-heatmap.pdf")
            
            # 2) Plot beh and task, ordered by rows in heatmap
            # - label with the cluster id

            # - reorder by clustering
            inds = cg.dendrogram_row.reordered_ind
            if len(inds)>n_max_drawings:
                inds = random_inds_uniformly_distributed(inds, n_max_drawings)

            # Plot (beh)
            list_strok = dfdat.iloc[inds]["strok"].values.tolist()
            list_strok_task = dfdat.iloc[inds]["strok_task"].values.tolist()
            list_shape = dfdat.iloc[inds]["shape"].values.tolist()

            fig, axes = self.plot_multiple_strok(list_strok, overlay=False, titles=list_shape, ncols=6)
            savefig(fig, f"{sdir}/{PLOT}-beh.pdf")
            fig, axes = self.plot_multiple_strok(list_strok_task, overlay=False, titles=list_shape, ncols=6)
            savefig(fig, f"{sdir}/{PLOT}-task.pdf")

            plt.close("all")

            # Plot beh, example trials
            if ParamsGeneral["version_trial_or_shapemean"]=="shapemean":
                key_subplots = "shape_char"
                figholder = self.plotshape_multshapes_egstrokes(key_subplots = key_subplots,
                            n_examples_total_per_shape =N_EXAMPLES_BEH, color_by=None, list_shape=list_shape);
                for i, (fig, axes) in enumerate(figholder):
                    savefig(fig, f"{sdir}/{PLOT}-beh_egtrials-sub{i}.pdf")

            plt.close("all")

        # Plot each cluster, sorting rows in identical fashion (i.e, using row order from other clustering)
        for REORDER_BY in ClustDict.keys():
            for PLOT in ClustDict.keys():

                # 1) get the orders
                ClReorder = ClustDict[REORDER_BY]
                cg = ClReorder.ClusterResults["hier_clust_seaborn"]
                inds_reorder = cg.dendrogram_row.reordered_ind

                # 2) Get the plot, and plot using this order
                ClDat = ClustDict[PLOT]

                # always reorder the columns using this representations col clustering.
                cgthis = ClDat.ClusterResults["hier_clust_seaborn"]
                inds_reorder_cols = cgthis.dendrogram_col.reordered_ind

                X = ClDat.Xinput[inds_reorder, :]
                X = X[:, inds_reorder_cols]
                fig, ax = plt.subplots(1,1, figsize=(8,8))
                sns.heatmap(X, ax=ax)
                
                ClReorder._hier_clust_label_axis(cgthis, "row")
                # _label_axis("row", REORDER_BY, ax)
                ClDat._hier_clust_label_axis(cgthis, "col")
                # _label_axis("col", PLOT, ax)
                
                savefig(fig, f"{sdir}/{PLOT}-REORDERBY-{REORDER_BY}-heatmap.pdf")
                
                plt.close("all")


    def clustergood_extract_saved_clusterobject(self, WHICH_FEATURE="beh_motor_sim"):
        # WHICH_FEATURE = "beh_motor_sim" 
        Cl = self.Clusters_ClustDict[WHICH_FEATURE]
        # list_shape_basis = ParamsDict[WHICH_FEATURE]["list_shape_basis"]
        # list_strok_basis = ParamsDict[WHICH_FEATURE]["list_strok_basis"]        
        Params = self.Clusters_ParamsDict[WHICH_FEATURE]
        return Cl, Params


    def clustergood_plot_single_dat(self, ind, WHICH_FEATURE="beh_motor_sim",
        savedir=None, prefix=None):
        ##### [Good] Plot exmaple trial, with score compared across all basis sets

        from pythonlib.tools.plottools import rotate_x_labels, saveMultToPDF

        Cl, Params = self.clustergood_extract_saved_clusterobject(WHICH_FEATURE)
        Y = Cl.Xinput
        assert Y.shape[0]==len(self.Dat)
        list_shape_basis = Params["list_shape_basis"]
        list_strok_basis = Params["list_strok_basis"]

        SIZE = 2
        fig1, ax = plt.subplots(figsize=(len(list_shape_basis*SIZE), 5))
        y = Y[ind, :]
        labels_col = Cl.LabelsCols
        ax.plot(labels_col, y, "-k")
        ax.set_ylim(0, 1)
        rotate_x_labels(ax, 45)

        # Plot basis set
        assert list_shape_basis == labels_col
        strokes = list_strok_basis
        fig2, axes = self.plot_multiple_strok(strokes, overlay=False, titles = list_shape_basis, ncols = len(list_shape_basis))

        # Plot drawing
        inddat, fig3, ax = self.plot_single_overlay_entire_trial(ind, overlay_beh_or_task="task")  
        _, fig4, _ = self.plot_single_overlay_entire_trial(ind, overlay_beh_or_task="beh")  

        if savedir:
            if prefix:
                path = f"{savedir}/{prefix}-ind_DS_{ind}"
            else:
                path = f"{savedir}/ind_DS_{ind}"
            print("Saving to: ", path)
            saveMultToPDF(path, [fig1, fig2, fig3, fig4])
            
        return fig1, fig2, fig3




                # 3) Plot the original plot
        #         cgthis.fig        

    # def cluster_compute_sim_matrix_helper(self, inds, do_preprocess=False, label_by="shape_oriented"):
    #     """ [OLDER CODE] Compute similarity matrix between each pair of trials in inds
    #     PARAMS:
    #     - label_by, either string (col name) or list of string (will make a new grouping 
    #     that conjunction of these)
    #     """

    #     strokes = self.extract_strokes("list_arrays", inds)


    #     # Preprocess storkes if descired
    #     if do_preprocess:
    #         strokes = self._process_strokes_inputed(strokes, min_stroke_length_percentile = None, 
    #             min_stroke_length = None, max_stroke_length_percentile = None)

    #     # labels
    #     if isinstance(label_by, list):
    #         assert False, "to do, take conjuctins, see grouping_append_and_return_inner_items"

    #     label_by = "shape_oriented"
    #     labels = self.Dat.iloc[inds][label_by]

    #     Cl = self._cluster_compute_sim_matrix(strokes, return_as_Clusters=True, 
    #         labels_for_Clusters = labels)

    #     # Plot
    #     Cl.plot_heatmap_data()
    #     Cl.plot_heatmap_data(sortver=0)
    #     Cl.plot_heatmap_data(sortver=2)

    #     return Cl



    ############### STROKE TRANSFORMATIONS/COMPTUATIONS/FEATURES
    def features_compute_velocity_binned(self, twind = (0., 0.2),
        plot_vel_timecourses = False, plot_histogram=False):
        """ Get meamn velocity scalars (x and y) by binning velocity 
        time series.
        PARAMS:
        - twind, window, relative to stroke onset, to extract data, in seconds.
        RETURNS:
        - new columns in self.Dat, "velmean_x", "velmean_y"
        NOTE:
        - 2/6/23 - 
        # Get first 200ms timewindow
        # chose this based on visualizing temporal profile of velocities.
        # - Given velocity and time window, get mean vel
        """
        from pythonlib.tools.nptools import bin_values

        # 2) mean velocity in time window.
        inds = list(range(len(self.Dat)))
        list_strokes_vel = self.extract_strokes_as_velocity(inds)
        out = []
        for strok in list_strokes_vel:
            tvals = strok[:,2]
            tvals = tvals-tvals[0] 
            inds = (tvals>=twind[0]) & (tvals<=twind[1]) # bool mask

            s = strok[inds, :2]
            velmean = np.mean(s,axis=0) # (x,y)

            out.append({
                "strok_vel":strok,
                "velmean_x":velmean[0],
                "velmean_y":velmean[1],
            })

        dfvels = pd.DataFrame(out)

        # save it in self.Dat
        cols_keep = ["velmean_x", "velmean_y"]
        for col in cols_keep:
            self.Dat[col] = dfvels[col]

        # convert to polar
        from pythonlib.tools.vectools import get_angle, bin_angle_by_direction, cart_to_polar
        list_angles = []
        list_magn = []
        for i in range(len(self.Dat)):
            # vec = [self.Dat.iloc[i]["velmean_x"], self.Dat.iloc[i]["velmean_y"]]
            a, norm = cart_to_polar(self.Dat.iloc[i]["velmean_x"], self.Dat.iloc[i]["velmean_y"])
            list_angles.append(a)
            list_magn.append(norm)    
        list_angles_binned = bin_angle_by_direction(list_angles, num_angle_bins=8)
        self.Dat["velmean_th"] = list_angles
        self.Dat["velmean_thbin"] = list_angles_binned
        self.Dat["velmean_norm"] = list_magn
        self.Dat["velmean_normbin"] = bin_values(list_magn, nbins=4)



        if plot_vel_timecourses:
            self.plotwrap_timecourse_vels_grouped_by_shape(5, also_plot_example_strokes=True)
        if plot_histogram:
            # Plot, duration distribution for all strokes
            list_dur = []
            list_strokes_vel = dfvels["strok_vel"].tolist()
            for i in range(len(list_strokes_vel)):
                dur = list_strokes_vel[i][-1, 2] - list_strokes_vel[i][0, 2]
                list_dur.append(dur)

            fig, ax = plt.subplots(1,1)
            ax.hist(list_dur, bins=20);
            ax.axvline(0, color="r")
            ax.set_xlabel("stroke durations (sec)")

            # Joint distribution of velocity scalars
            import seaborn as sns
            sns.pairplot(data=DS.Dat, vars=["velmean_x", "velmean_y"], hue="shape", markers=".")

    ################ Features
    def features_generate_dataset_singletrial(self, shape_key="shape"):
        """ For each row in self.Dat, extract useful information that can be used for 
        e.g., for cluster and similarity analyses
        RETURNS:
        - appends specific columns to self.Dat
        """

        inds = list(range(len(self.Dat)))

        self.Dat["strok_task"] = self.extract_strokes(inds=inds, ver_behtask="task_aligned_single_strok")
        print("Added column: strok_task")
            
        # self.Dat["shape_task"] = self.Dat["shape"]
        # print("Added column: shape_task")

        list_prims = self.extract_strokes(inds=inds, ver_behtask="task_aligned_single_primitiveclass")
        self.Dat["shape_cat_abstract"] = [x.ShapeNotOriented for x in list_prims]
        print("Added column: shape_cat_abstract")
        assert self.Dat["shape_cat_abstract"].tolist() == self.Dat["shapeabstract"].tolist(), "confused why these are diff..."


    def features_generate_dataset_mean_primshape(self, shape_key = "shape", best_n_by_dist=None):
        """ for eahc prim, generate feature vectors of various kinds (spaces),
        which can be used for clustering etc. One row per shape.
        RETURNS:
        - dfdat, dataframe holding one row per shape (i.e, string in shape_char)
        """

        self.shape_switch_to_this_kind(shape_key)

        # DATAPT = mean stroke for each shape

        # Prepare dataset
        list_shape = sorted(self.Dat["shape_oriented"].unique().tolist())
        list_strokmean = []
        self.dataset_append_column("Task")

        # Generate dataframe
        DAT = []
        for shape in list_shape:
            
            # Mean motor stroke (beh)
            strokmean = self.cluster_compute_mean_stroke_thisshape(shape, best_n_by_dist=best_n_by_dist)[0]
            list_strokmean.append(strokmean)
                
            # Image 
            strok_task = self.extract_strokes_example_for_this_shape(shape)

            # shape category
            idxs = self.find_indices_this_shape(shape, return_first_index=True)
            idx = idxs[0]
            list_prims = self.extract_strokes(inds=idxs, ver_behtask="task_aligned_single_primitiveclass")
            assert len(list_prims)==1
            shape_cat = list_prims[0].ShapeNotOriented
            
            # other info
            # character for the first trial that has this shape
            char = self.dataset_extract("character", idx)
            
            DAT.append({
                "strok":strokmean,
                "strok_task":strok_task,
                "shape":shape,
                "shape_cat_abstract":shape_cat,
                "char_first_instance":char
            })

        dfdat = pd.DataFrame(DAT)
        return dfdat


    # def features_generate_clusters_from_dataset(self, dfdat, 
    def features_wrapper_generate_all_features(self, version_trial_or_shapemean="trial",
        which_basis_set=None):
        """ 
        """

        ## GET DATA 
        if version_trial_or_shapemean=="trial":
            # one datapt per trial
            self.features_generate_dataset_singletrial() # Prepare necessary columns
            dfdat = self.Dat
        elif version_trial_or_shapemean=="shapemean":
            # one datapt per shape (defined by aligned task token)
            self.distgood_compute_beh_task_strok_distances()
            dfdat = self.features_generate_dataset_mean_primshape("shape")
        else:
            print(version_trial_or_shapemean)
            assert False

        ClustDict, ParamsDict = self._features_wrapper_generate_all_features(dfdat,
            which_basis_set=which_basis_set)

        ParamsGeneral = {
            "version_trial_or_shapemean":version_trial_or_shapemean
        }

        # Store within DatStrokes
        self.Clusters_ClustDict = ClustDict
        self.Clusters_ParamsDict = ParamsDict
        self.Clusters_ParamsGeneral = ParamsGeneral
        self.Clusters_dfdat = dfdat

        return ClustDict, ParamsDict, ParamsGeneral, dfdat


    def _features_wrapper_generate_all_features(self, dfdat,
        perform_clustering=True, compute_distance_matrix=True,
        which_basis_set=None):
        """ 
        Wrapper to generate all (or many) feature datasets.
        Generate feature vectors, each row in a matrix (ndat x ndim). 
        RETURNS:
        - ClustDict, keys are str representational spaces, and vals 
        are ClusterClass objects holding that data """
        from sklearn.preprocessing import OneHotEncoder
        from pythonlib.cluster.clustclass import Clusters

        ClustDict = {}
        ParamsDict = {}
        # 1) Motor similarity to basis set
        print("TODO: use held out stroke for basis set")
        Cl, params = self.clustergood_featurespace_project(dfdat, "strok_sim_motor",
            list_strok_basis=None, which_basis_set = which_basis_set,
            which_shapes = None,
            list_distance_ver=None)
        # list_strok = dfdat["strok"].values.tolist()
        # list_strok_basis = dfdat["strok"].values.tolist()
        # list_shape = dfdat["shape"].values.tolist()
        # Cl = self._cluster_compute_sim_matrix(list_strok, list_strok_basis,  
        #                                     rescale_strokes_ver=None, return_as_Clusters=True,
        #                                    labels_for_Clusters=list_shape)
        ClustDict["beh_motor_sim"] = Cl
        ParamsDict["beh_motor_sim"] = params

        # 2) Visual similarity to basis set
        Cl, params = self.clustergood_featurespace_project(dfdat, "strok_sim_task",
            list_strok_basis=None, which_basis_set = which_basis_set,
            which_shapes = None,
            list_distance_ver=None)

        # # - for each shape, return a single example task image
        # list_strok = dfdat["strok_task"].values.tolist()
        # list_strok_basis = dfdat["strok_task"].values.tolist()
        # list_shape = dfdat["shape"].values.tolist()

        # # Cluster
        # Cl = self._cluster_compute_sim_matrix(list_strok, list_strok_basis,
        #                                     rescale_strokes_ver=None, distancever="hausdorff_max", return_as_Clusters=True,
        #                                    labels_for_Clusters=list_shape)
        # # Cl = self._cluster_compute_sim_matrix(list_strok, list_strok_basis,
        # #                                     rescale_strokes_ver=None, distancever="hausdorff_centered", return_as_Clusters=True,
        # #                                    labels_for_Clusters=list_shape)
        ClustDict["task_image_sim"] = Cl
        ParamsDict["task_image_sim"] = params

        # 3) shape category (abstract)
        Cl, params = self.clustergood_featurespace_project(dfdat, "shape_cat_abstract",
            list_strok_basis=None, which_basis_set = which_basis_set,
            which_shapes = None,
            list_distance_ver=None)

        # # convert to one-hot

        # encoder = OneHotEncoder(handle_unknown='ignore')
        # X = encoder.fit_transform(dfdat[["shape_cat_abstract"]]).toarray()

        # labels_rows = list_shape
        # labels_cols = encoder.categories_[0].tolist()
        # ClustDict["task_shape_cat_abstract"] = Clusters(X, labels_rows=labels_rows, labels_cols=labels_cols)
        ClustDict["task_shape_cat_abstract"] = Cl
        ParamsDict["task_shape_cat_abstract"] = params

        # 4) curved vs. linear

        # TODO


        # (See notes for other)

        # TODO

        # Confirm that all inputed rows are identical across Data
        prev = None
        for k, Cl in ClustDict.items():
            if prev is not None:
                assert prev == Cl.Labels
            prev = Cl.Labels        

        ##################### Clustering
        if perform_clustering:
            for rep, Cl in ClustDict.items():
                Cl.plot_save_hier_clust()

        return ClustDict, ParamsDict




    ############################### DISTANCES (scoring)
    def _strokes_check_all_aligned_direction(self, strokes, plot_failure =False):
        """ Check that strokes are all aligned. For each adjacent pair in list (strokes),
        if their distance decreases after flipping one, then this means they are NOT aligned.
        Useful if strokes are all of same sahpe, and want to make sure they are same direction
        strokes. 
        Recenters strokes to onset before checking.
        RETURNS:
        - aligned, False if any of the adjacent items in strokes are misaligned
        """
        for strok1, strok2 in zip(strokes[:-1], strokes[1:]):
            strok1 = strok1 - np.mean(strok1, axis=0)
            strok2 = strok2 - np.mean(strok2, axis=0)
            # strok1 = strok1 - strok1[0,:]
            # strok2 = strok2 - strok2[0,:] 

            # COMPUTE similarities
            d_orig = self._cluster_compute_sim_matrix([strok1], [strok2])[0] # (1,)
            d_rev = self._cluster_compute_sim_matrix([strok1], [strok2[::-1]])[0] # (1,)

            # d_orig = self._dist_strok_pair(strok1, strok2, recenter_to_onset=True)
            # d_rev = self._dist_strok_pair(strok1, strok2[::-1], recenter_to_onset=True)
            if d_orig<0.75*d_rev: # 0.75 allows for some noise.
                if plot_failure:
                    # Plot these strokes
                    print("d_orig", d_orig)
                    print("d_rev", d_rev)
                    self.plot_multiple_strok([strok1, strok2], ver="beh", 
                        overlay=False, titles=None, ncols=5, size_per_sublot=2)
                return False
        return True
                
            

    def _dist_strok_pair(self, strok1, strok2, recenter_to_onset=False):
        """ compute doistance between strok1 and 2
        uses by defaiult hausdorff mean
        """

        if recenter_to_onset:
            strok1 = strok1 - strok1[0,:]
            strok2 = strok2 - strok2[0,:]
        from pythonlib.drawmodel.strokedists import distMatrixStrok
        return distMatrixStrok([strok1], [strok2], convert_to_similarity=False).squeeze().item()


    def _dist_alignedtask_to_beh(self, list_inds):
        """ COmpute and return distances between strok beh and task, one scalar
        for each index (in list_inds)
        RETURNS:
        - list of scalars, each a hsuasdorff dist, smaller the better
        """
        list_strok_beh = self.extract_strokes(inds=list_inds, ver_behtask="beh")
        list_strok_task = self.extract_strokes(inds=list_inds, ver_behtask="task_aligned_single_strok")

        list_dists = []
        for sb, st in zip(list_strok_beh, list_strok_task):
            d = self._dist_strok_pair(sb, st)
            list_dists.append(d)

        return list_dists



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


    ################################ UTILS
    def shape_switch_to_this_kind(self, shape_kind):
        """ Upodates what is the representation of shape that is in "shape_oriented" column,
        which is the "final" column. 
        PARAMS:
        - shape_kind, str. 
        --- if "shape_char", then values are "shape|{index_char}"
        --- if "shape", then values are "shape"
        NOTE:
        - can run this as many times as wants, is not history dependent.
        """
        self.Dat["shape_oriented"] = self.Dat[shape_kind]

    def strokerank_extract_semantic(self):
        """ using stroke indices, decide if is first, last, both_fl, or middle
        stroke
        RETURNS:
        - appends/modifies self.Dat["stroke_index_semantic"
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        def F(x):
            if x["stroke_index"]==0 and x["stroke_index_fromlast"]==-1:
                return "both_fl"
            elif x["stroke_index"]==0:
                return "first"
            elif x["stroke_index_fromlast"]==-1:
                return "last"
            else:
                return "middle"
        self.Dat = applyFunctionToAllRows(self.Dat, F, newcolname="stroke_index_semantic")

    # ################################# FIRST TOUCH
    # def firsttouch_extract_plot(self):
    #     ## Positions of first touch for each stroke

    #     assert False, "just copied from notebook. have not checked if working../"

    #     def F(x):
    #         return x["strok"][0][:2]
            
    #     DS.Dat = applyFunctionToAllRows(DS.Dat, F, 'pt_first_touch')


    #     # keep only specific location and size
    #     print(DS.Dat["gridloc"].value_counts())
    #     print(DS.Dat["gridsize"].value_counts())

    #     gloc = (0,0)
    #     dfthis = DS.Dat[DS.Dat["gridloc"]==gloc]

    #     gsize = "rig3_3x3_small"
    #     dfthis = dfthis[dfthis["gridsize"]==gsize]

    #     print("N= ", len(dfthis))

    #     # CONFIRM SAME LOC AND SIZE
    #     print(dfthis["gridloc"].value_counts())
    #     print(dfthis["gridsize"].value_counts())

    #     ##### Organize them into "sets" based on expected location of first touch.

    #     from pythonlib.tools.pandastools import grouping_get_inner_items, extract_trials_spanning_variable
    #     from pythonlib.tools.plottools import subplot_helper


    #     n = 15
    #     outdict, levels = extract_trials_spanning_variable(dfthis, "shape_oriented", n_examples=n, return_as_dict=True,
    #                                                       method_if_not_enough_examples = "prune_subset")

    #     getax, figholder, nplots = subplot_helper(5, 10, len(outdict), SIZE=4)

    #     for i, (shape, list_inds) in enumerate(outdict.items()):
    #         ax = getax(i)
            
    #         # get all pts
    #         pts = DS.Dat.iloc[list_inds]["pt_first_touch"]
            
    #         for pt in pts:
    #             ax.plot(pt[0], pt[1], 'x');
            
    #         # plot the task
    #         strok_task = DS.extract_strokes(inds=[list_inds[0]], ver_behtask ="task")[0]
    #         ax.plot(strok_task[:,0], strok_task[:,1], '-.k')
            
            
    #         ax.grid()
    #         ax.set_title(shape)

    #     # plot all shapes in order
    #     DS.plot_examples_grid(col_levels=levels)

    #################################### STROKES REPOSITORY
    def _stroke_shape_cluster_database_path(self, SDIR=f"{PATH_DATASET_BEH}/STROKES", 
            suffix=None, animal=None, expt=None, date=None):
        """ Generate the path storing the mean shapes strokes (repository)
        """
        if suffix is not None:
            suffix = f"-{suffix}"
        else:
            suffix = ""
        if expt is None:
            expt = "_".join(self.Dataset.expts())
        if animal is None:
            animal = "_".join(self.Dataset.animals())
        if date is None:
            date = "_".join(self.Dataset.Dat["date"].unique().tolist())

        sdir = f"{SDIR}/{animal}-{expt}-{date}{suffix}"
        path = f"{sdir}/strokes_each_shape_dataframe.pkl"

        return sdir, path


    def stroke_shape_cluster_database_save(self, suffix=None, overwrite_ok=False,
        plot_each_shape=True, best_n_by_dist=15):
        """
        Save strokes (np arrays) for each shape, after taking the mean. Useful
        clustering analyses.
        PARAMS:
        - best_n_by_dist, int, how many fo the top trials to take for computing
        mean stroke (top, by sim to task)
        RETURNS:
        - saves to , a dataframe with strokes.
        NOTE: default doesnt allow overwrite. to do so, input a new suffix
        """
        import os

        print("TODO: Extract and save plots of each indiv stroke to be sure all are good trials.")
        sdir, path = self._stroke_shape_cluster_database_path(suffix=suffix)
        os.makedirs(sdir, exist_ok=overwrite_ok)

        # 2) Extract strokes.
        dfdat = self.features_generate_dataset_mean_primshape("shape", best_n_by_dist=best_n_by_dist)
        dfdat.to_pickle(path)
        print("Saved to:", path)

        # Plot each stroke
        if False:
            strokes = dfdat["strok"].tolist()
            shapes = dfdat["shape"].tolist()
            fig, axes = self.plot_multiple_strok(strokes, overlay=False, titles=shapes)
            fig.savefig(f"{sdir}/mean_stroke_each_shape.pdf")
        else:
            # sort by name of prim
            dfdat = dfdat.sort_values("shape")
            list_strok = dfdat["strok"].tolist()
            list_shape = dfdat["shape"].tolist()
            # centerize task strokes, otherwise they are in space
            list_strok_task = dfbasis["strok_task"].tolist()
            list_strok_task = [x-np.mean(x, axis=0, keepdims=True) for x in list_strok_task]

            for i, titles in enumerate([None, list_shape]):
                fig, axes = DS.plot_multiple_strok(list_strok, overlay=False, ncols=9, titles=titles);
                savefig(fig, f"{sdir}/mean_stroke_each_shape-{i}-BEH.pdf")

                fig, axes = DS.plot_multiple_strok(list_strok_task, ver="task", overlay=False, ncols=9, titles=titles);
                savefig(fig, f"{sdir}/mean_stroke_each_shape-{i}-TASK.pdf")


    def stroke_shape_cluster_database_load(self, animal, expt, date, suffix):
        """ Load set of strokes, one for each sahpe, previously saved 
        RETURNS:
        - pd dataframe, each row a stroke instance (usually mean over trials).
        """

        sdir, path = self._stroke_shape_cluster_database_path(suffix=suffix,
            animal=animal, expt=expt, date=date)
        dfdat = pd.read_pickle(path)
        return dfdat, sdir

    def stroke_shape_cluster_database_load_helper(self, which_basis_set=None,
        which_shapes="all_basis", hand_entered_shapes = None, plot_examples=False,
        return_sdir=False):
        """ Helper, loads repositiory of basis set of stropkes, and slices to
        a desired set of shapeas.
        PARAMS:
        - which_basis_set, str, which file to load.
        - which_shapes, list of str, which shapes to pull out of this vbasis set
        - hand_entered_shapes, bool [optional], if which_shapes=="hand_enter"
        RETURNS:
        - dfbasis, dataframe, each row the shape
        - list_strok_basis, list of np array (strok)
        - list_shape_basis, list of str, matching order of list_strok_basis
        """
        from pythonlib.tools.pandastools import slice_by_row_label

        # v1, random subset of strokes.
        # v2, hard coded primitives database.
        # /gorilla1/analyses/database/STROKES/Pancho-priminvar3m-230104/strokes_each_shape_dataframe.pkl
        # dfbasis = DS.stroke_shape_cluster_database_load("Pancho", "priminvar3g", 221217, None)
        if which_basis_set is None:
            # Load the one for this dataset's subject
            if self.Dataset.animals()==["Pancho"]:
                WHICH_BASIS_SET = "standard_17"
            elif self.Dataset.animals()==["Diego"]:
                WHICH_BASIS_SET = "diego_all_minus_open_to_left"
            else:
                print(self.Dataset.animals())
                assert False
        elif which_basis_set=="Pancho":
            WHICH_BASIS_SET = "standard_17"
        elif which_basis_set=="Diego":
            WHICH_BASIS_SET="diego_all_minus_open_to_left"
        else:
            # Use the input
            WHICH_BASIS_SET = which_basis_set

        # Load the set
        if WHICH_BASIS_SET=="standard_17":
            dfbasis, sdir = self.stroke_shape_cluster_database_load("Pancho", "priminvar4", 220918, None)
            SHAPES_EXCLUDE = []
        elif WHICH_BASIS_SET=="diego_all_minus_open_to_left":
            # All except those three prims that open to the left. he struggled in rig.

            # Ingnore this one becuase it is missing lines.
            # dfbasis = self.stroke_shape_cluster_database_load("Diego", "primsingridall1c", 230418, None)

            # This one was the last one in rig before took long break doing grid tasks.
            # Looks almost identical to above. It includes the 3 tasks that open to left, but here
            # code exludes therm
            dfbasis, sdir = self.stroke_shape_cluster_database_load("Diego", "primsingridrand6b", 230223, None)
            SHAPES_EXCLUDE = ["V-2-1-0", "arcdeep-4-1-0", "usquare-1-1-0", "dot-2-1-0"]
        else:
            print(WHICH_BASIS_SET)
            assert False



        # Which shapes to extract from this set
        if which_shapes=="current_dataset":
            # v1: Use the shapes in the ground truth tasks.
            list_shape_basis = DS.Dat["shape"].unique().tolist()
            list_shape_basis = [shape for shape in list_shape_basis if shape not in SHAPES_EXCLUDE]
        elif which_shapes=="all_basis" or which_shapes is None:
            # use all within the basis set
            list_shape_basis = sorted(dfbasis["shape"].unique().tolist())
            list_shape_basis = [shape for shape in list_shape_basis if shape not in SHAPES_EXCLUDE]
        elif which_shapes=="hand_enter":
            assert hand_entered_shapes is not None
            list_shape_basis = hand_entered_shapes
        elif which_shapes=="Pancho":
            # Those in panchos... load it and get the shapes
            _, _, list_shape_basis = self.stroke_shape_cluster_database_load_helper("Pancho")
        else:
            assert False
        print("Basis set of strokes:", list_shape_basis)
        dfbasis = slice_by_row_label(dfbasis, "shape", list_shape_basis)
        list_strok_basis = dfbasis["strok"].values.tolist()

        # Plto some examples for sanity check
        if plot_examples:
            self.plot_multiple_strok(list_strok[:4])
            self.plot_multiple_strok(list_strok_basis)

        # print(list_shape_basis)
        # assert False
        if return_sdir:
            return dfbasis, list_strok_basis, list_shape_basis, sdir
        else:
            return dfbasis, list_strok_basis, list_shape_basis

    def shape_extract_indices_sorted_by_behtaskdist(self, shape):
        """ for this shape, extract its indices, sorted from low to high,
        for distnace beh to task stroke
        RETURNS:
        - inds, list of indices into self.Dat
        - dists, the matching distances.
        """

        from pythonlib.tools.listtools import random_inds_uniformly_distributed

        inds = self.Dat[self.Dat["shape"]==shape].index.tolist()        
        dists = self.Dat.iloc[inds]["dist_beh_task_strok"].tolist()

        # do sort
        this = [(i, d) for i, d in zip(inds, dists)]
        this = sorted(this, key=lambda x:x[1]) # increasing dists
        inds = [t[0] for t in this]
        dists = [t[1] for t in this]

        return inds, dists


    def find_indices_this_shape(self, shape, return_first_index=False,
            nrand = None):
        """
        Return Indices with this shape (in the shape_oriented) column
        PARAMS;
        - nrand, int, then get random, sorted, subsample. if None, then get all.
        RETURNS:
        - list of ints
        """
        tmp = self.find_indices_these_shapes([shape], return_first_index=return_first_index)
        inds = tmp[shape]
        if nrand is not None:
            import random
            if len(inds)>nrand:
                inds = sorted(random.sample(inds, nrand))
        return inds

    def find_indices_these_shapes(self, list_shapes, return_first_index=False,
        nrand_each = None):
        """ Return indices for these shapes. Returns the dict with keys in order
        of input list-shapes
        PARAMS:
        - nrand_each, int, then gets this many of each. None gets all
        RETURNS
        - dict, with shape:list of indices
        """
        grouping = self.grouping_get_inner_items(groupouter="shape_oriented", 
            groupinner="index", nrand_each=nrand_each)
        grouping = {shape:grouping[shape] for shape in list_shapes}
        # grouping = {k:v for k, v in grouping.items() if k in list_shapes}
        if return_first_index:
            for k, v in grouping.items():
                grouping[k] = [v[0]]
        return grouping


    def filter_dataframe(self, filtdict, modify_in_place=False):
        """ 
        PARMS

        Modifies self.Dat to retain the output after filtering self.Dat,
        with indices reset.
        """
        from pythonlib.tools.pandastools import filterPandas

        print("staritng legnth: ", len(self.Dat))
        df = filterPandas(self.Dat, filtdict)
        print("final legnth: ", df)

        if modify_in_place:
            print("Modified self.Dat!")
            self.Dat = df

        return df

    def copy(self):
        """ Make a copy
        Will copy data specific to self, but wil not copy the Trial Dataset
        """

        assert False, "in progres..."
        self.Dataset = Dataset
        self.Dat = None
        self.Params = {}
        self.Version = None


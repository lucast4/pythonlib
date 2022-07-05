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

        assert version=="beh", "not yhet coded!!"
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
            primlist, datsegs_behlength, _ = D.behclass_extract_beh_and_task(ind)
            
            # 2) Information about task (e..g, grid size)
            
            # 2) For each beh stroke, get its infor
            for i, (stroke, dseg) in enumerate(zip(primlist, datsegs_behlength)):
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
    def extract_strokes(self, version="list_arrays", inds=None):
        """ Methods to extract strokes across all trials
        PARAMS:
        - version, string, what format to output
        - inds, indices, if None then gets all
        RETURNS:
        - strokes
        """

        if inds is None:
            inds = range(len(self.Dat))

        if version=="list_arrays":
            # List of np arrays (i.e., "strokes" type)
            strokes = [self.Dat.iloc[i]["Stroke"]() for i in inds]
        elif version=="list_list_arrays":
            # i.e., like multiple strokes...
            strokes = [[self.Dat.iloc[i]["Stroke"]()] for i in inds]
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

    def plot_multiple(self, list_inds):
        """ Plot mulitple strokes, each on a subplot
        """
        strokes = self.extract_strokes("list_list_arrays", inds = list_inds)
        self.Dataset.plotMultStrokes(strokes)

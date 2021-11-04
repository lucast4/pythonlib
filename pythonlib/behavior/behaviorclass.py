""" 
Each instance is a single beh trial.
Consolidating the following:
- beh labeling, discretization, and processing
- motor timing, etc.
- lots of stuff that already done in Dataset, but here inlcude all the deeper analyses.
- and so on...
"""
import numpy as np

class BehaviorClass(object):
    """docstring for BehaviorClass"""
    
    def __init__(self, params, ver="dataset"):
        """ 
        PARAMS:
        - params. Different methods (ver) for params entry
        if ver =="dataset", then enter a single row in Dataset object.
        params["ind"], int, row in Dataset
        params["dataset"], Dataset instance
        if ver=="raw", then enter all these things by hand.
        params["strokes_beh"]
        params["strokes_task"]
        params["TaskClass"]
        """

        self.Dataset = None
        self.IndDataset = None


        if ver=="dataset":
            D = params["D"]
            ind = params["ind"]
            Task = D.Dat.iloc[ind]["Task"]

            strokes_task = D.Dat.iloc[ind]["strokes_task"]

            # 2) Get the "names" for all task strokes
            list_shapes = [s[0] for s in Task.Shapes]
            assert(len(list_shapes)==len(strokes_task))
            assert(all([isinstance(s, str) for s in list_shapes]))

            # 3) Compute best alignemnt with beh
            from pythonlib.tools.stroketools import assignStrokenumFromTask
            strokes_beh = D.Dat.iloc[ind]["strokes_beh"]
            
            # if False:
            #     strokes_beh = [np.copy(strokes_beh[0]), np.copy(strokes_beh[1])]
            # else:
            #     strokes_beh = [np.copy(s) for s in strokes_beh] + [np.copy(strokes_beh[0]), np.copy(strokes_beh[1])]
            if False:
                inds_task = assignStrokenumFromTask(strokes_beh, strokes_task, ver="stroke_stroke_lsa_usealltask")

            # 4) Plot 
            # D.plotSingleTrial(ind);
            # # reshuffle task strokes then plot in order
            if False:
                strokes_task_aligned = [strokes_task[i] for i in inds_task]
                D.plotMultStrokes([strokes_beh, strokes_task, strokes_task_aligned]);

            # Try abnother version
            inds_task, list_distances = assignStrokenumFromTask(strokes_beh, strokes_task, 
                ver="each_beh_stroke_assign_one_task")

            # 6) For each beh stroke, give it a label
            labels = []
            for i in inds_task:
                if len(i)==1:
                    labels.append(list_shapes[i[0]])
                elif len(i)==2:
                    # then concatenate into a string
                    tmp = sorted([list_shapes[ii] for ii in i])
                    labels.append("-".join(tmp))

            # Timing distributions
            motordat = D.get_motor_stats(ind)

            # - gap durations
            gap_durations = np.array(motordat["ons"])[1:] - np.array(motordat["offs"])[:-1]
            stroke_durations = np.array(motordat["offs"]) - np.array(motordat["ons"])


            ##### MASKS
            # mask, to pick out only the first sucesful getting of each prim.
            inds_task_done_sofar = []
            inds_task_mask = []
            for indthis in inds_task:
                if all([i in inds_task_done_sofar for i in indthis]):
                    # then this is not new
                    inds_task_mask.append(False)
                else:
                    # then you got something new
                    inds_task_mask.append(True)
                    inds_task_done_sofar.extend(indthis)

            # mask, true only if beh stroke matches only a single task stroke
            inds_task_single = np.ndarray(len(inds_task))
            for i, val in enumerate(inds_task):
                if len(val)>1:
                    inds_task_single[i] = np.nan
                else:
                    inds_task_single[i] = val[0]
            maskinds_single = ~np.isnan(inds_task_single)
            if False:
                print("here")
                print(inds_task)
                print(inds_task_single)
                print(maskinds_single)
                print(type(maskinds_single))

            # mask based on distance to match. if worse then threshold, then mask this off
            if False:
                shape_dist_thresholds = params["shape_dist_thresholds"]
                mask_shapedist = []
                for sh, dist in zip(labels, list_distances):
                    if dist>shape_dist_thresholds[sh]:
                        mask_shapedist.append(False)
                    else:
                        mask_shapedist.append(True)
                mask_shapedist = np.array(mask_shapedist)
            else:
                mask_shapedist = None


            self.Dat = {
                "strokes_task":strokes_task,
                "shapes_task":list_shapes,
                "strokes_beh":strokes_beh,
                "taskstrokeinds_beh":inds_task, # list of list
                "taskstrokeinds_dists":list_distances.squeeze(), # nparray
                "taskstrokeinds_beh_singleonly":inds_task_single, # np array, either single task ind, or nan
                "shapes_beh":labels,
                "gap_durations":gap_durations,
                "stroke_durations":stroke_durations,
                # "maskinds_repeated_beh": np.array(inds_task_mask, dtype=int)
                "maskinds_repeated_beh": np.array(inds_task_mask), # true if this beh mathces taskind that has not been gotetn yuet
                "maskinds_singlematch_beh": maskinds_single, # True if this beh matches only one task. 
                "maskinds_shapedist_beh": mask_shapedist, #
                "stroke_dists": np.array(motordat["dists_stroke"]),
                "gap_dists": np.array(motordat["dists_gap"]),
                "trialcode":D.Dat.iloc[ind]["trialcode"],
                "character":D.Dat.iloc[ind]["character"]}

            self.Dataset = D
            self.IndDataset = ind
        else:
            assert False, "not coded"

    ##################### SHAPES, LABELS, etc
    def shapes_nprims(self, ver="task", must_include_these_shapes=["circle", "line"]):
        """ count how oftne each shape occurs in the beh or task.
        PARAMS:
        - ver, whether to use "beh" or "task" shapes.
        - must_include_these_shapes, can be 0.
        RETURNS:
        - dict, num times each shape occured. e.g., {"circle":5, ...}
        """

        out = {}
        for sh in must_include_these_shapes:
            out[sh] = 0

        if ver=="task":
            list_shapes = self.Dat["shapes_task"]
        elif ver=="beh":
            list_shapes = self.Dat["shapes_beh"]
        else:
            assert False

        for sh in list_shapes:
            if sh in out.keys():
                out[sh]+=1
            else:
                out[sh]= 1
        return out


    ##################### MOTOR STATS
    def motor_extract_durations(self, list_inds, ver="stroke"):
        """ Extract either gap or stroke durations, given a list of beh stroke inds.
        PARAMS:
        - ver, str, either "stroke" or "gap"
        - list_inds, list of ints. interpretations depends on ver.
        --- if ver=="stroke", returns list of stroke durs. straighforwad
        --- if ver=="gap", then returns gaps between adjacent inds. asserts that inds are monotinic
        incresaing. e..g, [1,2,3], then returns gap betwen [1-2, 2-3]
        RETURNS:
        - array_durs, see above.
        """

        if ver=="stroke":
            return self.Dat["stroke_durations"][list_inds]
        elif ver=="gap":
            assert(np.all(np.diff(list_inds)==1.)), "need to be monotinoc increasing"
            return self.Dat["gap_durations"][list_inds[:-1]]

    def motor_extract_dists(self, list_inds, ver="stroke"):
        """ Extract dists in pixels for stroke or gaps. See 
        motor_extract_durations for params
        """
        if ver=="stroke":
            return self.Dat["stroke_dists"][list_inds]
        elif ver=="gap":
            assert(np.all(np.diff(list_inds)==1.)), "need to be monotinoc increasing"
            return self.Dat["gap_dists"][list_inds[:-1]]


    ##################### HELPERS TO EXTRACT
    def extract_strokes(self, ver="beh", list_inds=None):
        """ Extract strokes for this trial.
        PARAMS:
        - ver, str, from {'beh', 'task'}
        - list_inds, 
        --- None, then returns entire strokes
        --- list of ints, then returns just those trajectories in strokes. in the order
        provided
        RETURNS:
        - strokes, list of np arrays
        """

        if ver=="beh":
            strokes = self.Dat["strokes_beh"]
        elif ver=="task":
            strokes = self.Dat["strokes_task"]
        else:
            assert False

        if list_inds is None:
            return strokes
        else:
            return [strokes[i] for i in list_inds]

    def extract_taskstrokes_aligned_to_this_beh(self, ind, concat_task_strokes=False):
        """
        PARAMS:
        - ind, int, index into beh strokes
        - concat_task_strokes, then returns a single traj (concatted). might not be in a
        a good order...
        RETURNS:
        - inds_task, list of ints, the task inds matching this beh stroke
        - strokes_task, the strokes matching inds_task
        - dist, distance for this match (higher the worse)
        """

        taskinds = self.Dat["taskstrokeinds_beh"]
        taskinds_this = taskinds[ind]

        strokes_task = self.Dat["strokes_task"]
        strokes_task = [strokes_task[i] for i in taskinds_this]

        distances = self.Dat["taskstrokeinds_dists"]
        distances = distances[ind]

        if concat_task_strokes:
            strokes_task = [np.concatenate(strokes_task, axis=0)]


        return taskinds_this, strokes_task, distances



    ###################### FIND THINGS
    def find_labelmotif(self, labelmotif, ver="shapes", good_strokes_only=False):
        """ find cases where this labelmotif is present in this beh trial
        PARAMS:
        - labelmotif, list of tokens, either int or strings, depending on ver
        - ver, which beh representation to care about.
        --- "shapes", then is list of shape names
        --- "inds" then is list of task stroke inds.
        - good_strokes_only, then applies masks to get only good strokes. will not allow skips over
        bad storkes.
        RETURNS:
        - list_matches, list of list, where each inner list are the indices into list_beh which 
        would match motif. e.g., [[0,1], [4,5]]
        """
        
        if ver=="shapes":
            list_beh = self.Dat["shapes_beh"]
        elif ver=="inds":
            list_beh = self.Dat["taskstrokeinds_beh_singleonly"]
        else:
            assert False

        if good_strokes_only:
            # Generate mask, so only conisder good strokes
            maskinds = self.Dat["maskinds_repeated_beh"] & self.Dat["maskinds_singlematch_beh"]
        else:
            maskinds = None

        return self._find_motif_in_beh(list_beh, labelmotif, list_beh_mask=maskinds)



    def _find_motif_in_beh(self, list_beh, motif, list_beh_mask=None):
        """ Generic - given list of beh tokens, and a motif, find if/where this
        motif occurs.
        PARAMS:
        - list_beh, list of tokens, either int or str. e..g, [line, circle, ...]
        - motif, list of tokens, same type as in list_beh. this is the filter.
        - list_beh_mask, np array of bool int, same len as list_beh. baseically says that 
        only conisder substrings in list_beh for which all tokens are True in this mask.
        RETURNS:
        - list_matches, list of list, where each inner list are the indices into list_beh which 
        would match motif. e.g., [[0,1], [4,5]]
        """

        def _motifs_are_same(behstring, motif):
            assert len(behstring)==len(motif)
            for a, b in zip(behstring, motif):
                if not a==b:
                    return False
            return True

        nend = len(list_beh) - len(motif)
        nmotif = len(motif)
        if list_beh_mask is not None:
            assert len(list_beh)==len(list_beh_mask)
        
        if len(list_beh)<nmotif:
            return []

        list_matches = []
        for i in range(nend+1):
            
            behstring = list_beh[i:i+nmotif]

            if list_beh_mask is not None:
                behstring_mask = list_beh_mask[i:i+nmotif]
                if ~np.all(behstring_mask):
                    # skip this, since not all beh strokes are unmasked
                    continue

            if _motifs_are_same(behstring, motif):
                list_matches.append(list(range(i, i+nmotif)))

        return list_matches

    ##################################### PLOTS

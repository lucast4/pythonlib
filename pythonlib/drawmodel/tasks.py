""" stuff to do with tasks...
specifically for tasks in monkeylogic
"""
import numpy as np
from pythonlib.tools.stroketools import fakeTimesteps

class TaskClass(object):
    """ Holds a single task object.
    """

    def __init__(self, task):
        """ task can be formated in several ways:
        (1) ML2 task, which is a dict. 
        i.e., task = getTrialsTask(fd, 1), or
        _, tasklist = Probedat.pd2strokes(Probedat.Probedat) [this
        is same, but has also "strokes" and "fixpos"
        ... (nothing else coded so far)
        """
        self.Task = task
        if "strokes" not in self.Task.keys():
            assert False, "see Probedat.pd2strokes for how to get this."
        self.Strokes = self.Task["strokes"]
        self.Program = None

        if isinstance(task, dict):
            keys_to_check = ["x", "y", "stage"] # simply keys that are diagnostic of this being ml2 task
            if all(k in task.keys() for k in keys_to_check):
                # then this is ml2 task
                self._cleanMonkeylogic()

        # Default for affine transfomations of objects.
        self._features_default()

    def _features_default(self):
        """ default affine features"""
        self.FEATURES_DEFAULT = {
            "x":0.,
            "y":0.,
            "th":0.,
            "sx":1., 
            "sy":1.,
            "order":"trs"}
        for k, v in self.FEATURES_DEFAULT.items():
            if not isinstance(v, str):
                self.FEATURES_DEFAULT[k] = np.asarray(v)

    def _cleanMonkeylogic(self):
        """ clean monkeylogic tasks
        """

        # 1) remove redundant information
        del self.Task["x_before_sketchpad"]
        del self.Task["y_before_sketchpad"]
        del self.Task["x"]
        del self.Task["y"]
        del self.Task["x_rescaled"]
        del self.Task["y_rescaled"]



    ####################### ASSIGN EACH TASK STROKE A LABEL
    def assignLabelToEachStroke(self, method):
        """ label is generally string, but is like a class
        - method is either string, or a function, i..e,
        func(self.Task) --> <list of labels>.
        """
        if isinstance(method, str):
            assert False, "not coded yet"
        else:
            # assume method is a function
            lab = method(self.Task)
            if len(lab)==0:
                assert False
            self.TaskLabels = lab


    ####################### HELPERS
    def get_tasknew(self):
        """ extract tasknew, which is the one from makeDrawTasks
        """
        return self.Task["TaskNew"]

    ####################### GET FEATURES
    # (see efficiency cost model)


    ####################### GET PARSES
    # (see taskmodel and below)


    ####################### TAKE IN A BEHAVIORAL TRIAL AND DO SOMETHING
    def behAssignClosestTaskStroke(self, strokes_beh):
        """ for each beh stroke, it is "assigned" a task stroke (not vice versa)
        assign each stroke the task stroke that is the closest
        """
        from pythonlib.tools.vectools import modHausdorffDistance
        if len(strokes_beh)==0:
            assert False, "strokes_beh is empty" 
        strokes_task = self.Task["strokes"]
        assignments =[]
        for s_beh in strokes_beh:
            
            # get distnaces from this behavioal stroke to task strokes
            distances = [modHausdorffDistance(s_beh, s_task) for s_task in strokes_task]

            # if len(distances)==0:
            #     asdfasfsdf
            # assign the closest stroke
            assignments.append(np.argmin(distances))
        return assignments

    ######################## PLOTS
    def plotTaskOnAx(self, ax, plotkwargs = {}):
        """ plot task on axes"""
        from pythonlib.drawmodel.strokePlots import plotDatStrokes
        strokes = self.Strokes
        
        plotkwargs["clean_unordered"] = True
        plotDatStrokes(strokes, ax, **plotkwargs)

        # limits
        spad = self.Task["sketchpad"]
        ax.set_xlim(spad[:,0])
        ax.set_ylim(spad[:,1])


    ############# CONVERT TO OBJECTS
    # Objects is list of primitives, each with its own independent transofmation (locaiton, etc)

    def objects_extract(self):
        """ Initial extraction of objects (and their transforms).
        RETURNS:
        - modified self.Objects, which is list of dicts, one for each object.
        NOTE:
        - There are multiple sources of this information. Get all, and make sure they corroborate.
        Also useful for me so here saving notes that might be useful later.
        """
        T = self.get_tasknew()

        # V1: based on saved Objects
        if "Objects" in T.keys():
            # New version, like 6/2021 
            shapes = T["Objects"]["Features"]["shapes"] # dict
            shapes = self._program_line_dict2list(shapes)

        # V2: if this is "mixture2" task, then the mixture2 params
        Task = T["Task"]
        if Task["stage"]=="mixture2":
            # Then the input params for this (makeDrawTasks params) explicitly defines
            # objects in order.
            shapes = Task["info"]["params"]["objectsInOrder"]
            shapes = self._program_line_dict2list(shapes)
            
            tforms_subprog = Task["info"]["params"]["tformsSubprograms"]
            tforms_subprog = self._program_line_dict2list(tforms_subprog)

        # V3: analyze program, extract the primitives.
        self.program_extract()
        Objects = []
        for i in range(len(self.Program)):
            Objects.append(self.program_interpret_subprog(i))
        self.Objects = Objects



    ############# PROCESS MATLAB PROGRAMS TO USEFUL FORMAT
    def _program_line_dict2list(self, line):
        """ given dict, where each item is indexed by number (in string format),
        converts to list, where order is maintained, using values from dict,
        and taking care of convering numbers to squeezed arrays, 
        NOTE: Is recursive.
        e.g., 
        INPUT:
        - line={'1': 'transform', '2': array([[0.]]), '3': array([[0.]]), '4': array([[0.]]), '5': array([[1.]]), '6': array([[1.]]), '7': 'trs'}
        OUTPUT:
        - line_list = ['transform', array(0.), array(0.), array(0.), array(1.), array(1.), 'trs']"""

        def _f(x):
            if isinstance(x, str):
                return x
            elif isinstance(x, np.ndarray):
                return x.squeeze()
            elif isinstance(x, dict):
                # Then this is nested dict...
                return self._program_line_dict2list(x)
            else:
                print(type(x))
                assert False, "if this is dict, then you are looking at nested dict."
        line_list = []
        for i in range(len(line)):
            idx = f"{i+1}"
            val = _f(line[idx])
            line_list.append(val)
        return line_list


    def program_extract(self, skip_if_already_done=True):
        """ Initial extraction of program into useful format.
        OUTPUT:
        - assigns to self.Program
        NOTE: Skips running if self.Program is already present
        """

        if skip_if_already_done:
            if hasattr(self, "Program"):
                if self.Program is not None:
                    if len(self.Program)>0:
                        return

        program = self.get_tasknew()["Task"]["program"]
        if True:
            # use recursive function
            program_list = self._program_line_dict2list(program)
        else:
            # OLD, 
            # convert to nested lists.
            nlev1 = len(program)
            program_list = []
            for i in range(nlev1):
                idx = f"{i+1}"
                if doprint:
                    print("subprog:", idx)
                subprog = program[idx]
                nlev2 = len(subprog)
                
                # print(subprog)
                # assert False
                # Convert subprogram to list
                subprog_list = []
                for ii in range(nlev2):
                    idx = f"{ii+1}"
                    if doprint:
                        print("-- line:", idx)
                    line = subprog[idx]
                    line_list = self._program_line_dict2list(line)
                    if doprint:
                        print(line_list)
                    subprog_list.append(line_list)
                    
                program_list.append(subprog_list)
        
        self.Program = program_list

    def program_get_line(self, ind_subprog, ind_line):
        """ returns line from extracted Program (self.Program)
        e.g., program[ind_subprog][ind_line]
        - inds in set, {-1, 0, 1, ...}
        - EG:
        --- prog_get_line(program_list, 0, -1)
        """
        self.program_extract()
        program = self.Program

        if ind_line != -1 and len(program[ind_subprog])<=ind_line:
            print(len(program[ind_subprog]))
            print(ind_line)
            assert False, "you are asking for a line that doesnt exist"
            
        return program[ind_subprog][ind_line]

    def program_get_tform_dict(self, x=None, y=None, th=None, sx=None, sy=None, order=None):
        if not hasattr(self, "FEATURES_DEFAULT"):
            self._features_default()
        tform = self.FEATURES_DEFAULT.copy()
        if x:
            tform["x"] = x
        if y:
            tform["y"] = y
        if th:
            tform["th"] = th
        if sx:
            tform["sx"] = sx
        if sy:
            tform["sy"] = sy
        if order:
            tform["order"] = order
        return tform

    def program_compare_tform_dicts(self, dict1, dict2):
        keys = self.program_get_tform_dict().keys()
        for k in keys:
            if isinstance(dict1[k], str):
                if not dict1[k]==dict2[k]:
                    return False
            else:
                if not np.isclose(dict1[k], dict2[k]):
                    return False
        return True

    def program_interpret_line(self, line, return_none_if_tform_static=True):
        """ helper to interpret a single subprogram line, (list),
        into a dict with relevant features labeled. does things like: fills
        in missing variables using default features, etc.
        - pass in a single line (e.g, output of program_get_line)
        - return_none_if_tform_static, returns None if a line says to do a tformation
        that has no affect.
        NOTE:
        Assumptions: 
        - first iutem in a line is the kind, like "line", or "tform"
        - remaining items are tforms in order as defined by default params.
        """
        
        # Default things
        FEATURES_NAMES = ["x", "y", "th", "sx", "sy", "order"]
        if not hasattr(self, "FEATURES_DEFAULT"):
            self._features_default()
        FEATURES_DEFAULT = self.FEATURES_DEFAULT
        assert FEATURES_NAMES==list(FEATURES_DEFAULT.keys()), "order maintined for python 3.7+"
        
        # Break down this line into features
        kind = line[0]
        feats = line[1:]
        out = {} 

        def _affine_feats(feats):
            """ feats is variable length list, where position
            in list indicates what kind of feature, in this order:
            [x, y, th, sx, sy, order]. Can be empty. 
            RETURNS:
            - dict[feat] = val
            """
            outthis = {}
            for name, val in zip(FEATURES_NAMES, feats):
                outthis[name] = val
            return outthis
            
        prims_list = ["line", "circle", "dot"]
        if kind in prims_list:
            # Is a stroke 
            out["type"] = "stroke"
            out["kind"] = kind
            
            # what is affine transformation
            out["affine"] = self.program_get_tform_dict(*feats)

        elif kind=="transform":
            # Then this line is an affine transform
            out["type"] = "transform"
            out["kind"] = "affine"
            out["affine"] = self.program_get_tform_dict(*feats)
            if return_none_if_tform_static:
                # Check if this tformation doesnt do anything
                if out["affine"]==self.FEATURES_DEFAULT:
                    return None
        elif kind=="repeat":
            # Then this line is an affine transform

            N = feats[0]
            feats = feats[1:]

            out["type"] = "repeat"
            out["kind"] = "default"
            out["affine"] = self.program_get_tform_dict(*feats)
            out["N"] = N
            if return_none_if_tform_static:
                # Check if this tformation doesnt do anything
                if out["affine"]==self.FEATURES_DEFAULT:
                    return None

        elif kind=="reflect":
            assert False, "note: this is not standard ordered affine features, see https://github.com/lucast4/dragmonkey/blob/master/MonkeyLogicCode/task/drag/utils/tasks/taskDatabase.m"

        elif kind=="arc":
            # Is a stroke, but arc has 2 additional params before features
            arclength = feats[0] # in radians.
            ccw = feats[1] # if do ccw
            feats = feats[2:]

            out["type"] = "stroke"
            out["kind"] = kind
            
            # what is affine transformation
            out["affine"] = self.program_get_tform_dict(*feats)

            # additional params for arc
            out["arclength"] = arclength
            out["ccw"] = ccw

        elif kind=="curve":
            # Is a stroke, but has 1 additional params before features
            ctrlpts = feats[0] # in radians.
            feats = feats[1:]

            out["type"] = "stroke"
            out["kind"] = kind
            
            # what is affine transformation
            out["affine"] = self.program_get_tform_dict(*feats)

            # additional params for arc
            out["ctrlpts"] = ctrlpts

        else:
            print(kind)
            print(feats)
            print(prims_list)
            assert False, "confirm that is correct assumption that the 1: items are features in order."
        return out

    def program_interpret_subprog(self, ind_subprog):
        """ Decides if a subprogram is a "standard" object, like an 
        object primitive.
        And what is the final transformation in space
        NOTE:
        Assumptions:
        - each subprogram is one and only one object.
        - if is N lines, first N-2 or N-1 define the object, and the last lines define the flexible tform.
        """

        self.program_extract()
        program = self.Program

        def _compare_to_primitive_templates(lines_good, fail_if_no_match=True):
            """ checks if this matches predefined templates for action primitives.
            INPUTS:
            - lines_good, 
            --- these are lines are the first N-1 lines of a subprogram - assuming that the last
            --- line is the affine transofmr. 
            --- These are line dicts. i..e, output of prog_interpret_line
            - fail_if_no_match, if False, then returns None if no match.
            RETURNS:
            - either a string defining the primitive {"line", "circle", ...} or None, which means did not
            match (or fails, see above).
            """

            assert isinstance(lines_good, list)

            if len(lines_good)==1:
                # Check if this is a simple prim
                L = lines_good[0]
                if L["type"]=="stroke" and L["affine"]==self.FEATURES_DEFAULT:
                    return L["kind"]
            elif len(lines_good)==2:
                if lines_good[0]["kind"]=="arc":
                    if lines_good[0]["affine"] == self.program_get_tform_dict() \
                        and lines_good[1]["type"]=="transform" and lines_good[1]["affine"]==self.program_get_tform_dict(y=0.25, sx=0.5, sy=0.5, order='trs'):
                        return "arc"

                if lines_good[0]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[0]["affine"], self.program_get_tform_dict(y=0.28867513)) and \
                    lines_good[1]["type"]=="repeat" and lines_good[1]["N"]==3 and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(th = 2.0943951, order="rst")):
                    return "triangle1"

                if lines_good[0]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[0]["affine"], self.program_get_tform_dict(y=0.5)) and \
                    lines_good[1]["type"]=="repeat" and lines_good[1]["N"]==4 and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(th = 1.57079633, order="rst")):
                    return "square1"

            elif len(lines_good)==3:
                from math import pi

                if lines_good[0]["kind"]=="line" and lines_good[0]["affine"]==self.program_get_tform_dict(x=0.5) and \
                    lines_good[1]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(y=0.5, th=1.57079633, order="str")) and \
                    lines_good[2]["type"]=="transform" and self.program_compare_tform_dicts(lines_good[2]["affine"], self.program_get_tform_dict(sx=0.5, sy=0.5, order="rst")):
                    return "L"

                if lines_good[0]["kind"]=="curve" and self.program_compare_tform_dicts(lines_good[0]["affine"], self.program_get_tform_dict()) and \
                    lines_good[1]["type"]=="transform" and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(order="tsr")) and \
                    lines_good[2]["type"]=="transform" and self.program_compare_tform_dicts(lines_good[2]["affine"], self.program_get_tform_dict(y=-0.5, sx=0.71428571, sy=0.71428571, order="str")):
                    return "squiggle1"

            elif len(lines_good)==4:
                if lines_good[0]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[0]["affine"], self.program_get_tform_dict(x=0.5)) and \
                    lines_good[1]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[1]["affine"], self.program_get_tform_dict(x=0.5, th=0.78539816, order="srt")) and \
                    lines_good[2]["kind"]=="line" and self.program_compare_tform_dicts(lines_good[2]["affine"], self.program_get_tform_dict(x=0.20710678, y=0.70710678, order="str")) and \
                    lines_good[3]["type"]=="transform" and self.program_compare_tform_dicts(lines_good[3]["affine"], self.program_get_tform_dict(x=-0.35355339, y=-0.35355339, order="rst")):
                    return "zigzag1"

            if fail_if_no_match:
                print("here")
                # print(lines_good)
                print(len(lines_good))
                [print(l) for l in lines_good]
                assert False, "failed to find match..."

        def _compose_transformations(lines_good):
            """ given multiple lines, extract final transformation, 
            INPUT:
            - lines_good, list of dicts
            RETURNS:
            - dict
            - 
            """
            assert isinstance(lines_good, list), "maybe you passed in dict?"
            if len(lines_good)==1:
                L = lines_good[0]
                return L["affine"]
            else:
                assert len(lines_good)==1, "not yet coded for >1 lines, need to compose them"
            

        # 1) get all lines
        nlines = len(program[ind_subprog])
        lines_good = []
        for i in range(nlines):
            line = self.program_get_line(ind_subprog, i)
            line_good = self.program_interpret_line(line)
            if line_good is not None:
                lines_good.append(line_good)

        # 2) Ask whether is object :
        # first N-1 lines combine to define the object
        # Last line defines transformation.
        if len(lines_good)==1:
            # move the affine to a new line
            lines_tform = [{}] 
            lines_tform[0]["type"] = "transform"
            lines_tform[0]["kind"] = "affine"
            lines_tform[0]["affine"] = lines_good[0]["affine"].copy()
            lines_obj = lines_good
            lines_obj[0]["affine"] = self.program_get_tform_dict() # use default (static)
        elif len(lines_good)==2:
            # then assume it's [obj, tform]
            lines_obj = lines_good[:1]
            lines_tform = lines_good[1:]
        elif lines_good[0]["kind"] in ["arc"] and lines_good[0]["type"]=="stroke": 
            # primitives are defined by 2 lines.
            lines_obj = lines_good[:2]
            lines_tform = lines_good[2:]
        elif len(lines_good)==4 and lines_good[0]["kind"]=="line" and lines_good[1]["kind"]=="line" and lines_good[2]["type"]=="transform":
            # primitives defined by 3 lines
            lines_obj = lines_good[:3]
            lines_tform = lines_good[3:]
        elif len(lines_good)==4 and lines_good[0]["kind"]=="curve":
            # could be squiggles.
            lines_obj = lines_good[:3]
            lines_tform = lines_good[3:]
        elif len(lines_good)==5 and self.Task["stage"] in ["_zigzag"]:
            lines_obj = lines_good[:4]
            lines_tform = lines_good[4:]
        elif len(lines_good)==3 and self.Task["stage"] in ["_triangle1", "_square"]:
            lines_obj = lines_good[:2]
            lines_tform = lines_good[2:]
        else:
            print(self.Task)
            [print(l) for l in lines_good]
            assert False, "coudl be either (1) obj needs multipel lines, like square, or (2) multiple lines of tform (e.g., global). solution: try both cases (where take last 1, 2, 3,,..) lines as tform. stop when get that first N-1 lines are a valid primitive"
            # assume last line is tform
            lines_obj = lines_good[:-1]
            lines_tform = lines_good[-1:]

        # print(len(lines_good))
        # print(lines_tform)
        # asfdsaf
        obj = _compare_to_primitive_templates(lines_obj)
        if len(lines_tform)>0:
            tform = _compose_transformations(lines_tform)
        else:
            tform = None
        
        # --- Return a dict
        out = {
            "obj":obj,
            "tform":tform}
        
        return out


######################################## OTHER STUFF
def task2chunklist(task):
    """
    """

    chunks = task["TaskNew"]["Task"]["chunks"]

    # convert dict
    chunks = [v for v in chunks.values()]
    models = chunks[::2]
    stroke_assignments = chunks[1::2]
#     stroke_assignments = [s for s in stroke_assignments]

    chunkdict = {m:[] for m in models} 
    for m, s in zip(models, stroke_assignments):
        chunkdict[m].append([[int(vv[0]-1) for vv in v] for _, v in s.items()]) # append, since a model may have multiple chunks
    return chunkdict


def convertTask2Strokes(task, concat_timesteps=False, interp=None, fake_timesteps=None):
    """ given one task object, converts to a strokes (list of nparary)
    - concat_timesteps, then added 0,1,2,3 ... as a third dimension, (T x 3 output)
    - splits into multiple strokes based on assumption that NDOTS (points in a stroke)
    - 7/16/20 - FIXED so that accurately splits into strokes, instead of asuming that NDOTS is 
    15. Now doesn't use NDOTS, instead uses onsets saved by matlab.
    """
    if False:
        # OLD: does not split into multiple strokes.
        strokes = np.concatenate((task["x_rescaled"], task["y_rescaled"]), axis=0).T
        if concat_timesteps:
            strokes = np.concatenate((strokes, np.arange(strokes.shape[0]).reshape(-1,1)), axis=1)
            strokes = [strokes]
        if not interp is None:
            from pythonlib.tools.stroketools import strokesInterpolate
            strokes = strokesInterpolate(strokes, N=interp)
        return strokes
    else:

        strokes_flat = np.concatenate((task["x_rescaled"], task["y_rescaled"]), axis=0).T
        strokes = []

        if False:
            # -- split by NDOTS
            NDOTS = 15 # TODO: replace this with TrialRecord.User.GENERAL... (now saving this param)
            assert(len(strokes_flat)%NDOTS==0), "assumption of 15 dots is wrong?"
            # OLD METHOD - using hard codede NDOTS
            edges = np.linspace(0, len(strokes_flat), len(strokes_flat)/NDOTS+1)
            for e1, e2 in zip(edges[:-1], edges[1:]):
            #     print([e1, e2])
                strokes.append(strokes_flat[int(e1):int(e2),:])
        else:
            # NEW MOETHOD - using "onsets", which is coded in MATLAB.
            onsets = task["onsets"] # e..g, [1, 16, ...]
            if len(onsets)==0:
                onsets = [1]
            onsets = [int(o-1) for o in onsets]
            onsets.append(len(strokes_flat))
            for o1, o2 in zip(onsets[:-1], onsets[1:]):
                # print(o1)
                # print(o2)
                # print(strokes_flat)
                strokes.append(strokes_flat[o1:o2])

        # -- append fake timesteps
        strokes = fakeTimesteps(strokes, [], "in_order")
         
        # -- interpolate
        if not interp is None:
            from pythonlib.tools.stroketools import strokesInterpolate
            strokes = strokesInterpolate(strokes, N=interp)       

        if fake_timesteps=="from_orig":
            orig = getTrialsFix(filedata, trial)["fixpos_pixels"]
            strokes = fakeTimesteps(strokes, point=orig, ver="from_point")
        elif not fake_timesteps is None:
            assert False, "dont know this one"
        else:
            strokes = fakeTimesteps(strokes, point=[], ver="in_order")


        return strokes

# def chunklist2parses(chunklist, strokes, model, default="eachstroke"):
#     """ for this chunklist and model, extract
#     each way to chunking strokes. e.g.
#     [[0,1], 2] leads to chuning of strokes 0 and 1
#     Returns one strokes object for each way of chunking.
#     (note, can do:
#     chunklist = getTrialsTaskChunks(...))
#     - NOTE: 3rd dim (time) might not make sense).
#     - parses_list is len of num chunks, each element a strokes.
#     - default, what to do if dont find chunk? 
#     --- eachstroke, then will treat each stroke as chunk,
#     """
#     if model not in chunklist.keys():
#         if default=="eachstroke":
#             # then each stroke is a chunk
#             chunks = [[[i] for i in range(len(strokes))]]
#     else:
#         chunks = chunklist[model]


#     parses_list = []
#     for c in chunks:
#         strokesnew = [np.concatenate([strokes[i] for i in s], axis=0) for s in c]
#         parses_list.append(strokesnew)

#     # === remove temporal inforamtion from parses (since innacuarte)
#     parses_list = [[strok[:,[0,1]] for strok in strokes] for strokes in parses_list]

#     return parses_list


# def task2parses(task, model):
#     """ given task and model, get parses_list,
#     which is list of strokes, each strokes a 
#     permutation of strokes consistent with chuking for 
#     model in drawmodel.analysis."""
#     from tools.tasks import task2chunklist, convertTask2Strokes
#     from pythonlib.tools.stroketools import getAllStrokeOrders
    

#     chunklist = task2chunklist(task)
#     strokestask = convertTask2Strokes(task)
#     parses_list = chunklist2parses(chunklist, strokestask, model)

#     # === get all permutations for parses (output will be list of strokes)
#     parses_allperms = []
#     for strokes in parses_list:
#         parses_allperms.extend(getAllStrokeOrders(strokes)[0])

#     # == convert to format model wants
#     parses = []
#     for strokes in parses_allperms:
#         orders = None
#         parses.append({
#             "strokes":strokes, # is not using extra memory.
#             "order":orders,
#             })
#     return parses


def chunks2parses(chunks, strokes, default="eachstroke"):
    """ for this chunks (list of chunks) and model, extract
    each way to chunking strokes. e.g.
    [[0,1], 2] leads to chuning of strokes 0 and 1
    Returns one strokes object for each way of chunking. 
    (note, can do:
    chunklist = getTrialsTaskChunks(...))
    - NOTE: 3rd dim (time) might not make sense).
    - parses_list is len of num chunks, each element a strokes.
    - default, what to do if dont find chunk? 
    --- eachstroke, then will treat each stroke as chunk,
    """
    parses_list = []
    for c in chunks:
        strokesnew = [np.concatenate([strokes[i] for i in s], axis=0) for s in c]
        parses_list.append(strokesnew)

    # === remove temporal inforamtion from parses (since innacuarte)
    parses_list = [[strok[:,[0,1]] for strok in strokes] for strokes in parses_list]

    return parses_list

def chunklist2chunks(chunklist, strokes, model, default="eachstroke"):
    """ for this chunklist and model, extract
    each way to chunking strokes. e.g.
    [[0,1], 2] leads to chuning of strokes 0 and 1
    Returns one strokes object for each way of chunking.
    - default, what to do if dont find chunk? 
    --- eachstroke, then will treat each stroke as chunk,
    """
    if model not in chunklist.keys():
        if default=="eachstroke":
            # then each stroke is a chunk
            chunks = [[[i] for i in range(len(strokes))]]
    else:
        chunks = chunklist[model]
    return chunks

def task2parses(task, model):
    """ given task and model, get parses_list,
    which is list of strokes, each strokes a 
    permutation of strokes consistent with chuking for 
    model in drawmodel.analysis.
    - model, either string, or a function: task--> chunks"""

    # from tools.tasks import task2chunklist, convertTask2Strokes
    from pythonlib.tools.stroketools import getAllStrokeOrders
    strokestask = convertTask2Strokes(task)
    
    if isinstance(model, str):
        chunklist = task2chunklist(task)
        chunks = chunklist2chunks(chunklist, strokestask, model)
    elif callable(model):
        # then model is a fucntion
        chunks = model(task)
    else:
        print(model)
        assert False, "not sure what is"
        
    parses_list = chunks2parses(chunks, strokestask)

    # === get all permutations for parses (output will be list of strokes)
    parses_allperms = []
    for strokes in parses_list:
        parses_allperms.extend(getAllStrokeOrders(strokes)[0])

    # == convert to format model wants
    parses = []
    for strokes in parses_allperms:
        orders = None
        parses.append({
            "strokes":strokes, # is not using extra memory.
            "order":orders,
            })
    return parses
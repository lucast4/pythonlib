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

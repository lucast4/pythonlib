""" for working with chunks.
Each task can be chunked in different ways. Each instance here is a single way of chunking. 
Can then input beh and so on to compute things wrt this chunking. 

Also can keep track of the labels of task stroke, et.c, Line circle, etc., and can then do things
like search for beh that matches a given motif - e.g, line-->circle. 
Can do the same for chunks, e.g., chunk1 --> chunk2.
"""

import numpy as np

class ChunksClassList(object):
    """ A single ChunksClass instance represents all ways of chunking this Task given an
    expt and rule."""

    def __init__(self, Task, expt=None, rule=None):
        """
        PARAMS:
        - Task, TaskClass instnace
        - expt, str, the experiment name.
        - rule, str, the rule name
        NOTE:
        - expt and rule are used for deciding what constitites a "chunk". Leave them as None if
        want to enter other method for deciding chunk
        """
        from .chunks import find_chunks_wrapper

        self.Task = Task
        self.Expt = expt
        self.Rule = rule

        # Extract all chunks, hierarhchies, etc
        list_chunks, list_hier, list_fixed_order = find_chunks_wrapper(Task, 
            expt, rule)

        # - Make one ChunksClass instance for each chunk:
        # list_shapes = Task.tokens_generate({"expt":expt}, track_order=False)
        list_shapes = [s for s in Task.Shapes]
        self.ListChunksClass = []
        for chunks, hier, fixedorder in zip(list_chunks, list_hier, list_fixed_order):
            C = ChunksClass(chunks, hier, fixedorder, task_stroke_labels=list_shapes)
            self.ListChunksClass.append(C)


    def remove_chunks_that_concat_strokes(self):
        """ Remove chunks from ListChunksClass that concatenates strokes.
        i.e, wherever C.Chunks has a chunk with multiple strokes
        e.g., [[1,2], [0]].
        RETURNS:
        - modifies self.ListChunksClass
        """

        def _does_concat(C):
            chunks = C.Chunks
            for c in chunks:
                if len(c)>1:
                    return True
            else:
                return False

        self.ListChunksClass = [C for C in self.ListChunksClass if not _does_concat(C)]

    def print_summary(self):
        """
        """
        for i, C in enumerate(self.ListChunksClass):
            print("--- Chunk num:", i)
            C.print_summary()



class ChunksClass(object):
    """ Represents a single way of chunking a specific task"""
    def __init__(self, chunks, hier, fixed_order, task_stroke_labels = None,
        assume_all_strokes_used=True):
        """
        PARAMS:
        - chunks, list of list, like [[1 2], [0 3], [5]], meaing chunk 0 uses strokes 1 and 2...
        Can also be [[1 2] 3].
        Can only be 2 levels deep max (i.e., [[1 2] 3])
        Meanning of chunks: use it to concatenate strokes.
        - hier, same type as chunks, but meaning is not to concat strokes, but to structure the ordering
        - fixed_order, dict, which indicate whether teh elements at a given level are aloowed to 
        be reordered, or remain fixed in order. e.g. {0: False, 1: [False, False, False]}, if 
        hier is [[1, 0], [3, 2], [5, 4]],
        means level0 is not fixed, and the items within the 3 chunks are also not fixed.
        - task_stroke_labels, list of ints or str, which are names to give to each task stroke. e.g..
        ["L", "C", ..] for line and circle.
        - assume_all_strokes_used, asserts that each task stroke id is used only once. 
        assumes that all ids are used, but cannot know for sure since not passing in how many
        strokes there were in the task.
        """

        self.Chunks = chunks
        self.Hier = hier
        self.FixedOrder = fixed_order
        self.Labels = task_stroke_labels

        self._preprocess()


    def _preprocess(self):
        """ run once
        """

        # 1) no ints, make sure all items are lists.
        for i, c in enumerate(self.Chunks):
            if isinstance(c, int):
                self.Chunks[i] = [c]    
        
        for i, c in enumerate(self.Hier):
            if isinstance(c, int):
                self.Hier[i] = [c]    

        # N task strokes
        chunks_flat = [c for C in self.Chunks for c in C]
        n_taskstrokes = len(chunks_flat)
        assert len(chunks_flat)==len(set(chunks_flat)), "you netered same stroke multipel times across chunks"

        # 2) Check that shapes are correct
        assert len(self.FixedOrder[1]) == len(self.Hier)
        if self.Labels is not None:
            assert len(self.Labels)==n_taskstrokes


    def _find_group_for_this_ind(self, ind, ver="chunks"):
        """ 
        Generic: Find either the chunk or hier for this ind. see params.
        PARAMS:
        - ind, int, 
        --- if ver=="chunks", then ind is the original stroke id before chunking.
        --- if ver=="hier", then ind is the chunk id after chunking.
        """
        
        if ver=="chunks":
            groups = self.Chunks
        elif ver=="hier":
            groups = self.Hier
        else:
            print(ver)
            assert False

        for i, c in enumerate(groups):
            if ind in c:
                return i
        assert False, f"did not find this ind in this chunks, {ind}, {chunks}"

    def find_hier_for_this_taskstroke(self, indstroke):
        """ Given this original task stroke ind, return its hier chunk. takes into 
        account chunking, then hierarchy.
        """

        indchunk = self._find_group_for_this_ind(indstroke)
        hierthis = self._find_group_for_this_ind(indchunk, "hier")
        return hierthis

    def find_hier_for_list_taskstroke(self, list_indstrokes):
        """ Given list of taskstroke inds, return the sequence of hieraarchical chunks.
        Useful e.g., if list_indstrokes is seuqnece of task inds drawn by subject
        PARAMS:
        - list_indstrokes, list of ints
        RETURNS:
        - list_hier, list of ints.
        """

        return [self.find_hier_for_this_taskstroke(ind) for ind in list_indstrokes]


    def find_hier_for_list_list_taskstroke(self, list_list_indstrokes, only_one_hier_allowed=False):
        """ Similar to find_hier_for_list_taskstroke, but assumes that each beh stroke
        can potentialyl get multiple task strokes. For each beh stroke, returns the list of 
        all hier that this beh stroke got. So this works on on entire trial.
        PARAMS:
        - list_list_indstrokes, list of list of ints. e..g, [[1 2], [0], [0, 1]] means first 
        beh stroke got task strokes 1 and 2, etc.
        - only_one_hier_allowed, then for cases where a single beh stroke matches two task hier chunks,
        then replaces that with np.nan. 
        RETURNS:
        if only_one_hier_allowed==False"
        - list_list_indhier, list of list of ints, e.g., [[1], [0], [0, 1]] mean sfirst beh stroke
        got hierchunk 1, etc. Note: the items within each inner list (e..g, [0,1]) will not be
        in order given by input strokes. e..g, [[1], [0, 2], [2], [0], [1]]
        if only_one_hier_allowed==True:
        - array of ints, where np.nan means this beh stroke did not get just a single hier.
        e.g., array([ 1., nan,  2.,  0.,  1.])
        """

        list_list_indhier =[]
        for list_indstrokes in list_list_indstrokes:
            
            this = self.find_hier_for_list_taskstroke(list_indstrokes)

            # clean up, if any ind hier repeated for the same beh stroke, then combine them into one int
            this = list(set(this))

            list_list_indhier.append(this)


        if only_one_hier_allowed:
            out = np.ndarray(len(list_list_indhier))
            for i, val in enumerate(list_list_indhier):
                if len(val)>1:
                    out[i] = np.nan
                else:
                    out[i] = val[0]
            return out
        else:
            return list_list_indhier

    ########################### GROUDNING/CONVERTING TO CONCRETE SEQUENCES
    def search_permutations_chunks(self, ver="hier", max_perms=1000 ):
        """
        Returns all permutatoins of chunks or hierarchy
        PARAMS:
        - ver, str, either "chunks", or "hier", which one to return permutations of.
        """
        from .chunks import search_permutations_chunks

        if ver=="chunks":
            assert False, "currently only works for Hier, since Fixed Order is only defined for that. can easlity make one for CHunks"
        elif ver=="hier":
            return search_permutations_chunks(self.Hier, self.FixedOrder, max_perms=max_perms)
        else:
            assert False




    ########################### WORKING WITH LABELS


    ############################
    def print_summary(self):
        """ quick printing of chunks, etc
        """

        print("Chunks:", self.Chunks)
        print("Hier:" , self.Hier)
        print("Fixed Order:", self.FixedOrder)
        print("list labels/shapes:", self.Labels)
        

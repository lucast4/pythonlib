"""
7/9/21 - Good for parsing directly from strokes (task image)
"""

import numpy as np
import matplotlib.pyplot as plt

class Parser(object):
    """
    """

    def __init__(self):
        """
        """
        self.Params = {}

    def input_data(self, data, kind="strokes"):
        """ 
        General-purpose for inputing a single datapoint. 
        INPUT:
        - data, flexible type, depending on kind.
        "strokes", then is list of np arrays.
        """

        self.Params["data_kind"] = kind

        if kind=="strokes":
            self.Strokes = data
            self.Strokes = [s[:,:2] for s in self.Strokes]

        self.Translation = {}
        self.Skeleton = None

    def parse_pipeline(self, N=1000, nwalk_det = 10, max_nstroke=400, max_nwalk=100, 
            plot=False, quick=False):
        """
        Full pipeline to get unique parses
        - N, how many. might not be exact since will remove reduntant parses
        """

        if quick:
            N=50
            nwalk_det = 5
            max_nstroke=60
            max_nwalk=20

        # 1) Recenter strokes to positive coordinates
        self.strokes_translate_positive()
        self.strokes_interpolate_fine(plot=plot)
        self.skeletonize_strokes()
        self.graph_construct()

        # Walk on graph
        # self.walk_extract_parses(10,50,50)
        self.walk_extract_parses(nwalk_det=nwalk_det, 
            max_nstroke=max_nstroke, max_nwalk=max_nwalk)

        # Remove redundant
        self.parses_remove_redundant()

        # How many permutations to get?
        nnow = len(self.Parses)
        neach = int(np.ceil(N/nnow))
        print("Got this many base parses:", nnow)
        print("Getting this many permutations of each base parses:", neach)

        ### get all permutations
        # self.parses_get_all_permutations(n_each = 2)
        self.parses_get_all_permutations(n_each = neach)

        ### Remove redundant permutations
        self.parses_remove_redundant(False, False)

        # Keep top K parses
        # TODO

        if plot:
            self.plot_skeleton()
            self.plot_graph()
            self.summarize_parses()


    def _translate(self, pts):
        """ apply saved transfomration to any pts,
        where pts is Nx2 np array, or (2,)
        """
        if not pts.shape==(2,):
            assert pts.shape[1]==2
            assert len(pts.shape)==2

        return pts + self.Translation["xy"]

    def strokes_translate_back(self, strokes):
        """ applies transformation back to original coordiantes,
        for tihs strokes
        """
        from pythonlib.tools.stroketools import translateStrokes
        return translateStrokes(self.Strokes, -self.Translation["xy"])


    def strokes_translate_positive(self):
        """ translate strokes so that all value are ositive. they can then act as 
        indices for constructing skeleton image
        RETURNS:
        - modifies self.Strokes
        - saves current state in self.Translation
        """
        from pythonlib.drawmodel.image import convStrokecoordToImagecoord, get_sketchpad_edges_from_strokes, getSketchpadEdges
        from pythonlib.tools.stroketools import translateStrokes

        if len(self.Translation)>0:
            if self.Translation["is_translated"]:
                assert False, "it is currently translated"

        # Recenter, so that all indices are positive
        edges = get_sketchpad_edges_from_strokes([self.Strokes])

        # translate so that x and y start at index k
        k = 4
        xy = [-edges[0,0] + k, -edges[0,1] + k]
        self.Strokes = translateStrokes(self.Strokes, xy)

        # sanity check
        edges_new = get_sketchpad_edges_from_strokes([self.Strokes])
        assert (edges_new>0).all(), "did not translate enough?"

        # save current state
        self.Translation["xy"] = np.array(xy)
        self.Translation["is_translated"] = True


    def strokes_interpolate_fine(self, plot=False):
        """ interpolate strokes so that does not skip any pixel - spatial interpolation
        RETURNS:
        - new field: self.StrokesInterp. does not modify self.Strokes
        """
        from pythonlib.tools.stroketools import strokesInterpolate2

        interval = 0.8
        self.StrokesInterp = strokesInterpolate2(self.Strokes, 
            N = ["interval", interval], base="space", plot_outcome=plot)


    def skeletonize_strokes(self):
        """ convert strokes to image, and skeletonize
        RETURNS:
        - puts ske into self.Skeleton
        """
        from skimage import morphology
        strokes_interp = self.StrokesInterp

        # Find good size of image that captures entire stroke.
        xlim = int(np.max(np.concatenate(strokes_interp)[:,0]))
        ylim = int(np.max(np.concatenate(strokes_interp)[:,1]))

        # Initiate image
        image = np.zeros((xlim+5, ylim+5), dtype=np.int16)
        tmp = np.concatenate(strokes_interp)
        for pt in tmp:
            xind = int(np.round(pt[0]))
            yind = int(np.round(pt[1]))
            image[xind, yind] = 1

        # Skeletonize the image
        ske = image
        # ske = morphology.remove_small_objects(ske)
        # ske = morphology.remove_small_holes(ske)
        ske = morphology.skeletonize(ske).astype(np.uint16)

        self.Skeleton = ske

    def graph_construct(self):
        """ make skeleton graph using self.Skeleton
        RETURNS:
        - self.Graph hold sthe networkx graph.
        """
        from sknw import build_sknw
        self.Graph = build_sknw(self.Skeleton, multi=True)


    ########## WALKING ON GRAPH
    def walk_extract_parses(self, nwalk_det = 5, max_nstroke=250, max_nwalk=100):
        """ Extract parses from skeleton graph
        NOTE:
        - uses code by Reuben Feinmam for walking on graph.
        """
        from pybpl.bottomup.initialize import RandomWalker

        # initialize random walker and empty parse list
        walker = RandomWalker(self.Graph, self.Skeleton)
        
        # Keep track of all parses
        walker_list = []
        parses_strokes = []
        parses_nodes = []

        # add deterministic minimum-angle walks
        for i in range(nwalk_det):
            parses_strokes.append(walker.det_walk())
            # walker_list.append(walker)
            parses_nodes.append(walker.list_ws)

        # sample random walks until we reach capacity
        nwalk = len(parses_strokes)
        nstroke = sum([len(parse) for parse in parses_strokes])
        ct = 0
        while nstroke < max_nstroke and nwalk < max_nwalk:
            if ct%10==0:
                print(ct, nstroke, nwalk, "target:", max_nstroke, max_nwalk)
            walk = walker.sample()
            parses_strokes.append(walk)
            parses_nodes.append(walker.list_ws)

            nwalk += 1
            nstroke += len(walk)
            ct+=1

        # save
        self.Parses = []
        for s, n in zip(parses_strokes, parses_nodes):
            self.Parses.append(
                {
                "strokes":s,
                "list_walkers": n,
                "permutation_of":None
                })
        self.ParsesParams = {
            "nwalk_det":nwalk_det,
            "max_nstroke":max_nstroke,
            "max_nwalk":max_nwalk
        }

        # Convert strokes to represnetation using ParserStrokes
        self._strokes_convert_to_parsestrokes()
            

    def _walkerstroke_to_parserstroke(self, ws):
        """ converts a single WalkerStroke instance to a 
        single ParserStroke instance"""
        from .parser_stroke import ParserStroke
        PS = ParserStroke()
        PS.input_data(ws.list_ni, ws.list_ei)
        
        return PS

    def _strokes_convert_to_parsestrokes(self):
        """ for each parse, represent strokes each as a ParserStroke object,
        instead of WalkerStroke (bpl) which is default.
        - Usually run this right after extract parses)
        """

        for i in range(len(self.Parses)):
            list_ws = self.Parses[i]["list_walkers"]
            list_ps = [self._walkerstroke_to_parserstroke(w) for w in list_ws]
            self.Parses[i]["list_ps"] = list_ps
            del self.Parses[i]["list_walkers"]



    ################### DO THINGS WITH PARSES [INDIV]
    def parses_to_strokes(self, ind):
        """ extract strokes for this parse
        NOTE: automaticlaly check consistency, since this uses
        list_ni and list_ei, but other things here usual;yl use edges dir
        """
        return self._parses_to_walker(ind).S


    def _parses_to_walker(self, ind):
        """ returns RandomWalker object for this parse (ind)
        """
        from pybpl.bottomup.initialize.random_walker import RandomWalker
        R = RandomWalker(self.Graph, self.Skeleton)

        if False:
            R.list_ws = self.Parses[ind]["list_walkers"]
        else:
            # check consistnetcy
            [w._check_lists_match() for w in self.Parses[ind]["list_ps"]]
            # good.
            R.list_ws = [w for w in self.Parses[ind]["list_ps"]]
        return R





    ################### DO THINGS WITH PARSES [BATCH]
    def score_parses(self):
        """ quick and dirty to score parses, then can extract the top K parses.
        """
        assert False, "not done, probably just ignore this."
        from pythonlib.drawmodel.parsing import score_function
        origin = D.Dat["origin"][0]
        origin = P._translate(origin)
        score_ver = "travel_from_orig" # better, since differentiates 2 tasks thjat are just flipped (and so will not throw one of them out)
        score_norm = "negative"
        # score_fn = lambda parses: score_function(parses, ver=score_ver, 
        #                                          normalization=score_norm, use_torch=True, origin=origin)

        parses = self.extract_all_parses_as_list()
        scores = score_function(parses, ver=score_ver, normalization=score_norm, use_torch=False, origin=origin)


    def parses_fill_in_all_strokes(self):
        """ if a parse is missing strokes, then converts to strokes and saves
        in self.Parses[ind]["strokes"]
        - is smart about not runing if strokes already exists.
        """
        for i in range(len(self.Parses)):
            if self.Parses[i]["strokes"] is None:
                self.Parses[i]["strokes"] = self.parses_to_strokes(i)


    def parses_remove_redundant(self, direction_invariant=True,
        strokes_invariant=True):
        """ Finds parses that are the same and removes.
        by default, Same is defined as invariant to (1) stroke order,
        (2) within stroke direction and (3) for circle, circle permutation
        OUTPUT:
        - modifies self.Parses
        """

        # Find matches, return as dict[ind1] = [list of inds idential]
        # ind1 is not present unless list is len >0
        matches = {}
        nparses = len(self.Parses)
        for i in range(nparses):
            for ii in range(i+1, nparses):

                # compare parses
                # p1 = [p.list_ni for p in parses_tracker[i]]
                # p2 = [p.list_ni for p in parses_tracker[ii]]
                
                if self.check_parses_is_identical(i, ii, direction_invariant=direction_invariant,
                    strokes_invariant=strokes_invariant):
                    if i in matches:
                        matches[i].append(ii)
                    else:
                        matches[i] = [ii]
        #             matches.append((i,ii))
                    if False:
                        print("identical:")
                        print(p1)
                        print(p2)
        #         else:
        #             print("diff")
        #             print(p1)
        #             print(p2)
                    
        # throw out all parses identical
        # i.e., only keep the keys, which are the first instance found (unique).
        inds_to_remove = set([mm for m in matches.values() for mm in m])

        print("removing these inds", inds_to_remove)
        print("number", len(inds_to_remove))

        print("startring len of parses", len(self.Parses))
        self.Parses = [p for i, p in enumerate(self.Parses) if i not in inds_to_remove]
        print("ending len of parses", len(self.Parses))

    def parses_get_all_permutations(self, n_each = 5):
        """ gets all permutations for each parse. 
        Then appends them to the parses list.
        OUTPUT:
        """

        for i in range(len(self.Parses)):
            if i%5==0:
                print(i)
            parses_list = self.get_all_permutations(i, n=n_each)

            # add these to ends of self.Parses
            for p in parses_list:
                self.Parses.append(
                    {
                    "strokes":None,
                    "list_ps": p,
                    "permutation_of": i
                    })



        
    ########## GETTING PERMUTATIONS
    def get_all_permutations(self, ind, n=100, ver="parser"):
        """ get all permutations for this parse (ind)
        By default, get reordering, flipping.
        TODO: circular permtuation for any loops.
        NOTE: does on list of nodes (instead of strokes) but is identical, 
        since can convert to strokes after.
        INPUT:
        - ind, 
        - n, how many parses to keep. note that will search thru many (trials_per), picking
        out random (or select, if use scorefun). 
        OUT:
        - list of parses, each parses is a list of list.
        """
        # import sys
        # sys.path.append("/data1/code/python/GNS-Modeling/")
        # from gns.inference.parsing.top_k import search_parse
        from .search import search_parse

        if ver=="walker":
            assert False, "old - now converting WalkerStrokes to ParserStrokes"
            parse = self.Parses[ind]["list_walkers"]
            parse = [{"nodes":p.list_ni, "edges":p.list_ei, "flipped":False} for p in parse]
            assert False, "doesnt seem to be flipping..."
            assert False, "make a new class, just like walker class."
        elif ver=="parser":
            # each stroke is a ParserStroke
            parse = self.Parses[ind]["list_ps"]
        else:
            assert False, "need to fix below, cannot use nodelists, since doesnt keep track of the edge"
            # Instead, use edgelists. then convert from edge to nodes.
            # even better, just use codes for each stroke, e.g, [[]]
            # even better, just use the walk object. then give it a method to track whether flipped or not.

            parse = self.parses_extract_nodelists(ind)
            parse = self.parses_extract_edgelists(ind)

            # Uinsg strokes.
            # parses_out, _ = search_parse(parse, lambda parses: torch.rand(len(parses)))
            # parses_out = [[p.numpy() for p in parse] for parse in parses_out]

        parses_list, _ = search_parse(parse, configs_per=n, trials_per=800, max_configs=1e6)

        return parses_list


    ########## UTIls
    def extract_nodelists(self, ind):
        """ for this ind, extract list of nodes, 
        which is list of list, analogous to strokes, but using sequence of 
        nodes.
        INPUT:
        - ind, integer
        OUT:
        - list of list
        """
        return [p.list_ni for p in self.Parses[ind]["list_walkers"]]


    def extract_edgelists(self, ind):
        """ see above
        NOTE:
        - these do not influence the order of the final stroke, since final stroke will always
        make it so the edges emanate from the nodes.
        i.e., (1,2,0) does not necessarily mean stroke from node 1 to 2. 
        """
        return [p.list_ei for p in self.Parses[ind]["list_walkers"]]
        

    def extract_parses_wrapper(self, ind, kind="parser_stroke_class"):
        """ helper to extract list of strokes, represented in flexible waus
        OUT:
        - list of strokes.
        """

        if kind == "parser_stroke_class":
            key = "list_ps"
        else:
            assert False, "not coded"
        return [p for p in self.Parses[ind][key]]


    def extract_all_parses_as_list(self, kind="strokes"):
        """ wrapper to get list of parses
        kind, 
        "strokes"
        {"parser_stroke_class"}
        """
        if kind=="strokes":
            self.parses_fill_in_all_strokes()
            return [p["strokes"] for p in self.Parses]
        else:
            return [self.extract_parses_wrapper(i, kind) for i in range(len(self.Parses))]


    def check_parses_is_identical(self, ind1, ind2, direction_invariant=True,
        strokes_invariant=True):
        """
        returns true if parses are identical (with certain invariances, see below)
        INPUT:
        - ind1 and ind2 index tinto self.Parses (e.g., self.Parses[1])
        - direction_invariant, then order of storkes doenst matter
        - strokes_invariant, then the direction within each stroke doesnt matter.
        OUT: 
        - bool

        -----------------------
        list1 = [p.list_ni for p in self.Parses[i]["list_walkers"]]

        list1 and list2 are lists of lists, where the inner lists of integers.,
        returns true if lists 1 and 2 are eequal, allowing for the items in each inner lists to
        flipped either way, and for the inner lists to be in any order.
        - if think of these as parses, indicating sequence of nodes, then this detects if have
        identical parese (ignoring direction of each stroke, and order 
        e..g,:
        if a = [[3,2,1,0], [2,1,2]], then if 
        b = [[3,2,1,0], [2,1,2]] --> True
        b = [[2,1,2], [3,2,1,0]] --> True
        b = [[2,1,2], [0,1,2,3]] --> True
        b = [[2,1,2], [0,1,3,2]] --> False
        
        NOTE: also is invariant ot circular permutation. so [4,5,6,4] is smae as [5,6,4,5]. 
        e.g,:
        if a is [[3, 4, 7, 3], [7, 2, 4, 7]] and
        b = [[3, 7, 4, 3], [4, 2, 7, 4]], then these are equal.    
        """

        list1 = self.extract_parses_wrapper(ind1)
        list2 = self.extract_parses_wrapper(ind2)

        # for each stroke, find its hash
        list1 = [p.unique_path_id(invariant=strokes_invariant) for p in list1]
        list2 = [p.unique_path_id(invariant=strokes_invariant) for p in list2]

        # since dont care about direction, sort each list (of tuples of ints)
        if direction_invariant:
            list1 = sorted(list1)
            list2 = sorted(list2)
        
        # check if they are identical
        return list1==list2


    def _OLD_parses_is_identical_ignoredir_OLD(self, ind1, ind2):
        """
        returns true if parses are identical (with certain invariances, see below)
        INPUT:
        - ind1 and ind2 index tinto self.Parses (e.g., self.Parses[1])
        OUT: 
        - bool
        NOTE:
        - OLD, beucase doesnt take into accound the edge id (fails for multigraphs).

        -----------------------
        list1 = [p.list_ni for p in self.Parses[i]["list_walkers"]]

        list1 and list2 are lists of lists, where the inner lists of integers.,
        returns true if lists 1 and 2 are eequal, allowing for the items in each inner lists to
        flipped either way, and for the inner lists to be in any order.
        - if think of these as parses, indicating sequence of nodes, then this detects if have
        identical parese (ignoring direction of each stroke, and order 
        e..g,:
        if a = [[3,2,1,0], [2,1,2]], then if 
        b = [[3,2,1,0], [2,1,2]] --> True
        b = [[2,1,2], [3,2,1,0]] --> True
        b = [[2,1,2], [0,1,2,3]] --> True
        b = [[2,1,2], [0,1,3,2]] --> False
        
        NOTE: also is invariant ot circular permutation. so [4,5,6,4] is smae as [5,6,4,5]. 
        e.g,:
        if a is [[3, 4, 7, 3], [7, 2, 4, 7]] and
        b = [[3, 7, 4, 3], [4, 2, 7, 4]], then these are equal.    
        """
        assert False, 'use new ver'
        def _hash_linear(li):
            """ make the inner list in increasing order
            this ensures that sorting the outer list always leads to same
            result, assuming the inner lists are the same (modulo flipping)
            - hash, since returns list that ids a set of equivalent lists.
            """
            if li[-1]>li[0]:
                # then dont flip
                return li
            elif li[-1]<li[0]:
                return li[::-1]
            else:
                # they are equal
                return li
        
        def _hash_circular(l):
            """ given list of nodes, where first and list are identical, so is
            a loop, returns a unique list of nodes that identifies this loop.
            invariant to order and direction.
            e.g.,: 
            [4, 2, 7, 4] --> (2, 4, 7)
            [7, 2, 4, 7] --> (2, 4, 7)
            """

            # remove endpoint
            l = l[:-1]

            # for each direction, get all circular permutations
            l_rev = l[::-1]

            tmp = []
            for lthis in [l, l_rev]:
                from more_itertools import circular_shifts
                tmp.extend([list(x) for x in circular_shifts(lthis)])
            return sorted(tmp)[0]
        
        def _hash(l):
            if l[0]==l[-1]:
                return _hash_circular(l)
            else:
                return _hash_linear(l)

        assert False, "problem: doesnt take into account multiple paths between nodes"
        # e.g., sequence of nodes [2, 7, 4, 2] is paired with edges sequence: [(2, 7, 0), (7, 4, 1), (4, 2, 0)]
        # solution: represent sequence of nodes as something like [2, a, 7, b, 4, a, 2].
        # altearnive solution: any operation on stroke involves both nodelist and edgelist.

        
        list1 = self.parses_extract_nodelists(ind1)
        list2 = self.parses_extract_nodelists(ind2)

        # first, for each inner list, flip so that is incresaing.
        list1 = [_hash(x) for x in list1]
        list2 = [_hash(x) for x in list2]
        
        # second, sort each list
        list1 = sorted(list1)
        list2 = sorted(list2)
        
        # check if they are identical
        return list1==list2


    ########## PRINTING
    def print_parse_info(self, ind):

        for p in self.Parses[ind]["list_ps"]:
            p.print_elements(verbose=False)



    ########## PLOTTING
    def plot_skeleton(self):
        """ plots the bw image
        Note: the orientation might differ from the strokes. This is ok, since 
        once convert back to strokes, will fix the orientation
        """
        if self.Skeleton is not None:
            # plt.figure()
            # plt.imshow(image, cmap="gray")
            # plt.title("before skel")
            plt.figure()
            plt.imshow(self.Skeleton, cmap='gray')
            plt.title("after skel")
        else:
            print("need to first make skeleton")

    def plot_graph(self, graph=None):
        """ plots skeleton graph overlaid on skeleton
        """

        def _plot_graph(ax, nodecol="w"):
            # draw edges by pts
            for (s,e, i) in graph.edges:
                ps = graph.edges[(s, e, i)]['pts']
                ax.plot(ps[:,1], ps[:,0], "--")
                
            # draw node by o
            nodes = graph.nodes()
            ps = np.array([nodes[i]['o'] for i in nodes])
            ax.plot(ps[:,1], ps[:,0], 'o', color=nodecol)
            for i in nodes:
                pt = nodes[i]["o"]
                ax.text(pt[1], pt[0], i, color="r", size=15)

        if graph is None:
            graph = self.Graph
        
        # draw image
        fig, axes = plt.subplots(1,2, figsize=(25,8))

        # 1) Overlay on image
        ax = axes[0]
        ax.imshow(self.Skeleton, cmap='gray')
        _plot_graph(ax)

        # 2) graph alone
        ax = axes[1]
        # ax.imshow(self.Skeleton, cmap='gray')
        _plot_graph(ax, "k")






    def plot_parses(self):
        pass


    def summarize_parses(self, nmax = 50):
        from pythonlib.drawmodel.parsing import summarizeParses
        import random
        parses = self.extract_all_parses_as_list()
        if len(parses)>nmax:
            parses = random.sample(parses, nmax)
        summarizeParses(parses, plot_timecourse=True, ignore_set_axlim=True)

# from pythonlib.drawmodel.parsing import plotParses, summarizeParses

# # == PLOT PARSES (my plotting fucntion, which shows tiecourse as well
# # === 2) Given Task, general purpose parsing function
# from pythonlib.tools.stroketools import fakeTimesteps
# from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse

# plotParses(parses)





################ SCRATCH

# Testing the code for comparing two parses (see if identical)
# See by eye whether they are correct.
# dinv = True
# sinv = True
# for i in range(len(P.Parses)):
#     for ii in range(i+1, len(P.Parses)):
#         if P.parses_is_identical(i, ii, direction_invariant=dinv, strokes_invariant=sinv):
#             print("FOUND:")
#             print(i)
#             P.print_parse_info(i)
#             print(ii)
#             P.print_parse_info(ii)

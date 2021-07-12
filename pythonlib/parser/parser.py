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
        self.Finalized=False # to ensure that after process strokes, dont try to do more stuff.


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

    # def graph_and_parse_pipeline(self, N=1000, nwalk_det = 10, max_nstroke=400, max_nwalk=100, 
    #         plot=False, quick=False, graph_mods = None, graph_mods_params = None,
    #         direction_invariant=False, strokes_invariant=False):

    #     self.make_graph_pipeline(plot=plot)
    #     self.parse_pipeline(N=1000, nwalk_det = 10, max_nstroke=400, max_nwalk=100, 
    #         plot=False, quick=False, graph_mods = None, graph_mods_params = None,
    #         direction_invariant=False, strokes_invariant=False)

    def make_graph_pipeline(self, plot=False, graph_mods = None, graph_mods_params = None):

        # 1) Recenter strokes to positive coordinates
        self.strokes_translate_positive()
        self.strokes_interpolate_fine(plot=plot)
        self.skeletonize_strokes()
        self.graph_construct()

        # Graph mods
        for mod, params in zip(graph_mods, graph_mods_params):
            if mod=="merge":
                self.graphmod_merge_nodes_auto(**params)
            elif mod=="splitclose":
                self.graphmod_split_edges_auto(**params)
            elif mod=="strokes_ends":
                # endpoints for each stroke should match a node.
                self.graphmod_add_nodes_strokes_endpoints(**params)
            else:
                assert False

        if plot:
            self.plot_skeleton()
            self.plot_graph()


    def parse_pipeline(self, N=1000, nwalk_det = 10, max_nstroke=400, max_nwalk=100, 
            plot=False, quick=False, direction_invariant=False, strokes_invariant=False):
        """
        Full pipeline to get unique parses
        - N, how many. might not be exact since will remove reduntant parses
        - graph_mods, list of mods (strings) in order of application.
        - graph_mod_params, list of dicts, which are kwargs for each mod opertaion./
        """

        # self.make_graph_pipeline(plot=plot)

        # 1) Recenter strokes to positive coordinates
        # self.strokes_translate_positive()
        # self.strokes_interpolate_fine(plot=plot)
        # self.skeletonize_strokes()
        # self.graph_construct()
        
        if quick:
            N=50
            nwalk_det = 5
            max_nstroke=60
            max_nwalk=20

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
        self.parses_remove_redundant(direction_invariant=direction_invariant, 
            strokes_invariant=strokes_invariant)

        # Cleanup parses

        # Keep top K parses
        # TODO

        if plot:
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
        return translateStrokes(strokes, -self.Translation["xy"])

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




    ################# TOOLS, WROKING WOTH PARSERSTROKE
    def _walkerstroke_to_parserstroke(self, ws):
        """ converts a single WalkerStroke instance to a 
        single ParserStroke instance"""
        from .parser_stroke import ParserStroke
        PS = ParserStroke()
        PS.input_data(ws.list_ni, ws.list_ei)
        return PS

    def _listofpaths_to_parserstroke(self, path):
        """
        help convert list of paths to a single stroke
        - path, list of edges (tuples)
        """
        from .parser_stroke import ParserStroke
        PS = ParserStroke()
        PS.input_data_directed(path)
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


    ########## WALKING ON GRAPH (GET PARSES)
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
            
    def _check_edges_identical(self, edge1, edge2):
        if edge1[2]==edge2[2] and set(edge1[:2])==set(edge2[:2]):
            return True
        else:
            return False

    def manually_input_parse(self, list_of_paths, use_all_edges=True):
        """
        Manually enter a new parse by providing list of paths, where
        each path is a list of directed edges.
        e.g., 
        - list_of_paths = [
        [(0,1,0), (1,2,0)],
        [(2,3,0)]
        ]
        gives two paths (strokes).
        NOTE:
        - will check for correctness of each path
        - will check that all edges are used up, unless override.
        """

        if use_all_edges:
            # get set of all edges
            set_of_edges = set([edge for path in list_of_paths for edge in path])
            for ed in self.Graph.edges:
                if not any([self._check_edges_identical(ed, ed1) for ed1 in set_of_edges]):
                    print("edges inputed", set_of_edges)
                    print("edges in graph", self.Graph.edges)
                    assert False, "did not use all edges"

        list_ps = [self._listofpaths_to_parserstroke(path) for path in list_of_paths]


        newparse = {
            "strokes":None,
            "list_ps":list_ps,
            "permutation_of":None,
            "manual":True
        }

        self.Parses.append(newparse)
        print("Added new parse, ind:", len(self.Parses)-1)


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


    ########## GRAPH UTIls
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


    ########## SPLIT/MERGE OPERATIONS
    def find_closest_nodes_under_thresh(self, thresh):

        def _find_closest_nodes_under_thresh(thresh, G):
            """ find pairs of nodes that are closer than thresh
            OUT:
            - None if dosent find.
            - or the closest pair as a set, e..g, {2,3}
            """
            pairs = []
            dists = []
            for n in G.nodes:
                for nn in G.nodes:
                    if nn>n:
                        p1 = G.nodes[n]["o"]
                        p2 = G.nodes[nn]["o"]
                        d = np.linalg.norm(p1-p2)
                        if np.linalg.norm(p1-p2)<thresh:
                            pairs.append({n, nn})
                            dists.append(d)
            if len(pairs)==0:
                return None
            else:
                # take the closest pair
                li = [(a, b) for a,b in zip(pairs, dists)]
                li = sorted(li, key = lambda x: x[1])
                pairthis = li[0][0] # take the closest pair
                return pairthis
        
        return _find_closest_nodes_under_thresh(thresh, self.Graph)



    def merge_nodes(self, nodes):

        def _merge_nodes(nodes, G):
            """ 
            - nodes, set of nodes, e..g, {3,4}
            - G
            OUT:
            - modifies G to have new node (ind is max + 1), and new edges replacing old edges.
            """
            assert isinstance(nodes, set)
            pairthis = nodes
           
            # new center
            onew = np.round(np.mean(np.stack([G.nodes[n]["o"] for n in pairthis]), axis=0)).astype(np.int16)
            nnew = max(list(G.nodes))+1

            # remove all edges between these
            elist = [e for e in G.edges if set(e[:2])==pairthis]

            # add new edges to replace any old edges
            edges_to_add = []
            edges_to_remove = []
            edges_to_remove.extend(elist)
            for ni in [0, 1]:
                nthis = list(pairthis)[ni]
                othis = G.nodes[nthis]["o"]

                for e in G.edges:
                    if nthis in e:
                        edg = G.edges[e]

                        # conver this edge to a new edge, using same pts
                        nother = [n for n in e[:2] if n!=nthis][0]

                        if set([nthis, nother])==pairthis:
                            # this is edge between the pair you are merging.
                            continue

                        # construct a new edge
        #                 weight = edg["weight"]
        #                 visited = edg["visited"]
                        pts = edg["pts"]

                        # add the center to the end that is currently closest to the center
                        d1 = np.linalg.norm(pts[0,:]-othis)
                        d2 = np.linalg.norm(pts[-1,:]-othis)
                        if d1<d2:
                            pts = np.insert(pts, 0, onew.reshape(1,-1), axis=0)
                        else:
                            pts = np.append(pts, onew.reshape(1,-1), axis=0)

                        # Ideally remove the close pts, and replace with interpolated pts.
                        # Or snap onto the strokes.

                        # Add this new edge
                #         G.add_edge(nnew, nother, pts=pts, weight=weight)
                        edges_to_add.append((nnew, nother, {"pts":pts}))
        #                 edges_to_add.append((nnew, nother, {"pts":pts, "weight":weight}))
                        edges_to_remove.append((nthis, nother, e[2]))
                        
            nodes_to_add = [(nnew, {"o":onew, "pts":onew.reshape(1,-1)})]
            nodes_to_remove = list(pairthis)
            self.modify_graph(nodes_to_add=nodes_to_add, nodes_to_remove=nodes_to_remove, 
                        edges_to_add=edges_to_add, edges_to_remove=edges_to_remove)
            
            return G

        return _merge_nodes(nodes, self.Graph)


    def graphmod_merge_nodes_auto(self, thresh=50):
        """ automaticalyl merge airs of nodes that are closert than thershold until no
        such pairs remain
        """

        pairthis = self.find_closest_nodes_under_thresh(thresh=thresh)
        print("Merging this pair of nodes: ", pairthis)         
        while pairthis is not None:
            # Merge each pair
            # Make sure that if a node is part of multiple pairs, should do them separately, unless
            # they are all close together.
            self.merge_nodes(pairthis)

            pairthis = self.find_closest_nodes_under_thresh(thresh=thresh)
            if pairthis is None:
                print("No more pairs to merge - done!")
            else:
                print("Merging this pair of nodes: ", pairthis)         


    def graphmod_add_nodes_strokes_endpoints(self, strokes, thresh=10):
        """ add nodes for each endpoint for each stroke (or do nothing
        if it already exists)
        """
        for strok in strokes:
            for ind in [0, -1]:
                pt = strok[ind, :2]
                # check if this strokes endpoints both have nodes assigned
                list_ni = self.find_close_nodes_to_pt(pt, thresh=thresh)
                
                if len(list_ni)==0:
                    # then add a node
                    edges, dists, inds= self.find_close_edge_to_pt(pt, thresh=thresh)
                elif len(list_ni)>1:
                    assert False, "first do merge"
                else:
                    # fine, already has a single node.
                    continue

                # Split graph at each of the extracted edges
                for ed, ind in zip(edges, inds):
                    self.split_edge(ed, ind)

    def split_edge(self, edge, ind):
        def _split_edge(edge, ind, G):
            """ split this edge at this index along its pts
            """

            # First, make the new node
            # pt for the new node
            print("split edge start")
            print(edge)
            print(G.edges)
            onew = G.edges[edge]["pts"][ind]   
            
            # figure out orientation of pts
            pts = G.edges[edge]["pts"]
            n1 = edge[0]
            n2 = edge[1]
            o1 = G.nodes[n1]["o"]
            o2 = G.nodes[n2]["o"]
            if self.check_pts_orientation(pts, o1):
                # then pts goes from n1 to n2.
                assert self.check_pts_orientation(pts, o2)==False
            else:
                # n1 is always the one closest to pts[0]
                assert self.check_pts_orientation(pts, o2)==True
                n1 = edge[1]
                n2 = edge[0]

            # New pts
            nnew = max(G.nodes)+1
            onew = pts[ind,:]
            # print("Added node: ", nnew)
            # G.add_node(nnew, o=onew, pts=onew.reshape(1,-1))
            self.modify_graph(nodes_to_add = [(nnew, {"o":onew, "pts":onew.reshape(1,-1)})])
            
            # New edges
            edges_to_add = [
                (n1, nnew, {"pts":pts[:ind+1,:]}),
                (nnew, n2, {"pts":pts[ind:,:]})]
            
            # Remove edge
            edges_to_remove = [edge]
            
            self.modify_graph(edges_to_add=edges_to_add, edges_to_remove=edges_to_remove)
            
            return nnew

        return _split_edge(edge, ind, self.Graph)
        

    def find_closest_edge_to_split(self, thresh):
        """ find any pts that are close to an edge (that the pt is not on)
        return that edge, and the location on that edge that is closest to the pt
        """

        G = self.Graph
        edges = []
        dists = []
        inds = []
        nodes = []
        for n in G.nodes:
            for e in G.edges:
                if n not in e[:2]:
                    d, ind = self.min_dist_node_to_edge(n, e)
                    if d<thresh:
                        edges.append(e)
                        dists.append(d)
                        inds.append(ind)
                        nodes.append(n)
        if len(edges)==0:
            return None, None, None, None
        else:
            # get the edge that has the case of closest node.
            tmp = [(e, d, i, n) for e, d, i, n in  zip(edges, dists, inds, nodes)]
            tmp = sorted(tmp, key=lambda x:x[1])
            return tmp[0]


    def graphmod_split_edges_auto(self, thresh=25):
        """ automatilcaly split edgfes that re close to a pt, at
        the location on edge closest to the pt.
        """

        ct = 0
        e, d, i, n = self.find_closest_edge_to_split(thresh)
        while e is not None:
            
            if ct>20:
                print(G.nodes)
                print(G.edges)
                assert False, "recursive problem."
            print("**Splitting edge", e, "Merging with node:", n)
            nnew = self.split_edge(e, i)
            
            # Must then merge these two pts.
            print("Merging: ", {n, nnew})
            self.merge_nodes({n, nnew})

            e, d, i, n = self.find_closest_edge_to_split(thresh)



    # GRAPHMOD - tools
    def check_pts_orientation(self, pts, o, return_none_on_tie=False):
        """ figures out how pts is oriented relative to 
        pt1. returns True is pts[0] is closer, or False otherwise
        - pt1, Nx2 array
        - o, (2,) array
        - return_none_on_tie, otherwise fails.
        """
        
        # add the center to the end that is currently closest to the center
        d1 = np.linalg.norm(pts[0,:]-o)
        d2 = np.linalg.norm(pts[-1,:]-o)
        if d1==d2:
            if return_none_on_tie:
                return None
            else:
                assert False
        if d1<d2:
            return True
        else:
            return False

    def modify_graph(self, nodes_to_add=None, nodes_to_remove=None, edges_to_add=None, 
        edges_to_remove=None):
        
        G = self.Graph
        
        print("Graph before modifiyong:")
        print(G.nodes)
        print(G.edges)
        
        # Removing nodes
        if nodes_to_add is not None:
            print("Adding new nodes:", [[e for e in ed if not isinstance(e, dict)] for ed in nodes_to_add])
        #     G.add_node(nnew, o=onew, pts=onew.reshape(1,-1))
            G.add_nodes_from(nodes_to_add)
            
        if nodes_to_remove is not None:
            print("Removing old nodes:", nodes_to_remove)
            G.remove_nodes_from(nodes_to_remove)
            
        if edges_to_remove is not None:
            print("Removing edges:", edges_to_remove)
            G.remove_edges_from(edges_to_remove)

        if edges_to_add is not None:
            # Remove edges
            print("Adding edges", [[e for e in ed if not isinstance(e, dict)] for ed in edges_to_add])
        #     print(", which replace these removed edges", edges_to_remove)
            G.add_edges_from(edges_to_add)

        print("- Done modifying, new nodes and edges")
        print(G.nodes)
        print(G.edges)
        

    ##########
    # for each node, check how close it is to every other path
    def min_dist_node_to_edge(self, node, edge):
        """ returns min distance from this node to any pt along
        this path
        - node, int
        - edge, (n1, n2, key)
        OUT:
        - d, ind,
        where ind is index in pts (for edge) of min
        """
        assert node not in edge[:2], "must compare node to other edges"
        G = self.Graph
        o = G.nodes[node]["o"]
        pts = G.edges[edge]["pts"]
        
        dists = np.linalg.norm(pts-o, axis=1)
        ind = np.argmin(dists)
        return dists[ind], ind

    def min_dist_pt_to_edge(self, pt, edge):
        """
        Retunrs
        - the distance
        - the ind along edge that has this distance.
        """
        G = self.Graph
        pts = G.edges[edge]["pts"]
        
        dists = np.linalg.norm(pts-pt, axis=1)
        ind = np.argmin(dists)
        return dists[ind], ind


    def find_close_edge_to_pt(self, pt, thresh=10):
        """ Find list of edges which are close to pt.
        Also return the ind along the edge.
        RETURNS:
        - edges, list of edges
        - dists, 
        - inds, 
        (All lists)
        """
        G = self.Graph
        edges = []
        dists = []
        inds = []

        for e in G.edges:
            d, ind = self.min_dist_pt_to_edge(pt, e)
            if d<thresh:
                edges.append(e)
                dists.append(d)
                inds.append(ind)

        return edges, dists, inds



    def find_close_nodes_to_pt(self, pt, thresh=10):
        """
        OUT:
        - list_nodes, list of ints. empty if none.
        """

        list_nodes = []
        for n in self.Graph.nodes:
            o = self.Graph.nodes[n]["o"]
            if np.linalg.norm(pt-o)<thresh:
                list_nodes.append(n)
        return list_nodes

    # def map_pt_to_node(self, pt, thresh=15):
    #     """ 
    #     return node close to this pt.
    #     return None if none.
    #     if >1, then fails.
    #     """
    #     list_nodes = self.find_close_nodes_to_pt(pt, thresh)

    #     if len(list_nodes)==0:
    #         print(pt)
    #         assert False, "this pt not close to a node, need to add fifrst?"
    #     elif len(list_nodes)>1:
    #         print(pt)
    #         assert False, "found > 1 node, must merge close nodes first"
    #     else:
    #         node = list_nodes[0]


    def map_strokes_to_nodelists(self, strokes, thresh=15):
        """ finds nodes aligned to strokes
        OUT:
        - list of paths, each path a list of nodes of len =2
        NOTE:
        - will fail iuf cannot find for some reason.
        """

        list_of_paths = []
        for strok in strokes:
            path = []
            for ind in [0, -1]:
                pt = strok[ind,:2]
                list_nodes = self.find_close_nodes_to_pt(pt, thresh)

                if len(list_nodes)==0:
                    print(pt)
                    assert False, "this pt not close to a node, need to add fifrst?"
                elif len(list_nodes)>1:
                    print(pt)
                    assert False, "found > 1 node, must merge close nodes first"
                else:
                    node = list_nodes[0]

                path.append(node)
            list_of_paths.append(path)
        return list_of_paths




    ########## POST PROCESSING
    def finalize(self, convert_to_splines_and_deinterp=True,
        dist_int=10., fail_if_finalized=False):
        """ converts all parses into format for analysis, coordiantes, etc.
        INPUT:
        - fail_if_finalized, True, then fails otherwise, skips all steps. assumes your
        previous finalizing is what you wanted to do.
        NOTES:
        - Puts all finalized things into:
        - only modifies self.Parses[ind]["strokes"]
        - 
        """
        from pythonlib.drawmodel.splines import strokes2splines

        if hasattr(self, "Finalized"):
            if self.Finalized:
                if fail_if_finalized:
                    assert False, "already finaled.."
                else:
                    # skip
                    return

        self.parses_fill_in_all_strokes()

        def _process(strokes):
            # Translate coords back
            strokes = self.strokes_translate_back(strokes)
            # Spline and deinterp
            if convert_to_splines_and_deinterp:
                strokes = strokes2splines(strokes, dist_int=dist_int)
            return strokes

        # parses
        for i in range(len(self.Parses)):
            strokes = _process(self.Parses[i]["strokes"])
            assert strokes is not None
            self.Parses[i]["strokes"] = strokes 

        self.Finalized=True


    def strokes_translate_back_batch(self, convert_to_splines_and_deinterp=True,
        dist_int=10.):
        """ Return list of all parses, in orriginal coords
        IN:
        - convert_to_splines_and_deinterp, uses 10 pix in between pts. uses
        codes from pyBPL. Note does both processes in same code currently.
        OUT:
        - strokes_task, original strokes_task, but made sure to undergo same processing
        as the parses.
        - parses, list of parse
        """
        from pythonlib.drawmodel.splines import strokes2splines
        
        self.parses_fill_in_all_strokes()

        def _process(strokes):
            # Translate coords back
            strokes = self.strokes_translate_back(strokes)
            # Spline and deinterp
            if convert_to_splines_and_deinterp:
                strokes = strokes2splines(strokes, dist_int=dist_int)
            return strokes

        # Stroke
        strokes_task = _process(self.StrokesInterp)

        # parses
        parses_list = []
        for i in range(len(self.Parses)):
            strokes = _process(self.Parses[i]["strokes"])
            parses_list.append(strokes)
        
        return strokes_task, parses_list


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
        assert False, "see summarize_parses"
        pass

    def plot_single_parse(self, ind):
        from pythonlib.drawmodel.parsing import plotParses
        parse = self.Parses[ind]["strokes"]
        plotParses([parse], ignore_set_axlim=True)


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

"""
7/9/21 - Good for parsing directly from strokes (task image)
NOTES:
- each parse will be in different memory loation, even if it looks the same.
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
        self.Parses = []

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
        """
        didmod_all, True if actually modified graph
        """

        # 1) Recenter strokes to positive coordinates
        self.strokes_translate_positive()
        self.strokes_interpolate_fine(plot=plot)
        self.skeletonize_strokes()
        self.graph_construct()

        print(graph_mods)
        # print(graph_mods_params)
        # self.plot_graph()

        # Graph mods
        didmod_all = False
        if graph_mods is not None:
            for mod, params in zip(graph_mods, graph_mods_params):
                print("** APPLYING THIS GRAPHMOD: ", mod)
                if mod=="merge":
                    didmod = self.graphmod_merge_nodes_auto(**params) 
                elif mod=="splitclose":
                    didmod = self.graphmod_split_edges_auto(**params)
                elif mod=="strokes_ends":
                    # endpoints for each stroke should match a node.
                    # params["strokes"] can be strokes, or string attribute (e.g., "StrokesInterp")
                    if isinstance(params["strokes"], str):
                        params["strokes"] = getattr(self, params["strokes"])
                    self.graphmod_add_nodes_strokes_endpoints(**params)
                    didmod = False # This never modifies the pts.
                else:
                    print(mod)
                    assert False
                print("** DONE: ")
                didmod_all = didmod_all or didmod

        #     self.plot_graph()
        # print(self.Graph.edges)
        # assert False

        if plot:
            self.plot_skeleton()
            self.plot_graph()

        return didmod_all

        # print(mod, params)
        # self.plot_graph()
        # assert False

    def parse_pipeline(self, N=1000, nwalk_det = 10, max_nstroke=400, max_nwalk=100, 
            plot=False, quick=False, stroke_order_doesnt_matter=False, direction_within_stroke_doesnt_matter=False):
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
            N=500 # this is usually quick
            nwalk_det = 5
            max_nstroke=80
            max_nwalk=20

        # Walk on graph
        # self.walk_extract_parses(10,50,50)
        self.walk_extract_parses(nwalk_det=nwalk_det, 
            max_nstroke=max_nstroke, max_nwalk=max_nwalk)
        self.ParsesParams["N"] = N

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
        self.parses_remove_redundant(stroke_order_doesnt_matter=stroke_order_doesnt_matter, 
            direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)

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

    def strokes_translate_forward(self, strokes):
        """ applies original transform to new strokes.
        doesnt modify self, just returns copy of strokes, mod
        """
        from pythonlib.tools.stroketools import translateStrokes
        return translateStrokes(strokes, self.Translation["xy"])

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

        self._graph_remove_zerolength_edge()


    ################# TOOLS, WROKING WOTH PARSERSTROKE
    def _walkerstroke_to_parserstroke(self, ws):
        """ converts a single WalkerStroke instance to a 
        single ParserStroke instance"""
        from .parser_stroke import ParserStroke
        PS = ParserStroke()
        if len(ws.list_ni)<2 or len(ws.list_ei)<1:
            self.plot_graph()
            print(ws.list_ni)
            print(ws.list_ei)
            assert False, "why?"
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
            if "list_walkers" in self.Parses[i].keys():
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

    ############# MANUALLY ENTERING PARSES
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


    def manually_input_parse_from_strokes(self, strokes, apply_transform=True):
        """ wrapper, to auto find list of nodes, and input, for this storkes.
        strokes must match the strokes coordinates that make up the graph.
        - will fail if doesnt find a complete set of edges.
        INPUT:
        - apply_transform, then first transfomrs strokes (spatial) using same params
        as did for the initial strokes entry.
        """
        from pythonlib.tools.graphtools import path_through_list_nodes, path_between_nodepair

        if apply_transform:
            strokes = self.strokes_translate_forward(strokes)

        if False:
            # Old version, problem is if there are >1 path between endpoints, then this fails.
            # Find endpoint nodes for each stroke
            list_of_nodes = self.map_strokes_to_nodelists(strokes)
            # Conver these lenght 2 nodes to lists of directed edges
            list_of_paths = []
            for nodepair in list_of_nodes:
                path = path_between_nodepair(self.Graph, nodepair)
                list_of_paths.append(path)

        # else:
        #     list_of_nodes_traj = self.map_strokes_to_nodelists_entiretraj(strokes)
        #     list_of_paths = []
        #     for nodes in list_of_nodes_traj:
        #         path = path_through_list_nodes(self.Graph, nodes)
        #         list_of_paths.append(path)

        #     # Conver these lenght 2 nodes to lists of directed edges
        #     print(list_of_paths)
        #     assert FAlse
        else:
            list_of_paths = self.map_strokes_to_edgelist(strokes)

        # Inset this as a new parse
        self.manually_input_parse(list_of_paths)


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


    def parses_remove_redundant(self, stroke_order_doesnt_matter=True,
        direction_within_stroke_doesnt_matter=True):
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
                
                if self.check_parses_is_identical(i, ii, stroke_order_doesnt_matter=stroke_order_doesnt_matter,
                    direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter):
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

        if kind=="list_of_paths":
            # path is a list of directed edges.
            parses = self.extract_parses_wrapper(ind, "parser_stroke_class")
            return [p.extract_list_of_directed_edges() for p in parses]
        if kind == "parser_stroke_class":
            key = "list_ps"
        elif kind=="strokes":
            key = "strokes"
            # return [p["strokes"] for p in self.Parses[ind]]
        elif kind=="summary":
            # returns list of dicts, where each item has keys: "edges", "walker", "traj"
            # i.e. combines all of the others.
            list_of_edges = self.extract_parses_wrapper(ind, "list_of_paths")
            list_of_walkers = self.extract_parses_wrapper(ind)
            list_of_trajs = self.extract_parses_wrapper(ind, "strokes")
            out = []
            for e,w,t in zip(list_of_edges, list_of_walkers, list_of_trajs):
                out.append({"edgesdir":e, "walker":w, "traj":t})
            return out

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
        elif kind=="summary":
            return [self.extract_parses_wrapper(i, kind="summary") for i in range(len(self.Parses))]
        else:
            return [self.extract_parses_wrapper(i, kind) for i in range(len(self.Parses))]


    def check_parses_is_identical(self, ind1, ind2, stroke_order_doesnt_matter=True,
        direction_within_stroke_doesnt_matter=True):
        """
        returns true if parses are identical (with certain invariances, see below)
        INPUT:
        - ind1 and ind2 index tinto self.Parses (e.g., self.Parses[1])
        - stroke_order_doesnt_matter, then order of storkes doenst matter
        - direction_within_stroke_doesnt_matter, then the direction within each stroke doesnt matter.
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
        list1 = [p.unique_path_id(invariant=direction_within_stroke_doesnt_matter) for p in list1]
        list2 = [p.unique_path_id(invariant=direction_within_stroke_doesnt_matter) for p in list2]


        # since dont care about direction, sort each list (of tuples of ints)
        if stroke_order_doesnt_matter:
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




    ################ AUTOMATICALLY MODIFYING PARSE GRAPH
    def _graph_remove_zerolength_edge(self, dist_thresh=20):
        """
        Remove any edges from node to itself
        - dist_thresh, only removes if the cum dist along the edge is less
        than this.
        """
        edges_to_remove = []
        for ed in self.Graph.edges:
            if ed[0]==ed[1]:
                # print(self.Graph.edges[ed])
                # only remove if this edge has short distance for pts
                from pythonlib.drawmodel.features import strokeDistances
                d = strokeDistances([self.Graph.edges[ed]["pts"]])[0]
                # make sure each node has other edges that would still remain
                nodes = ed[:2]
                nedges_per_node = [len(list(self.Graph.edges(n))) for n in nodes]

                # only remoive if fails distance treshold, and ther are other exdges remining
                if d<dist_thresh and all([n>1 for n in nedges_per_node]):
                    edges_to_remove.append(ed)

        if len(edges_to_remove)>0:
            print("Removing edges that go from a node to istself:")
            self.modify_graph(edges_to_remove=edges_to_remove)


    def find_closest_nodes_under_thresh(self, thresh): 
        """ Finds nodes which are close to each other
        """
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


    def graphmod_merge_nodes_auto(self, thresh=50):
        """ automaticalyl merge airs of nodes that are closert than thershold until no
        such pairs remain
        """

        did_mod = False

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
            did_mod = True
        return did_mod


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
        didmod = False
        while e is not None:
            
            if ct>20:
                print(G.nodes)
                print(G.edges)
                assert False, "recursive problem."
            print("**Splitting edge", e, "Merging with node:", n)
            nnew = self.split_edge(e, i)
            
            # self.plot_graph() 
            # Must then merge these two pts.
            print("Merging: ", {n, nnew})
            self.merge_nodes({n, nnew})

            # self.plot_graph() 
            # Merge any other nodes that are now too close.
            self.graphmod_merge_nodes_auto(thresh=thresh)

            e, d, i, n = self.find_closest_edge_to_split(thresh)

            # self.plot_graph()
            # assert False

            didmod=True
        return didmod

    ############## TOOLS FOR MODIFYING NODES MANUALLY
    def remove_node(self, node):
        """ Carefully removes a node. Connects up other nodes that were previously
        connected to this node.
        NOTE:
        - only works if there are 2 and only 2 edges for this node. Otherwise it is not clear
        how to reconnect the new nodes.
        """

        edges_to_remove = []

        # Check that there are only 2 edges
        list_edges = self.find_edges_connected_to_this_node(node)
        if len(list_edges)<2:
            assert False, "this node is a loop with no other nodes.."
        if len(list_edges)>2:
            assert False, "multiple adjacent nodes, not sure how to reconnect them"

        # Delete these two edges
        edges_to_remove.extend(list_edges)

        # Make a new edge between the two adjacent edges
        nodes_adjacent = self.find_adjacent_nodes(node)
        pts = self.find_pts_between_these_nodes([nodes_adjacent[0], node, nodes_adjacent[1]], concat=True)
        new_edge = (nodes_adjacent[0], nodes_adjacent[1], {"pts":pts})
        # if False:
        #     # Old, where I made mistake, assumed needed new ind...
        #     list_inds = self.find_edgeindices_for_edges_between_these_nodes(nodes_adjacent[0],
        #         nodes_adjacent[1])
        #     new_ind = max(list_inds)+1
        #     new_edge = (nodes_adjacent[0], nodes_adjacent[1], new_ind)

        # DO MOD
        self.modify_graph(nodes_to_remove=[node], edges_to_add=[new_edge], 
            edges_to_remove=edges_to_remove)

    def add_node(self, node_info, ver):
        """ [GOOD] Flexible way to manually adda single node.
        Takes care to (i) find where to add the node and (ii) reconnect edges
        properly.
        INPUT:
        - ver, str, method to use
        - node_info, list, depends on ver
        """

        if ver=="pt_edge":
            """ give a pt (xy) and snaps it onto the nearest location in edge
            - node_info, [pt, edge], where pt is [x,y] and edge is (n1, n2, idx)
            """
            # edge = (2,6,1)
            # pt = [75, 5]
            pt = node_info[0]
            edge = node_info[1]

            dist, ind_along_edge = self.min_dist_pt_to_edge(pt, edge)
            self.split_edge(edge, ind_along_edge)
            
        elif ver=="pt_anyedge":
            """ Given pt (xy) snaps it onto nearest location on nearest edge
            """
            assert False, "test this"
            strokes = [np.array([pt, pt, pt])]
            P.graphmod_add_nodes_strokes_endpoints(strokes)


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
                    # Then skip. This is usally when edge of a line is close to another
                    # node, for natural reasons.
                    continue
                    # self.plot_graph()
                    # print(pt)
                    # print(list_ni)
                    # assert False, "first do merge"
                else:
                    # fine, already has a single node.
                    continue

                # Split graph at each of the extracted edges
                for ed, ind in zip(edges, inds):
                    print("**Splitting, adding strokes endpoints", ed, ind)
                    self.split_edge(ed, ind)

    def merge_nodes(self, nodes):
        """ Merges a pair of nodes intoa  new node, with position in between.
        Takes care to update edges correctly
        """

        def _merge_nodes(nodes, G):
            """ 
            - nodes, set of nodes, e..g, {3,4}
            - G
            OUT:
            - modifies G to have new node (ind is max + 1), and new edges replacing old edges.
            """
            assert isinstance(nodes, set)
            list_nodes = list(nodes)
            
            o1 = self.get_node_dict(list_nodes[0])["o"]
            o2 = self.get_node_dict(list_nodes[1])["o"]
            radius = np.linalg.norm(o1-o2)
            radius *= 2
            print("--- RADIUS", radius)

            # new center
            onew = np.round(np.mean(np.stack([G.nodes[n]["o"] for n in nodes]), axis=0)).astype(np.int16)
            nnew = max(list(G.nodes))+1

            # remove all edges between these
            elist = [e for e in G.edges if set(e[:2])==nodes]

            # add new edges to replace any old edges
            edges_to_add = []
            edges_to_remove = []
            edges_to_remove.extend(elist)
            for ni in [0, 1]:
                nthis = list_nodes[ni]
                othis = G.nodes[nthis]["o"]

                for e in G.edges:
                    if nthis in e[:2]:
                        edg = G.edges[e]

                        # conver this edge to a new edge, using same pts
                        nother = [n for n in e[:2] if n!=nthis]
                        assert len(nother)==1, "ytou might have repeat node? a-->a"
                        nother = nother[0]

                        if set([nthis, nother])==nodes:
                            # this is edge between the pair you are merging.
                            # you have already added it.
                            continue

                        # construct a new edge
        #                 weight = edg["weight"]
        #                 visited = edg["visited"]
                        pts = self.get_edge_pts(e, anchor_node=nthis)
                        # pts = np.insert(pts, 0, onew.reshape(1,-1), axis=0)
                        pts[0, :] = onew

                        if False:
                            pts = edg["pts"]

                            # add the center to the end that is currently closest to the center
                            d1 = np.linalg.norm(pts[0,:]-othis)
                            d2 = np.linalg.norm(pts[-1,:]-othis)
                            if d1<d2:
                                pts = np.insert(pts, 0, onew.reshape(1,-1), axis=0)
                            else:
                                pts = np.append(pts, onew.reshape(1,-1), axis=0)

                        # smoothening out the new edge
                        # - radius, based on distance between two nodes that you are merging
                        from pythonlib.tools.stroketools import smooth_pts_within_circle_radius
                        pts = smooth_pts_within_circle_radius(pts, radius)

                        # Add this new edge
                #         G.add_edge(nnew, nother, pts=pts, weight=weight)
                        edges_to_add.append((nnew, nother, {"pts":pts}))
        #                 edges_to_add.append((nnew, nother, {"pts":pts, "weight":weight}))
                        edges_to_remove.append((nthis, nother, e[2]))
                            

            nodes_to_add = [(nnew, {"o":onew, "pts":onew.reshape(1,-1)})]
            nodes_to_remove = list(nodes)
            self.modify_graph(nodes_to_add=nodes_to_add, nodes_to_remove=nodes_to_remove, 
                        edges_to_add=edges_to_add, edges_to_remove=edges_to_remove)
            
            return G

        return _merge_nodes(nodes, self.Graph)


    def split_edge(self, edge, ind):
        """
        Split this edge into two new edges at location within edge defined by ind.
        (i.e., add a node, and connect that node to the endpoints)
        Takes care to update all edges etc.
        """

        def _split_edge(edge, ind, G):
            """ split this edge at this index along its pts
            """

            # First, make the new node
            # pt for the new node
            print("split edge start")
            print("this edge", edge)
            print("at this ind", ind)
            print("len of edge", self.Graph.edges[edge]["pts"].shape)
            print("all edges", G.edges)
            onew = G.edges[edge]["pts"][ind]   
            
            # figure out orientation of pts
            pts = G.edges[edge]["pts"]
            n1 = edge[0]
            n2 = edge[1]
            o1 = G.nodes[n1]["o"]
            o2 = G.nodes[n2]["o"]
            if self.check_pts_orientation(pts, o1, return_none_on_tie=True) is None:
                # then either way is fine
                pass
            else:
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
        

    ################### LOW-LEVEL CODE FOR MANIPULATING GRAPH
    def modify_graph(self, nodes_to_add=None, nodes_to_remove=None, edges_to_add=None, 
        edges_to_remove=None):
        """ Low-level to add/remove nodes/edges. DO NOT use this unless you know what you doing,
        becuase this doesnt' autoatmically prune and add edges so that graph is appropriately
        connected.
        INPUTS:
        - nodes_to_add, list of nodes, where each is (nnew, {"o":onew, "pts":onew.reshape(1,-1)})]
        - nodes_to_remove, list of ints
        - edges_to_add, list of edges, where edges[0] = (node0, node1, {pts:[...]})
        - edges_to_remove, list of edges, where each is (node0, node1, index)
        """ 
        
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
            for ed in edges_to_add:
                assert isinstance(ed[2], dict), "ed[2] must be {pts:[...]}"
            G.add_edges_from(edges_to_add)

        print("- Done modifying, new nodes and edges")
        print(G.nodes)
        print(G.edges)



    ######################## GRAPHMOD - tools
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
                print(d1, d2)
                print(o)
                print(pts[0], pts[-1])
                assert False
        if d1<d2:
            return True
        else:
            return False



    ########## GRAPHMOD - helpers to find nodes and edges
    def find_edges_connected_to_this_node(self, node):
        """ return list of edges given a node (int)
        OUT:
        - list_edges, e.g, [(0,3,1), (..)...]
        NOTE:
        - if this node doesnt exist, returns empty list.
        """
        list_edges = [ed for ed in self.Graph.edges if node in ed[:2]]
        return list_edges

    def find_adjacent_nodes(self, node):
        """ Return a list of nodes connected to this node.
        """
        return list(self.Graph.neighbors(node))

    def find_edges_between_these_nodes(self, node1, node2):
        """ Returns list of all edges between thse nodes
        INPUT:
        - node1, node2, ints. order doesnt matter.
        OUT:
        - list of edges
        """
        return [ed for ed in self.Graph.edges if set(ed[:2])==set([node1, node2])]

    def find_edgeindices_for_edges_between_these_nodes(self, node1, node2):
        """ Returns list of indices for all edges that exist between thse nodes
        i.e,, if edge = (1,2,0) then 0 is the index.
        """
        list_edges = self.find_edges_between_these_nodes(node1, node2)
        return [ed[2] for ed in list_edges]

    def find_paths_between_these_nodes(self, list_nodes):
        """ 
        Find entire path (lsit of edges) in order between nodes in list_nodes.
        Must have unambiguous edge between each adjacent pari of noides in 
        list_nodes.
        e..g, list_ni = [0,5,2]
        --> [(0, 5, 0), (5, 2, 0)]
        """
        from pythonlib.tools.graphtools import path_through_list_nodes
        return path_through_list_nodes(self.Graph, list_nodes)

    def find_pts_between_these_nodes(self, list_nodes, concat=False):
        """ Returns the unambiguous (fails otherwise) path between 
        list of nodes in list_nodes. Must give enough info in list_nodes
        so that there is one and only one path.
        IN:
        - list_nodes, e.g, [0,7,2]
        OUT:
        - pts, 
        --- list of pts for each edge (if concat=False)
        --- Nx2 array (if concat==True)
        NOTE:
        - will ensure that the order of pts is correct
        """

        # get list of edges between these nodes
        list_edges = self.find_paths_between_these_nodes(list_nodes)

        # Get list of edges
        list_edges_pts = []
        for ed, anchor_node in zip(list_edges, list_nodes[:-1]):
            pts = self.get_edge_pts(ed, anchor_node=anchor_node)
            list_edges_pts.append(pts)
            
        if concat==True:
            return np.concatenate(list_edges_pts, axis=0)
        else:
            return list_edges_pts


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
        o = self.Graph.nodes[node]["o"]
        return self.min_dist_pt_to_edge(o, edge)

    def min_dist_pt_to_edge(self, pt, edge):
        """
        INPUT:
        - pt, xy coord.
        - edge, edge index.
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

    def find_close_nodes_to_pt(self, pt, thresh=10, only_one=False, take_closest=False):
        """
        only_one, then fails if >1. ok if 0.
        take_closest, then takes closest if there are multiple.
        OUT:
        - list_nodes, list of ints. empty if none.
        """

        list_nodes = []
        distances = []
        for n in self.Graph.nodes:
            o = self.Graph.nodes[n]["o"]
            d = np.linalg.norm(pt-o)
            if d<thresh:
                list_nodes.append(n)
                distances.append(d)
        if only_one:
            if len(list_nodes)>1:
                assert False
        if take_closest:
            if len(list_nodes)>0:
                tmp = [(n,d) for n,d in zip(list_nodes, distances)]
                tmp = sorted(tmp, key=lambda x:x[1])
                list_nodes = [tmp[0][0]]
        return list_nodes

    ############# USE STROKES TO FIND/MANIPULATE NODES AND EDGES
    def map_strokes_to_nodelists(self, strokes, thresh=5):
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
                    print([self.Graph.nodes[n]["o"] for n in self.Graph.nodes])
                    assert False, "this pt not close to a node, need to add fifrst?"
                elif len(list_nodes)>1:
                    print(pt)
                    print([self.Graph.nodes[n]["o"] for n in self.Graph.nodes])
                    assert False, "found > 1 node, must merge close nodes first"
                else:
                    node = list_nodes[0]

                path.append(node)
            list_of_paths.append(path)
        return list_of_paths

    def map_strokes_to_edgelist(self, strokes, thresh=5):
        """ 
        returns list of edges [(1,2,0), ...] that the input stroke is along a trajectory of.
        makes sure that if a pair of nodes
        have multiple possible edgse, takes teh edge that occurs more often (as you travel)
        along the pts
        """

        def _traj_to_edgelist(traj):
            """ traj is Nx2
            """

            last_visited_node = None
            tracker = {}
            list_of_edges = []
            for i, pt in enumerate(traj[:,:2]):
                nodes = self.find_close_nodes_to_pt(pt, only_one=True, thresh=thresh, take_closest=True)
                
                if len(nodes)==1:
                    if last_visited_node is None:
                        last_visited_node = nodes[0]

                    elif last_visited_node is not None and nodes[0]!=last_visited_node:
                        # then you have visisted a new node. 
                        # if change last_visited_node, then pick the most visited edge, then reset.
                        new_node = nodes[0]
                        # print(i)
                        # print(last_visited_node)
                        # print(nodes)
                        # print("HERE", tracker)
                        # # pick out candidate edges that explain the just finished traj
                        # edges_candidate = [ed for ed in tracker.keys() if set(ed[:2])==set([last_visited_node, new_node])]

                        # sort all candidate edges by how oten they visited
                        edges_candidate = [(ed, nvisit) for ed, nvisit in tracker.items() if set(ed[:2])==set([last_visited_node, new_node])]

                        # sort and pick out most highly visited edge.
                        edges_candidate = sorted(edges_candidate, key=lambda x: x[1]) # sort in ascending order.
                        edge_just_done = edges_candidate[-1][0]

                        if edge_just_done[0]==new_node:
                            edge_just_done = (edge_just_done[1], edge_just_done[0], edge_just_done[2]) # mnake sure is in order.
                        list_of_edges.append(edge_just_done)

                        # reset everything
                        last_visited_node = nodes[0]
                        tracker = {}
                    else:
                        # do nothing, since nthing changed
                        pass

                if i==0 or i==traj.shape[0]:
                    # print(i,pt)
                    # print([self.Graph.nodes[n]["o"] for n in self.Graph.nodes])
                    assert len(nodes)==1, "on and off should match a node..."

                # find edges
                edges, dists, inds = self.find_close_edge_to_pt(pt, thresh=thresh)

                # only keep edges that involve the current node
                edges = [ed for ed in edges if last_visited_node in ed[:2]]

                # update tracker
                for ed in edges:
                    if ed in tracker.keys():
                        tracker[ed] +=1
                    else:
                        tracker[ed] = 1
            return list_of_edges

        return [_traj_to_edgelist(traj) for traj in strokes]



    def map_strokes_to_nodelists_entiretraj(self, strokes, thresh=5):
        """ finds nodes aligned to strokes. includes all nodes, that are found along the 
        trajectory. 
        OUT:
        - list of paths, each path a list of nodes of at lest len=2, but more if more are encountered.
        - e.g,,: [[4, 2, 1], [3, 4, 5, 6], [8, 2, 0], [8, 5, 7]], means there are 4 strokes, each going
        thru those nodes in those order.

        NOTE:
        - can guarantee that these nodes should be in order (modulo reversal)
        - asserts that will find something for the first and last pts.
        """

        list_of_paths = []
        for strok in strokes:
            path = []
            # start from first pt, and continue
            for i in range(strok.shape[0]):
                pt = strok[i,:2]
                list_nodes = self.find_close_nodes_to_pt(pt, thresh)

                if i==0 or i==strok.shape[0]:
                    assert len(list_nodes)==1, "must find a node at start and end."

                if len(list_nodes)==1:
                    node = list_nodes[0]
                    if node not in path:
                        path.append(node)
                
                if len(list_nodes)>1:
                    print(pt)
                    print([self.Graph.nodes[n]["o"] for n in self.Graph.nodes])
                    assert False, "found > 1 node, must merge close nodes first"

            list_of_paths.append(path)
        return list_of_paths


    ########## LOW-LEVEL HELPERS WITH EDGES AND NODES
    def get_edge_helper(self, edge):
        """ Help return correct index for edge, given vareity of 
        ways of inputing edge here
        INPUT:
        - edge, e.g.,:
        --- edge = (1,2), then looks for one edge that is (1,2,x) where
        x is idx. 
        --- edge = (2,1,0), which returns the correct ordering of 1 and 2.
        OUT:
        - (n1, n2, idx)
        NOTE:
        - in all cases, fails if doiesnt find one and only one edge.
        """

        list_edges = []
        for ed in self.Graph.edges:
            if set(edge[:2]) != set(ed[:2]):
                continue

            if len(edge)>2:
                # also check the index
                if edge[2]!=ed[2]:
                    continue

            # passes all tests. keep
            list_edges.append(ed)

        if len(list_edges)==0 or len(list_edges)>1:
            print(list_edges)
            assert False, "didnt find exactly one edge"

        return list_edges[0]

    def get_edge_dict(self, edge):
        """ Return dict fo this edge
        e..g, dict[pts] = [...
        """
        edge = self.get_edge_helper(edge)
        return self.Graph.edges[edge]


    def get_edge_pts(self, edge, anchor_node=None):
        """ return pts, array (N,2)
        INPUT:
        - anchor_node, pts will be oriented (flipped) so that
        pts[0,:] will be closer to anchor_node, than will be pts[-1]
        --- if None, then returns in the saved order.
        """

        pts = self.get_edge_dict(edge)["pts"]
        if anchor_node is None:
            return pts
        else:
            anchor_pts = self.get_node_dict(anchor_node)["o"]
            doflip = not self.check_pts_orientation(pts, anchor_pts)
            if doflip:
                pts = np.flipud(pts)
            return pts

    def get_node_dict(self, node):
        """ Returns dict for this node.
        fails if doesnt exist
        """
        return self.Graph.nodes[node]
        
    ########## FILTER/CLEAN UP PARSES
    def filter_single_parse(self, ver, ind):
        """ returns True/False, where True means
        bad (should remove)
        """

        if ver=="any_path_has_redundant_edge":
            """ fails whether same or diff dir
            only checks within each path (turn on yourself).
            # 1) Find parses where there are paths where repeat an edge (in the same path)
            # could repeat in same or diff direction
            """
            list_of_paths = self.extract_parses_wrapper(ind, "list_of_paths")
            PS = self.extract_parses_wrapper(ind)[0]

            def _path_does_repeat_edge(path):
                """ returns True if this path has a repeated edge (can be either dir)
                """
                for i in range(len(path)):
                    for ii in range(i+1, len(path)):
                        if PS._edges_are_same(path[i], path[ii]):
                            return True
                return False

            # check each path
            if any([_path_does_repeat_edge(path) for path in list_of_paths]):
                return True
        if ver=="any_two_paths_have_common_edge":
            """ this common edge can be in either direction
            """
            list_of_paths = self.extract_parses_wrapper(ind, "list_of_paths")
            PS = self.extract_parses_wrapper(ind)[0]

            # conpare all pairs of paths
            for i in range(len(list_of_paths)):
                for ii in range(i+1, len(list_of_paths)):
                    path1 = list_of_paths[i]
                    path2 = list_of_paths[ii]
                    for e1 in path1:
                        for e2 in path2:
                            if PS._edges_are_same(e1, e2):
                                return True
        return False

    def filter_all_parses(self, ver, do_remove=True):
        """ 
        Rturns list of True False same len as parses, where True means is bad and should
        remove. 
        if do_remove, then does remove from self.Parses
        """
        badlist = [self.filter_single_parse(ver, i) for i in range(len(self.Parses))]
        self.Parses = [self.Parses[i] for i, x in enumerate(badlist) if not x]
        print(f"Removed {[i for i, x in enumerate(badlist) if x]}")
        print(f"Removed {sum(badlist)} parses.")
        return badlist
            


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

    def print_parse_concise(self, ind):
        out = f"{ind} = "
        for i, p in enumerate(self.Parses[ind]["list_ps"]):
            out+=f"{i}:{p.EdgesDirected} - "
            # p.print_elements(verbose=False)        
        print(out)

    def print_parse_concise_all(self):
        """ print all parses, each a single line"""
        for i in range(len(self.Parses)):
            self.print_parse_concise(i)

    def print_graph(self):
        nodes = self.Graph.nodes
        edges = self.Graph.edges
        print(f"NODES: {nodes} -- EDGES: {edges}")


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
            ax.set_xlabel("y")
            ax.set_ylabel("x")

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






    def plot_parses(self, inds=None, Nmax=50):
        """ new, quick, to just plot parses, not in cluding summary states, etc
        """
        from pythonlib.dataset.dataset import Dataset
        import random
        D = Dataset([])
        if inds==None:
            # then plot all
            inds = range(len(self.Parses))
        if len(inds)>Nmax:
            inds = random.sample(inds, Nmax)
        parses_list = self.extract_all_parses_as_list()
        parses_list = [parses_list[i] for i in inds]
        titles = inds
        titles = [str(i) for i in inds]
        # print(len(parses_list))
        D.plotMultStrokes(parses_list,titles=titles, jitter_each_stroke=True)

    def plot_single_parse(self, ind):
        from pythonlib.drawmodel.parsing import plotParses
        parse = self.Parses[ind]["strokes"]
        plotParses([parse], ignore_set_axlim=True)


    def summarize_parses(self, nmax = 50, inds=None):
        from pythonlib.drawmodel.parsing import summarizeParses
        import random
        if len(self.Parses)==0:
            return
        parses = self.extract_all_parses_as_list()
        self.plot_skeleton()
        self.plot_graph()
        if inds is not None:
            parses = [parses[i] for i in inds]
        print("FOUND THIS MANY PARSES: ", len(parses))
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

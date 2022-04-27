""" General TaskClass
(made during drawnn but should be extendable for other purposes
"""
from . import primitives as Prim
import numpy as np


class TaskClass(object):
    """ Abstract class for all tasks.
    All data exposed will have first already been converted to abstract cooridnates
    (e..g,, self.Strokes, self.Points)
    TODO: general-purpose self.Program and self.Plan
    """

    def __init__(self):
        self.Params = {}
        self._initialized = False
        self.Name = None
        self.Parser = None
        self.Parses = None
        self.ShapesOldCoords = None

    def initialize(self, ver, params, coord_force_same_aspect_ratio=True,
            convert_coords_to_abstract=True, split_large_jumps=False):
        """ General purpose, to initialize with params for
        this task. Is flexible, takes in many formats, converts
        each into general abstract format
        INPUT:
        - ver, str, in {"drawnn", "ml2"}, indicates what format
        - params, flexible, what format depends on what ver.
        - coord_force_same_aspect_ratio, then for drawnn and ml2 coords,
        - convert_coords_to_abstract, then self.Strokes, and others will be in abstract system.
        otherwise keep in whatever was input.
        will make sure is asme aspect ratio as asbtract coords. Does this
        by padding the smaller dimension to match abstract (equally on both ends)
        """

        # Save input params
        self.Params["input_ver"] = ver
        self.Params["input_params"] = params
        self.Params["coord_force_same_aspect_ratio"] = coord_force_same_aspect_ratio
        self.Params["convert_coords_to_abstract"] = convert_coords_to_abstract
        self.Params["split_large_jumps"] = split_large_jumps

        # Extract Strokes
        if ver=="drawnn":
            # New version of drawnn
            program = params["program"]
            shapes = params["shapes"]
            chunks = params["chunks"]
            self._initialize_drawnn(program, shapes, chunks)

        # Create things like self.StrokesOldCoords
        # Do not create self.Strokes here. This done next block.
        elif ver=="ml2":
            # Monkeylogic tasks
            taskobj = params
            self._initialize_ml2(taskobj) 
        else:
            print(ver)
            assert False, "not coded"

        # Convert strokes to appropriate coordinate system
        # and generate Points
        # convert things like self.StrokesOldCoords to self.Strokes
        if convert_coords_to_abstract:
            self._initialize_convcoords()
        else:
            # use the input coordinates
            # TODO: self.ProgramOldCoords and others.
            self.Strokes = self.StrokesOldCoords
            self.Shapes = self.ShapesOldCoords

        # Convert strokes to points
        self.Points = np.stack([ss for s in self.Strokes for ss in s], axis=0) # flatten Strokes

        # Check that nothing goes out of bounds
        self._check_out_of_bounds()

        # Split into distinct strokes, wherever task pts make dramatic jump.
        if split_large_jumps:
            self._split_strokes_large_jump()


    def _initialize_ml2(self, taskobj):
        """ Pass in a task from monkeylogic.
        - taskobj, task class from drawmodel.tasks

        NOTES:
        - self.Shapes is like list of lists, where lower lists are [shapename, tform params]
            e.g., for ml2 [note: might be different for different task versions]
            [['line',{'x': array(1.5),
               'y': array(0.),
               'th': array(0.),
               'sx': array(2.),
               'sy': array(2.),
               'order': 'trs'}],
             ['circle',
              {'x': array(0.),
               'y': array(1.5),
               'th': array(0.),
               'sx': array(0.7),
               'sy': array(0.7),
               'order': 'trs'}]
        """

        # Extract data
        self.StrokesOldCoords = taskobj.Strokes

        taskobj.program_extract() 
        self.ProgramOldCoords = taskobj.Program

        # Get abstract representaion of objects (primitives)
        taskobj.planclass_extract_all()
        if taskobj.PlanDat is None:
            # Then this is old version, before using planclass
            taskobj.objects_extract()
            self.ShapesOldCoords = taskobj.Objects
            self.ShapesOldCoords = [[S["obj"], S["tform"]] for S in self.ShapesOldCoords] # conver to general format
            for i in range(len(self.ShapesOldCoords)):
                if self.ShapesOldCoords[i][1] is not None:
                    self.ShapesOldCoords[i][1]["theta"] = self.ShapesOldCoords[i][1]["th"]
                    del self.ShapesOldCoords[i][1]["th"]
        else:
            # New version, use planclass representation of prims. Dont have to re-extract
            # here post-hoc. Dont use the "Shapes" thing anymore, instead represent them as
            # PrimitiveClass objects
            self.Primitives = taskobj.PlanDat["primitives"]
            self.PlanDat = taskobj.PlanDat
            
            # populate self.ShapesOldCoords, for backwards compativbility with other code.
            self.ShapesOldCoords = [p.extract_as("shape") for p in self.Primitives]

    def _initialize_drawnn(self, program=None, shapes=None, chunks=None):
        """
        To initialize a new task for drawnn training.
        PARAMS:
        - program is list of subprograms, each suprogram is list of 
        lines, where each line evaluates and takes in the accumulation of
        previous lines within the subprogram. e..g,:
            program = []
            program.append(
                [["line"],
                ["repeat", {"p":"$", "N":3, "x":0.5}]])
            program.append(
                [["circle", [None, None, 0.5, 0.5, None, None]],
                 ["line"],
                 ["repeat", {"y":0.5}]]
            )
        Note: each line is like [<function>, <params>], where params can be
        --- dict that will be passed in as kwargs. Note that "$" means take in
        accumulation of previousl lines in this subprogram as this argument. If 
        dont pass in $, then will assume to pass in previous lines into first
        position argument (e..g,, in repeat in 2nd suprogram).
        --- list, this must not miss any arguemnts, but can pass None if to use
        default. Can choose to pass in the $, or not, if not then uses as first arg.

        - shapes, is list of shape (single stroke) primitives. this will be automatcaly
        extracted from program. these are still symbolic. 
        --- can pass in shapes, in which case program will be None. I did this since 
        havent figured out best way to evaluate program.
        - strokes, is like shapes, but in numbers. This is computed automatically.
        - chunks, list of higher level chunking, where each chunk is [chunkname<str>, [shapes in this chunk]]
        e.g., chunks = [["lolli", [0, 1]], ["lolli", [4,3]]]. I added this so can use for 
        hierarhcical RNN.
        """


        a = program is not None
        b = shapes is not None
        assert a!=b, "provide either program or shapes"

        self.ProgramOldCoords = program

        if program is not None:
            self.ShapesOldCoords = self.program2shapes(self.ProgramOldCoords)
        else:
            self.ShapesOldCoords = shapes
            
        self.StrokesOldCoords = self.shapes2strokes(self.ShapesOldCoords)

        # NOT YET DONE - general coords, make these empty. following code will generate these
        # by converting coords.
        self.Program = None
        self.Shapes = None
        self.Strokes = None

        # Chunks
        if chunks is not None:
            self.Chunks = chunks

            # sanity checks
            for c in self.Chunks:
                inds = c[1]
                for i in inds:
                    assert i<len(self.ShapesOldCoords), "chunks are referencing shapes that dont exist"



    def _initialize_convcoords(self, out_="abstract"):
        """ initial conversion of input coordinate system into abstract system
        Modifies: Strokes and Points. 
        Converts things like self.StrokesOldCoords to self.Strokes.
        TODO: have not done for self.Program and self.Strokes
        print("TODO (drawmodel-->taskgeneral): convert coords for self.Shapes. Do this in ml2 taskobject")
        """

        in_ = self.Params["input_ver"]
        # out_ = "abstract"

        # print(in_, out_)
        # print(self.Strokes)

        self.Strokes = [self._convertCoords(s, in_, out_) for s in self.StrokesOldCoords]

        # TODO: self.Shapes = function(self.ShapesOldCoords) # have not coded this before. previously just
        # used the input coordinate system.

    def _preprocess(self):

        # Convert strokes to abstract coordinates (to be more general)
        from .image import convertCoordGeneral  
        pass


    def _split_strokes_large_jump(self):
        from pythonlib.tools.stroketools import split_strokes_large_jumps
        if self.Params["input_ver"]=="ml2":
            thresh = 50
        else:
            assert False, "what is good threshold where if adajncet pts greater, then split storke here?"
        self.Strokes = split_strokes_large_jumps(self.Strokes, thresh)

    def _convertCoords(self, pts, in_, out_):
        """ convert between coordinate systems.
        automatically determines sketchpads, etc.
        INPUTS:
        - in_, {"abstract", "ml2", "drawnn"}
        - out_, {"abstract", "ml2", "drawnn"}
        - pts, N x 2(or3), array
        """

        from pythonlib.drawmodel.image import convCoordGeneral
        # strokes = T.Strokes

        # make sure sketchpads are initialized
        self._make_sketchpads()

        # compute
        edges_in = self._Sketchpads[in_]
        edges_out = self._Sketchpads[out_]
        pts_out = convCoordGeneral(pts, edges_in, edges_out)

        return pts_out

    def get_sketchpad_final(self):
        """ Helps to return the sketchpad associated wtih the final coordinate
        system. i.e,., whgat' sin self.Points, etc.
        RETURNS:
        - skethpad, 2x2 np array, where sketchpad[0,:] are xlims...
        """

        if self.Params["convert_coords_to_abstract"]:
            # Then final pts are in abstract coords
            sketchpad = self._Sketchpads["abstract"]
        else:
            # then are in input coords
            if self.Params["input_ver"] == "ml2":
                sketchpad = self._Sketchpads["ml2"]
            else:
                assert False, "code this"
        return sketchpad

    def _make_sketchpads(self):
        """ get sketchpad coordinates, both hard coded and dynamic.
        for format, see _converCoords for detaisl
        """
        
        if not hasattr(self, "_Sketchpads"):
            self._Sketchpads = {
                "abstract":np.array([[-3.5, 3.5], [-3.5, 3.5]]),
                "drawnn":np.array([[-3.5, 3.5], [-3.5, 3.5]])    
            }

            if self.Params["input_ver"] == "ml2":
                self._Sketchpads["ml2"] = self.Params["input_params"].Task["sketchpad"].T
                # print("Generated self._Sketchpads to:")
            if self.Params["coord_force_same_aspect_ratio"]:
                from .image import coordsMatchAspectRatio

                edgesgood = self._Sketchpads["abstract"]

                # then makes sure each sketchpad is same aspect ratio as abstracat
                sketchpad_new = {}
                for k, edges in self._Sketchpads.items():

                    if k != "abstract":
                        edgesnew = coordsMatchAspectRatio(edgesgood, edges)
                        sketchpad_new[k] = edgesnew
                        sketchpad_new[f"{k}_orig"] = edges
                    else:
                        sketchpad_new[k] = edges

                        # self._Sketchpads[f"{k}_orig"] = edges
                        # self._Sketchpads[k] = edgesnew
                self._Sketchpads = sketchpad_new



    def _check_out_of_bounds(self):
        """
        check no pts out of bounds. would indicate mistake in sketchpad in.
        assumes hav already converted to abstract
        RETURNS:
        - fails (assert False) if pts not in bounds.
        """

        self._make_sketchpads()
        sketchpad = self.get_sketchpad_final()
        xlims = sketchpad[0,:]
        ylims = sketchpad[1,:]
        pts = self.Points

        a = np.all((pts[:,0]>xlims[0]) & (pts[:,0]<xlims[1]))
        b = np.all((pts[:,1]>ylims[0]) & (pts[:,1]<ylims[1]))
        if a==False or b==False:
            print(sketchpad)
            print(pts)
            print(self.Strokes)
            print(self.Params)
            assert False , "not in bounds"


    ########################

    # def get_task_id(self):
    #     """ get unique id for this task. 
    #     RETUNRS:
    #     tuple (category, number in cat)
    #     """
    #     if self.Params["input_ver"]=="ml2":
    #         taskcat = self.Params["input_params"].Task["stage"]
    #         taskstr = self.Params["input_params"].Task["str"]
    #         idx = taskstr.find(taskcat)
    #         tasknum = int(taskstr[idx+len(taskcat)+1:])
    #     else:
    #         assert False
    #     return taskcat, tasknum

    def get_task_category(self):
        """ [GOOD] Get human-interpretable task category
        This flexible depending ion task
        RETURNS:
        - taskcat, str, 
        """
        if self.Params["input_ver"]=="ml2":
            taskcat = self.Params["input_params"].info_name_this_task_category()   
            return taskcat
        else:
            assert False

    def get_los_id(self):
        """ Get load_old_set information, which means info for this as a fixed task
        that was loaded online during expt
        RETURNS:

        """
        if self.Params["input_ver"]=="ml2":
            return self.Params["input_params"].info_summarize_task()["los_info"]
        else:
            assert False

    def get_unique_name(self):
        """ Wrapper to get good unique name
        THIS is good version, identical to Dataset class
        RETURNS:
        - name, string name that is unique to this task.
        """
        if self.Params["input_ver"]=="ml2":
            Task = self.Params["input_params"]
            random_task = not Task.info_is_fixed()
            if random_task:
                return self._get_number_hash(ndigs=10, include_taskstrings=False,
                    include_taskcat_only=True, compact=True)
            else:
                return self._get_number_hash(ndigs=6, include_taskstrings=True)
        else:
            assert False

    def _get_number_hash(self, ndigs = 6, include_taskstrings=True, 
        include_taskcat_only=False, compact = False):
        """ Returns number, with digits ndigs, based on 
        coordinates of task. shodl be unique id.
        - include_taskstrings, False, then just number. True, then uses task 
        category, and numberical id (which entered during task creation)
        NOTE: confirmed that the strokes that go in here are identical to strokes in Dataset
        """
        # if include_taskcat_only:
        #     assert include_taskstrings is False, "choose either entier string, or just the cat"
        if self.Params["input_ver"]=="ml2":
            idnum = self.Params["input_params"].info_generate_unique_name(self.Strokes, 
                nhash=ndigs, include_taskstrings=include_taskstrings, include_taskcat_only=include_taskcat_only)

            if compact:
                # remove long "random" in string
                ind = idnum.find("random")
                if ind>0:
                    idnum = idnum[:ind+1] + idnum[ind+6:]

            return idnum
        else:
            assert False

    #############################
    def program2shapes(self, program):
        assert False, "not done, see __init__ for what need to do."
        shapes =[]
        for subprog in program:

            funcstring = evaluateProg(subprog) 
            subshapes = evaluateString(funcstring)
            shapes.extend(subshapes)


    def generatePlan(self):
        """ convert to sequential actions.
        e.g., based on simple sequential biases.
        """
        self.Plan = None
        assert False, "not coded"


    def shapes2strokes(self, shapes):
        def evaluateShape(shape, params):
            """ 
            - shape, is string name
            - params, different options:
            --- list, will be passed into transform in order.
            --- dict, will be passed into transform as kwargs.
            --- params order: [x=0, y=0, sx=1, sy=1, theta=0, order="trs"]
            --- Note: can pass in None to get defaults.
            === NOTES
            - now line is centered at 0,0. Original primitives centered at (0.5,0)
            === RETURNS
            [np.array]
            """
            def transform(p, x=None, y=None, sx=None, sy=None, theta=None, order=None):
                """ outputs stroke """
                T = Prim.makeAffine2(x, y, sx, sy, theta, order)
                return Prim._tform(p, T)

            if shape=="line":
                p = Prim.transform(Prim._line, x=-0.5)
            elif shape=="circle":
                p = Prim._circle
            else:
                print(shape)
                assert False, "not coded"

            if isinstance(params, list):
                return transform(p, *params)
            elif isinstance(params, dict):
                return transform(p, **params)
            else:
                print(params)
                assert False

        strokes =[]
        for sh in shapes:
            s = sh[0]
            if len(sh)==1:
                params = []
            else:
                params = sh[1]
            strokes.extend(evaluateShape(s, params))

        return strokes


    ############# SHAPES
    def _shape_to_string(self, shape):
        """
        Convert shape to a string, which is hashable.
        shape is an item in self.Shapes. 
        - shape, list, [shapename, dict_params], e./.g,"['line',
      {'x': array(-1.7),
       'y': array(1.7),
       'sx': array(1.55),
       'sy': array(1.55),
       'order': 'trs',
       'theta': array(1.57079633)}]
       
       RETURNS:
       - string, e.g., linex-1.7y1.7sx1.55sy1.55ortrsth1.57
       """
        
        import numpy as np
        outstring = ""
        
        # name of object (e..g, line)
        outstring += shape[0]
        
        # affine transofmr params
        prms_in_order = ["x", "y", "sx", "sy", "order", "theta"]
        for key in prms_in_order:
            this = shape[1][key]
            if isinstance(this, np.ndarray):
                this = str(this)[:4]
            outstring += f"{key[:2]}{this}"
        
        return outstring

    def get_shapes_hash(self):
        """ convert self.Shapes to a frozenset of strings, each string a unique shape based on 
        category and affine params.
        Operates on self.Shapes.
        OUT:
        - frozenset of strings. (frozen allows the output to be hashable)
        --- e.g.,{'circlex-1.7y0.0sx0.7sy0.7ortrsth0.0', 'linex-1.7y1.7sx1.55sy1.55ortrsth1.57'}
        NOTE:
        - raises error if self.Shapes is empty
        """

        assert len(self.Shapes)>0, "old version of task? no shapes"

        return frozenset((self._shape_to_string(sh) for sh in self.Shapes))




    ############ FILTERS
    def filter_by_shapes(self, F, kind):
        """ general purpose, for figuring whheter this task passes filter criteria. 
        supports different ways of formatting F
        INPUT
        - F, different versions:
        --- list of dicts, where each dict has the shape and range for params, where if fall
        within range for all p[arams then consider this a pass. e..g, 
        F = [{"shape":"circle"}]
        - kind, kind of filter.
        """

        if kind=="any_shape_in_range":
            # return True if any shape has features that all fall within ranges for the
            # features designated in F. Any features not designated will be ignored.
            # e.g., F = {"x":[-0, 0.8], "y":[-1.6, -1.5], "sy":[0, 0.5]}
            # NOTE: can pass in list len 0, if want to get exactly that value: e.g.,:
            # F = {"x":[0], "y":[-1.5]}

            def _check_shape(ind, feature, minval, maxval):
                # returns True if shape ind's feature is between minval and maxval.
                # is inclusive.
                if self.Shapes[ind][1][feature]>=minval and self.Shapes[ind][1][feature]<=maxval:
                    return True
                else:
                    return False

            for ind in range(len(self.Shapes)):
                check_list = []
                for feature, valrange in F.items():
                    assert isinstance(valrange, list)
                    if len(valrange)==1:
                        minval = valrange[0]
                        maxval = valrange[0]
                    else:
                        minval = valrange[0]
                        maxval = valrange[1]
                    check_list.append(_check_shape(ind, feature, minval, maxval))
                if all(check_list):
                    # at least one shape (this) has passed. that is enough.
                    return True

            # none of the shapes passed all checks
            return False

        else:
            print(kind)
            print(F)
            assert False, "not coded"

    ############ REPRESENT AS SEQUENCE OF DISCRETE TOKENS
    def get_grid_ver(self):
        """ Each task is either on a grid or off a grid. 
        RETURNS:
        - grid_ver, string, in 
        --- 'on_grid', meaning that all objects are centered on a global grid location
        --- 'on_rel', meaning that all objects are anchored to other obejcts, not to
        global grid
        """

        # Latest version, this is saved as a meta param

        # Older version, infer it from the saved centers.
        # if all centers are aligned with grid centers, then this is on grid.
        centers = self.PlanDat["CentersActual"]

        # Only consider centers for the first prim in each chunk. This is concvention.
        CLC = self.PlanDat["ChunksListClass"]
        # Use the first chunk, by convnetion
        Ch = CLC.ListChunksClass[0]
        print(centers)
        centers = [centers[h[0]] for h in Ch.Hier] # take first stroke in each chunk(hier)
        print(centers)

        grid_x = self.get_grid_params()["grid_x"]
        grid_y = self.get_grid_params()["grid_x"]
        grid_centers = []
        for x in grid_x:
            for y in grid_y:
                grid_centers.append(np.asarray([x, y]))

        if np.all(np.isin(centers, grid_centers)):
            # Then each stroke center matches a grid center
            grid_ver = "on_grid"
        else:
            # Then at least one stroke is off grid.
            grid_ver = "on_rel"

        return grid_ver



    def get_grid_params(self):
        """ What is the grid structure (if any) for this task?
        This is saved now (3/2022) in PlanClass in matlab.
        RETURNS:
        - gridparams, dict holding things like {grid_x, grid_y, ...}, with np array values
        """
        T = self.extract_as('ml2')
        return T.PlanDat["TaskGridClass"]["GridParams"]



    def tokens_generate(self, params, inds_taskstrokes=None, track_order=True):
        """
        Designed for grid tasks, where each prim is a distinct location on grid,
        so "relations" are well-defined based on adjacency and direction
        PARAMS:
        - params, dict, like things defining the grid params for this expt
        - inds_taskstrokes, list of ints, order for the taskstrokes. e..g,, [0,4,2]
        if None, then uses the default order.
        - track_order, bool, whether order is relevant. if True (e.g, for behavior)
        then tracks things related to roder (e.g., sequential relations). If False
        (e.g. for chunks where care only about grouping not order), then ignore those
        features.
        RETURNS:
        - datsegs, list of dicts, each a token.
        """
        from math import pi

        # Extract shapes, formatted correctly
        objects = self.ShapesOldCoords
        if inds_taskstrokes is not None:
            objects = [objects[i] for i in inds_taskstrokes]
        
        # Some grid params for this task
        # - was this on grid?
        grid_ver = self.get_grid_ver()

        # - grid spatial params 
        expt = params["expt"]
        if expt in ["gridlinecircle", "chunkbyshape2"]:
            xgrid = np.linspace(-1.7, 1.7, 3)
            ygrid = np.linspace(-1.7, 1.7, 3)
        else:
            # Look for this saved
            gridparams = self.get_grid_params()
            xgrid = gridparams["grid_x"]
            ygrid = gridparams["grid_y"]
        nver = len(ygrid)
        nhor = len(xgrid)

        # METHODS (doesnt matter if on grid)
        def _orient(i):
            # Angle, orientation
            if np.isclose(objects[i][1]["theta"], 0.):
                return "horiz"
            elif np.isclose(objects[i][1]["theta"], pi):
                return "horiz"
            elif np.isclose(objects[i][1]["theta"], pi/2):
                return "vert"
            elif np.isclose(objects[i][1]["theta"], 3*pi/2):
                return "vert"
            else:
                print(objects[i])
                assert False

        def _shape(i):
            # return string
            return objects[i][0]
        
        def _shape_oriented(i):
            # different name depedning on orientaion
            if _shape(i)=="line" and _orient(i)=="horiz":
                return "hline"
            elif _shape(i)=="line" and _orient(i)=="vert":
                return "vline"
            elif _shape(i)=="circle":
                return "circle"
            else:
                return _shape(i)

        ################## GRID THINGS
        if grid_ver in ["on_grid"]:
            # Sanity check that the hard coded things are correct.
            for o in objects:
                assert np.any(xgrid == o[1]["x"])
                assert np.any(ygrid == o[1]["y"])

            # 1) assign each object a grid location
            locations = []
            for o in objects:
                xloc = o[1]["x"]
                yloc = o[1]["y"]
                xind = int(np.where(xgrid==xloc)[0]) - int((nhor-1)/2)
                yind = int(np.where(ygrid==yloc)[0]) - int((nver-1)/2)
                locations.append((xind, yind))

            def _posdiffs(i, j):
                # return xdiff, ydiff, 
                # in grid units.
                pos1 = locations[i]
                pos2 = locations[j]
                return pos2[0]-pos1[0], pos2[1] - pos1[1]

            def _direction(i, j):
                # only if adjacnet on grid.
                xdiff, ydiff = _posdiffs(i,j)
                if np.isclose(xdiff, 0.) and np.isclose(ydiff, 1.):
                    return "up"
                elif np.isclose(xdiff, 0.) and np.isclose(ydiff, -1.):
                    return "down"
                elif xdiff ==-1. and np.isclose(ydiff, 0.):
                    return "left"
                elif xdiff ==1. and np.isclose(ydiff, 0.):
                    return "right"
                else:
                    return "far"

            def _relation_from_previous(i):
                # relation to previous stroke
                if i==0:
                    return "start"
                else:
                    return _direction(i-1, i)

            def _relation_to_following(i):
                if i==len(objects)-1:
                    return "end"
                else:
                    return _direction(i, i+1)       

            def _horizontal_or_vertical(i, j):
                xdiff, ydiff = _posdiffs(i,j)
                if np.isclose(xdiff, 0.) and np.isclose(ydiff, 0.):
                    assert False, "strokes are on the same grid location, decide what to call this"
                elif np.isclose(xdiff, 0.):
                    return "vertical"
                elif np.isclose(ydiff, 0.):
                    return "horizontal"
                else: # xdiff, ydiff are both non-zero
                    return "diagonal" 

            def _horiz_vert_move_from_previous(i):
                if i==0:
                    return "start"
                else:
                    return _horizontal_or_vertical(i-1, i)

            def _horiz_vert_move_to_following(i):
                if i==len(objects)-1:
                    return "end"
                else:
                    return _horizontal_or_vertical(i, i+1)   
        else:
            assert grid_ver in ["on_rel"]


        # Create sequence of tokens
        datsegs = []
        for i in range(len(objects)):

            # 1) Things that don't depend on grid
            datsegs.append({
                "shape":_shape(i),
                "shape_oriented":_shape_oriented(i),
                })
            
            # 2) Things that depend on grid
            if grid_ver=="on_grid":
                # Then this is on grid, so assign grid locations.
                datsegs[-1]["gridloc"] = locations[i]
                if track_order:
                    datsegs[-1]["rel_from_prev"] = _relation_from_previous(i)
                    datsegs[-1]["rel_to_next"] = _relation_to_following(i)
                    datsegs[-1]["h_v_move_from_prev"] = _horiz_vert_move_from_previous(i)
                    datsegs[-1]["h_v_move_to_next"] = _horiz_vert_move_to_following(i)
            elif grid_ver=="on_rel":
                # Then this is using relations, not spatial grid.
                pass
                # _get_rel_features(i)
                # assert False, 'complete this'
            else:
                print(grid_ver)
                assert False, "code it"

        return datsegs


                        
    ############ PARSER
    def input_parser(self, Parser):
        """ Parser class, input, is fine even if Parser is already done
        """
        self.Parser = Parser

    def input_parses_strokes(self, parses_list):
        """ Input list of parses, where each parse is a strokes
        """
        self.Parses = parses_list


    ############ Extract in different formats
    def extract_as(self, this):
        """ Extract this task in different fromats
        """

        if this=="ml2":
            """ Extract as ml2 task. fails if this is not actually an ml2 task
            """
            if self.Params["input_ver"]=="ml2":
                return self.Params["input_params"]
            else:
                print(self.Params)
                assert False, "this is not an ml2 task"
        else:
            print(this)
            assert False

    ############ SAVING
    def save(self, sdir, suffix):
        """ save all this as pickle
        saves as f"{sdir}/task_{suffix}.pkl"
        """
        import pickle
        
        path = f"{sdir}/task_{suffix}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved to: {path}")





#################### TO WORK WITH LISTS OF TASKS
def getSketchpad(TaskList, kind):
    """ will fail in any Tasks in Tasklist have different sketchpads
    RETURN:
    - np array, 2,2
    """
    tmp = np.stack([task._Sketchpads[kind] for task in TaskList]) # combine all tasks
    tmp = tmp.reshape(tmp.shape[0], -1) 
    assert np.all(np.diff(tmp, axis=0)==0.) # chekc that all are the same.
    spad = TaskList[0]._Sketchpads[kind]
    return spad





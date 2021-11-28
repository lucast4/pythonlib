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

        # Extract Strokes
        if ver=="drawnn":
            # New version of drawnn
            program = params["program"]
            shapes = params["shapes"]
            chunks = params["chunks"]
            self._initialize_drawnn(program, shapes, chunks)

        elif ver=="ml2":
            # Monkeylogic tasks
            taskobj = params
            self._initialize_ml2(taskobj) 
        else:
            print(ver)
            assert False, "not coded"

        # Convert strokes to appropriate coordinate system
        # and generate Points
        if convert_coords_to_abstract:
            self._initialize_convcoords()

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

        taskobj.objects_extract()
        self.ShapesOldCoords = taskobj.Objects
        self.ShapesOldCoords = [[S["obj"], S["tform"]] for S in self.ShapesOldCoords] # conver to general format
        for i in range(len(self.ShapesOldCoords)):
            if self.ShapesOldCoords[i][1] is not None:
                self.ShapesOldCoords[i][1]["theta"] = self.ShapesOldCoords[i][1]["th"]
                del self.ShapesOldCoords[i][1]["th"]

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
        TODO: have not done for self.Program and self.Strokes
        print("TODO (drawmodel-->taskgeneral): convert coords for self.Shapes. Do this in ml2 taskobject")
        """

        in_ = self.Params["input_ver"]
        # out_ = "abstract"

        # print(in_, out_)
        # print(self.Strokes)

        self.Strokes = [self._convertCoords(s, in_, out_) for s in self.StrokesOldCoords]

        # print(self.Strokes)
        # assert False

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
                self._Sketchpads["ml2"] = edges_in = self.Params["input_params"].Task["sketchpad"].T

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
        """

        print("TODO: check out of bounds")


    ########################
    def get_task_id(self):
        """ get unique id for this task
        RETUNRS:
        tuple (category, number in cat)
        """

        if self.Params["input_ver"]=="ml2":
            taskcat = self.Params["input_params"].Task["stage"]
            taskstr = self.Params["input_params"].Task["str"]
            idx = taskstr.find(taskcat)
            tasknum = int(taskstr[idx+len(taskcat)+1:])
        else:
            assert False
        return taskcat, tasknum

    def get_category_setnum(self):
        if self.Params["input_ver"]=="ml2":
            taskcat = self.Params["input_params"].info_name_this_task_category()   
            return taskcat
        else:
            assert False


    def get_number_hash(self, ndigs = 6, include_taskstrings=True, 
        include_taskcat_only=False, compact = False):
        """ Returns number, with digits ndigs, based on 
        coordinates of task. shodl be unique id.
        - include_taskstrings, False, then just number. True, then uses task 
        category, and numberical id (which entered during task creation)
        NOTE: confirmed that the strokes that go in here are identical to strokes in Dataset
        """
        # if include_taskcat_only:
        #     assert include_taskstrings is False, "choose either entier string, or just the cat"
        idnum = self.Params["input_params"].info_generate_unique_name(self.Strokes, 
            nhash=ndigs, include_taskstrings=include_taskstrings, include_taskcat_only=include_taskcat_only)

        if compact:
            # remove long "random" in string
            ind = idnum.find("random")
            if ind>0:
                idnum = idnum[:ind+1] + idnum[ind+6:]

        return idnum

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


                        
    ############ PARSER
    def input_parser(self, Parser):
        """ Parser class, input, is fine even if Parser is already done
        """
        self.Parser = Parser

    def input_parses_strokes(self, parses_list):
        """ Input list of parses, where each parse is a strokes
        """
        self.Parses = parses_list



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





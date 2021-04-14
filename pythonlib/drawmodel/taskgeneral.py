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

    def initialize(self, ver, params, coord_force_same_aspect_ratio=True):
        """ General purpose, to initialize with params for
        this task. Is flexible, takes in many formats, converts
        each into general abstract format
        INPUT:
        - ver, str, in {"drawnn", "ml2"}, indicates what format
        - params, flexible, what format depends on what ver.
        - coord_force_same_aspect_ratio, then for drawnn and ml2 coords,
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
            self._initialize_drawnn(program, shapes)

        elif ver=="ml2":
            # Monkeylogic tasks
            taskobj = params
            self._initialize_ml2(taskobj)
        else:
            print(ver)
            assert False, "not coded"

        # Convert strokes to appropriate coordinate system
        # and generate Points
        self._initialize_convcoords()

        # Check that nothing goes out of bounds
        self._check_out_of_bounds()


    def _initialize_ml2(self, taskobj):
        """ Pass in a task from monkeylogic.
        - taskobj, task class from drawmodel.tasks
        """

        # Extract data
        self.Strokes = taskobj.Strokes

        print("TODO: convert to program")
        self.Program = None
        # self.Program = Told.Task["TaskNew"]["Task"]["program"]
        # self.Name = self.Params[""]

    def _initialize_drawnn(self, program=None, shapes=None):
        """
        To initialize a new task for drawnn training.
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
        """

        a = program is not None
        b = shapes is not None
        assert a!=b, "provide either program or shapes"

        self.ProgramOldCoords = program

        if program is not None:
            self.ShapesOldCoords = self.program2shapes(self.ProgramOldCoords)
        else:
            self.ShapesOldCoords = shapes
        
        self.Strokes = self.shapes2strokes(self.Shapes)

        # NOT YET DONE - general coords
        self.Program = None
        self.Shapes = None


    def _initialize_convcoords(self):
        """ initial conversion of input coordinate system into abstract system
        Modifies: Strokes and Points. 
        TODO: have not done for self.Program and self.Strokes
        """

        in_ = self.Params["input_ver"]
        out_ = "abstract"

        self.Strokes = [self._convertCoords(s, in_, out_) for s in self.Strokes]

        # Convert strokes to points
        self.Points = np.stack([ss for s in self.Strokes for ss in s], axis=0) # flatten Strokes


    def _preprocess(self):

        # Convert strokes to abstract coordinates (to be more general)
        from .image import convertCoordGeneral  
        pass


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
            else:
                self._Sketchpads["ml2"] = None 

                # print("Generated self._Sketchpads to:")
            if self.Params["coord_force_same_aspect_ratio"]:
                from .image import coordsMatchAspectRatio

                edgesgood = self._Sketchpads["abstract"]

                # then makes sure each sketchpad is same aspect ratio as abstracat
                for k, edges in list(self._Sketchpads.items()):
                    if k != "abstract":
                        edgesnew = coordsMatchAspectRatio(edgesgood, edges)
                        self._Sketchpads[f"{k}_orig"] = edges
                        self._Sketchpads[k] = edgesnew



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
        return taskcat, tasknum



    #############################3
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
            --- params order: [x=0, y=0, sx=1, sy=1, th=0, order="trs"]
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





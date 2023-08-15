""" Represents a single primitive, usually a task stroke.
Usually this is a base primitive. Usually corresponds to a single stroke, 
but doesnt have to (i.e. could be a chunk, which is multiple strokes concatenated
into a single new stroke)
3/29/22 - Try to use this for representations of tasks
"""

import numpy as np

class PrimitiveClass(object):
    """
    """

    def __init__(self):
        self.ParamsAbstract = None
        self.ParamsConcrete = None
        self.Stroke = None # strokeclass representation

    def input_prim(self, ver, params, traj=None):
        """ Initialize data, represnting a single primtiive
        PARAMS:
        - traj, optional stroke, np array (N,2 or 3). If input, then creates strokeclass
        TODO:  Give it a hash code (since it is missing prim indices) within PrimitiveClass")
        """


        if ver=="prototype_prim_abstract":
            """ A prototype prim as in matlab ml2 code, where each prim is
            (shape, scale, rotation), where each are abstract categories.
            This doesn't explicitly know the concrete prim params
            - See drawmodel.tasks.planclass_extract_all, where uses planclass prims
            """

            if "shape" in params.keys():
                self.ShapeNotOriented = params["shape"]
            else:
                self.ShapeNotOriented = None

            # self.ParamsAbstract = {
            #     "reflect":int(params["reflect"]),
            #     "scale":int(params["scale"]),
            #     "rotation":int(params["rotation"]),
            # }
            self.ParamsAbstract = {}
            list_keys_abstract = ["reflect", "scale", "rotation"]
            for k in list_keys_abstract:
                if k not in params.keys() or params[k] is None:
                    self.ParamsAbstract[k] = None
                else:
                    self.ParamsAbstract[k] = int(params[k])

            self.ParamsConcrete = {}
            list_keys_concrete = ["x", "y", "theta", "sx", "sy", "order"]
            for k in list_keys_concrete:
                if k not in params.keys() or params[k] is None:
                    self.ParamsConcrete[k] = None
                elif isinstance(params[k], str):
                    self.ParamsConcrete[k] = params[k]
                else:
                    self.ParamsConcrete[k] = np.around(params[k], 3)
            # self.ParamsConcrete = {
            #     "x":np.around(params["x"], 3),
            #     "y":np.around(params["y"], 3)
            # }

            for k in params.keys():
                assert k in ["shape"] + list_keys_concrete + list_keys_abstract, "typo in entry"

            if traj is not None:
                self.strokeclass_generate(traj)

        elif ver=="prototype_prim_concrete":
            """ A prototype prim, but inputing concrete tform variables.
            This used in drawmodel.tasks.objects_extract. Prefer abstract over this,
            since objects_extract has a lot of ad-hoc code for parsing what prim a 
            subprogram corresponds to.
            """
            assert False, "code it, see below"
            # Entry will have params like this
            # T.Objects
            # [{'obj': 'line',
            #   'tform': {'x': array(1.7),
            #    'y': array(1.7),
            #    'th': array(2.35619449),
            #    'sx': array(1.4),
            #    'sy': array(1.4),
            #    'order': 'trs'}},
            #  {'obj': 'L',
            #   'tform': {'x': array(0.1),
            #    'y': array(1.7),
            #    'th': array(6.28318531),
            #    'sx': array(1.9),
            #    'sy': array(1.9),
            #    'order': 'trs'}},
            #  {'obj': 'line',
            #   'tform': {'x': array(1.7),
            #    'y': array(0.1),
            #    'th': array(1.57079633),
            #    'sx': array(1.4),
            #    'sy': array(1.4),
            #    'order': 'trs'}},
            #  {'obj': 'squiggle1',
            #   'tform': {'x': array(0.1),
            #    'y': array(0.1),
            #    'th': array(1.57079633),
            #    'sx': array(1.2),
            #    'sy': array(1.2),
            #    'order': 'trs'}}]
        else:
            print(ver)
            assert False, "code it"

    def print_summary(self):
        """ quickly summarize this prim
        """
        print(f"- Shape: {self.ShapeNotOriented}")
        print("- abstract params:")
        for k, v in self.ParamsAbstract.items():
            print(f"{k}: {v}")
        print("- concrete params:")
        for k, v in self.ParamsConcrete.items():
            print(f"{k}: {v}")

    def shape_oriented(self, include_scale=True):
        """ Returns shape oriented, e.g., line-3-0
        """
        prms = self.ParamsAbstract
        def _sh(x):
            # SHORTHAND
            if x is None:
                return "x"
            else:
                return x
        if include_scale:
            return f"{self.ShapeNotOriented}-{_sh(prms['scale'])}-{_sh(prms['rotation'])}-{_sh(prms['reflect'])}"
        else:
            return f"{self.ShapeNotOriented}-{_sh(prms['rotation'])}-{_sh(prms['reflect'])}"

    def extract_as(self, output="primtuple", include_scale=True):
        """ [Wrapper] Help extract ths primitives in different formats
        PARAMS:
        - output, string name for how to extract
        """

        if output=="primtuple":
            return self._convert_to_primtuple()
        if output=="primtuple_string":
            return self._convert_to_primtuple(as_string=True)
        elif output=="shape":
            """ TaskGeneral.Shape, which is like ["line", {"x":x, "y":y, ...}]
            """
            par = self._extract_params(include_scale=include_scale)
            return [par["shape_rot"], {
                "x":par["cnr_x"],
                "y":par["cnr_y"],
                "sx":par["cnr_sx"],
                "sy":par["cnr_sy"],
                "theta":par["cnr_theta"],
                "order":par["cnr_order"],
                }
                ] # leave the tform as None, ideally extract this from combining (x,y) in plan and sizes/rotation for this primtive
        elif output=="params":
            """ A dict holding relevant params"""
            return self._extract_params()
        elif output=="loc_concrete":
            # 2-tuple, x and y coords.
            par = self._extract_params(include_scale=include_scale)
            return (par["cnr_x"], par["cnr_y"])
        else:
            print(output)
            assert False, "code it"

    def _extract_params(self, include_scale=True):
        """ Extract params for this prim in a dict format
        RETURNS:
        - params, dict, where keys are things like scale, rotation, etc.
        """

        out = {}
        out["shape"] = self.ShapeNotOriented

        for k, v in self.ParamsAbstract.items():
            out[f"abs_{k}"] = v
        for k, v in self.ParamsConcrete.items():
            out[f"cnr_{k}"] = v
        
        # assert False, 'also add reflection here'
        # print("Also add reflection here")
        out["shape_rot"] = self.shape_oriented(include_scale=include_scale)
        
        return out

    def _convert_to_primtuple(self, use_abstract=True, as_string=False):
        """ A hashable represetnation of this prim, 
        either as tuple (defualt) or string (concatenation)
        - is unique id for this prim
        """

        params = self._extract_params()

        if use_abstract:
            scale = params["abs_scale"]
            rot = params["abs_rotation"]
            refl = params["abs_reflect"]
        else:
            scale = params["cnr_scale"]
            rot = params["cnr_theta"]
            refl = params["abs_reflect"]

        # shape x rotation defines prim
        primtuple = (params["shape_rot"], params["shape"], scale, rot, refl, params["cnr_x"], params["cnr_y"])

        if as_string:
            primtuple = "_".join([f"{x}" for x in primtuple])
        return primtuple

    ##################### REPRESENTING AS STROKECLASS
    def strokeclass_generate(self, traj):
        """ Represent this primitive as a concrete stroke using 
        StrokeClass. Allows using all the methods there.
        PARAMS:
        - traj, np.array, (N,2 or 3).
        MODIFIES:
        - self.StrokeClass is replaced.
        """
        from pythonlib.behavior.strokeclass import StrokeClass
        self.Stroke = StrokeClass(traj)






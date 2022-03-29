""" Represents a single primitive.
Usually this is a base primitive. Usually corresponds to a single stroke, 
but doesnt have to (i.e. could be a chunk, which is multiple strokes concatenated
into a single new stroke)
3/29/22 - Try to use this for representations of tasks
"""

class PrimitiveClass(object):
    """
    """

    def __init__(self):
        self.ParamsAbstract = None
        self.ParamsConcrete = None

    def input_prim(self, ver, params):
        """ Initialize data, represnting a single primtiive
        """
        if ver=="prototype_prim_abstract":
            """ A prototype prim as in matlab ml2 code, where each prim is
            (shape, scale, rotation), where each are abstract categories.
            This doesn't explicitly know the concrete prim params
            - See drawmodel.tasks.planclass_extract_all, where uses planclass prims
            """

            self.Shape = params["shape"]
            self.ParamsAbstract = {
                "scale":int(params["scale"]),
                "rotation":int(params["rotation"]),
            }
            self.ParamsConcrete = {
                "x":params["x"],
                "y":params["y"]
            }

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
        print(f"- Shape: {self.Shape}")
        print("- abstract params:")
        for k, v in self.ParamsAbstract.items():
            print(f"{k}: {v}")
        print("- concrete params:")
        for k, v in self.ParamsConcrete.items():
            print(f"{k}: {v}")




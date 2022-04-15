""" Holds a single stroke, usualyl behavior, but could be task as well.
Contrast this with PrimitiveClass, which also holds a single storke (uusally) but is about
a symbolic represnetaiotn, so usually is task.
Here is wrapper for (N,3) np array.
Think of BehaviorClass as holding a sequence of strokeclasses
"""

class StrokeClass(object):
    """
    """
    def __init__(self, stroke=None):
        """
        PARAMS:
        - stroke, (N,3) np array. 
        """
        if stroke is None:
            self.Stroke = None
        else:
            self.input_data("nparray", stroke)

    def __call__(self):
        """ Returns the np array"""
        return self.Stroke

    def input_data(self, ver, data, params=None):
        """ initilaize this stroke.
        PARAMS:
        - ver, string for different methods fo data entyr
        - data, flexible  type, depends on ver.
        - params, dict, holding params.
        """

        if ver=="nparray":
            """ data is (N,3) numpy array, where N is num timestemps, and (x,y,t) are columns
            """
            self.Stroke = data
        else:
            print(ver)
            assert False, "code it"


    ############# STROKE FEATURES
    def extract_single_feature(self, featurename):
        """ Extract this single feature, usually a scalar value.
        Wrapper for various methods written over the years to do this.
        PARAMS:
        - featurename, string name,
        RETURNS:
        - val,
        """

        if featurename=="circularity":
            assert False, "see drawmodel.features"

        assert False, "finish this."
    




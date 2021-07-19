


class Scorer(object):
    
    def __init__(self):

        self.ScoreFunction = None
        self.NormFunction = None
        self.Params = None # dict["score"] = (thetas)
        self._do_score_with_params = False
        self._do_norm_with_params = False

    def input_score_function(self, func):
        """ 
        IN:
        - func, you decide what its arugments
        --- e.g, if Prior, wnat to (strokes, TaskObject=None)
        --- if Likeli, wnat ...
        OUT:
        - assingns to self.ScoreFunction.
        """

        self.ScoreFunction = func

    def input_norm_function(self, func):
        """ 
        Normalizes raw scores. 
        """
        self.NormFunction = func

    def score(self, *args):
        if self._do_score_with_params:
            return self.ScoreFunction(*args, params=self.Params["score"])
        else:
            return self.ScoreFunction(*args)

    def norm(self, scores):
        """ scores is np array (N,) shape
        """
        if self._do_norm_with_params:
            probs = self.NormFunction(scores, params=self.Params["norm"])
        else:
            probs = self.NormFunction(scores)
        return probs

    def score_and_norm(self, *args):
        scores = self.score(*args)
        probs = self.norm(scores)
        return probs
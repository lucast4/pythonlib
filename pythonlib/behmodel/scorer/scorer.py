


class Scorer(object):
    
    def __init__(self):

        self.ScoreFunction = None
        self.NormFunction = None

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
        return self.ScoreFunction(*args)

    def score_and_norm(self, *args):
        scores = self.score(*args)
        return self.NormFunction(scores)

    # def score_task(task):
    #     """
    #     task is taskObject class (general).
    #     it should have at least fields "Strokes" and "Parses", but also anything
    #     else that might be used for scoring prior.
    #     """

    #     assert False, "add a Parser field to TaskObject"

    # def _score_strokes(strokes):
    #     """ directly score these strokes (e.g., a single parse)
    #     """

    #     return self.ScoreFunction(strokes)







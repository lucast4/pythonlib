
assert False, "use scorer inheritence"


class Prior(object):
    
    def __init__(self):

        self.ScoreFunction = None


    def input_score_function(self, func):
        """ 
        IN:
        - func(strokes, TaskObject=None)
        OUT:
        - assingns to self.ScoreFunction.
        """

        self.ScoreFunction = func

    def score_task(task):
        """
        task is taskObject class (general).
        it should have at least fields "Strokes" and "Parses", but also anything
        else that might be used for scoring prior.
        """

        assert False, "add a Parser field to TaskObject"

    def _score_strokes(strokes):
        """ directly score these strokes (e.g., a single parse)
        """

        return self.ScoreFunction(strokes)






    
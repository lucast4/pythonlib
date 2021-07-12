

class BehModel(object):
    """
    """

    def __init__(self):
        """
        """
        self.Params = {}

        self.Prior = None
        self.Likeli = None
        self.Poster = None


        self._prior_scores = None
        self._likeli_scores = None
        self._behavior = None
        self._task = None
        self._posterior_score = None
        
    def input_model_components(self, prior_scorer, likeli_scorer, posterior_scorer):
        """ 
        IN:
        - all scorers are instances of the Scorer class.
        - signatures:
        --- prior_scorer.score(parse_single)
        --- likeli_scorer.score(beh_single, parse_single)
        --- posterior_scorer.score(prior_scores, likeli_scores)
        --- where parse_single and beh_single are strokes types.
        OUT:
        - assigns to fields
        """

        self.Prior = prior_scorer
        self.Likeli = likeli_scorer
        self.Poster = posterior_scorer



    def score_single_trial(self, behavior, task):
        """ scores this behavior on this task, under model
        INPUT:
        - behavior, Behavior() class
        - task, Task() class
        OUT:
        - score, scalar posteiro score.
        """

        self._behavior = behavior
        self._task = task

        parses_list = [p["strokes"] for p in task.Parser.Parses]

        # 1) prior score for all parses
        # parses_list = task.Parses
        # parser = task.Parser
        priors = []
        for parse in parses_list:
            priors.append(self.Prior.score(parse))
        self._prior_scores = priors

        # 2) Likelis (one behavior vs. all parses)
        likelis = []
        strokes_beh = behavior.Strokes
        for parse in parses_list:
            likelis.append(self.Likeli.score(strokes_beh, parse))
        self._likeli_scores = likelis

        # 3) Posterior
        self._posterior_score = self.Poster.score(likelis, priors)
        return self._posterior_score









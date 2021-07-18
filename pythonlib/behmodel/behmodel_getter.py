from .scorer.likeli_functions import *
from .scorer.poster_functions import *
from .scorer.prior_functions import *
from pythonlib.behmodel.behmodel import BehModel
from pythonlib.behmodel.scorer.scorer import Scorer


def models_monkey_vs_self(GROUPING_LEVELS):
    """
    OUT:
    - each D will be modified to have the strokes as fake parses (only for a specific groyping level)
    - returns list of BehModels
    """
    import numpy as np
    from pythonlib.drawmodel.strokedists import distscalarStrokes
    
    list_models = []
    list_model_names = []
    for i, grp in enumerate(GROUPING_LEVELS):
        list_model_names.append(grp)
#         grp = "straight"
        parsesname = f"strokes_beh_group_{grp}"

        # Prior
        Pr = prior_feature_extractor_base()
        def F(D, ind, modelname):
            parsesname = f"strokes_beh_group_{modelname}"
            nparses = len(D.Dat.iloc[ind][parsesname])
            return np.ones(nparses)    
        Pr.input_score_function(F)

        # Likeli
        def F(D, ind, modelname):
            strokes_beh = D.Dat.iloc[ind]["strokes_beh"]
            parsesname = f"strokes_beh_group_{modelname}"
            list_of_parsestrokes = D.Dat.iloc[ind][parsesname]
            scores = np.array([distscalarStrokes(strokes_beh, strokes_parse, "dtw_segments") for strokes_parse in list_of_parsestrokes])
            return 1/scores
        Li = likeli_dataset()
        Li.input_score_function(F)

        # Post
        Po = poster_dataset()

        # Model
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po)
        
        # need to know this to pass in correct args
        BM._list_input_args = ["dat", "trial", "modelname"]

        # activate names
        BM.unique_id(grp)

        list_models.append(BM)

        
    return list_models, list_model_names


def quick_getter(modelclass, vectorized_priors=False):
    """ For this modelclass (string) returns
    list of beh models (BMs).
    Sort of hard coded currently, just for testing)
    """

    if modelclass=="lines5":

        # Generate models
        list_mod = []
        list_modnames = []
        for rule_model in ["straight", "bent"]:

            # Prior
            if vectorized_priors:
                Pr = prior_scorer_quick(ver="lines5", input_parsesflat=True)
            else:
                Pr = prior_scorer_quick(ver="lines5", input_parsesflat=False)

            # Likeli
            Li = likeli_scorer_quick(ver="base")

            # Post
            Po = poster_dataset()

            # Model
            BM = BehModel()
            BM.input_model_components(Pr, Li, Po)
            
            # BM._list_input_args_likeli = ("dat", "trial", "modelname")
            # BM._list_input_args_prior = ("parsesflat")
            BM._list_input_args_likeli = ("dat", "trial", "modelname")
            
            if vectorized_priors:
                BM._list_input_args_prior = ("parsesflat", "trialcode", "modelname")
            else:
                BM._list_input_args_prior = ("dat", "trial", "modelname")

            list_mod.append(BM)
            list_modnames.append(rule_model)
    else:
        assert False

    return list_mod, list_modnames

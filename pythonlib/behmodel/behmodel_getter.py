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


from .scorer.likeli_functions import *
from .scorer.poster_functions import *
from .scorer.prior_functions import *
from pythonlib.behmodel.behmodel import BehModel
from pythonlib.behmodel.scorer.scorer import Scorer



def quick_getter(modelclass, vectorized_priors=False):
    """ For this modelclass (string) returns
    list of beh models (BMs).
    Sort of hard coded currently, just for testing)
    """
    assert False, "obsolete, see below."

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


def get_single_model(ver, params=None):
    """ GOOD , get single BM.
    given ver and params, should always give same object
    """
    from pythonlib.behmodel.scorer.prior_functions import prior_function_database
    from pythonlib.behmodel.scorer.likeli_functions import likeli_function_database

    if ver=="lines5":
        Pr = prior_function_database("lines5")        
        Li = likeli_function_database("lines5")
        Po = poster_dataset(ver="logsumexp")
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,
            list_input_args_likeli = ("dat", "trial"),
            list_input_args_prior = ("parsesflat", "trialcode"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True)
        BM.Prior.Params = {
            "norm":(1.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }
    elif ver=="mkvsmk":
        Pr = prior_function_database("monkey_vs_monkey")
        Li = likeli_function_database("monkey_vs_monkey")
        Po = poster_dataset(ver="logsumexp")
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,             
            list_input_args_likeli = ("dat", "trial", "modelname"),
            list_input_args_prior = ("dat", "trial", "modelname"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True)
        BM.Prior.Params = {
            "norm":(1.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }
    else:
        assert False

    return BM



def quick_getter_with_params(modelclass, list_mrules):
    """ For this modelclass (string) returns
    list of beh models (BMs).
    Sort of hard coded currently, just for testing)
    NOTE:
    - with_params, means params for prior are exposed, to allow fitting.
    """
    list_mod = []
    list_modnames = []

    parsers_to_flatten = ['parser_graphmod', 'parser_nographmod']

    if modelclass=="lines5":
        # list_mrules = ["straight", "bent"]
        for rule_model in list_mrules:
            BM = get_single_model("lines5")
            list_mod.append(BM)
            list_modnames.append(rule_model)
        allow_separate_likelis = False
    elif modelclass=="mkvsmk":
        # list_mrules = ["straight", "bent"]
        for i, grp in enumerate(list_mrules):
            BM = get_single_model("mkvsmk")
            list_mod.append(BM)
            list_modnames.append(grp)
        allow_separate_likelis = True
    else:
        assert False

    kwargs = {
        "allow_separate_likelis":allow_separate_likelis,
        "parsers_to_flatten":parsers_to_flatten
    }
    return list_mod, list_modnames, kwargs

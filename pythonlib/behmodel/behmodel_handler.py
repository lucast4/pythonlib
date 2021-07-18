import numpy as np

class BehModelHandler(object):


    def __init__(self):
        """
        """

        # self.Likelis = None # hold likeli for each parse for each trial
        self.PriorProbs = None
        self.Posteriors = None

    def input_data(self, dataset, list_models, 
        list_model_ids=None, allow_separate_likelis=False,
        parsers_to_flatten = ['parser_graphmod', 'parser_nographmod']):
        """
        IN:
        - dataset, a single D object
        - list_models, list of BehModel objects
        - list_model_ids, list of strings. if dont enter, then will generate
        auto based on timestamps.
        - allow_separate_likelis, then issue is will take longer if they dont share.
        - parsers_to_flatten, by default will flatten these parsers into list of parses, 
        where ach parse is a list of walks. This will go into a new column: "parses_behmod"
        OUT:
        - assigns to self.Dat, self.ListModels
        NOTE:
        - Requirements:
        --- Assumes all models use the same likeli scoring function (so can share likelis)
        --- Assume all models applied to the same one dataset.
        --- Otherwise models hve different priors and posetiro scorers.
        """

        self._parsers_to_flatten = parsers_to_flatten
        self._allow_separate_likelis = allow_separate_likelis

        self.D = dataset
        self.ListModels = list_models

        if list_model_ids is not None:
            assert len(list_model_ids)==len(list_models)
            for M, name in zip(self.ListModels, list_model_ids):
                M.unique_id(set_id_to_this=name)
        else:
            list_model_ids = []
            for M in self.ListModels:
                list_model_ids.append(M.unique_id(ver="random_num"))

        self.ListModelsIDs = list_model_ids

        # dict from id to model
        self.DictModels = {name:model for name, model in zip(self.ListModelsIDs, self.ListModels)}

        assert len(self.DictModels)==len(self.ListModels), "probably keys are same."

        if self._allow_separate_likelis:
            self.Likelis = {name:[] for name in self.DictModels.keys()} # list of likelsi.
        else:
            self.Likelis = []  # model --> list of likelis

        # Preprocess
        self.preprocess()

    def preprocess(self):
        """
        """
        # Flatten parsers
        self.D.parser_flatten(self._parsers_to_flatten)

        if self._allow_separate_likelis==False:
            # self.Likeli = self.ListModels[0].Likeli # asssuems all use same likeli
            if len(self.ListModels)>1:
                # Sanity check, make sure likelis do same thing
                list_likelis = [] # one for each model
                for name, Mod in self.DictModels.items():
                    list_likelis.append(self._score_helper("likeli", name, 0))
                    # list_likelis.append(Mod.Likeli.score(self.D, 0))

                nparse = len(list_likelis[0])
                for n in range(nparse):
                    a = [likelis[n] for likelis in list_likelis] # (x1,x2,x3) if 3 models, on this parse for first trial.
                    for aa in a[1:]:
                        if not np.isclose(aa,a[0]):
                            print(a)
                            assert False, "model likeli functions are different. just make separate BehModelHandlers if this is the case, or set allow_separate_likelis True"



    def _score_helper(self, scorever, modelname, i):
        """
        - General purpose for extracting list of scores (priors or likelis) for this trial(i)
        this model (modelname, string)
        NOTE:
        - this useful becuase figures out what arguments to pass in (based on params I set
        when initializing)
        """
        Mod = self.DictModels[modelname]

        # what args this needs?

        def _get_args(list_of_argstrings):
            args = None
            if list_of_argstrings==("dat","trial"):
                args = (self.D, i)
            elif list_of_argstrings == ("dat", "trial", "modelname"):
                    args = (self.D, i, modelname)
            elif list_of_argstrings == ("parsesflat"):
                assert False, "if this, then shouldnt pass in args one by one... should use applyFuncToAllRows"
                args = (self.D, i, modelname)
            else:
                assert False

            if args is None:
                print(list_of_argstrings)
                assert False
            return args


        if scorever=="prior":
            args = _get_args(Mod._list_input_args_prior)
            return Mod.Prior.score_and_norm(*args)
        elif scorever=="likeli":
            args = _get_args(Mod._list_input_args_likeli)
            return Mod.Likeli.score(*args)
        else:
            print(scorever)
            assert False, "not coded"


    ##### PROCESS FOR SCORING
    def compute_store_likelis(self, force_run=False):
        """ doesnt care about model. just compute likelis and save them, for each
        line in datsaet.
        NOTE;
        - uses the likeli scorer for the first model, since assumes all models use same likeli.
        """

        def _get_list_likelis(modname):
            list_likelis = []
            for i in range(len(self.D.Dat)):
                if i%50==0:
                    print("likeli", i)
                likelis = self._score_helper("likeli", modname, i)
                # likelis = Mod.Likeli.score(self.D, i)
                list_likelis.append(likelis)
                # self.Likelis.append(likelis)
            return list_likelis

        if self._allow_separate_likelis==False:
            if len(self.Likelis)==0 or force_run==True:
                # use the first
                name = self.ListModelsIDs[0]
                self.Likelis = _get_list_likelis(name)
            else:
                print("skipping compute likelis since done")
        else:
            for name in self.DictModels.keys():
                if len(self.Likelis[name])==0 or force_run==True:
                    self.Likelis[name] = _get_list_likelis(name)
                else:
                    print("skipping compute likelis since done, mod= ", name)


    def compute_store_priorprobs(self, force_run=False): 
        """
        - force_run, then redoes, even if already donea nd saved.
        OUT:
        - svaes into : self.PriorProbs[modelid][trial]
        """

        if self.PriorProbs is not None and force_run==False:
            print("skipping prior compute, since already done")
            return

        self.PriorProbs = {}
        for name, Mod in self.DictModels.items():
            self.PriorProbs[name] = []
            for indtrial in range(len(self.D.Dat)):
                if indtrial%50==0:
                    print("priors", name, indtrial)
                probs = self._score_helper("prior", name, indtrial)
                # probs = Mod.Prior.score_and_norm(self.D, indtrial)
                self.PriorProbs[name].append(probs)

    def compute_store_priorprobs_vectorized(self):
        """ 
        - Assumes that prior scorer takes in list of parses as an argument.
        this way can vectorize computation over all rows of Dat
        NOTE:
        This seems to be about 4-5x faster than compute_store_priorprobs, on datsaet
        with about 250 rows (trials) and about 20-100 parses each.
        Tested on Red - lines5 - straight.
        See notebook: devo_taskmodel_finalized_071121
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        if self.PriorProbs is not None and force_run==False:
            print("skipping prior compute, since already done")
            return

        self.PriorProbs = {}
        for name, Mod in self.DictModels.items():
            print("probs, getting vectorized across all trials", name)
            def F(x):
                if Mod._list_input_args_prior==("parsesflat", "trialcode", "modelname"):
                    args = (x["parses_behmod"], x["trialcode"], name)
                else:
                    print(Mod._list_input_args_prior)
                    assert False
                probs = Mod.Prior.score_and_norm(*args)
                return probs
            dfthis = applyFunctionToAllRows(self.D.Dat, F, "priorprobs")
            self.PriorProbs[name] = dfthis["priorprobs"].to_list()


    def compute_store_posteriors(self):
        """
        OUT:
        - svaes into : self.Posteriors[modelid][trial]
        """

        self.Posteriors = {}
        for name, Mod in self.DictModels.items():
            self.Posteriors[name] = []
            for indtrial in range(len(self.D.Dat)):
                if indtrial%50==0:
                    print("posterior", name, indtrial)

                likelis = self._get_list_likelis(name, indtrial)
                probs = self.PriorProbs[name][indtrial]

                post = Mod.Poster.score(likelis, probs)
                self.Posteriors[name].append(post)


    ##### GETTERS
    def _get_list_likelis(self, modname, indtrial):
        if self._allow_separate_likelis:
            return self.Likelis[modname][indtrial]
        else:
            return self.Likelis[indtrial]


    ##### PLOTTING


    ##### POST-PROCESS
    def results_to_dataset(self, suffix=""):
        """ assign results (posterior scores, one for each row) 
        back into the dataset.
        will make new column called:
        "modelpost_{modename}", for each model
        """

        list_col_names = []
        for name, scores in self.Posteriors.items():
            assert len(self.D.Dat)==len(scores)
            newcol = f"behmodpost_{name}_{suffix}"

            self.D.Dat[newcol] = scores
            list_col_names.append(newcol)
            d = self.D.identifier_string()
            print(f"scores into Dat: {d} , col:", newcol)

        return list_col_names




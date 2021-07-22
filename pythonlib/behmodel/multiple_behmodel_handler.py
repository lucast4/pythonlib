
from .behmodel_handler import *

class MultBehModelHandler(object):
    """ holds multipel H (behmodel handlers).
    - Ideally each H represents a single model class, with potentially multiple
    individual model rules (all applied to the same dataset)
    - Across H, could apply to same or different dataset. But I envision this applying to same,
    where this single dataset includes all "test" datasets concatenated. I..e, the same task 
    can be present across multipel epochs in this datsaet.
    - The result is each H is (beh epoch) x (model rule, for a single class), and 
    each H is a different model class.
    - Ideally this is more analysis and plotting, not for training models.
    """

    def __init__(self):
        
        self.DictTrainedH = {}

    def _load_pretrained_models(self, SDIR, list_mclass, list_mrule):
        """ 
        """

        train_or_test = "train"

        ## First, reload pre-saved models
        DictH = {} # mclass, mrule
        for model_class in list_mclass:
            for mrule in list_mrule:
                ListBMH, list_dsets, ListH = bmh_load(SDIR, model_class, [mrule], train_or_test)
                # Convert ListBMH to DictBMH
                for L in ListBMH:
                    model_rule = L["id_mod"]
                    key = (model_class, model_rule)
                    DictH[key] = L["H"]
        self.DictTrainedH = DictH

        # save
        self.ListMclass = list_mclass
        self.ListMrule = {}
        self.DictMrule = {}
        for mclass in self.ListMclass:
            list_rules = [x[1] for x in self.DictTrainedH.keys() if x[0]==mclass]
            self.DictMrule[mclass] = list_rules

        print("** OUTCOME:")
        print("classes:", self.ListMclass)
        print("rules:", self.DictMrule)
        print("extracted trained models: ", self.DictTrainedH)


    def apply_models_to_single_new_dataset(self, D):
        """ applies pretrained models to new dataset D.
        IN: 
        - D, a single dataset. holding multiple epochs for exampkle, if want to apply all models to all epochs data
        OUT:
        - Modifies self, to add self.DictTestH_SameDset
        """

        DlistThis = [D]
        self.DictTestH_SameDset = {}
        for mclass in self.ListMclass:
            list_mrule = self.DictMrule[mclass]
            ListBMH, list_dsets, ListH= cross_dataset_model_wrapper_params(DlistThis, mclass, list_mrule)
            assert len(ListH)==1
            H = ListH[0]

            # 2) For each mrule, update with trained params.
            # get statedict from trained model
            for mrule in list_mrule:
                print("*** APPLYING TRAINED PARAMS FOR: ", (mclass, mrule))
                state_trained = self.DictTrainedH[(mclass, mrule)].extract_state()
                H.params_prior_set(mrule, state_trained["prior_params"][mrule])

                # 3) Evaluate
                H.compute_all(mode="test")
                
            # 4) assign to D
            H.results_to_dataset(mclass)

            # 5) save this H
            self.DictTestH_SameDset[mclass] = H

        self.Dtest_SameDset = D


    def apply_models_to_mult_new_dataset(self, Dlist):
        """ applies pretrained models to multiple new dataset D.
        IN: 
        - D, a single dataset. holding multiple epochs for exampkle, if want to apply all models to all epochs data
        OUT:
        - Modifies self, to add self.DictTestH_SameDset
        """
        self.DictTestDH = {}
        self.DictTestD = {}

        for D in Dlist:
            d_id = D.identifier_string()
            DlistThis = [D]
            for mclass in self.ListMclass:
                list_mrule = self.DictMrule[mclass]
                print(DlistThis)
                print(mclass)
                print(list_mrule)
                ListBMH, list_dsets, ListH= cross_dataset_model_wrapper_params(DlistThis, 
                    mclass, list_mrule)
                assert len(ListH)==1
                H = ListH[0]

                # 2) For each mrule, update with trained params.
                # get statedict from trained model
                for mrule in list_mrule:
                    print("*** APPLYING TRAINED PARAMS FOR: ", (mclass, mrule))
                    state_trained = self.DictTrainedH[(mclass, mrule)].extract_state()
                    H.params_prior_set(mrule, state_trained["prior_params"][mrule])

                    # 3) Evaluate
                    H.compute_all(mode="test")
                    
                # 4) assign to D
                H.results_to_dataset(mclass)

                # 5) save this H
                self.DictTestDH[(d_id, mclass)] = {
                    "D":D,
                    "H":H
                }
            self.DictTestD[d_id] = D

        self.initialize_colnames()

        # get difference of scores
        self.dataset_get_diffs()


    def initialize_colnames(self):
        """ assumes that each class has same set of rules..
        """
        from pythonlib.dataset.beh_model_comparison import ColNames
        CN = ColNames(self.ListMclass, self.DictMrule[self.ListMclass[0]])
        self.ColNames = CN


    def dataset_get_diffs(self):
        """ get all score diffs"
        """
        for D in self.list_D():
            for mclass in self.ListMclass:
                coldiff = self.ColNames.colnames_minus_usingnames(mclass)
                col1 = self.ColNames.colnames_score(mclass)[1]
                col0 = self.ColNames.colnames_score(mclass)[0]
                D.Dat[coldiff] = D.Dat[col1] - D.Dat[col0]
                
                print(coldiff, "from ", col1, "minus", col0)



    ##### GETTERS

    def list_H(self, mclass=None):
        """ return list of H
        """
        if mclass is not None:
            return [v["H"] for k, v in self.DictTestDH.items() if k[1]==mclass]
        else:
            return [v["H"] for v in self.DictTestDH.values()]

    def list_D(self):
        """ return unique list of D
        """
        return list(set([v["D"] for v in self.DictTestDH.values()]))



    #### EXTRACT
    def extract_concatenated_dataset(self):
        """ 
        Returns single D
        """
        from pythonlib.dataset.dataset import concatDatasets

        Dlist = [D for D in self.DictTestD.values()]
        return concatDatasets(Dlist)

    #### PRING
    def print_pretrained_models(self):
        assert False, "not checked"
        for k, H in DictH.items():
            print("---")
            print(k)
            print(H.params_prior(k[1]))
        #     print(k, v.D.Dat["task_stagecategory"].value_counts())


if __name__=="__main__":
    from pythonlib.behmodel.score_dataset import score_dataset
    # from pythonlib.behmodel.scorer.prior_functions import prior_feature_extractor
    from pythonlib.behmodel.scorer.likeli_functions import likeli_dataset
    from pythonlib.behmodel.scorer.poster_functions import poster_dataset
    from pythonlib.behmodel.behmodel import BehModel
    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.dataset.dataset import Dataset
    from pythonlib.dataset.dataset import concatDatasets
    from pythonlib.dataset.analy import preprocessDat
    import numpy as np
    import matplotlib.pyplot as plt
    from pythonlib.dataset.beh_model_comparison import plots_cross_prior_and_model_combined, ColNames
    from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper_params
    from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper, bmh_optimize_single, bmh_results_to_dataset, cross_dataset_model_wrapper_params, bmh_save, bmh_load
    from pythonlib.dataset.analy import extract_strokes_monkey_vs_self


    # Load datasets
    SDIR = "/data2/analyses/main/model_comp/planner"

    SUBSAMPLE = False
    animal = "Red"
    expt = "lines5"
    rule_list = ["straight", "bent"]
    LOAD_ALL_PARSES = True
    ONLY_SUMMARY_DATES = True

    # LOAD ALL DATASETS
    Dlist  = []
    for rule in rule_list:
        D = Dataset([])
        D.load_dataset_helper(animal, expt, rule=rule)
        
        if SUBSAMPLE:
            D.subsampleTrials(1,10)
            
        D.load_tasks_helper()
        
        D.filterPandas({"random_task":[False], "insummarydates":[True]}, "modify")
        
        if LOAD_ALL_PARSES:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":"fixed_True"},
                {"quick":True, "ver":"nographmod", "savenote":"fixed_True"}]
            list_suffixes = ["graphmod", "nographmod"]
        else:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":"fixed_True"}]
            list_suffixes = ["graphmod"]
        D.parser_load_presaved_parses(list_parse_params, list_suffixes)
        Dlist.append(D)
            
            
            
    # Preprocess
    D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(Dlist[0], expt)


    ### SPLIT DATASET
    DictD = {}

    for D, rule in zip(Dlist, rule_list):

        inds_train, inds_val, inds_test = D.splitTrainTestMonkey(expt, epoch=rule)
        
        DictD[(rule, "train")] = D.subsetDataset(inds_train)
        DictD[(rule, "test")] = D.subsetDataset(inds_test)

        print("**", (rule, "train"), "this many:", len(inds_train))
        print("**", (rule, "test"), "this many:", len(inds_test))


    from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper, bmh_optimize_single, bmh_results_to_dataset, cross_dataset_model_wrapper_params, bmh_save, bmh_load
    import os

    from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
    dset = f"{animal}-{expt}"
    ts = makeTimeStamp()
    SDIR = f"/data2/analyses/main/model_comp/planner/pilot-{ts}/{dset}"
    os.makedirs(SDIR, exist_ok=True)

    # Prepare strokes (other monkey) in dataset
    extract_strokes_monkey_vs_self(Dlist, GROUPING, GROUPING_LEVELS)

    GROUPING_LEVELS = rule_list
    # model_class = "mkvsmk"
    # list_model_class = ["mix_features_bdn", "mix_features_bd", "mix_features_bn", "mix_features_dn", 
    #                     "mix_features_b", "mix_features_d", "mix_features_n", "mkvsmk", "lines5"]
    # list_model_class = ["mix_features_bd", "mix_features_dn", "mix_features_d", "mkvsmk"]
    list_model_class = ["mix_features_dn", "mix_features_d", "mkvsmk"]
    for model_class in list_model_class:
        for D, rule in zip(Dlist, rule_list):


            # Get Dataset
            Dlist_this = [DictD[(rule, "train")]]
            GROUPING_LEVELS_this = [rule]
            ListBMH, list_dsets, ListH= cross_dataset_model_wrapper_params(Dlist_this, model_class, GROUPING_LEVELS_this)

            # Optimize
            if model_class=="mkvsmk":
                for H in ListH:
                    H.compute_store_priorprobs_vectorized()
                    H.compute_store_likelis()
                    H.compute_store_likelis_logprobs()
                    H.compute_store_posteriors()
    #         if model_class=="lines5":
            else:
                bmh_optimize_single(ListBMH, D.identifier_string(),  rule)
        

            ### Save
            bmh_save(SDIR, Dlist_this, model_class, GROUPING_LEVELS_this, ListH, "train")

            bmh_results_to_dataset(ListBMH, suffix=model_class)
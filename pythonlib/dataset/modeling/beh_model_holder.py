"""
Abstract class to hold results from modeling, for any kiund of modeling where a set of 
models are compared against a set of behaviloral data, yeulkding scalar scores for 
each trial (across different models).

See jupyter notebook for development of this code:
- 221219_analy_multiplemodels_general


"""
import numpy as np
import matplotlib.pyplot as plt

class BehModelHolder(object):
    """
    """
    def __init__(self, data, dict_modelclass_to_rules, input_ver="default"):
        """
        Initialize with a dataframe in default format, holding model-beh scores,
        with different scorers(models). 
        PARAMS:
        - data, wide-form dataframe. each row is a single trial. Must have the following
        columns:
        --- epoch, str, the actual rule for monkey. This usually matches at least one of the rules in
        scorenames
        --- character, str, the name (id) of the task. 
        --- score_names [multiple cols each a score using a diff model] behmodpost_{rulename}_{modelclass},
        where rulename is string name of the rule for that model, at least one of the
        coluns (for ach class) should match the rule of the epoch. modelclass is higher level
        class of the model, e..g, different hyperparametres.
        --- trialcode (optional), trialcode, to help link back to Dataset
        - dict_modelclass_to_rules, dict mapping from each class to a list of rules existing for
        that class.
        --- taskgroup (optional), value that identified how to group tasks, e.g, could mean 
        different kinds of proibe tasks. this is flexible.
        """

        if input_ver=="default":
            self.Dat = data
        elif input_ver=="long_form":
            self._input_data_long_form(data)
        elif input_ver=="mbh":
            self._input_data_from_multiple_behmodel_handler(data)
        else:
            print(input_ver)
            assert False

        if dict_modelclass_to_rules is None:
            assert False, "write code to autoamtically extract"
        else:
            self.DictMclassToRules = dict_modelclass_to_rules

        # Sanituy check of data
        self._preprocess_sanity_check()

        ### ALIGNMENT - rank, compute all
        self.analy_compute_alignment_wrapper()

        ## Get colnames
        self.colnames_extract_alignment()


    def _preprocess_sanity_check(self):
        print("TODO! _preprocess_sanity_check")

    def _input_data_long_form(self, data):
        """ Help to melt from long-form to necessary wide form 
        """
        assert False, "code it"

    def _input_data_from_multiple_behmodel_handler(self, data):
        """ Input modeling using continuos parses, stored in a 
        MBH object. 
        e.g., Used this for gridlinecircle, working with continuous parses
        """
        assert False, "code it"

    def _initialize_colnames(self):
        """ assumes that each class has same set of rules..
        """
        from pythonlib.dataset.beh_model_comparison import ColNames
        list_mclass = list(self.DictMclassToRules.keys())
        CN = ColNames(list_mclass, self.DictMclassToRules[self.ListMclass[0]])
        self.ColNames = CN

    def colnames_extract_scores(self, list_mclass_get=None, list_rule_get=None):

        list_colname = []
        for mclass, list_rule in self.DictMclassToRules.items():
            if list_mclass_get is not None:
                if mclass not in list_mclass_get:
                    continue
            for rule in list_rule:
                if list_rule_get is not None:
                    if rule not in list_rule_get:
                        continue
                colname = f"behmodpost_{rule}_{mclass}"
                list_colname.append(colname)
        return list_colname


    def colnames_extract_alignment(self, alignmnet_ver="diff", list_mclass_get=None):

        list_colname = []
        for mclass, list_rule in self.DictMclassToRules.items():
            if list_mclass_get is not None:
                if mclass not in list_mclass_get:
                    continue
            colname = f"alignment_{alignmnet_ver}_{mclass}"
            list_colname.append(colname)
        return list_colname


    def extract_concatenated_aggregated_dataset(self, Dat=None, monkey_prior_col_name="epoch", 
        monkey_prior_list=None, list_classes=None, model_score_name_list =None,
        list_cols_keep = None):
        """ 
        Returns single D, but also returns variations after different
        kinds of aggregations.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows, filterPandas, aggregGeneral, summarize_feature

        # Get Single dataset
        # D = self.extract_concatenated_dataset()
        if Dat is None:
            Dat = self.Dat

        # Get list of monkye priors
        if monkey_prior_list is None:
            monkey_prior_list = sorted(Dat[monkey_prior_col_name].unique().tolist())


        # Get all scores
        if model_score_name_list is None:
            model_score_name_list = self.colnames_extract_scores()
            # # Then get it 
            # if list_classes is not None:
            #     model_score_name_list = []
            #     for c in list_classes:
            #         model_score_name_list.extend(self.ColNames.colnames_score(c))
            # else:
            #     list_classes = self.ColNames.Classes
            #     model_score_name_list = self.ColNames.colnames_score()

        # get list of alignments column names
        list_colnames_alignment = [col for col in Dat.columns if "alignment_" in col]

        print(monkey_prior_list)
        print("Monkey priors (datasets): ", monkey_prior_list)
        print("Model scores (colnames): ", model_score_name_list)
        print("Model alignments (colnames): ", list_colnames_alignment)

        LIST_COLS_KEEP =  ["character", "trialcode", "taskgroup", "probe"]
        if list_cols_keep:
            LIST_COLS_KEEP = LIST_COLS_KEEP + list_cols_keep

        # 1) wide-form to long-form (a single column of "scores") (inlcude alignments)
        
        DatWide, _ = summarize_feature(Dat, "epoch", model_score_name_list+list_colnames_alignment, 
            LIST_COLS_KEEP, newcol_variable="model", newcol_value="score")

        # 1) wide-form to long-form (a single column of "scores")
        _, DatFlat = summarize_feature(Dat, "epoch", model_score_name_list, LIST_COLS_KEEP, 
            newcol_variable="model", newcol_value="score")

        # 1) wide-form to long-form (a single column of "scores") (inlcude alignments)
        if len(list_colnames_alignment)>0:
            _, DatFlatAlignment = summarize_feature(Dat, "epoch", list_colnames_alignment, LIST_COLS_KEEP, 
                newcol_variable="model", newcol_value="score")
            # 3) Long, agg over tasks
            DatFlatAlignmentAgg = aggregGeneral(DatFlatAlignment, group = ["character", monkey_prior_col_name, "model"], 
                                       values=["score"])
        else:
            DatFlatAlignment = None
            DatFlatAlignmentAgg = None

        # 2) Wide, agg over tasks
        DatThisAgg = aggregGeneral(Dat, group = ["character", monkey_prior_col_name], 
                                   values=model_score_name_list)
        
        # 3) Long, agg over tasks
        DatFlatAgg = aggregGeneral(DatFlat, group = ["character", monkey_prior_col_name, "model"], 
                                   values=["score"])

        return Dat, DatWide, DatFlat, DatThisAgg, DatFlatAgg, DatFlatAlignment, DatFlatAlignmentAgg

    def analy_compute_alignment_wrapper(self, ploton=False, plot_alignment_ver="diff"):
        """ All ways of computing alignemnt:
        In general, for each trial, how well does the model
        with same rule for that trial score beh, compared to all
        altenraitve models?
        NOTE: see dataset.analy.score_alignment. There was only 2x2
        INPUT:
        - ploton, plots alignment in rank order, and also prints out the trial inds so that 
        you can inspect each trial by hand if you want.
        - plot_alignment_ver, which alignmetn score to use for plotting
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def _alignment(x, mclass, ver="rank"):
            """ Compute different versions of alignemnt, for this row
            Only gets within all rules(models) for this mclass
            """
            rthis = x["epoch"]

            # print("mclass",mclass)
            # print("rthis",rthis)
            list_names = self.colnames_extract_scores([mclass], [rthis])
            
            # this means that epoch doesn't match any rules
            if not list_names:
                return np.nan

            colname_this = list_names[0]

            list_rules_this = [r for r in self.DictMclassToRules[mclass] if not r==rthis]
            colname_others = self.colnames_extract_scores([mclass], list_rules_this)

            score_this = x[colname_this]
            score_others = [x[c] for c in colname_others]

            if ver=="rank":
                # 0, 1, .., best to worst
                rank = sum(np.array(score_others)>score_this)
                return rank
            elif ver=="this_minus_meanothers":
                # more positive the better.
                return score_this - np.mean(score_others)
            elif ver=="diffindex":
                # (this - other)/(this + other)
                tmp = np.mean(score_others)
                return (score_this - tmp)/(score_this + tmp)
            else:
                assert False

        # For each row, compute alignment in various ways.
        # 1) Alignment rank
        for mclass, list_rules in self.DictMclassToRules.items():
            colthis = f"alignment_rank_{mclass}"
            self.Dat = applyFunctionToAllRows(self.Dat, lambda x: _alignment(x, mclass, "rank"), colthis)
            print("DONE", mclass, colthis)
            
            # 2) Alignment (score diff from mean of others)
            colthis = f"alignment_diff_{mclass}"
            self.Dat = applyFunctionToAllRows(self.Dat, lambda x: _alignment(x, mclass, "this_minus_meanothers"), colthis)
            print("DONE", mclass, colthis)
            
            # 3) Alignment - score index?
            colthis = f"alignment_diffindex_{mclass}"
            self.Dat = applyFunctionToAllRows(self.Dat, lambda x: _alignment(x, mclass, "diffindex"), colthis)
            print("DONE", mclass, colthis)

        # For each trial, alignment, as rank out of all model scores.
        if False:
            sns.catplot(data=DatWide, x="epoch", y=colthis)
            sns.catplot(data=DatWide, x="epoch", y=colthis, kind="box")    

        if ploton:
            # Plot alignment across all trials in a sorted fasion. 

            ncols = 2
            nrows = int(np.ceil(len(self.DictTestDH)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*7, nrows*4))

            ct = 0
            for dset, mclass, D, H, _ in self.iter_test_dat():
                col = f"alignment_{plot_alignment_ver}_{mclass}"
                print(D.identifier_string(), ' -- ', mclass)

                # sort dataframe by a column
                dfthis = D.Dat.sort_values(col)
                inds_sorted_best2worst = dfthis.index.tolist()
                alignments = dfthis[col].tolist()
                
                print(f"dataset: {dset}, indstrials, low to high value, based on {col}")
                print(inds_sorted_best2worst)
            #     (alignments)
                
                ax = axes.flatten()[ct]
                ax.plot(np.arange(len(alignments)), alignments)
            #     ax.plot(inds_sorted_best2worst, alignments)
                ax.set_ylabel(col)
                ax.set_title((dset, mclass))
                ax.set_xlabel('rank order')
                ax.axhline(0)
                ct+=1        


    def plotwrapper_overview_all(self):
        """ Overview of all beh epochs and models
        datapt at both level of trials and characters
        """

        # from pythonlib.dataset.modeling.beh_model_comparison import plots_cross_prior_and_model_anynum
        # plots_cross_prior_and_model_anynum(self)
        self.plot_score_cross_prior_model(self.Dat)

        # Separate plots,split by taskgroup and by probe
        for split_by in ["taskgroup", "probe"]:
            self.plot_score_cross_prior_model_splitby(self.Dat, split_by=split_by)

        ############### STUFF PULLED IN FROM  plots_cross_prior_and_model
        # there is only for 2, so uses mod minus mod. here apply those plots for each pair of models...
        model_score_name_list = self.colnames_extract_scores()
        self.plot_score_scatter_compare_models(model_score_name_list[0], model_score_name_list[1])        
        self.plot_score_scatter_compare_epochs(model_score_name_list[0], epoch1, epoch2)        


    def plot_score_cross_prior_model_splitby(self, df=None, split_by="taskgroup"):
        """ bar plots crossing prior and model, plot separately for differelt levels of the variable
        split_by
        PARAMS;
        - split_by, variabl eto split by, e.g., "taskgroup"
        """
        import seaborn as sns

        if df is None:
            df = self.Dat

        Dat, DatWide, DatFlat, DatThisAgg, DatFlatAgg = self.extract_concatenated_aggregated_dataset(
            df, "epoch", list_cols_keep=[split_by])[:5]

        # combine in single plot (all taskgroups)
        fig = sns.catplot(data=DatFlat, x="epoch", y="score", hue="model", col=split_by, col_wrap=3, kind="bar")
        return fig


    def plot_score_cross_prior_model(self, df, monkey_prior_col_name="epoch", monkey_prior_list=None,
        list_classes=None, model_score_name_list =None, ALPHA = 0.2, sdir=None):
        """
        Summary plot of test dataset against models (scores), when num models is >2, this still works.
        NOTE: modified 12/19/22 to work with beh_model_holder, not with multiple... (which should be 
        changed to be a wrapper for beh_model_holder)
        INPUT:
        - monkey_prior_col_name
        --- e..g, "epoch"
        - GROUPING_LEVELS [aka monkey_prior_list]
        """
        from pythonlib.tools.pandastools import pivot_table
        from pythonlib.tools.plottools import plotScatter45
        import seaborn as sns
        
        # Get Single dataset
        Dat, DatWide, DatFlat, DatThisAgg, DatFlatAgg = self.extract_concatenated_aggregated_dataset(
            df, monkey_prior_col_name, monkey_prior_list, list_classes, model_score_name_list)[:5]

        # 1) Plot score fr all combo of dataset and model
        fig = sns.catplot(data=DatFlat, x=monkey_prior_col_name, y="score", hue="model", aspect=3, kind="bar")
        if sdir:
            fig.savefig(f"{sdir}/meanscore_epoch_by_rule_alltrials.pdf")
        # 2) same, agg over trials
        fig = sns.catplot(data=DatFlatAgg, x=monkey_prior_col_name, y="score", hue="model", aspect=3, kind="bar")
        if sdir:
            fig.savefig(f"{sdir}/meanscore_epoch_by_rule_allchars.pdf")

        # if column exists for binary_tuple then plot
        # to create 'binary_rule_tuple' column: run dataset.modeling.discrete.add_binary_rule_tuple_col
        if 'binary_rule_tuple' in Dat.columns:
            fig,ax = plt.subplots(figsize=(8,4))
            sns.histplot(data=Dat,x='epoch',hue='binary_rule_tuple',ax=ax,multiple="dodge",shrink=0.8)
            fig,ax = plt.subplots(figsize=(8,4))
            sns.histplot(data=Dat,x='binary_rule_tuple',hue='epoch',ax=ax,multiple="dodge",shrink=0.8)

        # For each trial, alignment, as rank out of all model scores.
        for colthis in self.colnames_extract_alignment():
            # colthis = "alignment_rank_chunks" # was this, for gridlinecircle.
            sns.catplot(data=Dat, x="epoch", y=colthis)
            sns.catplot(data=Dat, x="epoch", y=colthis, kind="boxen")
            sns.catplot(data=Dat, x="epoch", y=colthis, kind="swarm")

        return DatWide, DatFlat, DatThisAgg, DatFlatAgg


    def plot_score_scatter_compare_models(self, score_name_1, score_name_2, ax=None,
            savedir= None, list_epoch=None):
        """
        For a given epoch, plot scatter of score for each model, plotted as pairwise
        scatterplot (for a given pair of models)
        """
        from pythonlib.tools.plottools import plotScatter45

        if ax is None:
            fig, ax = plt.subplots(1,1)


        ############### STUFF PULLED IN FROM  plots_cross_prior_and_model
        # there is only for 2, so uses mod minus mod. here apply those plots for each pair of models...
        ALPHA = 0.2

        if list_epoch is None:
            list_epoch = sorted(self.Dat["epoch"].unique().tolist())
        monkey_prior_list = list_epoch
        monkey_prior_col_name = "epoch"
        plot_level = "trial"
        # model_score_name_list = BM.colnames_extract_scores()
        # score_name_1 = model_score_name_list[0]
        # score_name_2 = model_score_name_list[1]

        fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True)
        for i, lev in enumerate(monkey_prior_list):
            if plot_level == "trial":
                dfthis = DatWide[DatWide[monkey_prior_col_name]==lev]
            elif plot_level =="char":
                dfthis = DatThisAgg[DatThisAgg[monkey_prior_col_name]==lev]
            else:
                assert False

            x1 = dfthis[score_name_1]
            x2 = dfthis[score_name_2]

            # Scatter
            ax = axes.flatten()[i]
            plotScatter45(x1, x2, ax=ax, alpha=ALPHA, means=True)

            ax.set_xlabel(score_name_1)
            ax.set_ylabel(score_name_2)
            ax.set_title(lev)
        if savedir:
            fig.savefig(f"{savedir}/all_trials_origscores_scatter.pdf")

    def plot_score_scatter_compare_epochs(self, ax, scorename, epoch1, epoch2):
        """ For a given score, compare it across 2 epochs (in a scatter plot),
        plotting etierh trials for charavcters
        """

        assert False, "in progress. this pulled from plots_cross_prior_and_model"
        col1 = monkey_prior_list[0]
        col2 = monkey_prior_list[1]
        dfthis = DatThisAggPaired["mod2_minus_mod1"]
        x = dfthis[col1]
        y = dfthis[col2]
        tasknames = [v.split("-")[0] for v in DatThisAggPaired["character"].values]

        fig, ax = plt.subplots()
        plotScatter45(x, y, ax, dotted_lines="plus", means=True, labels = None)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title("mod2 - mod1")
        if savedir:
            fig.savefig(f"{savedir}/aggbytask_scatter_mod2minus1.pdf")

        fig, ax = plt.subplots(figsize=(15,15))
        plotScatter45(x, y, ax, dotted_lines="plus", means=False, labels = tasknames)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title("mod2 - mod1")
        if savedir:
            fig.savefig(f"{savedir}/aggbytask_scatter_mod2minus1_largetext.pdf")






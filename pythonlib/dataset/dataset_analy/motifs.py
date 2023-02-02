""" Given dataset, extract motifs (ordered list of tokens, where the relevatn features that define
a token are user-defined) and some basic analysis, stats, and plotting. 

Main goal are tools for extraction, filtering the motifs.

1/31/23 
Builds on:
- notebook: 220707_analy_spatial_prims_motifs, which I used to set up tasks for
experiments to have common subsequence (motifs). [REPLACES THIS]
- diagnostic modeling, comparing the actual motifs (beh) to the expected motifs
given specific models. in progress.
"""

import numpy as np
import matplotlib.pyplot as plt


def preprocess_dat(D, only_prims_on_grid=True):
    """ Helper to generate MotifsClass given a dataset D
    """

    if only_prims_on_grid:
        # 1) Only use prims in grid
        # D.filterPandas({"task_kind":["prims_on_grid"]}, "modify")
        D.filterPandas({"task_kind":["prims_on_grid"]}, "modify")

    # 1) preprocess
    params = ["no_supervision", "only_success", "correct_sequencing"]
    D.preprocessGood(params =params)

    # Generate tokens
    D.behclass_preprocess_wrapper()


class MotifsClass(object):
    """
    NOTE: each instance should work with only one kind of motif. If you want multiple
    motifs, just make multiple instances.
    """
    def __init__(self, Dataset):

        self.Dataset = Dataset.copy(dfdeep=False) # 

    def preprocess_motifs_extract(self, features=["shape_oriented", "gridloc"],
        which_sequence="beh_firsttouch", nprims=2, shapes_to_ignore=None):
        """ Generate initial set of all motifs.
        RETURNS:
        - saves motifs in self.MotifsAllDict, where keys are motifs (tuples of tokens) and itmes are 
        list of indices, where and index is (trial, starting strokenum)
        e..g, (('line-8-4-0', (1, 0)), ('line-8-4-0', (1, 1))): [(0, 0),
          (20, 0),
          (45, 0),
          (64, 0),
          (65, 0)]
        - self.MotifsList, list of motifs, sorted.
        """

        from pythonlib.dataset.dataset_analy.datseg_motifs import generate_dict_of_all_used_motifs
        # if e in ["primsingrid2", "primsingrid2sub", "shapedirsequence1"]:
        #     shapes_to_ignore = []
        # else:
        #     assert False
        #     # not sure what this means (shapes_to_ignore = ['squiggle3-2-0', 'V-2-0' ,'Lcentered-4-0'])
        #     shapes_to_ignore = ['squiggle3-2-0', 'V-2-0' ,'Lcentered-4-0']
        motifs_all_dict = generate_dict_of_all_used_motifs(self.Dataset, nprims, features, 
                                                            WHICH_DATSEGS = which_sequence, 
                                                           shapes_to_ignore=shapes_to_ignore)
        self._MotifsAllDict = motifs_all_dict
        self.MotifsList = sorted(list(motifs_all_dict.keys()))

    def motifs_all_dict_extract(self):
        """ The original reprenstiaton, before moved here in to Class
        """
        # Restructure data to fit older code below.
        motifs_all_dict = {}
        for k, v in self._MotifsAllDict.items():
            motifs_all_dict[k] = [vv[1] for vv in v]
        return motifs_all_dict

    def preprocess_motifgroups_extract(self):
        from pythonlib.dataset.dataset_analy.datseg_motifs import generate_motifgroup_data

        # Restructure data to fit older code below.
        motifs_all_dict = self.motifs_all_dict_extract()
        # for k, v in self._MotifsAllDict.items():
        #     motifs_all_dict[k] = [vv[1] for vv in v]
        self.MotifGroupsAllDict, self.MotifGroupsAllKinds = generate_motifgroup_data(motifs_all_dict)

    def _plot_single_motif_motifdat(self, motifdat, ax, which_overlay, underlay_single_color):
        trial = motifdat["trial"]
        # self.Dataset.plot_single_trial_highlighting_overlaid_strokes(md["trial"], 
        #     md["strokeinds"], ax, "beh", "beh", underlay_single_color="k")
        strokes = self._motifdat_strokes_extract(motifdat, which_overlay)
        self.Dataset._plot_single_trial_highlighting_overlaid_strokes(motifdat["trial"], 
            strokes, ax, "beh", underlay_single_color=underlay_single_color)

    def plot_single_motif_motifdats(self, list_motifdat, which_overlay="beh",
            underlay_single_color="k"):
        """ Plot each motifdat as a subplot, overlaying on the original trial
        """
        n= len(list_motifdat)
        ncols = 5
        nrows = int(np.ceil(n/ncols))
        SIZE = 2.5
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE),
            sharex=True, sharey=True)
        for md, ax in zip(list_motifdat, axes.flatten()):
            self._plot_single_motif_motifdat(md, ax, which_overlay, 
                    underlay_single_color=underlay_single_color)
        return fig, axes


    def plot_single_motif_trials(self, indmotif, trials=None, nrand=20, which_overlay="beh"):
        """ Plot trials for a single motif
        PARAMS:
        - trials,  if None, then plots all trials (nrand), otherwise a list of ints,
        trials into Dataset
        """
        assert trials is None, "not coded, this shoud index ibnto the trials"
        # if trials is None:
        #     import random
        #     trials = range(len(self.Dataset))


        # indmotif = random.sample(range(len(motifs_all_dict)), 1)[0]
        # motif = self.MotifsList[indmotif]

        list_motifdat = self.motifdat_extract_all(indmotif, strokes_extract=True)
        self.plot_single_motif_motifdats(list_motifdat, which_overlay="beh")


    def motifindex_to_str(self, indmotif):
        """
        Reutnrs string, input can be int or str
        """
        if isinstance(indmotif, int):
            motif = self.MotifsList[indmotif]
        else:
            motif = indmotif
        return motif

    def motifdat_extract_all(self, indmotif, strokes_extract=False):
        """ Extract all instances for this motif.
        RETURNS:
        - list of dicts, each an instance
        """ 
        motif = self.motifindex_to_str(indmotif)
        inds = range(len(self._MotifsAllDict[motif]))
        list_out = [self.motifdat_extract(indmotif, i, strokes_extract=strokes_extract) for i in inds]
        return list_out

    def motifdat_extract(self, indmotif, ind_withinmotif, strokes_extract=False):
        """ Helper to extract in useful terms
        PARAMS:
        - indmotif, index into self.MotifsList (if int) or self.MotifsAllDict,
        if str
        - ind_withinmotif, index into self.MotifsAllDict[motifstr]
        RETURNS:
        - dict, summarizing this motif instance
        """

        motif = self.motifindex_to_str(indmotif)
        trial_strok = self._MotifsAllDict[motif][ind_withinmotif][1]
        tokens = self._MotifsAllDict[motif][ind_withinmotif][0]
        if strokes_extract:
            strokes = [t["Prim"].Stroke() for t in tokens]
        else:
            strokes = None

        n = len(motif)
        out = {
            "motif":motif,
            "indmotif":indmotif,
            "trial":trial_strok[0],
            "strokeinds":list(range(trial_strok[1], trial_strok[1]+n)),
            "tokens":tokens,
            "strokes_task":strokes
        }
        return out

    def _motifdat_indsbeh_extract(self, motifdat):
        """ Returns the list of ints
        """

        # print(motifdat)
        inds_inds = [t["ind_behstrokes"] for t in motifdat["tokens"]]
        inds = [i for inds in inds_inds for i in inds] # flatten
        # print(inds)
        # assert False
        return inds

    def _motifdat_strokes_extract(self, motifdat, which="beh"):
        """ Return the strokes for this datmotif,
        """
        if which=="task":
            strokes = motifdat["strokes_task"]
        elif which=="beh":
            inds_beh = self._motifdat_indsbeh_extract(motifdat)
            trial = motifdat["trial"]
            strokes_beh = self.Dataset.Dat.iloc[trial]["strokes_beh"]
            strokes = [strokes_beh[i] for i in inds_beh]
        else:
            assert False
        return strokes

    def print_summary(self):

        for i, motif in enumerate(self.MotifsList):
            n = len(self._MotifsAllDict[motif])
            print(i, ' -- ', motif, ' -- ', n)

    def print_summary_thismotif(self, indmotif):

        list_md = self.motifdat_extract_all(indmotif)
        for md in list_md:
            print(md["trial"], ' -- ', md["strokeinds"])

    ################ MOTIFGROUP
    def print_motifgroup_summary(self):
        """ Print summary of all the motif groups
        """
        DatGroups = self.MotifGroupsAllDict
        list_group_kind = self.MotifGroupsAllKinds

        # Summarize, how many groups found and how many trials each
        do_print_shapes = True

        def _print(DatGroups, which_group, inds_motifgroup_ordered=None):
            if inds_motifgroup_ordered is None:
                inds_motifgroup_ordered = range(len(self.MotifGroupsAllDict[which_group]))

            for i in inds_motifgroup_ordered:
                # 1) info about the motifs/tokens/shapes actually found for this group
                list_motifs, motifgroup, list_tokens, list_shapes = self.motifgroup_extract_motifs_simple(which_group, i)
                        
                # 2) Info about the n distributions
                Dat = DatGroups[which_group][i]
                ntrials_per_unique_motif = sorted(Dat["n_trials_per_motif"], reverse=True)
                ntrials_per_unique_motif = [n for n in ntrials_per_unique_motif if n>0]
                if do_print_shapes:
                    print("#", i, "|n trials tot:", Dat["n_trials_summed_across_motifs"], "|n motifs:", Dat["n_unique_motifs_with_at_least_one_trial"], "|n trials/motif:", ntrials_per_unique_motif, "|shapes:", list_shapes)
                else:
                    print("#", i, "|n trials tot:", Dat["n_trials_summed_across_motifs"], "|n motifs:", Dat["n_unique_motifs_with_at_least_one_trial"], "|n trials/motif:", ntrials_per_unique_motif)
        
        for which_group in list_group_kind:
            print(" === ", which_group)
            _print(DatGroups, which_group, None)
    
    def motifgroup_count_num_motifkeys(self, which_group):
        """ Return int, the number of keys within this group
        NOTE: a key is a set of motifs which have some shared feature.
        """
        dat = self.MotifGroupsAllDict[which_group]
        return len(dat)

    # def motifgroup_extract_trials(self, which_group, ind_within_group, doprint=False,
    #         NMAX_PER_MOTIF=None):
    #     """ Get all trials (and stroke num) that are in this which_group (this indx within)
    #     # indtoplot = 1 # where 0 means the best (group with most cases)
    #     # which_group = "diff_sequence"
    #     # which_group = "diff_location"
    #     # which_group = "same"
    #     """
    #     assert False, "use motifgroup_extract_all_motifs instead"
    #     from pythonlib.dataset.dataset_analy.datseg_motifs import get_inds_this_motifgroup

    #     motifs_all_dict = self.motifs_all_dict_extract()

    #     # Extract all the specific trials that are part of this group
    #     dat = self.MotifGroupsAllDict[which_group]
    #     motifgroup = dat[ind_within_group]["key"]
    #     inds_idxs_all, list_motifs, _ = get_inds_this_motifgroup(self.MotifGroupsAllDict, which_group, motifs_all_dict, 
    #                                                              motifgroup, ntrials_per_motif=NMAX_PER_MOTIF)
    #     inds_trials_all = [idx[0] for idx in inds_idxs_all]
    #     inds_strokenums_all = [idx[1] for idx in inds_idxs_all]

    #     if doprint:
    #         print("Thiese trials and strokes and motifs:")
    #         for mot, inds in zip(list_motifs, inds_idxs_all):
    #             print(mot, ' --- ', inds)
    #         # print(inds_idxs_all)
    #         # print(list_motifs)
    #     return inds_trials_all, inds_strokenums_all, list_motifs

    def motifgroup_extract_motifs_thiskey(self, which_group, motifkey_index, doprint=False,
            NMAX_PER_MOTIF=None, return_as_list_motifdat=False):
        """ Get all motif instances that are in this which_group (this indx within)
        PARAMS
        - which_group, str, what group of motifs? 
        e.g., "same", "diff_sequence", "diff_location"
        - motifkey_index, index within the sets of motifs that are cases for,
        where 0 means the best (group with most cases)
        RETURNS:
        - list of motif instances, which are either dicts or motifdats, depending on 
        whether return_as_list_motifdat is True.
        """
        from pythonlib.dataset.dataset_analy.datseg_motifs import get_inds_this_motifgroup

        # motifs_all_dict = self.motifs_all_dict_extract()

        # Extract all the specific trials that are part of this group
        dat = self.MotifGroupsAllDict[which_group]
        motifkey = self.MotifGroupsAllDict[which_group][motifkey_index]["key"]
        n_trials_per_motif = self.MotifGroupsAllDict[which_group][motifkey_index]["n_trials_per_motif"]
        
        list_list_md = []
        out = []
        for i, (motif, ntrials) in enumerate(zip(motifkey, n_trials_per_motif)):
            if ntrials>0:
                list_md = self.motifdat_extract_all(motif)
                for md in list_md:
                    out.append({
                        "motifdat":md,
                        "motif":motif,
                        "motif_index":i,
                        "motifkey":motifkey,
                        "motifkey_index":motifkey_index,
                        "which_group":which_group,
                        "trial":md["trial"],
                        "indstroke_on":md["strokeinds"][0]
                        })
        if return_as_list_motifdat:
            out = [d["motifdat"] for d in out]

        return out

    def motifgroup_extract_motifs_all_filtered(self, which_group, feature_vary=None, 
            feature_constant=None):
        """ Clever way to represent motifs, related to getting variation in character (for
        example). e.g,, if want to get motifs which exist across characters, holding epoch
        constant.
        PARAMS:
        - which_group, str, collects all sets of motifs (i.e.,, motif keys) in this group, e.g.
        "same", "diff_prims"
        - feature_vary, str, whether to filter, so that only get motif_keys which have trials across
        >1 level for this feature. e.g., if want motif_keys that have motifs present in at least 2
        characters, make this "character"
        - feature_constant, str, what to condition on, as you check feature_vary. e.g., if want to extract
        only cases which, for each epoch, have trials across levels of character, make this "epoch"
        E.G. motifgroup_extract_motifs_all_filtered("same", feature_vary=epoch, feature_constant=character)
        will extract only conjunction of motif_key and character which have cases of "same" motif across epochs
        RETURNS: 
        - OUT, list of dict, where each dict represents a single motif isntance on a specific trial. 
        """

        D = self.Dataset

        if feature_constant is not None:
            # get each level.
            list_levels_feature_constant = sorted(self.Dataset.Dat[feature_constant].unique().tolist())
        else:
            # then dont care about this
            list_levels_feature_constant = [None]

        # for each group, get its list of trials
        nkeys = self.motifgroup_count_num_motifkeys(which_group)

        OUT = []
        for i in range(nkeys):
            
            list_motifdict = self.motifgroup_extract_motifs_thiskey(which_group, i, return_as_list_motifdat=False)
            inds_trials_all = [x["trial"] for x in list_motifdict]
                
            for feat_con in list_levels_feature_constant:

                # Restrict just to trials within this feature_con
                if feat_con is not None:
                    trials_good = D.Dat[(D.Dat[feature_constant]==feat_con)].index.tolist()
                    inds_trials_good = [i_ for i_ in inds_trials_all if i_ in trials_good]
                    feat_con_save = feat_con
                else:
                    inds_trials_good = inds_trials_all
                    feat_con_save = "IGNORE"

                if len(inds_trials_good)==0:
                    print(i, feat_con, "failed inds_trials_good...")
                    continue

                # chekc if have multipel llevels of desired feature
                if feature_vary:
                    list_feat_unique = D.Dat.iloc[inds_trials_good][feature_vary].unique().tolist()
                    if len(list_feat_unique)<2:
                        print(i, feat_con, "failed list_feat_unique...")
                        continue

                # extract the motifdats 
                list_motifdict_this = [x for x in list_motifdict if x["trial"] in inds_trials_good]
                list_motifdat = [x["motifdat"] for x in list_motifdict_this]
                list_trialsthis = [x["trial"] for x in list_motifdict_this]
                # For each motif, get its level
                if feature_vary:
                    list_feature_vary_levels = D.Dat.iloc[list_trialsthis][feature_vary].tolist()
                else:
                    list_feature_vary_levels = None

                for motifdict in list_motifdict_this:
                    trial = motifdict["trial"]

                    OUT.append({
                        "trial": trial,
                        "motifdict":motifdict,
                        "which_group":which_group,
                        "motifkey":i,
                        "feature_constant":feat_con_save,
                        "feature_vary":feature_vary,
                        "feature_vary_level":D.Dat.iloc[trial][feature_vary] if feature_vary is not None else None
                        })
        return OUT
                        


    def motifgroup_extract_motifs_simple(self, which_group="same", index=0):
        """ Return the list of motifs found for this motifgrop
        e.g,: returns:
        ([(('line-8-4-0', (-1, 0)), ('line-8-4-0', (-1, 1)))],
             ((('line-8-4-0', (-1, 0)), ('line-8-4-0', (-1, 1))),),
             [('line-8-4-0', (-1, 1)), ('line-8-4-0', (-1, 0))],
             ['line-8-4-0'])
        """
        from pythonlib.dataset.dataset_analy.datseg_motifs import get_all_motifs_found_for_this_motifgroup
        # if inds_motifgroup_ordered is None:
        inds_motifgroup_ordered = range(len(self.MotifGroupsAllDict[which_group]))
        return get_all_motifs_found_for_this_motifgroup(self.MotifGroupsAllDict,
            which_group, index) 

    def plot_motifgroup_singlegroup_trials(self, which_group, ind_within_group, 
        NMAX_PER_MOTIF = 30, underlay_single_color="k"):
        if False:
            inds_trials_all, inds_strokenums_all, list_motifs = self.motifgroup_extract_trials(
                which_group, ind_within_group, doprint=True, NMAX_PER_MOTIF=NMAX_PER_MOTIF)
        else:
            list_motifdat = self.motifgroup_extract_motifs_thiskey(which_group, ind_within_group, 
                NMAX_PER_MOTIF=NMAX_PER_MOTIF, return_as_list_motifdat=True)

        #### Plot each trial for this group
        if False:
            # for trial, snum in zip(inds_trials_all, inds_strokenums_all):
            self.Dataset.plotMultTrials2(inds_trials_all, titles = inds_strokenums_all);
            self.Dataset.plotMultTrials(inds_trials_all, "strokes_task", titles = inds_strokenums_all);
        else:
            fig, axes = self.plot_single_motif_motifdats(list_motifdat, 
                underlay_single_color=underlay_single_color)
            # self.plot_single_motif_motifdats([dat[0]["motifdat"]], ax)

        return fig, axes, list_motifdat


def plotwrapper_drawings_trials_spanning_feature(D, nprims, SDIR, which_group="same", feature_vary = None,
    feature_constant = None):
    """ Plot drawings, where each plot is all drawins across levels of a given feature
    (feature_vary, e.g., epoch), for a single level of the feature (feature_constant, e.g, char).
    both feature_vary and feature_constant can be None, in which case ignore it. All drawings depict
    instances of motifs within a motif_key, under the grouping which_group (e..g, same). 
    (SEE e.g.);
    PARAMS;
    - nprims, motifs with this many prims.
    EXAMPLE:
    e..g, for this, each plot is a single motif (shape-location) for all trials across all chars it 
    is found (and skipping this modif if doesnt find across chars), one plot for each epoch-motif
    conjunction.
    plotwrapper_drawings_trials_spanning_feature(D, 2, SDIR, which_group="same", feature_vary = "char",
        feature_constant = "epoch"):
    """

    # # Find groups that have trials spanning a levels of a vairble (ee..g epoch)
    # which_group="same"
    # # feature_vary = "epoch" # want trials across all levels of this feature
    # feature_vary = None # want trials across all levels of this feature
    # # feature_constant = "character" # perform analysis separately for each level of this feature
    # feature_constant = None # perform analysis separately for each level of this feature
    import pandas as pd
    from pythonlib.tools.expttools import writeStringsToFile

    sdir = f"{SDIR}/nprims_{nprims}-matchacrosstrials-grp_{which_group}-varyby_{feature_vary}-constby_{feature_constant}"
    import os
    os.makedirs(sdir, exist_ok=True)
    print("Saving at:", sdir)

    # 1) Generate MotifsClass and preprocess
    MC = MotifsClass(D)
    MC.preprocess_motifs_extract(nprims=nprims)
    MC.preprocess_motifgroups_extract()

    # 2) Extract motifs across features.
    OUT = MC.motifgroup_extract_motifs_all_filtered(which_group, feature_vary, feature_constant)
    dfout = pd.DataFrame(OUT)

    #### Write all trials to a text file.
    list_str = []
    list_str_trialcodes = []
    list_motifkey = dfout["motifkey"].unique().tolist()
    list_feature_constant = dfout["feature_constant"].unique().tolist()
    list_feature_level = dfout["feature_vary_level"].unique().tolist()
    for key in list_motifkey:
        for fc in list_feature_constant:
            n = sum((dfout["motifkey"]==key) & (dfout["feature_constant"]==fc))
            if n>0:
                list_str.append(f"#{key} -- {fc}")
                list_str_trialcodes.append(f"#{key} -- {fc}")
                for lev in list_feature_level:
                    dfthis = dfout[(dfout["motifkey"]==key) & (dfout["feature_constant"]==fc) & (dfout["feature_vary_level"]==lev)]
                    list_trial = dfthis["trial"].tolist()
                    list_motif = [x["motif"] for x in dfthis["motifdict"].tolist()]
                    list_str.append(f"  {lev} -- {len(dfthis)}")    
                    list_tc = D.Dat.iloc[list_trial]["trialcode"].tolist()
                    for tc, mot in zip(list_motif, list_tc):
                        list_str_trialcodes.append(f"  {lev}  |  {tc}  |  {mot}")
                    
    # how many trials across each level of feature vary
    fname = f"{sdir}/eachnum_trials.txt"
    writeStringsToFile(fname, list_str)
    fname = f"{sdir}/each_trial_motif.txt"
    writeStringsToFile(fname, list_str_trialcodes)

    #### Plot each drawing
    for key in list_motifkey:
        for fc in list_feature_constant:
            dfthis = dfout[(dfout["motifkey"]==key) & (dfout["feature_constant"]==fc)]
            if len(dfthis)>0:
                print(f"#{key} -- {fc}")
                list_mdict = dfthis["motifdict"].tolist()
                list_motifdat = [x["motifdat"] for x in list_mdict]
                list_featvar_levels = dfthis["feature_vary_level"].tolist()
                list_trials = dfthis["trial"].tolist()
                list_trialcode = D.Dat.iloc[list_trials]["trialcode"]
                # for each trial, get its value across desired levsls.
                fig, axes = MC.plot_single_motif_motifdats(list_motifdat, underlay_single_color=None)

                # show its feature
                titles = [f"{tcode}-{feat}" for tcode, feat in zip(list_trialcode, list_featvar_levels)]
    #             titles = D.Dat.iloc[inds_trials_good][feature_vary].tolist()
                
                for ax, tit in zip(axes.flatten(), titles):
                    ax.set_title(tit)

                # save
                fig.savefig(f"{sdir}/motifkey_{key}-featconst_{fc}.pdf")

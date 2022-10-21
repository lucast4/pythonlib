""" To study learning of rules/grammars.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.snstools import rotateLabel
import pandas as pd
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles


def preprocess_dataset(D):
    """ Preprocess Dataset as basis for all subsetquence grammar/learning analyses.
    RETURNS:
    - dfGramScore, dataframe holding each trial, whether is success (beh sequence matches task sequebce,
    where latter is from the active chunk, and alignemnet is done with alignment matrix.
    - list_blocksets_with_contiguous_probes, list of list of ints, where inner lists hold
    blocks that are continusous and which all have probe tasks. these are useful for making
    separate plots for each. 
    - SDIR, string path to directory for all saving of grammar.
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows

    ################## Create save directiory
    SDIR = D.make_savedir_for_analysis_figures("grammar")
    savedir= f"{SDIR}/summary"
    os.makedirs(savedir, exist_ok=True) 

    ################## Generate behclass
    D.behclass_preprocess_wrapper()

    ################## Extract dataframe computing matches of beh sequence to task sequence
    # 2) For each trial, determine whether (to waht extent) beh matches task inds sequence
    gramscoredict = []
    # def nmatch_(order_beh, order_correct):
    print("frac strokes gotten in progress -> use a string edit distance")

    for ind in range(len(D.Dat)):
        gramdict = D.sequence_extract_beh_and_task(ind)

        # frac strokes gotten
        order_beh = gramdict["taskstroke_inds_beh_order"]
        order_correct = gramdict["taskstroke_inds_correct_order"]

        # binary fail/success
        success_binary = order_beh==order_correct
        
        # note cases where did not complete the trial, but sequence was correct so far (exclude those)
        def beh_good_ignore_length(order_beh, order_correct):
            """ True if beh is good up unitl the time beh fails.
            e.g., if beh is [0, 3, 2], and correct is [0, 3, 2, 1], then
            this is True
            """
            for x, y in zip(order_beh, order_correct):
                if x!=y:
                    return False
            return True
        beh_sequence_wrong = not beh_good_ignore_length(order_beh, order_correct)
        beh_too_short = len(order_beh) < len(order_correct)
        
        # exclude cases where beh was too short, but order was correct
        exclude_because_online_abort = beh_too_short and not beh_sequence_wrong
        
        # combination of (epoch, supervision_tuple)
        epoch_superv = (gramdict["epoch"], gramdict["supervision_tuple"])

        # block (sanity check)
        block = D.Dat.iloc[ind]["block"]
        
        # taskgroup
        taskgroup = D.Dat.iloc[ind]["taskgroup"]
        
        # character
        char = D.Dat.iloc[ind]["character"]
        
        # COLLECT
        gramscoredict.append({
            "success_binary":success_binary,
            "beh_sequence_wrong":beh_sequence_wrong,
            "beh_too_short":beh_too_short,
            "exclude_because_online_abort":exclude_because_online_abort,
            "epoch_superv":epoch_superv,
            "block": block,
            "datind":ind,
            "taskgroup":taskgroup,
            "character":char
        })
    dfGramScore = pd.DataFrame(gramscoredict)

    ##### 1) Extract blocksets (contiguous blocks with probe tasks)
    blocks = sorted(D.probes_extract_blocks_with_probe_tasks())
    list_blockset = []
    current_blockset = []
    for i, bk in enumerate(blocks):
    #     print(list_blockset, ' -- ', current_blockset)
        if len(current_blockset)==0:
            # then this starts a new blockset
            current_blockset.append(bk)
        else:
            if bk == current_blockset[-1]+1:
                # it continues. include it
                current_blockset.append(bk)
            else:
                # a gap in blocks. therefore start a new blcokset
                list_blockset.append(current_blockset)
                current_blockset = [bk]
    # add the last blockset
    list_blockset.append(current_blockset)
                
    print("Got these sets of blocks: ", list_blockset)

    # assign each ind to a blockset (if exists)
    # if doesnt exist, assign -1.
    mapper = {} # bk --> bkset
    for i, bkset in enumerate(list_blockset):
        for bk in bkset:
            mapper[bk] = i
    def F(x):
        if x["block"] in mapper.keys():
            return mapper[x["block"]]
        else:
            return -1
    dfGramScore = applyFunctionToAllRows(dfGramScore, F, "which_probe_blockset")
    list_blocksets_with_contiguous_probes = list_blockset

    return dfGramScore, list_blocksets_with_contiguous_probes, SDIR


def print_useful_things(dfGramScore):
    ## PRINT useful things
    from pythonlib.tools.pandastools import grouping_get_inner_items
    print("---- 1")
    display(grouping_get_inner_items(dfGramScore, "epoch_superv", "block"))
    print("---- 2")
    display(grouping_get_inner_items(dfGramScore, "taskgroup", "epoch_superv"))
    print("---- 3")
    display(grouping_get_inner_items(dfGramScore, "taskgroup", "block"))
    print("---- 4")
    display(grouping_get_inner_items(dfGramScore, "block", "epoch_superv"))    


def plot_performance_all(dfGramScore, list_blockset, SDIR):
    """ Standard plots summarizing perforamnce, slicing in different ways, includiong
    aggragating to characters (or sticking with trials). Saves automatically.
    """
    from pythonlib.tools.pandastools import aggregGeneral

    # Make save dir
    savedir= f"{SDIR}/summary"
    os.makedirs(savedir, exist_ok=True) 
    if checkIfDirExistsAndHasFiles(savedir)[1]:
        print("[SKIPPING, since SDIR exists and has contents: ", savedir)
        return


    def plot_(dfthis):
        """ Helper to make all plots """
        
        listfigs = []
        
        # sns.catplot(data=dfGramScore, x="block", y="epoch_superv", hue="taskgroup", kind="strip", alpha=0.5, height=10)
        if "block" in dfthis.columns:
            sns.catplot(data=dfthis, x="block", y="epoch_superv", col="taskgroup", col_wrap=2, kind="strip", 
                        hue="success_binary", alpha=0.5, height=10)

        fig = sns.catplot(data=dfthis, x="epoch_superv", y="success_binary", col="taskgroup", col_wrap=2, kind="bar",
                    height=5)
        rotateLabel(fig)
        listfigs.append(fig)

        fig = sns.catplot(data=dfthis, x="epoch_superv", y="success_binary", col="taskgroup", col_wrap=2, height=5)
        rotateLabel(fig)
        listfigs.append(fig)

        try:
            fig = sns.catplot(data=dfthis, x="epoch_superv", y="success_binary", hue="taskgroup", kind="bar",
                        height=5, aspect=1.5)
            rotateLabel(fig)
            listfigs.append(fig)
    #         fig = sns.catplot(data=dfthis, x="epoch_superv", y="success_binary", hue="taskgroup", 
    #                     alpha=0.5, height=5, aspect=1.5)
    #         rotateLabel(fig)
        except:
            pass
        
        # fig = sns.catplot(data=dfGramScore, x="taskgroup", y="success_binary", col="epoch_superv", col_wrap=2, kind="bar",
        #             hue="epoch_superv", alpha=0.5, height=2, aspect=1.5)
        fig = sns.catplot(data=dfthis, x="taskgroup", y="success_binary", col="epoch_superv", col_wrap=2, kind="bar",
                          height=2, aspect=1.5)
        rotateLabel(fig)
        listfigs.append(fig)

        fig = sns.catplot(data=dfthis, x="taskgroup", y="success_binary", col="epoch_superv", col_wrap=2, 
                          alpha=0.7, height=2, aspect=1.5)
        rotateLabel(fig)
        listfigs.append(fig)

        try:
            fig = sns.catplot(data=dfthis, x="taskgroup", y="success_binary", hue="epoch_superv", kind="bar",
                              height=5, aspect=1.5)
            rotateLabel(fig)
            listfigs.append(fig)
            
    #         fig = sns.catplot(data=dfthis, x="taskgroup", y="success_binary", hue="epoch_superv", 
    #                           alpha=0.5, height=5, aspect=1.5)
    #         rotateLabel(fig)
        except:
            pass
        
        return listfigs

    # How many trial ssuccess/failure?
    fig = sns.catplot(data=dfGramScore, x="success_binary", y="beh_sequence_wrong", 
                hue="beh_too_short", col="exclude_because_online_abort", kind="strip", alpha=0.2)
    fig.savefig(f"{savedir}/distribution_success.pdf")
    rotateLabel(fig)


    ### [All data, datapt=trial] Only consider cases where completed the entire sequence
    dfthis = dfGramScore[dfGramScore["exclude_because_online_abort"]==False]
    figs = plot_(dfthis)
    for i, f in enumerate(figs):
        f.savefig(f"{savedir}/successrate-summary-blocks_all-datapt_trial-{i}.pdf")
    plt.close("all")

    ## [Blocks of interest (blocks with interleaved trials), datapt=trial]  Prune to specific set of blocks
    for blocks_keep in list_blockset:
        dfthis = dfGramScore[
            (dfGramScore["block"].isin(blocks_keep)) & 
            (dfGramScore["exclude_because_online_abort"]==False)
        ]
        figs = plot_(dfthis)
        for i, f in enumerate(figs):
            f.savefig(f"{savedir}/successrate-summary-blocks_{_blocks_to_str(blocks_keep)}-datapt_trial-{i}.pdf")
    plt.close("all")

    ## The same, but datapt=char (aggregate over trials)
    dfthis = dfGramScore[dfGramScore["exclude_because_online_abort"]==False]
    dfthisAgg = aggregGeneral(dfthis, group=["epoch_superv", "taskgroup", "character"], values=["success_binary"])

    ### [All data, datapt=char]
    figs = plot_(dfthisAgg)
    for i, f in enumerate(figs):
        f.savefig(f"{savedir}/successrate-summary-blocks_all-datapt_char-{i}.pdf")

    for blocks_keep in list_blockset:
        dfthis = dfGramScore[
            (dfGramScore["block"].isin(blocks_keep)) & 
            (dfGramScore["exclude_because_online_abort"]==False)]
        dfthisAgg = aggregGeneral(dfthis, group=["epoch_superv", "taskgroup", "character"], values=["success_binary"])
        figs = plot_(dfthisAgg)
        for i, f in enumerate(figs):
            f.savefig(f"{savedir}/successrate-summary-blocks_{_blocks_to_str(blocks_keep)}-datapt_char-{i}.pdf")
    plt.close("all")


def plot_performance_timecourse(dfGramScore, list_blockset, SDIR):
    """ Plot perforamcne as function of blocks (not bloques!)
    """
    sdirthis = f"{SDIR}/summary"
    dfthis = dfGramScore[
        (dfGramScore["exclude_because_online_abort"]==False)
    ]
    # taskgroup", y="success_binary", col="epoch_superv
    fig = sns.catplot(data=dfthis, x="block", y="success_binary", row="taskgroup", hue="epoch_superv", kind="point", 
                aspect=2, alpha=0.4, ci=68)
    fig.savefig(f"{sdirthis}/timecourse-blocks-datapt_trial.pdf")
    rotateLabel(fig)


def plot_performance_static_summary(dfGramScore, list_blockset, SDIR):
    """ Bar plots of "static" performance, avberaged within
    blocksets (each a contigous set of plots containing probe tasks)
    """

    from pythonlib.tools.expttools import writeStringsToFile
    sdirthis = f"{SDIR}/summary"
    # Exract only no online abort
    dfthis = dfGramScore[
        (dfGramScore["exclude_because_online_abort"]==False)
    ]

    # 1) Print summary and save it (perforamnce, as function of hierarhcal params)
    N_MIN = 5 # skip printing for cases with fewer trials
    list_taskgroups = sorted(dfthis["taskgroup"].unique())
    list_epochsuperv = sorted(dfthis["epoch_superv"].unique())
    list_textstrings = [] # for saving
    for i, blockset in enumerate(list_blockset):
        s = f"Blockset #{i}: [{_blocks_to_str(blockset)}]"
        list_textstrings.append(s)
        print(s)

        for taskgroup in list_taskgroups:
            s = f" Taskgroup: {taskgroup}"
            list_textstrings.append(s)
            print(s)
            for epoch_superv in list_epochsuperv:
                # get performance of tasks
                inds = (dfthis["block"].isin(blockset)) & (dfthis["taskgroup"]==taskgroup) & (dfthis["epoch_superv"]==epoch_superv)
                nthis = sum(inds)
                if nthis>N_MIN:
    #                 print(dfthis[inds]["success_binary"])
    #                 print(sum(inds))
                    s = f"   {epoch_superv}: --> {100 * np.mean(dfthis[inds]['success_binary']):.0f}% (N={nthis})"
                    list_textstrings.append(s)
                    print(s)
    # SAVE
    path = f"{sdirthis}/staticsummary.txt"
    writeStringsToFile(path, list_textstrings)

    # 2) Plot bar plot (same infor as in list_textstrings)
    fig = sns.catplot(data = dfthis, x="taskgroup", y="success_binary", hue = "epoch_superv", kind="bar", ci=68,
               row="which_probe_blockset", aspect=2, height=3)
    rotateLabel(fig)
    fig.savefig(f"{sdirthis}/staticsummary.pdf")


def _blocks_to_str(blocks):
    """ Helper to conver tlist of ints (blocks) to a signle string"""
    return "|".join([str(b) for b in blocks])
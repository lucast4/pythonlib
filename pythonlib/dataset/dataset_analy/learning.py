""" General-purpose stuff to analyze learning, where here extracts various meta params, and
holds various plots for timecourse, etc.
"""

print("TODO!!!")
# Merge this with other learning-related code:
# - characters
# - timecourse stuff (see Evernote for where this is located)
# Make this code generally work for analysis of timecourses.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.snstools import rotateLabel
import pandas as pd
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def preprocess_dataset(D):
    """ Preprocess Dataset as basis for all subsetquence grammar/learning analyses.
    PARAMS:
    - grammar_recompute_parses, bool, if False, then uses the saved "active chunk" used 
    during the task. This works for training tasks with single correct parse,
    but not (in general) for testing, which may have multiple correct parses. Deal with latter
    by recomputing parses using D.grammar_parses_generate. 
    - grammar_correct_rule, string, must be inputed if you use grammar_recompute_parses.
    This defined what will be the set of correct orders to compare with beh.
    RETURNS:
    - dfGramScore, dataframe holding each trial, whether is success (beh sequence matches task sequebce,
    where latter is from the active chunk, and alignemnet is done with alignment matrix.
    - list_blocksets_with_contiguous_probes, list of list of ints, where inner lists hold
    blocks that are continusous and which all have probe tasks. these are useful for making
    separate plots for each. 
    - SDIR, string path to directory for all saving of grammar.
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows

    # ################## Create save directiory
    # SDIR = D.make_savedir_for_analysis_figures("grammar")
    # savedir= f"{SDIR}/summary"
    # os.makedirs(savedir, exist_ok=True) 

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
    D.Dat = applyFunctionToAllRows(D.Dat, F, "which_probe_blockset")
    list_blocksets_with_contiguous_probes = list_blockset


    return list_blocksets_with_contiguous_probes


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
        if len(dfthis)>0:
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
        if len(dfthis)>0:
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


def plot_performance_static_summary(dfGramScore, list_blockset, SDIR, 
        only_cases_got_first_stroke=False):
    """ Bar plots of "static" performance, avberaged within
    blocksets (each a contigous set of plots containing probe tasks)
    """

    from pythonlib.tools.expttools import writeStringsToFile
    sdirthis = f"{SDIR}/summary"
    # Exract only no online abort

    if only_cases_got_first_stroke:
        dfthis = dfGramScore[
            (dfGramScore["exclude_because_online_abort"]==False) &
            (dfGramScore["beh_got_first_stroke"]==True)
        ]
    else:
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
    path = f"{sdirthis}/staticsummary-first_strk_correct_{only_cases_got_first_stroke}.txt"
    writeStringsToFile(path, list_textstrings)

    # 2) Plot bar plot (same infor as in list_textstrings)
    fig = sns.catplot(data = dfthis, x="taskgroup", y="success_binary", hue = "epoch_superv", kind="bar", ci=68,
               row="which_probe_blockset", aspect=2, height=3)
    rotateLabel(fig)
    fig.savefig(f"{sdirthis}/staticsummary-first_strk_correct_{only_cases_got_first_stroke}-1.pdf")

    # 2) More compact bar plot
    fig = sns.catplot(data = dfthis, x="which_probe_blockset", y="success_binary", 
                hue = "epoch_superv", kind="bar", ci=68,
               col="taskgroup", col_wrap=3, aspect=2, height=3)
    rotateLabel(fig)
    fig.savefig(f"{sdirthis}/staticsummary-first_strk_correct_{only_cases_got_first_stroke}-2.pdf")

def plot_counts_heatmap(dfGramScore, SDIR):
    """ Heatmap for n trials for each combo of conditions (e.g., epoch|supervsion, and taskgroup)
    i.e. rows are epoch/sup and cols are taskgroup. plots and saves figures.
    """
    from pythonlib.tools.pandastools import convert_to_2d_dataframe, aggregGeneral
    sdirthis = f"{SDIR}/summary"

    def _plot_counts_heatmat(dfGramScore, col1, col2, ax=None):
        
        if col2=="taskgroup":
            # then datapt is characters (not trials)
            dfGramScoreAgg = aggregGeneral(dfGramScore, group=[col1, col2, "character"], values=["success_binary"])
            df = dfGramScoreAgg
        else:
            df = dfGramScore

        dfthis, fig, ax, _ = convert_to_2d_dataframe(df, col1, col2, True)
            
        return dfthis, fig, ax

    fig, axes = plt.subplots(3,3)

    col1 = "epoch_superv"
    col2 = "taskgroup"
    dfthis, fig, ax = _plot_counts_heatmat(dfGramScore, col1, col2)
    path = f"{sdirthis}/staticsummary.txt"
    fig.savefig(f"{sdirthis}/counts_heatmap_{col2}.pdf")

    col1 = "epoch_superv"
    col2 = "character"
    dfthis, fig, ax = _plot_counts_heatmat(dfGramScore, col1, col2)
    fig.savefig(f"{sdirthis}/counts_heatmap_{col2}.pdf")

    taskgroups = dfGramScore["taskgroup"].unique()
    for tg in taskgroups:
        dfthis = dfGramScore[(dfGramScore["taskgroup"] == tg)]
        col1 = "epoch_superv"
        col2 = "character"
        dfthis, fig, ax = _plot_counts_heatmat(dfthis, col1, col2)
        fig.savefig(f"{sdirthis}/counts_heatmap_{col2}-taskgroup_{tg}.pdf")

def plot_performance_each_char(dfGramScore, D, SDIR):
    """ Plot separate fro each characters, organized in different ways
    """

    from pythonlib.tools.snstools import rotateLabel

    # skip for now, since if many chars the plots are huge. should fix this somethow.
    if False:
        sdirthis = f"{SDIR}/summary"
        dfthis = dfGramScore[
            (dfGramScore["exclude_because_online_abort"]==False)
        ]

        fig = sns.catplot(data = dfthis, x="character", y="success_binary", hue="epoch_superv", 
                    col="which_probe_blockset", row="taskgroup", kind="bar", aspect=2)
        rotateLabel(fig)
        path = f"{sdirthis}/eachchar-overview-1.pdf"
        fig.savefig(path)

        fig = sns.catplot(data = dfthis, x="character", y="success_binary", hue="block", 
                    col="which_probe_blockset", row="taskgroup", kind="bar", aspect=2)
        rotateLabel(fig)
        path = f"{sdirthis}/eachchar-overview-2.pdf"
        fig.savefig(path)

        fig = sns.catplot(data = dfthis, x="block", y="success_binary", hue="character", 
                    col="taskgroup", row="epoch_superv", kind="bar", aspect=2)
        rotateLabel(fig)
        path = f"{sdirthis}/eachchar-overview-3.pdf"
        fig.savefig(path)



def plot_performance_trial_by_trial(dfGramScore, D, SDIR):
    """ Plots of timecourse of performance, such as trial by trial
    """

    list_sess = D.Dat["session"].unique().tolist()
    for sess in list_sess:

        dfthis = D.Dat[
            (D.Dat["exclude_because_online_abort"]==False) & 
            (D.Dat["session"]==sess) 
        ]

        fig, axes = plt.subplots(5,1, figsize=(35,11))

        # 1) info, plot blocks and etc.
        ax = axes.flatten()[0]
        sns.scatterplot(data=dfthis, x="trial", y="block", hue="which_probe_blockset", style="epoch", ax=ax, alpha = 0.75)
        ax.grid(True)

        # 2) trial by trial performance
        ax = axes.flatten()[1]
        ax.grid(True)
        sns.scatterplot(data=dfthis, x="trial", y="success_binary", hue="epoch_superv", ax=ax, 
                       alpha = 0.75)
        smwin = 20
        dfthis_rolling = dfthis.rolling(window=smwin, center=True).mean()
        # sns.lineplot(ax=ax, data=dfthis_rolling, x="trial", y="success_binary")
        sns.scatterplot(ax=ax, data=dfthis_rolling, x="trial", y="success_binary")
        blockver = "block"
        idx_of_bloque_onsets = []
        for i in np.argwhere(dfthis[blockver].diff().values):
            idx_of_bloque_onsets.append(i[0])
        bloque_onsets = dfthis["trial"].values[idx_of_bloque_onsets]
        # bloque_nums = df["bloque"].values[idx_of_bloque_onsets]
        # blokk_nums = dfthis["blokk"].values[idx_of_bloque_onsets]
        block_nums = dfthis["block"].values[idx_of_bloque_onsets]
        for b, x in zip(block_nums, bloque_onsets):
            ax.axvline(x)
            ax.text(x, ax.get_ylim()[1], f"k{b}\nt{x}", size=8)
        ax.grid(True)

        ax = axes.flatten()[2]
        ax.grid(True)
        sns.scatterplot(data=dfthis, x="trial", y="success_binary", hue="epoch", style="epoch_superv", ax=ax, 
                       alpha = 0.75)

        ## NOT EXCLUDING ONLINE ABORTS
        dfthis = D.Dat[(D.Dat["session"]==sess)]
        ax = axes.flatten()[3]
        ax.grid(True)
        ax.set_title('NOT excluding abort trials')
        sns.scatterplot(data=dfthis, x="trial", y="success_binary", hue="exclude_because_online_abort", style="epoch", ax=ax, 
                       alpha = 0.75)

        # Plotting reason for failure.
        ax = axes.flatten()[4]
        ax.grid(True)
        ax.set_title('NOT excluding abort trials')
        sns.scatterplot(data=dfthis, x="trial", y="exclude_because_online_abort", hue="success_binary", style="epoch", ax=ax, 
                       alpha = 0.75)


        sdirthis = f"{SDIR}/summary"
        path = f"{sdirthis}/timecourse-trials-overview-sess_{sess}.pdf"
        fig.savefig(path)

def _blocks_to_str(blocks):
    """ Helper to conver tlist of ints (blocks) to a signle string"""
    return "|".join([str(b) for b in blocks])
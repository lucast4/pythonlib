""" plots/processing for comparing beh and models. 
Currently optimized for draw nn models, but shoud be flexible nough with minor
mods to apply to toher momdels (e.g., bpl)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.pandastools import applyFunctionToAllRows, filterPandas, aggregGeneral
import os
from pythonlib.tools.expttools import makeTimeStamp, findPath
import seaborn as sns


def plots_cross_prior_and_model(DatThisTest, monkey_prior_col_name, monkey_prior_list, 
                                model_score_name_list, savedir=None, return_DatThisAggPaired=False):
    """ All plots, where take cross product of prior model (monkey) and model (anythign else)
    Assumes that there are only 2 priors and 2 models, for now.
    NOTES:
    - column_dsets = "monkey_prior" # names the dartasets.
    - column_dsets = "epoch" # names the dartasets.

    """
    from pythonlib.tools.pandastools import pivot_table
    from pythonlib.tools.plottools import plotScatter45
    ALPHA = 0.2

    assert len(model_score_name_list)==2, "not coded for otherwise"
    assert len(monkey_prior_list)==2, "not coded for otherwise"

    
    # --- subtract model1 from model2 score
    DatThisTest["mod2_minus_mod1"] = DatThisTest[model_score_name_list[1]] - DatThisTest[model_score_name_list[0]]
    
    # aggregate over tasks
    # tasklist = set(DatThisTest["character"])
    DatThisAgg = aggregGeneral(DatThisTest, group = ["character", monkey_prior_col_name], 
                               values=[model_score_name_list[0], model_score_name_list[1], "mod2_minus_mod1"])

    # Pivot
    DatThisAggPaired = pivot_table(DatThisAgg, index=["character"], columns = [monkey_prior_col_name], 
                                   values=[model_score_name_list[0], model_score_name_list[1], "mod2_minus_mod1"])
    # cleanup remove any rows with nans
    DatThisAggPaired = DatThisAggPaired.dropna().reset_index(drop=True)


    ################################### ALL TRIALS
    # plotting scores (not score minus score)
    fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True)
    for i, lev in enumerate(monkey_prior_list):
        dfthis = DatThisTest[DatThisTest[monkey_prior_col_name]==lev]

        x1 = dfthis[model_score_name_list[0]]
        x2 = dfthis[model_score_name_list[1]]

        # Scatter
        ax = axes.flatten()[i]
        plotScatter45(x1, x2, ax=ax, alpha=ALPHA, means=True)

        ax.set_xlabel(model_score_name_list[0])
        ax.set_ylabel(model_score_name_list[1])
        ax.set_title(lev)
    if savedir:
        fig.savefig(f"{savedir}/all_trials_origscores_scatter.pdf")


    # == plotting scores (not score minus score)
    fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True, figsize=(10,4))

    for i, lev in enumerate(monkey_prior_list):
        dfthis = DatThisTest[DatThisTest[monkey_prior_col_name]==lev]

        x1 = dfthis[model_score_name_list[0]]
        x2 = dfthis[model_score_name_list[1]]

        # histograms
        ax = axes.flatten()[i]
        ax.hist(x1, label="model1", density=True, histtype="step")
        ax.hist(x2, label="model2", density=True, histtype="step")

        ax.set_title(lev)
        ax.legend()
    if savedir:
        fig.savefig(f"{savedir}/all_trials_origscores_hist.pdf")


    #  PLOT OVERVIEW, ALL TASKS
    fig = sns.catplot(data=DatThisTest, x=monkey_prior_col_name, y="mod2_minus_mod1", aspect=3)
    plt.axhline(0, color="k", alpha=0.5)
    if savedir:
        fig.savefig(f"{savedir}/all_trials_catplot_summary_mod2minus1.pdf")


    fig = sns.displot(data=DatThisTest, hue=monkey_prior_col_name, x="mod2_minus_mod1", 
                aspect=3, kind="hist", stat="probability", common_norm=False)
    plt.axvline(0, color="r")
    if savedir:
        fig.savefig(f"{savedir}/all_trials_summary_mod2minus1.pdf")



    ########### ONE PLOT PER TASK
    fig = sns.displot(data=DatThisAgg, hue=monkey_prior_col_name, x="mod2_minus_mod1", 
                aspect=2, kind="hist", stat="probability", common_norm=False, bins=20,
               element="bars", fill=True)
    plt.axvline(0, color="r")
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_summary_mod2minus1.pdf")

    # === 
    fig = sns.catplot(data=DatThisAgg, x=monkey_prior_col_name, y="mod2_minus_mod1", aspect=3)
    plt.axhline(0, color="k", alpha=0.5)
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_catplot_summary_mod2minus1.pdf")

    ALPHA = ALPHA*2
    fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True)
    for i, lev in enumerate(monkey_prior_list):
        dfthis = DatThisAgg[DatThisAgg[monkey_prior_col_name]==lev]

        x1 = dfthis[model_score_name_list[0]]
        x2 = dfthis[model_score_name_list[1]]

        # Scatter
        ax = axes.flatten()[i]
        plotScatter45(x1, x2, ax=ax, alpha=ALPHA, means=True)

        ax.set_xlabel(model_score_name_list[0])
        ax.set_ylabel(model_score_name_list[1])
        ax.set_title(lev)
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_origscores_scatter.pdf")

    # == plotting scores (not score minus score)
    fig, axes = plt.subplots(1,len(monkey_prior_list), sharex=True, sharey=True, figsize=(10,4))

    for i, lev in enumerate(monkey_prior_list):
        dfthis = DatThisAgg[DatThisAgg[monkey_prior_col_name]==lev]

        x1 = dfthis[model_score_name_list[0]]
        x2 = dfthis[model_score_name_list[1]]

        # histograms
        ax = axes.flatten()[i]
        ax.hist(x1, label="model1", density=True, histtype="step")
        ax.hist(x2, label="model2", density=True, histtype="step")

        ax.set_title(lev)
        ax.legend()
    if savedir:
        fig.savefig(f"{savedir}/aggbytask_origscores_hist.pdf")

    ###############################################
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

    if return_DatThisAggPaired:
        return DatThisAggPaired


def plots_cross_prior_and_model_combined():
    """ combines plots for positive control (monkey-monkey) and model
    model-monkey - e..g, a single bar plot comparing alignemnt"""
    print("See notebook drawnn --> test_on_monkey_tasks...")

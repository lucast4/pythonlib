"""
For loading data across all microstim days (for syntax) and then plotting various summaries.
Need to have first extracted each day

See notebook: /home/lucas/code/drawmonkey/drawmonkey/notebooks_datasets/250426_analy_microstim_summary.ipynb

"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
from pythonlib.tools.snstools import rotateLabel
import pandas as pd


if __name__ == "__main__":
    from pythonlib.tools.pandastools import grouping_print_n_samples

    ### PARAMS
    nperms = 10000
    
    ### RUN
    import sys
    animal = sys.argv[1]
    SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/2_syntax/microstim/overall/{animal}"
    os.makedirs(SAVEDIR, exist_ok=True)

    ### Load data across all dates (pre-saved)
    import pandas as pd
    from glob import glob
    from  pythonlib.dataset.dataset_analy.microstim import mult_load_all_days
    DFALL, DATES_TO_SKIP = mult_load_all_days(animal)
    
    ##### Additional postprocessing
    # Keep only epochsets that have same character across epohcs
    savepath = f"{SAVEDIR}/counts_before_prune_same_epochset.txt"
    grouping_print_n_samples(DFALL, ["date", "epochset", "epoch_orig"], savepath=savepath)
    n1 = len(DFALL)
    tmp = []
    for epochset in DFALL["epochset"]:
        tmp.append(epochset[0]=="same")
    print(sum(tmp), len(DFALL))
    DFALL = DFALL[tmp].reset_index(drop=True)
    assert len(DFALL)>0.1 * n1
    savepath = f"{SAVEDIR}/counts_after_prune_same_epochset.txt"
    grouping_print_n_samples(DFALL, ["date", "epochset", "epoch"], savepath=savepath)
    
    # Take just from shapes epoch
    savepath = f"{SAVEDIR}/counts_before_prune_shape_epochorig.txt"
    grouping_print_n_samples(DFALL, ["date", "epoch_orig"], savepath=savepath)
    if animal == "Pancho":
        epochs_shape = ["AnBmCk2"]
        epochs_direction = ["L"]
    elif animal == "Diego":
        epochs_shape = ["llCV3b", "gramD5"]
        epochs_direction = []
    else:
        assert False
    # Check inputed all
    assert all(DFALL["epoch_orig"].isin(epochs_shape + epochs_direction)), "user should update the lists"
    DFALL = DFALL[DFALL["epoch_orig"].isin(epochs_shape)].reset_index(drop=True)
    savepath = f"{SAVEDIR}/counts_after_prune_shape_epochorig.txt"
    grouping_print_n_samples(DFALL, ["date", "epoch_orig"], savepath=savepath)

    # Rename, so they are groyuped
    DFALL["epoch_orig"] = "SHAPES"

    ### Get agg
    from pythonlib.tools.pandastools import aggregGeneral
    DFALL_AGG = aggregGeneral(DFALL, ["date", "epochset", "num_grammars", "epoch_orig", "epoch", "bregion_expt", "microstim_status", "character"], ["success_binary_quick"])
    print(len(DFALL))
    print(len(DFALL_AGG))

    ### IGNORE, catplots, they are not that informative.
    if False:
        # For each date, get the change in performance as a fraction
        from pythonlib.tools.pandastools import summarize_featurediff
        # dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = summarize_featurediff(DFALL, 
        #                                                                                     "microstim_status", ["on", "off"], 
        #                                                                                     ["success_binary_quick"], ["date", "bregion_expt"],
        #                                                                                     return_dfpivot=True)

        dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = summarize_featurediff(DFALL, 
                                                                                            "microstim_status", ["on", "off"], 
                                                                                            ["success_binary_quick"], ["date", "bregion_expt", "character"],
                                                                                            return_dfpivot=True)



        import seaborn as sns
        fig = sns.catplot(data=dfsummaryflat, x="bregion_expt", y="value", hue="bregion_expt")
        fig = sns.catplot(data=dfsummaryflat, x="bregion_expt", y="value", kind="bar", errorbar="se")
        # fig = sns.catplot(data=dfsummaryflat, x="bregion_expt", y="value", kind="violin")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        rotateLabel(fig)
        import seaborn as sns
        from pythonlib.tools.snstools import rotateLabel
        fig = sns.catplot(data=DFALL, x="microstim_bregion", y="success_binary_quick", col="date", kind="bar", errorbar="se", col_wrap=4)
        rotateLabel(fig)

    ##### PLOTS
    ### (1) Scatter plots
    from pythonlib.dataset.dataset_analy.microstim import mult_stats_plots_scatter
    mult_stats_plots_scatter(DFALL, DFALL_AGG, DATES_TO_SKIP, SAVEDIR)

    ### Get significance for each bregion -- shuffle whether stim or not
    from pythonlib.dataset.dataset_analy.microstim import mult_stats_permutation_test
    skip=True
    for n_grammars in [1, None]:
        if n_grammars is None:
            # Combine all
            dfthis = DFALL[(~DFALL["date"].isin(DATES_TO_SKIP))].reset_index(drop=True)
        else:
            dfthis = DFALL[(~DFALL["date"].isin(DATES_TO_SKIP)) & (DFALL["num_grammars"]==n_grammars)].reset_index(drop=True)

        savedir = f"{SAVEDIR}/permutation-ngrams={n_grammars}-skipdates={skip}-nperms={nperms}"
        os.makedirs(savedir, exist_ok=True)
        dfres, dfres_flat = mult_stats_permutation_test(dfthis, nperms=nperms, savedir=savedir)

        ##### Plot results from shuffling.
        from pythonlib.dataset.dataset_analy.microstim import mult_stats_permutation_test_plot
        mult_stats_permutation_test_plot(dfres, dfres_flat, savedir)
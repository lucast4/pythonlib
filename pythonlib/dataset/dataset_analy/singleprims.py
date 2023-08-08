""" Daily plots for single prim expts
"""

from pythonlib.tools.plottools import savefig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pythonlib.tools.pandastools import applyFunctionToAllRows
import os
import seaborn as sns

def preprocess_dataset(D, PLOT=True):
    """
    """
    SAVEDIR = D.make_savedir_for_analysis_figures("singleprims")
    D.taskclass_gridsize_assign_column()

    from pythonlib.dataset.dataset_strokes import DatStrokes
    DS = DatStrokes(D)

    if PLOT:
        ### PLOt drawings of all conjucntions
        plot_drawings_grid_conjunctions(DS, SAVEDIR)
        # DS.plotshape_multshapes_egstrokes_grouped_in_subplots(key_to_extract_stroke_variations_in_single_subplot="gridsize", n_examples=1);


        # Plot (row, col, location in suplot) = (size, shape, loc)
        sdir = f"{SAVEDIR}/shape_size_loc_drawings"
        os.makedirs(sdir, exist_ok=True)
        niter = 5
        for i in range(niter):
            fig_beh, fig_task = DS.plotshape_row_col_size_loc()
            savefig(fig_beh, f"{sdir}/iter_{i}-beh.pdf")      
            savefig(fig_task, f"{sdir}/iter_{i}-task.pdf")      
            plt.close("all")
            
        ### PRint and plot conjucntions for neural analy
        from neuralmonkey.metadat.analy.anova_params import conjunctions_print_plot_all
        # which_level="trial"
        # ANALY_VER = "seqcontext"
        # animal = D.animals()[0]
        conjunctions_print_plot_all([D], SAVEDIR, ANALY_VER="singleprim")
        plt.close("all")

        ### Plot scores (beh)
        sdir = f"{SAVEDIR}/beh_eval_scores"
        os.makedirs(sdir, exist_ok=True)

        list_block = D.Dat["block"].unique().tolist()
        for bk in list_block:
            df = D.Dat[(D.Dat["block"]==bk)]
            if len(df)>15:
                for yval in ["rew_total", "beh_multiplier"]:

                    print("Plotting for (block, yval): ", bk, yval)

                    fig = sns.catplot(data=df, x="gridsize", y=yval, hue="aborted", row="session", aspect=3, height=2, jitter=True, alpha=0.2)
                    savefig(fig, f"{sdir}/block_{bk}-yval_{yval}-1.pdf")

                    fig = sns.catplot(data=df, x="gridsize", y=yval, hue="aborted", row="session", aspect=3, height=2, kind="bar")
                    savefig(fig, f"{sdir}/block_{bk}-yval_{yval}-2.pdf")      

                fig = sns.catplot(data=df, x="gridsize", y="aborted", row="session", aspect=3, height=2, kind="bar")
                savefig(fig, f"{sdir}/block_{bk}-aborted.pdf")      
                
                plt.close("all")

    return DS, SAVEDIR


def plot_drawings_grid_conjunctions(DS, SAVEDIR):
    
    # pull out strokes into a column called "strokes"
    list_stroke = []
    for ind in range(len(DS.Dat)):
        list_stroke.append([DS.Dat.iloc[ind]["Stroke"]()])
    DS.Dat["strokes_beh"] = list_stroke

    ### Plot grid showing example conjunctions of size, shape, and location
    sdir = f"{SAVEDIR}/drawings_grid_conjunctions"
    os.makedirs(sdir, exist_ok=True)
    niter = 3
    for i in range(niter):
        fig, _ = DS.plotshape_row_col_vs_othervar(rowvar="gridsize", colvar="shape", n_examples_per_sublot=3)
        path = f"{sdir}/gridsize-vs-shape-{i}.pdf"
        savefig(fig, path)

        fig, _ = DS.plotshape_row_col_vs_othervar(rowvar="gridloc", colvar="shape", n_examples_per_sublot=3)
        path = f"{sdir}/gridloc-vs-shape-{i}.pdf"
        savefig(fig, path)

        fig, _ = DS.plotshape_row_col_vs_othervar(rowvar="gridsize", colvar="gridloc", n_examples_per_sublot=3)
        path = f"{sdir}/gridsize-vs-gridloc-{i}.pdf"
        savefig(fig, path)

        plt.close("all")

    
    


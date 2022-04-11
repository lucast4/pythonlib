""" General summaries that can apply for all experiments.
- Plotting raw data organized by epohcs, tasks, etc.
- Timecourses of scores

This is related to notebook:
analy_dataset_summarize_050621
"""


def plot_summary_drawing_examplegrid(Dthis, SAVEDIR_FIGS, subfolder, yaxis_ver="date", 
        LIST_N_PER_GRID = [1], strokes_by_order=True):
    """ 
    Plot grid, where y axis is usually date (or epoch) and x axis are each unique task.
    Plots one (or n) examples per grid cell. Plots multiple iterations to get many random examples, 
    in separate plots.
    Useful for comparing same task across conditions.
    This relates to flag PLOT_EXAMPLE_DATEVTASK
    PARAMS:
    - Dthis, Dataset instance
    - SAVEDIR_FIGS, base dir for figures
    - subfolder, string name of subfolder within SAVEDIR_FIGS. makes it if doesnt exist.
    - yaxis_ver, string name, indexes column. usually "date" or "epoch"
    - LIST_N_PER_GRID, list of ints, where each int is n examples to plot per grid.
    RETURNS:
    - saves figures.
    """
    from pythonlib.dataset.plots import plot_beh_grid_grouping_vs_task
    import os
    
    # 1) Generate save dirs
#     sdirthis = f"{SAVEDIR_FIGS}/date_vs_task/stage_{s}"
    sdirthis = f"{SAVEDIR_FIGS}/{yaxis_ver}_vs_task/{subfolder}"
    os.makedirs(sdirthis, exist_ok=True)
    print(" ** SAVING AT : ", sdirthis)
    
    # 2) one plot for each task category
    taskcats = Dthis.Dat["task_stagecategory"].unique()
    for tc in taskcats:
        tasklist = Dthis.Dat[Dthis.Dat["task_stagecategory"]==tc]["character"].unique()
        for max_n_per_grid in LIST_N_PER_GRID:
            
            # How many iterations?
            if max_n_per_grid==1:
                n = 4
            else:
                n=1
            
            # Plots Iterate, since is single plots.
            for i in range(n):
                figb, figt = plot_beh_grid_grouping_vs_task(Dthis.Dat, yaxis_ver, 
                                                            tasklist, 
                                                            max_n_per_grid=max_n_per_grid, 
                                                            plotkwargs={"strokes_by_order":strokes_by_order})
                figb.savefig(f"{sdirthis}/{tc}-npergrid{max_n_per_grid}-iter{i}-beh.pdf");
                figt.savefig(f"{sdirthis}/{tc}-npergrid{max_n_per_grid}-iter{i}-task.pdf");
                assert False
            plt.close("all")
            
            
            
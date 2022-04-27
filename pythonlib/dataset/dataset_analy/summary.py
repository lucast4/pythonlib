""" General summaries that can apply for all experiments.
- Plotting raw data organized by epohcs, tasks, etc.
- Timecourses of scores

This is related to notebook:
analy_dataset_summarize_050621
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_summary_drawing_examplegrid(Dthis, SAVEDIR_FIGS, subfolder, yaxis_ver="date", 
        LIST_N_PER_GRID = [1], strokes_by_order=True, how_to_split_files = "task_stagecategory"):
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
    - how_to_split_files, string col name in Dat, will make seprate plots for each level of this.

    RETURNS:
    - saves figures.
    """
    print("*** FIX - don't subsample trials. instead make mulitpel plots")
    from pythonlib.dataset.plots import plot_beh_grid_grouping_vs_task
    import os
    
    # 1) Generate save dirs
#     sdirthis = f"{SAVEDIR_FIGS}/date_vs_task/stage_{s}"
    sdirthis = f"{SAVEDIR_FIGS}/{yaxis_ver}_vs_task/{subfolder}"
    os.makedirs(sdirthis, exist_ok=True)
    print(" ** SAVING AT : ", sdirthis)
    
    # 2) one plot for each task category
    taskcats = Dthis.Dat[how_to_split_files].unique()
    for tc in taskcats:
        tasklist = Dthis.Dat[Dthis.Dat[how_to_split_files]==tc]["character"].unique()
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
            plt.close("all")
            
            
############## PRINT THINGS


def print_save_task_information(D, SDIR_MAIN):
    """ Print into text file summary of tasks presented across experiment, broken down
    into task categories and then unique tasks.
    Also print sample sizes
    """

    def _print_save_task_information(D, SDIR_MAIN, date=None):
        """ Print into text file summary of tasks presented across experiment, broken down
        into task categories and then unique tasks.
        Also print sample sizes
        """
        from pythonlib.tools.expttools import writeDictToYaml
        
        df = D.Dat
        
        # Specific date
        if date is not None:
            df = df[df["date"]==date]
            suffix = f"-{date}"
        else:
            suffix = "-ALLDATES"
        sdir = f"{SDIR_MAIN}/task_printed_summaries"
        os.makedirs(sdir, exist_ok=True)
        
        # get task categories    
        taskcats = df["task_stagecategory"].unique()

        catdict = {}
        catndict = {}
        for cat in taskcats:
            dfthis = df[df["task_stagecategory"] == cat]

            taskdict = {}
            tasks = sorted(dfthis["unique_task_name"].unique())
            for t in tasks:
                n = sum(dfthis["unique_task_name"]==t)
                taskdict[t] = n

            catdict[cat] = taskdict

            # Information about this category

            catndict[cat] = {
                "n_trials":len(dfthis),
                "n_unique_tasks":len(tasks),
                "min_ntrials_across_tasks":min(taskdict.values()),
                "max_ntrials_across_tasks":max(taskdict.values()),
            }

        writeDictToYaml(catdict, f"{sdir}/all_tasks_bycategory{suffix}.yaml")
        writeDictToYaml(catndict, f"{sdir}/all_categories{suffix}.yaml")

    # Across all dates
    _print_save_task_information(D, SDIR_MAIN)

    # Separate for each date
    for date in D.Dat["date"].unique():
        _print_save_task_information(D, SDIR_MAIN, date)
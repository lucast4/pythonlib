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


def plotall_summary(animal, expt, rulelist=[], savelocation="main"):
    """
    PARAMS:
    - rulelist, list of str, rules, each a datsaet. leave None to load all rules that you find.
    - savelocation, string code for where to save, in {'main', 'daily'}
    """

    ##### Related to "Task scores" (porting from probedat analyses, will replace it)
    ### NOTE: currently is partly hard coded for biasdir expts (angle stuff). need to modify to 
    # be more general.

    # Load dataset

    # INPUTS:
    # for animal in ["Pancho"]:
    #     for expt in ["lines5"]:

    # animal = "Pancho"
    # expt = "lines5"
    # rule = ["straight", "bent"]
    # for animal in ["Diego", "Pancho"]:
    # #     animal = "Pancho"
    #     expt = "linecircle"
    #     rule = ["null"]

    from pythonlib.dataset.dataset import Dataset
    from pythonlib.globals import PATH_ANALYSIS_OUTCOMES, PATH_DATA_BEHAVIOR_RAW

    # --------------- INPUT PARAMS
    PLOT_OVERVIEW = True

    # - related to fixed test tasks:
    PLOT_EXAMPLE_DATEVTASK = True
    PLOT_ALL_EACHTRIAL = True
    PLOT_TEST_TASK_IMAGES = False
    PLOT_ALL_DRAWINGS_IN_GRID = False # example trials x unique tasks, one plot for each task set. [may take a while]

    # - related to train tasks
    PLOT_TRAIN_GRID = True
    PLOT_TRAIN_REPEATED = False # things repeated even though random.

    # - other stuff
    PLOT_RANDOM_GRID_EXAMPLES= False

    # base_dir = "/gorilla1"
    # drawmonkey_dir = "/home/lucast4/drawmonkey"

    # EXPT PARAMS
    # expt = "concrete1" #run for chunkbyshape1, chunkbyshape2
    # RULELIST = []


    if rulelist is None:
        # Then find all the rules automatically
        from pythonlib.dataset.dataset_preprocess.general import get_rulelist
        rulelist = get_rulelist(animal, expt)

    # Saving location
    if savelocation=="main":
        SDIR_MAIN = f"{PATH_ANALYSIS_OUTCOMES}/main/simple_summary/{animal}-{expt}-{'_'.join(rulelist)}"
    else:
        datethis = rulelist[0]
        assert len(rulelist)==1, "if daily, then this must be a single date"
        SDIR_MAIN = f"{PATH_DATA_BEHAVIOR_RAW}/Pancho/{datethis}/figures/dataset"
    # SDIR_MAIN = f"{base_dir}/analyses/main/simple_summary/{animal}-{expt}-{'_'.join(rulelist)}"
    os.makedirs(SDIR_MAIN, exist_ok=True)
    SAVEDIR_FIGS = f"{SDIR_MAIN}/FIGS/drawfigs"
    SDIR_FIGS = SAVEDIR_FIGS
    os.makedirs(SAVEDIR_FIGS, exist_ok=True)
    MAX_COLS = 20

    D = Dataset([])
    D.load_dataset_helper(animal, expt, ver="mult", rule=rulelist)
    GROUPING = D.MetadatPreprocess["GROUPING"]
    GROUPING_LEVELS = D.MetadatPreprocess["GROUPING_LEVELS"]
    FEATURE_NAMES = D.MetadatPreprocess["FEATURE_NAMES"]
    SCORE_COL_NAMES = D.MetadatPreprocess["SCORE_COL_NAMES"]
    
    # 1) extract supervision params to Dat
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    def F(x, abbrev = True):
        S = x["supervision_params"]

        # sequence
        if S["sequence_on"]==False:
            seq = "off"
        else:
            seq = S["sequence_ver"]

        # dynamic
        dyna = S["guide_dynamic_strokes"]
        if dyna:
            d = 1
        else:
            d=0

        # color
        colo = S["guide_colored_strokes"]
        if colo:
            c=1
        else:
            c=0

        # give a condensed name
        if seq=="off":
            s = 0
        elif seq=="v3_remove_and_show":
            s = 3
        elif seq=="v3_remove_and_fadein":
            s = 4
        elif seq=="v3_noremove_and_show":
            s = 5
        elif seq=="v3_noremove_and_fadein":
            s = 6
        elif seq=="v4_fade_when_touch":
            s=1
        elif seq=="v4_remove_when_touch":
            s=2
        elif seq=="unknown":
            s=7
        elif seq=="objectclass_active_chunk":
            s=8
        else:
            print(seq)
            print("what is this?")
            assert False

        if abbrev:
            return (s,d,c)
        else:
            return seq, dyna, colo
    if expt in ["chunkbyshape1"]:
        # TODO: define supervision stage using ObjectCalss only
        if False:
            # Start from here
            TT = T.Params["input_params"]
            Tnew = TT.get_tasknew()
            Tnew["Objects"]["Params"]
        else:
            # For now, just hack it 
            Fthis = lambda x: (0,0,0)
    else:
        Fthis = F
    D.Dat = applyFunctionToAllRows(D.Dat, Fthis, "supervision_stage")

    
    ##### THINGS THAT SHOULD ALWAYS DO, SINCE IS QUICK (PRINTING THINGS)
    from pythonlib.dataset.dataset_analy.summary import print_save_task_information
    print_save_task_information(D, SDIR_MAIN)

    #### PLOT OVERVIEW OF EXPERIMENT
    if PLOT_OVERVIEW:
        figlist = D.plotOverview()
        for i, fig in enumerate(figlist):
            fig.savefig(f"{SDIR_MAIN}/overview_{i}.pdf")

        # Plot overview, separating by supervision
        fig = sns.catplot(data=D.Dat, x="block", y="supervision_stage", col="task_stagecategory", row="date")
        fig.savefig(f"{SDIR_MAIN}/overview_supervisionstage.pdf")
        plt.close("all")

    ##### Any random tasks that are repeated?


    ## Plot all test tasks stimuli
    # Get all uique test tasks
    if PLOT_TEST_TASK_IMAGES:
        # TODO: split by set category, and features.
        # get list of unique tasks
        df = D.filterPandas({"monkey_train_or_test":["test"]}, "dataframe")
        tasklist = sorted(df["character"].unique())
        if len(tasklist)>50:
            print(len(tasklist))
            assert False, "break it up into multiple plots"

        # Only keep one per unique task
        indices = []
        for task in tasklist:
            ind = D.filterPandas({"character":[task]})[0] # get the first
            indices.append(ind)

        # Plot and save
        sdirthis = f"{SAVEDIR_FIGS}/all_test_task_images"
        os.makedirs(sdirthis, exist_ok=True)

        fig, axes, _ = D.plotMultTrials2(indices, "strokes_task", titles=tasklist, SIZE=3.2, 
                          plotkwargs={"naked_axes":True});
        fig.savefig(f"{sdirthis}/all_images_1.pdf")
        plt.close("all")

    if PLOT_ALL_DRAWINGS_IN_GRID:
        from pythonlib.dataset.dataset_analy.summary import plot_all_drawings_in_grid
        plot_all_drawings_in_grid(D, SAVEDIR_FIGS, MAX_ROWS=10)
        plt.close("all")

    ##### Plot drawing behavior over entire experiment.
    ############ 1) Fixed tasks (train and test), plot mult trials per task, in structured way (GOOD)
    # sort based on supervision stage
    stages = D.Dat["supervision_stage"].unique()
    for s in stages:
        for traintest in ["test", "train"]:

            Dthis = D.filterPandas({"random_task":[False], "monkey_train_or_test":[traintest], "supervision_stage":[s]}, "dataset")
            if len(Dthis.Dat)>0:

                #### 2) A single category, over all time
                if PLOT_EXAMPLE_DATEVTASK:
                    from pythonlib.dataset.dataset_analy.summary import plot_summary_drawing_examplegrid
                    plot_summary_drawing_examplegrid(Dthis, SAVEDIR_FIGS, f"{traintest}/stage{s}", 
                                     "date")
                    plot_summary_drawing_examplegrid(Dthis, SAVEDIR_FIGS, f"{traintest}/stage{s}", 
                                     "epoch")

                #### 1) A single task, over time.
                if PLOT_ALL_EACHTRIAL:

                    sdirthis = f"{SAVEDIR_FIGS}/each_task_all_trials/{traintest}/stage_{s}"
                    os.makedirs(sdirthis, exist_ok=True)

                    tasklist = Dthis.Dat["character"].unique()

                    from pythonlib.dataset.plots import plot_beh_grid_singletask_alltrials, plot_beh_grid_flexible_helper
                    for task in tasklist:
                    #     task = "mixture2_1-savedset-50-39276"
                        row_variable = "date"
                        if False:
                            # Old version
                            figb, figt = plot_beh_grid_singletask_alltrials(D, task, row_variable, max_cols = MAX_COLS)
                        else:
                            # New version, plotting trialcode, and coloring by order
                            Dthisthis = Dthis.filterPandas({"character":task}, "dataset")
#                             # if too many trials, split this into multiple datasets
#                             ntot = len(Dthis.Dat)
                            figb, figt = plot_beh_grid_flexible_helper(Dthisthis, "date", "trial", 
                                                                       plotkwargs={"strokes_by_order":True}, 
                                                                       max_cols=150)
                        figb.savefig(f"{sdirthis}/{task}-beh.pdf");
                        figt.savefig(f"{sdirthis}/{task}-task.pdf");
                        plt.close("all")


    ######### PLOT RANDOM EXAMPLES IN GRID, EACH PLOT A SEPARATE TASK SET.
    if PLOT_RANDOM_GRID_EXAMPLES:
        sdirthis = f"{SAVEDIR_FIGS}/random_grid_examples"
        os.makedirs(sdirthis, exist_ok=True)
        import random
        nplot = 20 # per grid
        niter = 3
        for i in range(niter):
            for taskset, inds in D.grouping_get_inner_items("task_stagecategory", "index").items():
                # get subset of inds
                if len(inds)>nplot:
                    inds = random.sample(inds, nplot)
                inds = sorted(inds)
                fig1 = D.plotMultTrials(inds, "strokes_task", titles=inds);
                fig2 = D.plotMultTrials(inds, color_by="order", add_stroke_number=False);

                fig1.savefig(f"{sdirthis}/{taskset}-iter_{i}-task.pdf")
                fig2[0].savefig(f"{sdirthis}/{taskset}-iter_{i}-beh.pdf")

    ######### 2) Plot all training tasks (lumps random and fixed) into random grids.
    if PLOT_TRAIN_GRID:
        Dtrain = D.filterPandas({"monkey_train_or_test":["train"]}, "dataset")

        # 1) plot random train tasks (beh + test)
        # separate plots for each level of grouping (e..g, epoch), each block, each supervision kind, date
        stages = Dtrain.Dat["supervision_stage"].unique()
        dates = Dtrain.Dat["date"].unique()
        blocks = Dtrain.Dat["block"].unique()

        nrand = 20
        niter = 1 
        nmin = 3

        for lev in GROUPING_LEVELS:
            for d in dates:
                for s in stages:

                    for b in blocks:
                        inds = Dtrain.filterPandas({
                            GROUPING:[lev],
                            "date":[d],
                            "supervision_stage":[s],
                            "block":[b]})
                        if len(inds)>nmin:
                            print("running: ", (lev, d, s, b))

                            # Savedir
                            sdirthis = f"{SDIR_MAIN}/FIGS/drawfigs/traintasks_randomgrid/{lev}/{d}/{s}"
                            os.makedirs(sdirthis, exist_ok=True)

                            if len(inds)<1.5*nrand:
                                # dont plot multipel times if not that many trials.
                                niterthis = 1
                            else:
                                niterthis = niter

                            for i in range(niterthis):
                                # Plot these inds
                                figbeh, _, indsthis = Dtrain.plotMultTrials(inds, "strokes_beh", return_idxs=True, nrand=nrand,
                                                                        naked_axes=True, add_stroke_number=False)
                                figtask = Dtrain.plotMultTrials(indsthis, "strokes_task", return_idxs=False, nrand=nrand,
                                                                       naked_axes=True, add_stroke_number=False)

                                # SAVE
                                figbeh.savefig(f"{sdirthis}/block{b}-iter{i}-beh.pdf")
                                figtask.savefig(f"{sdirthis}/block{b}-iter{i}-task.pdf")             

                            plt.close("all")

    ####### Random train tasks that were repeated.
    if PLOT_TRAIN_REPEATED:
        nmin_trials= 6 # min num repeated trials.
        nmin_tasksplot = 100 # take top 100 tasks.
        Dtrain = D.filterPandas({"random_task":[True], "monkey_train_or_test":["train"]}, "dataset")

        # Hash each task shapes
        list_shapeshash = [] # each tasks shape hash

        for i in range(len(Dtrain.Dat)):
            list_shapeshash.append(Dtrain.Dat.iloc[i]["Task"].get_shapes_hash())


        ## Get all inds that have the same task (defined by shapes)
        # 1) assign back into dat the hash
        Dtrain.Dat["Task_shapeshash"] = list_shapeshash

        # 2) plot one
        if False:
            import random
            thishash = random.choice(list_shapeshash)
            inds = Dtrain.filterPandas({"Task_shapeshash":[thishash]})
            print(inds)
            Dtrain.plotMultTrials(inds);
            inds = inds[:20]
            Dtrain.plotMultTrials(inds, "strokes_task");

        sdirthis = f"{SDIR_MAIN}/FIGS/drawfigs/traintasks_repeatedtrials"
        sdirthis_raw = f"{SDIR_MAIN}/FIGS/drawfigs/traintasks_repeatedtrials/each_task"
        os.makedirs(sdirthis, exist_ok=True)
        os.makedirs(sdirthis_raw, exist_ok=True)

        # How often where tasks repeated?
        a = Dtrain.Dat["Task_shapeshash"].value_counts().tolist()

        fig = plt.figure()
        plt.hist(a, range(max(a)), log=True);
        plt.ylabel('counts');
        plt.xlabel('num repetitions per unique task')
        plt.title(f"This many unique tasks: {len(a)}")
        fig.savefig(f"{sdirthis}/num_repeats_per_task-1.pdf")

        fig = plt.figure()
        plt.hist(a, range(max(a)), log=False);
        plt.ylabel('counts');
        plt.xlabel('num repetitions per unique task')
        plt.title(f"This many unique tasks: {len(a)}")
        fig.savefig(f"{sdirthis}/num_repeats_per_task-2.pdf")


        ### For each unique task, plot across days
        # get all unique tasks with greater than N trials
        a = Dtrain.Dat["Task_shapeshash"].value_counts()
        a = a[a>nmin_trials]
        a = a[:nmin_tasksplot]

        # save a dict of names
        from pythonlib.tools.expttools import writeDictToYaml
        namesdict = {i:a.index[i] for i in range(len(a))}
        writeDictToYaml(namesdict, f"{sdirthis}/tasknames.yaml")

        for i in range(len(a)):
            name = a.index[i]
            inds = Dtrain.filterPandas({"Task_shapeshash":[name]})
            inds = inds[:20]
            figbeh, _, _ = Dtrain.plotMultTrials(inds);
            figtask = Dtrain.plotMultTrials(inds, "strokes_task");

            figbeh.savefig(f"{sdirthis_raw}/task{i}-beh.pdf")
            figtask.savefig(f"{sdirthis_raw}/task{i}-task.pdf");

            plt.close("all")    

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
                                                            max_n_trials=200,
                                                            plotkwargs={"strokes_by_order":strokes_by_order})
                figb.savefig(f"{sdirthis}/{tc}-npergrid{max_n_per_grid}-iter{i}-beh.pdf");
                figt.savefig(f"{sdirthis}/{tc}-npergrid{max_n_per_grid}-iter{i}-task.pdf");
            plt.close("all")
            

def plot_all_drawings_in_grid(Dthis, SAVEDIR_FIGS, MAX_COLS = 150, MAX_ROWS = 5, strokes_by_order = True):
    """ Plot beh and task in separate grids, where y axis is examples and x is unique tasks, where
    each plot is a specific task group (e.g., task set).
    """
    from ..plots import plot_beh_grid_flexible_helper, plot_beh_grid_grouping_vs_task

    # Make saving directory
    sdirthis = f"{SAVEDIR_FIGS}/alltrial_vs_task_bycat"
    os.makedirs(sdirthis, exist_ok=True)

    # One plot for each task category
    list_taskcat = Dthis.filterPandas({"random_task":[False]}, "dataframe")["task_stagecategory"].unique().tolist()
    for cat in list_taskcat:
        Dthiscat = Dthis.filterPandas({"task_stagecategory":[cat]}, "dataset")
        figbeh, figtask = plot_beh_grid_flexible_helper(Dthiscat, row_group ="trial", col_group="unique_task_name", 
                                      max_rows=MAX_ROWS, max_cols=MAX_COLS, plotkwargs={"strokes_by_order":strokes_by_order})
        
        figbeh.savefig(f"{sdirthis}/{cat}-beh.pdf");
        figtask.savefig(f"{sdirthis}/{cat}-task.pdf");
                
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

    # Overrivew of each trial-block-epoch
    D.print_trial_block_epoch_summary(savedir=SDIR_MAIN)



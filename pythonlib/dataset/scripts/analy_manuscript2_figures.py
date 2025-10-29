"""
All behaviroal figures in manuscript
Fig 1 - 3
Linked to notebook: 240912_MANUSCRIPT_FIGURES_1_shapes.

"""

# %cd ..
# from tools.utils import * 
# from tools.plots import *
# from tools.analy import *
# from tools.calc import *
# from tools.analyplot import *
# from tools.preprocess import *
# from tools.dayanalysis import *

# from pythonlib.drawmodel.analysis import *
# from pythonlib.tools.stroketools import *
# import pythonlib
from pythonlib.dataset.dataset import load_dataset_notdaily_helper, load_dataset_daily_helper
import pickle
import seaborn as sns
import os
import matplotlib.pyplot as plt
from pythonlib.tools.snstools import rotateLabel
import pandas as pd
from pythonlib.tools.plottools import savefig
import numpy as np

SAVEDIR_ALL = "/lemur2/lucas/analyses/manuscripts/2_syntax"

def fig1_generalize_wrapper(animal, date):
    """
    Load, extract, preprocess, plot, and then save, for a single date.
    Example:
        animal = "Diego"
        date = 230118 # ss
    """

    # (1) Load data
    D = load_dataset_daily_helper(animal, date)
    
    # (2) Preprocess and extract
    bm, DM, SDIR = fig1_generalize_1_extract_preprocess(D)

    # (3) Merge into D
    fig1_generalize_1b_merge_with_D(D, bm, DM)

    # Save things
    D.save(SDIR)

    # (4) Plots
    fig1_generalize_2_plot(D, bm, DM, SDIR)


def fig1_generalize_1_extract_preprocess(D):
    """
    Extract scores (diff kinds, eg, n circles), for monkey and
    different models. 
    """
    from pythonlib.drawmodel.task_features import shapes_has_separated_cases_of_shape
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.dataset.modeling.discrete import tasks_categorize_based_on_rule_mult
    import os
    from pythonlib.tools.pandastools import stringify_values_column
    from pythonlib.behmodelholder.preprocess import generate_scored_beh_model_data_matlabrule, generate_scored_beh_model_data_long, generate_diagnostic_model_data
    from pythonlib.tools.expttools import writeDictToYaml, writeDictToTxtFlattened

    SDIR = D.make_savedir_for_analysis_figures("diagnostic_model")
    animal = D.animals(force_single=True)[0]
    expt = D.expts(force_single=True)[0]
    if False:
        if animal=="Diego" and "gridlinecircle3" in expt:
            list_rule_across_dates = ["LCr2", "CLr2", "LolDR"]
            shape_key = "shapeabstract"
        elif animal == "Pancho" and ("grammar2" in expt):
            list_rule_across_dates = ["(AB)n", "AnBm1a", "AnBm2", "AnBmHV"]
            shape_key = "shape"
        else:
            print(animal)
            print(expt)
            assert False
    else:
        # Decided to not try to use rules from across dates, as this leads to a maess.
        list_rule_across_dates = None
        shape_key = D.grammarparses_rules_shape_AnBmCk_get_shapekey()

    # shapes_across_dates = D.grammarparses_rules_shape_AnBmCk_get_shapes_UNORDERED(list_rule_across_dates)    
    
    # Get rules infor for this date
    rulesinfo = D.grammarparses_rules_shapes_summary_simple()
    list_rule_this_dates = rulesinfo["rules"]
    # shapes_this_date = rulesinfo["shapes_used"]
    # Only score for the shapes that at some point act as the first shape
    # Get the sahpes that play role of "A" (ie first shape in sequence)
    # first_shapes_this_date = []
    # for rule in list_rule_this_dates:
    #     shapes = D._grammarparses_rules_shape_AnBmCk_get_shape_order(rule)
    #     first_shapes_this_date.append(shapes[0])
    if False:
        first_shapes_this_date = rulesinfo["shapes_used_first_in_seq"]
    else:
        # This works better generally. For two-shape epochs, gets both epochs, while above just gets one.
        first_shapes_this_date = []
        for ep in D.Dat["epoch"].unique().tolist():
            shapes = D._grammarparses_rules_shape_AnBmCk_get_shape_order(ep)
            first_shapes_this_date.append(shapes[0])

    writeDictToYaml(rulesinfo, f"{SDIR}/rules_info.yaml")
    writeDictToTxtFlattened(rulesinfo, f"{SDIR}/rules_info.txt")

    ### PREPROCESS
    ### Figure out if lines and circles are separated in space (e..g, 2 lines, separated by circle)
    # For each task, give it flag for whether it has separated shapes
    if False:
        # This becomes unwieldy with more than 2 shapes, which is the case if I automticaly input the shapes from above.
        # I don't use it, so just ignpre for now.
        for shape_to_check, shape_jump_over in [
            ("line", "circle"),
            ("circle", "line"),
            ]:
            def F(x):
                Task = x["Task"]
                # return shapes_has_separated_cases_of_this_shape(Task, shape_to_check)
                return shapes_has_separated_cases_of_shape(Task, shape_to_check, [shape_jump_over], ploton=False, shape_key=shape_key)
            D.Dat = applyFunctionToAllRows(D.Dat, F, f"shape_separated_{shape_to_check}")

    ## GrammarDat class, all things below
    # Categorize tasks based on rules
    list_rule_across_dates = tasks_categorize_based_on_rule_mult(D, list_rule_across_dates, return_list_rule=True)

    ### EXTRACT DATA
    # First, rename character so that it uses the actual (shape, loc) config
    # - this helps in that "random" tasks will be given real names
    D.Dat["character_orig"] = D.Dat["character"]
    D.Dat["character"] = D.Dat["taskconfig_shploc"]
    stringify_values_column(D.Dat, "character"
                            )
    # Which models to use in diagnostic model?
    # Which motifs to use?
    if False: # Entered by hand
        # LIST_MOTIFS = [
        #     ("repeat", {"shapekey":"shape",
        #                  "shape":"line-8-4-0",
        #                  "nmin":1,
        #                  "nmax":6,
        #                  "allowed_rank_first_token":[0]
        #                }),
        #     ("repeat", {"shapekey":"shape",
        #                  "shape":"line-8-3-0",
        #                  "nmin":1,
        #                  "nmax":6,
        #                  "allowed_rank_last_token":[-1]
        #                }),
        #     ("repeat", {"shapekey":"shape",
        #                  "shape":"line-11-1-0",
        #                  "nmin":1,
        #                  "nmax":6,
        #                  "allowed_rank_first_token":[0]
        #                }),
        #     ("repeat", {"shapekey":"shape",
        #                  "shape":"line-11-2-0",
        #                  "nmin":1,
        #                  "nmax":6,
        #                  "allowed_rank_last_token":[-1]
        #                })
        # ]

        LIST_MOTIFS = [
            ("repeat", {"shapekey":shape_key,
                        "shape":"line",
                        "nmin":1,
                        "nmax":6,
                        "allowed_rank_first_token":[0]
                    }),
        #     ("repeat", {"shapekey":shape_key,
        #                  "shape":"line",
        #                  "nmin":1,
        #                  "nmax":6,
        #                  "allowed_rank_last_token":[-1]
        #                }),
            ("repeat", {"shapekey":shape_key,
                        "shape":"circle",
                        "nmin":1,
                        "nmax":6,
                        "allowed_rank_first_token":[0]
                    }),
            ("lolli", {"list_orientation":["right", "down"],
                        "list_first_shape":["line"]}),
        ]

    else:
        # Get motifs automatically
        LIST_MOTIFS = []
        # for sh in shapes_this_date:
        for sh in first_shapes_this_date:
            LIST_MOTIFS.append((
                "repeat",
                {"shapekey":shape_key,
                "shape":sh,
                "nmin":1,
                "nmax":7,
                "allowed_rank_first_token":[0]
                }
            ))
        
        # Append more motifs (ie scores)
        if animal=="Diego":
            LIST_MOTIFS.append(
                ("lolli", {"list_orientation":["right", "down"],
                            "list_first_shape":["line"]}),
            )

    # Define the models (i.e., the rulestrings.)
    if True: # 
        list_rulestring = []
        for rule in list_rule_this_dates:
            rd = D.grammarparses_ruledict_rulestring_extract_flexible(rule)
            list_rulestring.append(rd["rulestring"])    
        LIST_MODELS = list_rulestring + ["rand-null-uni"]
        if animal=="Diego":
            LIST_MODELS = LIST_MODELS + ["ss-rankdir_nmax2-LCr2", "chmult-dirdir-LolDR"]
    else:
        # Hand-coded, older.
        LIST_MODELS = ["ss-rankdir-CLr2", "ss-rankdir-LCr2", "ss-rankdir_nmax2-LCr2", "chmult-dirdir-LolDR", "rand-null-uni"]

    COLS_TO_KEEP = ["taskcat_by_rule"]
    # COLS_TO_KEEP = ["taskcat_by_rule", "shape_separated_line", "shape_separated_circle"]
    bm, DM = generate_diagnostic_model_data(D, LIST_MODELS, LIST_MOTIFS, COLS_TO_KEEP)

    ### POSTPROCESS Other things.
    bm.DatLong["epoch_orig"] = bm.DatLong["epoch"]
    
    writeDictToYaml(LIST_MOTIFS, f"{SDIR}/LIST_MOTIFS.yaml")
    writeDictToYaml(DM.Params, f"{SDIR}/DIAGNOSTIC_MODEL_PARAMS.yaml")

    # Debuggin
    if False:
        g = D.GrammarDict["230118-1-3"]
        from pythonlib.dataset.modeling.discrete import rules_map_rulestring_to_ruledict
        rules_map_rulestring_to_ruledict("ss-rankdir-CLr2")
        g.ChunksListClassAll = {}
        g.parses_generate("ss-rankdir-CLr2")
        g.print_plot_summary()
        g.print_plot_summary()
        rs = "ss-rankdir-CLr2"
        D._grammarparses_parses_extract(0, [rs])
        D.grammarparses_print_plot_summarize(0)
        bm.DatLong[(bm.DatLong["agent"] == "model-ss-rankdir-CLr2") & (bm.DatLong["score_name"] == "ntokens-repeat-circle")]

    if False: # THis fails, unless you input all the rules in the above step of tasks_categorize_based_on_rule_mult()
        if animal=="Diego" and "gridlinecircle3" in expt:
            ### Further preprocessing, hand-coded
            # Extract n circles, lines, lollis.
            from pythonlib.tools.pandastools import applyFunctionToAllRows  

            def F(x):
                nline = x["taskcat_by_rule"][0][0]
                return nline
            bm.DatLong = applyFunctionToAllRows(bm.DatLong, F, "nline")

            def F(x):
                ncir = x["taskcat_by_rule"][0][2]
                return ncir
            bm.DatLong = applyFunctionToAllRows(bm.DatLong, F, "ncircle")


            def F(x):
                nlolli = x["taskcat_by_rule"][2]
                return nlolli
            bm.DatLong = applyFunctionToAllRows(bm.DatLong, F, "nlolli")



    return bm, DM, SDIR

def fig1_generalize_1b_merge_with_D(D, bm, DM):
    """
    RETURNS:
    - (noting) Modifies D.Dat
    """
    from pythonlib.tools.pandastools import slice_by_row_label

    animal = D.animals(force_single=True)[0]
    expt = D.expts(force_single=True)[0]

    # Extract useful things into D.Dat (so that you dont need BM to make pltos)
    # [Plots using D.Dat] [not using agents]
    D.grammarparses_successbinary_score_wrapper(False)

    # Get n circ and n lines
    # Get scores (just for monk)
    tcs = D.Dat["trialcode"].tolist()
    # DM._get_list_scores(1)
    if False:
        dftmp = slice_by_row_label(bm.DatWide, "trialcode", tcs, assert_exactly_one_each=True)
    else:
        for score_name in DM.scorenames_extract():
            # This works, but it lacks nline and ncircle
            # dftmp = DM.Dat[(DM.Dat["score_name"] == score_name ) & (DM.Dat["agent_kind"] == "monkey" )].reset_index(drop=True)
            dftmp = bm.DatLong[(bm.DatLong["score_name"] == score_name ) & (bm.DatLong["agent_kind"] == "monkey" )].reset_index(drop=True)
            dftmp = slice_by_row_label(dftmp, "trialcode", tcs, assert_exactly_one_each=True)
            D.Dat[score_name] = dftmp["score"]
            # if animal=="Diego" and "gridlinecircle3" in expt:
            #     D.Dat["shape_separated_line"] = dftmp["shape_separated_line"]
            #     D.Dat["shape_separated_circle"] = dftmp["shape_separated_circle"]
            #     D.Dat["nline"] = dftmp["nline"]
            #     D.Dat["ncircle"] = dftmp["ncircle"]
            #     D.Dat["nlolli"] = dftmp["nlolli"]
                
    if False:
        sns.catplot(data=D.Dat, x="ncircle", y="ntokens-repeat-circle", col="exclude_because_online_abort", jitter=True, alpha=0.25)

def fig1_generalize_2_plot(D, bm, DM, SDIR):
    """
    Plot results
    """
    from pythonlib.tools.pandastools import stringify_values

    DFLONG = stringify_values(bm.DatLong)

    # PRint things
    sdir = f"{SDIR}/repeats_aligned_to_onset_offset"
    os.makedirs(sdir, exist_ok=True)

    ### summary plots
    if False:
        bm.plot_score_cross_prior_model_splitby_v2(split_by="taskgroup", savedir=sdir)
    plot_each_char = False
    bm.plot_score_cross_prior_model_splitby_v2(split_by="taskcat_by_rule", savedir=sdir, plot_each_char=plot_each_char)


    ##### Correlation plot
    def _get_agent_kind_rule(agent):
        """ Helper just to get string name
        agent = "Diego-CLr2"
        """
        df = DFLONG[DFLONG["agent"]==agent]
        this = df["agent_kind"].unique().tolist()
        assert len(this)==1
        agent_kind = this[0]

        this = df["agent_rule"].unique().tolist()
        assert len(this)==1
        agent_rule = this[0]
        
        return agent_kind, agent_rule

    ### Scatterplots (monkey vs. model)
    from pythonlib.tools.plottools import plotScatter45

    # Extract Agged data
    bm.datextract_datlong_agg()
    bm._input_data_long_form()

    list_score_name = bm.DatLongAgg["score_name"].unique().tolist()
    list_agent_model = bm.DatLongAgg[bm.DatLongAgg["agent_kind"]=="model"]["agent"].unique().tolist()
    list_agent_monk = bm.DatLongAgg[bm.DatLongAgg["agent_kind"]=="monkey"]["agent"].unique().tolist()

    nmod = len(list_agent_model)
    nscor = len(list_score_name)
    SIZE=2.8
    xs = []
    ys = []
    for monk in list_agent_monk:
        fig, axes = plt.subplots(nmod, nscor, sharex=True, sharey=True, figsize=(nscor*SIZE, nmod*SIZE), squeeze=False)
        for i, model in enumerate(list_agent_model):
            for j, scorename in enumerate(list_score_name):
                ax = axes[i][j]
    #             dfthis = (bm.DatLongAgg["score_name"]==scorename) & (bm.DatLongAgg["score_name"]==scorename)
                modelkind, modelrule = _get_agent_kind_rule(model)
                vals_mod = bm.DatWideAgg[f"{scorename}|{modelrule}|{modelkind}"]
            
                kind, rule = _get_agent_kind_rule(monk)
                vals_monk = bm.DatWideAgg[f"{scorename}|{rule}|{kind}"]

                # remove nans
    #             ax.plot(vals_mod, vals_monk, 'ok')
                # print(max(vals_mod), max(vals_monk))
                ax.scatter(vals_mod, vals_monk, alpha=0.25)
                
                # sns.scatterplot(x=vals_mod, y=vals_monk, ax=ax, alpha=0.25, marker="o")
                # sns.histplot(x=vals_mod, y=vals_monk, ax=ax)
                
                # plotScatter45(vals_mod, vals_monk, ax, alpha=0.4)
                # if max(vals_monk)>4:
                #     assert False
    #             plotScatter45(vals_monk, vals_mod, ax, alpha=0.4)
                ax.set_xlabel(modelrule)
                ax.set_ylabel("monkey")
                ax.set_title(scorename)

                xs.extend(vals_mod)
                ys.extend(vals_monk)

        for ax in axes.flatten():
            from pythonlib.tools.plottools import set_axis_lims_square_bounding_data_45line
            set_axis_lims_square_bounding_data_45line(ax, xs, ys, 0.1)
            
        savefig(fig, f"{sdir}/scatter-{monk}.pdf")
    plt.close("all")

    ### Summary plots
    if "ncircle" in DFLONG.columns:
        for xval in ["nline", "ncircle", "nlolli"]:
            fig = sns.catplot(data=DFLONG, x=xval, y="score", hue="agent", col="score_name", kind="point")
            savefig(fig, f"{sdir}/scoremean_vs_{xval}.pdf")
        plt.close("all")

        ### Seprataed lines and circles (in space)
        dfthis = DFLONG[DFLONG["shape_separated_line"]==True]
        if len(dfthis)>0:
            for xval in ["nline", "ncircle", "nlolli"]:
                fig = sns.catplot(data=dfthis, x=xval, y="score", hue="agent", col="score_name", kind="point")
                savefig(fig, f"{sdir}/scoremean_vs_{xval}-shape_separated_line.pdf")

            for xval in ["nline", "ncircle", "nlolli"]:
                fig = sns.catplot(data=dfthis, x=xval, y="score", row="agent", col="score_name", alpha=0.3, height=3, jitter=True)
                savefig(fig, f"{sdir}/score_vs_{xval}-shape_separated_scatter-shape_separated_line.pdf")
            
            plt.close("all")

        dfthis = DFLONG[DFLONG["shape_separated_circle"]==True]
        if len(dfthis)>0:
            for xval in ["nline", "ncircle", "nlolli"]:
                fig = sns.catplot(data=dfthis, x=xval, y="score", hue="agent", col="score_name", kind="point")
                savefig(fig, f"{sdir}/scoremean_vs_{xval}-shape_separated_circle.pdf")

            for xval in ["nline", "ncircle", "nlolli"]:
                fig = sns.catplot(data=dfthis, x=xval, y="score", row="agent", col="score_name", alpha=0.3, height=3, jitter=True)
                savefig(fig, f"{sdir}/score_vs_{xval}-shape_separated_scatter-shape_separated_circle.pdf")
            plt.close("all")

        dfthis = DFLONG
        for xval in ["nline", "ncircle", "nlolli"]:
            fig = sns.catplot(data=dfthis, x=xval, y="score", row="agent", col="score_name", alpha=0.3, height=3, jitter=True)
            savefig(fig, f"{sdir}/score_vs_{xval}-shape_separated_scatter.pdf")
        plt.close("all")
        
        ### 2d plot (nlines, ncircles)
        from pythonlib.tools.pandastools import convert_to_2d_dataframe
        list_agent = DFLONG["agent"].unique().tolist()
        list_score_name = DFLONG["score_name"].unique().tolist()

        yval = "ncircle"
        xval = "nline"

        for agent in list_agent:
            for score_name in list_score_name:
                dfthis = DFLONG[(DFLONG["score_name"]==score_name) & (DFLONG["agent"]==agent)]
                
                _, fig, _, _ = convert_to_2d_dataframe(dfthis, xval, yval, True, agg_method="mean", val_name="score")
                fig.savefig(f"{sdir}/score_2d_heat-{yval}_vs_{xval}-{score_name}-{agent}.pdf")
                
                # fig = sns.catplot(data=dfthis, x="ncircle", hue="nline", y="score", kind="point", ci=68, alpha=0.2)
                fig = sns.catplot(data=dfthis, x="ncircle", hue="nline", y="score", kind="point", errorbar="se")
                fig.savefig(f"{sdir}/score_lines-{yval}_vs_{xval}-{score_name}-{agent}.pdf")
                plt.close("all")            

    #############################
    ### Other more detailed plots

    ### Plot example training tasks in a grid
    D.plotwrapper_training_task_examples(SDIR, niter = 3, nrand = 10)

    ### Plot example tasks categorized by rules
    sdir = f"{SDIR}/example_tasks_categorized_by_rules"
    os.makedirs(sdir, exist_ok=True)
    # plot exmaple tasks for each taskcat -- sanity check.
    for taskcat, inds in D.grouping_get_inner_items("taskcat_by_rule").items():
        fig = D.plotMultTrials(inds, which_strokes="strokes_task", add_stroke_number=True, nrand=20)
        savefig(fig, f"/tmp/{taskcat}.pdf")
        plt.close("all")

    ### Plot examples, scores, parses for a single character
    DM.plotwrapper_example_allcharacter(SDIR)

def fig1_generalize_3_postprocess_D(D, savedir, plot_examples=False, expect_max_two_shapes_each_day=False):
    """
    Modifies D to store useful things related to grammar behavior.

    PARAMS:
    - expect_max_two_shapes_each_day, bool, hacky, forcig you to voluterailty turn this True. If true, then 
    defines a new column "success_binary_quick_v2", which is true if  atrial repeats the first shape to start the trial
    and also gets all task shapes. This gets hihgher scores than success_binary_quick.
    expect_max_two_shapes_each_day is like AABBB. If this day also has AAABBC, then cannot do this. Need to solve by then asking
    if got all As and Bs in order.
    """
    # NOTE: Confirmed that <exclude_because_online_abort> means that aborted even though sequence was correct so far.

    assert expect_max_two_shapes_each_day==True, "see docs"
    shape_key = D.grammarparses_rules_shape_AnBmCk_get_shapekey()
    info = D.grammarparses_rules_shapes_summary_simple()

    # Only keep trials which are composed of only the shapes in the rule.
    # -- e.g, sometimes had probe trials inlcuding novel shapes. If inlcude, then will
    # get errors below.
    list_keep = []
    for ind in range(len(D.Dat)):
        shapes = D.taskclass_shapes_extract(ind, shape_kind=shape_key)
        keep = all([sh in info["shapes_used"] for sh in shapes])
        list_keep.append(keep)
    if sum(list_keep)/len(list_keep) < 0.9:
        print(sum(list_keep), len(list_keep))
        assert False, "probes should not be so common.."
    D.Dat = D.Dat.iloc[list_keep].reset_index(drop=True)

    # Classify the kind of error on each trial (semantic)
    # NOTE: also see D.grammarparsesmatlab_score_wrapper(), which is too detailed for here.
    list_errors = []
    for i, row in D.Dat.iterrows():    
        # errordict = D.grammarparses_classify_sequence_error(i, shape_key=shape_key)

        # Deter
        # print(i)
        # n_parses = D.grammarparses_parses_extract_trial(i)
        rule_includes_location = D.grammarparses_rules_shape_AnBmCk_locationmatters(ind)

        if row["success_binary_quick"] == True:
            # No error
            error_class = "success"
        elif row["exclude_because_online_abort"]:
            # Sequence fine, stroke bad
            error_class = "abort_stroke_quality"
        elif (rule_includes_location==True) and (D.grammarparses_classify_sequence_error(i, shape_key=shape_key)["error_same_shape"]):
            # Failed, but due to location error (shape was not error)
            error_class = "shape_correct_location_wrong"
        else:
            # Failed, due to shape error
            error_class = "shape_wrong"
        
        list_errors.append(error_class)
    D.Dat["sequence_error_kind"] = list_errors

    # Plot examples for each error
    if plot_examples:
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        cols = ["success_binary_quick", "aborted", "exclude_because_online_abort", "probe", "sequence_error_kind"]
        print(cols)
        grpdict = grouping_append_and_return_inner_items_good(D.Dat, cols)
        nrand = 20
        for grp, inds in grpdict.items():
            print(grp, inds)

            # Plot exmaples of this class
            # - randomly sample inds
            figbeh, _, indsthis = D.plotMultTrials(inds, "strokes_beh", return_idxs=True, nrand=nrand,
                                                    naked_axes=True, add_stroke_number=False)
            figtask = D.plotMultTrials(indsthis, "strokes_task", return_idxs=False,
                                                naked_axes=True, add_stroke_number=False)

            savefig(figbeh, f"{savedir}/examples-beh-{grp}.pdf")
            savefig(figtask, f"{savedir}/examples-task-{grp}.pdf")
            plt.close("all")

    # Same, but plot metric (e.g., n-circle)
    # - First, make new column, which is (metric relevant fro the epoch) (actual n items for that) (totla strokes)
    # Map each epoch to the correct score
    map_epoch_to_scorename = {}
    map_epoch_to_firstshape = {}
    for ep in D.Dat["epoch"].unique().tolist():
        if False:
            first_shape = info["map_rule_to_shapeseq"][ep][0]
        else:
            shape_ordered = D._grammarparses_rules_shape_AnBmCk_get_shape_order(ep)
            first_shape = shape_ordered[0]
        score_name = f"ntokens-repeat-{first_shape}"
        map_epoch_to_scorename[ep] = score_name
        map_epoch_to_firstshape[ep] = first_shape

    # for each row, determine its score
    D.Dat["scorename_this_epoch"] = D.Dat["epoch"].map(lambda x:map_epoch_to_scorename[x])
    D.Dat["firstshape_this_epoch"] = D.Dat["epoch"].map(lambda x:map_epoch_to_firstshape[x])

    def F(x):
        score_name = map_epoch_to_scorename[x["epoch"]]
        try:
            return x[score_name]
        except Exception as err:
            print("Probably this score_name was not added as a column in Diagnostic Model code.")
            raise err
    tmp = []
    for _, row in D.Dat.iterrows():
        tmp.append(F(row))
    D.Dat["ntokens-repeat-thisepoch"] = tmp

    # For each trial, figure out how many of each shape there are
    for shape_find in info["shapes_used"]:
        list_n =[]
        for ind in range(len(D.Dat)):
            shapes = D.taskclass_shapes_extract(ind, shape_kind=shape_key)
            n = sum([sh==shape_find for sh in shapes])
            list_n.append(n)
        D.Dat[f"n_{shape_find}"] = list_n

    # Sanity check that sum of shapes is n task strokes
    cols = [f"n_{sh}" for sh in info["shapes_used"]]
    # display(D.Dat[~(D.Dat.loc[:, cols].sum(axis=1) == D.Dat["seqc_nstrokes_task"])])
    assert(all(D.Dat.loc[:, cols].sum(axis=1) == D.Dat["seqc_nstrokes_task"])), "you missed some shapes in info[shapes_used], probably"
    
    from pythonlib.tools.datetools import trialcode_to_scalar
    D.Dat["trialcode_scal"] = [trialcode_to_scalar(tc) for tc in D.Dat["trialcode"]]
    D.Dat["supervision_stage_concise"].value_counts()
    
    # Group conditions based on whether they are success/fail considering shapes, or others (which are more ambigous)
    from pythonlib.tools.pandastools import append_col_with_grp_index
    def F(row):
        if row["sequence_error_kind"] in ["success", "shape_wrong"]:
            sequence_error_kind_v2="shape"
        elif row["sequence_error_kind"] in ["abort_stroke_quality", "shape_correct_location_wrong"]:
            sequence_error_kind_v2=row["sequence_error_kind"]
        else:
            print(row["sequence_error_kind"])
            assert False
        return sequence_error_kind_v2
    D.Dat["sequence_error_kind_v2"] = [F(row) for _, row in D.Dat.iterrows()]

    if True: # Doing preprocessing for D, just for debugging
        # ### Get time in scalar value
        # from pythonlib.tools.datetools import standardizeTime
        # # datestart = "230111-000000"
        # datestart = sorted(D.Dat["datetime"])[0][:6] + "-000000"
        # D.Dat["datetime_scal"] = [standardizeTime(dt, datestart=datestart, daystart=0, dayend=1) for dt in D.Dat["datetime"]]

        ### Get score of first shape
        tmp = []
        for _, row in D.Dat.iterrows():
            col = "n_" + row["firstshape_this_epoch"]
            tmp.append(row[col])
        D.Dat["n_first_shape"] = tmp
        D.Dat["n_first_shape"] = D.Dat["n_first_shape"].astype(int)

        ### Get the first instance of each (epoch, character)
        from pythonlib.tools.pandastools import append_col_with_grp_index
        D.Dat = append_col_with_grp_index(D.Dat, ["epoch", "character"], "epoch_char")

        # import numpy as np
        # assert np.all(np.diff(D.Dat["datetime_scal"])>0)
        # assert np.all(np.diff(D.Dat["trialcode_scal"])>0)
        # D.Dat["index"] = D.Dat.index

        from pythonlib.tools.pandastools import append_col_with_grp_index
        D.Dat = append_col_with_grp_index(D.Dat, ["n_first_shape", "seqc_nstrokes_task"], "nfirst_ntot")

        D.Dat["seqc_nstrokes_task_gotten"] = [len(D.taskclass_tokens_extract_wrapper(ind, "beh_firsttouch")) for ind in range(len(D.Dat))]
        D.Dat["success_binary_got_all_taskstrokes"] = (D.Dat["seqc_nstrokes_task_gotten"] == D.Dat["seqc_nstrokes_task"]) # got all of the task strokes
        D.Dat["success_binary_got_all_firstshape"] = D.Dat["n_first_shape"] == D.Dat["ntokens-repeat-thisepoch"]

        # Recalculate success, since some were caled not but actually are.
        # As long as only two shapes, and got correct n repesat for first sahpe, and got all task strokes, then is success
        if expect_max_two_shapes_each_day:
            successes = []
            for _, row in D.Dat.iterrows():
                rule = row["epoch"]
                shapes_ordered = D._grammarparses_rules_shape_AnBmCk_get_shape_order(rule)
                # succ = (len(shapes_ordered)==2) & (row["success_binary_got_all_firstshape"]) & (row["success_binary_got_all_taskstrokes"])
                succ = (row["success_binary_got_all_firstshape"]) & (row["success_binary_got_all_taskstrokes"])
                successes.append(succ)
            D.Dat["success_binary_quick_v2"] = successes

            assert sum((D.Dat["success_binary_quick_v2"]==False) & (D.Dat["success_binary_quick"]==True))==0, "must be mistake in success_binary_quick"

        from math import factorial
        def p(i, n, k):
            assert i <= n
            assert n < k
            return (factorial(n)/factorial(n-i)) * (factorial(k-i)/factorial(k)) * (k-n)/(k-i)

        def expected_n_repeated_at_start(n_actual, k):
            """
            Expected value of n, if you sample randomly with replacement, and
            you count how many draws you make at the start
            PARAMS:
            - n_actual, the number of items of this shape in the image
            - k, the totla num of items (across shapes)
            """

            # Expected value
            import numpy as np
            # print(i, n_actual, k)
            # print([(i, p(i, n_actual, k)) for i in range(n_actual+1)])
            expected_n = np.sum([i*p(i, n_actual, k) for i in range(n_actual+1)])
            # print(expected_n)

            return expected_n

        def expected_success_rate(n_actual, k):
            """
            Expected value of n, if you sample randomly with replacement, and
            you count how many draws you make at the start
            PARAMS:
            - n_actual, the number of items of this shape in the image
            - k, the totla num of items (across shapes)
            """

            # Expected value
            import numpy as np
            rate = p(n_actual, n_actual, k)
            return rate

        tmp1 = []
        tmp2 = []
        for i, row in D.Dat.iterrows():
            n_actual = row["n_first_shape"]
            k = row["seqc_nstrokes_task"]
            expected_n = expected_n_repeated_at_start(n_actual, k)
            expected_rate = expected_success_rate(n_actual, k)
            tmp1.append(expected_n)
            tmp2.append(expected_rate)

        D.Dat["ntokens-repeat-thisepoch_RAND"] = tmp1
        D.Dat["success_binary_got_all_firstshape_RAND"] = tmp2

    ### Classify probes (new location, more n strokes)
    # NOTE: This does within day, which is probably ok, as the training tasks each day are good representaton of
    # all training up to now (cumulative). There is the potential for this to call some things probes even when
    # they are not.
    from pythonlib.dataset.dataset_preprocess.probes import compute_features_each_probe
    def get_loc_nstrok(dict_probe_features, row):
        feats = dict_probe_features[(row["epoch"], row["character"])]
        return (feats["novel_location_shape_combo"], feats["more_n_strokes"])
    dict_probe_features, dict_probe_kind, list_tasks_probe = compute_features_each_probe(D)
    D.Dat["probe_kind"] = [dict_probe_kind[(row["epoch"], row["character"])] if row["probe"] else "not_probe" for _, row in D.Dat.iterrows()]
    D.Dat["probe_loc_nstrok"] = [get_loc_nstrok(dict_probe_features, row) if row["probe"] else "not_probe" for _, row in D.Dat.iterrows()]

def fig1_generalize_3_extract_sketchpad_xylim(list_D):
    """
    For behavior, get overall min and max for x and y, across all tasks.
    """
    import numpy as np
    xlims = []
    ylims = []
    for D in list_D:
        _xlim, _ylim = D.recomputeSketchpadEdgesXlimYlim("strokes_task")
        xlims.append(_xlim)
        ylims.append(_ylim)

    tmp = np.stack(xlims, axis=0)
    XLIM = (np.min(tmp[:, 0]), np.max(tmp[:, 1]))

    tmp = np.stack(ylims, axis=0)
    YLIM = (np.min(tmp[:, 0]), np.max(tmp[:, 1]))

    return XLIM, YLIM

def fig1_generalize_3_plot_task_variability(DFALL, PLOT, savedir, XLIM, YLIM, window_size = 20, drawing_plot_interval=75):
    """
    For behavior, compute and store variables related to variability of unique characters across tasks. Goal is to show that 
    trials were very varied.

    And also plot timecourses and drawings.
    NOTE: Best to do both compute and plots or else will be easy to lose mapping between stats (aligned to "index") and
    drawings after you slide dataset

    PARAMS:
    - window_size, running window, to score n unique chracters in running window (20)
    """
    ### Plots to show variability in tasks

    if PLOT:
        assert savedir is not None

    ### For each trial, is this a novel task?
    assert all(sorted(DFALL["index"]) == DFALL["index"])
    assert all(np.diff(DFALL["index"])==1)

    tasks_done_at_least_once = []
    novels = []
    for _, row in DFALL.iterrows():
        key = (row["epoch"], row["character"])
        if key in tasks_done_at_least_once:
            novel_task_this_trial = False
        else:
            novel_task_this_trial = True
            tasks_done_at_least_once.append(key)

        # print(novel_task_this_trial)
        novels.append(novel_task_this_trial)

    DFALL["novel_epochchar_this_trial"] = novels

    ### Cumulative count of novel tasks
    DFALL["novel_epochchar_cumsum"] = np.cumsum(DFALL["novel_epochchar_this_trial"])

    ### Running average of the number of unique tasks within a time window
    # assert window_size%2==1, "thus can define the center index"
    plot_drawings = False
    PRINT = False
    tmp = np.full(len(DFALL), np.nan)
    map_idxcenter_to_window = {}
    for i1 in range(len(DFALL) - window_size + 1):
        i2 = i1 + window_size

        chars = DFALL.iloc[i1:i2]["character"]
        n_unique_chars = len(set(chars))

        if PRINT:
            print(i1, i2)
            print(chars)
            print(n_unique_chars, window_size)  

        # index, take the center value
        index = int((i1 + i2-1)/2)
        assert index == DFALL.iloc[index]["index"], "you should run this on the original DFALL"
        tmp[index] = n_unique_chars
        map_idxcenter_to_window[index] = (i1, i2)

        if False:
            if n_unique_chars==5:
                plot_drawings = True

            if plot_drawings:
                strokes_beh = DFALL.iloc[i1:i2]["strokes_beh"].tolist()
                strokes_task = DFALL.iloc[i1:i2]["strokes_task"].tolist()
                successes = DFALL.iloc[i1:i2]["success_binary_quick_v2"].tolist()

                D.plotMultStrokes(strokes_beh, titles=successes)
                D.plotMultStrokes(strokes_task, is_task=True)
                assert False

        if PLOT:
            from pythonlib.dataset.dataset import Dataset
            D = Dataset([])
            ### PLOT example trials
            if index%drawing_plot_interval==0:
                strokes_beh = DFALL.iloc[i1:i2]["strokes_beh"].tolist()
                strokes_task = DFALL.iloc[i1:i2]["strokes_task"].tolist()
                successes = DFALL.iloc[i1:i2]["success_binary_quick_v2"].tolist()
                successes = [bool(s) for s in successes]
                error_kinds = DFALL.iloc[i1:i2]["sequence_error_kind"].tolist()
                supervisions = DFALL.iloc[i1:i2]["supervision_online"].tolist()
                titles2 = [f"sprv={int(superv)}-succ={int(succ)}" for superv, succ in zip(supervisions, successes)]

                figbeh, axes_beh = D.plotMultStrokes(strokes_beh, titles=error_kinds, naked_axes=True, number_from_zero=False)
                figtask, axes_task = D.plotMultStrokes(strokes_task, is_task=True, titles=titles2, naked_axes=True)

                for ax in axes_beh.flatten():
                    ax.set_xlim(XLIM)
                    ax.set_ylim(YLIM)

                for ax in axes_task.flatten():
                    ax.set_xlim(XLIM)
                    ax.set_ylim(YLIM)

                savefig(figbeh, f"{savedir}/trialindex={index}-n_uniq_char={n_unique_chars}-BEH.pdf")
                savefig(figtask, f"{savedir}/trialindex={index}-n_uniq_char={n_unique_chars}-TASK.pdf")
                
                plt.close("all")

    DFALL["num_uniq_char_in_sld_wind"] = tmp

    ### Plot
    # Across entire experiment
    if False:
        do_clean = True
        probe_only = False
        first_trial = False
        no_supervision = False
        dfall = extract_data_subset(do_clean, probe_only, first_trial, no_supervision)
    else:
        dfall = DFALL

    for row in [None, "epoch", "seqc_nstrokes_task", "probe_loc_nstrok", "train_test_class_v2"]:

        fig = sns.relplot(dfall, x="index", y="novel_epochchar_this_trial", row=row, aspect=1.5, alpha=0.2)
        figmod_overlay_date_lines(fig, dfall, "index", timevar="datetime_scal")
        savefig(fig, f"{savedir}/timecourse-y=novel_epochchar_this_trial-row={row}.pdf")

        fig = sns.relplot(dfall, x="index", y="novel_epochchar_cumsum", row=row, aspect=1.5, kind="line")
        figmod_overlay_date_lines(fig, dfall, "index", timevar="datetime_scal")
        savefig(fig, f"{savedir}/timecourse-y=novel_epochchar_cumsum-row={row}-1.pdf")

        fig = sns.relplot(dfall, x="index", y="novel_epochchar_cumsum", row=row, aspect=1.5, alpha=0.5)
        figmod_overlay_date_lines(fig, dfall, "index", timevar="datetime_scal")
        savefig(fig, f"{savedir}/timecourse-y=novel_epochchar_cumsum-row={row}-2.pdf")

        fig  = sns.relplot(dfall, x="index", y="num_uniq_char_in_sld_wind", row=row, aspect=1.5, kind="line")
        figmod_overlay_date_lines(fig, dfall, "index", timevar="datetime_scal")
        for ax in fig.axes.flatten():
            ax.set_ylim([0, window_size])
        savefig(fig, f"{savedir}/timecourse-y=num_uniq_char_in_sld_wind-row={row}-1.pdf")

        fig  = sns.relplot(dfall, x="index", y="num_uniq_char_in_sld_wind", row=row, aspect=1.5, alpha=0.5)
        figmod_overlay_date_lines(fig, dfall, "index", timevar="datetime_scal")
        for ax in fig.axes.flatten():
            ax.set_ylim([0, window_size])
        savefig(fig, f"{savedir}/timecourse-y=num_uniq_char_in_sld_wind-row={row}-2.pdf")

        plt.close("all")

    return map_idxcenter_to_window


def figmod_overlay_date_lines(fig, dfall_this, xvar="index", timevar="datetime_scal"):
    """
    Modify timeocurse plots, 
    by overlay vertical lines at the first trial for each date.
    """
    
    def F(x):
        index = x.sort_values(timevar, axis=0).iloc[0][xvar]
        return index
    
    dfboundaries = dfall_this.groupby("date").apply(F).reset_index()

    for _, row in dfboundaries.iterrows():
        date = row["date"]
        index = row[0]

        for ax in fig.axes.flatten():
            ax.axvline(index, color="k", alpha=0.4)
            YLIM = ax.get_ylim()
            ydelt = YLIM[1] - YLIM[0]
            ax.text(index, YLIM[0] + 0.9*ydelt, date, color="k")


def fig1_generalize_3_plot_task_variability_plot(DFALL, map_idxcenter_to_window, window_size, XLIM, YLIM, savedir):
    """
    PLot timecourse showing task variability, 
    """

    assert False, "moved to fig1_generalize_3_plot_task_variability -- see resaoning there"
    from pythonlib.dataset.dataset import Dataset
    D = Dataset([])

    ### Plot
    for row in [None, "epoch", "seqc_nstrokes_task", "probe_loc_nstrok", "train_test_class_v2"]:

        fig = sns.relplot(DFALL, x="index", y="novel_epochchar_this_trial", row=row, aspect=1.5, alpha=0.2)
        figmod_overlay_date_lines(fig, DFALL, "index", timevar="datetime_scal")
        savefig(fig, f"{savedir}/timecourse-y=novel_epochchar_this_trial-row={row}.pdf")

        fig = sns.relplot(DFALL, x="index", y="novel_epochchar_cumsum", row=row, aspect=1.5, kind="line")
        figmod_overlay_date_lines(fig, DFALL, "index", timevar="datetime_scal")
        savefig(fig, f"{savedir}/timecourse-y=novel_epochchar_cumsum-row={row}-1.pdf")

        fig = sns.relplot(DFALL, x="index", y="novel_epochchar_cumsum", row=row, aspect=1.5, alpha=0.5)
        figmod_overlay_date_lines(fig, DFALL, "index", timevar="datetime_scal")
        savefig(fig, f"{savedir}/timecourse-y=novel_epochchar_cumsum-row={row}-2.pdf")

        fig  = sns.relplot(DFALL, x="index", y="num_uniq_char_in_sld_wind", row=row, aspect=1.5, kind="line")
        figmod_overlay_date_lines(fig, DFALL, "index", timevar="datetime_scal")
        for ax in fig.axes.flatten():
            ax.set_ylim([0, window_size])
        savefig(fig, f"{savedir}/timecourse-y=num_uniq_char_in_sld_wind-row={row}-1.pdf")

        fig  = sns.relplot(DFALL, x="index", y="num_uniq_char_in_sld_wind", row=row, aspect=1.5, alpha=0.5)
        figmod_overlay_date_lines(fig, DFALL, "index", timevar="datetime_scal")
        for ax in fig.axes.flatten():
            ax.set_ylim([0, window_size])
        savefig(fig, f"{savedir}/timecourse-y=num_uniq_char_in_sld_wind-row={row}-2.pdf")

        plt.close("all")

    list_trial_index_plot = range(0, DFALL["index"].max(), 75)
    for trial_index in list_trial_index_plot:

        idx = np.argmin(np.abs(DFALL["index"] - trial_index))

        if idx in map_idxcenter_to_window:
            i1, i2 = map_idxcenter_to_window[trial_index]
            # i1 = int(idx - window_size/2)
            # i2 = int(idx + window_size/2)

            n_unique_chars = DFALL.iloc[idx]["num_uniq_char_in_sld_wind"]

            print(n_unique_chars)
            strokes_beh = DFALL.iloc[i1:i2]["strokes_beh"].tolist()
            strokes_task = DFALL.iloc[i1:i2]["strokes_task"].tolist()
            successes = DFALL.iloc[i1:i2]["success_binary_quick_v2"].tolist()
            successes = [bool(s) for s in successes]
            error_kinds = DFALL.iloc[i1:i2]["sequence_error_kind"].tolist()
            supervisions = DFALL.iloc[i1:i2]["supervision_online"].tolist()
            chars = DFALL.iloc[idx]["character"].tolist()

            titles2 = [f"sprv={int(superv)}-succ={int(succ)}" for superv, succ in zip(supervisions, successes)]

            figbeh, axes_beh = D.plotMultStrokes(strokes_beh, titles=error_kinds, naked_axes=True, number_from_zero=False)
            figtask, axes_task = D.plotMultStrokes(strokes_task, is_task=True, titles=titles2, naked_axes=True)

            for ax in axes_beh.flatten():
                ax.set_xlim(XLIM)
                ax.set_ylim(YLIM)

            for ax in axes_task.flatten():
                ax.set_xlim(XLIM)
                ax.set_ylim(YLIM)

            savefig(figbeh, f"{savedir}/trialindex={trial_index}-n_uniq_char={n_unique_chars}-BEH.pdf")
            savefig(figtask, f"{savedir}/trialindex={trial_index}-n_uniq_char={n_unique_chars}-TASK.pdf")
            
            plt.close("all")

if __name__=="__main__":
    
    import sys


    PLOTS_DO = [1]
    # PLOTS_DO = [5.1]

    ###
    for plot_do in PLOTS_DO:
        if plot_do==1:
            animal = sys.argv[1]
            date = sys.argv[2]
            fig1_generalize_wrapper(animal, date)
        else:
            assert False
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pythonlib.drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows
from ..analy_dlist import extract_strokes_monkey_vs_self
from pythonlib.globals import PATH_DRAWMONKEY_DIR
from .dates import preprocess_dates
from pythonlib.dataset.modeling.discrete import _get_default_grouping_map_tasksequencer_to_rule

def _groupingParams(D, expt):
    """ Filter and grouping variable to apply to 
    Dataset"""

    assert isinstance(expt, str)

    # Reassign epochs/rules
    # By fedefault, replace epochs with rules used in metadat
    # Replace epoch with rule, if that exists
    def F(x):
        idx = x["which_metadat_idx"]
        if D.Metadats[idx]["rule"]:
            return D.Metadats[idx]["rule"]
        else:
            return idx+1
    D.Dat = applyFunctionToAllRows(D.Dat, F, "epoch")

    # grouping_levels = ["short", "long"]
    # vals = ["nstrokes", "hausdorff", "time_go2raise", "time_raise2firsttouch", 
    #     "dist_raise2firsttouch", "sdur", "isi", 
    #     "time_touchdone", "dist_touchdone", "dist_strokes", 
    #     "dist_gaps", "sdur_std", "gdur_std", "hdoffline", "alignment"]

    # feature_names = vals + ["stroke_speed", "gap_speed", "onset_speed", "offset_speed", "total_distance",
    #                 "total_time", "total_speed", "dist_per_gap", "dist_per_stroke"]

    # features_to_remove_nan =  ["dist_strokes", "sdur", "dist_gaps" , "isi", "dist_raise2firsttouch", "time_raise2firsttouch", "nstrokes"]
    # features_to_remove_outliers = ["dist_strokes", "sdur", "dist_gaps" , "isi", "dist_raise2firsttouch", "time_raise2firsttouch"] # distnaces and times.

    #### DEFAULTS
    # def _get_defaults(D):
    F = {}
    grouping = "epoch"
    plantime_cats = {}
    features_to_remove_nan =  []
    features_to_remove_outliers = []
    grouping_levels = None
    feature_names = ["hdoffline", "num_strokes", "circ", "dist"]    

    # -- whether reassign "epoch"
    grouping_reassign = False
    # grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
    grouping_reassign_methods_in_order = ["tasksequencer"]

    # -- WHether reassign train/test 
    traintest_reassign_method = "probes"

    # -- WHether renaming taskgroups
    mapper_taskset_to_category = {}
    mapper_auto_rename_probe_taskgroups = False

    # # - Merge epochs names into a single new name
    # epoch_merge_dict = {
    #   "LCr2":["LCr1", "LCr2"]
    # }
    epoch_merge_dict = {}

    # 2) Overwrite defaults    
    if expt == "neuralprep2":
        F = {
            "block":[17, 18]
        }
        grouping = "block"
        assert False, "get plantime_cats"
    elif expt=="neuralprep3":
        F = {
            "block":[9],
            "hold_time_exists":[True]
        }
        grouping = "hold_time_string"
        assert False, "get plantime_cats"
    elif expt in ["neuralprep4", "neuralprep5", "neuralprep6", "neuralprep8"]:
        F = {
            "block":[16],
            "plan_time":[0., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1000.: "long"}
    elif expt in ["neuralprep7"]:
        F = {
            "block":[16],
            "plan_time":[150., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {150.: "short", 1000.: "long"}
    elif expt in ["plan3"]:
        F = {
            "block":[11],
            "plan_time":[0., 1200.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1200.: "long"}
        feature_names = [f for f in feature_names if f not in 
        ["time_touchdone", "dist_touchdone", "offset_speed"]]
    elif expt in ["plan4", "plan5"]:
        F = {
            "block":[11],
            "plan_time":[0., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1000.: "long"}
        feature_names = [f for f in feature_names if f not in 
            ["time_touchdone", "dist_touchdone", "offset_speed"]]
    elif expt in ["plandir1"]:
        print("ONLY FOCUS ON l-->r BLOCK FOR NOW!! NEED TO ALSO ADD BLOCK 1")
        F = {
            "block":[11],
            "plan_time":[0., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1000.: "long"}
        feature_names = [f for f in feature_names if f not in 
            ["time_touchdone", "dist_touchdone", "offset_speed", "alignment"]] # exclude alignment if there is only one trial per task.
    elif expt in ["plandir2"]:
        print("ONLY FOCUS ON r-->l BLOCK FOR NOW!! NEED TO ALSO ADD BLOCK 1")
        F = {
            "block":[18],
            "plan_time":[0., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1000.: "long"}
        feature_names = [f for f in feature_names if f not in 
            ["time_touchdone", "dist_touchdone", "offset_speed", "alignment"]] # exclude alignment if there is only one trial per task.
    elif expt=="lines5":
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = ["straight", "bent"]
        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]
    elif expt=="linecircle":
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = ["null"]
        feature_names = ["hdoffline"]
    elif expt=="figures9":
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = ["straight", "bent"]
        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]        
    elif expt == "gridlinecircle":
        grouping_reassign = False
        # grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]

        animal = D.animals()
        if len(animal)>1:
            assert False, "params different for each animal.."
        else:
            animal = animal[0]
        F = {}
        grouping = "epoch"
        # plantime_cats = {}
        # features_to_remove_nan =  []
        # features_to_remove_outliers = []
        if animal in ["Diego", "Pancho"]:
            grouping_levels = ["baseline", "linetocircle", "circletoline", "lolli"]
        elif animal in ["Red"]:
            grouping_levels = ["baseline", "Ltoline"]
        else:
            assert False
        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]                
        traintest_reassign_method = "fixed"
    elif expt in ["chunkbyshape1", "chunkbyshape1b", "chunkbyshape2", "chunkbyshape2b"]:
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = ["horiz", "vert"]
        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]     
        traintest_reassign_method = "fixed"
    elif expt=="character34":
        assert False, "fix this, see here"
        # epoch 1 (line) the test tasks were not defined as probes. Add param here , which
        # should have affect in subsewuen code redefining monkye train test.

    elif "neuralbiasdir" in expt:
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = True

    # elif expt in ["neuralbiasdir3", "neuralbiasdir3b"]:
    #     grouping_reassign = True
    #     grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
    #     traintest_reassign_method = "supervision_except_color"
    #     mapper_taskset_to_category = {
    #         ("neuralbiasdir", 4, tuple([13,14,15,16,17,18])): "heldout_E_spatial", # novel spatial config
    #         ("neuralbiasdir", 4, tuple([1,3,5,7,9,11,13])): "heldout_E_shapes"} # novel shapes order

    # elif expt in ["neuralbiasdir3c", "neuralbiasdir3d", "neuralbiasdir3e"]:
    #     grouping_reassign = True
    #     grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
    #     traintest_reassign_method = "supervision_except_color"
    #     mapper_taskset_to_category = {
    #         ("neuralbiasdir", 6, tuple([6, 7, 8, 9, 10, 11])): "heldout_E_spatial",  # novel spatial config
    #         ("neuralbiasdir", 5, tuple([5,6,7,8,9])): "heldin_same_RD_LU",  # samebeh, right and down.
    #         ("neuralbiasdir", 6, tuple([3])): "heldin_same_RD_LU", # samebeh, 
    #         ("neuralbiasdir", 5, tuple([14,15,16,17,18])): "heldin_same_RU_LD", # samebeh, 
    #         ("neuralbiasdir", 6, tuple([12])): "heldin_same_RU_LD"} # samebeh, 

    # elif 'neuralbiasdir' in expt:
    #     grouping_reassign = True
    #     grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
    #     traintest_reassign_method = "supervision_except_color"

    elif "shapesequence" in expt:
        # Reassign rules: each epoch is based on tasksequencer rule
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = True
    elif "shapedirseq" in expt:
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = True

    # elif expt in ["grammar1b", "grammardir1"]:
    #     # Reassign rules: each epoch is based on tasksequencer rule
    #     grouping_reassign = True
    #     grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
    #     traintest_reassign_method = "supervision_except_color"
    #     mapper_auto_rename_probe_taskgroups = True

    elif "grammardircolor1" in expt or "grammardircolor2" in expt or "grammardircolor3" in expt:
        # Reassign rules first using tasksequencer, then taking conjuctionw ith color instruction/
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        mapper_taskset_to_category = {
            ("grammar", 41, tuple([16, 18, 21, 22, 23, 27, 28])): "diff_beh",
            ("grammar", 41, tuple([17, 20, 24, 26])): "diff_beh_probes",
            ("grammar", 41, tuple([15, 19, 25])): "same_beh"}
        mapper_auto_rename_probe_taskgroups = True
    elif "grammardircolor" in expt:
        # Reassign rules first using tasksequencer, then taking conjuctionw ith color instruction/
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        mapper_taskset_to_category = {
            ("neuralbiasdir", 4, tuple([1, 3, 6, 17])): "diff_beh_probes",
            ("grammar", 41, tuple([16, 18, 21, 22, 23, 27, 28])): "diff_beh",
            ("grammar", 41, tuple([17, 20, 24, 26])): "diff_beh_probes",
            ("grammar", 41, tuple([15, 19, 25])): "same_beh"}
        mapper_auto_rename_probe_taskgroups = True

    elif "dircolor" in expt:
        # 10/17/22 - e..g, dircolro3b    
        # Reassign rules first using tasksequencer, then taking conjuctionw ith color instruction/
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        mapper_taskset_to_category = {
            ("neuralbiasdir", 31, tuple([1,3,6])): "heldout_I", # novel shape order x spatial config
            ("neuralbiasdir", 31, tuple([8,9])): "heldout_E_config"} # novel spatial config.
        mapper_auto_rename_probe_taskgroups = True

    elif "dirshapecolor2" in expt:
        # 10/21/22 - 3 rules.
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        mapper_taskset_to_category = {
            ("neuralbiasdir", 30, tuple([1, 6, 11, 16, 21, 26])):"train_test_same_RShape", 
            ("neuralbiasdir", 31, tuple([7])):"train_test_same_RShape", 
            ("neuralbiasdir", 30, tuple([5, 15, 20, 25, 30])):"train_test_same_LShape", 
            ("neuralbiasdir", 31, tuple([2, 12])):"train_test_same_LShape", 
            ("neuralbiasdir", 31, tuple([3, 4, 6, 10])): "heldout_I"} # heldout, diff across all 3 rules.
        mapper_auto_rename_probe_taskgroups = True

    elif "dirshapecolor" in expt:
        # 10/17/22 - e..g, dircolro3b    
        # Reassign rules first using tasksequencer, then taking conjuctionw ith color instruction/
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        mapper_taskset_to_category = {
            ("neuralbiasdir", 30, tuple([5, 15, 20, 25, 30])): "train_test_same", # same beh, test blocks only (fb on)
            ("neuralbiasdir", 31, tuple([2, 12])): "train_test_same", # 
            ("neuralbiasdir", 30, tuple([1, 24])): "heldout_I", # heldout, diff beh
            ("neuralbiasdir", 31, tuple([3, 4, 6, 10])): "heldout_I"} # .
        mapper_auto_rename_probe_taskgroups = True            

    elif "dirshape" in expt:
        # 10/17/22 - e..g, dircolro3b    
        # Reassign rules first using tasksequencer, then taking conjuctionw ith color instruction/
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = True            

    elif "shapecolor" in expt:
        # 10/17/22 - e..g, dircolro3b    
        # Reassign rules first using tasksequencer, then taking conjuctionw ith color instruction/
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        # mapper_taskset_to_category = {
        #     ("neuralbiasdir", 30, tuple([5, 15, 20, 25, 30])): "train_test_same", # same beh, test blocks only (fb on)
        #     ("neuralbiasdir", 31, tuple([2, 12])): "train_test_same", # 
        #     ("neuralbiasdir", 30, tuple([1, 24])): "heldout_I", # heldout, diff beh
        #     ("neuralbiasdir", 31, tuple([3, 4, 6, 10])): "heldout_I"} # .
        mapper_auto_rename_probe_taskgroups = True            

    elif "charstrokeseq" in expt:
        grouping_reassign = False

    elif "grammar" in expt:
        # Assume that if grammar in name, it has rules.
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = True
    elif "gridlinecircle" in expt:
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = True
        epoch_merge_dict = {
            "LCr2":["LCr1", "LCr2"]
        }

    elif "dirfullvar" in expt or "dirdir" in expt:
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer", "color_instruction"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = True

    elif "rowcol" in expt:
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = False
    elif "linecirclerow" in expt:
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["tasksequencer"]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = False
    elif expt in ["primsingridvar2", "primsingridvar2b", "primsingridvar3"]:
        # Then epochs are a group of blocks. manually enter them here.
        grouping_reassign = True
        grouping_reassign_methods_in_order = ["manual_by_block"]
        if expt=="primsingridvar2":
            map_epoch_to_block = {
                "oneloc_varyshp":[2, 4, 6],
                "oneshp_varyloc":[8, 10, 12]
            }
        elif expt=="primsingridvar2b":
            map_epoch_to_block = {
                "varyloc_varyshp":[14],
                "novary":[16, 18, 20],
                "oneloc_varyshp":[2,4,6],
                "oneshp_varyloc":[8,10,12],
            }
        elif expt=="primsingridvar3":
            map_epoch_to_block = {
                "singleprim":[2],
                "oneloc_varyshp":[9, 16, 23],
                "oneshp_varyloc":[30, 37, 44],
                "varyloc_varyshp":[51],
            }
        else:
            assert False

        grouping_reassign_params_in_order = [map_epoch_to_block]
        traintest_reassign_method = "supervision_except_color"
        mapper_auto_rename_probe_taskgroups = False
    else:
        # pass, just use defaults
        pass

    ### always reassign grouping by color instruction (even if not using it, doesnt do anythign)
    if "color_instruction" not in grouping_reassign_methods_in_order:
        # grouping_reassign = Tr
        grouping_reassign_methods_in_order.append("color_instruction")


    ############### OPTIONAL:
    # Filter dataframe
    if len(F)>0:
        print("*** Filtering dataframe using this filter:")
        print(F)
        D = D.filterPandas(F, return_ver="dataset")

    # classify based on plan times
    if len(plantime_cats)>0:
        print("*** Reassigning plan_time category names, using this filter:")
        print(plantime_cats)
        F = lambda x: plantime_cats[x["plan_time"]]
        D.Dat = applyFunctionToAllRows(D.Dat, F, "plan_time_cat")
    
    if grouping_reassign:
        # Rename epochs, applying reassignment methods in order
        for i, grmeth in enumerate(grouping_reassign_methods_in_order):
            if grmeth=="tasksequencer":
                # Look into objectclass tasksequencer.
                print(" ")
                print("*** Rules/epochs reassigning using the following rules:")
                grouping_map_tasksequencer_to_rule = _get_default_grouping_map_tasksequencer_to_rule()
                print(grouping_map_tasksequencer_to_rule)
                assert len(grouping_map_tasksequencer_to_rule)>0, "need to define how to remap rules."
                epoch_grouping_reassign_by_tasksequencer(D, grouping_map_tasksequencer_to_rule)
            elif grmeth=="color_instruction":
                print(" ")
                D.supervision_reassign_epoch_rule_by_color_instruction()
            elif grmeth=="manual_by_block":
                # get a mapping from block to epoch.
                map_epoch_to_block = grouping_reassign_params_in_order[i]
                # Generate map from block to epoch.
                map_block_to_epoch = {}
                for ep, list_bk in map_epoch_to_block.items():
                    for bk in list_bk:
                        assert bk not in map_block_to_epoch.keys(), "block is not allowed to be present in multipel epochs"
                        map_block_to_epoch[bk] = ep
                print("[preprocess.general] map_block_to_epoch:")
                print(map_block_to_epoch)
                # Assert that every block is accounted for
                for bk in D.Dat["block"].unique().tolist():
                    if bk not in map_block_to_epoch.keys():
                        print(map_block_to_epoch)
                        print(D.Dat["block"].unique().tolist())
                        assert False, "this block doesnt have an epoch assigned yet... give it..."
                        
                # now reassing each blcokt o a new epoch.
                def F(x):
                    return map_block_to_epoch[x["block"]]
                D.Dat = applyFunctionToAllRows(D.Dat, F, "epoch")
            else:
                print(grmeth)
                assert False

    # # fix a problem, sholdnt throw out epoch name
    # D.supervision_epochs_extract_orig() 

    if grouping_levels is None:
        # Then you did not enter it manually. extract it
        grouping_levels = D.Dat[grouping].unique().tolist() # note, will be in order in dataset (usually chron)

    return D, grouping, grouping_levels, feature_names, features_to_remove_nan, \
        features_to_remove_outliers, traintest_reassign_method, mapper_taskset_to_category, \
        mapper_auto_rename_probe_taskgroups, epoch_merge_dict

def taskgroup_reassign_by_mapper(D, mapper_taskset_to_category, 
        mapper_character_to_category=None, append_probe_status=True,
        what_use_for_default = "task_stagecategory"):
    """ Reassign values to D.Dat["taskgroup"], which represent meaningful group[s of
    tasks (e..g, task sets meant to test different kinds of generalization behavior), which
    can be given human-readable names.
    PARAMS
    - mapper_taskset_to_category, dict, mapping from set of tasks to string name of category.
    set of tasks defined by los (setname, setnum, taskinds), i.e, fixed tasks. 
    e.g., mapper_taskset_to_category = {
    ("grammar", 41, tuple([16, 18, 21, 22, 23, 27, 28])): "diff_beh",
    ("grammar", 41, tuple([17, 20, 24, 26])): "diff_beh_probes",
    ("grammar", 41, tuple([15, 19, 25])): "same_beh"}
    means that tasks with setname grammar, setnum 41, and in {15, 19, 25} will be 
    mapped to name "same_beh"
    - mapper_character_to_category, dict mapping from character name to category. 
    If using this, then mapper_taskset_to_category must be None. If character not included,
    then will default to taskset
    - append_probe_status, then appends string "-P", if this is probe
    - what_use_for_default, string, name of col in D.Dat, what to use if cant find a task
    in the inputed mapper.
    RETURNS:
    - for each ind, renames its taskgroup column, one of the following:
    --- the value in task_stagecategory, if this is random task.
    --- the value in task_stagecategory, if los info not in mapper_taskset_to_category
    --- the string category in mapper_taskset_to_category (see above).
    """

    if mapper_character_to_category is not None:
        assert mapper_taskset_to_category is None, "can only use one or the other"
    else:
        assert mapper_character_to_category is None, "can only use one or the other"

    if False:
        # {TODO}
        # Method 1, use tstruct index saved in dragmonkey. This is quick and auto., should do this always.
        # -- if 10/11/22 onwards
        D.taskclass_extract_tstruct_index()
        # -- if before (in progress)
        bp = D.blockparams_extract_single_combined_task_and_block(ind)
        bp["TaskSet"]
        print("todo: extract it from TaskSet by findind taskset that matches a given task's los")
    else:
        # Method 2: enter by hand, based on los.
        def taskgroup_rename(D, ind):
            isrand = D.Dat.iloc[ind]["random_task"]
            
            def _final_name(name):
                """ appends probe status if needed"""
                if append_probe_status:
                    isprobe = D.Dat.iloc[ind]["probe"]
                    if isprobe:
                        return f"{name}-P"
                    else:
                        return name
                else:
                    return name

            if isrand:
                # use the current random tasksetclass
                return _final_name(D.Dat.iloc[ind]["task_stagecategory"])
            else:

                if mapper_taskset_to_category is not None:
                    # OPTION 1 - use taskset
                    setname, setnum, taskind = D.taskclass_extract_los_info(ind)
                    if setname is not None:
                        # Then is not older tasks where cant find the set
                        # try to find this set entered by hand
                        for k, v in mapper_taskset_to_category.items():
                            setname_mapper = k[0]
                            setnum_mapper = k[1]
                            taskinds_mapper = k[2] # list

                            if setname==setname_mapper and setnum==setnum_mapper and taskind in taskinds_mapper:
                                # found this trial's category
                                return _final_name(v)
                    # if got to here, then you did not include this task in the mapper. use its taskset
                    # e..g, grammar-ss-41
                    return _final_name(D.Dat.iloc[ind][what_use_for_default])
                else:
                    # OPTION 2 - use character
                    charname = D.Dat.iloc[ind]["character"]
                    if charname in mapper_character_to_category.keys():
                        tg = mapper_character_to_category[charname]
                    else:
                        # use default, the taskset
                        tg = D.Dat.iloc[ind][what_use_for_default]
                    return _final_name(tg)

        list_taskgroup = []
        for ind in range(len(D.Dat)):
            list_taskgroup.append(taskgroup_rename(D, ind))

        # replace the column
        D.Dat["taskgroup"] = list_taskgroup
        print("[taskgroup_reassign_by_mapper], reassigned values in column: taskgroup")

    


def epoch_grouping_reassign_by_tasksequencer(D, map_tasksequencer_to_rule):
    """ Decide what is the epoch and level for each trial, based on the tasksequencer rule
    that was applied, using the objectclass version.
    - Looks into blockparams, so this assumes all tasks in a block are givent he same rule. 
    - Also uses the default blockparams (not hotkey updated).
    PARAMS:
    - D, Dataset
    - map_tasksequencer_to_rule, dict, where keys represent tasksequencer rule, and vals 
    are string names to assign to "epoch" column. (see eg. below)
    RETURNS:
    - moidifies the "epoch" column in D.Dat
    EXAMPLE:
    # Map from tasksequencer to epoch rule
    map_tasksequencer_to_rule = {}
    map_tasksequencer_to_rule[(None, None)] = "baseline"
    map_tasksequencer_to_rule[("directionv2", ("lr",))] = "rightward"
    map_tasksequencer_to_rule[("directionv2", ("rl",))] = "leftward"
    map_tasksequencer_to_rule[("directionv2", ("ud",))] = "downward"
    map_tasksequencer_to_rule[("directionv2", ("du",))] = "upward"
    """

    # Alternative version, look into object, but this doesnt have the category infomation
    # T = D.Dat.iloc[100]["Task"]
    # TT = T.Params["input_params"]
    # TT.get_tasknew()["Objects"]["ChunkList"]

    # Sanity check of types.
    for k, v in map_tasksequencer_to_rule.items():
        if k[0] is not None:
            assert isinstance(k[0], str)
            assert isinstance(k[1], tuple) or isinstance(k[1], str)

    # Based on the tasksequencer rule
    def _convert_to_prim(x):
        """
        PARAMS:
        - x, [str, nparray, nparay], represnting [shape, leve, rot]
        -- OR, [str], to be more general/abstract, e.g, [line]
        RETURNS:
        - <shape>-<level>-<rot>, e.g. # line-8-3
        -- OR <shape>, e.g, line
        """
        if len(x)==1:
            assert isinstance(x[0], str) # should be like "line"
            return x[0]
        elif len(x)==3:
            # like [str, nparray, nparay]
            return '-'.join([x[0], str(int(x[1])), str(int(x[2]))])
        else:
            print(x)
            assert False, "not sure why"

    def _index_to_rule(ind):
        tp = D.blockparams_extract_single_taskparams(ind)
        # new, post 9/19/22 - using taskparams
        ver = tp["task_objectclass"]["tasksequencer_ver"]
        prms = tp["task_objectclass"]["tasksequencer_params"] # list.

        if len(ver)==0 and len(prms)==0:
            # Then no supervision
            ver = None
            p = None
        elif ver=="directionv2":
            p = tuple(prms[0])
        elif ver=="direction":
            # The params is an angle in radians:
            # e..g,  prms= [array(3.14159265), 'default', array(1.)]

            assert prms[1]=="default", "assuming take remix of default chunk"
            assert prms[2]==1.

            p = f"{prms[0]:.2f}"


        elif ver in ["prot_prims_in_order", "prot_prims_chunks_in_order"]:
            # convert to sequence of prims.
            # e..g, p = ('line-8-3', 'V-2-4', 'Lcentered-4-3')
            list_prims = prms[0]
            list_prims_str = [_convert_to_prim(x) for x in list_prims]
            p = tuple(list_prims_str)
        elif ver=="hack_220829":
            # (AB)(n), like lollipop, but hacked for today, hard coded for specific prims today.
            # 8/29/22
            p = tuple(["hack_220829"])
        elif ver=="prot_prims_in_order_AND_directionv2":
            # TYhen is hierarchcayl, first order by shape, then within shapes order by direction
            list_prims = prms[0]
            # each prim in list prims could be 3-list, like ['line', 1, 1] or 1-list, ['line']
            list_prims_str = [_convert_to_prim(x) for x in list_prims] # list of strings
            direction = prms[1][0] # topright
            p = tuple(list_prims_str + [direction])
        elif ver=="randomize_strokes":
            assert len(prms)==0
            p = tuple([ver])
        elif ver=="shape_chunk_concrete":
            # ver = shape_chunk_concrete
            # prms = ['lolli', ['D', 'R']]
            # converts to:
            # p = ('lolli', ('D', 'R'))
            if not len(prms)==2: 
                print(prms)
                assert False, "code up for this"

            chunkname = prms[0] # "lolli"
            prmsinner = prms[1]
            if len(prms)>2:
                print(prms)
                assert False, "code it, "
            assert isinstance(chunkname, str)
            p = [chunkname]
            p.append(tuple(prmsinner))
            p = tuple(p)
        elif ver in ["cols_direction", "rows_direction"]:
            # prms == ['down', 'right']
            for this in prms:
                assert isinstance(this, str)
            p = tuple(prms)
        else:
            print(ver)
            print(prms)
            assert False

        return map_tasksequencer_to_rule[(ver, p)]

    # For each trial, get its rule
    list_rule =[]
    for ind in range(len(D.Dat)):
        list_rule.append(_index_to_rule(ind))
    # Assign rule back into D.Dat
    D.Dat["epoch"] = list_rule
    D.Dat["epoch_rule_tasksequencer"] = list_rule # since epoch _might_ change, save a veresion here.
    print("Modified D.Dat[epoch]")
    print("These counts for epochs levels: ")
    print(D.Dat["epoch"].value_counts())
    print("These counts for epoch_rule_tasksequencer levels: ")
    print(D.Dat["epoch_rule_tasksequencer"].value_counts())

def preprocessDat(D, expt, get_sequence_rank=False, sequence_rank_confidence_min=None,
    remove_outliers=False, sequence_match_kind=None, extract_motor_stats=False,
    score_all_pairwise_within_task=False, extract_features = False, 
    only_keep_trials_across_groupings=False):
    """ wrapper for preprocessing, can differ for each expt, includes
    both general and expt-specific stuff.
    INPUT:
    - get_sequence_rank, then gets rank of beh sequence using parses (rank of efficiency 
    of best-matching parse). NOTE: this removes all trials where strokes_beh cannot be compared
    to strokes_task - by defualt removes cases where num strokes not equal. Should be qwuick, since
    requires preprocessing with D.planner... methods.
    - only_if_sequence_different_across_grouping, then only trials where sequence is not used in other groups.
    THis ideally goes with get_sequence_rank=True, sequence_rank_confidence_min=0.1, as only make sense if
    sequence assignment is accurate/confident.
    NOTE:
    - if D.Dat ends up being empty, then returns None
    - if all flags False, then doesnt do any mods to D, just returns groupings, etc.
    """
    from pythonlib.tools.pandastools import filterPandas, aggregGeneral, applyFunctionToAllRows
    from pythonlib.drawmodel.strokedists import distscalarStrokes
    from pythonlib.dataset.dataset_preprocess.probes import taskgroups_assign_each_probe

    if hasattr(D, "_analy_preprocess_done"):
        if D._analy_preprocess_done:
            assert False, "already done preprocess!! You probably ran this in Dataset(). You can just skip this current call to preprocessDat()"

    # (1) Warnings
    if expt=="neuralprep4":
        print("First day (210427) need to take session 5. Second day take all. ..")
        assert False

    # (2) Extract new derived varaibles
    # -- Plan time
    if "holdtime" in D.Dat.columns and "delaytime" in D.Dat.columns:
        tmp = D.Dat["holdtime"] - D.Dat["delaytime"]
        tmp[tmp<0.] = 0.
        D.Dat["plan_time"] = tmp

    # (3) Apply grouping variabples + prune dataset
    print("- starting/ending len (grouping params):")
    print(len(D.Dat))
    D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, features_to_remove_nan, \
        features_to_remove_outliers, traintest_reassign_method, \
        mapper_taskset_to_category, mapper_auto_rename_probe_taskgroups, epoch_merge_dict \
        = _groupingParams(D, expt)
    print(len(D.Dat))

    # Only keep characters that have at lesat one trial across all grouping levels.
    if only_keep_trials_across_groupings:
        D.removeTrialsExistAcrossGroupingLevels(GROUPING, GROUPING_LEVELS)

    # -- std of stroke and gaps
    if extract_motor_stats:
        if "motorevents" in D.Dat.columns:
            def F(x, ver):
                ons = x["motorevents"]["ons"]
                offs = x["motorevents"]["offs"]
                if len(ons)==1:
                    return np.nan
                strokedurs = np.array(offs) - np.array(ons)
                gapdurs = np.array(ons)[1:] - np.array(offs)[:-1]
                if ver=="stroke":
                    return np.std(strokedurs)
                elif ver=="gap":
                    return np.std(gapdurs)
            D.Dat = applyFunctionToAllRows(D.Dat, lambda x: F(x, "stroke"), "sdur_std")
            D.Dat = applyFunctionToAllRows(D.Dat, lambda x: F(x, "gap"), "gdur_std")

    ### EXTRACT FEATURES
    # - hausdorff, offline score
    if extract_features:
        D.extract_beh_features(feature_list = FEATURE_NAMES)
        D.score_visual_distance()

    # -- pull out variables into separate columns
    if extract_motor_stats:
        for col in ["motortiming", "motorevents"]:
            keys = D.Dat[col][0].keys()
            for k in keys:
                def F(x):
                    return x[col][k]
                D.Dat = applyFunctionToAllRows(D.Dat, F, k)

        # - derived motor stats
        D.Dat["stroke_speed"] = D.Dat["dist_strokes"]/D.Dat["sdur"]
        D.Dat["gap_speed"] = D.Dat["dist_gaps"]/D.Dat["isi"]
        D.Dat["onset_speed"] = D.Dat["dist_raise2firsttouch"]/D.Dat["time_raise2firsttouch"]
        D.Dat["offset_speed"] = D.Dat["dist_touchdone"]/D.Dat["time_touchdone"]

        # D.Dat["total_distance"] = D.Dat["dist_strokes"] + D.Dat["dist_gaps"] + D.Dat["dist_raise2firsttouch"] + D.Dat["dist_touchdone"]
        # D.Dat["total_time"] = D.Dat["sdur"] + D.Dat["isi"] + D.Dat["time_raise2firsttouch"] + D.Dat["time_touchdone"]
        # D.Dat["total_speed"] = D.Dat["total_distance"]/D.Dat["total_time"]
        D.Dat["total_distance"] = D.Dat["dist_strokes"] + D.Dat["dist_gaps"]
        D.Dat["total_time"] = D.Dat["sdur"] + D.Dat["isi"]
        D.Dat["total_speed"] = D.Dat["total_distance"]/D.Dat["total_time"]

        D.Dat["dist_per_gap"] = D.Dat["dist_gaps"]/(D.Dat["nstrokes"]-1)
        D.Dat["dist_per_stroke"] = D.Dat["dist_strokes"]/(D.Dat["nstrokes"])


    # (4) Sequences more similar within group than between?
    if score_all_pairwise_within_task:
        from pythonlib.dataset.analy import score_all_pairwise_within_task
        from pythonlib.dataset.analy import score_alignment
        DIST_VER = "dtw_split_segments"

        # - score all pairwise, trials for a given task
        SCORE_COL_NAMES = score_all_pairwise_within_task(D, GROUPING, GROUPING_LEVELS,
            DIST_VER, DONEG=True)

        # - score alignment
        score_alignment(D, GROUPING, GROUPING_LEVELS, SCORE_COL_NAMES)
    else:
        SCORE_COL_NAMES = []


    # - score beh sequence rank relative to parses
    if get_sequence_rank:
        sequence_get_rank_vs_task_permutations_quick(D)
        FEATURE_NAMES = sorted(set(FEATURE_NAMES + ["effic_rank", "effic_summary", "effic_confid"]))
    # print("pruning by confidence of rank")
    # print(len(D.Dat))
    # print(D.Dat["effic_confid"])
    # print(len(np.isnan(D.Dat["effic_confid"])))
    print("- starting/ending len (getting sequence):")
    print(len(D.Dat))
    if get_sequence_rank and sequence_rank_confidence_min is not None:
        D.Dat = D.Dat[D.Dat["effic_confid"]>=sequence_rank_confidence_min]
        D.Dat = D.Dat.reset_index(drop=True)
    if len(D.Dat)==0:
        return None
    print(len(D.Dat))

    # =========
    if sequence_match_kind in ["same", "diff"]:
        print("-- Doing only_if_sequence_different_across_grouping")
        print(len(D.Dat))
        D.analy_match_sequence_discrete_per_task(groupby=GROUPING, 
            grouping_levels=GROUPING_LEVELS, ver = sequence_match_kind, 
            print_summary=True)
        print(len(D.Dat))      
    else:
        assert sequence_match_kind is None

    # ======== CLEAN, REMOVE NAN AND OUTLIERS
    # - Remove nans
    D.removeNans(columns=features_to_remove_nan) 
    # - Replace outliers with nans
    for F in features_to_remove_outliers:
        D.removeOutlierRowsTukey(F, niqr=2.5, replace_with_nan=True)
    if remove_outliers:
        D.removeOutlierRows(FEATURE_NAMES, [0.1, 99.9])

    # FIgure out task probes (0 and 1)
    probe_list = []
    for i in range(len(D.Dat)):
        t_p = D.Dat.iloc[i]["Task"].Params["input_params"]
        probe_val = t_p.info_summarize_task()["probe"]["probe"]
        probe_list.append(probe_val)
    D.Dat["probe"] = probe_list

    ### Rename things as monkey train test depenidng on expt
    if expt in ["gridlinecircle", "chunkbyshape1", "resize1"]:
        assert traintest_reassign_method=="fixed", "I had previously hard coded as such"
    preprocess_task_train_test(D, expt, method=traintest_reassign_method)

    # () Note that preprocess done
    D._analy_preprocess_done=True

    # Append fixed taks setds information
    D.taskclass_extract_los_info_append_col()

    #### SUPERVISION - get supervision stages, i.e, tuples
    D.supervision_summarize_into_tuple(method="verbose", new_col_name = "supervision_stage_new")

    # Reassign taskgroup. by default uses value in task_stagecategory.
    # i) first, do automatic detection of probe categories for each character
    if D.taskclass_is_new_version(-1):
        # 1) always start by redefining taskgroup using task_stagecategory
        def F(x):
            # Rename taskgroup if it is undefined.
            if x["taskgroup"]=="undefined":
                # Then replace 
                return x["task_stagecategory"]
            else:
                # dont replace
                return x["taskgroup"]
        D.Dat = applyFunctionToAllRows(D.Dat, F, "taskgroup")

        # 2) Autoamticlaly categorize probes based on task features?
        if mapper_auto_rename_probe_taskgroups:
            map_task_to_taskgroup = taskgroups_assign_each_probe(D)

            # append_probe_status = False, so that only appends 1x (the last call)
            taskgroup_reassign_by_mapper(D, None, map_task_to_taskgroup, append_probe_status=False,
                what_use_for_default="taskgroup")

        # 3) Finally, use hand-entered if they exist, overwriting anything detected auto 
        # (not doing any auto, therefore what_use_for_default = taskgroup)
        taskgroup_reassign_by_mapper(D, mapper_taskset_to_category, append_probe_status=True,
            what_use_for_default="taskgroup")


    # Print outcomes
    print("GROUPING", GROUPING)
    print("GROUPING_LEVELS", GROUPING_LEVELS)
    print("FEATURE_NAMES", FEATURE_NAMES)
    print("SCORE_COL_NAMES", SCORE_COL_NAMES)

    # Get new column: date_epoch
    x = []
    for ind in range(len(D.Dat)):
        x.append(D.Dat.iloc[ind]["date"][2:])
    D.Dat["date_MMDD"] = x

    D.grouping_append_col(["date_MMDD", "epoch"], "date_epoch", use_strings=True, strings_compact=True)

    # Extract concise supervision stage
    D.supervision_summarize_into_tuple(method="concise", new_col_name="supervision_stage_concise")
    D.supervision_semantic_string_append("supervision_stage_semantic")
    
    def F(x):
        return (x["epoch"], x["supervision_stage_concise"])
    D.Dat = applyFunctionToAllRows(D.Dat, F, "epoch_superv")

    # Merge epochs;
    if len(epoch_merge_dict)>0:
        print("MERGING EPOCHS...")
        for epoch_new, list_epoch_old in epoch_merge_dict.items():
            print("Merging these epochs:", list_epoch_old, "... into this:", epoch_new)
            D.supervision_epochs_merge_these(list_epoch_old, epoch_new)        
            
    # Since epoch might change...
    D.Dat["epoch_rule_tasksequencer"] = D.Dat["epoch"] # since epoch _might_ change, save a veresion here.

    # fix a problem, sholdnt throw out epoch name
    D.supervision_epochs_extract_orig() 

    # if taskgroup name is too long, then prune it. otherwise can lead to seaborn plotting errors...
    def F(x):
        if len(x["task_stagecategory"])>36:
            return x["task_stagecategory"][:15] + "+" + x["task_stagecategory"][-15:]
        else:
            return x["task_stagecategory"]
    D.Dat = applyFunctionToAllRows(D.Dat, F, "task_stagecategory")

    def F(x):
        if len(x["taskgroup"])>36:
            return x["taskgroup"][:15] + "+" + x["taskgroup"][-15:]
        else:
            return x["taskgroup"]
    D.Dat = applyFunctionToAllRows(D.Dat, F, "taskgroup")

    # Generate behclass. This is used frequenctly, so I decided to aklways do this
    D.behclass_preprocess_wrapper(skip_if_exists=False)
    
    ############## OLD THINGS, delete if not using (to avoid confusion)
    del D.Dat["supervision_params"] # from drawmonkey.

    return D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES


def preprocess_task_train_test(D, expt, method="probes"):
    """ Clean up naming of tasks as train or test
    in the monkey_train_or_test key of D.Dat. Makes sure that all rows have either
    "train" or "test" for this column.
    PARAMS:
    - method, str, how to define train/test... This may depend on expt, etc. and may
    need to be hand-tuned.
    RETURNS:
    - modifies D.Dat
    """ 

    if method=="fixed":
        # randoom = train, fixed = test
        # train were all random tasks, test were all fixed.
        key = "random_task"
        list_train = [True]
        list_test = [False]
    elif method=="probes":
        # probes==1 --> test, ... probes==0 --> train
        key = "probe"
        list_train = [0]
        list_test = [1]
    elif method=="supervision_except_color":
        # online supervision (dont care about color) --> train
        # no online supervision --> test
        if "supervision_online" not in D.Dat.columns:
            # Extract it
            D.supervision_summarize_whether_is_instruction()
        key = "supervision_online"
        list_train = [True]
        list_test = [False]
    elif method=="probe_and_supervision":
        # Test, if probe==1 and supervision is off
        assert False, "code it, take conjuntion of two things above"
    else:
        print(method)
        assert False

    # Do reassignment
    D.analy_reassign_monkeytraintest(key, list_train, list_test)


def extract_expt_metadat(expt=None, animal=None, rule=None, 
        metadat_dir = f"{PATH_DRAWMONKEY_DIR}/expt_metadat", 
        metadat_dir_daily = f"{PATH_DRAWMONKEY_DIR}/expt_metadat_daily"):
    """ Get matadata for this expt, without having to load a Dataset first. useful if want to
    know what rules exits (for exampel) before extracting those datsaets.
    This is looks for expt metadat yaml files defined usually in drawmonkey repo.
    PARAMS:
    - expt, animal, rule, all strings or None. If None, then gets ignores this feature (e.g., animal).
    e.g., expt="lines5" gets all rules and animals under lines5.
    - metadat_dir, string name of path where yaml files saved. 
    RETURNS:
    - list_expts, list of lists of strings, where inner lists are like ['chunkbyshape1', 'rule2', 'Pancho']
    """
    from pythonlib.tools.expttools import findPath, extractStrFromFname

    # construct the path wildcard
    pathwildcard = []
    if expt is not None:
        pathwildcard.append(f"{expt}-")
    if rule is not None:
        pathwildcard.append(f"{rule}-")
    if animal is not None:
        pathwildcard.append(f"{animal}.")

    list_path = findPath(metadat_dir, [pathwildcard], None)
    list_expts = [(extractStrFromFname(path, "-", "all")) for path in list_path]

    if len(list_expts)==0:
        # Search for daily
        list_path = findPath(metadat_dir_daily, [pathwildcard], None)
        list_expts_daily = [(extractStrFromFname(path, "-", "all")) for path in list_path]

        list_expts = list_expts+list_expts_daily
        
    return list_expts


def get_rulelist(animal, expt):
    """ Autoamtically Extract all existing rules
    RETURNS:
    - list of str, unique rules
    """
    list_expts = extract_expt_metadat(animal=animal, expt=expt, metadat_dir=f"{PATH_DRAWMONKEY_DIR}/expt_metadat") ##CHANGE## metadat_dir if necessary
    rulelist = [e[1] for e in list_expts]
    assert len(rulelist)>0
    if not len(set(rulelist))==len(rulelist):
        print("rulelist:", rulelist)
        print(animal, expt)
        assert False, "some redundant?"
    return rulelist




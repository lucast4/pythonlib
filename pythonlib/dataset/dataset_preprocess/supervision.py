""" To extract supervision params implemented online on each trial, and related tings.

"""

import numpy as np

def extract_supervision_params(D, ind):
    """ Extract the supervision params for this trial index in this dataset
    NOTE: some of these requires objectclass version of tasks, and so may break. 
    """

    # Get the blockparams and taskparams
    taskparams = D.blockparams_extract_single_taskparams(ind)
    blockparams = D.blockparams_extract_single(ind)


    ########### ABORT MODES
    ABORT_ON = D.Dat.iloc[ind]["abort_params"]["on"]
    ABORT_MODES = list(D.Dat.iloc[ind]["abort_params"]["modes"])
    ABORT_PUNISH = blockparams["params_task"]["enableAbort_punish_if_abort"]==1
    if "failed_rule" in ABORT_MODES:
        # Then also include aborts due to failing ObjectClass rules
        for rule in taskparams["task_objectclass"]["RuleList"]:
            assert len(rule[1])==0, "this rule has a param, have to incorportae this info into ABORT_MODES somehow"
            ABORT_MODES.append(rule[0])


    ############## SEQUENCE SUPERVISION
    seq = blockparams["sequence"]

    if seq["ver"]=="objectclass" and seq["on"]==1 and seq["manipulations"] == ['alpha', 'active_chunk'] and taskparams["task_objectclass"]["Params"]["ChunksDone"]["order"]=="any_order":
        SEQUENCE_SUP = "active_chunk"
        assert taskparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer"]==1
        SEQUENCE_ALPHA = taskparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer_alpha"]
        
        # chunk active abort [usually with chunk active]
        assert ['chunks_any_order', {}] in taskparams["task_objectclass"]["RuleList"]
        assert "failed_rule" in ABORT_MODES
        
    elif seq["ver"]=="objectclass" and seq["on"]==1 and seq["manipulations"] == ['alpha', 'mask'] and taskparams["task_objectclass"]["Params"]["ChunksDone"]["order"]=="in_order":
        #SEQUENCE_SUP = "mask"
        assert taskparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer"]==1
        SEQUENCE_ALPHA = taskparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer_alpha"]
        
        # Sequence abort (in order) [usually with mask]
        if ['chunks_in_order', {}] in taskparams["task_objectclass"]["RuleList"] and "failed_rule" in ABORT_MODES:
            SEQUENCE_SUP = "mask"
        else:
            SEQUENCE_SUP = "char_strokes"

    elif seq["on"]==0:
        SEQUENCE_SUP = "off"
        #assert taskparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer"]==1
        SEQUENCE_ALPHA = taskparams["task_objectclass"]["Params"]["UpdateVisibleChunks"]["show_lowalpha_layer_alpha"]
    else:
        print(seq)
        print(taskparams["task_objectclass"]["Params"])
        assert False


    ############# COLOR
    ### COLOR
    #print(blockparams["columns"])
    COLOR_ON = blockparams["colormod"]["strokes"]["on"]==1

    if "color_method" not in blockparams["colormod"]["strokes"].keys():
        if blockparams["colormod"]["strokes"]["randomize_each_stroke"]==1:
            COLOR_METHOD = 'randomize_each_stroke'
        else:
            COLOR_METHOD = ''
    elif len(blockparams["colormod"]["strokes"]["color_method"])>0:
        COLOR_METHOD = blockparams["colormod"]["strokes"]["color_method"]
    elif blockparams["colormod"]["strokes"]["randomize_each_stroke"]==1:
        COLOR_METHOD = 'randomize_each_stroke'
    else:
        COLOR_METHOD = ''



    ################ ONLINE VISUAL FB
    p = taskparams["task_objectclass"]["AllPtsVisCriteria"]
    c = taskparams["task_objectclass"]["CtrlPtsVisCriteria"]

    if p == [['always_vis', {}, 'and', 0]] and c == [['always_vis', {}, 'and', 0]]:
        VISUALFB_METH = "none"
    elif p == [['not_touched', {}, 'and', 0]] and c == [['not_touched', {}, 'and', 0]]:
        VISUALFB_METH = "all"
    elif p == [['always_vis', {}, 'and', 0]] and c == [['not_touched', {}, 'and', 0], ['no_isolated_pts', {}, 'or', 1, 1]]:
        VISUALFB_METH = "control_pts"
    elif p == [['always_vis', {}, 'and', 0]] and c == [['not_touched', {}, 'and', 0]]:
        VISUALFB_METH = "control_pts_norespawn"
    else:
        assert False



    ################# guide dynamic
    GUIDEDYN_ON = blockparams["guide_dynamic"]["on"]==1
    GUIDEDYN_VER = blockparams["guide_dynamic"]["version"]
    GUIDEDYN_CHUNKSFLY = blockparams["guide_dynamic"]["chunks_fly_in_together"] ==1;
    GUIDEDYN_DURATION = blockparams["guide_dynamic"]["duration"] ==1;



    ############### suonds

    assert blockparams["DonenessTracker"]["make_sound_on_chunk_switch"]==0

    SOUNDS_HIT_VER = taskparams["task_objectclass"]["Feedback"]["hit_sound_ver"]
    if taskparams["task_objectclass"]["Feedback"]["chunk_active_change_sound_multiplier"]>0:
        SOUNDS_CHUNK_CHANGE = taskparams["task_objectclass"]["Feedback"]["chunk_active_change_sound_ver"]
    else:
        SOUNDS_CHUNK_CHANGE = "none"

    SOUNDS_STROKES_DONE = taskparams["task_objectclass"]["Feedback"]["stroke_done_sound_ver"]
    SOUNDS_TRIAL_ONSET = blockparams["sounds"]["play_sound_trial_onset"]==1
    SOUNDS_ALLPTSDONE = blockparams["sounds"]["play_done_sound_on"]==1


    ############# CONTROL PTS
    CONTROL_SIZE = blockparams["control_pts"]["size_mult"]
    CONTROL_COLOR_DIFF = blockparams["control_pts"]["color_diff"]
    CONTROL_VISIBLE = CONTROL_SIZE>1 or np.any(CONTROL_COLOR_DIFF!=0)
    CONTROL_GEN_METHOD = taskparams["task_objectclass"]["Params"]["ControlPts"]["generate_method"]

    ################ PEANUT
    PEANUT_ALPHA = blockparams["sizes"]["PeanutAlpha"]



    ############## REPORT DONE KMETHOD
    def extract_taskdone_criteria(taskparams):
        # BlockParams(ind).task_objectclass.TaskDoneCriteria = struct;
        # BlockParams(ind).task_objectclass.TaskDoneCriteria(1).ver = 'ctrl_pts_all';
        # BlockParams(ind).task_objectclass.TaskDoneCriteria(1).params = {};
        # BlockParams(ind).task_objectclass.TaskDoneBool = 'and';
        
        if isinstance(taskparams["task_objectclass"]["TaskDoneCriteria"], dict):
            ver = taskparams["task_objectclass"]["TaskDoneCriteria"]["ver"]
            assert len(taskparams["task_objectclass"]["TaskDoneCriteria"]["params"])==0, "incorproate this info into ver..."
        else:
            print(taskparams["task_objectclass"])
            assert False, "is this multiple criteria?"
        return ver

    if blockparams["params_task"]["donebutton_criterion"]:
        # Used done button
        meth = blockparams["params_task"]["donebutton_criterion"]
        DONE_METHOD = f"donebutton-{meth}"
    #     DONE_PARAMS = meth
    else:
    #     BlockParams(ind).params_task.reportDoneMethod = ...
    #     {'taskobject', 'numstrokes_rel_task'}; % nstrokes, so that can end if keeps trying and failing.
    #     BlockParams(ind).params_task.reportDoneParam= ...
    #         {{}, 4};

        DONE_METHOD = []
    #     DONE_PARAMS = []
        rdm = blockparams["params_task"]["reportDoneMethod"]
        rdp = blockparams["params_task"]["reportDoneParam"]
        for meth, par in zip(rdm, rdp):
            if meth=="taskobject":
                # then look into taskparams to figure out
                methodthis = extract_taskdone_criteria(taskparams)
                assert len(par)==0
            else:
                print(meth)
                print(par)
                assert False, "convert (meth, par) into tuple?"
            DONE_METHOD.append(methodthis)
    #         DONE_PARAMS.append(paramsthis)

    
    ##### SCREEN POST VER
    SCREENPOST_ON = blockparams["scenes"]["sceneSchedule"]["samp1"][3]==1
    bpthis = blockparams["postscene"]
    spd = bpthis["dynamic"]
    spd_ver = bpthis["dynamic_ver"]

    assert bpthis["samp1_only_show_untouched"]==0
    SCREENPOST_ALPHA = bpthis["samp1_force_alpha_to"]
    SCREENPOST_SIZE = bpthis["samp1_force_size_to"]

    assert bpthis["pnut1_interp"]["on"]==1
    assert bpthis["pnut1_interp"]["params"] == ['upsample', 3.]
    
    # print(spd)
    # print(bpthis.ke)
    # print(spd["ver"])
    if spd["on"]==1 and spd_ver=="flash_in_order" and bpthis["dynamic_params"] == [100., 'all_strokes']:
        SCREENPOST_DYNAMIC_VER = "all_strokes"
    elif spd["on"]==1 and spd_ver=="flash_in_order" and bpthis["dynamic_params"] == [100., 'active_chunk']:
        SCREENPOST_DYNAMIC_VER = "flash_active_chunk"
    elif spd["on"]==1 and spd_ver=="flash_in_order" and bpthis["dynamic_params"] == [100., 'strokes_notdone']:
        SCREENPOST_DYNAMIC_VER = "flash_missed_taskstrokes"
    elif spd["on"]==0:
        SCREENPOST_DYNAMIC_VER = "off"
    else:
        print(1)
        print(spd)
        print(2)        
        print(spd_ver)
        print(3)
        print(bpthis["dynamic_params"])
        assert False
        

    params = {
        "ABORT_ON":ABORT_ON,
        "ABORT_MODES":ABORT_MODES,
        "ABORT_PUNISH":ABORT_PUNISH,
        
        "SEQUENCE_SUP":SEQUENCE_SUP,
        "SEQUENCE_ALPHA":SEQUENCE_ALPHA,
        
        "COLOR_ON":COLOR_ON,
        "COLOR_METHOD":COLOR_METHOD,
        
        "VISUALFB_METH":VISUALFB_METH,
        
        "GUIDEDYN_ON":GUIDEDYN_ON,
        "GUIDEDYN_VER":GUIDEDYN_VER,
        "GUIDEDYN_CHUNKSFLY":GUIDEDYN_CHUNKSFLY,
        "GUIDEDYN_DURATION":GUIDEDYN_DURATION,

        "SOUNDS_HIT_VER":SOUNDS_HIT_VER,
        "SOUNDS_STROKES_DONE":SOUNDS_STROKES_DONE,
        "SOUNDS_TRIAL_ONSET":SOUNDS_TRIAL_ONSET,
        "SOUNDS_ALLPTSDONE":SOUNDS_ALLPTSDONE,

        "CONTROL_SIZE":CONTROL_SIZE,
        "CONTROL_COLOR_DIFF":CONTROL_COLOR_DIFF,
        "CONTROL_VISIBLE":CONTROL_VISIBLE,
        "CONTROL_GEN_METHOD":CONTROL_GEN_METHOD,

        "PEANUT_ALPHA":PEANUT_ALPHA,

        "DONE_METHOD":DONE_METHOD,

        "SCREENPOST_ON":SCREENPOST_ON,
        "SCREENPOST_ALPHA":SCREENPOST_ALPHA,
        "SCREENPOST_SIZE":SCREENPOST_SIZE,
        "SCREENPOST_DYNAMIC_VER":SCREENPOST_DYNAMIC_VER
    }



    return params 
            
                





if __name__=="__main__":
    from pythonlib.dataset.dataset import Dataset
    from pythonlib.drawmodel.parsing import *
    import numpy as np
    import torch

    # Parse a single datset, but iter over multiple datsetes. 
    # Parses are one for each unique task, this is more efficiecnt that each row of datsaet.
    # Default is to add extra_junction to make sure to get corners.

    ##################### PARAMS
    # OK version (good balance between fast and variety of parses)
    # params_parse = {
    #     "configs_per":10,
    #     "trials_per":50,
    #     "max_ntrials":75, 
    #     "max_nwalk":75,
    #     "max_nstroke":100
    # }

    # # FAST VERSION
    # params_parse = {
    #     "configs_per":5,
    #     "trials_per":20,
    #     "max_ntrials":50, 
    #     "max_nwalk":50,
    #     "max_nstroke":50
    # }
    # VERY FAST
    params_parse = {
        "configs_per":5,
        "trials_per":10,
        "max_ntrials":25, 
        "max_nwalk":25,
        "max_nstroke":25
    }
    return_in_strokes_coords = True
    kparses = 10
    # animal = "Red"

    use_extra_junctions=True
    # score_ver = "travel"
    score_ver = "travel_from_orig" # better, since differentiates 2 tasks thjat are just flipped (and so will not throw one of them out)
    score_norm = "negative"
    image_WH = 105

    #################### RUN
    for animal in ["Pancho"]:
        # Load datasets
        if animal == "Red":
            path_list = [
                "/data2/analyses/database/Red-lines5-formodeling-210329_005719",
                "/data2/analyses/database/Red-arc2-formodeling-210329_005550",
                "/data2/analyses/database/Red-shapes3-formodeling-210329_005200",
                "/data2/analyses/database/Red-figures89-formodeling-210329_005443"
            ]
            # path_list = [
            #     "/data2/analyses/database/Red-figures89-formodeling-210329_005443"
            # ]
        elif animal=="Pancho":
            path_list = [
                "/data2/analyses/database/Pancho-lines5-formodeling-210329_014835",
                "/data2/analyses/database/Pancho-arc2-formodeling-210329_014648",
                "/data2/analyses/database/Pancho-shapes3-formodeling-210329_002448",
                "/data2/analyses/database/Pancho-figures89-formodeling-210329_000418"
            ]
            # path_list = [
            #     "/data2/analyses/database/Pancho-shapes3-formodeling-210329_002448",
            #     "/data2/analyses/database/Pancho-figures89-formodeling-210329_000418"
            # ]
            
        append_list = None

        for paththis in path_list:
            # Load single dataset
            D = Dataset([paththis], append_list)

            # No need to preprocess data, since can just apply any preprocess afterwards to both task and beahviro

            # Get sketchpad edges
            maxes = []
            for k, v in D.Metadats.items():
                maxes.append(np.max(np.abs(v["sketchpad_edges"].flatten())))
            canvas_max_WH = np.max(maxes)

            # origin for tasks, is generally around (0, 50) to (0,100). so just hard code here.
            origin = torch.tensor([image_WH/2, -(0.45*image_WH)])
            # alternatively, could look into D.Dat["origin"], and convert to image coords [(1,105), (-105, -1)]

            # For each row, parse its task
            score_fn = lambda parses: score_function(parses, ver=score_ver, 
                                                     normalization=score_norm, use_torch=True, origin=origin)

            if False:
                # Just testing, pick a random trial
                import random
                ind = random.sample(range(len(D.Dat)), 1)[0]
                strokes = D.Dat["strokes_task"].values[ind]

                if False:
                    from pythonlib.tools.stroketools import strokesInterpolate2
                    strokes = strokesInterpolate2(strokes, N=["npts", 100])
                else:
                    pass
                parses, log_probs_k = get_parses_from_strokes(strokes, canvas_max_WH, 
                                                              use_extra_junctions=use_extra_junctions, plot=True,
                                                             return_in_strokes_coords=True, k=5)

            # save params
            params_parse["canvas_max_WH"] = canvas_max_WH
            params_parse["use_extra_junctions"] = use_extra_junctions
            params_parse["return_in_strokes_coords"] = return_in_strokes_coords
            params_parse["score_ver"] = score_ver
            params_parse["score_norm"] = score_norm

            # OLD VERSION - GET EACH ROW
        #     # Collect parses
        #     PARSES = []
        #     for row in D.Dat.iterrows():
        #         if row[0]%100==0:
        #             print(row[0])
        #         strokes = row[1]["strokes_task"]
        #         index = row[0]
        #         trial_id = row[1]["trialcode"]
        #         unique_task_name = row[1]["unique_task_name"]
        #         character = row[1]["character"]

        #         parses, log_probs = get_parses_from_strokes(strokes, canvas_max_WH, 
        #                                                   use_extra_junctions=use_extra_junctions, plot=False,
        #                                                  return_in_strokes_coords=return_in_strokes_coords, k=k,
        #                                                    configs_per = params_parse["configs_per"],
        #                                                    trials_per = params_parse["trials_per"],
        #                                                    max_ntrials = params_parse["max_ntrials"],
        #                                                    max_nstroke = params_parse["max_nstroke"],
        #                                                    max_nwalk = params_parse["max_nwalk"],
        #                                                    )

        #         PARSES.append(
        #             {"strokes_task":strokes,
        #              "index_dat":index,
        #              "trial_id":trial_id,
        #              "unique_task_name":unique_task_name,
        #              "character":character,
        #              "parses":parses,
        #              "parses_log_probs":log_probs}
        #         )

            # NEW VERSION - only do once for each unique task
            tasklist = sorted(list(set(D.Dat["unique_task_name"])))
            # Collect parses
            PARSES = []

            for i, task in enumerate(tasklist):
                # find the first row that has this task
                row = D.Dat[D.Dat["unique_task_name"]==task].iloc[0]
                assert row["unique_task_name"]==task

        #         if i%20==0:
        #             print(i, "-",  task)
                print(i, "-",  task)
                strokes = row["strokes_task"]

                parses, log_probs = get_parses_from_strokes(strokes, canvas_max_WH, 
                                                          use_extra_junctions=use_extra_junctions, 
                                                          score_fn=score_fn,
                                                            plot=False, image_WH=image_WH,
                                                            return_in_strokes_coords=return_in_strokes_coords, 
                                                            k=kparses, configs_per = params_parse["configs_per"],
                                                           trials_per = params_parse["trials_per"],
                                                           max_ntrials = params_parse["max_ntrials"],
                                                           max_nstroke = params_parse["max_nstroke"],
                                                           max_nwalk = params_parse["max_nwalk"],
                                                           )
                assert len(parses)>0, "why?"

                PARSES.append(
                    {"strokes_task":strokes,
                     "unique_task_name":task,
                     "parses":parses,
                     "parses_log_probs":log_probs}
                )

            # === save as dataframe
            import os
            import pandas as pd
            import pickle
            
            DIR, FNAME = os.path.split(paththis)
            fname_parse = f"{DIR}/{FNAME}/parses.pkl"
            print("Saving at:")
            print(fname_parse)

            PARSES  = pd.DataFrame(PARSES)
            PARSES.to_pickle(fname_parse)

            fname_params = f"{DIR}/{FNAME}/parses_params.pkl"
            with open(fname_params, "wb") as f:
                pickle.dump(params_parse, f)
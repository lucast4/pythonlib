""" Dataset preprocessed so that 
has information about sequential context
"""


import pandas as pd

def preprocess_dataset(D, n_strok_max = 9, version="tokens"):
    """ Extract new columns into self.Dat, for each trial noting sequence 
    context inforamtion ,such as n strokes, shape of first stroke, etc
    PARAMS:
    - n_strok_max, int, max num strokes to etract info for, large is fine.
    - version, where to find stroke shape labels from.
    """
    
    D.behclass_preprocess_wrapper()

    datall = []
    list_dat = []
    for i in range(len(D.Dat)):

        dat = {}
        # trial-level info
        # trialcode = D.Dat.iloc[i]["trialcode"]   
        if version == "tokens":
            # use the tokens, e.g., for PIG
            tokens_beh = D.taskclass_tokens_extract_wrapper(i, which_order="beh")
            COLLECT_LOCATION = True
        elif version == "char_clust_labels":
            assert False, "this is obsolete, instead, shapes are already integrated into tokens, in general.preprocess"
            assert False, "otherwise this runs into problem becuase not chagin shapes eraly enoguh."
            # use labels extracted from clkustering strokes, usualyl for chars
            shape_seq = D.Dat.iloc[i]["charclust_shape_seq"] # list of str.
            # Fake it into list of dicts
            tokens_beh = [{"shape":sh} for sh in shape_seq]
            COLLECT_LOCATION = False
        else:
            print(version)
            assert False

        # NOTE: tokens_beh is not necesarily length of strokes_beh
        if False:
            if len(tokens_beh)!=len(D.Dat.iloc[i]["strokes_beh"]):
                # This occurs when datsegs throws out a beh stroke that doesnt match
                # any task stroke.
                print("weird: ", i)
                print(len(tokens_beh))
                print(len(D.Dat.iloc[i]["strokes_beh"]))
                print(tokens_beh)
                D.plotSingleTrial(i)
                assert False, "why?"

        nstrokes_beh = len(tokens_beh)
        dat["nstrokes_beh"] = nstrokes_beh
        dat["nstrokes_task"] = len(D.Dat.iloc[i]["strokes_task"])

        # shapes in order
        for j in range(n_strok_max):
            if j<nstrokes_beh:
                tok = tokens_beh[j]
                dat[f"{j}_shape"] = tok["shape"]
                if COLLECT_LOCATION:
                    dat[f"{j}_loc"] = tok["gridloc"]
                    dat[f"{j}_loc_local"] = tok["gridloc_local"]
            else:
                # Use same type as the actuals.
                dat[f"{j}_shape"] = "IGN"
                if COLLECT_LOCATION:
                    dat[f"{j}_loc"] = ("IGN", "IGN")
                    dat[f"{j}_loc_local"] = ("IGN", "IGN")
            if COLLECT_LOCATION:
                dat[f"{j}_loc_shape"] = (dat[f"{j}_loc"], dat[f"{j}_shape"]) # conjunction

        list_dat.append(dat)

    # Put back into D.Dat
    dfdat = pd.DataFrame(list_dat)
    D.Dat["seqc_nstrokes_beh"] = dfdat["nstrokes_beh"]
    D.Dat["seqc_nstrokes_task"] = dfdat["nstrokes_task"]
    for i in range(n_strok_max):
        D.Dat[f"seqc_{i}_shape"] = dfdat[f"{i}_shape"]
        if COLLECT_LOCATION:
            D.Dat[f"seqc_{i}_loc"] = dfdat[f"{i}_loc"]
            D.Dat[f"seqc_{i}_loc_local"] = dfdat[f"{i}_loc_local"]
            D.Dat[f"seqc_{i}_loc_shape"] = dfdat[f"{i}_loc_shape"]

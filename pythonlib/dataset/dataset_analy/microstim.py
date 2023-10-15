"""Analhysis of beh effects of microstim
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_overview_behcode_timings(D, sdir, STIM_DUR = 0.5):
    """ Wuick plot , for each stim trial, of 
    of timings of behe vents, strokes, and stim onsets.
    Based on beh codes extracted.
    Only plots for trials with detercted stim, based on beh codes.
    PARAMS:
    - STIM_DUR, num in sec, duiratin of stim, for use in plotting
    """  
    
    from pythonlib.tools.plottools import plot_beh_codes

    D.ml2_extract_raw()
    PRINT=False

    for ind in range(len(D.Dat)):

        # get times of stim
        # codes_keep = [141, 142, 151, 152] # 152 ignore. is TTL off
        codes_keep = [141, 142, 151] 
        codes, times = D.ml2_utils_getTrialsBehCodes(ind, codes_keep, PRINT=PRINT)
        
        ms_fix = D.blockparams_extract_single_taskparams(ind)["microstim_fix"]
        ms_stroke = D.blockparams_extract_single_taskparams(ind)["microstim_stroke"]
        
        if len(codes)>0:
                    
            assert ms_fix["on"]==1
            assert ms_stroke["on"]==1
            
            fig, ax = plt.subplots(figsize=(12,4))

            # Plot times of stim
            plot_beh_codes(codes, times, ax=ax, color="r", yval=2)
            # Plot offsets
            times_offset = [t+STIM_DUR for t in times]
            plot_beh_codes(codes, times_offset, ax=ax, color="m", yval=2)

            # get times of strokes and trial events
            codes, times = D.ml2_utils_getTrialsBehCodes(ind, keep_only_codes_standard=True, PRINT=PRINT)
            plot_beh_codes(codes, times, ax=ax, color="k", yval=0)

            # get times of strokes
            strokes = D.Dat.iloc[ind]["strokes_beh"]

            ons = [s[0,2] for s in strokes]
            offs = [s[-1,2] for s in strokes]

            plot_beh_codes(["on" for _ in range(len(ons))], ons, ax=ax, color="b", yval=1)
            plot_beh_codes(["off" for _ in range(len(offs))], offs, ax=ax, color="k", yval=1)

            ax.set_ylim([-3, 5])
            ax.set_ylabel("0:trial events; 1:strokes: 2: stim")
            fig.savefig(f"{sdir}/trial_{ind}.png")
        else:
            assert ms_fix["on"]==0
            assert ms_stroke["on"]==0
            
        plt.close("all")    

def preprocess_assign_stim_code(D, map_ttl_region):
    """
    Give each trial a string code for its stim params, which can vary across 
    expts.
    PARAMS:
    - map_ttl_region, dict mapping from int ttl values to string , usually
    breian regions, which will be the code.
    """

    # if HACK:
    #   map_ttl_region = {
    #       3:"M1",
    #       4:"pSMA"
    #   }
    # else:
    #   assert False, "code this input"

    list_stim_code = []
    for ind in range(len(D.Dat)):
        ms_fix = D.blockparams_extract_single_taskparams(ind)["microstim_fix"]
        ms_str = D.blockparams_extract_single_taskparams(ind)["microstim_stroke"]

        stim_code = []

        ttl_codes = [] #collect across all stims.

        # FIxation
        stim_code.append(ms_fix["on"]==1)
        if ms_fix["on"]==1:
            ttls = [int(x[0]) for x in ms_fix["stimlist"]]
            assert len(set(ttls))==1
            stim_code.append(ttls[0])
            ttl_codes.append(ttls[0])

        # Strokes
        stim_code.append(ms_str["on"]==1)
        if ms_str["on"]==1:
            ttls = [int(x[2]) for x in ms_str["stimlist"]]
            assert len(set(ttls))==1
            stim_code.append(ttls[0])
            ttl_codes.append(ttls[0])
            
        # print(ms_fix)
        # print(ms_str)
        # print(stim_code)

        # SHORTHANDS
        if all([x==False for x in stim_code]):
            # Stim off
            sc = "off"
        else:
            # Code it by its ttl
            # ttls = stim_code[1::2]
            assert len(set(ttl_codes))==1
            sc = map_ttl_region[ttl_codes[0]]

        list_stim_code.append(sc)


    # place stim code back into data
    D.Dat["microstim_epoch_code"] = list_stim_code

    print("New column: microstim_epoch_code")
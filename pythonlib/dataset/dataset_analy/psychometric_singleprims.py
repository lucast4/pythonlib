"""
5/20/24 - To deal with (usually single prim) expts with single sahpes, varying in graded fashion (therefore called
"psychometric"), doing the extraction of the psycho params and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pythonlib.tools.plottools import savefig
import os
import seaborn as sns

def params_has_intermediate_shape(animal, date):
    """ Within the ones that are smooth (no swithcing), which
    has intermediate shape?
    RETURNS:
    - list of  morphsets (ints) which have intermeiate shape. empy list means no intermed morphsets
    """
        
    date = int(date)

    if (animal, date) == ("Diego", 240515):
        morphset_good = []
    elif (animal, date) == ("Diego", 240517):
        morphset_good = [0, 4, 7]
    elif (animal, date) == ("Diego", 240521):
        morphset_good = [] 
    elif (animal, date) == ("Diego", 240523):
        morphset_good = [] # 
    elif (animal, date) == ("Diego", 240731):
        morphset_good = [2, 6] # 
    elif (animal, date) == ("Diego", 240801):
        morphset_good = [2] # 
    elif (animal, date) == ("Diego", 240802):
        morphset_good = [] # 

    elif (animal, date) == ("Pancho", 240516):
        morphset_good = [0, 3] # 
    elif (animal, date) == ("Pancho", 240521):
        morphset_good = [3] # 
    elif (animal, date) == ("Pancho", 240524):
        morphset_good = [] # 
    elif (animal, date) == ("Pancho", 240801):
        morphset_good = [0, 3, 4] # 
    elif (animal, date) == ("Pancho", 240802):
        morphset_good = [] # 
    else:
        print(animal, date)
        assert False    
    return morphset_good

def params_good_morphsets_no_switching(animal, date):
    """ 
    Those morphsets which have no clear switching between 
    different beh strategies -- smooth psychometric.
    Here incliudng all, so maybe some arent cleanest, e..g 
    some switching (rare).
    """

    date = int(date)

    if (animal, date) == ("Diego", 240515):
        morphset_good = [0, 1]
    elif (animal, date) == ("Diego", 240517):
        morphset_good = [0, 3, 4, 7]
    elif (animal, date) == ("Diego", 240521):
        morphset_good = [] # HAS NONE!
    elif (animal, date) == ("Diego", 240523):
        morphset_good = [6] # 
    elif (animal, date) == ("Diego", 240731):
        morphset_good = [0, 1, 2, 3, 4, 5, 6] # 
    elif (animal, date) == ("Diego", 240801):
        morphset_good = [0, 1, 2, 3, 4] # 
    elif (animal, date) == ("Diego", 240802):
        morphset_good = [0, 1, 2, 3, 4, 5, 6, 7] # 

    elif (animal, date) == ("Pancho", 240516):
        morphset_good = [0, 1, 3] # 
    elif (animal, date) == ("Pancho", 240521):
        morphset_good = [3, 7, 8] # 
    elif (animal, date) == ("Pancho", 240524):
        morphset_good = [2, 4, 5, 7] # 
    elif (animal, date) == ("Pancho", 240801):
        morphset_good = [0, 1, 2, 3, 4] # 
    elif (animal, date) == ("Pancho", 240802):
        morphset_good = [0, 1, 2, 3, 4, 5, 6, 7] # 
    else:
        print(animal, date)
        assert False    
    return morphset_good
        
def params_remap_angle_to_idx_within_morphset(animal, date):
    """
    Uyseful if there exist angle morphs to left and right of base prims.
    """
    date = int(date)
    if (animal, date) in [("Diego", 240731)]:
        # If have flank to left and right of base prims...  
        map_angleidx_to_finalidx  ={
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            99:99,
            6:100,
            7:-1}
        split_by_morphset = False
    elif (animal, date) in [("Diego", 240801)]:
        map_angleidx_to_finalidx = {}

        for morphset in [0,1,3,4]:
            _map  ={
                0:0,
                1:1,
                2:2,
                3:3,
                4:4,
                5:5,
                99:99,
                6:100,
                7:-1}
            map_angleidx_to_finalidx[morphset] = _map
        for morphset in [2]:
            _map  ={
                11:-1,
                0:0,
                1:1,
                2:2,
                3:3,
                4:4,
                10:5,
                5:6,
                6:7,
                7:8,
                8:9,
                99:99,
                9:100,
            }
            map_angleidx_to_finalidx[morphset] = _map

        split_by_morphset = True

    elif (animal, date) in [("Pancho", 240801)]:
        map_angleidx_to_finalidx = {}

        for morphset in [0,1,2,3]:
            _map  ={
                0:0,
                1:1,
                2:2,
                3:3,
                4:4,
                5:5,
                99:99,
                6:100,
                7:-1}
            map_angleidx_to_finalidx[morphset] = _map
        for morphset in [4]:
            _map  ={
                11:-1,
                0:0,
                1:1,
                2:2,
                3:3,
                4:4,
                10:5,
                5:6,
                6:7,
                7:8,
                8:9,
                99:99,
                9:100,
            }
            map_angleidx_to_finalidx[morphset] = _map

        split_by_morphset = True

    else:
        map_angleidx_to_finalidx = None
        split_by_morphset = None

    return map_angleidx_to_finalidx, split_by_morphset

def params_extract_psycho_groupings_manual_using_tsc_inds(animal, date):
    """
    Store params, manually entred, to be used with psychogood_preprocess_wrapper_using_tsc_inds.
    For defining morphsets for structured morph.
    Inoput the indices that were used in tsc params in dragmonkey.

    Assumes that morphsets are contiguous indices in TSC, and same n in each morphset.

    Approach:
    """

    date = int(date)

    if (animal, date) == ("Diego", 240802):
        
        # Params for morphs
        # - Method: Easy: just see the generative code in matlab
        morphs_tsc_n_sets = 8 # jhow many sets
        morphs_tsc_n_each_set = 7 # n in each set (excluring base prims).
        morphs_tsc_idx_start = 31 # first index in TSC
        morphs_tsc_idx_end = 86 # last index in TSC
        assert morphs_tsc_n_sets * morphs_tsc_n_each_set == morphs_tsc_idx_end - morphs_tsc_idx_start + 1

        # Map from index rank to the final index within. 
        # morphs_tsc_map_to_these_indices = [1, 2, 3, 4, 5] # This is standard
        morphs_tsc_map_to_these_indices = [-1, 1, 2, 3, 4, 5, 100] # if the flanking are nbefore and after the base prims (0, 99)

        # Params for base prims.
        # list is len num morphsets.
        # Within each morphset, Each is (base1, base2).
        # Sometimes multkple candidate LOS fore each base prim. If so, enter them as a tuple and will collect all and combine.
        # - Method: pull up saved images and manaulyl enter them here. To knwo what image sto lok for for each  morphset,
        # you can iterate by entering this e,mpy, running the preporcess_wrapper_using_tsc, and see it prints the morphsets,
        # look at those images (morphs) and use thgat to find these base prims.
        list_example_base_los = [
            [("singleprims_psycho", 25, (4, 5, 7, 25)), ("singleprims_psycho", 25, (3, 19, 20, 30))],
            [("singleprims_psycho", 25, (5, 4, 7, 25)) , ("singleprims_psycho", 25, (8, 6, 9, 22))],
            [("singleprims_psycho", 25, 10) , ("singleprims_psycho", 25, (2, 24, 12, 14, 23, 24, 26))],
            [("singleprims_psycho", 25, 16) , ("singleprims_psycho", 25, 20)],
            [("singleprims_psycho", 25, 28), ("singleprims_psycho", 25, 13)],
            [("singleprims_psycho", 25, 29), ("singleprims_psycho", 25, (2, 24, 12, 14, 23, 24, 26))],
            [("singleprims_psycho", 25, (11, 18)) , ("singleprims_psycho", 25, (2, 24, 12, 14, 23, 24, 26))],
            [("singleprims_psycho", 25, (11, 18)) , ("singleprims_psycho", 25, (8, 6, 9, 22))],
        ]   

        nlocations_expected = 2 # for each LOS. Used for saniyt checi that got all data.

        # Sanity check misses LOS -- these are allowed to fail. Usualyl these are orig base prims.
        los_allowed_to_exclude = [
            ('singleprims_psycho', 26, 3),
            ('singleprims_psycho', 26, 20),
            ('singleprims_psycho', 26, 4),
            ('singleprims_psycho', 26, 16),
            ('singleprims_psycho', 26, 9),
            ('singleprims_psycho', 26, 10),
            ('singleprims_psycho', 26, 8),
            ('singleprims_psycho', 26, 17),
            ('singleprims_psycho', 26, 5),
            ('singleprims_psycho', 26, 1),
            ('singleprims_psycho', 26, 19),
            ('singleprims_psycho', 26, 11),
            ('singleprims_psycho', 26, 18),
            ('singleprims_psycho', 26, 14),
            ('singleprims_psycho', 26, 7),
            ('singleprims_psycho', 26, 12),
            ('singleprims_psycho', 26, 6),
            ('singleprims_psycho', 26, 2),
            ('singleprims_psycho', 26, 13),
            ('singleprims_psycho', 26, 15),
        ]
    
    elif (animal, date) == ("Pancho", 240802):

        # Params for morphs
        # - Method: Easy: just see the generative code in matlab
        # - find the indices in tasksetclass_database.m
        morphs_tsc_n_sets = 8 # jhow many sets
        morphs_tsc_n_each_set = 7 # n in each set (excluring base prims).
        morphs_tsc_idx_start = 31 # first index in TSC
        morphs_tsc_idx_end = 86 # last index in TSC
        assert morphs_tsc_n_sets * morphs_tsc_n_each_set == morphs_tsc_idx_end - morphs_tsc_idx_start + 1

        # Map from index rank to the final index within. 
        # morphs_tsc_map_to_these_indices = [1, 2, 3, 4, 5] # This is standard
        morphs_tsc_map_to_these_indices = [-1, 1, 2, 3, 4, 5, 100] # if the flanking are nbefore and after the base prims (0, 99)

        # Params for base prims.
        # list is len num morphsets.
        # Within each morphset, Each is (base1, base2).
        # Sometimes multkple candidate LOS fore each base prim. If so, enter them as a tuple and will collect all and combine.
        # - Method: pull up saved images and manaulyl enter them here. To knwo what image sto lok for for each  morphset,
        # you can iterate by entering this e,mpy, running the preporcess_wrapper_using_tsc, and see it prints the morphsets,
        # look at those images (morphs) and use thgat to find these base prims.
        setind = 28
        list_example_base_los = [
            [("singleprims_psycho", setind, (6, 28)), ("singleprims_psycho", setind, 9)],
            [("singleprims_psycho", setind, 7), ("singleprims_psycho", setind, (11, 4, 17, 20))],
            [("singleprims_psycho", setind, (6, 28)), ("singleprims_psycho", setind, (10, 26, 29))],
            [("singleprims_psycho", setind, 8), ("singleprims_psycho", setind, (11, 4, 17, 20))],
            [("singleprims_psycho", setind, 5), ("singleprims_psycho", setind, (11, 4, 17, 20))],
            [("singleprims_psycho", setind, 3), ("singleprims_psycho", setind, (10, 26, 29))],
            [("singleprims_psycho", setind, (25, 32)), ("singleprims_psycho", setind, (10, 26, 29))],
            [("singleprims_psycho", setind, (25, 32)), ("singleprims_psycho", setind, (11, 4, 17, 20))],
        ]   

        nlocations_expected = 2 # for each LOS. Used for saniyt checi that got all data.

        # Sanity check misses LOS -- these are allowed to fail. Usualyl these are orig base prims.
        los_allowed_to_exclude = [
            ('singleprims_psycho', 29, 13),
            ('singleprims_psycho', 29, 15),
            ('singleprims_psycho', 29, 12),
            ('singleprims_psycho', 29, 3),
            ('singleprims_psycho', 29, 1),
            ('singleprims_psycho', 29, 18),
            ('singleprims_psycho', 29, 9),
            ('singleprims_psycho', 29, 11),
            ('singleprims_psycho', 29, 7),
            ('singleprims_psycho', 29, 10),
            ('singleprims_psycho', 29, 14),
            ('singleprims_psycho', 29, 5),
            ('singleprims_psycho', 29, 6),
            ('singleprims_psycho', 29, 17),
            ('singleprims_psycho', 29, 8),
            ('singleprims_psycho', 29, 4),
            ('singleprims_psycho', 29, 2),
            ('singleprims_psycho', 29, 16),
            ]

    else:
        print(animal, date)
        assert False
    
    return morphs_tsc_idx_start, morphs_tsc_idx_end, morphs_tsc_n_sets, morphs_tsc_n_each_set, morphs_tsc_map_to_these_indices, list_example_base_los, nlocations_expected, los_allowed_to_exclude



def params_extract_psycho_groupings_manual(animal, date):
    """
    Repo of params for each (animal ,date), for psychogood. 
    See code within for how to generate these params.
    
    """
    PARAMS = []
    date = int(date)

    if animal == "Diego" and date == 240523:
        if False:
            # 0.33 (not enought data)
            # PARAMS.append(
            # {'los_within': ('singleprims_psycho', 5, 25), 
            # 'los_base_1': ('singleprims_psycho', 6, 2), 
            # 'los_base_2': ('singleprims_psycho', 6, 16), 
            #  'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 0, 'tform_key': 'sx'}}
            # )

            PARAMS.append(
            {'los_within': ('singleprims_psycho', 5, 61), 
            'los_base_1': ('singleprims_psycho', 6, 2), 
            'los_base_2': ('singleprims_psycho', 6, 16), 
            'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 0, 'tform_key': 'sx'}}
            )

            # 0.66
            PARAMS.append(
            {'los_within': ('singleprims_psycho', 5, 67), 
            'los_base_1': ('singleprims_psycho', 6, 2), 
            'los_base_2': ('singleprims_psycho', 6, 16), 
            'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 0, 'tform_key': 'sx'}}
            )

            # 0.83
            PARAMS.append(
            {'los_within': ('singleprims_psycho', 5, 19), 
            'los_base_1': ('singleprims_psycho', 6, 2), 
            'los_base_2': ('singleprims_psycho', 6, 16), 
            'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 0, 'tform_key': 'sx'}}
            )
        else:
            # Combine all...
            PARAMS.append(
            {'los_within': ('singleprims_psycho', 5, 19), 
            'los_base_1': ('singleprims_psycho', 6, 2), 
            'los_base_2': ('singleprims_psycho', 6, 16), 
            'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': [0,2], 'tform_key': 'sx'}}
            )


        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 16), 
        #  'los_base_1': ('singleprims_psycho', 5, 19), 
        #  'los_base_2': ('singleprims_psycho', 5, 32), 
        'los_base_1': ('singleprims_psycho', 6, 2), 
        'los_base_2': ('singleprims_psycho', 6, 12), 
        'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 2, 'tform_key': 'sx'}
        }
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 18), 
        'los_base_1': ('singleprims_psycho', 6, 11),  
        'los_base_2': ('singleprims_psycho', 6, 5), 
        'psycho_params': {'psycho_ver': 'attachpt1_interp', 'idx_motif_char_flex': 0, 'idx_rel': 1, 'convert_extra_tforms_to_sign': True}
        }
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 30), 
        'los_base_1': ('singleprims_psycho', 6, 6), 
        'los_base_2': ('singleprims_psycho', 6, 17),  
        'psycho_params': {'psycho_ver': 'attachpt1_interp', 'idx_motif_char_flex': 0, 'idx_rel': 2, 'idx_rel_within': 1, 'convert_extra_tforms_to_sign': True}}
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 14), 
        'los_base_1': ('singleprims_psycho', 6, 3), 
        'los_base_2': ('singleprims_psycho', 6, 12),  
        'psycho_params': {'psycho_ver': 'attachpt1_interp', 'idx_motif_char_flex': 0, 'idx_rel': 1, 'idx_rel_within': 1, 'convert_extra_tforms_to_sign': False}}
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 44), 
        'los_base_1': ('singleprims_psycho', 6, 12), 
        'los_base_2': ('singleprims_psycho', 6, 1),  
        'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 1, 'tform_key': 'th'}
        }
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 33), 
        'los_base_1': ('singleprims_psycho', 6, 14), 
        'los_base_2': ('singleprims_psycho', 6, 15),  
        'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 1, 'tform_key': 'th'}
        }
        )


        los_allowed_to_miss = [('singleprims', 86, 1),
            ('singleprims', 86, 2),
            ('singleprims', 86, 3),
            ('singleprims', 86, 4),
            ('singleprims', 86, 5),
            ('singleprims', 86, 6),
            ('singleprims', 86, 7),
            ('singleprims', 86, 8),
            ('singleprims', 86, 9),
            ('singleprims', 86, 10),
            ('singleprims', 86, 11),
            ('singleprims', 86, 12),
            ('singleprims', 86, 13),
            ('singleprims', 86, 14),
            ('singleprims', 86, 15),
            ('singleprims', 86, 16),
            ('singleprims', 86, 17),
            ('singleprims', 86, 18),
            ('singleprims', 86, 19),
            ('singleprims', 86, 20),
            ('singleprims', 86, 21),
            ('singleprims', 86, 22),
            ('singleprims', 86, 23),
            ('singleprims', 86, 24),
            ('singleprims', 86, 25),
            ('singleprims', 86, 26),
            ('singleprims', 86, 27),
            ('singleprims', 86, 28),
            ('singleprims', 86, 29),
            ('singleprims', 86, 30),
            ('singleprims', 86, 31),
            ('singleprims', 86, 32),
            ('singleprims', 86, 33),
            ('singleprims', 86, 34),
            ('singleprims', 86, 35),
            ('singleprims', 86, 36),
            ('singleprims', 86, 37),
            ('singleprims', 86, 38),
            ('singleprims', 86, 39),
            ('singleprims', 86, 40),
            ('singleprims', 86, 41),
            ('singleprims', 86, 42),
            ('singleprims', 86, 43),
            ('singleprims', 86, 44),
            ('singleprims', 86, 45),
            ('singleprims', 86, 46),
            ('singleprims', 86, 47),
            ('singleprims', 86, 48),
            ('singleprims_psycho', 6, 13),
            ('singleprims_psycho', 6, 18),
        ]   
    elif animal == "Pancho" and date == 240524:
        #### GOOD
        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 8),
            "los_within":("singleprims_psycho", 8, 11),
            "los_base_2":("singleprims_psycho", 7, 10),
            "psycho_params":{
                "psycho_ver":"extra_tforms_each_prim",
                "idx_prim": 1,
                "tform_key": "sx"
            }
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 8),
            "los_within":("singleprims_psycho", 8, 15),
            "los_base_2":("singleprims_psycho", 7, 4),
            "psycho_params":{
                "psycho_ver":"extra_tforms_each_prim",
                "idx_prim": 0,
                "tform_key": "sx"}
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 18),
            "los_within":("singleprims_psycho", 8, 12),
            "los_base_2":("singleprims_psycho", 7, 15),
            "psycho_params":{
                "psycho_ver":"extra_tforms_each_prim",
                "idx_prim": 0,
                "tform_key": "sx"
            }    
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 10),
            "los_within":("singleprims_psycho", 8, 6),
            "los_base_2":("singleprims_psycho", 7, 17),
            "psycho_params":{
                "psycho_ver":"attachpt1_interp",
                "idx_motif_char_flex": 0,
                "idx_rel": 1,
            }
        })


        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 14),
            "los_within":("singleprims_psycho", 8, 7),
            "los_base_2":("singleprims_psycho", 7, 7),
            "psycho_params":{
                "psycho_ver":"attachpt1_interp",
                "idx_motif_char_flex": 0,
                "idx_rel": 1,
            }    
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 20),
            "los_within":("singleprims_psycho", 8, 13),
            "los_base_2":("singleprims_psycho", 7, 22),
            # Here two things are varying: interp and extra tforms. Solve this problem by convering tform to its sign.
            "psycho_params":{
                "psycho_ver":"attachpt1_interp",
                "idx_motif_char_flex": 0, # Usuall dont change (if this is one "prim"), i.e,. this is the outer index.
                "idx_rel": 1, # This is the index into the relations, within each plan
                "exclude_extra_tform_values":False,
                "exclude_interp_values":False,
                "convert_extra_tforms_to_sign":True 
            }    
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 5),
            "los_within":("singleprims_psycho", 8, 16),
            "los_base_2":("singleprims_psycho", 7, 9),
            "psycho_params":{
                "psycho_ver":"attachpt1_interp",
                "idx_motif_char_flex": 0, # Usuall dont change (if this is one "prim"), i.e,. this is the outer index.
                "idx_rel": 1, # This is the index into the relations, within each plan
                "exclude_extra_tform_values":False,
                "exclude_interp_values":False,
                "convert_extra_tforms_to_sign":True 
            }    
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 3),
            "los_within":("singleprims_psycho", 8, 35),
            "los_base_2":("singleprims_psycho", 7, 17),
            "psycho_params":{
                "psycho_ver":"extra_tforms_each_prim",
                "idx_prim": [0,1],
                "tform_key": "th"
            }    
        })


        # Arc --> For each one find both locations.
        PARAMS.append({
            "los_base_1":("singleprims", 114, 9), # arc
            "los_within":("singleprims_psycho", 9, 1),
            "los_base_2":("singleprims", 114, 30), # circle
            "psycho_params":{
                "psycho_ver":"morph_frac",
                "replace_morph_params_with_binary_whether_exists":True,
            }
        })

        los_allowed_to_miss = [('singleprims', 114, 1),
            ('singleprims', 114, 2),
            ('singleprims', 114, 3),
            ('singleprims', 114, 4),
            ('singleprims', 114, 5),
            ('singleprims', 114, 6),
            ('singleprims', 114, 7),
            ('singleprims', 114, 8),
            ('singleprims', 114, 10),
            ('singleprims', 114, 11),
            ('singleprims', 114, 12),
            ('singleprims', 114, 13),
            ('singleprims', 114, 14),
            ('singleprims', 114, 15),
            ('singleprims', 114, 16),
            ('singleprims', 114, 17),
            ('singleprims', 114, 19),
            ('singleprims', 114, 20),
            ('singleprims', 114, 21),
            ('singleprims', 114, 22),
            ('singleprims', 114, 23),
            ('singleprims', 114, 24),
            ('singleprims', 114, 25),
            ('singleprims', 114, 26),
            ('singleprims', 114, 28),
            ('singleprims', 114, 29),
            ('singleprims', 114, 31),
            ('singleprims', 114, 32),
            ('singleprims', 114, 33),
            ('singleprims', 114, 34),
            ('singleprims_psycho', 7, 2),
            ('singleprims_psycho', 7, 6),
            ('singleprims_psycho', 7, 12),
            ('singleprims_psycho', 7, 21)]

    elif animal == "Diego" and date == 240730:
        # GOOD

        PARAMS.append(
            {'los_base_1': ('singleprims_psycho', 15, 2), 
             'los_base_2': ('singleprims_psycho', 15, 10), 
             'los_within': ('singleprims_psycho', 14, 26), 
             'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': [0, 1], 'tform_key': 'th'}}
        )

        PARAMS.append(
            {'los_base_1': ('singleprims_psycho', 15, 15), 
             'los_base_2': ('singleprims_psycho', 15, 16), 
             'los_within': ('singleprims_psycho', 14, 44), 
             'psycho_params': {'psycho_ver': 'attachpt1_interp', 'idx_motif_char_flex': 0, 'idx_rel': 1, 'convert_extra_tforms_to_sign': True}}
        )
        PARAMS.append(
            {'los_base_1': ('singleprims_psycho', 15, 15), 
             'los_base_2': ('singleprims_psycho', 15, 19), 
             'los_within': ('singleprims_psycho', 14, 31), 
             'psycho_params': {'psycho_ver': 'attachpt1_interp', 'idx_motif_char_flex': 0, 'idx_rel': 1, 'convert_extra_tforms_to_sign': True}}
        )
        PARAMS.append(
            {'los_base_1': ('singleprims_psycho', 15, 6), 
             'los_base_2': ('singleprims_psycho', 15, 2), 
             'los_within': ('singleprims_psycho', 14, 41), 
             'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': [0, 1, 2], 'tform_key': 'th'}}
        )
        PARAMS.append(
            {'los_base_1': ('singleprims_psycho', 15, 5), 
             'los_base_2': ('singleprims_psycho', 15, 3), 
             'los_within': ('singleprims_psycho', 14, 18), 
             'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': [0, 1, 2], 'tform_key': 'th'}}
        )

        # # Arc --> For each one find both locations.
        PARAMS.append(
            {'los_base_1': ('singleprims', 86, 11), 
             'los_base_2': ('singleprims', 86, 26), 
             'los_within': ('singleprims_psycho', 13, 1), 'psycho_params': {'psycho_ver': 'morph_frac', 'replace_morph_params_with_binary_whether_exists': True}
             })

        los_allowed_to_miss = [
            ('singleprims', 86, 1),
            ('singleprims', 86, 2),
            ('singleprims', 86, 3),
            ('singleprims', 86, 4),
            ('singleprims', 86, 5),
            ('singleprims', 86, 6),
            ('singleprims', 86, 7),
            ('singleprims', 86, 8),
            ('singleprims', 86, 9),
            ('singleprims', 86, 10),
            ('singleprims', 86, 13),
            ('singleprims', 86, 14),
            ('singleprims', 86, 15),
            ('singleprims', 86, 16),
            ('singleprims', 86, 17),
            ('singleprims', 86, 18),
            ('singleprims', 86, 20),
            ('singleprims', 86, 21),
            ('singleprims', 86, 22),
            ('singleprims', 86, 23),
            ('singleprims', 86, 24),
            ('singleprims', 86, 25),
            ('singleprims', 86, 27),
            ('singleprims', 86, 28),
            ('singleprims', 86, 29),
            ('singleprims', 86, 30),
            ('singleprims', 86, 31),
            ('singleprims', 86, 32),
            ('singleprims', 86, 33),
            ('singleprims', 86, 34),
            ('singleprims', 86, 35),
            ('singleprims', 86, 36),
            ('singleprims', 86, 37),
            ('singleprims', 86, 38),
            ('singleprims', 86, 39),
            ('singleprims', 86, 40),
            ('singleprims', 86, 41),
            ('singleprims', 86, 42),
            ('singleprims', 86, 43),
            ('singleprims', 86, 44),
            ('singleprims', 86, 45),
            ('singleprims', 86, 46),
            ('singleprims', 86, 47),
            ('singleprims', 86, 48),
        ]   
    else:
        assert False

    return PARAMS, los_allowed_to_miss

def preprocess_and_plot(D, var_psycho, PLOT=True):
    """
    For angle morphs.

    Wrapper to do preprocess and plots, two kinds, either with strong restriction on strokes (only
    clean like singleprims) or lenient, including all even abort (useful to see vacillation).
    """

    if var_psycho == "angle":
        # OLD, should place into below, general purpose methods.
        SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("psycho_singleprims")
        SAVEDIR = f"{SAVEDIR}/{var_psycho}"

        DS, DSlenient, map_shape_to_psychoparams = preprocess(D, var_psycho)

        if PLOT:
            # Plot both strict and lenient
            plot_overview(DS, D, f"{SAVEDIR}/clean_strokes_singleprim", var_psycho=var_psycho)
            plot_overview(DSlenient, D, f"{SAVEDIR}/lenient_strokes", var_psycho=var_psycho)

        #### ADDITIOANL PLOTS, USING DSMORPHSETS
        animal = D.animals(True)[0]
        date = D.dates(True)[0]
        map_angleidx_to_finalidx, split_by_morphset = params_remap_angle_to_idx_within_morphset(animal, date)
        DSmorphsets = preprocess_angle_to_morphsets(DSlenient, map_angleidx_to_finalidx, split_by_morphset)
        # for morphset, idx_within in 

        psychogood_plot_morphset_wrapper(D, DS, DSmorphsets, SAVEDIR)

        if False:
            savedir = f"{SAVEDIR}/drawings_morphsets"
            os.makedirs(savedir, exist_ok=True)
            psychogood_plot_morphset_drawings(D, DSmorphsets, savedir, PLOT_EACH_TRIAL=True)        
        
        #####################
        # Also make plot of mean_sim_score (trial by trial var)
        savedir = f"{SAVEDIR}/using_primitivenessv2"
        os.makedirs(savedir, exist_ok=True)

        if False: # This is done in DSmorphset stuff now.
            from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline
            # First, extract all the derived metrics
            PLOT=True
            plot_methods = ("tls", "drw")
            DSnew, _, dfres, grouping = preprocess_plot_pipeline(D, PLOT=PLOT, plot_methods=plot_methods)
            _apply_psychoparams_to_ds(DSnew.Dat, map_shape_to_psychoparams, var_psycho)
            _apply_psychoparams_to_ds(dfres, map_shape_to_psychoparams, var_psycho)

            # Make plots
            fig = sns.catplot(data=dfres, x="angle_idx_within_shapeorig", y="mean_sim_score", col="shapeorig", 
                kind="point", sharey=True)
            savefig(fig, f"{savedir}/mean_sim_score-1.pdf")
            fig = sns.catplot(data=dfres, x="angle_idx_within_shapeorig", y="mean_sim_score", col="shapeorig", 
                alpha=0.5, sharey=True)
            savefig(fig, f"{savedir}/mean_sim_score-2.pdf")

            plt.close("all")

        return DS, DSlenient, map_shape_to_psychoparams
    
    elif var_psycho == "cont_morph":
        DS, DSmorphsets, map_morph_set_idx_to_shapes, _, _, SAVEDIR = preprocess_cont_morph(D)
        plot_overview_cont_morph(D, DS, DSmorphsets, map_morph_set_idx_to_shapes, SAVEDIR)

        return DS, DSmorphsets, map_morph_set_idx_to_shapes
    
    else:
        print(var_psycho)
        assert False


def _apply_psychoparams_to_ds(df, map_shape_to_psychoparams, var_psycho):
    """

    """
    var_psycho_unique = f"{var_psycho}_unique"
    var_psycho_str = f"{var_psycho}_str"
    var_psycho_idx = f"{var_psycho}_idx_within_shapeorig"

    names = ["shapeorig_psycho", "shapeorig", var_psycho_unique, var_psycho_str, var_psycho_idx]
    for i, na in enumerate(names):
        # D.Dat[f"seqc_0_{na}"] = [map_shape_to_psychoparams[sh][i] for sh in D.Dat["seqc_0_shape"]]
        df[na] = [map_shape_to_psychoparams[sh][i] for sh in df["shape"]]
        # DSlenient.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DSlenient.Dat["shape"]]

def preprocess(D, var_psycho="angle", SANITY=True, clean_ver="singleprim_psycho",
               ):
    """
    For angle morph.
    Determines for each stroke what its original shape is, and 
    its angle relative to that shape. 
    """
    from pythonlib.tools.pandastools import find_unique_values_with_indices
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import append_col_with_index_of_level, append_col_with_index_of_level_after_grouping
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.checktools import check_objects_identical
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    from math import pi

    assert var_psycho == "angle", "coded only for this currently."

    # First, for each token, extract variables that reflect the psycho variable/param.
    token_ver = "task"
    for i, row in D.Dat.iterrows():
        # for each shape get its concrete params
        Tk = D.taskclass_tokens_extract_wrapper(i, token_ver, return_as_tokensclass=True)
        Tk.features_extract_wrapper(["loc_on", "angle"], angle_twind=[0, 2])
    df = D.tokens_extract_variables_as_dataframe(["shape", "loc_on", "angle", "Prim", "gridloc"], token_ver)
    
    # Extract the original shape, which should have been overwritten in rprepovessing, but is useufl as a category
    # to anchor the variations.
    list_shape_orig = []
    for P in df["Prim"]:
        list_shape_orig.append(P.shape_oriented())
    df["shape_orig"] = list_shape_orig

    # Mark which shapes are base shpaes (i.e. no rotation tform)
    base_shapes = []
    for ind in range(len(D.Dat)):
        # tforms = D.taskclass_extract_prims_extra_params_tforms(ind)
        # assert isinstance(tforms, list)
        # assert isinstance(tforms[0], dict)
        # tform_first_stroke = tforms[0]
        tform_first_stroke = _extract_extra_tform_params_helper(D, ind)
        if tform_first_stroke is None or len(tform_first_stroke)==0:
            # Then this is base shape
            base_shapes.append(D.taskclass_shapes_extract(ind)[0])
    base_shapes = list(set(base_shapes))
    df["is_base_shape"] = df["shape"].isin(base_shapes)

    # For some days, there are three base shapes in a sequence [base1 morph base2 morph base3]. Deal with this by renaming base2 to
    # morph, with appropriate angle. This is hacky, but ok.
    if (D.animals(True)[0] in ["Diego", "Pancho"]) and int(D.dates(True)[0])==240801:
        ###########
        list_shape_orig = []
        list_angle = []
        list_base = []
        for i, row in df.iterrows():
            if (row["shape_orig"] == "line-8-1-0") and (row["angle"]==0.0):
                list_shape_orig.append("line-8-4-0")
                list_angle.append(pi/4)
                list_base.append(False)
            else:
                list_shape_orig.append(row["shape_orig"])
                list_angle.append(row["angle"])
                list_base.append(row["is_base_shape"])

        df["shape_orig"] = list_shape_orig
        df["angle"] = list_angle
        df["is_base_shape"] = list_base

    # For each shape_orig, get an ordered indices for angle. 
    if True:
        ################### FIND THE ANGLE RELATIVE TO THE BASE_PRIM'S ANGLE. THis will be used for defining the psycho index
        # Get list of all shapes which dont have extra tform
        # base_shapes = []
        # for ind in range(len(D.Dat)):
        #     # tforms = D.taskclass_extract_prims_extra_params_tforms(ind)
        #     # assert isinstance(tforms, list)
        #     # assert isinstance(tforms[0], dict)
        #     # tform_first_stroke = tforms[0]
        #     tform_first_stroke = _extract_extra_tform_params_helper(D, ind)
        #     if tform_first_stroke is None or len(tform_first_stroke)==0:
        #         # Then this is base shape
        #         base_shapes.append(D.taskclass_shapes_extract(ind)[0])
        # base_shapes = list(set(base_shapes))

        # df["is_base_shape"] = df["shape"].isin(base_shapes)

        # For each shape_orig, find its one base shape
        from pythonlib.tools.pandastools import find_unique_values_with_indices

        map_shape_orig_to_angle_base = {}
        for grp in df.groupby(["shape_orig"]):
            shape_orig = grp[0][0]
            dfthis = df[(df["shape_orig"] == shape_orig) & (df["is_base_shape"]==True)]
            dfthis = grp[1][(grp[1]["is_base_shape"]==True)]

            unique_values, _, _ = find_unique_values_with_indices(dfthis, "angle")
            # print(unique_values)
            if not len(unique_values)==1:
                print(unique_values)
                print(len(dfthis))
                assert False, "This means there are multiple trials without tform, which are thus called base prims, for this shape_orig"

            angle_base = unique_values[0]

            map_shape_orig_to_angle_base[shape_orig] = angle_base

        # Map the base angle to each shape_orig
        df["angle_base"] = df["shape_orig"].map(map_shape_orig_to_angle_base)

        # Calculate the relative angle
        df["angle_rel_base"] = df["angle"] - df["angle_base"]

        # Adjust angles that are negative
        df.loc[df["angle_rel_base"] < 0, "angle_rel_base"] += 2 * pi
        df.loc[df["angle_rel_base"] > 0.99*2*pi, "angle_rel_base"] = 0 # anything 6.28 should actually be 0.

        # HACK - change name to angle
        HACK = True
        if HACK:
            df["angle"] = df["angle_rel_base"]
            var_psycho = f"angle"
        else:
            var_psycho = f"{var_psycho}_rel_base"


    # col = var_psycho
    var_psycho_unique = f"{var_psycho}_unique"
    var_psycho_str = f"{var_psycho}_str"
    var_psycho_idx = f"{var_psycho}_idx_within_shapeorig"
    unique_values, indices, map_index_to_value = find_unique_values_with_indices(df, var_psycho, 
        append_column_with_unique_values_colname=var_psycho_unique)
    df[var_psycho_str] = [f"{a:.2f}" for a in df[f"{var_psycho}_unique"]]
    # df[var_psycho_str] = [f"{a:.2f}" for a in df[f"{var_psycho}"]]
    # print(var_psycho)
    # assert False
    
    ##########################
    # if SANITY: STop doing this -- now is using clusters, not hash.
    #     df["shape_hash"] = [P.label_classify_prim_using_stroke(return_as_string=True, version="hash") for P in df["Prim"]]
    #     assert np.all(df["shape_hash"] == df["shape"]), "for psycho, you need to turn on reclassify_shape_using_stroke_version=hash in preprocess/general"

    # Redefine each token to have a unique identifier (shape-psycho)
    df = append_col_with_grp_index(df, ["shape_orig", var_psycho_str], "shape_pscho", True)

    # Convert psycho var levels in to indices that start at 0 for each shape-orig.
    df = append_col_with_index_of_level_after_grouping(df, ["shape_orig"], var_psycho_str, var_psycho_idx)

    # Finally, get map from shape to psycho stuff
    map_shape_to_psychoparams = {}
    for i, row in df.iterrows():
        shape = row["shape"]
        params = (row["shape_pscho"], row["shape_orig"], row[var_psycho_unique], row[var_psycho_str], row[var_psycho_idx])
        if shape in map_shape_to_psychoparams:
            if not check_objects_identical(map_shape_to_psychoparams[shape], params):
                print(shape)
                print(map_shape_to_psychoparams[shape])
                print(params)
                assert False
        else:
            map_shape_to_psychoparams[shape] = params

    # if SANITY:
    #     grouping_print_n_samples(df, ["shape_pscho", "shape_hash", var_psycho_idx, "gridloc"])
    #     df.groupby(["shape_pscho", "shape_hash", "shape_orig", "shape", var_psycho_str]).size().reset_index()
    
    if False: # Not needed
        D.tokens_assign_dataframe_back_to_self_mult(df, tk_ver=token_ver)

    ############# EXTRACT DS
    DSlenient = preprocess_dataset_to_datstrokes(D, clean_ver)
    DS = preprocess_dataset_to_datstrokes(D, "singleprim")

    # Assign columns to DS and D
    names = ["shapeorig_psycho", "shapeorig", var_psycho_unique, var_psycho_str, var_psycho_idx]
    for i, na in enumerate(names):
        D.Dat[f"seqc_0_{na}"] = [map_shape_to_psychoparams[sh][i] for sh in D.Dat["seqc_0_shape"]]
        # DS.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DS.Dat["shape"]]
        # DSlenient.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DSlenient.Dat["shape"]]
    _apply_psychoparams_to_ds(DS.Dat, map_shape_to_psychoparams, var_psycho)
    _apply_psychoparams_to_ds(DSlenient.Dat, map_shape_to_psychoparams, var_psycho)

    return DS, DSlenient, map_shape_to_psychoparams

def preprocess_angle_to_morphsets(DS, map_angleidx_to_finalidx, split_by_morphset):
    """
    Convert DS, which has one datapt for stroke, to DSmorphset, which has indepednent data for each morphset, which is
    (baseprim1, [morph prims], baseprim2).

    DSmorphsets is the base object I want to use for all code.

    Takes output of preprocess().
    """

    # Hand coded map from baseprim to otherbasepimr, assuming rotation that is CCW, as is all the convention here.
    # To add to this, just look at the plot of (idx_within vs. baseprim) or task images.
    map_shapeorig_to_otherbaseprim = {
        'zigzagSq-1-2-0':'zigzagSq-1-1-0',
        'Lcentered-4-3-0':'Lcentered-4-4-0',
        'zigzagSq-1-1-1':'zigzagSq-1-2-1',
        'zigzagSq-1-2-1':'zigzagSq-1-1-1',
        'line-8-1-0':'line-8-2-0',
        'line-8-4-0':'line-8-3-0',
        'arcdeep-4-2-0':'arcdeep-4-3-0',
        'arcdeep-4-3-0':'arcdeep-4-4-0',
        'Lcentered-4-2-0':'Lcentered-4-3-0',
        'V-2-3-0':'V-2-4-0',
        'zigzagSq-1-1-0':'zigzagSq-1-2-0',
        'Lcentered-4-1-0':'Lcentered-4-2-0',
        'Lcentered-4-4-0':'Lcentered-4-1-0',
        'line-8-2-0':'line-8-1-0',
        'V-2-2-0':'V-2-3-0',
        'squiggle3-3-2-1':'squiggle3-3-1-1',
        'squiggle3-3-2-0':'squiggle3-3-1-0',
        'squiggle3-3-1-0':'squiggle3-3-2-0',
        'squiggle3-3-1-1':'squiggle3-3-2-1',
        'arcdeep-4-1-0':'arcdeep-4-2-0',
        'V-2-4-0':'V-2-1-0',
        'line-8-3-0':'line-8-2-0',
        'V-2-1-0':'V-2-2-0',
        'arcdeep-4-4-0':'arcdeep-4-1-0',
        'usquare-1-3-0':'usquare-1-4-0',
        }


    # Go thru each baseprim, find its other base prim and morph angles, and call that a single morphset        
    list_baseprim = DS.Dat[DS.Dat["angle_idx_within_shapeorig"] == 0]["shapeorig"].unique().tolist()
    morphset = 0
    list_df = []
    for baseprim in list_baseprim:
        list_angles = DS.Dat[DS.Dat["shapeorig"]==baseprim]["angle_unique"].values
        if any(list_angles!=0):
            # Then this baseprim has rotation.

            if baseprim not in map_shapeorig_to_otherbaseprim:
                print(baseprim)
                assert False, "Not big deal, just add this baseprim, mapping to its other prim, by inspection of plots shapeorig-vs-idx-1-task-iter0 (see above)."
        
            baseprim_other = map_shapeorig_to_otherbaseprim[baseprim]

            # # Find the most extreme angle.
            # in progress - auto matching largest rotation to the closest other base prim, by motor distances.
            # angle_max = list_angles[np.argmax(np.abs(list_angles))] # 0 is baseprim

            if False:
                print(baseprim, list_angles, baseprim_other)

            # Extract data
            # - base 1
            df_base1 = DS.Dat[(DS.Dat["shapeorig"]==baseprim) & (DS.Dat["angle_idx_within_shapeorig"]==0)].copy()
            df_base1["morph_set_idx"] = morphset
            df_base1["morph_idxcode_within_set"] = 0
            df_base1["morph_is_morphed"] = False

            # - base 2
            df_base2 = DS.Dat[(DS.Dat["shapeorig"]==baseprim_other) & (DS.Dat["angle_idx_within_shapeorig"]==0)].copy()
            df_base2["morph_set_idx"] = morphset
            df_base2["morph_idxcode_within_set"] = 99
            df_base2["morph_is_morphed"] = False

            # - morphs
            df_morph = DS.Dat[(DS.Dat["shapeorig"]==baseprim) & (DS.Dat["angle_idx_within_shapeorig"]>0)].copy()
            df_morph["morph_set_idx"] = morphset
            df_morph["morph_idxcode_within_set"] = df_morph["angle_idx_within_shapeorig"]
            df_morph["morph_is_morphed"] = True

            try:
                assert len(df_base1)>0
                assert len(df_base2)>0
                assert len(df_morph)>0
            except Exception as err:
                print(len(df_base1), len(df_base2), len(df_morph))
                assert False, "bad task design this day? Suolution - skip this morphs et"

            # Count
            morphset+=1

            list_df.extend([df_morph, df_base1, df_base2])

    DF = pd.concat(list_df).reset_index(drop=True)
    DSmorphsets = DS.copy()
    DSmorphsets.Dat = DF

    # Add variables to match names across methods.
    DSmorphsets.Dat["psycho_value"] = DSmorphsets.Dat["angle_unique"]

    # Hacky, remap indices
    if map_angleidx_to_finalidx is not None:
        if split_by_morphset:
            # each morpjhset has different maping
            tmp = []
            for _, row in DSmorphsets.Dat.iterrows():
                morphset = row["morph_set_idx"]
                idx_within = row["morph_idxcode_within_set"]
                idx_within_new = map_angleidx_to_finalidx[morphset][idx_within]
                tmp.append(idx_within_new)
            DSmorphsets.Dat["morph_idxcode_within_set"] = tmp
        else:
            # print(map_angleidx_to_finalidx)
            # assert False
            DSmorphsets.Dat["morph_idxcode_within_set"] = [map_angleidx_to_finalidx[idx] for idx in DSmorphsets.Dat["morph_idxcode_within_set"]]

    return DSmorphsets


def plot_overview(DS, D, SAVEDIR, var_psycho="angle"):
    """
    For angle morph.
    Make all plots.
    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import grouping_print_n_samples

    assert var_psycho == "angle", "code it for others, need to change variable strings below"

    # Only include first stroke in these plots
    DS = DS.copy()
    DS.Dat = DS.Dat[DS.Dat["stroke_index"] == 0].reset_index(drop=True)
    DS.distgood_compute_beh_task_strok_distances()

    ### PLOT DRAWINGS
    savedir = f"{SAVEDIR}/drawings"
    os.makedirs(savedir, exist_ok=True)

    niter = 1
    for _iter in range(niter):

        print("Plotting drawings...")
        if False: # not useful, the other plots are better
            figholder = DS.plotshape_multshapes_egstrokes("shapeorig_psycho", 6, ver_behtask="beh");
            for i, (fig, axes) in enumerate(figholder):
                savefig(fig, f"{savedir}/egstrokes-{i}-iter{_iter}.pdf")

            figholder = DS.plotshape_multshapes_egstrokes("shapeorig_psycho", 6, ver_behtask="task_aligned_single_strok");
            for i, (fig, axes) in enumerate(figholder):
                savefig(fig, f"{savedir}/egstrokes-{i}-task-iter{_iter}.pdf")
            plt.close("all")

        figbeh, figtask = DS.plotshape_row_col_vs_othervar("angle_idx_within_shapeorig", "shapeorig", 
                                                           n_examples_per_sublot=8, plot_task=True);
        savefig(figbeh, f"{savedir}/shapeorig-vs-idx--beh-iter{_iter}.pdf")
        savefig(figtask, f"{savedir}/shapeorig-vs-idx--task-iter{_iter}.pdf")
        plt.close("all")

    ### PLOT COUNTS
    print("Plotting counts...")
    savedir = f"{SAVEDIR}/counts"
    os.makedirs(savedir, exist_ok=True)

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_idx_within_shapeorig", "shapeorig", ["gridloc"])
    savefig(fig, f"{savedir}/counts-idx-vs-shapeorig.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_str", "shapeorig", ["gridloc"])
    savefig(fig, f"{savedir}/counts-str-vs-shapeorig.pdf")
    
    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_binned", "angle_idx_within_shapeorig", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-angle_binned-vs-idx.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "loc_on_clust", "angle_idx_within_shapeorig", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-loc_on_clust-vs-idx.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_idx_within_shapeorig", "shapeorig_psycho", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-idx-vs-shapeorig_psycho.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "shape", "shapeorig_psycho", ["gridloc"])
    savefig(fig, f"{savedir}/counts-shape-vs-shapeorig_psycho.pdf")

    savepath = f"{savedir}/groupings.txt"
    grouping_print_n_samples(DS.Dat, 
        ["shapeorig_psycho", "shapeorig", "angle_idx_within_shapeorig", "angle_str", "shape", "character"], 
        savepath=savepath)
    plt.close("all")

    #### ANALYSES (e..,g, timing and velocity)
    savedir = f"{SAVEDIR}/analyses"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")

    ## First stroke
    dfthis = DS.Dat
    dfthis = dfthis[dfthis["gap_from_prev_dur"]<5].reset_index(drop=True)

    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="angle", hue="gridloc", col="shapeorig", alpha=0.5)
    savefig(fig, f"{savedir}/angle-vs-idx-1.pdf")
    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="angle", hue="gridloc", col="shapeorig", kind="point")
    savefig(fig, f"{savedir}/angle-vs-idx-2.pdf")

    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="loc_on_clust", hue="gridloc", col="shapeorig", jitter=True, alpha=0.5)
    savefig(fig, f"{savedir}/loc_on_clust-vs-idx-1.pdf")
    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="loc_on_clust", hue="gridloc", col="shapeorig", jitter=True, alpha=0.5)
    savefig(fig, f"{savedir}/loc_on_clust-vs-idx-1.pdf")
    plt.close("all")

    # Slower for psycho?
    for sharey in [True, False]:
        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="velocity", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/velocity-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="velocity", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/velocity-vs-idx-sharey={sharey}-2.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="gap_from_prev_dur", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/gap_from_prev_dur-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="gap_from_prev_dur", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/gap_from_prev_dur-vs-idx-sharey={sharey}-2.pdf")
        plt.close("all")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="dist_beh_task_strok", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/dist_beh_task_strok-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="dist_beh_task_strok", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/dist_beh_task_strok-vs-idx-sharey={sharey}-2.pdf")
        plt.close("all")

    ## Dataset
    savedir = f"{SAVEDIR}/analyses_dataset_lenient_strokes"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")

    D = D.copy()
    # D.extract_beh_features(["num_strokes_beh"])
    # frac_touched_min = 0.6
    # ft_decim_min = 0.3
    # shortness_min = 0.2
    # D.preprocessGood(params=["beh_strokes_at_least_one",
    #                             "no_supervision",
    #                             "remove_online_abort"],
    #                     frac_touched_min=frac_touched_min,
    #                     ft_decim_min=ft_decim_min,
    #                     shortness_min = shortness_min)
    D.extract_beh_features(["num_strokes_beh"])
    D.preprocessGood(params=["beh_strokes_at_least_one",
                                "no_supervision"])

    # Number of strokes used
    dfthis = D.Dat

    fig = sns.catplot(data=dfthis, x="seqc_0_angle_idx_within_shapeorig", y="FEAT_num_strokes_beh", col="seqc_0_shapeorig", 
        alpha=0.4, jitter=True)
    savefig(fig, f"{savedir}/FEAT_num_strokes_beh-vs-idx-1.pdf")

    fig = sns.catplot(data=dfthis, x="seqc_0_angle_idx_within_shapeorig", y="FEAT_num_strokes_beh", col="seqc_0_shapeorig", kind="box")
    savefig(fig, f"{savedir}/FEAT_num_strokes_beh-vs-idx-2.pdf")
    plt.close("all")

def preprocess_cont_morph(D, clean_ver="singleprim_psycho"):
    """
    ENtire preprocess pipeline for continuous morph (i.e., 2 base shapes, and a single, or multiple, shapes that interpolate bewteen them)

    """
    from pythonlib.tools.pandastools import find_unique_values_with_indices
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import append_col_with_index_of_level, append_col_with_index_of_level_after_grouping
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.checktools import check_objects_identical
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    from math import pi

    # Check that it is one stroke.
    assert all(D.Dat["task_kind"] == "prims_single"), "assuming this to make things eaiser..."
    
    _, map_shape_orig_to_shape = D.shapesemantic_taskclass_map_between_shape_and_shape_orig()

    ############################################
    # Map from shape to its base shapes
    map_shape_to_base_prims = {}
    # map_shape_to_base_LOSs = {}
    for ind in range(len(D.Dat)):

        # get the shape name 
        shape = D.taskclass_shapes_extract(ind)[0]
        morph_info = D.shapesemantic_taskclass_cont_morph_extract_params(ind)

        if morph_info is None:
            base_shapes = None
            # base_LOSs = None
        else:
            base_shapes = [mi["shape"] for mi in morph_info] # list of two 3-tuples
            # base_LOSs = [mi["los"] for mi in morph_info] 

            for sh in base_shapes:
                if sh == shape:
                    print(ind)
                    print(morph_info)
                    print(base_shapes)
                    print(shape)
                    assert False, "a shape cannot be its own morph shape... doesnt make sense."

        # Base shapes -- convert these to whatever they are called now,w hich might be diff, beaause of redefine
        # name by stroke hash
        if base_shapes is not None:
            base_shapes = ["-".join([str(x) for x in shtuple]) for shtuple in base_shapes]
            print(map_shape_orig_to_shape.keys())
            
            base_shapes = [map_shape_orig_to_shape[sh] for sh in base_shapes]

        if shape in map_shape_to_base_prims.keys():
            # Shape already added. check that previous item matches current.
            assert map_shape_to_base_prims[shape] == base_shapes
            # print(map_shape_to_base_LOSs[shape])
            # print(base_LOSs)
            # assert map_shape_to_base_LOSs[shape] == base_LOSs
        else:
            # Newly encountered shape. Add it
            map_shape_to_base_prims[shape] = base_shapes
            # map_shape_to_base_LOSs[shape] = base_LOSs

    ###################### sanity check, mapping other way around is also unique
    allow_incorrect_shape_names = True

    # Base shapes
    map_base_prims_to_morphed_shape = {}
    for shape, base_prims in map_shape_to_base_prims.items():
        if base_prims is not None:
            base_prims = tuple(base_prims)
            if base_prims in map_base_prims_to_morphed_shape.keys():
                if not allow_incorrect_shape_names:
                    assert map_base_prims_to_morphed_shape[base_prims] == shape
                else:
                    map_base_prims_to_morphed_shape[base_prims].append(shape)
            else:
                map_base_prims_to_morphed_shape[base_prims] = [shape]
    
    # Base LOSs
    if False:
        map_base_LOSs_to_morphed_shape = {}
        for shape, LOSs in map_shape_to_base_LOSs.items():
            if LOSs is not None:
                LOSs = tuple(LOSs)
                if LOSs in map_base_LOSs_to_morphed_shape.keys():
                    if not allow_incorrect_shape_names:
                        assert map_base_LOSs_to_morphed_shape[LOSs] == shape
                    else:
                        map_base_LOSs_to_morphed_shape[LOSs].append(shape)
                else:
                    map_base_LOSs_to_morphed_shape[LOSs] = [shape]


    ############# EXTRACT DS
    # DSlenient = preprocess_dataset_to_datstrokes(D, "singleprim_psycho")
    # DS = preprocess_dataset_to_datstrokes(D, "singleprim")
    # - Decided to just get this, since it would be too much to run through both versions below.
    DS = preprocess_dataset_to_datstrokes(D, clean_ver)

    # Append column with base prims.
    # DSlenient.Dat["base_shapes"] = [map_shape_to_base_prims[sh] for sh in DSlenient.Dat["shape"]]
    DS.Dat["base_shapes"] = [map_shape_to_base_prims[sh] for sh in DS.Dat["shape"]]

    # Minor dumb things
    DS.Dat["loc_on_x"] = [loc[0] for loc in DS.Dat["loc_on"]]
    DS.Dat["loc_on_y"] = [loc[1] for loc in DS.Dat["loc_on"]]

    if False:
        D.Dat["seqc_0_base_LOSs"] = [map_base_LOSs_to_morphed_shape[sh] for sh in D.Dat["seqc_0_shape"]]
        # DSlenient.Dat["base_LOSs"] = [map_base_LOSs_to_morphed_shape[sh] for sh in DSlenient.Dat["shape"]]
        DS.Dat["base_LOSs"] = [map_base_LOSs_to_morphed_shape[sh] for sh in DS.Dat["shape"]]

    # Also save other things related to morphs and shape sets.
    # for ds in [DS, DSlenient]:
    for ds in [DS]:
        ds.Dat["morph_is_morphed"] = [bs is not None for bs in ds.Dat["base_shapes"]] # bool, whether is base (False) or morphed (True)
    if False: # I dont actualyl use this??
        D.Dat["seqc_0_base_shapes"] = [map_shape_to_base_prims[sh] for sh in D.Dat["seqc_0_shape"]]
        D.Dat["seqc_0_morphed"] = [bs is not None for bs in D.Dat["seqc_0_base_shapes"]]

    # Also extract motor params for DS (i.e,, primitivenessv2). 
    # SHould do this here, so that the next step has all the relevant motor params already
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_directly_from_DS
    DS, _ = preprocess_directly_from_DS(DS, prune_strokes=False, microstim_version=False)

    ### And then get version of DF that groups into morph sets
    # Get groups of shape including the base shapes, and all morphed shapes
    # Here defines all variables that depends on cosnidering entire set of shapes (base and morphs).
    list_df = []
    map_morph_set_idx_to_shapes = {}

    map_shape_to_base_prims = {k:map_shape_to_base_prims[k] for k in sorted(map_shape_to_base_prims.keys())}
    map_shape_orig_to_shape = {k:map_shape_orig_to_shape[k] for k in sorted(map_shape_orig_to_shape.keys())}
    map_base_prims_to_morphed_shape = {k:map_base_prims_to_morphed_shape[k] for k in sorted(map_base_prims_to_morphed_shape.keys())}
        
    if False:
        print(1)
        for shape, base_prims in map_shape_to_base_prims.items():
            print(shape, " -- ", base_prims)

        print(2)
        for x, y in map_shape_orig_to_shape.items():
            print(x, " -- ", y)

        print(3)
        for x, y in map_base_prims_to_morphed_shape.items():
            print(x, " -- ", y)

        assert False, "not sure why"

    for i, (base_prims, morphed_shapes) in enumerate(map_base_prims_to_morphed_shape.items()):

        # Pull out rows 
        dfbase = DS.Dat[DS.Dat["shape"].isin(base_prims)].copy()
        dfmorph = DS.Dat[DS.Dat["shape"].isin(morphed_shapes)].copy()

        dfbase["morph_set_idx"] = i
        if not all(dfbase["morph_is_morphed"] == False):
            print(1)
            print(dfbase)
            print(1)
            print(dfbase["morph_is_morphed"])
            print(1)
            print(i, (base_prims, morphed_shapes))
            print(1)
            for shape, base_prims in map_shape_to_base_prims.items():
                print(shape, " -- ", base_prims)

            print(1)
            for x, y in map_shape_orig_to_shape.items():
                print(x, " -- ", y)

            print(1)
            for x, y in map_base_prims_to_morphed_shape.items():
                print(x, " -- ", y)

            assert False, "not sure why"

        # For each base row, determine if it is the first or second base
        dfbase["morph_idxcode_within_set"] = [base_prims.index(sh) for sh in dfbase["shape"]]
        assert all(dfbase["morph_idxcode_within_set"].isin([0,1]))

        dfmorph["morph_set_idx"] = i
        assert all(dfmorph["morph_is_morphed"] == True)
        dfmorph["morph_idxcode_within_set"] = -1 # morphed
        
        list_df.append(dfbase)
        list_df.append(dfmorph)

        # Save mapping from morph index to prims
        # idx:[base_prim_1, [morph prims], base_prim_2]
        base_prim_1 = dfbase[dfbase["morph_idxcode_within_set"]==0]["shape"].values[0]
        base_prim_2 = dfbase[dfbase["morph_idxcode_within_set"]==1]["shape"].values[0]
        morph_prims = sorted(dfmorph["shape"].unique().tolist())
        map_morph_set_idx_to_shapes[i] = [
            base_prim_1, 
            morph_prims,
            base_prim_2
        ]

    DF = pd.concat(list_df).reset_index(drop=True)    
    DSmorphsets = DS.copy()
    DSmorphsets.Dat = DF

    # Create save dir
    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("psycho_singleprims")
    SAVEDIR = f"{SAVEDIR}/cont_morph"
    
    return DS, DSmorphsets, map_morph_set_idx_to_shapes, map_shape_to_base_prims, map_base_prims_to_morphed_shape, SAVEDIR


def psychogood_plot_morphset_drawings(D, DSmorphsets, SAVEDIR, PLOT_EACH_TRIAL = True):
    """
    All drawing-related plots that separate out fore ach morphset. i.e, not the plots that 
    combine all morphsetse intoa  single grid.

    KINDS OF PLOTS:
    - One figure for overlaied data, across indices (draw or beh).
    - One figure foir each idx_within, showing all trials
    
    """
    from pythonlib.tools.plottools import savefig
    # if PLOT_DRAWINGS:
    for psycho_group in DSmorphsets.Dat["morph_set_idx"].unique().tolist():
    # Plot drawings of all morphsets

        dfres_all = DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"] == psycho_group].reset_index(drop=True)

        inds_base1 = D.Dat[D.Dat["trialcode"].isin(dfres_all[dfres_all["morph_idxcode_within_set"] == 0]["trialcode"].tolist())].index.tolist()
        inds_base2 = D.Dat[D.Dat["trialcode"].isin(dfres_all[dfres_all["morph_idxcode_within_set"] == 99]["trialcode"].tolist())].index.tolist()

        savedir = f"{SAVEDIR}/psycho_group_{psycho_group}"
        import os
        os.makedirs(savedir, exist_ok=True)

        # # Print results 
        # fig = grouping_plot_n_samples_conjunction_heatmap(dfres_all, "los", "morph_idxcode_within_set", ["morph_is_morphed"])
        # savefig(fig, f"{savedir}/counts-los-vs-morph_idxcode_within_set.pdf")

        # fig = grouping_plot_n_samples_conjunction_heatmap(dfres_all, "psycho_value", "morph_idxcode_within_set", ["morph_is_morphed"])
        # savefig(fig, f"{savedir}/counts-psycho_value-vs-morph_idxcode_within_set.pdf")

        for ver in ["beh", "task"]:

            if ver=="beh":
                nrand = 3
            elif ver =="task":
                # then you want to plot all, to sanity check that correctly detected
                nrand = None
            else:
                assert False

            SIZE = 4
            list_morph_idx = sorted(dfres_all["morph_idxcode_within_set"].unique())

            n = len(list_morph_idx) + 2
            ncols = 6
            nrows = int(np.ceil(n/ncols))

            fig_combined, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)

            # base 1
            ax = axes.flatten()[0]
            # inds = dfres_base_1["idx_dat"].tolist()
            D.plot_mult_trials_overlaid_on_axis(inds_base1, ax, ver=ver, single_color="b", nrand=nrand)
            ax.set_title(f"base_prim 1")

            # - also plot individual trials
            if False: # is included with morph index 0 and end
                fig, _, _ = D.plotMultTrials2(inds, "strokes_beh")
                savefig(fig, f"{savedir}/indivtrials-base_1-beh.pdf")
                fig, _, _  = D.plotMultTrials2(inds, "strokes_task")
                savefig(fig, f"{savedir}/indivtrials-base_1-task.pdf")

            # Within, in order
            # inds_base1 = dfres_base_1["idx_dat"].tolist()
            # inds_base2 = dfres_base_2["idx_dat"].tolist()

            ct = 1
            for morph_idx in list_morph_idx:
                ax = axes.flatten()[ct]

                if ver == "task":
                    # Ovelray base inds
                    D.plot_mult_trials_overlaid_on_axis(inds_base1, ax, ver=ver, single_color="b", alpha=0.25, nrand=10)
                    D.plot_mult_trials_overlaid_on_axis(inds_base2, ax, ver=ver, single_color="r", alpha=0.25, nrand=10)

                # inds = dfres_all[dfres_all["morph_idxcode_within_set"] == morph_idx]["idx_dat"].tolist()
                inds = D.Dat[D.Dat["trialcode"].isin(dfres_all[dfres_all["morph_idxcode_within_set"] == morph_idx]["trialcode"].tolist())].index.tolist()
                D.plot_mult_trials_overlaid_on_axis(inds, ax, ver=ver, single_color="k")

                ax.set_title(f"morph_idx: {morph_idx}")
                ct += 1

                if PLOT_EACH_TRIAL:
                    # - also plot individual trials
                    fig, _, _  = D.plotMultTrials2(inds, "strokes_beh")
                    savefig(fig, f"{savedir}/indivtrials-morph_idx_{morph_idx}-beh.pdf")
                    if False: # not needed, since tasks are already plot in the overlaid plot
                        fig, _, _  = D.plotMultTrials2(inds, "strokes_task")
                        savefig(fig, f"{savedir}/indivtrials-morph_idx_{morph_idx}-task.pdf")
        
                plt.close("all")

            # base 2
            ax = axes.flatten()[ct]
            # inds = dfres_base_2["idx_dat"].tolist()
            D.plot_mult_trials_overlaid_on_axis(inds_base2, ax, ver=ver, single_color="r", nrand=nrand)
            ax.set_title(f"base_prim 2")

            # - also plot individual trials
            if False: # is included with morph index 0 and end
                fig, _, _  = D.plotMultTrials2(inds, "strokes_beh")
                savefig(fig, f"{savedir}/indivtrials-base_2-beh.pdf")
                fig, _, _  = D.plotMultTrials2(inds, "strokes_task")
                savefig(fig, f"{savedir}/indivtrials-base_2-task.pdf")

            savefig(fig_combined, f"{savedir}/all_overlaid-{ver}.pdf")

        plt.close("all")


def _plot_contmorph_drawings_each_morph_set(DS, map_morph_set_idx_to_shapes, savedir):
    """
    For each morph set, plot the base prims and the morphed
    """
    from pythonlib.tools.plottools import savefig

    for morph_set_idx, (base_prim_1, morphed_prims, base_prim_2) in map_morph_set_idx_to_shapes.items():

        list_shape = [base_prim_1] + morphed_prims + [base_prim_2]

        figholder = DS.plotshape_multshapes_egstrokes_grouped_in_subplots(levels_subplots=list_shape, n_examples=3)
        for j, (fig, _) in enumerate(figholder):
            path = f"{savedir}/morph_set_idx={morph_set_idx}-base_prims={base_prim_1}|{base_prim_2}-sub={j}-BEH.pdf"
            savefig(fig, path)
            
        figholder = DS.plotshape_multshapes_egstrokes_grouped_in_subplots(levels_subplots=list_shape, n_examples=1, ver_behtask="task")
        for j, (fig, _) in enumerate(figholder):
            path = f"{savedir}/morph_set_idx={morph_set_idx}-base_prims={base_prim_1}|{base_prim_2}-sub={j}-TASK.pdf"
            savefig(fig, path)
        
        plt.close("all")


def plot_overview_cont_morph(D, DS, DSmorphsets, map_morph_set_idx_to_shapes, SAVEDIR,
                             use_task_stroke_or_los="stroke"):
    """
    [GOOD] All plots for continuous morph (betwene two shapes)
    This should be folded into psychogood code. and here should subsuming the angle stuff. 
    Here is based on that but shoudl be more general.
    """

    # For every pair of base prims, plot them along with their interpolated across all locations.
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import grouping_print_n_samples

    # Only include first stroke in these plots
    DS = DS.copy()
    DS.Dat = DS.Dat[DS.Dat["stroke_index"] == 0].reset_index(drop=True)

    ### PLOT DRAWINGS
    savedir = f"{SAVEDIR}/drawings"
    os.makedirs(savedir, exist_ok=True)
    _plot_contmorph_drawings_each_morph_set(DS, map_morph_set_idx_to_shapes, savedir)

    ### PLOT COUNTS
    print("Plotting counts...")
    savedir = f"{SAVEDIR}/counts"
    os.makedirs(savedir, exist_ok=True)

    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "shape", "morph_is_morphed", ["gridloc"])
    savefig(fig, f"{savedir}/counts-shape-vs-ismorph.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "shape", "morph_set_idx", None, FIGSIZE=12)
    savefig(fig, f"{savedir}/MORPHSETS-counts-shape-vs-morph_set_idx.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "shape", "morph_set_idx", ["morph_is_morphed"], FIGSIZE=12)
    savefig(fig, f"{savedir}/MORPHSETS-counts-shape-vs-morph_set_idx-2.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "morph_is_morphed", "morph_set_idx", ["gridloc"])
    savefig(fig, f"{savedir}/MORPHSETS-counts-morph_is_morphed-vs-morph_set_idx_gridloc.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "morph_idxcode_within_set", "morph_set_idx", ["gridloc"])
    savefig(fig, f"{savedir}/MORPHSETS-counts-morph_idxcode_within_set-vs-morph_set_idx_gridloc.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "morph_idxcode_within_set", "morph_set_idx", None)
    savefig(fig, f"{savedir}/MORPHSETS-counts-morph_idxcode_within_set-vs-morph_set_idx.pdf")


    from pythonlib.tools.pandastools import grouping_print_n_samples
    savepath = f"{savedir}/groupings_gridloc.txt"
    grouping_print_n_samples(DS.Dat, ["morph_is_morphed", "shape", "gridloc", "character"], savepath=savepath)

    savepath = f"{savedir}/groupings_shape_semantic.txt"
    grouping_print_n_samples(DS.Dat, ["morph_is_morphed", "shape", "shape_semantic"], savepath=savepath)

    savepath = f"{savedir}/MORPHSETS-groupings_morphsets.txt"
    grouping_print_n_samples(DSmorphsets.Dat, 
                             ["morph_set_idx", "morph_is_morphed", "morph_idxcode_within_set", "shape", "gridloc", "character"], 
                             savepath=savepath)
    plt.close("all")

    ### PLOT SCORES
    _plot_overview_scores(D, DSmorphsets, SAVEDIR, use_task_stroke_or_los, DS)



def _plot_overview_scores(D, DSmorphsets, SAVEDIR, use_task_stroke_or_los, DS=None):
    """
    In general, plot scores given DSmorphsets.
    """
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline_directly_from_DS
    from pythonlib.tools.vectools import angle_diff_ccw
    import pandas as pd
   

    # #### ANALYSES (e..,g, timing and velocity)
    savedir = f"{SAVEDIR}/analyses"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")

    # Get angle relative to  base
    group_mins = DSmorphsets.Dat[DSmorphsets.Dat["morph_idxcode_within_set"] == 0].groupby(['morph_set_idx'])["angle"].mean().reset_index()
    group_mins["angle"] = group_mins["angle"] - 1.5 # to make sure ALL trials are now positive.
    group_mins = group_mins.rename(columns={"angle": f'angle_min'})
    DSmorphsets.Dat = pd.merge(DSmorphsets.Dat, group_mins, on=["morph_set_idx"], how='left')
    DSmorphsets.Dat["angle_rel_base0"] = [angle_diff_ccw(row["angle_min"], row["angle"]) for _, row in DSmorphsets.Dat.iterrows()]

    ## First stroke
    dfthis = DSmorphsets.Dat
    dfthis = dfthis[dfthis["gap_from_prev_dur"]<5].reset_index(drop=True)
    for y in ["angle_rel_base0", "angle", "loc_on_x", "loc_on_y", "velocity", "gap_from_prev_dur", "dist_beh_task_strok"]:
        for sharey in [True, False]:
            fig = sns.catplot(data=dfthis, x="morph_idxcode_within_set", y=y, hue="gridloc", col="morph_set_idx", col_wrap=8,
                alpha=0.5, sharey=sharey)
            savefig(fig, f"{savedir}/{y}-vs-idx-sharey={sharey}-1.pdf")

            fig = sns.catplot(data=dfthis, x="morph_idxcode_within_set", y=y, hue="gridloc", col="morph_set_idx", col_wrap=8,
                kind="point", sharey=sharey)
            savefig(fig, f"{savedir}/{y}-vs-idx-sharey={sharey}-2.pdf")

            plt.close("all")

    ## Dataset
    savedir = f"{SAVEDIR}/analyses_dataset_lenient_strokes"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")


    # Number of strokes used
    if "seqc_0_morphed" in D.Dat.columns:
        D = D.copy()
        D.extract_beh_features(["num_strokes_beh"])
        D.preprocessGood(params=["beh_strokes_at_least_one",
                                    "no_supervision"])
        
        dfthis = D.Dat
        fig = sns.catplot(data=dfthis, x="seqc_0_morphed", y="FEAT_num_strokes_beh", 
            alpha=0.4, jitter=True)
        savefig(fig, f"{savedir}/FEAT_num_strokes_beh-1.pdf")

        fig = sns.catplot(data=dfthis, x="seqc_0_morphed", y="FEAT_num_strokes_beh", 
            kind="boxen")
        savefig(fig, f"{savedir}/FEAT_num_strokes_beh-2.pdf")


    if DS is not None:
        grouping = [var, "epoch", "block", "morph_is_morphed"]
        context = var
        contrast = "morph_is_morphed"
        plot_methods = ("grp", "tls", "tl", "tc", "drw")
        preprocess_plot_pipeline_directly_from_DS(DS, grouping=grouping, 
                                                contrast=contrast, context=context,
                                                prune_strokes=prune_strokes, 
                                                plot_methods=plot_methods)
    plt.close("all")
        

def psychogood_preprocess_wrapper(D, PLOT_DRAWINGS = True, PLOT_EACH_TRIAL = True,  PLOT_SCORES=True,
    clean_ver="singleprim_psycho"):
    """
    GOOD - For semi-auto detecting psycho groups, and making all plots.
    [Written for structured morph, shold subsume all, esp cont morph]

    Does entire preprocess and plot pipeline.

    """
    from pythonlib.tools.listtools import stringify_list
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_find_tasks_in_this_psycho_group_wrapper, params_extract_psycho_groupings_manual
    from pythonlib.tools.pandastools import append_col_with_index_of_level_after_grouping, append_col_with_index_of_level
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.plottools import savefig

    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("psycho_singleprims")
    SAVEDIR = f"{SAVEDIR}/general"

    animal = D.animals(True)[0]
    date = D.dates(True)[0]

    # Extract pre-saved params
    PARAMS, los_allowed_to_miss = params_extract_psycho_groupings_manual(animal, date)

    # Generate data for eahc psycho group
    list_dfres = []
    for psycho_group, _params in enumerate(PARAMS):
        los_base_1 = _params["los_base_1"]
        los_within = _params["los_within"]
        los_base_2 = _params["los_base_2"]
        psycho_params = _params["psycho_params"]

        # Extract trials that are in this psycho group
        dfres_within = psychogood_find_tasks_in_this_psycho_group_wrapper(D, los_within, psycho_params)
        dfres_base_1 = psychogood_find_tasks_in_this_psycho_group_wrapper(D, los_base_1, None)
        dfres_base_2 = psychogood_find_tasks_in_this_psycho_group_wrapper(D, los_base_2, None)

        # If there are any base prims that were detected by dfres_within, remove them from dfres_within
        # dfres_within = dfres_within[~dfres_within["los"].isin([los_base_1, los_base_2])].reset_index(drop=True)
        dfres_within = dfres_within[~dfres_within["los"].isin(dfres_base_1["los"].unique().tolist())].reset_index(drop=True)
        dfres_within = dfres_within[~dfres_within["los"].isin(dfres_base_2["los"].unique().tolist())].reset_index(drop=True)

        # Add features
        dfres_within = append_col_with_index_of_level_after_grouping(dfres_within, None, "psycho_value", "morph_idxcode_within_set")
        dfres_within["morph_is_morphed"] = True
        dfres_within["morph_idxcode_within_set"] += 1

        dfres_base_1["morph_is_morphed"] = False
        dfres_base_1["morph_idxcode_within_set"] = 0

        dfres_base_2["morph_is_morphed"] = False
        # dfres_base_2["morph_idxcode_within_set"] = max(dfres_within["morph_idxcode_within_set"]) + 1
        dfres_base_2["morph_idxcode_within_set"] = 99

        dfres_all = pd.concat([dfres_base_1, dfres_base_2, dfres_within]).reset_index(drop=True)
        dfres_all["morph_set_idx"] = psycho_group

        list_dfres.append(dfres_all)
    DFRES = pd.concat(list_dfres).reset_index(drop=True)

    # CHECK that all los have been assigned to something
    los_all = D.Dat["los_info"]
    tmp = DFRES["los"].unique().tolist()
    los_missing = []
    for los in los_all:
        if los not in tmp:
            los_missing.append(los)

    los_missing = sorted(set(los_missing))
    los_missing_actual = [los for los in los_missing if los not in los_allowed_to_miss]
    for los in los_missing_actual:
        print(f"{los},")
    assert len(los_missing_actual)==0, "missing these los... is that ok? Copy and paste the los that you are fine missing, and place above into los_allowed_to_miss"

    ######### have data for each morph set and each stroke.
    ############# EXTRACT DS
    # DSlenient = preprocess_dataset_to_datstrokes(D, "singleprim_psycho")
    # DS = preprocess_dataset_to_datstrokes(D, "singleprim")
    # - Decided to just get this, since it would be too much to run through both versions below.
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    DS = preprocess_dataset_to_datstrokes(D, clean_ver)
    DS.dataset_append_column("los_info")
    
    # # Map from psycho group to trial info
    # map_group_to_los = {}
    # for morph_group in DFRES["morph_set_idx"].unique().tolist():
    #     list_idx = DFRES[DFRES["morph_set_idx"]==morph_group]["morph_idxcode_within_set"].unique().tolist()
    #     map_group_to_los[morph_group] = {idx:None for idx in list_idx}
    #     for idx_within_group in list_idx:
    #         los = DFRES[(DFRES["morph_set_idx"]==morph_group) & (DFRES["morph_idxcode_within_set"]==idx_within_group)]["los"].unique().tolist()
    #         map_group_to_los[morph_group][idx_within_group] = los

    ##################################### PLOTS/ANALYSIS
    DSmorphsets = psychogood_preprocess_generate_DSmorphset_and_plot(D, DFRES, SAVEDIR, 
                                                PLOT_SCORES, PLOT_DRAWINGS, clean_ver=clean_ver)

    return DFRES, DSmorphsets, PARAMS, los_allowed_to_miss
        

def psychogood_preprocess_generate_DSmorphset_and_plot(D, DFRES, SAVEDIR, 
                                              PLOT_SCORES=True, PLOT_DRAWINGS=True, clean_ver = "singleprim_psycho"):
    """
    WRapper for all merthods for extracting DFmorphset, scoring and plotting, for structured morphs,
    where the input is the output of the psychogood_preprocess_wrapper... functions.

    PARAMS:
    - DFRES, each row is single (morphset, idx_within, los, trialcode). Holds meta information mapping between these
    variables. A giventrial could be multiple rows.
    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

    ### EXTRACT DS
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    DS = preprocess_dataset_to_datstrokes(D, clean_ver)
    DS.dataset_append_column("los_info")

    # Map from psycho group to trial info
    map_group_to_los = {}
    for morph_group in DFRES["morph_set_idx"].unique().tolist():
        list_idx = DFRES[DFRES["morph_set_idx"]==morph_group]["morph_idxcode_within_set"].unique().tolist()
        map_group_to_los[morph_group] = {idx:None for idx in list_idx}
        for idx_within_group in list_idx:
            los = DFRES[(DFRES["morph_set_idx"]==morph_group) & (DFRES["morph_idxcode_within_set"]==idx_within_group)]["los"].unique().tolist()
            map_group_to_los[morph_group][idx_within_group] = los

    ##### GENMERATE DSmorphsets
    # (1) DS, concatting df where each df is a single morph set, e.g., morphj idnices [0 1 2 3... 99], where 0 and 99 are base prims
    list_df = []
    for group, idx_dict in map_group_to_los.items():
        for idx, list_los in idx_dict.items():
            df = DS.Dat[DS.Dat["los_info"].isin(list_los)].copy()
            df["morph_set_idx"] = group
            df["morph_idxcode_within_set"] = idx
            df["morph_is_morphed"] = idx not in [0, 99]
            list_df.append(df)
    DF = pd.concat(list_df).reset_index(drop=True)
    DSmorphsets = DS.copy()
    DSmorphsets.Dat = DF

    ### PLOT
    DSmorphsets = psychogood_plot_morphset_wrapper(D, DS, DSmorphsets, SAVEDIR, PLOT_DRAWINGS, PLOT_SCORES)

    return DSmorphsets

    
def psychogood_plot_morphset_wrapper(D, DS, DSmorphsets, SAVEDIR, PLOT_DRAWINGS=True, PLOT_SCORES=True):
    """
    Given DSmorphset, make all the plots. Wrapper.
    Ideally this is  gthe only plotting function called when using DSmorphsets
    """
    from pythonlib.tools.listtools import stringify_list
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_find_tasks_in_this_psycho_group_wrapper, params_extract_psycho_groupings_manual
    from pythonlib.tools.pandastools import append_col_with_index_of_level_after_grouping, append_col_with_index_of_level
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.plottools import savefig

    DSmorphsets.dataset_append_column("los_info")
    DSmorphsets.Dat["los"] = "los_info"
    DS.dataset_append_column("los_info")
    DS.Dat["los"] = "los_info"

    # Hacky, if this isnt angle morph, then might not have psycho continuos value. Just give it something that allows correct ordering.
    if "psycho_value" not in DSmorphsets.Dat.columns:
        DSmorphsets.Dat["psycho_value"] = DSmorphsets.Dat["morph_idxcode_within_set"]

    # Plot counts
    # Print results 
    # if "shapeorig_psycho" in DSmorphsets.Dat.columns:
    #     var_each_image = "shape"
    # else:

    savedir = f"{SAVEDIR}/counts_each_morphset"
    os.makedirs(savedir, exist_ok=True)
    for morphset in DSmorphsets.Dat["morph_set_idx"].unique():
        dfres = DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"] == morphset].reset_index(drop=True)
        fig = grouping_plot_n_samples_conjunction_heatmap(dfres, "los", "morph_idxcode_within_set", ["morph_is_morphed"])
        savefig(fig, f"{savedir}/morphset={morphset}-counts-los-vs-morph_idxcode_within_set.pdf")

        fig = grouping_plot_n_samples_conjunction_heatmap(dfres, "psycho_value", "morph_idxcode_within_set", ["morph_is_morphed"])
        savefig(fig, f"{savedir}/morphset={morphset}-counts-psycho_value-vs-morph_idxcode_within_set.pdf")


    ######## LWAYS DO THIS
    savedir = f"{SAVEDIR}/analyses_using_morphsets"
    os.makedirs(savedir, exist_ok=True)
    _plot_overview_scores(D, DSmorphsets, savedir, use_task_stroke_or_los="los")

    ################# Plot drawings
    if PLOT_DRAWINGS:

        # Single plot holding all
        from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_plot_drawings_morphsets
        savedir = f"{SAVEDIR}/drawings_combined"
        os.makedirs(savedir, exist_ok=True)
        psychogood_plot_drawings_morphsets(DSmorphsets, savedir)

        # All trials for each morphset,  many plots
        savedir = f"{SAVEDIR}/drawings_each"
        os.makedirs(savedir, exist_ok=True)
        psychogood_plot_morphset_drawings(D, DSmorphsets, savedir, True)


    ##################################### PLOTS
    if PLOT_SCORES:
        # THen you need to regenerate DSMorphsets. This is becuase you want to preprocess motor stats in DS, which is much smaller
        # dataset than DSmorphsets, and then re-split it to DSMorphsets

        from pythonlib.tools.pandastools import find_unique_values_with_indices
        from pythonlib.tools.pandastools import append_col_with_grp_index
        from pythonlib.tools.pandastools import append_col_with_index_of_level, append_col_with_index_of_level_after_grouping
        from pythonlib.tools.pandastools import grouping_print_n_samples
        from pythonlib.tools.checktools import check_objects_identical
        from math import pi
        from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_directly_from_DS
        from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline_directly_from_DS


        # Also extract motor params for DS (i.e,, primitivenessv2). 
        # SHould do this here, so that the next step has all the relevant motor params already
        DS, _ = preprocess_directly_from_DS(DS, prune_strokes=False, microstim_version=False)
        # Minor dumb things
        if DS is not None:
            DS.Dat["loc_on_x"] = [loc[0] for loc in DS.Dat["loc_on"]]
            DS.Dat["loc_on_y"] = [loc[1] for loc in DS.Dat["loc_on"]]

        # Params for PRIMITIVENESS plots
        use_task_stroke_or_los = "los"
        if use_task_stroke_or_los == "stroke":
            var = "stroke"
        elif use_task_stroke_or_los == "los":
            var = "los_info"
        else:
            assert False
        prune_strokes = False

        # Map from psycho group to trial info
        map_group_to_los = {}
        for morph_group in DSmorphsets.Dat["morph_set_idx"].unique().tolist():
            list_idx = DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"]==morph_group]["morph_idxcode_within_set"].unique().tolist()
            map_group_to_los[morph_group] = {idx:None for idx in list_idx}
            for idx_within_group in list_idx:
                los = DSmorphsets.Dat[(DSmorphsets.Dat["morph_set_idx"]==morph_group) & (DSmorphsets.Dat["morph_idxcode_within_set"]==idx_within_group)]["los_info"].unique().tolist()
                map_group_to_los[morph_group][idx_within_group] = los

        ##### Two variations on DS
        # (1) DS, concatting df where each df is a single morph set, e.g., morphj idnices [0 1 2 3... 99], where 0 and 99 are base prims
        list_df = []
        for group, idx_dict in map_group_to_los.items():
            for idx, list_los in idx_dict.items():
                df = DS.Dat[DS.Dat["los_info"].isin(list_los)].copy()
                df["morph_set_idx"] = group
                df["morph_idxcode_within_set"] = idx
                df["morph_is_morphed"] = idx not in [0, 99]
                list_df.append(df)
        DF = pd.concat(list_df).reset_index(drop=True)
        DSmorphsets = DS.copy()
        DSmorphsets.Dat = DF

        _plot_overview_scores(D, DSmorphsets, SAVEDIR, use_task_stroke_or_los="los")

        grouping = [var, "epoch", "block", "morph_is_morphed", "morph_set_idx"]
        contrast = "morph_is_morphed"
        context = "morph_set_idx"
        plot_methods = ("grp", "tls", "tl", "tc")
        preprocess_plot_pipeline_directly_from_DS(DSmorphsets, grouping=grouping, 
                                                contrast=contrast, context=context,
                                                prune_strokes=prune_strokes, plot_methods=plot_methods,
                                                savedir_suffix="MORPH_SETS")


        # (2) DS, where each df is a single morph set x morph index, e.g.., morph indices [0 1 99], [0 2 99] will be separated dfs that are concatted
        list_df = []
        morph_set_idx_global = 0
        for group, idx_dict in map_group_to_los.items():
            for idx, list_los in idx_dict.items():
                if idx not in [0, 99]:
                    # Then this is a morph:

                    df_morph = DS.Dat[DS.Dat["los_info"].isin(list_los)].copy()
                    df_morph["morph_set_idx"] = group
                    df_morph["morph_set_idx_global"] = morph_set_idx_global
                    df_morph["morph_idxcode_within_set"] = idx
                    df_morph["morph_is_morphed"] = True

                    list_los_base_1 = idx_dict[0] # base 1
                    df_base1 = DS.Dat[DS.Dat["los_info"].isin(list_los_base_1)].copy()
                    df_base1["morph_set_idx"] = group
                    df_base1["morph_set_idx_global"] = morph_set_idx_global
                    df_base1["morph_idxcode_within_set"] = 0
                    df_base1["morph_is_morphed"] = False

                    list_los_base_2 = idx_dict[99] # base 2
                    df_base2 = DS.Dat[DS.Dat["los_info"].isin(list_los_base_2)].copy()
                    df_base2["morph_set_idx_global"] = morph_set_idx_global
                    df_base2["morph_set_idx"] = group
                    df_base2["morph_idxcode_within_set"] = 99
                    df_base2["morph_is_morphed"] = False

                    assert len(df_morph)>0
                    assert len(df_base1)>0
                    assert len(df_base2)>0
                    
                    list_df.extend([df_morph, df_base1, df_base2])
                    morph_set_idx_global += 1

        DF = pd.concat(list_df).reset_index(drop=True)
        DSmorphsetsSingle = DS.copy()
        DSmorphsetsSingle.Dat = DF

        grouping = [var, "epoch", "block", "morph_is_morphed", "morph_set_idx_global"]
        contrast = "morph_is_morphed"
        context = "morph_set_idx_global"
        plot_methods = ("grp", "tls", "tl", "tc")
        preprocess_plot_pipeline_directly_from_DS(DSmorphsetsSingle, grouping=grouping, 
                                                contrast=contrast, context=context,
                                                prune_strokes=prune_strokes, plot_methods=plot_methods,
                                                savedir_suffix="MORPH_SETS_EACH_IDX_GLOBAL")
    # else:
    #     # Still return a DSmorph.
    #     # NOTE: must run again, since above it runs after primitiveness2 stuff, but that takes a while, dont' want to do 
    #     # it here.
        
    #     # (1) DS, concatting df where each df is a single morph set, e.g., morphj idnices [0 1 2 3... 99], where 0 and 99 are base prims
    #     list_df = []
    #     for group, idx_dict in map_group_to_los.items():
    #         for idx, list_los in idx_dict.items():
    #             df = DS.Dat[DS.Dat["los_info"].isin(list_los)].copy()
    #             df["morph_set_idx"] = group
    #             df["morph_idxcode_within_set"] = idx
    #             df["morph_is_morphed"] = idx not in [0, 99]
    #             list_df.append(df)
    #     DF = pd.concat(list_df).reset_index(drop=True)
    #     DSmorphsets = DS.copy()
    #     DSmorphsets.Dat = DF

    return DSmorphsets

def psychogood_preprocess_wrapper_using_tsc_inds(D, morphs_tsc_idx_start, morphs_tsc_idx_end, morphs_tsc_n_sets, morphs_tsc_n_each_set,
                                        morphs_tsc_map_to_these_indices, list_example_base_los, nlocations_expected, los_allowed_to_exclude,
                                        print_summary = True,  PLOT_SCORES=True, PLOT_DRAWINGS = True):
    """
    For extraction and plotting of structured morphs (and any morph in general), as alterantive to psychogood_preprocess_wrapper,
    where here you manaulyl define morphsets using the matlab TSC indices.

    See within for how to mnaully define params for morphsets.

    This has advatnage of not needing to deal with taskparams.

    One diadvantage is that it doenst have psycho variable (it just hacks it with the idx within).


    PARAMS
    [Input params, see params_extract_psycho_groupings_manual_using_tsc_inds]
    
    RETURNS;
    - DFRES, each row is single trial.
    - DSmorphsets, concatted DS after first splitting by morphset.
    """

    from pythonlib.tools.pandastools import grouping_print_n_samples

    assert morphs_tsc_n_sets * morphs_tsc_n_each_set == morphs_tsc_idx_end - morphs_tsc_idx_start + 1

    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("psycho_singleprims")
    SAVEDIR = f"{SAVEDIR}/general"

    # Helper functino
    def extract_all_los_if_they_exist(los_example, 
                                    res, morphset, morph_idxcode_within_set, morph_is_morphed, psycho_value,
                                    allow_to_not_find=False):
        """
        Helper function to append data to res. 
        See use within.
        Datapt is single trial.
        """
        assert isinstance(los_example, tuple)
        assert len(los_example)==3
        assert isinstance(los_example[0], str)
        assert isinstance(los_example[1], int)
        assert isinstance(los_example[2], int)
        
        if allow_to_not_find == False:
            assert los_example in map_los_to_tscind

        if los_example in map_los_to_tscind:
            tsc_ind = map_los_to_tscind[los_example]
            list_los = map_tscind_to_listlos[tsc_ind] # exand to all los that match this inputed los
            for los in list_los:
                # get all trials
                for tc in D.Dat[D.Dat["los_info"] == los]["trialcode"].tolist():
                    res.append({
                        "trialcode":tc,
                        "morph_set_idx":morphset,
                        "morph_idxcode_within_set":morph_idxcode_within_set,
                        "morph_is_morphed":morph_is_morphed,
                        "psycho_value":psycho_value,
                        "los":los
                    })

    ### PREPARE
    # First, get mapping between TSC and los
    # (1) append tsc_ind to dataset
    tsc_inds = []
    for ind in range(len(D.Dat)):
        plan = D.taskclass_extract_planclass(ind)
        si = int(plan["Info"]["TaskSetClass"]["sampled_prims_inds"])
        assert isinstance(si, int), "multi-prim plan... why?"
        # assert len(si)==1, "multi-prim plan... why?"
        tsc_inds.append(si)
    D.Dat["tsc_ind"] = tsc_inds
    # (2) Get mapping
    map_tscind_to_listlos = {}
    map_los_to_tscind = {}
    for _, row in D.Dat.iterrows():
        tsc_ind = row["tsc_ind"]
        los = row["los_info"]

        if tsc_ind in map_tscind_to_listlos:
            if los not in map_tscind_to_listlos[tsc_ind]:
                map_tscind_to_listlos[tsc_ind].append(los)
        else:
            map_tscind_to_listlos[tsc_ind] = [los]
        
        if los in map_los_to_tscind:
            assert map_los_to_tscind[los] == tsc_ind
        else:
            map_los_to_tscind[los] = tsc_ind

    ########################## Get morphed tasks
    morphset_tscinds = []
    on = morphs_tsc_idx_start
    for _ in range(morphs_tsc_n_sets):
        off = on+morphs_tsc_n_each_set
        morphset_tscinds.append(list(range(on,off)))
        on = off

    res = []
    ### (1)  Get all the morphed los
    morph_is_morphed = True
    HACK_PSYCHO_VALUE = True
    for morphset, tscinds in enumerate(morphset_tscinds):
        for _i, _tscind in enumerate(tscinds):

            idx_within = morphs_tsc_map_to_these_indices[_i]

            if print_summary:
                print("-----------")
                print(morphs_tsc_map_to_these_indices)
                print(_i, idx_within)
                print(morphset, " -- ", idx_within, " -- ", los)
            
            if HACK_PSYCHO_VALUE:
                # psycho_value = _i # HACKKY
                psycho_value = idx_within # HACKKY
            else:
                assert False, "what method to use?"

            list_los = map_tscind_to_listlos[_tscind]

            for los in list_los:
                extract_all_los_if_they_exist(los, 
                                                res, morphset, idx_within, morph_is_morphed, psycho_value)
    
    ### (2) Base prim los
    for morphset, (example_los_base_1, example_los_base_2) in enumerate(list_example_base_los):

        if isinstance(example_los_base_1[2], tuple):
            # Then you gave multiple candidate los. try each one, and take whatever find.
            for _ind in example_los_base_1[2]:
                _los = (example_los_base_1[0], example_los_base_1[1], _ind)
                extract_all_los_if_they_exist(_los, res, morphset, 0, False, 0, allow_to_not_find=True)
        else:
            # Then you gave a single candidate los. it must exist or else fail.
            extract_all_los_if_they_exist(example_los_base_1, res, morphset, 0, False, 0)

        if isinstance(example_los_base_2[2], tuple):
            for _ind in example_los_base_2[2]:
                _los = (example_los_base_2[0], example_los_base_2[1], _ind)
                extract_all_los_if_they_exist(_los, res, morphset, 99, False, 99, allow_to_not_find=True)
        else:
            extract_all_los_if_they_exist(example_los_base_2, res, morphset, 99, False, 99)

    ### Finalize
    DFRES = pd.DataFrame(res)    
    if print_summary:
        from pythonlib.tools.pandastools import stringify_values, grouping_print_n_samples
        _dfres = stringify_values(DFRES)
        grouping_print_n_samples(_dfres, ["morph_set_idx", "morph_idxcode_within_set", "morph_is_morphed", "psycho_value", "los"],
                                 print_header=True)        

    ############## Sanity checks 
    # (1) Each idx_within must have data at all locations.
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    grpvar = ["morph_set_idx", "morph_idxcode_within_set", "morph_is_morphed", "psycho_value"]
    grpdict = grouping_append_and_return_inner_items_good(DFRES, grpvar)
    for grp, inds in grpdict.items():
        if len(inds)<nlocations_expected:
            print("This grp didnt have corrent expected unique tasks (i.e., locations)")
            print(grpvar)
            print(grp)
            print(inds)
            assert False
    
    # (2) Check that not too many los were not included in the above.
    # Print all the los that were not included.
    los_exist_all = D.Dat["los_info"].unique().tolist()
    los_found = DFRES["los"].unique().tolist()
    los_excluded = [los for los in los_exist_all if los not in los_found]
    print("These LOS were not included in morph or base prims...")
    for x in los_excluded:
        print(x)
    n = sum(D.Dat["los_info"].isin(los_excluded))
    ntot = len(D.Dat["los_info"])
    print(f"counting for this mnay trials: {n}/{ntot}")
    assert n/ntot<0.2, "why excludd so many? this might be expected"

    if not all([los in los_allowed_to_exclude for los in los_excluded]):
        print("Los ecluded, but not allowed to: ")
        for los in los_excluded:
            if not los in los_allowed_to_exclude:
                print(f"{los},")
        assert False, "if is fine to fail, then add them to params"

    #################### ANALYSIS
    DSmorphsets = psychogood_preprocess_generate_DSmorphset_and_plot(D, DFRES, SAVEDIR, 
                                                PLOT_SCORES, PLOT_DRAWINGS)
    
    return DFRES, DSmorphsets

def simplify_morph_info_dict(morph_info):
    """
    Return simplified version of morph info, human-readable.
    RETURNS:
    - list of 2 dicts, one for each of 2 morph endpoints
    [
      {'los': ('singleprims', 114, 9), 'shape': ('arcdeep', 4, 4, 0), 'frac': 0.16666666666666666, 'flip_task1': False, 'flip_task2': False, 'frac_func': array([], shape=(0, 0), dtype=float64)}, 
      {'los': ('singleprims', 114, 30), 'shape': ('circle', 6, 1, 0), 'frac': 0.16666666666666666, 'flip_task1': False, 'flip_task2': False, 'frac_func': array([], shape=(0, 0), dtype=float64)}
    ]
    """

    if morph_info is None or len(morph_info)==0:
        return None
    else:   
        assert isinstance(morph_info, list)
        for x in morph_info:
            assert isinstance(x, dict)
            assert "frac" in x.keys()
        return morph_info

def simplify_extra_tform_dict(extra_tform_params):
    """
    Returns human readable version of tforms
    E.g., if no global tform, but have specifi tform for sencond prim:
    RETURNS:
    - methods_tforms, either None (no tform) or dict (e.g., {sx:1.2, th:0.2})
    - methods_tforms_prim, list of len strokes, where each stroke is represnted liek above, 
    either NOne, or dict

        example:
        (None, [None, {'sx': array(1.)}])
    """
    
    if extra_tform_params is None:
        return None, None
    
    if "tforms" in extra_tform_params and len(extra_tform_params["tforms"])>0:
        # methods_tforms = [_tform[0] for _tform in extra_tform_params["tforms"]]
        assert len(list(extra_tform_params.keys())) == len(set(list(extra_tform_params.keys()))), "you have tforms that repeat. in matlab this would compound. how to deal with here?"
        methods_tforms = {_tform[0]:_tform[1] for _tform in extra_tform_params["tforms"]}
        if len(methods_tforms)==0:
            methods_tforms = None
    else:
        methods_tforms = None

    if "tforms_each_prim_p" in extra_tform_params and len(extra_tform_params["tforms_each_prim_p"])>0:
        methods_tforms_prim = []
        for prim_tform in extra_tform_params["tforms_each_prim_p"]:
            if len(prim_tform)==0:
                methods_tforms_prim.append(None)
            else:
                _keys = [_tform[0] for _tform in prim_tform]
                assert len(_keys) == len(set(_keys)), "you have tforms that repeat. in matlab this would compound. how to deal with here?"
                methods_tforms_prim.append({_tform[0]:_tform[1].item() for _tform in prim_tform})
                # methods_tforms_prim = sorted(methods_tforms_prim)
    else:
        print("Adsad", extra_tform_params)
        assert False
        methods_tforms_prim = None
    
    return methods_tforms, methods_tforms_prim

        
def psychogood_make_task_template_to_find_psycho_variants(plan, extra_tform_params, morph_info, psycho_params):
    """ Generate a template (reprenetaiton of thet ask) which will be identical across all
    tasks in the same psycho group, by returning task after excising infomation about the 
    value of the psycho param

    out==out can be used to find tasks in the same psycho
    
    PARAMS:
    - plan
    - extra_tform_params
    - morph_info
    - psycho_params
    ---> See calling code for how get these.

    """
    from pythonlib.tools.listtools import stringify_list
    import copy

    # Simplify the inputs
    methods_tforms, methods_tforms_prim = simplify_extra_tform_dict(extra_tform_params)
    morph_info = simplify_morph_info_dict(morph_info)

    idx_prim_plan = 0

    def _remove_tform_values(methods_tforms, methods_tforms_prim, idx_prim_remove):
        """
        Remove any information aboiut the valuye of the trforms
        for tforms, keep just the keys
        PARAMS:
        EXAMPLE:
            methods_tforms_prims = [{'sx': 0.16666666666666666}, None, {'sx': 0.5}]
            -->
            (if idx_prim_remove == 0)
            methods_tforms_prim_keys = [['sx'], None, ['sx', 0.5]]

            (if idx_prim_remove == "all")
            methods_tforms_prim_keys = [['sx'], None, ['sx']]

            (if idx_prim_remove == [0, 2])
            methods_tforms_prim_keys = [['sx'], None, ['sx']]

            (if idx_prim_remove == 2)
            methods_tforms_prim_keys = [['sx', 0.16666666666666666], None, ['sx']]

            (if idx_prim_remove == None)
            methods_tforms_prim_keys = [['sx', 0.16666666666666666], None, ['sx', 0.5]]

        """

        if methods_tforms is not None:
            methods_tforms_keys = list(methods_tforms.keys())
        else:
            methods_tforms_keys = None

        methods_tforms_prim_keys = []
        if methods_tforms_prim is not None:
            for _i, tform in enumerate(methods_tforms_prim): # each prim
                if tform is not None and idx_prim_remove=="all":
                    # Then this is the idx that you want to remove values from
                    methods_tforms_prim_keys.append(list(tform.keys()))
                elif tform is not None and _i == idx_prim_remove:
                    # Then this is the idx that you want to remove values from
                    methods_tforms_prim_keys.append(list(tform.keys()))
                elif tform is not None and isinstance(idx_prim_remove, list) and _i in idx_prim_remove:
                    # Then this is the idx that you want to remove values from
                    methods_tforms_prim_keys.append(list(tform.keys()))
                elif tform is not None:
                    # Keep all values
                    methods_tforms_prim_keys.append(list(tform.keys()) + list(tform.values()))
                else:
                    methods_tforms_prim_keys.append(None)
        else:
            methods_tforms_prim_keys = None

        return methods_tforms_keys,  methods_tforms_prim_keys

    # Based on psychoparams, decide what strings to exclude from template, i.e., exclude the values for the psycho var.
    if psycho_params is None:
        # Then template will include all info, i.e., doesnt excise out the psycho value.
        # useful for finding base prims, at diff locations (which ahve same plan but diff LOS)
        plan_prim = plan[idx_prim_plan] # Ignore the global rels
        template = stringify_list(plan_prim) + stringify_list(methods_tforms) + stringify_list(methods_tforms_prim) + stringify_list(morph_info)
    else:
        psycho_ver = psycho_params["psycho_ver"]
        if "exclude_extra_tform_values" in psycho_params:
            exclude_extra_tform_values = psycho_params["exclude_extra_tform_values"]
        else:
            exclude_extra_tform_values = False

        if "exclude_interp_values" in psycho_params:
            exclude_interp_values = psycho_params["exclude_interp_values"]
        else:
            exclude_interp_values = False

        if "convert_extra_tforms_to_sign" in psycho_params:
            convert_extra_tforms_to_sign = psycho_params["convert_extra_tforms_to_sign"]
        else:
            convert_extra_tforms_to_sign = False

        if "replace_morph_params_with_binary_whether_exists" in psycho_params:
            replace_morph_params_with_binary_whether_exists = psycho_params["replace_morph_params_with_binary_whether_exists"]
        else:
            replace_morph_params_with_binary_whether_exists = False

        # Clean up morph info.
        if morph_info is None or len(morph_info)==0:
            morph_info = [] 
        else:
            if replace_morph_params_with_binary_whether_exists:
                # Still note down that params exist, just exclude the values.
                morph_info = [{"exists":"exists"}, {"exists":"exists"}]
            else:
                # Inculde all params
                tmp = [None, None]  
                tmp[0] = {k:v for k, v in morph_info[0].items() if not k=="frac"}
                tmp[1] = {k:v for k, v in morph_info[1].items() if not k=="frac"}
                morph_info = tmp
        
        ### Generate template
        if psycho_ver == "attachpt1_interp":
            idx_motif_char_flex = psycho_params["idx_motif_char_flex"]
            idx_rel = psycho_params["idx_rel"]
            if "idx_rel_within" in psycho_params:
                idx_rel_within = psycho_params["idx_rel_within"]
            else:
                idx_rel_within = 0

            # (1) Remove any informaiton about the value of the interp
            # make sure this is the correct index
            try:
                # print(plan[idx_motif_char_flex][1][1][2*idx_rel+1][2][0])
                # print(idx_motif_char_flex)
                # print(idx_rel)
                # print("PLAN:", plan[idx_motif_char_flex])
                # assert False
                if "interp_end1_end2" in plan[idx_motif_char_flex][1][1][2*idx_rel+1][2][idx_rel_within]:
                    # print(plan[idx_motif_char_flex][1][1][2*idx_rel+1][2][0])
                    plan = copy.deepcopy(plan)
                    # Then this is has the interp at the correct place. excise the value.
                    # replace it by removing the actual interp vallue
                    plan[idx_motif_char_flex][1][1][2*idx_rel+1][2][idx_rel_within] = "interp_end1_end2"
            except IndexError as err:
                # HACKY, this plan just doesnt even have this index...
                pass
            except Exception as err:
                raise err
            
            plan_prim = plan[idx_prim_plan] # Ignore the global rels

            # Optional ways to modify the template.
            if convert_extra_tforms_to_sign and methods_tforms_prim is not None:
                # useful if sign of this variable distinguishes this psycho group from another similar one.
                methods_tforms_prim = [{k:np.sign(v) for k, v in tform.items()} for tform in methods_tforms_prim if tform is not None]
            if convert_extra_tforms_to_sign and methods_tforms is not None:
                methods_tforms = {k:np.sign(v) for k, v in methods_tforms.items()}
            
            if exclude_extra_tform_values:
                methods_tforms, methods_tforms_prim = _remove_tform_values(methods_tforms, methods_tforms_prim, idx_prim_remove="all")
            
            template = stringify_list(plan_prim) + stringify_list(methods_tforms) + stringify_list(methods_tforms_prim) + stringify_list(morph_info)

        elif psycho_ver in ["extra_tforms", "extra_tforms_each_prim"]:
            # Tforms are in extra tforms.
            
            idx_prim_remove = psycho_params["idx_prim"]
            
            methods_tforms_keys, methods_tforms_prim_keys = _remove_tform_values(methods_tforms, methods_tforms_prim, idx_prim_remove)

            plan_prim = plan[idx_prim_plan] # Ignore the global rels
            template = stringify_list(plan_prim) + stringify_list(methods_tforms_keys) + stringify_list(methods_tforms_prim_keys) + stringify_list(morph_info)
        
        elif psycho_ver in ["morph_frac"]:
            # value is in morph info

            plan_prim = plan[idx_prim_plan] # Ignore the global rels

            if morph_info is None or len(morph_info)==0:
                morph_info = [] 
            else:
                # Inculde all params
                tmp = [None, None]  
                tmp[0] = {k:v for k, v in morph_info[0].items() if not k=="frac"}
                tmp[1] = {k:v for k, v in morph_info[1].items() if not k=="frac"}
                morph_info = tmp

            template = stringify_list(plan_prim) + stringify_list(methods_tforms) + stringify_list(methods_tforms_prim) + stringify_list(morph_info)
        else:
            print(psycho_ver)
            assert False
        
        if exclude_interp_values:
            template = [t for t in template if "interp_end1_end2" not in t]

    return template


def psychogood_extract_psycho_var(plan, extra_tform_params, morph_info, psycho_params):
    """ 
    Extract scalar variable that defines vairation that defines differences across
    tasks within same psychoemtric group
    PARAMS:
    - plan, 
    - extra_tform_params,
    - morph_info
    - psycho_params
    ---> See calling code for how get above.
    RETURNS:
    - psycho_var, scalar value of the psycho variable
    """

    psycho_ver = psycho_params["psycho_ver"]
    
    if psycho_ver == "attachpt1_interp":
        # Interp of attach pt.

        # plan = 
        # (['motif_char_flex',
        #   ['direct_chunk_entry',
        #    [['line',
        #      ['prot',
        #       array(18.),
        #       array(2.),
        #       array([], shape=(0, 0), dtype=float64),
        #       array(0.)]],
        #     ['translate_xy', array(0.), ['center', 'center', array([0., 0.])]],
        #     ['line',
        #      ['prot',
        #       array(18.),
        #       array(1.),
        #       array([], shape=(0, 0), dtype=float64),
        #       array(0.)]],
        #     ['translate_xy',
        #      array(-1.),
        #      ['interp_end1_end2_50', 'end1', array([0., 0.])]]]]],
        #  ['translate_xy',
        #   array(0.),
        #   ['center_prim_sketchpad', 'center_prim_sketchpad', array([ 1. , -0.2])]])
        # ---> pulls out 50 (from interp_end1_end2_50)

        idx_motif_char_flex = psycho_params["idx_motif_char_flex"]
        idx_rel = psycho_params["idx_rel"]
        if "idx_rel_within" in psycho_params:
            idx_rel_within = psycho_params["idx_rel_within"]
        else:
            idx_rel_within = 0

        assert plan[idx_motif_char_flex][1][0] == "direct_chunk_entry"
        _prims = plan[idx_motif_char_flex][1][1][0::2]
        _rels = plan[idx_motif_char_flex][1][1][1::2]

        if psycho_ver == "attachpt1_interp":
            interp_string = _rels[idx_rel][2][idx_rel_within] # e.g, interp_end1_end2_50
            i = interp_string.find("interp_end1_end2")
            interp_frac = int(interp_string[i+17:]) # .e.g, 50
        
        psycho_value = interp_frac

    elif psycho_ver == "extra_tforms_each_prim":
        # Variable in extra tforms

        idx_prim = psycho_params["idx_prim"]
        tform_key = psycho_params["tform_key"]

        methods_tforms, methods_tforms_prim = simplify_extra_tform_dict(extra_tform_params)

        if isinstance(idx_prim, list):
            # Take sum of the values... Hacky way to convert 2d psycho to 1d...
            _vals = [methods_tforms_prim[_i][tform_key] for _i in idx_prim]
            _val_diff = np.abs(max(_vals) - min(_vals)) # Take diff, as tiebreak
            _val_diff_2 = _vals[0] - _vals[1] # Take diff, as tiebreak
            
            tform_value = sum(_vals) - 0.1*_val_diff - 0.01*_val_diff_2

        else:
            tform_value = methods_tforms_prim[idx_prim][tform_key]

        psycho_value = tform_value
    elif psycho_ver == "morph_frac":
        # Morph between two base prims.

        # morph_info = D.shapesemantic_taskclass_cont_morph_extract_params(1113)
        
        # morph_info = 
        # [{'los': ('singleprims', 114, 9),
        # 'shape': ('arcdeep', 4, 4, 0),
        # 'frac': 0.6666666666666666,
        # 'flip_task1': False,
        # 'flip_task2': False,
        # 'frac_func': array([], shape=(0, 0), dtype=float64)},
        # {'los': ('singleprims', 114, 30),
        # 'shape': ('circle', 6, 1, 0),
        # 'frac': 0.6666666666666666,
        # 'flip_task1': False,
        # 'flip_task2': False,
        # 'frac_func': array([], shape=(0, 0), dtype=float64)}]

        assert np.all(np.isclose(morph_info[0]["frac"], morph_info[1]["frac"]))
        
        psycho_value = morph_info[0]["frac"].item()
    else:
        print(psycho_ver)
        assert False, "code it for this psycho_ver"

    return psycho_value

def psychogood_find_tasks_in_this_psycho_group_wrapper(D, los_within, psycho_params):
    """
    Good, Return all trials that are in the psycho group that includes this los_within, and
    with variation consistent with pyscho_params.
    PARAMS:
    - los_within, (str, int, int).
    - psycho_params, dict of params defining what the psycho variable is.
    NOTE: if None, then looks for identical task plans (allowing for variation in location)
    """
    df = D.Dat[D.Dat["los_info"] == los_within]
    if len(df)==0:
        print(los_within)
        assert False, "Cant find any trials with this los... you made mistake in entering params"

    ind = D.Dat[D.Dat["los_info"] == los_within].index[0]

    # (1) Make a template for a single case within the interp
    plan = D.taskclass_extract_low_level_plan(ind)
    
    # list_extra_tform_params = D.taskclass_extract_prims_extra_params_tforms(ind)
    # assert len(list_extra_tform_params)==1, "currently asusming is single stroke. allows taking first stroke, legacy code."
    # extra_tform_params = list_extra_tform_params[0]
    extra_tform_params = _extract_extra_tform_params_helper(D, ind)

    morph_info = D.shapesemantic_taskclass_cont_morph_extract_params(ind)
    template = psychogood_make_task_template_to_find_psycho_variants(plan, extra_tform_params, morph_info, psycho_params)

    # (2) Find all trials that match this template -- i.e. tasks within same psycho group
    res = []
    for ind in range(len(D.Dat)):
        plan_this = D.taskclass_extract_low_level_plan(ind)
        extra_tform_params_this = _extract_extra_tform_params_helper(D, ind)
        morph_info_this = D.shapesemantic_taskclass_cont_morph_extract_params(ind)
        template_this = psychogood_make_task_template_to_find_psycho_variants(plan_this, extra_tform_params_this, morph_info_this, psycho_params)

        if template == template_this:
            # Found a match to los_within.
            los_this = D.taskclass_extract_los_info(ind)

            if psycho_params is not None:
                psycho_value = psychogood_extract_psycho_var(plan_this, extra_tform_params_this, morph_info_this, psycho_params)
            else:
                psycho_value = np.nan

            res.append({
                "idx_dat":ind,
                "psycho_value":psycho_value,
                "los":los_this
            })
            # print(ind, " --- los:", los_this, " -- value:", psycho_value)
    
    return pd.DataFrame(res)

def psychogood_find_tasks_in_this_psycho_group_wrapper_manual_helper(D, example_los_1, example_los_2, psycho_params_try):
    """ 
    Helper to identify what params to input to manually enter required info
    for identifying psyho groups uisng find_tasks_in_this_psycho_group_wrapper. Does this by entering two
    example los that are within group, and a guess as params. 
    Determines whether the psycho params inputed successfully calls los1 and los1 as same group. If so, then copy and paster
    the output in to PARAMS> If Not ,then read what's printed, and updated psycho_params_try accordingly...

    READ the comments in code to see steps for using this.

    See 210506_.. notebook for example how to run this.

    PARAMS:
    - example_los_1, los (str, num, num), which is one of two los that are within this group for use in testing 
    these params.
    - example_los_2, los, the other that is also in the group.
    RETURNS:
    - Prints to console --> bool, whether these two los would be identified as being in the same psycho group,
    given these psycho_params. If they are then you can use these params in find_tasks_in_this_psycho_group_wrapper:
    -- los_within = <example_los_1> or <example_los_2>
    -- psycho_params = psycho_params_try
    """
    # [GOOD HELPER] To decode what psycho params to enter

    # (1) Input tasks that are in the same psycho group, such that they have one param difference, the param
    # you want to vary. Find these using your notes for this experiment, and findings tasks in the makeDrawTaskSets folder.
    # los1 = ("singleprims_psycho", 9, 1)
    # los2 = ("singleprims_psycho", 10, 1)
    los1 = example_los_1
    los2 = example_los_2

    # (2) Enter candidate psycho params. This picks out the specific one param that is expected to vary across los in this group.
    # Here, can enter and iterate, with goal that template1 == template2 below.
    psycho_params = psycho_params_try

    # Example ways to entee:
    
    # (i) Varying a prim-specific tform [structured morphs]
    # psycho_params = {
    #     "psycho_ver":"extra_tforms_each_prim",
    #     "idx_prim": 1, # the substroke index
        # "idx_prim": [0, 1], # take a hacky combination 0 and 1.
    #     "tform_key": "sx" # scale x
    # }
    # -- Example task:
    # Extra tform params:
    # {'tforms': {}, 'tforms_each_prim_p': [{}, [['sx', array(0.16666667)]]]}
    # Extra tform params:
    # {'tforms': {}, 'tforms_each_prim_p': [{}, [['sx', array(0.5)]]]}

    # (ii) Varying attach pt [structured morphs]
    # psycho_params = {
    #     "psycho_ver":"attachpt1_interp",
    #     "idx_motif_char_flex": 0, # Usuall dont change (if this is one "prim"), i.e,. this is the outer index.
    #     "idx_rel": 1, # This is the index into the relations, within each plan
    #     "exclude_extra_tform_values":False, # Turn this True if the morphs also have diff in extra tforms (each prim), and you want to be invariant over those
    # }
    # -- Example task:
    # (['motif_char_flex',
    #   ['direct_chunk_entry',
    #    [['line',
    #      ['prot',
    #       array(18.),
    #       array(1.),
    #       array([], shape=(0, 0), dtype=float64),
    #       array(0.)]],
    #     ['translate_xy', array(0.), ['center', 'center', array([0., 0.])]],
    #     ['line',
    #      ['prot',
    #       array(18.),
    #       array(2.),
    #       array([], shape=(0, 0), dtype=float64),
    #       array(0.)]],
    #     ['translate_xy',
    #      array(-1.),
    #      ['interp_end1_end2_17', 'end2', array([0., 0.])]]]]],
    #  ['translate_xy',
    #   array(0.),
    #   ['center_prim_sketchpad', 'center_prim_sketchpad', array([-1. ,  0.8])]])
    # Plan:
    # (['motif_char_flex',
    #   ['direct_chunk_entry',
    #    [['line',
    #      ['prot',
    #       array(18.),
    #       array(1.),
    #       array([], shape=(0, 0), dtype=float64),
    #       array(0.)]],
    #     ['translate_xy', array(0.), ['center', 'center', array([0., 0.])]],
    #     ['line',
    #      ['prot',
    #       array(18.),
    #       array(2.),
    #       array([], shape=(0, 0), dtype=float64),
    #       array(0.)]],
    #     ['translate_xy',
    #      array(-1.),
    #      ['interp_end1_end2_50', 'end2', array([0., 0.])]]]]],
    #  ['translate_xy',
    #   array(0.),
    #   ['center_prim_sketchpad', 'center_prim_sketchpad', array([-1. ,  0.8])]])

    # (iii) Varying attach pt [structured morphs] and also varying extra tforms, but you want to be invariant over tforms,
    # so you convert tforms to their sign (I did this when + and - tforms indicated two different psycho groups)
    # psycho_params = {
    #     "psycho_ver":"attachpt1_interp",
    #     "idx_motif_char_flex": 0, # Usuall dont change (if this is one "prim"), i.e,. this is the outer index.
    #     "idx_rel": 1, # This is the index into the relations, within each plan
    #     "exclude_extra_tform_values":False,
    #     "exclude_interp_values":False,
    #     "convert_extra_tforms_to_sign":True 
    # }    

    # (iv) Morphing between two tasks (e.g,, arc <--> circle)
    # psycho_params = {
    #     "psycho_ver":"morph_frac",
    #     "replace_morph_params_with_binary_whether_exists":True, # Turn True so that is not affected by minute differences in params.
    # }


    # (3) Extract task infomraiton
    print("----- Printing information about each task")

    # Task 1
    ind = D.Dat[D.Dat["los_info"] == los1].index[0]
    plan1 = D.taskclass_extract_low_level_plan(ind)
    # list_extra_tform_params1 = D.taskclass_extract_prims_extra_params_tforms(ind)
    # assert len(list_extra_tform_params1)==1
    # extra_tform_params1 = list_extra_tform_params1[0]
    extra_tform_params1 = _extract_extra_tform_params_helper(D, ind)

    morph_info1 = D.shapesemantic_taskclass_cont_morph_extract_params(ind)
    template1 = psychogood_make_task_template_to_find_psycho_variants(plan1, extra_tform_params1, morph_info1, psycho_params)
    psycho_var_1 = psychogood_extract_psycho_var(plan1, extra_tform_params1, morph_info1, psycho_params)

    print("= Task 1")
    print("  Plan:")
    display(plan1)
    print("  Extra tform params:")
    print(extra_tform_params1)
    print("  Morph info:")
    print(morph_info1)
    print("  Psycho var:")
    print(psycho_var_1)

    # Task 2
    ind = D.Dat[D.Dat["los_info"] == los2].index[0]
    plan2 = D.taskclass_extract_low_level_plan(ind)
    # list_extra_tform_params2 = D.taskclass_extract_prims_extra_params_tforms(ind)
    # assert len(list_extra_tform_params2)==1
    # extra_tform_params2 = list_extra_tform_params2[0]
    extra_tform_params2 = _extract_extra_tform_params_helper(D, ind)

    morph_info2 = D.shapesemantic_taskclass_cont_morph_extract_params(ind)
    template2 = psychogood_make_task_template_to_find_psycho_variants(plan2, extra_tform_params2, morph_info2, psycho_params)
    psycho_var_2 = psychogood_extract_psycho_var(plan2, extra_tform_params2, morph_info2, psycho_params)

    print("= Task 2")
    print("  Plan:")
    display(plan2)
    print("  Extra tform params:")
    print(extra_tform_params2)
    print("  Morph info:")
    print(morph_info2)
    print("  Psycho var:")
    print(psycho_var_2)

    ### TEST whether these tasks are considered same psycho group
    print(" ")
    print("------ Results from comparing these two tasks:")
    same_psycho_group = template1 == template2
    print("Were tasks identified as same psycho group? ", same_psycho_group)

    if not same_psycho_group:
        print("Differences between task represnetations (Use these to determine how to change psycho params, if needed):")
        print("len of templates: ", len(template1), len(template2))
        for i, (x, y) in enumerate(zip(template1, template2)):
            if not x==y:
                print(f"idx in template = {i} -- values: {x}, {y}")
    
    # print(f"los_within = {example_los_1}")
    # print(f"psycho_params = {psycho_params}")
    print("If this is same group, then copy and paste the following in to PARAMS:")

    print(f"'los_base_1': ('singleprims_psycho', FILL_IN, FILL_IN), 'los_base_2': ('singleprims_psycho', FILL_IN, FILL_IN), 'los_within': {example_los_1}, 'psycho_params': {psycho_params}")

    
def psychogood_decide_if_tasks_are_ambiguous(DSmorphsets, PLOT_SAVEDIR=None):
    """
    Decide whether task is ambiguous, in that it is matched with eitehr base 1 or base 2 on different
    trials. 
    
            
    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    
    def find_morphset_morphidx(DSmorphsets, morphset, idx_in_morphset):
        return DSmorphsets.Dat[
            (DSmorphsets.Dat["morph_set_idx"] == morphset) & 
            (DSmorphsets.Dat["morph_idxcode_within_set"] == idx_in_morphset)].index.tolist()

    grpdict = grouping_append_and_return_inner_items_good(DSmorphsets.Dat, ["morph_set_idx", "morph_idxcode_within_set"])
    assignments = ["null" for _ in range(len(DSmorphsets.Dat))]
    min_trials = 5
    min_frac = 0.3

    for (morphset, idx_in_morphset), _inds_this in grpdict.items():

        # if PLOT_SAVEDIR is not None:
        #     savedir = f"{PLOT_SAVEDIR}/morphset={morphset}--idx_in_morphset={idx_in_morphset}"
        #     os.makedirs(savedir, exist_ok=True)
        # else:
        #     savedir = None
        
        # These are the base prims (on endpoints)
        if idx_in_morphset==0:
            for i in _inds_this:
                assignments[i] = "base1"
            continue 
        if idx_in_morphset==99:
            for i in _inds_this:
                assignments[i] = "base2"
            continue 
            
        # Find cases with variable behavior ()

        # across trials, get its similarity to each of the endpoint shapes
        INDS_THIS = find_morphset_morphidx(DSmorphsets, morphset, idx_in_morphset)
        assert INDS_THIS == _inds_this
        
        if len(INDS_THIS)<=min_trials:
            for i in _inds_this:
                assignments[i] = "not_enough_trials"
            continue

        #################### GOOD, enough data to run
        inds_base_1 = find_morphset_morphidx(DSmorphsets, morphset, 0)
        inds_base_2 = find_morphset_morphidx(DSmorphsets, morphset, 99)

        # Collect all stroke data
        strokes = DSmorphsets.extract_strokes(inds=INDS_THIS)
        strokes_base1 = DSmorphsets.extract_strokes(inds=inds_base_1)
        strokes_base2 = DSmorphsets.extract_strokes(inds=inds_base_2)

        # plot each stroke in order
        if False:
            DSmorphsets.plot_multiple_strok(strokes)
            DSmorphsets.plot_multiple_strok(strokes_base1)
            DSmorphsets.plot_multiple_strok(strokes_base2)

        # Get the average stroke for the base strokes
        from pythonlib.tools.stroketools import strokes_average
        strokmean_base1, _ = strokes_average(strokes_base1, center_at_onset=True)
        strokmean_base2, _ = strokes_average(strokes_base2, center_at_onset=True)

        if False:
            DSmorphsets.plot_multiple_strok([strokmean_base1, strokmean_base2])

        # DSmorphsets._dist_strok_pair(strokes[0], strokes[1], recenter_to_onset=True)
        Cl = DSmorphsets._cluster_compute_sim_matrix(strokes, [strokmean_base1, strokmean_base2], "dtw_vels_2d", return_as_Clusters=True)
        if PLOT_SAVEDIR:
            fig, X, labels_col, labels_row, ax = Cl.plot_heatmap_data()
            savefig(fig, f"{PLOT_SAVEDIR}/morphset={morphset}--idx_in_morphset={idx_in_morphset}--simmat_heatmap.pdf")

        # Assign whether is base1 or base 2 aligned
        indmax = np.argmax(Cl.Xinput, axis=1)
        inds_assigned_to_base1 = [x[0] for x in np.argwhere(indmax==0).tolist()]
        inds_assigned_to_base2 = [x[0] for x in np.argwhere(indmax==1).tolist()]

            
        # Plot 
        if PLOT_SAVEDIR:
            fig, axes = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10,10))

            ax = axes[0][0]
            DSmorphsets.plot_multiple_strok([strokes[i] for i in inds_assigned_to_base1], ax=ax, alpha=0.2)
            ax.set_title("strokes assigned to base 1")

            ax = axes[0][1]
            DSmorphsets.plot_multiple_strok([strokes[i] for i in inds_assigned_to_base2], ax=ax, alpha=0.2)
            ax.set_title("strokes assigned to base 2")

            ax = axes[1][0]
            DSmorphsets.plot_multiple_strok(strokes_base1, ax=ax, alpha=0.2)
            ax.set_title("strokes base 1")

            ax = axes[1][1]
            DSmorphsets.plot_multiple_strok(strokes_base2, ax=ax, alpha=0.2)
            ax.set_title("strokes base 2")
            savefig(fig, f"{PLOT_SAVEDIR}/morphset={morphset}--idx_in_morphset={idx_in_morphset}--drawings_final_assignments.pdf")

        # Save the assignemnets 
        # - call this ambiguous only if it shows bimodal matching to base prims.
        bimodal_by_frac = min([len(inds_assigned_to_base1), len(inds_assigned_to_base2)])/len(strokes) >= min_frac
        bimodal_by_count = min([len(inds_assigned_to_base1), len(inds_assigned_to_base2)])>=2


        enough_trials = len(strokes)>=min_trials
        
        ### Are strokes variable across trials.
        if False: # stop doing this, as it fails for circles, and also you don't have to flip to be ambiguous, so thistt
            # throws too much stuff out.
            flips = [DSmorphsets.stroke_shape_align_flip_vs_task(ind) for ind in INDS_THIS]
            ntot = len(flips)
            nflips = sum(flips)
            nnotflips = ntot - nflips
            flips_are_variable = min([nflips, nnotflips])>=2
            
            is_ambigious = (bimodal_by_frac or bimodal_by_count) and enough_trials and flips_are_variable
        else:
            print("TODO: best method is to test bimodality in distnaces to base prims..")
            is_ambigious = (bimodal_by_frac or bimodal_by_count) and enough_trials

        # - for each index give its assigned 
        if is_ambigious:
            # idxs = [INDS_THIS[i] for i in inds_assigned_to_base1]
            for i in inds_assigned_to_base1:
                assert assignments[INDS_THIS[i]] == "null", "already assigned,shouldnt be possible"
                assignments[INDS_THIS[i]] = "ambig_base1"

            # idxs = [INDS_THIS[i] for i in inds_assigned_to_base1]
            for i in inds_assigned_to_base2:
                assert assignments[INDS_THIS[i]] == "null", "already assigned,shouldnt be possible"
                assignments[INDS_THIS[i]] = "ambig_base2"
        else:
            # then still assign to base 1 or 2
            if len(inds_assigned_to_base1) > len(inds_assigned_to_base2):
                assigned_base = "not_ambig_base1"
            else:
                assigned_base = "not_ambig_base2"

            for i in inds_assigned_to_base1:
                assert assignments[INDS_THIS[i]] == "null", "already assigned,shouldnt be possible"
                assignments[INDS_THIS[i]] = assigned_base

            # idxs = [INDS_THIS[i] for i in inds_assigned_to_base1]
            for i in inds_assigned_to_base2:
                assert assignments[INDS_THIS[i]] == "null", "already assigned,shouldnt be possible"
                assignments[INDS_THIS[i]] = assigned_base
            
        print("morphset, idx_in_morphset:", morphset, idx_in_morphset)
        print(f"inds assigned to base1: {inds_assigned_to_base1}, to base2: {inds_assigned_to_base2}")
        print("is_ambigious:", is_ambigious)


        plt.close("all")    

    assert sum([a=="null" for a in assignments])==0, "not sure why, failed to assign some trials."

    # Place assignments into DS
    DSmorphsets.Dat["morph_assigned_to_which_base"] = assignments

    return assignments

def _extract_extra_tform_params_helper(D, ind):
    """
    Befucae of legacy code, expects dict, but now taskclass_extract_prims_extra_params_tforms returns
    list of dict. So here takes the first item, assertiung it is length 1 (i.e, single ;prims tasks)
    """

    list_extra_tform_params_this = D.taskclass_extract_prims_extra_params_tforms(ind)
    if list_extra_tform_params_this is None:
        return None
    else:
        assert len(list_extra_tform_params_this)==1, "currently asusming is single stroke. allows taking first stroke, legacy code."
        extra_tform_params_this = list_extra_tform_params_this[0]
        return extra_tform_params_this
    
def psychogood_prepare_for_neural_analy(D, DSmorphsets):
    """
    Helper to extract mappings useful for neural analysis, did this during decode_moment...
    For useful things for plots, etc.

    NOTE:
    - should first preprocess to just one beh stroke per trial, or else will fail.
    """

    assert all(DSmorphsets.Dat["stroke_index"])==0, "or else teh following will fail -- should first preprocess to just one stroke per trial"

    # Code: given a (morphset, idx), get all sets of trialcodes
    def find_morphset_morphidx(DSmorphsets, morphset, idx_in_morphset, return_as_trialcodes=True):
        """ Return indices in DSmorphsets that match morphset and idx_in_morphset
        """
        from pythonlib.tools.pandastools import _check_index_reseted
        # _check_index_reseted(DSmorphsets.Dat)
        inds = DSmorphsets.Dat[
            (DSmorphsets.Dat["morph_set_idx"] == morphset) & 
            (DSmorphsets.Dat["morph_idxcode_within_set"] == idx_in_morphset)].index.tolist()
        
        if return_as_trialcodes:
            return DSmorphsets.Dat.iloc[inds]["trialcode"].tolist()
        else:
            return inds


    from pythonlib.tools.pandastools import _check_index_reseted
    _check_index_reseted(DSmorphsets.Dat)

    # PLOT_SAVEDIR = "/tmp"
    PLOT_SAVEDIR = None
    assignments = psychogood_decide_if_tasks_are_ambiguous(DSmorphsets, PLOT_SAVEDIR)

    from pythonlib.tools.pandastools import grouping_print_n_samples


    # for morphset in sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist()):
    #     df = DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"] == morphset]

    #     from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    #     grouping_plot_n_samples_conjunction_heatmap(df, "los_info", "morph_idxcode_within_set", ["morph_set_idx"])

    ##### Condition DSmorphsets
    # Each base shape is defined by (morphset, idx_within)  -- i.e.e the "shape" variable is wrong - its the task component.
    from pythonlib.tools.pandastools import append_col_with_grp_index
    DSmorphsets.Dat = append_col_with_grp_index(DSmorphsets.Dat, ["morph_set_idx", "morph_idxcode_within_set"], "morph_id_both", use_strings=False)
    DSmorphsets.Dat = append_col_with_grp_index(DSmorphsets.Dat, ["morph_set_idx", "morph_idxcode_within_set"], "morph_id_both_str", use_strings=True)
    ##### Generate various mappings (beh features)
    # Generate maps of this kind -- MAP: tc --> stuff [GOOD]

    import numpy as np

    # Map from trialcode to whether is base or morph
    map_tc_to_morph_info = {}
    for tc in D.Dat["trialcode"]:
        if tc not in DSmorphsets.Dat["trialcode"].tolist():
            map_tc_to_morph_info[tc] = "no_exist"
            print("This tc in D.Dat but not in DSmorphsets...", tc)
        else:
            tmp = DSmorphsets.Dat[DSmorphsets.Dat["trialcode"] == tc]

            morph_is_morphed_list = tmp["morph_is_morphed"].unique().tolist()
            assert len(morph_is_morphed_list)==1

            if morph_is_morphed_list[0]:
                # Is morph -- Then this is not base prim.
                map_tc_to_morph_info[tc] = "morphed"
            else:
                # THis is base prim. But could be participant in multiple morhph sets, so 
                # Just give it the first id
                mid = tmp["morph_id_both_str"].unique().tolist()
                map_tc_to_morph_info[tc] = mid[0]

    # Generate maps of this kind -- morphset --> stuff

    # For base prims, map from (morphset) --> (base1, base2) where base1 and base2 are the codes used in decoder.\

    # {0: ('0|0', '0|99'),
    #  1: ('0|0', '0|99'),
    #  2: ('2|0', '2|99'),
    #  3: ('2|0', '3|99'),
    #  4: ('4|0', '0|99'),
    #  5: ('0|99', '5|99'),
    #  6: ('6|0', '4|0')}

    list_morphset = DSmorphsets.Dat["morph_set_idx"].unique()
    map_morphset_to_basemorphinfo = {}
    for morphset in list_morphset:

        trialcodes = find_morphset_morphidx(DSmorphsets, morphset, 0)
        mis = [map_tc_to_morph_info[tc] for tc in trialcodes]
        assert len(set(mis))==1
        base1_mi = mis[0]
        
        trialcodes = find_morphset_morphidx(DSmorphsets, morphset, 99)
        mis = [map_tc_to_morph_info[tc] for tc in trialcodes]
        assert len(set(mis))==1
        base2_mi = mis[0]
        
        map_morphset_to_basemorphinfo[morphset] = (base1_mi, base2_mi)
    # Generate maps of this kind -- MAP: from (tc, morphset) --> stuff

    map_tcmorphset_to_idxmorph = {} # (tc, morphset) --> idx_in_morphset | "not_in_set"
    map_tcmorphset_to_info = {} # (tc, morphset) --> (amibig, base1, base2)

    for i, row in DSmorphsets.Dat.iterrows():
        tc = row["trialcode"]
        morphset = row["morph_set_idx"]
        morph_idxcode_within_set = row["morph_idxcode_within_set"]

        # (1) 
        assert (tc, morphset) not in map_tcmorphset_to_idxmorph,  "probably multiple strokes on this trial..."
        map_tcmorphset_to_idxmorph[(tc, morphset)] = morph_idxcode_within_set

        # (2)
        if (tc, morphset) in map_tcmorphset_to_info:
            print(tc, morphset, row["morph_idxcode_within_set"], row["stroke_index"])
            assert False, "probably multiple strokes on this trial..."
        else:
            if False:
                # Get its base prims
                _inds = find_morphset_morphidx(DSmorphsets, morphset, 0, False)
                _tmp = DSmorphsets.Dat.iloc[_inds]["morph_id_both_str"].unique()
                assert len(_tmp)==1
                base1_mi = _tmp[0]

                _inds = find_morphset_morphidx(DSmorphsets, morphset, 99, False)
                _tmp = DSmorphsets.Dat.iloc[_inds]["morph_id_both_str"].unique()
                assert len(_tmp)==1
                base2_mi = _tmp[0]
            else:
                base1_mi = map_morphset_to_basemorphinfo[morphset][0]
                base2_mi = map_morphset_to_basemorphinfo[morphset][1]

            map_tcmorphset_to_info[(tc, morphset)] = (row["morph_assigned_to_which_base"], base1_mi, base2_mi)
            
    # Fill in the missing ones
    list_morphset = DSmorphsets.Dat["morph_set_idx"].unique().tolist()
    list_tc = D.Dat["trialcode"].tolist()
    for morphset in list_morphset:
        for tc in list_tc:
            if (tc, morphset) not in map_tcmorphset_to_idxmorph:
                map_tcmorphset_to_idxmorph[(tc, morphset)] = "not_in_set"
    # Generate maps of this kind -- (morphset, idx within) --> stuff

    map_morphsetidx_to_assignedbase_or_ambig = {}
    # map_morphsetidx_to_assignedbase = {}
    for i, row in DSmorphsets.Dat.iterrows():
        morphset = row["morph_set_idx"]
        morph_idxcode_within_set = row["morph_idxcode_within_set"]  

        key = (morphset, morph_idxcode_within_set)

        # Convert to avalue that is same across trials.
        if row["morph_assigned_to_which_base"] in ["ambig_base2", "ambig_base1"]:
            value = "is_ambig"
        else:
            value = row["morph_assigned_to_which_base"]

        if key in map_morphsetidx_to_assignedbase_or_ambig:
            assert map_morphsetidx_to_assignedbase_or_ambig[key] == value
        else:
            map_morphsetidx_to_assignedbase_or_ambig[key] = value

        # if key in map_morphsetidx_to_assignedbase:
        #     assert map_morphsetidx_to_assignedbase[key] == row["morph_assigned_to_which_base"]
        # else:
        #     map_morphsetidx_to_assignedbase[key] = row["morph_assigned_to_which_base"]

    map_morphsetidx_to_assignedbase_or_ambig = {k:map_morphsetidx_to_assignedbase_or_ambig[k] for k in sorted(map_morphsetidx_to_assignedbase_or_ambig.keys())}
    # map_morphsetidx_to_assignedbase = {k:map_morphsetidx_to_assignedbase[k] for k in sorted(map_morphsetidx_to_assignedbase.keys())}


    # list_morphset = sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist())    
    # map_morphsetidx_to_assignedbase = {}
    # for morphset in list_morphset:
    #     has_switched = False
    #     list_idx = sorted(DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"] == morphset]["morph_idxcode_within_set"].unique().tolist())    
    #     for idx in list_idx:
    #         k = (morphset, idx)

    #         if map_morphsetidx_to_isambig[k]:
    #             has_switched = True
    #             assigned_base = "is_ambig"
    #         else:
    #             if has_switched or idx==99:
    #                 assigned_base = "base2"
    #             else:
    #                 assigned_base = "base1"
            
    #         map_morphsetidx_to_assignedbase[k] = assigned_base

    #         print(idx, map_morphsetidx_to_isambig[k], assigned_base)
        
    # Map from trialcode to morph information
    # NOTE - this doesnt work, since a given TC can be in multiple morph sets
    map_tc_to_morph_status = {}
    ct_missing = 0
    ct_present = 0
    for tc in D.Dat["trialcode"]:
        if tc not in DSmorphsets.Dat["trialcode"].tolist():
            print("This tc in D.Dat but not in DSmorphsets...", tc)
            ct_missing += 1
        else:
            tmp = DSmorphsets.Dat[DSmorphsets.Dat["trialcode"] == tc]["morph_is_morphed"].unique().tolist()
            assert len(tmp)==1
            ct_present += 1

            # map_tc_to_morph_info[tc] = (tmp["morph_set_idx"].values[0], tmp["morph_idxcode_within_set"].values[0], tmp["morph_assigned_to_which_base"].values[0], tmp["morph_is_morphed"].values[0])

    print("Missing / got:", ct_missing, ct_present)
    from pythonlib.tools.pandastools import grouping_print_n_samples
    grouping_print_n_samples(DSmorphsets.Dat, ["shape", "morph_idxcode_within_set", "morph_is_morphed"])

    return DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status


def psychogood_plot_drawings_morphsets(DSmorphsets, savedir, n_iter=2):
    """ 
    General -- plot grid of beh and task images, organized as (idx_within, morphset), useful for all
    psycho tasks
     
    One large figure for all dataset
    """

    for _iter in range(n_iter):

        figbeh, figtask = DSmorphsets.plotshape_row_col_vs_othervar("morph_idxcode_within_set", "morph_set_idx", 
            n_examples_per_sublot=8, plot_task=True, ver_behtask="task_entire")
        savefig(figbeh, f"{savedir}/morph_idxcode_within_set-vs-morph_set_idx-beh-iter{_iter}.pdf")
        savefig(figtask, f"{savedir}/morph_idxcode_within_set-vs-morph_set_idx-task-iter{_iter}.pdf")
        plt.close("all")


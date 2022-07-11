""" 
7/7/22 - for extracting sequences from (e.g) n prims in grid, based on things like:
- diff prims, but same sequence and location.
Assumes clean tasks (prims in grid)

Works using datsegs (either aligned to beh or task strokes)

See notebook: analy_spatial_prims_motifs_220707
"""

VARIABLES_KEEP = ["shape_oriented", "gridloc"]

def generate_dict_of_all_used_motifs(D, nprims=2, 
    variables_keep = VARIABLES_KEEP,
    WHICH_DATSEGS = "task", shapes_to_ignore=None):
    """ Generate dict holding all motifs used in this dataset.
    PARAMS:
    - D, Dataset
    - nprims, number of prims in motif. Currently only coded for 2
    - variables_keep, list of str keys into datsegs. order of this list
    defines the feature tuple for each token. e..g, ["shape_oriented", "gridloc"]
    - WHICH_DATSEGS, str, {'beh', 'task'}, defines which datsegs to use, ether
    aligned to beh or task.
    - shapes_to_ignore, list of str of shapes_oriented, ignores any motifs that inlcude this
    shape. None to not apply this
    RETURNS:
    - motifs_all_dict, dict, where keys are motifs (tuples of tokens) and itmes are 
    list of indices, where and index is (trial, starting strokenum)
    """

    assert nprims==2, "not coded for other yet"
    motifs_all_dict = {}
    def prune_token_variables(token, variables_keep):
        """
        keep only the variables you care about.
        token = datsegs[i]
        RETURNS:
        tuple of items, in order of variables_keep
        """
    #     token = {k:v for k, v in token.items() if k in variables_keep}
        token_tuple = tuple([token[k] for k in variables_keep])
        return token_tuple

    for ind in range(len(D.Dat)):
        Beh = D.Dat.iloc[ind]["BehClass"]
        
        primlist, datsegs_behlength, datsegs_tasklength = D.behclass_extract_beh_and_task(ind)
        if WHICH_DATSEGS=="task":
            datsegs = datsegs_tasklength
        elif WHICH_DATSEGS=="beh":
            datsegs = datsegs_behlength
        else:
            assert False
        
        for i in range(len(datsegs)-1):
            motif = datsegs[i:i+2]
            
    #         # MOtifs that should obviously not keep
    #         if motif[0][1]
            
            # save this motif
            # use the motif as a key (hash it)
            # just keep the variables that you care about
            motif = tuple([prune_token_variables(token, variables_keep) for token in motif])
            
            # This is hashable. use as key in dict
            index = (ind, i)
            if motif in motifs_all_dict.keys():
                motifs_all_dict[motif].append(index)
            else:
                motifs_all_dict[motif] = [index]

    # prune motifs
    if shapes_to_ignore is not None:
        # First, check you didnt make mistake netering shapes
        list_shapes = extract_list_shapes_loc(motifs_all_dict)[0]
        for sh in shapes_to_ignore:
            if sh not in list_shapes:
                print(sh)
                print(list_shapes)
                assert False

        # Second, clean up the motifs
        def _is_bad(motif):
            for token in motif:
                if token[0] in shapes_to_ignore:
                    return True
            return False

        motifs_all_dict_new = {}
        for motif, v in motifs_all_dict.items():
            if _is_bad(motif):
                print("removing motif: ", motif)
                continue
            else:
                motifs_all_dict_new[motif] = v
        motifs_all_dict = motifs_all_dict_new

    print("Found this many motifs: ", len(motifs_all_dict))
    sorted(list(motifs_all_dict.keys()))
    print("This many instances per motif: ", [len(v) for k, v in motifs_all_dict.items()])
    return motifs_all_dict

def extract_list_shapes_loc(motifs_all_dict):
    """ get the list of all unqiue shapes and locations
    """
    # 1) Get list of all unique shapes and locations in the dataset.
    list_shapes = []
    list_locs = []
    for motif in motifs_all_dict.keys():
        for token in motif:
            loc = token[1]
            shape = token[0]
            list_locs.append(loc)
            list_shapes.append(shape)
    list_locs = sorted(list(set(list_locs)))
    list_shapes = sorted(list(set(list_shapes)))
    return list_shapes, list_locs

def generate_motifgroup_data(motifs_all_dict, 
        list_group_kind= ["same", "diff_sequence", "diff_location", "diff_prims"]):
    """ Main extraction of computed data. For each way of grouping (e.g., same prim, same seq, diff location),
    pull out each "motifgroup" (which is a set of all motifs which are related under this grouping), and
    for each motifgroup save the trials that have these motifs. 
    RETURNS:
    - DatGroups, dict where keys are each kind of grouping (e.g, diff_sequence), and items 
    are lists of dicts, where each dict is a motifgroup and its data.
    """

    list_shapes, list_locs = extract_list_shapes_loc(motifs_all_dict)
    
    # 2) Extract data
    DatGroups = {}
    for group_kind in list_group_kind:

        def get_all_possible_motifs_for_this_motifs_group(motif):
            """
            RETURNS:
            - motif_group, tuple of tples, sorted, where one of the
            tuples if motif, and the rest are the other motifs in this 
            group for this motif. This works becauseof symmatry -i.e,
            each motif is only in one possible group
            """

            list_motif = []
            if group_kind=="diff_sequence":
                # Get motif and its reverse
                list_motif = [motif, motif[::-1]]
            elif group_kind=="diff_location":
                # get this sequence of prims, at all possibel locations
                shape1 = motif[0][0]
                shape2 = motif[1][0]
                for i in range(len(list_locs)):
                    for ii in range(len(list_locs)):
                        if i!=ii: # different locations
                            # This is a unique motif
                            loc1 = list_locs[i]
                            loc2 = list_locs[ii]
                            motif = ((shape1, loc1), (shape2, loc2))
                            list_motif.append(motif)
            elif group_kind=="diff_prims":
                # Get this sequence of locations, all possible prim location pairs.
                loc1 = motif[0][1]
                loc2 = motif[1][1]
                for shape1 in list_shapes:
                    for shape2 in list_shapes:
                        # This is a unique motif
                        motif = ((shape1, loc1), (shape2, loc2))
                        list_motif.append(motif)
            elif group_kind=="same":
                # simply: motifs should be identical
                list_motif = [motif]
            else:
                assert False

            # assert len(list_motif)>1, "even if you want same motif, just include it twice."
            # Convert to sorted tuple
            list_motif = sorted(list_motif)
            motif_group = tuple(list_motif)
            return motif_group

        def count_num_instances_this_motif(motif):
            # How often did this motif occur in dataset?
            if motif in motifs_all_dict.keys():
                return len(motifs_all_dict[motif])
            else:
                return 0

        # Got thru all motifs. check which group it is in
        dict_motifgroups_all = {}
        list_motifs_iter = list(motifs_all_dict.keys())
        for motif in list_motifs_iter:

            motif_group = get_all_possible_motifs_for_this_motifs_group(motif)

            if motif_group in dict_motifgroups_all.keys():
                # Then skip, since this already done
                continue
            else:
                # if group_kind=="same":
                #     print(motif_group)
                #     assert False
                # Then go thru each motif in this group, and see how many instances there are
                list_n = [count_num_instances_this_motif(motif_this) for motif_this in motif_group]    
                # if motif_group==((('V-4-0', (1, 1)), ('Lcentered-3-0', (0, 0))), ):
                #     print(list_n)
                #     assert False

                dict_motifgroups_all[motif_group] = list_n

        
        #  Convert to a list of dicts
        dat = []
        for k, v in dict_motifgroups_all.items():
            n_unique_motifs_with_at_least_one_trial = sum([vv>0 for vv in v])
            n_trials_per_motif = v
            n_trials_summed_across_motifs = sum(v)
            dat.append({
                "n_unique_motifs_with_at_least_one_trial":n_unique_motifs_with_at_least_one_trial,
                "key":k, # motifgroup
                "n_trials_per_motif":n_trials_per_motif,
                "n_trials_summed_across_motifs":n_trials_summed_across_motifs
            })
            # print({
            #     "n_unique_motifs_with_at_least_one_trial":n_unique_motifs_with_at_least_one_trial,
            #     "key":k, # motifgroup
            #     "n_trials_per_motif":n_trials_per_motif,
            #     "n_trials_summed_across_motifs":n_trials_summed_across_motifs
            # })
            # assert False

        del dict_motifgroups_all # inefficient code, went thru this intermediate
            
        # sort motifgroups in order from most to least cases 
        dat = sorted(dat, key=lambda x:(x["n_unique_motifs_with_at_least_one_trial"], x["n_trials_summed_across_motifs"]), reverse=True)
        DatGroups[group_kind] = dat

    # Prune, only keep motifgroups that have enough trials to actually use these for an 
    # experiment.
    for which_group in list_group_kind:
        # 1) only keep cases with at least two unique motifs.
        if which_group!="same":
            list_dat = []
            for Dat in DatGroups[which_group]:
                if Dat["n_unique_motifs_with_at_least_one_trial"]>1:
                    list_dat.append(Dat)
            DatGroups[which_group] = list_dat
        
        # 2) Only keep cases with >1 trial
        list_dat = []
        for Dat in DatGroups[which_group]:
            if Dat["n_trials_summed_across_motifs"]>1:
                list_dat.append(Dat)
        DatGroups[which_group] = list_dat



    return DatGroups, list_group_kind
        

def extract_motifgroup_dict(DatGroups, which_group, motifgroup):
    """ Help to extract a single data dict, for this motifgroup
    - DatGroups, dict, where each key is a way of grouping (e..g, "diff_sequence") and
    each item is DAT the data (list of dicts, each dict corresponding to one group of 
    motifs (motifgroup))
    - which_group, string, indexes DatGroups
    - motifgroup, either:
    --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
    --- tuple (motif), tuple of motifs, where motif is a tuple of tokens, where token is
    a tuple of features (e..g, (circle, [0,0]) for (shape, location)
    This is a group that indexes DAT
    RETURNS:
    - datdict, a dict for this motifgroup
    Or: None if cant find it
    """

    dat = DatGroups[which_group]
    
    # Get DAT
    if isinstance(motifgroup, int):
        # Then get in order of mostcases --> leastcases
        # Assumes it is already sorted
        if motifgroup>=len(dat):
            return None, None
        datdict = dat[motifgroup]
    else:
        # Then search for this motifgroup
        datdict = [d for d in dat if d["key"]==motifgroup][0]

    motifgroup = datdict["key"]
    return datdict, motifgroup



def get_all_motifs_found_for_this_motifgroup(DatGroups, which_group, motifgroup):
    """
    PARAMS:
    - DatGroups, dict, where each key is a way of grouping (e..g, "diff_sequence") and
    each item is DAT the data (list of dicts, each dict corresponding to one group of 
    motifs)
    - which_group, string, indexes DatGroups
    - motifgroup, either:
    --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
    --- tuple (motif), see above.tuple, a group that indexes DAT
    RETURNS:
    - list_motifs, list of motifs for which >0 cases found, 
    in this motif group
    - motifgroup, the tuple holding all possible motifs
    - list_tokens, list of unique tokens across used motifs
    - list_shapes, list of unique shapes across used motifs
    """
    
    datdict, motifgroup = extract_motifgroup_dict(DatGroups, which_group, motifgroup)

    if datdict is None:
        return None, None

    # Get motifs found in this group
    list_motifs = []
    for motif, ncases in zip(datdict["key"], datdict["n_trials_per_motif"]):
        if ncases>0:
            list_motifs.append(motif)

    # tokens and shapes
    list_tokens = list(set([token for motif in list_motifs for token in motif]))
    list_shapes = list(set([token[0] for token in list_tokens]))

    return list_motifs, motifgroup, list_tokens, list_shapes
    
def get_inds_this_motif(motifs_all_dict, motif, random_subset=None):
    """ REturn list of indices that have this motif
    PARAMS:
    - motifs_all_dict, dict where keys are motifs, each vals are list of index where 
    this is found in dataset, where index is (trialnum, strokenum start)
    - random_subset, either None or int, if the latter, then extracts maximum this many trials
    RETURNS:
    - list_idxs, each is (trial, strokenum)
    """
    assert motif in motifs_all_dict.keys(), "asking for motif that was never detected for this dataset..."
    list_idxs = motifs_all_dict[motif]

    # Take random subset
    if random_subset and len(list_idxs)>random_subset:
        import random
        list_idxs = random.sample(list_idxs, random_subset)

    # break out trials and strokenums
    return list_idxs
    
def get_inds_this_motifgroup(DatGroups, which_group, motifs_all_dict, motifgroup,
        ntrials_per_motif=None):
    """ For this motifgroup, find all indices across all motifs
    that were found within this group.
    PARAMS;
    - DatGroups, see avbove
    - which_group, str, see above
    - motifs_all_dict, see above
    - motifgroup, either:
    --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
    --- tuple (motif), see above.
    - ntrials_per_motif, int, if not None, then only keeps this many trials per motif, i.e., if any
    motifs have > this many trials, picks random subset
    RETURNS:
    - list_trials, list of ints into D.Dat
    - list_strokenums, list of strokenums taht start each motif instance,
    paired with list_trials
    - list_motifs_all, aligned with the above, the motif for each.
    """
        
    list_motifs, motifgroup = get_all_motifs_found_for_this_motifgroup(DatGroups, which_group, motifgroup)[:2]
    # for each motif, extract trials

    if list_motifs is None:
        # Then this motifgroup doesnt exist?
        return None, None, None

    inds_idxs_all = []
    inds_strokenums_all =[]
    list_motifs_all = []
    for motif in list_motifs:
        inds_idxs = get_inds_this_motif(motifs_all_dict, motif, 
            random_subset=ntrials_per_motif)
        inds_idxs_all.extend(inds_idxs)
        for _ in inds_idxs:
            list_motifs_all.append(motif)

    return inds_idxs_all, list_motifs_all, motifgroup

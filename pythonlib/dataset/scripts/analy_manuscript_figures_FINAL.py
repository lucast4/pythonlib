import pickle
import os


SAVEDIR_ALL = "/lemur2/lucas/analyses/manuscripts/1_action_symbols/REPRODUCED_FIGURES"


def fig2k_cluster_label(animal, date, SAVEDIR_THIS, WHICH_BASIS_SET, which_shapes, HACK=False):
    """
    Extract all cluster labels, starting from DS (raw strokes) and save
    """
    from pythonlib.dataset.dataset_analy.characters import initial_clustering_extract_and_plot_clusters

    WHICH_LEVEL = "trial"
    WHICH_FEATURE = "beh_motor_sim" # For clustering/scoring.
    if WHICH_BASIS_SET is None:
        WHICH_BASIS_SET = animal

    # Load DS
    savedir = f"{SAVEDIR_THIS}/DS_before_cluster"
    path = f"{savedir}/{animal}-{date}.pkl"
    with open(path, "rb") as f:
        DSorig = pickle.load(f)

    # (1) Perform clustering
    # becuase overwrites below
    DS = DSorig.copy()

    ### Run --> to assign cluster label
    savedir = f"{SAVEDIR_THIS}/DS_after_cluster/basis_{WHICH_BASIS_SET}-shapes_{which_shapes}"
    os.makedirs(savedir, exist_ok=True)
    print("Running... ", savedir)

    # savedir = f"{SAVEDIR_THIS}/{animal}-{date}/{WHICH_LEVEL}-basis_{WHICH_BASIS_SET}-shapes_{which_shapes}"
    if HACK:
        # Quickly load old version
        DS, params_dict = DS.clustergood_load_saved_cluster_shape_classes(which_basis=WHICH_BASIS_SET, which_shapes=which_shapes)            
    else:
        _savedir = f"{savedir}/{animal}-{date}"
        initial_clustering_extract_and_plot_clusters(DS, WHICH_LEVEL, WHICH_BASIS_SET, which_shapes, WHICH_FEATURE, _savedir, 
                                                    do_save=False)
    # Save DS
    DS.save(savedir, filename=f"{animal}-{date}")

def fig5_load_data(animal, date):
    """
    Load data
    """
    from neuralmonkey.classes.population_mult import dfpa_concat_normalize_fr_split_multbregion_flex

    savedir = f"{SAVEDIR_ALL}/fig5"

    ### Save it
    if False:
        path = f"{savedir}/DFallpa-{animal}-{date}.pkl"
        with open(path, "rb") as f:
            DFallpa = pickle.load(f)
    else:
        DFallpa = None

    path = f"{savedir}/map_morphset_to_dfallpa-{animal}-{date}.pkl"
    with open(path, "rb") as f:
        map_morphset_to_dfallpa = pickle.load(f)

    path = f"{savedir}/map_tcmorphset_to_idxmorph-{animal}-{date}.pkl"
    with open(path, "rb") as f:
        map_tcmorphset_to_idxmorph = pickle.load(f)

    path = f"{savedir}/list_morphset-{animal}-{date}.pkl"
    with open(path, "rb") as f:
        list_morphset = pickle.load(f)

    path = f"{savedir}/map_tcmorphset_to_info-{animal}-{date}.pkl"
    with open(path, "rb") as f:
        map_tcmorphset_to_info = pickle.load(f)

    # Normalize FR
    fr_mean_subtract_method = "across_time_bins"
    for dfallpa in map_morphset_to_dfallpa.values():
        # pa = dfallpa["pa"].values[0]
        # pa.plotNeurHeat(0)
        dfpa_concat_normalize_fr_split_multbregion_flex(dfallpa, fr_mean_subtract_method)

    return DFallpa, map_morphset_to_dfallpa, map_tcmorphset_to_idxmorph, list_morphset, map_tcmorphset_to_info


def fig6_load_data(animal, date):
    """
    Load data
    """
    from neuralmonkey.classes.population_mult import dfpa_concat_normalize_fr_split_multbregion_flex

    path = f"{SAVEDIR_ALL}/fig6/DFallpa-{animal}-{date}.pkl"

    with open(path, "rb") as f:
        DFallpa = pickle.load(f)
    
    # Normalize firing rates
    fr_mean_subtract_method = "across_time_bins"
    dfpa_concat_normalize_fr_split_multbregion_flex(DFallpa, fr_mean_subtract_method, PLOT=False)

    return DFallpa

if __name__=="__main__":
    import sys

    
    # PLOTS_DO = [2.1]
    # # PLOTS_DO = [5.1]
    
    plot_do = sys.argv[1]

    ### RUN
    if plot_do=="2kp":
        LIST_BASIS_SHAPE = [
            ("Diego_shuffled", None),
            ("Pancho_shuffled", None),
            ("Diego_shuffled_1", None),
            ("Pancho_shuffled_1", None),
            ("Diego_shuffled_2", None),
            ("Pancho_shuffled_2", None),
            ("Diego", "main_21"),
            ("Pancho", None),
            ]
        
        animal = sys.argv[2]
        date = sys.argv[3]

        SAVEDIR_THIS = f"{SAVEDIR_ALL}/fig2kp"
        os.makedirs(SAVEDIR_THIS, exist_ok=True)

        for WHICH_BASIS_SET, which_shapes in LIST_BASIS_SHAPE:    
            fig2k_cluster_label(animal, date, SAVEDIR_THIS, WHICH_BASIS_SET, which_shapes)

    elif plot_do=="5dh":

        from neuralmonkey.scripts.analy_decode_moment_psychometric import analy_switching_GOOD_euclidian_index

        DO_RSA_HEATMAPS = True # May take time..
        manuscript_version = True

        animal = sys.argv[2]
        date = sys.argv[3]

        _, map_morphset_to_dfallpa, map_tcmorphset_to_idxmorph, list_morphset, map_tcmorphset_to_info = fig5_load_data(animal, date)

        SAVEDIR_THIS = f"{SAVEDIR_ALL}/fig5dh"
        os.makedirs(SAVEDIR_THIS, exist_ok=True)
        savedir = f"{SAVEDIR_THIS}/{animal}-{date}"
        os.makedirs(savedir, exist_ok=True)

        for morphset in list_morphset:

            dfallpa = map_morphset_to_dfallpa[morphset]
            analy_switching_GOOD_euclidian_index(dfallpa, savedir, map_tcmorphset_to_idxmorph, 
                                                        [morphset], map_tcmorphset_to_info,
                                                        make_plots=True, save_df=True,
                                                        DO_RSA_HEATMAPS=DO_RSA_HEATMAPS,
                                                        manuscript_version=manuscript_version)

    elif plot_do == "6f":
        
        from neuralmonkey.scripts.analy_euclidian_chars_sp import euclidian_time_resolved_fast_shuffled

        SAVEDIR_THIS = f"{SAVEDIR_ALL}/fig6f"
        os.makedirs(SAVEDIR_THIS, exist_ok=True)

        animal = sys.argv[2]
        date = sys.argv[3]

        DFallpa = fig6_load_data(animal, date)

        DO_RSA_HEATMAPS = False
        for DO_REGRESS_HACK in [True, False]:
            
            savedir = f"{SAVEDIR_THIS}/{animal}-{date}-regrhack={DO_REGRESS_HACK}"
            os.makedirs(savedir, exist_ok=True)
            print(savedir)

            euclidian_time_resolved_fast_shuffled(DFallpa, animal, date, savedir, DO_RSA_HEATMAPS=DO_RSA_HEATMAPS, 
                                    HACK = True, DEBUG_bregion_list=None, DO_REGRESS_HACK=DO_REGRESS_HACK,
                                    manuscript_version=True)
    else:
        print(plot_do)
        assert False
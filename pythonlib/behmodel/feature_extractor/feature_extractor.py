""" one job - take in a single datapoint (e..g, parse) applies stored parameters to 
score this datapoint """
import numpy as np


class FeatureExtractor(object):
    def __init__(self, STROKE_DIR_DOESNT_MATTER=True):
        """ params is dict. flexible, and whatever keys are active will dictate what
        features are considered
        - Main focus is working with parses in Parser format.
        - Does not score or save any scoring-related params.
        NOTE:
        - STROKE_DIR_DOESNT_MATTER, mainly placeholder, in future maybe modify. this just 
        now used for hashing the edges.
        """

        self.Params = {}
        self.Params["STROKE_DIR_DOESNT_MATTER"] = STROKE_DIR_DOESNT_MATTER

    # def list_featurevec_from_list_p(self, list_of_p, hack_lines5 = False):
    #     """ 
    #     IN:
    #     - list_of_p is list of parses for a single trial.
    #     """

    #     """ 
    #     Given single parser (one trial/task, multipel parses, one skeleton graph)
    #     returns a list of featuresvecs.
    #     IN:
    #     - P, parser object, holds list of parses
    #     OUT:
    #     - list of featurevecs, one for each parse in P
    #     """
    #     from pythonlib.drawmodel.features import strokeCircularity, strokeDistances

    #     tracker = {}
    #     STROKE_DIR_DOESNT_MATTER = self.Params["STROKE_DIR_DOESNT_MATTER"]

    #     def _path_to_key(walker):
    #         return walker.unique_path_id(invariant=STROKE_DIR_DOESNT_MATTER)

    #     def _path_to_feature_vec(traj):
    #         feat = (
    #             strokeCircularity([traj])[0],
    #             strokeDistances([traj])[0])
    #         return feat

    #     def feature_single_p(p):
    #         # first check if this is already saved
    #         key = _path_to_key(p["walker"])
    #         if key in tracker:
    #             return tracker[key]
    #         else:
    #             # get new
    #             feat = _path_to_feature_vec(p["traj"])
    #             tracker[key]=feat
    #             return feat

    #     def featurevec_single_trial(list_of_p):
    #         """ a single trial is a list of ps
    #             e..g, for trial ind, 
    #             list_of_p = P.extract_parses_wrapper(ind, "summary")
    #         OUT:
    #         - feature_vec, (N,) shape np array.
    #         """

    #         list_of_features = []
    #         for p in list_of_p:
    #             list_of_features.append(feature_single_p(p))

    #         # Take average over all strokes
    #         mat_of_features = np.stack(list_of_features) # N x nfeat
    #         feature_vec = np.mean(mat_of_features, axis=0)

    #         # Add other features
    #         # - max circ across trajs
    #         feature_vec = np.append(feature_vec, np.max(mat_of_features[:,0]))

    #         #- nstrokes
    #         nstrokes = len(list_of_p)
    #         feature_vec = np.append(feature_vec, nstrokes)

    #         return feature_vec
        
    #     list_featurevecs = []
    #     for ind in range(len(P.Parses)):
    #         list_of_p = P.extract_parses_wrapper(ind, "summary")
    #         feature_vec = featurevec_single_trial(list_of_p)
    #         list_featurevecs.append(feature_vec)
            
    #     return list_featurevecs



    def list_featurevec_from_mult_parser(self, D, ind, 
        parser_names = ["parser_graphmod", "parser_nographmod"],
        hack_lines5 = False):
        """ Get list of feature vecs, convatenated across all parses
        IN:
        - D, Dataset
        - ind, index into D
        - parser_names, columns for D
        OUT:
        - list of np arrays.
        """
        
        list_of_parsers = D.parser_list_of_parsers(ind, parser_names=parser_names)
        list_featurevec_cat = []
        for P in list_of_parsers:
            if hack_lines5:
                list_featurevec = self.list_featurevec_from_parser_lines5(P)
            else:
                assert False, "need general purpose feature vec parasm"
            list_featurevec_cat.extend(list_featurevec)
        return list_featurevec_cat


    def list_featurevec_from_parser_lines5(self, P):
        """ 
        Given single parser (one trial/task, multipel parses, one skeleton graph)
        returns a list of featuresvecs.
        IN:
        - P, parser object, holds list of parses
        OUT:
        - list of featurevecs, one for each parse in P
        """
        from pythonlib.drawmodel.features import strokeCircularity, strokeDistances

        tracker = {}
        STROKE_DIR_DOESNT_MATTER = self.Params["STROKE_DIR_DOESNT_MATTER"]

        def _path_to_key(walker):
            return walker.unique_path_id(invariant=STROKE_DIR_DOESNT_MATTER)

        def _path_to_feature_vec(traj):
            feat = (
                strokeCircularity([traj])[0],
                strokeDistances([traj])[0])
            return feat

        def feature_single_p(p):
            # first check if this is already saved
            key = _path_to_key(p["walker"])
            if key in tracker:
                return tracker[key]
            else:
                # get new
                feat = _path_to_feature_vec(p["traj"])
                tracker[key]=feat
                return feat

        def featurevec_single_trial(list_of_p):
            """ a single trial is a list of ps
                e..g, for trial ind, 
                list_of_p = P.extract_parses_wrapper(ind, "summary")
            OUT:
            - feature_vec, (N,) shape np array.
            """

            list_of_features = []
            for p in list_of_p:
                list_of_features.append(feature_single_p(p))

            # Take average over all strokes
            mat_of_features = np.stack(list_of_features) # N x nfeat
            feature_vec = np.mean(mat_of_features, axis=0)

            # Add other features
            # - max circ across trajs
            feature_vec = np.append(feature_vec, np.max(mat_of_features[:,0]))

            #- nstrokes
            nstrokes = len(list_of_p)
            feature_vec = np.append(feature_vec, nstrokes)

            return feature_vec
        
        list_featurevecs = []
        for ind in range(len(P.Parses)):
            list_of_p = P.extract_parses_wrapper(ind, "summary")
            feature_vec = featurevec_single_trial(list_of_p)
            list_featurevecs.append(feature_vec)
            
        return list_featurevecs



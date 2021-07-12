""" quickly coded likeli functions

"""



def makeLikeliFunction(ver="segments", norm_by_num_strokes=True, 
    standardize_strokes=False, asymmetric=False):
    """ returns a function that can pass into Model,
    which does the job of computing likeli (ie., model human dist) 
    CONVENTION: more positive is better match.
    NOTE: by defai;t tje split_segments version does take into acocunt directionality.
    # 
    - norm_by_num_strokes this default, shoudl be, for cases where distance is
    summed over strokes. This is true for distanceDTW.
    - standardize_strokes, then will subtract mean and divide by x liomits range
    - combine_beh_segments, then combines all segments for animal into a single stroke.
    """
    if ver in ["split_segments", "timepoints"]:
        # these differ depending on direction fo strokes
        print("NOTE: should get parses in both directions [default for getParses()], since this distance function cares about the chron order.")

    def likeliFunction(t):
        dists_all = []
        for p in t["model_parses"]:
            strokes_parse = p["strokes"]
            if ver=="modHaussdorf":
                dist = -distanceBetweenStrokes(t["behavior"]["strokes"], strokes_parse)
                if norm_by_num_strokes:
                    dist = dist/len(t["behavior"]["strokes"])
            elif ver in ["segments", "combine_segments"]:
                from pythonlib.tools.stroketools import distanceDTW
                if ver=="combine_segments":
                    S = [np.concatenate(t["behavior"]["strokes"], axis=0)]
                    v = "segments"
                else:
                    S = t["behavior"]["strokes"]
                    v = "segments"
                dist = -distanceDTW(S, strokes_parse, 
                    ver=v, norm_by_numstrokes=norm_by_num_strokes, asymmetric=asymmetric)[0]
            else:
                assert False, "not coded"

            dists_all.append(dist)
        return dists_all
    return likeliFunction


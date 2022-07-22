"""
works with sf, dataframe where each column is single stroke.
    this can be extracted into Dataset class,
    using methods prefixed by "sf"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from ..dataset.dataset_strokes import DatStrokes


def computeSimMatrixGivenBasis(Dat, DatBasis, rescale_strokes_ver, distancever, npts_space):
    """ Compute sim matrix between all strokes in SF vs. the specific inputed basis strokes
    PARAMS:
    - Dat, 
    --- dataframe of strokes with col name "strok"
    --- or strokes (list of np array)
    - DatBasis, same, but the basis set
    --- or strokes (list of np array)
    """
    from pythonlib.tools.stroketools import rescaleStrokes, strokesInterpolate2
    from .strokedists import distMatrixStrok
    
    ############# Extract data
    # data
    if isinstance(Dat, pd.core.frame.DataFrame):
        # Then is pandas dataframe 
        stroklist = Dat["strok"].values.tolist()
    else:
        stroklist = Dat
    idxs_stroklist_dat = range(len(stroklist))
    
    # data (basis set) (append)
    if isinstance(DatBasis, pd.core.frame.DataFrame):
        strokes_basis = DatBasis["strok"].values.tolist()
    else:
        strokes_basis = DatBasis
    stroklist.extend(strokes_basis)
    idxs_stroklist_basis = range(len(idxs_stroklist_dat), len(stroklist))

    ############# Preprocess
    # rescale
    if rescale_strokes_ver=="stretch_to_1":
        stroklist = [rescaleStrokes([s])[0] for s in stroklist]
    else:
        assert False, "which?"
        print("keeping strokes scale unchaged")

    # interpolate
    if distancever in ["euclidian", "euclidian_diffs"]:
        # then need to be same length
        stroklist = strokesInterpolate2(stroklist, N=["npts", npts_space], base="space")
    else:
        assert False, "which?"

    ### Cmpute sim matrix
    similarity_matrix = distMatrixStrok(idxs_stroklist_dat, idxs_stroklist_basis, stroklist=stroklist,
                       normalize_cols_range01=True, distancever=distancever)

    return similarity_matrix


def computeSimMatrix(SF, rescale_strokes_ver="idxs_stroklist_dat", 
        distancever="euclidian_diffs", npts_space=50, Nbasis=300):
    """ Get distance matrix, entire dataset, with random instances chosen for basis
    PARAMS:
    """

    if Nbasis>len(SF):
        print(Nbasis)
        print(len(SF))
        assert False, "not enough data..."

    idxs_stroklist_dat = list(range(len(SF)))
    idxs_stroklist_basis = random.sample(range(len(SF)), Nbasis)

    # Get sim matrix
    similarity_matrix = computeSimMatrixGivenBasis(SF, SF.iloc[idxs_stroklist_basis], 
        rescale_strokes_ver, distancever, npts_space)

    return similarity_matrix, idxs_stroklist_basis


def preprocessStroksList(strokes, params):
    """ Preprioess strokes same as
    preprocessStroks, but takes in list of np arrays
    PARAMS:
    - strokes, list of np arrays
    - params, dict, see preprocessStroks
    RETURNS:
    - strokes, list of np arrays modified. does not modify input.
    """

    dfthis = pd.DataFrame({"strok":strokes})
    dfthis = preprocessStroks(dfthis, params)
    strokes = dfthis["strok"].tolist()
    return strokes

def preprocessStroks(df, params):
    """
    preprocessing, including normalizing, centering, etc, on a set of strok data.
    - df, dataframe with column "strok", holding a np.array for its strok
    - params, dict, telling what preprocessing to do.
    RETURNS:
    - df, shape shape as df, but with strok replaced with processed.
    (no modify in place)
    NOTE: resets index for df in return.
    NOTE:
    - Tested speed using two methods, either modify each row in df, or first convert to
    list then do list comprehension. 
    Run: %timeit preprocessStroks(SF, params)
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.tools.stroketools import rescaleStrokes, strokesInterpolate2

    # ANOTHER METHOD, applied to df directly. is slower, see notse above, so do not use
#     if params["align_to_onset"]:
#         def F(x):
#             return x["strok"] - x["strok"][0,:]
#         df = applyFunctionToAllRows(df, F, "stroknew")
# #             df["strok"] = df["stroknew"] # replace old
# #             del df["stroknew"]
# #             print(df.iloc[0])
            
    # Fill gaps with defaults
    params_default = {
        "align_to_onset":False,
        "min_stroke_length_percentile":None,
        "min_stroke_length":None,
        "max_stroke_length_percentile":None,
        "centerize":False,
        "rescale_ver":None
    }
    for k, v in params_default.items():
        if k not in params.keys():
            params[k] = v

    # Sanity check
    assert params["align_to_onset"] != params["centerize"], "cannot do both"

    print("starting length of dataframe:")
    print(len(df))

    # pre-extract features,. if needed
    if params["max_stroke_length_percentile"] or params["min_stroke_length_percentile"] or params["min_stroke_length"]:
        # then get distance(length) for each strok
        if "distance" not in df.columns:
            from pythonlib.drawmodel.features import strokeDistances
            strok_lengths = strokeDistances(list(df["strok"].values))
            df["distance"] = strok_lengths
    
    if params["max_stroke_length_percentile"] is not None:
        thresh =  np.percentile(df["distance"], [params["max_stroke_length_percentile"]])
        # if False:
        #     plt.figure(figsize=(15,5))
        #     plt.hist(SF["distance"], 100);
        #     plt.axvline(thresh)
        #     print(thresh)
        #     assert False
            
        def F(x):
            return x["distance"]<thresh
        df = applyFunctionToAllRows(df, F, "keep")
        df = df[df["keep"]==True]
        del df["keep"]
        # df = df.reset_index()

    # if "min_stroke_length_percentile" in params.keys():
    if params["min_stroke_length_percentile"] is not None:
        thresh =  np.percentile(df["distance"], [params["min_stroke_length_percentile"]])
        # if False:
        #     plt.figure(figsize=(15,5))
        #     plt.hist(SF["distance"], 100);
        #     plt.axvline(thresh)
        #     print(thresh)
        #     assert False
        def F(x):
            return x["distance"]>thresh
        df = applyFunctionToAllRows(df, F, "keep")
        df = df[df["keep"]==True]
        del df["keep"]
        # df = df.reset_index()

    # if "min_stroke_length" in params.keys():
    if params["min_stroke_length"] is not None:
        thresh =  params["min_stroke_length"]
        def F(x):
            return x["distance"]>thresh
        df = applyFunctionToAllRows(df, F, "keep")
        df = df[df["keep"]==True]
        del df["keep"]
        # df = df.reset_index()

    # === pull out stroks.
    stroklist = list(df["strok"].values)
    
    if params["align_to_onset"]:
        def F(strok):
            return strok - strok[0,:]
        stroklist = [F(strok) for strok in stroklist]
        
    if params["centerize"]:
        def F(strok):
            c = np.mean(strok[:,:2], axis=0)
            s = strok.copy()
            s[:,:2] = s[:,:2] - c
            return s
        stroklist = [F(strok) for strok in stroklist]

    if params["rescale_ver"] is not None:
        stroklist = [rescaleStrokes([s], ver=params["rescale_ver"])[0] for s in stroklist]

    # === finally, put back into df
    df["strok"] = stroklist
    
    print("after filtering:")
    print(len(df))

    df = df.reset_index(drop=True)
    
    return df
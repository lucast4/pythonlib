""" based around parsing in pyBPL package (Reuben Feinman)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

def extractJunctions(strokes, sketchpad_edges, image_edges):
    """ automatically converts strokes to image coordinates, then gets posiotions of junctions
    - strokes, in monkey sketchpad space
    - can first run:
    --- sketchpad_edges, image_edges = getSketchpadEdges(canvas_max_WH, image_WH)
    # This section deals with issue that corners are in general not used as junctions in 
    # the undirected graph that represents the drawing during parsing.
    # For monkey, they often break down L-shaped things into two lines, so we want to consider
    # parses where corners can have a junction.

    # Solution: determine coordinates of all "segment" endpoints, which will generally include
    # corners (since things like L's are represnted as two lines usually). 
    # Will just pass in all endpoints, since anything that is redundant with the 
    # BPL junctions will be automaticalyl discarded.
    from pythonlib.drawmodel.parsing import extractJunctions

    if stim_ver=="task_image":
        extra_junctions = extractJunctions(strokes, sketchpad_edges, image_edges)
        print(extra_junctions)
    elif stim_ver=="monkey_strokes":
        # then dont need to do this, since if monkey raises finger at corner, that will be
        # detected by BPL, since it will not be a perfectly clean corner, and so will
        # be considered a "cross". [have not verified this by testing]
        pass
        

    """
    from pythonlib.drawmodel.image import convStrokecoordToImagecoord, getSketchpadEdges
    
    # use this to convert strokes to image space
    extra_junctions = []
    for pts in strokes:
        pts_image_inds = convStrokecoordToImagecoord(pts, sketchpad_edges, image_edges)
        extra_junctions.append(pts_image_inds[0])
        extra_junctions.append(pts_image_inds[-1])

    extra_junctions = np.stack(extra_junctions, axis=0)
    return extra_junctions


def get_parses_from_strokes(strokes, canvas_max_WH, image_WH=105, k=20,     
    score_fn = lambda parses: score_function(parses, ver="travel", normalization="negative", use_torch=True), 
    use_extra_junctions=False, plot=False):
    """ 

    """
    from pythonlib.drawmodel.parsing import extractJunctions
    from pythonlib.drawmodel.image import getSketchpadEdges
    # from gns.inference.parsing.top_k import get_topK_parses    

    # convert to images
    from pythonlib.drawmodel.image import strokes2image
    I = strokes2image(strokes, canvas_max_WH, image_WH, plot=plot)

    # Extract junctions
    if use_extra_junctions:
        sketchpad_edges, image_edges = getSketchpadEdges(canvas_max_WH, image_WH)
        extra_junctions = extractJunctions(strokes, sketchpad_edges, image_edges)
    else:
        extra_junctions = None

    # Parse
    grp_kwargs = {"ver":"lucas", "extra_junctions":extra_junctions}
    pp_kwargs = {"do_fit_spline":False}
    k = 20
    parses_k, log_probs_k = get_topK_parses(I, k, score_fn, pp_kwargs=pp_kwargs, **grp_kwargs)

    # Conver to np
    def _torch2np(strokes):
        return [s.numpy() for s in strokes]
    parses_k = [_torch2np(strokes) for strokes in parses_k]
    log_probs_k = log_probs_k.numpy()

    return parses_k, log_probs_k

def score_function(parses, ver="ink", normalization = "inverse", test=False,
                  use_torch=False):
    """ 
    - ver, str, determines what score to use
    --- "ink", then total distnace traveled on page
    --- "travel", then total distance traveled, including
    gaps, starting from position of first touch.
    - normalization, how to normalize raw distnace. distance will
    be that more positive is more worse. 
    --- inverse take inverse, so that now less positive is worse.
    --- negative, ...
    """
    from pythonlib.drawmodel.features import strokeDistances, computeDistTraveled

    if test:
        # then just return random number, one for each parse
        return torch.tensor([random.random() for _ in range(len(parses))])    
    
    if ver=="ink":
        # === Total ink used
        distances = [np.sum(strokeDistances(strokes)) for strokes in parses]
    elif ver=="travel":
        # conisder origin to be onset of first storke.
        distances_traveled = [computeDistTraveled(strokes, origin=strokes[0][0,[0,1]]) for strokes in parses]
        distances = distances_traveled
    elif ver=="nstrokes":
        # num strokes
        # == plit histogram of num strokes
        nstrokes = [len(p) for p in parses]        
    else:
        print(ver)
        assert False, "not codede"
        
    if use_torch:
        distances = torch.tensor(distances)
    else:
        distances = np.array(distances)
        
    if normalization=="inverse":
        return 1/distances
    elif normalization=="negative":
        return -distances
    else:
        print(normalization)
        assert False, "not coded"
    

def convertCoordParseToStrokes(parse, canvas_max_WH, image_WH=105):
    """ Convert parses back into monkey sketchpad coords
    parses, output of pyPBL based parsing, convert back to monkey sketchpad
    - parse, is list of np arraus, (like "strokes")
    """
    from pythonlib.drawmodel.image import convCoordGeneral

    parse_edges = np.array([[1, image_WH-1], [-(image_WH-1), -1]]) # since parsing has translated y to negatives.
    sketchpad_edges = np.array([[-canvas_max_WH, canvas_max_WH], [-canvas_max_WH, canvas_max_WH]])
    strokes = [convCoordGeneral(S, parse_edges, sketchpad_edges) for S in parse]

    return strokes


################# COPIES FROM GNS LIBRARY, BUT ADDING PARAMS
def process_parse(parse, device=None, do_fit_spline=True):
    """ same as gns.inference.parsing.top_k, but allowing to skip fitting 
    spline """
    from pybpl.data import unif_space
    from gns.omniglot.minimal_splines import fit_minimal_spline

    parse_ = []
    for stk in parse:
        # for ntraj = 1, set spline as the original stroke
        if len(stk) == 1:
            spl = torch.tensor(stk, dtype=torch.float, device=device)
            parse_.append(spl)
            continue
        # for ntraj > 1, fit minimal spline to the stroke
        stk = unif_space(stk)
        stk = torch.tensor(stk, dtype=torch.float, device=device)
        if do_fit_spline:
            spl = fit_minimal_spline(stk, thresh=0.7, max_nland=50)
        else:
            spl = stk
        parse_.append(spl)
    return parse_


def get_topK_parses(img, k, score_fn, configs_per=100, trials_per=800,
                    device=None, seed=3, pp_kwargs=None, **grp_kwargs):
    """ copied from
    gns.inference.parsing.top_k, except allwoing for pp_kwargs, which is useful 
    if want to skip spline fit optimiziation step
    """
    from pybpl.matlab.bottomup import generate_random_parses
    from gns.inference.parsing.top_k import search_parse    

    # generate random walks (the "base parses")
    base_parses = generate_random_parses(I=img, seed=seed, **grp_kwargs)
    # convert strokes to minimal splines
    base_parses = [process_parse(parse, device, **pp_kwargs) for parse in base_parses]

    # search for best stroke ordering & stroke direction configurations
    np.random.seed(seed)
    n = len(base_parses)
    parses = []; log_probs = []
    for i in range(n):
        parses_i, log_probs_i = search_parse(
            base_parses[i], score_fn, configs_per, trials_per)
        parses.extend(parses_i)
        log_probs.append(log_probs_i)
    log_probs = torch.cat(log_probs)

    # refine to unique & sort
    log_probs, idx = np.unique(log_probs.cpu().numpy(), return_index=True)
    log_probs = torch.from_numpy(log_probs).flip(dims=[0])
    idx = torch.from_numpy(idx).flip(dims=[0])
    parses = [parses[i] for i in idx]

    return parses[:k], log_probs[:k]
""" relating strokes to images
"""
import numpy as np
import matplotlib.pyplot as plt

def getSketchpadEdges(canvas_max_WH, image_WH):
    """ helper to get sketchpad edges in common format
    """
    sketchpad_edges = np.array([[-canvas_max_WH, canvas_max_WH], [-canvas_max_WH, canvas_max_WH]])
    image_edges = np.array([[1, image_WH-1], [1, image_WH-1]]) # 1 on edges, since there is a slight border.
    return sketchpad_edges, image_edges


def strokes2image(strokes, canvas_max_WH, image_WH, smoothing=0.9, bin_thresh= 0.6,
    plot=False):
    """ default params are used for converitng to binary pixel image,, used for
    parsing.
    - canvas_max_WH, coordinates of the sketchpad used by monkey
    max width or hiegh (whichever one greater), num pixels for
    half of page, so that sketchpad will be square with edges in 
    both dimensions of : (-WH, WH). e.g.; canvas_max_WH = np.max(np.abs(metadat["sketchpad_edges"])) # smallest square that bounds all the stimuli
    - image_WH, num pixels per dimension for the output image
    image_WH = 105 # 105 is used for BPL
    - smoothing, set to 1.
    - bin_thresh, for binarizing, 0.6 is useful for smoothing=1

    """
    from pythonlib.drawmodel.primitives import prog2pxl
    from pythonlib.drawmodel.strokePlots import plotDatStrokes
    from pythonlib.drawmodel.image import convStrokecoordToImagecoord

    I = prog2pxl(strokes, WHdraw=canvas_max_WH*2, WH=image_WH, smoothing=smoothing)
    I = np.array(I>bin_thresh) # binarize

    if plot:
        # ----- PLOTS
        # plot strokes
        fig, ax = plt.subplots(figsize=(10,10))
        plotDatStrokes(strokes, ax, each_stroke_separate=True)

        # plot hist of values
        plt.figure()
        plt.hist(I[:], log=True)

        # plot
        plt.figure()
        plt.imshow(I, cmap="gray")
        plt.colorbar()
        plt.title("after binarize")

        print(I.shape)
    return I


def convCoordGeneral(pts, edges1, edges2):
    """ general purpose conversion, assumes that only translating and rescaling image, but no
    rotation 
    INPUT:
    - pts. N x 2 array of pts in space of edges1.
    - edges1, [xmin, xmax; ymin, ymax], 2 x 2 array
    - edges2, [xmin, xmax; ymin, ymax], 2 x 2 array
    (lower bound is inclusize, upper is eclusive, 
    so xmin and xmax are possible values that pts could take), so
    if xmin and xmax are [1,2], then assumes window is 1 unit wide (i.e., is [1, 2)]
    """

    if pts.shape[1]==3:
        t = pts[:,2] # save
        pts = pts[:, [0,1]]
        do_append=True  
    else:
        do_append=False
    
    pts = pts - np.repeat(edges1[:,0].T[None, :], pts.shape[0], axis=0) # subtract x and y min
    pts = pts/np.repeat(np.diff(edges1, axis=1).T, pts.shape[0], axis=0) # rescale between 0 and 1.
        
    # expand to size of image
    tmp = np.repeat(np.diff(edges2, axis=1).T, pts.shape[0], axis=0)
    pts = pts * tmp

    # add on the start position
#     print(image_edges[:,0][None, :].shape)
    pts = pts + np.repeat(edges2[:,0][None, :], pts.shape[0], axis=0)

    # append time
    if do_append:
        pts = np.c_[pts, t]
    
    return pts

def convStrokecoordToImagecoord(pts, sketchpad_edges, image_edges, flip="xy"):
    """ given a coordinate  return coordinate in 
    image space, 
    INPUT:
    - pts. N x 2 array of pts in monkey pixel space.
    (in original monkey pixel space,
    i.e., which is centered at 0,0, and x gets more negative 
    towards left, y more negative towards down)
    - sketchpad_edges, [xmin, xmax; ymin, ymax], 2 x 2 array
    - image_edges, same format, where x and y are what would see if plot in
    imshow (ie., x here is y in sketchpad coords)
    (for which 0,0 is at top left corner, and x gets larger as go down,
    y larger as go to right, like in plt.imshow). is inclusize, so if
    use [0, 104], then those indices exist. 
    e..g, if give [3, 105; 0; 105], this means top of sketchpad should start
    3 pixels below top of image.
    - flipxy, useful if comvert from strokes to bitmap, since in imshow a bitmap x coord
    is vertical, but in strokes it is horizontal.
    -- set to "y" if converting from parses to strokes, since y is made to be negetivae.
    RETURN:
    - pts_out, N x 2 array of pts in image space.
    NOTE: throws out 3rd column (time) if you pass in N x 2 array
    """
#     t = pts[:,2] # save
    pts = pts[:, [0,1]]
    
    
    pts = pts - np.repeat(sketchpad_edges[:,0].T[None, :], pts.shape[0], axis=0) # subtract x and y min
    pts = pts/np.repeat(np.diff(sketchpad_edges, axis=1).T, pts.shape[0], axis=0) # rescale between 0 and 1.
    
    # use 1-y, since care about distance from top of image (0,0)
    pts[:,1] = 1-pts[:,1]
    
    # flip x and y
    if flip=="xy":
        pts = np.fliplr(pts)
    elif flip=="y":
        pts[:,1] = -pts[:,1]
    else:
        print(flip)
        assert False, "not coded"
    
    # expand to size of image
    tmp = np.repeat(np.diff(image_edges, axis=1).T, pts.shape[0], axis=0)
    pts = pts * tmp
    
    # round to nearest integer, since these are indices
    pts = np.round(pts).astype(int)
    
    # add on the start position
#     print(image_edges[:,0][None, :].shape)
    pts = pts + np.repeat(image_edges[:,0][None, :], pts.shape[0], axis=0)
    
    # append time
#     pts = np.c_[pts, t]
    
    return pts
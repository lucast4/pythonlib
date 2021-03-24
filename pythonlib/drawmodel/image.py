""" relating strokes to images
"""
import numpy as np


def convStrokecoordToImagecoord(pts, sketchpad_edges, image_edges):
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
    pts = np.fliplr(pts)
    
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
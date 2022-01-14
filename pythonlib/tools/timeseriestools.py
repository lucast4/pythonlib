
import numpy as np

def stackTrials(Xlist, ver="append_nan"):
    """ 
    take list of neural activations, stack into single matrix. 
    Useful if, e.g., each trial different length, and want to 
    preprocess in standard ways to make them all one stack.
    - Xlist, list, length trials, each element (nunits, timebins).
    can have diff timebine, but shoudl have same nunits.
    RETURNS:
    - Xout, (ntrials, nunits, timebins), array
    """

    assert len(list(set([X.shape[0] for X in Xlist])))==1, "all need to have same n units"
    nunits = Xlist[0].shape[0]
    
    if ver=="append_nan":
        # appends nans to each trial so they all length same as max
        
        # 1) max length
        lens = [X.shape[1] for X in Xlist]
        maxlen = max(lens)
        print(f"appending nans to each trial timebins so match max time: {maxlen}")
        # 2) append nans
        Xlistout = []
        for X in Xlist:
            nt = X.shape[1]
            nanarray = np.empty_like(X, shape=(nunits, maxlen-nt))
            nanarray[:] = np.nan
            
            Xout = np.concatenate((X, nanarray), axis=1)
            Xlistout.append(Xout)
        
        # 3) stack
        Xout = np.array(Xlistout)
        
        
    elif ver=="cut_to_min":
        # cuts length of all trials to min len over trials
        assert False
    elif ver=="time_warp":
        # time warp
        assert False, "not coded"
    else:
        assert False, "not coded"
    
    print(f"shape of output X: {Xout.shape}")
    return Xout


def smoothDat(x, window_len=11, window='hanning', flip_at_edges=False):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    Deals with edges by assuming that is flat at edges (appends enough based on window size)
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer. autoamtiaclly
        makes odd if not.
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    assert window_len-int(window_len)==0
    if window_len%2==0:
        window_len+=1
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if np.any(np.isnan(x)):
        raise ValueError("does nto deal with nans well! will leave a gap where there is nan, without appending to data at each end of nan. better to first segment then pass in each segment")
        
    if flip_at_edges:
        # The signal is prepared by introducing reflected copies of the signal (with the window size) in both ends so that transient parts are minimized in the begining and end part of the output signal.
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        assert False, "need to modify, since the output length of y will be N+2*((windowsize-1)/2)"
    else:
        # signal prepared by repeating the endpoint value 
        s = np.r_[np.ones(int((window_len-1)/2))*x[0], x, np.ones(int((window_len-1)/2))*x[-1]]
        
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    # print("--")
    # print(len(x))
    # print(len(s))
    # print(window_len)
    # print("==")
    y=np.convolve(w/w.sum(),s,mode='valid')
    # print(len(y))
    return y


def getSampleTimesteps(T, fs, force_T=False):
    """ get timesteps from 0, ..., T, 
    where T is in sec, and sample rate
    of fs. if T is not a whole number of 
    samples, then returns shorter sequence,
    up to last sample before T.
    - if force_T, then will allow slight change in fs
    to make sure t is 0, ..., T. will also return fs_new
    """
    
    if force_T:
        nsamp = int(np.round(T*fs))
        t = np.linspace(0, T, nsamp+1)
        fs_new = 1/(T/nsamp)
        return t, fs_new
    else:
        nsamp = int(np.floor(T*fs))
        Tnew = nsamp/fs
        t = np.linspace(0, Tnew, nsamp+1)
        return t



def DTW(x, y, distfun, asymmetric=True):
    """ dynamic time warping between two arrays x and y. can be 
    lists, or np arrays (e.g T x 2 vs. N x 2). 
    - distfun is distance function that takes in two elements and 
    outputs scalar, where larger is further distances 
    - assymetric means that will output min distances such that uses
    up all of x, but not constrined to use up all of y. This is didfferent 
    from standard algo where endpoints have to be pinned to each other.
    - 
    """
    table = {}
    # this table will store the cost of the minimum paths
    # TODO: make this assymetric, so that cannot go from (0,0) to (0,1)

    # def distance(n, m):
    #     return (n - m)**2

    def minimumCostPath(i,j):
        # figures out the cost of the shortest path which uses the first i members of x and the first j members of y
        if (i,j) in table: return table[(i,j)]

        cost = distfun(x[i], y[j])
        if i > 0 and j > 0:        
            cost += min(minimumCostPath(i - 1, j - 1),
                        minimumCostPath(i - 1, j),
                        minimumCostPath(i, j - 1))
        elif i > 0:
            cost += minimumCostPath(i - 1, j)
        elif j > 0:
            cost += minimumCostPath(i, j - 1)

        table[(i,j)] = cost

        return cost

    def optimalAlignment(i,j):
        assert (i,j) in table, "first you have to compute the minimum cost path"
        from math import isclose 

        thisCost = table[(i,j)]
        residual = thisCost - distfun(x[i], y[j])

        if i > 0 and j > 0 and isclose(table[(i - 1, j - 1)], residual):
            alignment = optimalAlignment(i - 1, j - 1)
        elif i > 0 and isclose(table[(i - 1, j)], residual):
            alignment = optimalAlignment(i - 1, j)
        elif j > 0 and isclose(table[(i, j - 1)], residual):
            alignment = optimalAlignment(i, j - 1)
        elif j == 0 and i == 0:
            alignment = []
        else:
            print([i, j])
            print(table[(i - 1, j - 1)])
            print(residual)
            assert False, "this should be impossible"

        alignment.append((i,j))

        return alignment

    # ==== OUTPUT
    m=len(x)-1
    n=len(y)-1

    # -- compute distances
    minimumCostPath(m, n)
    
    if False:
        print("the alignment with this minimum cost is",
          optimalAlignment(len(x) - 1,
                           len(y) - 1))
    
    # print(table)
    if asymmetric:
        D = np.array([table[(m, j)] for j in range(n+1)])
        distmin = np.min(D)
        alignment = optimalAlignment(m, np.argmin(D))
    else:
        distmin = table[(m, n)]
        alignment = optimalAlignment(m, n)

    return distmin, alignment


def getChangePoints(vals):
    """ find when vals changes
    only makes sense when vals are in blocks, and sudden switches, 
    e.g., vals = [1 1 1 1 2 2 1 3 3 3], here would extract [4 6 7]               ,
    the indices of first val in a block.
    - Returns as np array of indices.
    """
    vals = np.array(vals)
    idx_of_bloque_onsets = []
    idx_of_bloque_onsets = (np.argwhere(np.diff(vals))+1).reshape(-1)
    return idx_of_bloque_onsets



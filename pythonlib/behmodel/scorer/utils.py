
import numpy as np
def normscore(scores_all, ver, params=None):
    """given lsit of scalars (scores_all)  
    across parses, and a method (ver), output probabilsitys in 
    a list.
    TODO: dont use divide. 
    (it doesnt work well if there exist different signs.)
    - params, tuple of params, flexible depending on ver
    """

    if len(scores_all)==0:
        print(ver)
        print(params)
        assert False

    if ver=="divide":
        # simple, just divide by sum of alls cores
        # if any is negative, then subtracts it.
        # is very hacky!!!
        if np.any(scores_all>0) and np.any(scores_all<0):
            scores_all = scores_all - np.min(scores_all)
        s_sum = np.sum(scores_all)
        return scores_all/s_sum
        # return (score)/np.sum(scores_all)
    elif ver=="softmax":
        # softmax. first normalize scores so that within similar range
        # dividing by mean of absolute values of scores.. this is hacky...
        from scipy.special import softmax
        # scores_all = np.array([-5, 10, 25])
        # sumabs = np.sum(np.absolute(scores_all))

        # standardize scores. first subtract mean score. then divide by MAD
        invtemp = 1
        if params is None:
            scores_all = scores_all - np.mean(scores_all)
            meanabs = np.mean(np.absolute(scores_all))
            if meanabs>0:
                scores_all = scores_all/meanabs
        elif params=="raw":
            # do nothing to raw scores
            pass
        else:
            invtemp = params[0]
            subtrmean = params[1]

            if subtrmean==True:
                # then subtract mean. this useful if large variation across trials for mean
                scores_all = scores_all - np.mean(scores_all)
        scores_all = invtemp*scores_all
        probs = softmax(scores_all) 
        return probs
    else:
        assert False, "not coded!"


def posterior_score(likelis, priors, ver):
    if ver=="top1":
        c = np.random.choice(np.flatnonzero(priors == priors.max())) # this randomly chooses, if there is tiebreaker.
        post = likelis[c]
    elif ver=="maxlikeli":
        # is positive control, take the maximum likeli parse
        c = np.random.choice(np.flatnonzero(likelis == likelis.max()))
        post = likelis[c]
    elif ver=="weighted":
        # baseline - weighted sum of likelihoods by prior probabilities
        post = np.average(likelis, weights=priors)
    elif ver=="likeli_weighted":
        # uses likelihoods as weights.. this is like apositive control..
        # 1) convert likelis to probabilities by softmax
        probs = self.normscore(likelis, ver="softmax")
        post = np.average(likelis, weights=probs)
    else:
        assert False, "not coded"
    return post

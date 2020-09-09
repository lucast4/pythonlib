
import numpy as np


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

               


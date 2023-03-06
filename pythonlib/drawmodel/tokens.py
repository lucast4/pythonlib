

class Tokens(object):
    """
    """
    def __init__(self, tokens):
        self.Tokens = tuple(tokens) # order is immutable



    def feature_location(self, i):
        xloc, yloc = self.Tokens[i]["gridloc"]
        return xloc, yloc

    def featurepair_gridposdiff(self, i, j):
        """ difference in grid position
        returns (x, y)
        """
        pos1 = self.feature_location(i)
        pos2 = self.feature_location(j)
        return pos2[0]-pos1[0], pos2[1] - pos1[1]

    def featurepair_griddist(self, i, j):
        """ Distance in grid, euclidian scalar
        """
        (x,y) = self.featurepair_gridposdiff(i, j)
        return (x**2 + y**2)**0.5




""" 
Can be used for either beh or task ordered tokens
"""

import numpy as np
class Tokens(object):
    """
    """
    def __init__(self, tokens, version=None):
        """
        PARAMS:
        - version, string, e.g, "beh" or "task"
        """

        if version is not None:
            assert version in ["beh", "task"]
        self.Version = version
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

    def chunks_update_by_shaperank(self, map_shape_to_rank):
        """ Assign each token to a chunk and a ordinal withn the chunk, based
        on rule that shapes are done in order, such as line --> circle.
        E.g, if did [L, L, L, C, C, L] then will return:
        - chunk_rank = [0 0 0 1 1 2]
        - chunk_within_rank = [0 1 2 0 1 0]
        """

        tokens = self.Tokens

        # 1) get cjhunk rank
        chunk_rank = [map_shape_to_rank[t["shape"]] for t in tokens]

        # 2) get ordinal within the chunk
        # e.g [0 0 1 1 1 0] will map to [0 1 0 1 2 0]
        chunk_within_rank = []
        r_prev = None
        for r in chunk_rank:
            if r==r_prev:
                o = o+1
            else:
                # reset
                o = 0
            r_prev = r
            chunk_within_rank.append(o)
        
        # 3) modify tokens
        for t,r,o in zip(tokens, chunk_rank, chunk_within_rank):
            t["chunk_rank"] = r
            t["chunk_within_rank"] = o
        
        return chunk_rank, chunk_within_rank


    def sequence_context_relations_calc(self):
        """ Get each item's sequence context, entering as new features
        for each tok.
        NOTE: derived from taskgeneral._tokens_generate_relations
        """

        tokens = self.Tokens

        grid_ver = "on_grid"
        for dseg in tokens:
            if dseg["gridloc"] is None:
                grid_ver = "on_rel"
        
        def _location(i):
            xloc, yloc = tokens[i]["gridloc"]
            return xloc, yloc

        def _posdiffs(i, j):
            # return xdiff, ydiff, 
            # in grid units.
            pos1 = _location(i)
            pos2 = _location(j)
            return pos2[0]-pos1[0], pos2[1] - pos1[1]

        def _direction(i, j):
            # only if adjacnet on grid.
            xdiff, ydiff = _posdiffs(i,j)
            if np.isclose(xdiff, 0.) and np.isclose(ydiff, 1.):
                return "up"
            elif np.isclose(xdiff, 0.) and np.isclose(ydiff, -1.):
                return "down"
            elif xdiff ==-1. and np.isclose(ydiff, 0.):
                return "left"
            elif xdiff ==1. and np.isclose(ydiff, 0.):
                return "right"
            else:
                return "far"

        def _relation_from_previous(i):
            # relation to previous stroke
            if i==0:
                return "start"
            else:
                return _direction(i-1, i)

        def _horizontal_or_vertical(i, j):
            xdiff, ydiff = _posdiffs(i,j)
            if np.isclose(xdiff, 0.) and np.isclose(ydiff, 0.):
                return "same_location"
                # assert False, "strokes are on the same grid location, decide what to call this"
            elif np.isclose(xdiff, 0.):
                return "vertical"
            elif np.isclose(ydiff, 0.):
                return "horizontal"
            else: # xdiff, ydiff are both non-zero
                return "diagonal" 

        def _horiz_vert_move_from_previous(i):
            if i==0:
                return "start"
            else:
                return _horizontal_or_vertical(i-1, i)


        for i, dseg in enumerate(tokens):
            # 2) Things that depend on grid
            if grid_ver=="on_grid":
                # Then this is on grid, so assign grid locations.
                dseg["rel_from_prev"] = _relation_from_previous(i)
                # dseg["rel_to_next"] = _relation_to_following(i)
                dseg["h_v_move_from_prev"] = _horiz_vert_move_from_previous(i)
                # dseg["h_v_move_to_next"] = _horiz_vert_move_to_following(i)
            elif grid_ver=="on_rel":
                # Then this is using relations, not spatial grid.
                # give none for params
                # dseg["gridloc"] = None
                dseg["rel_from_prev"] = None
                # datsegs[-1]["rel_to_next"] = None
                dseg["h_v_move_from_prev"] = None
                # datsegs[-1]["h_v_move_to_next"] = None
            else:
                print(grid_ver)
                assert False, "code it"

            #### Context
            if i==0:
                loc_prev = (0, "START") # tuple, so that is same type as gridloc
                shape_prev = "START"
            else:
                loc_prev = tokens[i-1]["gridloc"]
                shape_prev = tokens[i-1]["shape"]
            dseg["CTXT_loc_prev"] = loc_prev
            dseg["CTXT_shape_prev"] = shape_prev

            if i==len(tokens)-1:
                loc_next = ("END", 0)
                shape_next = "END"
            else:
                loc_next = tokens[i+1]["gridloc"]
                shape_next = tokens[i+1]["shape"]
            dseg["CTXT_loc_next"] = loc_next
            dseg["CTXT_shape_next"] = shape_next






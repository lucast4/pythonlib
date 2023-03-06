""" Methods that generally take in TaskGeneral object, and returns various features
about it. Doesnt care about behavior, just the task.
Often operates with tokens.

NOTE: doesnt usualyl do things related to sequences/groups, which is instead done in 
motifs_search, or discrete. 

3/5/23
"""



def shapes_n_each_extract(Task, list_shapes, shape_key="shapeabstract"):
    """ Return list with n times each shape exists in this task
    PARAMS;
    - list_shapes, list of str.
    RETURNS;
    - list_n, list of int, how often each shape occur,
    - n_left_over, int, how many shapes in Task were not detected?
    """

    tokens = Task.tokens_generate()
    shapes = [t[shape_key] for t in tokens]
    
    # ignore order. just count how many A, B, C, ... etc
    nshapes = []
    for sh in list_shapes:
        n = sum([sh==x for x in shapes])
        nshapes.append(n)

    # shapes left over?
    n_left_over = len([x for x in shapes if x not in list_shapes])

    return nshapes, n_left_over



def shapes_has_separated_cases_of_this_shape(Task, shape_of_interest, ploton=False):
    """ Returns True if mjultiple strokes of this shape exist, and are seprated in space
    such that at least one of them is closer to a different shape than it is to the same
    shape. E..g, lines are far apart, with circle in between.
    PARAMS:
    - shape_of_interest, the shape for wich to return True if it is separated.
    RETURNS;
    - bool.
    """

    from pythonlib.drawmodel.tokens import Tokens

    # def _min_distances(shapes_in_order, distances, shapes):
    #     distances_in_order = []
    #     for sh in shapes_in_order:
    #         distances_shape = [d for d, s in zip(distances, shapes) if s==sh]
    #         if len(distances_shape)>0:
    #             dthis = min(distances_shape)
    #             distances_in_order.append(dthis)
    #     return distances_in_order

    # Genreate tokens
    tokens = Task.tokens_generate()
    Tok = Tokens(tokens)

    if ploton:
        Task.plotStrokes(ordinal=True)

    res = []
    for i in range(len(tokens)):

        # Collect distances to each other token
        distances = []
        shapes = []
        shape_this = Tok.Tokens[i]["shapeabstract"]
        if not shape_this==shape_of_interest:
            # then ignore.
            continue

        for j in range(len(tokens)):
            # compare this tok to all other toks.
            if j==i:
                continue

            # get their distnace
            d = Tok.featurepair_griddist(i, j)
            sh1 = Tok.Tokens[i]["shapeabstract"]
            sh2 = Tok.Tokens[j]["shapeabstract"]
            res.append({
                "dist":d,
                "sh1":sh1,
                "sh2":sh2
            })

            distances.append(d)
            shapes.append(sh2)

        # Collect distances to this shape and other shapes.
        # shape_this = Tok.Tokens[i]["shapeabstract"]
        dists_shape_this = [d for d, sh in zip(distances, shapes) if sh==shape_this]
        dists_shape_other = [d for d, sh in zip(distances, shapes) if not sh==shape_this]
        # if shape_this=="line":
        #     shape_other = "circle"
        # else:
        #     shape_other ="line"

        if ploton:
            print("conditioned on token #:", i, shape_this)
            print("dists to this shape:", dists_shape_this)
            print("dists to other shapes:", dists_shape_other)

        if len(dists_shape_this)>0 and len(dists_shape_other)>0:
            min_dist_this = min(dists_shape_this)
            min_dist_other = min(dists_shape_other)
            if min_dist_other<0.99*min_dist_this:
                # then distnace to other is closer than to this same... 
                # means these two instances of same shape are separted by other shape.
                return True

    return False



def decompose_string(s, sep="-"):
    """ 
    Returns list of substrings, which are
    separated by sep in s
    e.g;
    s = "11-2-3"
    returns:
    ['11', '2', '3'], or [] if sep doesnt exist.
    """
    import re
    inds = [m.start() for m in re.finditer(sep, s)]
    if len(inds)==0:
        # Then didnt find this
        return []
    inds = [-1] + inds + [len(s)+1]
    substrings = []
    for i1, i2 in zip(inds[:-1], inds[1:]):
        substrings.append(s[i1+1:i2])
    return substrings
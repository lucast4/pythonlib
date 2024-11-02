

def decompose_string(s, sep="-"):
    """ 
    Returns list of substrings, which are
    separated by sep in s
    e.g;
    s = "11-2-3"
    returns:
    ['11', '2', '3'], or [] if sep doesnt exist.
    """

    assert sep in ["-", "_"], "| did not work in one case not sure wy."
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

def trialcode_to_scalar(tc):
    """
    Convert trialcode to a scalar that is sortable (globally, within a subject, guaranteed to be currect)

    date.1
    240115.28 means
    date = 240115
    sess = 2
    trial = 800
    
    """
    from pythonlib.tools.stringtools import decompose_string
    
    tc_tuple = trialcode_to_tuple(tc)
    if tc_tuple is None:
        return None
    else:
        assert tc_tuple[1] < 10, "then divide by more, so that is mapped to max 0.,000 is mapped to 0.01"
        assert tc_tuple[2] < 10000, "then divide by more, so that 100,000 is mapped to 0.01"

        a = tc_tuple[0]
        b =  tc_tuple[1]/10
        c = tc_tuple[2]/10000
        # mainting the code...
        assert b<1
        assert c<0.1
        return a+b+c
    
def trialcode_to_tuple(tc):
    """
    PARAMS:
    - tc, any type, but a valid tc is yyyy-mm-dd
    Return eitehr (int, int, int) or None otherwise (all ways in which this can faiL)
    """
    from pythonlib.tools.stringtools import decompose_string
    
    if not isinstance(tc, str):
        return None 
    
    tmp = decompose_string(tc)
    
    if not len(tmp)==3:
        return None
    else:
        a, b, c = tmp
        # if (not str(int(a))==a) or (not str(int(b))==b) or (not str(int(c))==c):
        #     return None
        try:
            return (int(a), int(b), int(c))
        except ValueError:
            # usualyl becuase a b or c are not numbers
            return None
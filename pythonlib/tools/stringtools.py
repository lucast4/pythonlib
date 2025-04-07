import numpy as np

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

def trialcode_to_scalar(tc, allow_failure=True, input_tuple_directly=False):
    """
    Convert trialcode to a scalar that is sortable (globally, within a subject, guaranteed to be currect)

    date.1
    240115.28 means
    date = 240115
    sess = 2
    trial = 800
    
    """
    from pythonlib.tools.stringtools import decompose_string

    if input_tuple_directly:
        tc_tuple = tc
    else:
        tc_tuple = trialcode_to_tuple(tc)

    if tc_tuple is None:
        if not allow_failure:
            print(tc)
            assert False
        return None
    else:
        assert tc_tuple[1] < 10, "then divide by more, so that is mapped to max 0.,000 is mapped to 0.01"
        assert tc_tuple[2] < 100000, "then divide by more, so that 100,000 is mapped to 0.01"

        a = tc_tuple[0]
        b =  tc_tuple[1]/10
        c = tc_tuple[2]/100000
        # mainting the code...
        try:
            assert b<1
            assert c<0.1
            return a+b+c
        except Exception as err:
            print(tc)
            print(a, b, c)
            raise err

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
        
def trialcode_extract_rows_within_range(list_trialcode, tc_start, tc_end, input_tuple_directly=False):
    """
    Return indices (into list_trialcode) which have truialcode within range of [tc_start, tc_end], inclusive.
    Checks based on real time, not based on location within dataframe
    PARAMS:
    - list_trialcode, list of string trialcodes
    - input_tuple_directly, if True, then input tc's as tuples of ints, e.g, (240522, 1, 20)
    RETURNS:
    - inds, trialcodes, both lists, those that fall within the range.
    """
    from pythonlib.tools.stringtools import trialcode_to_scalar
    
    list_trialcode_scalar = np.array([trialcode_to_scalar(tc) for tc in list_trialcode])

    tc_start_scalar = trialcode_to_scalar(tc_start, allow_failure=False, input_tuple_directly=input_tuple_directly)
    tc_end_scalar = trialcode_to_scalar(tc_end, allow_failure=False, input_tuple_directly=input_tuple_directly)

    inds = np.where((list_trialcode_scalar >= tc_start_scalar) & (list_trialcode_scalar <= tc_end_scalar))[0]
    trialcodes = [list_trialcode[i] for i in inds]
    
    # convert to list
    inds = list(inds)

    return inds, trialcodes

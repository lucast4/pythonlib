import numpy as np

def check_objects_identical(obj1, obj2):
    """ Works with commonly used objects, checks if identical,
    Returns False if any difference found
    otherwise True.
    """

    # Go thru all cases. if any are false, returns False
    if not type(obj1)==type(obj2):
        return False
    elif isinstance(obj1, list):
        if not len(obj1)==len(obj2):
            return False
        for x, y in zip(obj1, obj2):
            good = check_objects_identical(x,y)
            if not good:
                return False
    elif isinstance(obj1, dict):
        if not sorted(obj1.keys()) == sorted(obj2.keys()):
            return False
        for k in obj1.keys():
            good = check_objects_identical(obj1[k], obj2[k])
            if not good:
                return False
    elif isinstance(obj1, np.ndarray):
        if not np.all(np.isclose(obj1, obj2)):
            return False
    elif isinstance(obj1, (int, str)):
        if not obj1==obj2:
            return False
    else:
        print(obj1)
        print(obj2)
        print(type(obj1))
        print(type(obj2))
        assert False, "add this type"

    # got here, passed all tests.
    return True
        

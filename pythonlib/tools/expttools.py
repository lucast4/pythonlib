""" verious functions useful for running experiemnts/analysis"""


def makeTimeStamp(exptID="", datefirst=True):
    """useful for saving things, 
    gives yymmdd_hhmmss_{exptID}"""
    import time
    ts = time.localtime()
    tstamp = time.strftime("%y%m%d_%H%M%S", ts)

    if len(exptID)>0:
        if not datefirst:
            exptID+="-"
        else:
            exptID = "-" + exptID

    if datefirst==False:
        return f"{exptID}{tstamp}"
    else:
        return f"{tstamp}{exptID}"


def getDateList(sdate=None, edate=None):
    """    sdate = 200226, for Feb 26 2020
    """
    
    from datetime import date, timedelta, datetime
    
    if sdate is None:
        sdate=date(2020, 1, 1)
    else:
        sdate = datetime.strptime(f"{sdate}", '%y%m%d')
    
    if edate is None:
        edate =  date.today()
    else:
        edate = datetime.strptime(f"{edate}", '%y%m%d')

    delta = edate - sdate       # as timedelta
    date_list = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    # for i in range(delta.days + 1):
    #     day = sdate + timedelta(days=i)
    #     print(day)
    date_list = [d.strftime("%y%m%d") for d in date_list]
    return date_list

def writeStringsToFile(fname, stringlist, silent=True):
    """ atuoamtically does newline """
    with open(fname, "w") as f:
        for s in stringlist:
            if not silent:
                print(s)
            f.write(f"{s}\n")

def writeDictToYaml(dictdat, path):
    """ path shold include extension (.yaml)"""
    import yaml

    with open(path, 'w') as f:
        # yaml.dump(dictdat, f, default_flow_style=False)
        yaml.dump(dictdat, f)

def extractStrFromFname(fname, sep, pos, return_entire_filename=False):
    """ given fname like '/data2/animals/Pancho/201030/201030_164324_arc2_Pancho_3.h5'
    pull out arc2 if give me sep="_" and pos = "2"
    - pos is 0,1, ...
    - if return_entire_filename, then overwrites sep and pos, and instead returns 
    201030_164324_arc2_Pancho_3
    """
    import os
    import re
    
    # 1) get the filename without leading dir, or training extension
    # e.g., gets 201030_164324_arc2_Pancho_3
    path = os.path.split(fname)[1]
    path = os.path.splitext(path)[0]

    if return_entire_filename:
        return path

    # 2) get posoptions of the separator
    idxs = [m.start() for m in re.finditer(sep, path)]
    idxs.append(len(path)) # to allow getting of last substr
    idxs.insert(0, -1) # to allow getting of first
    
    # 2) get substring
#     print(idxs)
    if len(idxs)<pos+2:
        print("this pos otu of bounds (returning None)")
        return None
    else:
        return path[idxs[pos]+1:idxs[pos+1]]

def checkIfDirExistsAndHasFiles(dirname):
    """ returns (exists?, hasfiles?), 
    - hasfiles is obviously always False if exists
    is False
    """
    import os
    if os.path.isdir(dirname):
        exists = True
        if len(os.listdir(dirname))>0:
            hasfiles = True
        else:
            hasfiles = False
    else:
        exists = False
        hasfiles = False

    return (exists, hasfiles)

    

def findPath(path_base, path_hierarchy, path_fname="", ext="",
    return_without_fname=False):
    """ get list of data, searches thru paths.
    INPUT:
    - path_base, str, static, common across all possible
    paths
    - path_hierarchy, list, where each element is another list,
    where each element in that list is a string, each of which
    will be separated by wildcards to find paths. 
    - ext, extensions for files. leave empty to not care.
    - return_without_fname, then returns path name but splt off from the
    final file name. (i..e, jsut gets the dir)s
    NOTES:
    The order
    matter. e..g,: 
    path_hierarchy = [['pancho', 'beh'], ['figures']) 
    ext = ".pkl"
    means look for 
    {path_base}/*pancho*beh*/*figures*/*.pkl
    NOTE:
    - length of path_hierarchy must match the hierarchy of the paths.
    EXAMPLE USAGE:
        path_base = "/data2/analyses/database/clustering/bysimilarity"
        path_hierarchy = [
            ["combined"],
            ["Red", "Pancho"],
            ["Pancho", "Red"],
        ]
        # path_hierarchy = [
        # ]
        path_fname = "SAVEDAT"
        ext = ".pkl"
        findPath(path_base, path_hierarchy, path_fname, ext)

    """
    import glob
    
    def _summarize(pathlist):
        print("Found this many paths:")
        print(len(pathlist))
        for p in pathlist:
            print("---")
            print(p)
    
    # Construct path
    path = path_base
    
    for p in path_hierarchy:
        path += "/*"
        for pp in p:
            path += f"{pp}*"
    
    path += "/*" + path_fname + "*" + ext
    
    print("Searching using this string:")
    print(path)
    
    # Search
    pathlist = glob.glob(path)

    # Process
    if return_without_fname:
        print("-- Splitting off dir from fname")
        # pull out just the path before the file name
        import os
        pathlist = [os.path.split(p)[0] for p in pathlist]
        pathlist = list(set(pathlist))

    pathlist = sorted(pathlist)
    _summarize(pathlist)

    return pathlist
    
def get_common_path(pathlist):
    """ if pathlist len>1, then gets the most specific path 
    that is common across all. if len==1, then gets the final directory.
    """
    import os

    assert isinstance(pathlist, list)

    if len(pathlist)>1:
        path_shared = os.path.commonpath(pathlist)
    elif len(pathlist)==1:
        path_shared = os.path.split(pathlist[0])[0]
    else:
        assert False
    return path_shared

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

def fileparts(path):
    """ breaks down path into dir, path, ext. always works regaridless of whether give 
    relative or fullpath.
    NOTES:
    - if path is full, then pathdir will be string.
    - if path is rel, then pathdir is empyt string.
    - if path is full, but only one level (e/g. path = /tmp.ext), pathdir is "/"
    - Will alkways be able to reconstruct path as pathdir + pathname + ext
    """
    # f = "/210517_152802_plan3_Red_1.pkl"
    import os
    pathdir, pathend = os.path.split(path)
    pathname, ext = os.path.splitext(pathend)
    if len(pathdir)>0 and pathdir!="/":
        pathdir+="/" # to allow reconstruction as pathdir + pathname + ext
   
    return pathdir, pathname, ext

    
def modPathFname(path, prefix=None, suffix=None):
    """ moves file in path, to nbew filename either
    {prefix}-{path} or {path, without ext}-{suffix}-{ext}
    NOTES:
    - ok to entire full path, this code will pick out the lowest level filename.
    RETURNS:
    - new pathname
    """
    from pathlib import Path
    pathdir,pathname, ext = fileparts(path)

    if prefix:
        pathname = prefix + "-" + pathname
    if suffix:
        pathname = pathname + "-" + suffix
    pathout = pathdir + pathname + ext

    Path(path).rename(pathout)
    print("Renamed path1 to path2:")
    print(path)
    print(pathout)
    return pathout


def findPath(path_base, path_hierarchy, path_fname="", ext="",
    return_without_fname=False, sort_by="name", path_hierarchy_wildcard_on_ends=True,
    strings_to_exclude_in_path=[]):
    """ get list of data, searches thru paths.
    INPUT:
    - path_base, str, static, common across all possible
    paths
    - path_hierarchy, list, where each element is another list,
    where each element in that list is a string, each of which
    will be separated by wildcards to find paths. 
    - path_fname, {str, None} make this None if want to have the last level of path_hierarchy
    be the filename.
    - ext, extensions for files. leave empty to not care.
    - return_without_fname, then returns path name but splt off from the
    final file name. (i..e, jsut gets the dir)s
    - sort_by, if "name", then alphabetically. otherwise {"size", "date"}. always incresaing order.
    - path_hierarchy_wildcard_on_ends, if true, allows /*[]*[]*/... otherwise does /[]*[]/
    - strings_to_exclude_in_path, list of strings, will exclude  apath if any of these strings occurs anywhere in the
    path.

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
        if path_hierarchy_wildcard_on_ends:
            path += "/*"
        else:
            path += "/"
        for pp in p:
            path += f"{pp}*"
        if not path_hierarchy_wildcard_on_ends:
            path = path[:-1] # remove the last wildcard
    
    if path_fname is None:
        path += "*" + ext
    else:
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

    if sort_by == "name":
        pathlist = sorted(pathlist)
    elif sort_by == "size":
        import os
        fsizes = [os.path.getsize(f) for f in pathlist]

        tmp = [(s, p) for s, p in zip(fsizes, pathlist)]
        tmp = sorted(tmp, key=lambda x:x[0])
        pathlist = [t[1] for t in tmp]
    elif sort_by == "date":
        import os
        fsizes = [os.path.getmtime(f) for f in pathlist]
        tmp = [(s, p) for s, p in zip(fsizes, pathlist)]
        tmp = sorted(tmp, key=lambda x:x[0])
        pathlist = [t[1] for t in tmp]
    else:
        print(sort_by)
        assert False, "not coded"

    def string_check(path):
        """ returns True if none of the strings to check are in the path
        """
        return all([s not in path for s in strings_to_exclude_in_path])

    pathlist = [path for path in pathlist if string_check(path)]

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


def load_yaml_config(path):
    """ 
    Load a file.yaml into an output dict
    NOTE:
    - path should have .yaml extension.
    """
    import yaml
    with open(path) as file:
        outdict = yaml.load(file, Loader=yaml.FullLoader)
    return outdict

""" verious functions useful for running experiemnts/analysis"""


def makeTimeStamp(exptID="", datefirst=True):
    """useful for saving things, 
    gives yymmdd_hhmmss_{exptID}"""
    import time
    ts = time.localtime()
    tstamp = time.strftime("%y%m%d_%H%M%S", ts)

    if datefirst==False:
        return f"{exptID}-{tstamp}"
    else:
        return f"{tstamp}-{exptID}"


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


def extractStrFromFname(fname, sep, pos):
    """ given fname like '/data2/animals/Pancho/201030/201030_164324_arc2_Pancho_3.h5'
    pull out arc2 if give me sep="_" and pos = "2"
    - pos is 0,1, ...
    """
    import os
    import re
    
    # 1) get the filename without leading dir, or training extension
    # e.g., gets 201030_164324_arc2_Pancho_3
    path = os.path.split(fname)[1]
    path = os.path.splitext(path)[0]

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

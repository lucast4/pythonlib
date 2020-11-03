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

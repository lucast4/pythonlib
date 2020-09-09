    # # how to compare dates
    # import datetime
    # d = datetime.datetime.strptime(str(date), "%y%m%d")

    # d1 = datetime.datetime.today()
    # d<d1


def datenum2obj(date, dtformat="%y%m%d"):
    """ converts date (in str or int) to a datetime object
    - by default, date in format yymmdd (e..g, 200130) 
    - can be string or int
    - returns datetime obnject"""
    from datetime import datetime
    
    if isinstance(date, int):
        dt = datetime.strptime(str(date), dtformat)
    elif isinstance(date, str):
        dt = datetime.strptime(date, dtformat)
    return dt
    

def getDateList(sdate=None, edate=None):
    """ get list of dates between (including) sdate and edate
    - dates can either be None (defualt, see below), strings, or
    ints (in format yymmdd).
    - default, sdate is 1/1/20;
    - default, edate is today.
    - date format will be yymmdd (e..g, 200130)
    """
#     sdate = date(2020, 1, 1)   # start dat
    from datetime import date, timedelta, datetime
    if sdate is None:
        sdate=datetime(2020, 1, 1)
    else:
        sdate = datenum2obj(sdate)
        
    if edate is None:
        edate =  datetime.today()
    else:
        edate = datenum2obj(edate)
        
    delta = edate - sdate       # as timedelta
    date_list = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    # for i in range(delta.days + 1):
    #     day = sdate + timedelta(days=i)
    #     print(day)
    date_list = [d.strftime("%y%m%d") for d in date_list]
    return date_list

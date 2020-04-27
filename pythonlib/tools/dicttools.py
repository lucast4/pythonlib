"""generally takes in list of dicts, where all dicts have same keys"""


def printOverviewKeyValues(summarydict_all, keystoget=[]):
    """Prints all keys and their unique values. ignoers some keys that have numerical vbalues."""
    if len(keystoget)==0:
        keystoget = set([kk for k in summarydict_all for kk in k.keys()])
    for key in keystoget:
        if key not in ["planner", "res", "cost", "x0", "x1", "x2", "x3", "x4", "x5", "x6"]:
            try:
                values = set([s[key] for s in summarydict_all])
            except:
                values = set([", ".join(s[key]) for s in summarydict_all])
            N = len(values)
            print("-----")
            print("- {} (N = {})".format(key, N))
            if N>100:
                print("SKIPPING PRINTING OF VALUES FOR {}, SINCE LEN {}".format(key, len(values)))
                continue
            [print(v) for v in values]

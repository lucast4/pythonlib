""" script, extracting parses, using the Parser (good) code.
This runs and saves thru Dataset code.
"""
from pythonlib.dataset.dataset import Dataset

QUICK = True # quick means fast parsing.
 
def get_dataset(a,e,r, FIXED):
    D = Dataset([])
    D.load_dataset_helper(a, e, ver="mult", rule=r)
    D.load_tasks_helper()

    # Keep only the fixed tasks?
    if FIXED:
        D = D.filterPandas({"random_task":[False]}, "dataset")
    else:
        D = D.filterPandas({"random_task":[True]}, "dataset")
    return D

def run(a,e,r,v, FIXED):
    D = get_dataset(a,e,r, FIXED)
    M = D.Metadats[0]
    SDIR = f"/data2/analyses/database/PARSES_GENERAL/{M['expt']}"
    D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}",     
        SDIR=SDIR, save_using_trialcode=False)


# animal_list = ["Red", "Pancho"]
# expt_list = ["lines5"]
# rule_list = ["straight", "bent"]
# ver_list = ["graphmod", "nographmod"]
# for a in animal_list:
#     for e in expt_list:
#         for r in rule_list:
#             D = get_dataset(a,e,r,fixed_only=FIXED)
#             for v in ver_list:
#                 D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}")



# animal_list = ["Diego", "Pancho"]
# expt_list = ["linecircle"]
# rule_list = ["null"]
# ver_list = ["graphmod", "nographmod"]
# for a in animal_list:
#     for e in expt_list:
#         for r in rule_list:
#             D = get_dataset(a,e,r,fixed_only=FIXED)
#             for v in ver_list:
#                 D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}")


if __name__=="__main__":
    MULTI= False
    FIXED = False # only do fixed tasks.

    if MULTI:
        # Multiprocessing.
        args1, args2, args3, args4, args5 = [], [], [], [], []
        # animal_list = ["Red", "Pancho"]
        # animal_list = ["Pancho"]
        # ver_list = ["graphmod"]
        expt_list = ["gridlinecircle"]
        rule_list = ["baseline", "circletoline", "linetocircle", "lolli"]
        animal_list = ["Pancho"]
        ver_list = ["graphmod"]
        for a in animal_list:
            for e in expt_list:
                for r in rule_list:
                    for v in ver_list:
                        args1.append(a)
                        args2.append(e)
                        args3.append(r)
                        args4.append(v)
                        args5.append(FIXED)
                        # D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}")

        from multiprocessing import Pool
        ncores = 4
        with Pool(4) as pool:
            pool.starmap(run, zip(args1, args2, args3, args4, args5))
    else:
        # rule_list = ["baseline", "circletoline", "linetocircle", "lolli"]
        rule_list = ["baseline", "circletoline", "linetocircle", "lolli"]
        a = "Pancho"
        e = "gridlinecircle"
        v = "graphmod"
        for r in rule_list:
            run(a,e,r,v, FIXED)



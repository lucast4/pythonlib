""" script, extracting parses, using the Parser (good) code.
This runs and saves thru Dataset code.
"""
from pythonlib.dataset.dataset import Dataset

QUICK = False # quick means fast parsing.
FIXED = False # only do fixed tasks.
 
def get_dataset(a,e,r):
    D = Dataset([])
    D.load_dataset_helper(a, e, ver="mult", rule=r)
    D.load_tasks_helper()

    # Keep only the fixed tasks?
    if FIXED:
        D = D.filterPandas({"random_task":[False]}, "dataset")
    return D

def run(a,e,r,v):
    D = get_dataset(a,e,r)
    D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}")


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



if True:
    # Multiprocessing.
    args1, args2, args3, args4 = [], [], [], []
    # animal_list = ["Red", "Pancho"]
    animal_list = ["Pancho"]
    expt_list = ["lines5"]
    rule_list = ["straight", "bent"]
    ver_list = ["graphmod", "nographmod"]
    for a in animal_list:
        for e in expt_list:
            for r in rule_list:
                for v in ver_list:
                    args1.append(a)
                    args2.append(e)
                    args3.append(r)
                    args4.append(v)
                    # D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}")

    from multiprocessing import Pool
    with Pool(2) as pool:
        pool.starmap(run, zip(args1, args2, args3, args4))


# Multiprocessing.
if False:
    args1, args2, args3, args4 = [], [], [], []
    animal_list = ["Diego"]
    expt_list = ["linecircle"]
    rule_list = ["null"]
    ver_list = ["graphmod", "nographmod"]
    for a in animal_list:
        for e in expt_list:
            for r in rule_list:
                for v in ver_list:
                    args1.append(a)
                    args2.append(e)
                    args3.append(r)
                    args4.append(v)
                    # D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}")

    from multiprocessing import Pool
    with Pool(5) as pool:
        pool.starmap(run, zip(args1, args2, args3, args4))

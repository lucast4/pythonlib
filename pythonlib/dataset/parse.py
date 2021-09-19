""" script, extracting parses, using the Parser (good) code.
This runs and saves thru Dataset code.
"""
from pythonlib.dataset.dataset import Dataset

QUICK = False # quick means fast parsing.
INITIAL_EXTRACTION = False
BEH_ALIGNED_EXTRACTION = True

def get_dataset(a,e,r, FIXED):
    D = Dataset([])
    D.load_dataset_helper(a, e, ver="mult", rule=r)
    D.load_tasks_helper()

    # Keep only the fixed tasks?
    if FIXED:
        D = D.filterPandas({"random_task":[False]}, "dataset")
    return D

def run(a,e,r,v, FIXED):
    SDIR = f"/data2/analyses/database/PARSES_GENERAL/{e}"
    D = get_dataset(a,e,r, FIXED)

    if INITIAL_EXTRACTION:
        # D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}",     
        #     SDIR=SDIR, save_using_trialcode=False)
        D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote="",     
            SDIR=SDIR, save_using_trialcode=False)

    if BEH_ALIGNED_EXTRACTION:
        # extract all all the beh-aligned parses too
        list_parse_params = [{"quick":QUICK, "ver":v, "savenote":""}]
        list_suffixes = [x["ver"] for x in list_parse_params]
        pathbase = SDIR
        name_ver = "unique_task_name"
        EXTRACT_BEH_ALIGNED_PARSES = True
        D.parser_load_presaved_parses(list_parse_params, 
            list_suffixes, pathbase=pathbase, name_ver=name_ver,
            ensure_extracted_beh_aligned_parses=EXTRACT_BEH_ALIGNED_PARSES)


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
    FIXED = True # only do fixed tasks.

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
        rule_list = ["baseline", "circletoline", "linetocircle", "lolli"]
        # rule_list = ["linetocircle"]
        for a in ["Pancho", "Diego"]:
            e = "gridlinecircle"
            v = "graphmod"
            for r in rule_list:
                run(a,e,r,v, FIXED)



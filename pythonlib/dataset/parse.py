
from pythonlib.dataset.dataset import Dataset

animal_list = ["Pancho", "Red"]
expt_list = ["lines5"]
rule_list = ["straight", "bent"]
ver_list = ["graphmod", "nographmod"]
for a in animal_list:
    for e in expt_list:
        for r in rule_list:
            D = Dataset([])
            D.load_dataset_helper(a, e, ver="mult", rule=r)
            D.load_tasks_helper()
            for v in ver_list:
                D.parser_extract_and_save_parses(ver=v)



animal_list = ["Diego", "Pancho"]
expt_list = ["linecircle"]
rule_list = ["null"]
ver_list = ["graphmod", "nographmod"]
for a in animal_list:
    for e in expt_list:
        for r in rule_list:
            D = Dataset([])
            D.load_dataset_helper(a, e, ver="mult", rule=r)
            D.load_tasks_helper()
            for v in ver_list:
                D.parser_extract_and_save_parses(ver=v)




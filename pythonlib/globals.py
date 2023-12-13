
import platform
MACHINE = platform.node()
# see https://stackoverflow.com/questions/4271740/how-can-i-use-python-to-get-the-system-hostname

if MACHINE == "lucast4-MS-7B98":
	# gorilla	
	HOME = "/home/lucast4"
	PATH_DRAWMONKEY_DIR = f"{HOME}/code/drawmonkey"
	PATH_NEURALMONKEY = f"{HOME}/code/neuralmonkey/neuralmonkey"
	PATH_DATA_NEURAL_RAW = "/mnt/Freiwald/ltian/recordings"
	PATH_DATA_NEURAL_PREPROCESSED = "/gorilla1/neural_preprocess"
	PATH_DATA_BEHAVIOR_RAW = "/gorilla1/animals"
	PATH_DATA_BEHAVIOR_RAW_SERVER = "/mnt/Freiwald_kgupta/kgupta/macaque_data"
	PATH_ANALYSIS_OUTCOMES = "/gorilla1/analyses"
	PATH_ANALYSIS_OUTCOMES_SERVER = "/mnt/Freiwald_kgupta/kgupta/analyses"
	# PATH_DATASET_BEH = "/gorilla1/analyses/database"
	PATH_DATASET_BEH = "/mnt/Freiwald_kgupta/kgupta/analyses/database"
	PATH_MATLAB = "/gorilla1/programs/MATLAB/R2021a/bin/matlab"
elif MACHINE == "kggs-macbook.rockefeller.edu" or MACHINE == "kggs-macbook.lan" or MACHINE == "kggs-macbook.lan.rockefeller.edu":
	PATH_DRAWMONKEY_DIR = "/Users/kdu/Desktop/rockefeller/drawmonkey"
	PATH_DATA_BEHAVIOR_RAW = "/Users/kdu/data2/animals"
	PATH_MATLAB = "/Users/kdu/data1/programs/MATLAB/R2022a/bin/matlab"
	PATH_ANALYSIS_OUTCOMES = "/Volumes/kdot/analyses"
	PATH_DATASET_BEH = "/Volumes/kdot/analyses/database"
# elif MACHINE == "bonobo":
# 	PATH_NEURALMONKEY = "/data1/code/python/neuralmonkey/neuralmonkey"
# 	PATH_DATA_NEURAL_RAW = "/mnt/hopfield_data01/ltian/recordings"
# 	PATH_DATA_NEURAL_PREPROCESSED = "/data3"
# 	PATH_DATA_BEHAVIOR_RAW = "/home/lucast4/data2/animals"
# 	PATH_ANALYSIS_OUTCOMES = f"{HOME}/data2/analyses"
# 	PATH_DATASET_BEH = f"{HOME}/data2/analyses/database"
# 	PATH_MATLAB = "/data1/programs/MATLAB/R2021a/bin/matlab"
elif MACHINE == "ltbonobo":
	HOME = "/home/kgg"
	PATH_DRAWMONKEY_DIR = "/home/kgg/Desktop/drawmonkey"
	PATH_DATA_BEHAVIOR_RAW = "/home/kgg/mnt/Freiwald/kgupta/macaque_data"
	PATH_DATA_BEHAVIOR_RAW_SERVER = "/home/kgg/mnt/Freiwald/kgupta/macaque_data"
	PATH_NEURALMONKEY = f"{HOME}/Desktop/neuralmonkey/neuralmonkey"
	PATH_DATA_NEURAL_RAW = "/home/kgg/mnt/Freiwald/ltian/recordings"
	PATH_DATA_NEURAL_PREPROCESSED = "/data4/Kedar/neural_preprocess"
	# PATH_MATLAB = "/usr/local/MATLAB/R2022b/bin/matlab"
	PATH_MATLAB = "/data1/programs/MATLAB/R2021a/bin/matlab"
	# PATH_ANALYSIS_OUTCOMES = "/home/kgg/Desktop/analyses"
	PATH_ANALYSIS_OUTCOMES = "/home/kgg/mnt/Freiwald/kgupta/analyses"
	PATH_ANALYSIS_OUTCOMES_SERVER = "/home/kgg/mnt/Freiwald/kgupta/analyses"
	# PATH_DATASET_BEH = "/home/kgg/Desktop/analyses/database"
	PATH_DATASET_BEH = "/home/kgg/mnt/Freiwald/kgupta/analyses/database"
elif MACHINE=="Lucass-MacBook-Air.local":
	HOME = "/Users/lucastian"
	PATH_DRAWMONKEY_DIR = f"{HOME}/code/drawmonkey"
	PATH_NEURALMONKEY = f"{HOME}/code/neuralmonkey/neuralmonkey"
	PATH_DATA_NEURAL_RAW = "DOESNTEXIST1"
	PATH_DATA_NEURAL_PREPROCESSED = "/Users/lucastian/DATA/neural_preprocess"
	PATH_DATA_BEHAVIOR_RAW = "DOESNTEXIST2"
	PATH_DATA_BEHAVIOR_RAW_SERVER = "DOESNTEXIST3"
	PATH_ANALYSIS_OUTCOMES = "/Users/lucastian/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DatasetBeh"
	PATH_ANALYSIS_OUTCOMES_SERVER = "/Users/lucastian/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DatasetBeh"
	PATH_DATASET_BEH = "/Users/lucastian/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DatasetBeh/database"
	PATH_MATLAB = ""
else:
	print(MACHINE)
	assert False, "add this machine"


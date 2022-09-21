
import platform
MACHINE = platform.node()
# see https://stackoverflow.com/questions/4271740/how-can-i-use-python-to-get-the-system-hostname

if MACHINE == "lucast4-MS-7B98":
	# gorilla	
	HOME = "/home/lucast4"
	PATH_DRAWMONKEY_DIR = f"{HOME}/code/drawmonkey"
	PATH_NEURALMONKEY = f"{HOME}/code/neuralmonkey/neuralmonkey"
	PATH_DATA_NEURAL_RAW = "/mnt/hopfield_data01/ltian/recordings"
	PATH_DATA_NEURAL_PREPROCESSED = "/gorilla1/neural_preprocess"
	PATH_DATA_BEHAVIOR_RAW = "/gorilla1/animals"
	PATH_ANALYSIS_OUTCOMES = "/gorilla1/analyses"
	PATH_DATASET_BEH = "gorilla1/analyses/database"
	PATH_MATLAB = "/gorilla1/programs/MATLAB/R2021a/bin/matlab"

elif MACHINE == "bonobo":
	PATH_NEURALMONKEY = "/data1/code/python/neuralmonkey/neuralmonkey"
	PATH_DATA_NEURAL_RAW = "/mnt/hopfield_data01/ltian/recordings"
	PATH_DATA_NEURAL_PREPROCESSED = "/data3"
	PATH_DATA_BEHAVIOR_RAW = "/home/lucast4/data2/animals"
	PATH_ANALYSIS_OUTCOMES = f"{HOME}/data2/analyses"
	PATH_DATASET_BEH = f"{HOME}/data2/analyses/database"
	PATH_MATLAB = "/data1/programs/MATLAB/R2021a/bin/matlab"
else:
	print(MACHINE)
	assert False, "add this machine"


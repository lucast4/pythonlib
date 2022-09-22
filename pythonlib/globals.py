
import platform
MACHINE = platform.node()
# see https://stackoverflow.com/questions/4271740/how-can-i-use-python-to-get-the-system-hostname

if MACHINE == "lucast4-MS-7B98":
	# gorilla	
	PATH_DRAWMONKEY_DIR = "/home/lucast4/code/drawmonkey"
	PATH_NEURALMONKEY = "/home/lucast4/code/neuralmonkey/neuralmonkey"
	PATH_DATA_NEURAL_RAW = "/mnt/hopfield_data01/ltian/recordings"
	PATH_DATA_NEURAL_PREPROCESSED = "/gorilla1/neural_preprocess/recordings"
elif MACHINE == "kggs-macbook.rockefeller.edu" or MACHINE == "kggs-macbook.lan":
	PATH_DRAWMONKEY_DIR = "/Users/kdu/Desktop/rockefeller/drawmonkey"
	PATH_DATA_BEHAVIOR_RAW = "/Users/kdu/data2/animals"
	PATH_MATLAB = "/Users/kdu/data1/programs/MATLAB/R2022a/bin/matlab"
elif MACHINE == "bonobo":
	PATH_NEURALMONKEY = "/data1/code/python/neuralmonkey/neuralmonkey"
	PATH_DATA_NEURAL_RAW = "/mnt/hopfield_data01/ltian/recordings"
	PATH_DATA_NEURAL_PREPROCESSED = "/data3/recordings"
else:
	print(MACHINE)
	assert False, "add this machine"


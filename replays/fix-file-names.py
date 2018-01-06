import os
datas = ['halite-mellendo', 'halite-prisoner3d', 'halite-recurs3', 'halite-zxqfl']
dir = '/Users/redstar/Documents/Programming/Python/Halite/replays/'
for d in datas:
	os.chdir(dir+d)
	for file in os.listdir(os.getcwd()):
		if "_" not in file: continue
		print("Fixing {}".format(file))
		newname = file.split("_")[1]
		os.rename(file, newname)

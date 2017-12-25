import os
datas = ['halite-fakepsyho']
dir = '/Users/redstar/Documents/Programming/Python/Halite/replays/'
for d in datas:
	os.chdir(dir+d)
	for file in os.listdir(os.getcwd()):
		if "_" not in file: continue
		print("Fixing {}".format(file))
		newname = file.split("_")[1]
		os.rename(file, newname)

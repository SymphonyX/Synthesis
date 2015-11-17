import pickle
import math
import os


l = os.listdir("./")
pkls = []

for f in l:
	fi = f.split(".")
	if fi[-1] == "pkl":
		pkls.append(f)

pkls.sort()
train = open("train.txt", "w")

for i in pkls:
	val = i.split(".")[0]
	x = 200 * math.cos(float(val[0]+"."+val[1]))
	y = 200 * math.sin( float(val[0]+"."+val[1]))
	f = open(i, "r")
	data = pickle.load(f)
	params = data[0]
	best_goals = data[1]
	output = ""
	print i
	for j in range(len(params)):
		output += str(params[j]) + ", "

	for j in range(len(best_goals)):
		if j == len(best_goals)-1:
			output += str(best_goals[j])
		else:
			output += str(best_goals[j]) + ", "
	# train.write(val[0]+"."+val[1]+"; "+output+"\n")
	#train.write(str(x)+", "+str(y)+"; "+output+"\n")
	train.write(str(x)+", "+str(y)+", "+val[0]+"."+val[1]+"; "+output+"\n")


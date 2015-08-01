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
	y = -200 * math.sin( float(val[0]+"."+val[1]))
	f = open(i, "r")
	params = pickle.load(f)
	output = ""
	for j in range(params.shape[0]):
		output += str(params[j]) + ", "
	train.write(str(x)+", "+str(y)+"; "+output+"\n")


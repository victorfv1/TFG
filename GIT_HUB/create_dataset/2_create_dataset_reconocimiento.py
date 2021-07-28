import os
import random
import sys

#photos = set(list(os.listdir("no-ortho")))     
maps = set(list(os.listdir("ok-ortho")))



os.system("mkdir training")
os.system("mkdir training/input")
#os.system("mkdir training/output") ESTE NO SE MODIFICA EN LAS 2 EJECUCIONES

files = list(maps)
random.shuffle(files)

n = int(sys.argv[1])

for file in files[:int(n*0.9)]:
	#os.system("mv no-ortho/"+ file + " training/input")
	os.system("mv ok-ortho/"+ file + " training/input")

os.system("mkdir test")
os.system("mkdir test/input")
#os.system("mkdir test/output") ESTE NO SE MODIFICA EN LAS 2 EJECUCIONES

for file in files[int(n*0.9):int(n*0.95)]:
	#os.system("mv no-ortho/"+ file + " test/input")
	os.system("mv ok-ortho/"+ file + " test/input")
    
### added then
os.system("mkdir validation")
os.system("mkdir validation/input")
#os.system("mkdir validation/output") ESTE NO SE MODIFICA EN LAS 2 EJECUCIONES

for file in files[int(n*0.95):n]:
    #os.system("mv no-ortho/"+ file + " validation/input")
    os.system("mv ok-ortho/"+ file + " validation/input")


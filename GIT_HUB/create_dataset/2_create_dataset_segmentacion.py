import os
import random
import sys

photos = set(list(os.listdir("photo")))
maps = set(list(os.listdir("map")))



os.system("mkdir training")
os.system("mkdir training/input")
os.system("mkdir training/output")

files = list(maps & photos)
random.shuffle(files)

n = int(sys.argv[1])

for file in files[:int(n*0.9)]:
    os.system("mv photo/"+ file + " training/input")
    os.system("mv map/"+ file + " training/output")

os.system("mkdir test")
os.system("mkdir test/input")
os.system("mkdir test/output")

for file in files[int(n*0.9):int(n*0.95)]:
    os.system("mv photo/"+ file + " test/input")
    os.system("mv map/"+ file + " test/output")

os.system("mkdir validation")
os.system("mkdir validation/input")
os.system("mkdir validation/output")

for file in files[int(n*0.95):n]:
    os.system("mv photo/"+ file + " validation/input")
    os.system("mv map/"+ file + " validation/output")

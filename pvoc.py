
from  pascal_voc_writer import Writer
import os
path = "../asl-alphabet/asl_alphabet_train/B/"

for filename in os.listdir(path):
    if filename.endswith("jpg"):
        writer = Writer(path, 200,200)

        writer.addObject('a',0,0,200,200)
        name = os.path.splitext(filename)[0]
        print(path + name+'.xml')
        # writer.save(path + name+'.xml')

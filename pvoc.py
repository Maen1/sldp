
from  pascal_voc_writer import Writer
import os
#path = "../tf-dataset/test/"
path = "../asl-alphabet/asl_alphabet_test/"

for filename in os.listdir(path):
    if filename.endswith("jpg"):
        name = os.path.splitext(filename)[0]
        object = name[0]
        writer = Writer(path, filename, 200,200)

        writer.addObject(object,0,0,200,200)
        # print(path + name+'.xml')
        writer.save(path + name+'.xml')

print("done...")


from  pascal_voc_writer import Writer

path = "/Downloads/asl-alphabet/asl_alphabet_train/A/A1.jpg"
writer = Writer(path, 200,200)

writer.addObject('a',50,50,150,150)

writer.save('./a.xml')

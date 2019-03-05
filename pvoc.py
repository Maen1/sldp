
from  pascal_voc_writer import Writer
import os
path = "../asl-alphabet/asl_alphabet_train/B/"
# writer = Writer(path, 200,200)

# writer.addObject('a',0,0,200,200)

# writer.save('./a.xml')
for filename in os.listdir(path):
    if filename.endswith("jpg"): 
        # Your code comes here such as
        name = os.path.splitext(filename)[0]
        print(name)

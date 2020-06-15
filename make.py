from __future__ import print_function

import numpy as np
from PIL import Image
import os
import sys
import glob
import shutil

from argparser import args

train_examples_per_shape = args.n
test_examples_per_shape = int(max(1, train_examples_per_shape / 10))

image_w = 400
image_h = 400
shape_w = 50
shape_h = 50

max_x_noise = args.noise

max_y_noise = args.noise
#50 - 48%, 39%
#100 - 67$, 58%
#150 - 80%, 77%
#200 - 80%, 84%, 81%

hor_locations = {
    "left": args.e,#int(image_w / 4.0 - (shape_w / 2.0)),
    "right": image_w - shape_w - args.e,#int(3 * (image_w / 4.0) - (shape_w / 2.0)),
    "center": int(image_w / 2.0 - (shape_w / 2.0))    
}
top_location = int(image_h / 2.0 - (shape_h / 2.0))

sim_id_t = 0

shapes = {}

def clean_up(output_dir):
    for f in glob.glob('{}/*'.format(output_dir)):
        shutil.rmtree(f)
        
def gen_stimulus(shape_no, location, output_dir, category, scramble = False):
    global sim_id_t, shapes, shape_h, shape_w
    
    img = Image.new(mode='RGB', size=(image_w, image_h), color=(255, 255, 255))
    if shape_no in shapes:
        shape = shapes[shape_no]
    else:
        shape = Image.open("{}/{}.png".format(shapes_dir, shape_no))
        #shape = shape.convert("L")
        #shape = ImageOps.invert(shape)
        shape.thumbnail((shape_h, shape_w))                
        shapes[shape_no] = shape
               
    noise_x = int((max_x_noise / 2.0) - np.random.rand() * max_x_noise)
    noise_y = int((max_y_noise / 2.0) - np.random.rand() * max_y_noise)
    
    x = hor_locations[location] + noise_x
    y = top_location + noise_y
    img.paste(shape,box=(x, y))

    shape_output_dir = "{}/{}".format(output_dir, category)
    if not(os.path.exists(shape_output_dir)):
        os.makedirs(shape_output_dir)
        
    sim_id_t += 1

    stim_id = sim_id_t
    stim_file = "{}/{}.png".format(shape_output_dir, stim_id)
    img.save(stim_file)
    
#cleaning up trainng and test set    
clean_up("./data/train/{}.{}".format(args.e, args.m))
clean_up("./data/test/{}.{}".format(args.e, args.m))
    
#Ryan's data, Leek    
shapes_dir = "./stimuli/"
print("Generating train set (n = {}) ...".format(train_examples_per_shape), end='')
sys.stdout.flush()
for i in range(train_examples_per_shape):
    
    if (args.m == 1):
        for shape_no in range(1, 13):
            gen_stimulus(shape_no, "center", "./data/train/{}.{}.{}".format(args.e, args.m, args.noise), category = 1)
            gen_stimulus(shape_no + 12, "center", "./data/train/{}.{}.{}".format(args.e, args.m, args.noise), category = 2)
            
    if (args.m == 2):
        for shape_no in range(1, 7):
            gen_stimulus(shape_no, "left", "./data/train/{}.{}.{}".format(args.e, args.m, args.noise), category = 1)
        for shape_no in range(7, 13):
            gen_stimulus(shape_no, "right", "./data/train/{}.{}.{}".format(args.e, args.m, args.noise), category = 1)
        for shape_no in range(13, 19):
            gen_stimulus(shape_no, "left", "./data/train/{}.{}.{}".format(args.e, args.m, args.noise), category = 2)
        for shape_no in range(19, 25):
            gen_stimulus(shape_no, "right", "./data/train/{}.{}.{}".format(args.e, args.m, args.noise), category = 2)

print("Done.")

print("Generating test set (n = {}) ...".format(test_examples_per_shape), end='')
sys.stdout.flush()
for _ in range(test_examples_per_shape):
    if (args.m == 1):
        for shape_no in range(1, 13):
            gen_stimulus(shape_no, "right", "./data/test/{}.{}.{}".format(args.e, args.m, args.noise), category = 1)
            gen_stimulus(shape_no + 12, "right", "./data/test/{}.{}.{}".format(args.e, args.m, args.noise), category = 2)

            
    if (args.m == 2):
        for shape_no in range(1, 7):
            gen_stimulus(shape_no, "right", "./data/test/{}.{}.{}".format(args.e, args.m, args.noise), category = 1)
        for shape_no in range(7, 13):
            gen_stimulus(shape_no, "left", "./data/test/{}.{}.{}".format(args.e, args.m, args.noise), category = 1)
        for shape_no in range(13, 19):
            gen_stimulus(shape_no, "right", "./data/test/{}.{}.{}".format(args.e, args.m, args.noise), category = 2)
        for shape_no in range(19, 25):
            gen_stimulus(shape_no, "left", "./data/test/{}.{}.{}".format(args.e, args.m, args.noise), category = 2)
            
print("Done")

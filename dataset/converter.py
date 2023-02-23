from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import asarray
from numpy import savetxt


def custom_mask( input_data ):
    size_data = len( input_data )
    np_output = np.zeros( size_data )
    int_index = 0
    for img_intensity in input_data:
        converted_intensity = 255 - img_intensity
        np_output[int_index] = converted_intensity
        int_index = int_index + 1
    return np_output


filepath = os.getcwd()
sample_path = filepath + "\\sample"
output_path = filepath + "\\out"

list_img = [ f for f in listdir(sample_path) if isfile( join(sample_path, f) ) ]

print("Detected Images: ")
for file_index in list_img:
    print( file_index )

max_iteration = len( list_img )
print( "The Number of Images: ", max_iteration )

output_list = list()
for int_index in range(max_iteration):
    current_path = sample_path + "\\" + str(list_img[int_index])
    current_img = Image.open(current_path)
    resized_img = current_img.resize((28,28))
    converted_img = resized_img.convert('L')
    np_img = (np.array( converted_img )).flatten()
    np_converted_img = custom_mask( np_img )
    output_list.append(np_converted_img)
    
    current_out = output_path + "\\" + str(int_index) + ".png" 
    converted_img.save(current_out)

output_np = np.asarray( output_list )
print("Integrity Check: ")
print("\t\t1st type: ", type(output_np))
for np_item in output_np:
    print("\t\t2nd type: ", type(np_item))

print("========================")
print("Example: ")
print(output_np[0])

savetxt("dataset.csv", output_np, delimiter=',')

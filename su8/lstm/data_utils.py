
import os
import numpy


from numpy import array
from numpy import split
from numpy import transpose


pi = 3.1415926535897932385

def load_data(data_path):

    input_data = []
    output_data = []

    with open(data_path, 'r') as input:
        lines = input.readlines()
        
        for k in range(0, len(lines)-1,2):
            
            # input: csv file
	    # Row order
            # U_n ... U_1
            # U
 
            output_values = lines[k].strip().split(',')
            input_values = lines[k+1].strip().split(',')

            output_values = [float(z) for z in output_values]
            input_values = [float(z) for z in input_values]

            input_values = split(array(input_values),8)

            # 64 x 10 outputs
            # split into 80 x 8
            output_values = array(output_values)

            input_data.append(input_values)
            output_data.append(output_values)

    return (array(input_data), array(output_data))







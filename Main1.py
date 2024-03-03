#!/usr/bin/python
'''
    Main File.
'''
import os
import sys
import pandas as pd

from interpolation  import interpolate  # assuming interpolate is a function in interpolate module
from normalization import normalize  # assuming normalize is a function in normalize module

def main(file_path, output_dir):
    '''
        Run Pipeline of processes on a single file.
    '''

    file_dataframe = pd.read_csv(file_path)

    cols = ['Rain Fall (mm)','Fertilizer(urea) (kg/acre)','Temperature (°C)','Nitrogen (N)','Phosphorus (P)','Potassium (K)','Yeild (Q/acre)']  # You might want to specify columns for interpolate and normalize

    file_dataframe = interpolate(file_dataframe, cols)

    file_dataframe = normalize(file_dataframe, cols)

    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, file_name)
    file_dataframe.to_csv(output_file_path, encoding='utf-8')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_directory")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

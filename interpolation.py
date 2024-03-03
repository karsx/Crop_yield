#!/usr/bin/env python

import os
import sys
import pandas as pd

def interpolate_yield(dataframe, cols_to_interpolate):
    for col in cols_to_interpolate:
        dataframe[col] = dataframe[col].interpolate('spline', order=2)
    return dataframe

def main(dir_path):
    files = os.listdir(dir_path)
    
    for file_name in files:
        dataframe = pd.read_csv(os.path.join(dir_path, file_name))
        cols_to_interpolate = ['Rain Fall (mm)', 'Fertilizer(urea) (kg/acre)', 'Temperature (°C)', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Yeild (Q/acre)']
        dataframe = interpolate_yield(dataframe, cols_to_interpolate)
        print(dataframe)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py directory_path")
        sys.exit(1)
    main(sys.argv[1])

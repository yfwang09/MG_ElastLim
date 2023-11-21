import glob
import numpy as np
import pandas as pd

# Read all the txt files in the folder
files = glob.glob('*.txt')

# Iterate through the files
for file in files:
    print(file)
    try:
        # Read the txt file into numpy array
        header  = np.genfromtxt(file, dtype=str, delimiter=',', max_rows=1, comments=None).tolist()
        rawdata = np.loadtxt(file, skiprows=1)
    except:
        print('Error reading file: ' + file)
        continue
    # convert the numpy array into pandas dataframe
    print(header)
    header_list = header + ['%d'%(k+1) for k in range(rawdata.shape[1] - len(header))]
    df = pd.DataFrame(rawdata, columns=header_list)
    
    print(header_list)
    # print(df)

    # Write the dataframe into excel file
    df.to_excel(file[:-4] + '.xlsx', index=False, header=header_list)

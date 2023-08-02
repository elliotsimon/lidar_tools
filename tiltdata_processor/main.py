__author__ = 'Elliot I. Simon'
__email__ = 'ellsim@dtu.dk'
__version__ = 'May 16, 2022'

import sys
import os
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob


def init(lidar=sys.argv[1], inpath=sys.argv[2], outpath=sys.argv[3]):
    #print(lidar, inpath, outpath)
    pd.options.display.float_format = '{:.5f}'.format
    plt.rcParams['figure.figsize'] = [10, 8]

    # Globe #3 doesn't have the Gyro. All the rest do
    if lidar == 'Globe3':
        numchannels = 11
    else:
        numchannels = 26

    # Offsets to each system for X_inc, Y_inc, Pitch, Roll when turbine is shutdown
    if lidar == 'Globe1':
        offsets = {"X_inc": -0.154947536334635, "Y_inc": -0.177644474477131, "Pitch": -0.214801533144300,
                   "Roll": -0.464211139666754}
    elif lidar == 'Globe2':
        offsets = {"X_inc": -0.255054184824409, "Y_inc": 0.113357280184561, "Pitch": -0.461399496923802,
                   "Roll": -0.174543354268145}
    elif lidar == 'Globe3':
        offsets = {"X_inc": -0.353579546911074, "Y_inc": 0.188849477698021}
    elif lidar == 'Globe4':
        offsets = {"X_inc": -0.021659047603667, "Y_inc": 0.153910290983092, "Pitch": 0.065695291306036,
                   "Roll": 0.428415175688223}
    elif lidar == 'Globe5':
        offsets = {"X_inc": -0.158326260830941, "Y_inc": 0.070065860364364, "Pitch": -0.281081676294176,
                   "Roll": 0.028367376694063}
    elif lidar == 'Globe6':
        offsets = {"X_inc": -0.152711623352406, "Y_inc": 0.034443344317334, "Pitch": -0.293758098109923,
                   "Roll": -0.361286697960697}
    elif lidar == 'Globe7':
        offsets = {"X_inc": -0.124288173237089, "Y_inc": -0.011538277345918, "Pitch": -0.390946279844547,
                   "Roll": 0.048530260966130}
    return lidar, inpath, outpath, numchannels, offsets


def extract(inpath, outpath):
    # Find all zipfiles in path
    zipfiles = glob.glob(inpath + '\**\*.zip', recursive=True)
    #print(zipfiles)

    # Windows method to unzip (only extract .tim files from archive)
    print('..Extracting..')
    for zf in zipfiles:
        with zipfile.ZipFile(zf,"r") as zip_ref:
            [zip_ref.extract(file, path=outpath+'\\bin') for file in zip_ref.namelist() if file.endswith('.tim')]
    print('..Done!..')


def bin2asc(outpath, numchannels):
    binfiles = glob.glob(outpath + '\\bin\\' + '*.tim')
    #print(binfiles)

    # Run binary to ascii conversion on all binary files
    print('..Converting binary to ascii..')
    for f in binfiles:
        asciifile = f.split('\\')[-1].split('.tim')[0] + '.tsv'
        asciifile = outpath + '\\ascii\\' + asciifile
        #print(asciifile)
        os.system("bin2asc.exe {0} {1} {2} {3}".format(f, asciifile, numchannels, "%6.3f"))
    print('..Done!..')


def collate_format(lidar, offsets, outpath):
    # List converted ascii files
    tsvfiles = glob.glob(outpath + '\\ascii\\*.tsv')
    tsvfiles = sorted(([str(f) for f in tsvfiles]))
    #print(tsvfiles)

    # Load all data into dataframe
    df = pd.concat(map(lambda f: pd.read_csv(f, delim_whitespace=True, header=None), tsvfiles))

    # Rename columns to match channel names
    if lidar == 'Globe3':
        df.columns = ['YY', 'MT', 'DD', 'HH', 'MM', 'SS', 'RTC_MS', 'Temp_inc', 'X_inc', 'Y_inc', 'Inc_stat']
    else:
        df.columns = ['YY', 'MT', 'DD', 'HH', 'MM', 'SS', 'RTC_MS', 'Gyro_MS', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'HX',
                      'HY', 'HZ',
                      'Roll', 'Pitch', 'Yaw', 'Temp_gyro', 'Gyro_stat', 'Temp_inc', 'X_inc', 'Y_inc', 'Inc_stat']

    # Apply offsets
    if lidar == 'Globe3':
        df['X_inc'] -= offsets['X_inc']
        df['Y_inc'] -= offsets['Y_inc']
    else:
        df['X_inc'] -= offsets['X_inc']
        df['Y_inc'] -= offsets['Y_inc']
        df['Pitch'] -= offsets['Pitch']
        df['Roll'] -= offsets['Roll']

    # Fix edge case, where timestamp is NaN
    df = df[df['SS'].notna()]

    # Force conversion to int to remove trailing decimals, then force conversion to str to comply with datetime formatter

    df['YY'] = df['YY'].astype(int).astype(str).apply(lambda x: x.zfill(4))
    df['MT'] = df['MT'].astype(int).astype(str).apply(lambda x: x.zfill(2))
    df['DD'] = df['DD'].astype(int).astype(str).apply(lambda x: x.zfill(2))
    df['HH'] = df['HH'].astype(int).astype(str).apply(lambda x: x.zfill(2))
    df['MM'] = df['MM'].astype(int).astype(str).apply(lambda x: x.zfill(2))
    df['SS'] = df['SS'].astype(int).astype(str).apply(lambda x: x.zfill(2))
    # Milliseconds must be converted to microseconds and zero-padded to use with to_datetime() %f string formatter
    df['US'] = (df['RTC_MS']*1000).astype(int).astype(str).apply(lambda x: x.zfill(6))

    # Build datetime string from clock channels
    df['dt'] = pd.to_datetime(df['YY'] + '-' + df['MT'] + '-' + df['DD'] + ' ' +
                              df['HH'] + ':' + df['MM'] + ':' +
                              df['SS'] + ':' + df['US'],
                              format='%Y-%m-%d %H:%M:%S:%f')

    df.set_index(df['dt'], inplace=True)

    # Delete unneeded columns to save space
    del df['dt']
    del df['YY']
    del df['MT']
    del df['DD']
    del df['HH']
    del df['MM']
    del df['SS']
    del df['US']

    print('Final DataFrame shape = ' + str(df.shape))

    return df


def save_hdf(df, outpath, lidar):
    # Save result to hdf5 file
    # Make filename from the time range contained in the dataframe
    outfile = str(df.first_valid_index()).split('.')[0].replace(' ', '_').replace('-', '_').replace(':', '_') + '-' + \
              str(df.last_valid_index()).split('.')[0].replace(' ', '_').replace('-', '_').replace(':', '_')
    outfile = outpath + '\\hdf\\' + lidar + '_motiondata_' + outfile + '.hdf'
    print(outfile)
    df.to_hdf(outfile, key='df', complib='blosc')
    print('..Saved HDF5 file!..')


def plot(df, lidar, outpath):
    if lidar == 'Globe3':
        f, axarr = plt.subplots(2, 1, sharex=True)
        plt.sca(axarr[0])
        df['X_inc'].plot(c='c')
        plt.sca(axarr[1])
        df['Y_inc'].plot(c='m')
    else:
        f, axarr = plt.subplots(10, 1, sharex=True)
        plt.sca(axarr[0])
        df['X_inc'].plot(c='c')
        plt.sca(axarr[1])
        df['Y_inc'].plot(c='m')
        plt.sca(axarr[2])
        df['Pitch'].plot(c='y')
        plt.sca(axarr[3])
        df['Roll'].plot(c='b')
        plt.sca(axarr[4])
        df['AX'].plot(c='k')
        plt.sca(axarr[5])
        df['AY'].plot(c='r')
        plt.sca(axarr[6])
        df['AZ'].plot(c='g')
        plt.sca(axarr[7])
        df['GX'].plot(c='brown')
        plt.sca(axarr[8])
        df['GY'].plot(c='purple')
        plt.sca(axarr[9])
        df['GZ'].plot(c='grey')

    start = str(df.first_valid_index().replace(microsecond=0))
    end = str(df.last_valid_index().replace(microsecond=0))
    plt.suptitle(lidar + ' motion data between: ' + start + ' and ' + end)
    f.legend(loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=5, fancybox=True, shadow=True)
    start = start.replace(':', '_')
    start = start.replace(' ', '_')
    end = end.replace(':', '_')
    end = end.replace(' ', '_')
    plt.savefig(outpath + '\\plots\\' + str(lidar) + '_' + start + '_' + end + '.png', dpi=250)
    print('..Saved plot!..')


def purge_temp_data(outpath):
    # Delete temporary bin and ascii files
    files = glob.glob(outpath + '\\ascii\\*.tsv')
    for f in files:
        os.remove(f)
    files = glob.glob(outpath + '\\bin\\*.tim')
    for f in files:
        os.remove(f)
    print('..Purged temp files!..')


def main():
    # Initialize parameters
    lidar, inpath, outpath, numchannels, offsets = init()
    # Extract .tim files from .zips
    extract(inpath, outpath)
    # Convert binary to ascii format
    bin2asc(outpath, numchannels)
    # Collate data files and format into pandas DataFrame
    df = collate_format(lidar, offsets, outpath)
    # Save dataframe to HDF5 file
    save_hdf(df, outpath, lidar)
    # Make and save plots
    plot(df, lidar, outpath)
    # Purge temp data (.tim and .tsv files)
    purge_temp_data(outpath)


if __name__ == '__main__':
    main()

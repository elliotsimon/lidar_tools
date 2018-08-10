#!/usr/bin/python

__author__ = 'Elliot Simon'
__email__ = 'ellsim@dtu.dk'
__date__ = 'July 15, 2018'
__credits__ = ["DTU-Risø"]

'''
Used for reconstructing DBS (Doppler beam swing) lidar measurements in 5-beam configuration.
Useage is from commandline, with arguments:
path to data file (windscanner format), cone angle, cnr lower limit, cnr upper limit, radial speed lower limit,
radial speed upper limit
'''

# Imports

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import datetime
import xarray as xr
import argparse
import matplotlib.dates as mdates

# Configs

plt.style.use('classic')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Serif'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['savefig.dpi'] = 500

# Functions


# Function to get the starting column number for a specific range gate
def get_col_num(range_gate):
    spacing = range_gates[1] - range_gates[0]
    colnum = int((((range_gate - range_gates[0])/spacing)*4)+8)
    return colnum


# Function to convert labview to datetime stamp
# Time is stamped at beginning of 10min period in UTC format!!!
def convtime(labviewtime):
    unixtime = labviewtime - 2082844800
    timestamp = datetime.datetime.utcfromtimestamp(int(unixtime))
    # - pd.Timedelta(minutes=10)
    # + pd.Timedelta(hours=1)
    return timestamp

# Inputs

# Get inputs from command line arguments, or use default values
parser = argparse.ArgumentParser(description='Reconstruct 5-beam DBS from windscanner measurements')
parser.add_argument('filename', nargs='?', type=str, help='The file path to the measurements')
parser.add_argument('cone_angle', nargs='?', type=int, default=15, help='The DBS cone angle in degrees, default = 15 deg')
parser.add_argument('range_gate', nargs='?', type=int, default=50, help='The range gate to make plots and DataFrame for (all are reconstructed, only one is plotted and extracted)')
parser.add_argument('angle_tolerance', nargs='?', type=int, default=1, help='The tolerance (in degrees) for selecting beams. Applies to both azimuth and elevation. Default = ` degree')
parser.add_argument('cnr_low', nargs='?', type=int, default=-26, help='The lower bound CNR limit to use. default = -26 dB')
parser.add_argument('cnr_high', nargs='?', type=int, default=0, help='The upper bound CNR limit to use. default = 0 dB')
parser.add_argument('rs_low', nargs='?', type=int, default=-20, help='The lower bound RadSpeed limit to use. default = -20 m/s')
parser.add_argument('rs_high', nargs='?', type=int, default=20, help='The upper bound RadSpeed limit to use. default = 20 m/s')
args = vars(parser.parse_args())

# Get scenario name
scn_name = str(args['filename'].split('\\')[-1].split('_')[0])
print(scn_name)

# Read in windscanner wind_data file
df_ws = pd.read_csv(args['filename'], sep=';', header=None)
print('File input complete. Shape = ')
print(df_ws.shape)

# Data pre-processing

# If file is not all complete scans, then chop partial scan at end
if len(df_ws) % 5 != 0:
    num_to_chop = len(df_ws) % 5
    df_ws = df_ws.iloc[:-num_to_chop]
    print('Chopped! New shape = ')
    print(df_ws.shape)
else:
    print('No need to chop!')

# Rename certain columns for ease of reading
df_ws = df_ws.rename(columns={4: 'dt_start', 5: 'dt_stop', 6: 'azim', 7: 'elev'})

# Get timestamp of first and last row
start = convtime(df_ws['dt_stop'].iloc[0])
end = convtime(df_ws['dt_stop'].iloc[-1])
print('Scenario times = ')
print('Start: ' + start.strftime("%Y-%m-%d %H:%M:%S"))
print('End: ' + end.strftime("%Y-%m-%d %H:%M:%S"))

# Total time of scenario
print('Total measurement time: ')
print(end - start)

# Apply time conversion
df_ws['dt_start'] = df_ws['dt_start'].apply(lambda x: convtime(x))
df_ws['dt_stop'] = df_ws['dt_stop'].apply(lambda x: convtime(x))

# Gets list of range gates to use for plotting radial distances
range_gates = df_ws.iloc[0:1,8::4].values[0].tolist()

# Check that the selected range gate is included in the measurements
if not args['range_gate'] in range_gates:
    print(range_gates)
    raise ValueError('Error! The measurement file is not standard. Range gates are not correct!')
else:
    print('Range gate selected: ' + str(args['range_gate']) + ' is valid! ')

# Filtering

# Loops through list of range gates. Applies CNR mask to set radspeed & CNR to NaN if CNR below or above threshold
for dist in range_gates:
    rg = get_col_num(dist)
    mask = df_ws[rg+2] < args['cnr_low']
    df_ws.loc[mask, rg+1] = np.nan
    df_ws.loc[mask, rg+2] = np.nan

    mask = df_ws[rg+2] > args['cnr_high']
    df_ws.loc[mask, rg+1] = np.nan
    df_ws.loc[mask, rg+2] = np.nan

# Loops through list of range gates. Applies RS mask to set radspeed & CNR to NaN if RS below or above threshold
for dist in range_gates:
    rg = get_col_num(dist)
    mask = df_ws[rg+1] < args['rs_low']
    df_ws.loc[mask, rg+1] = np.nan
    df_ws.loc[mask, rg+2] = np.nan

    mask = df_ws[rg+1] > args['rs_high']
    df_ws.loc[mask, rg+1] = np.nan
    df_ws.loc[mask, rg+2] = np.nan

# Data quality checks

# Beam positions hardcoded for now
north = [75, 0]
south = [105, 0]
east = [105, 90]
west = [75, 90]
vertical = [90]

# Compute angle ranges for selecting rows
n_high = [x + args['angle_tolerance'] for x in north]
n_low = [x - args['angle_tolerance'] for x in north]
s_high = [x + args['angle_tolerance'] for x in south]
s_low = [x - args['angle_tolerance'] for x in south]
e_high = [x + args['angle_tolerance'] for x in east]
e_low = [x - args['angle_tolerance'] for x in east]
w_high = [x + args['angle_tolerance'] for x in west]
w_low = [x - args['angle_tolerance'] for x in west]
v_high = [x + args['angle_tolerance'] for x in vertical]
v_low = [x - args['angle_tolerance'] for x in vertical]

# Check that all indexes for all beams are properly separated by 5 integer positions
north_check = df_ws[((df_ws['elev'] > n_low[0]) & (df_ws['elev'] < n_high[0])) & ((df_ws['azim'] > n_low[1]) & (df_ws['azim'] < n_high[1]))].iloc[:,9::4].index.values
south_check = df_ws[((df_ws['elev'] > s_low[0]) & (df_ws['elev'] < s_high[0])) & ((df_ws['azim'] > s_low[1]) & (df_ws['azim'] < s_high[1])) ].iloc[:,9::4].index.values
east_check = df_ws[((df_ws['elev'] > e_low[0]) & (df_ws['elev'] < e_high[0])) & ((df_ws['azim'] > e_low[1]) & (df_ws['azim'] < e_high[1])) ].iloc[:,9::4].index.values
west_check = df_ws[((df_ws['elev'] > w_low[0]) & (df_ws['elev'] < w_high[0])) & ((df_ws['azim'] > w_low[1]) & (df_ws['azim'] < w_high[1])) ].iloc[:,9::4].index.values
vertical_check = df_ws[(df_ws['elev'] > v_low[0]) & (df_ws['elev'] < v_high[0])].iloc[:,9::4].index.values

checks = [north_check, south_check, east_check, west_check, vertical_check]
for d, e in enumerate(checks):
    diff = np.diff(e)
    if np.unique(diff).size != 1:
        #print(count)
        raise ValueError(
            'Data dimensions are incorrect. Beams are not consistently separated by 5 rows. Error occurs at index: ')
print('All good! Quality check passed!')

# Reconstruction

# Initialize empty ndarrays with correct dimensions
l = int(len(df_ws) / 5 - 1)
north = np.empty(((len(df_ws)-5), len(range_gates)))
south = np.empty(((len(df_ws)-5), len(range_gates)))
east = np.empty(((len(df_ws)-5), len(range_gates)))
west = np.empty(((len(df_ws)-5), len(range_gates)))
vertical = np.empty(((len(df_ws)-5), len(range_gates)))

# Loop over possible combination of beam pairs, select all corresponding points for given beam angles
# Take only values, and trim to correct length
for i in range(0,5,1):
    #print(i)
    north[i::5] = df_ws.iloc[i:,:][((df_ws['elev'] > n_low[0]) & (df_ws['elev'] < n_high[0])) & ((df_ws['azim'] > n_low[1]) & (df_ws['azim'] < n_high[1])) ].iloc[:,9::4].values[:l,:]
    south[i::5] = df_ws.iloc[i:,:][((df_ws['elev'] >  s_low[0]) & (df_ws['elev'] < s_high[0])) & ((df_ws['azim'] > s_low[1]) & (df_ws['azim'] < s_high[1])) ].iloc[:,9::4].values[:l,:]
    east[i::5] = df_ws.iloc[i:,:][((df_ws['elev'] >  e_low[0]) & (df_ws['elev'] < e_high[0])) & ((df_ws['azim'] > e_low[1]) & (df_ws['azim'] < e_high[1])) ].iloc[:,9::4].values[:l,:]
    west[i::5] = df_ws.iloc[i:,:][((df_ws['elev'] > w_low[0]) & (df_ws['elev'] < w_high[0])) & ((df_ws['azim'] > w_low[1]) & (df_ws['azim'] < w_high[1])) ].iloc[:,9::4].values[:l,:]
    vertical[i::5] = df_ws.iloc[i:,:][((df_ws['elev'] > v_low[0]) & (df_ws['elev'] < v_high[0]))].iloc[:,9::4].values[:l,:]


# Check result shapes
if (north.shape == south.shape == east.shape == west.shape == vertical.shape):
    print('All good! Shapes match')
else:
    print(north.shape)
    print(south.shape)
    print(east.shape)
    print(west.shape)
    print(vertical.shape)
    raise ValueError('Something went wrong! Shapes do not match! ')


# Calculations for u,v,w (using other 4 beams)
# Also horizontal wind speed and direction
u = (west - east) / (2 * np.sin(np.deg2rad(args['cone_angle'])))
v = (north - south) / (2 * np.sin(np.deg2rad(args['cone_angle'])))
w = vertical
w_calc = (north + south + east + west) / (4 * np.cos(np.deg2rad(args['cone_angle'])))
wsp = np.sqrt(u**2 + v**2)
# There is an offset in wind direction
wdir = np.rad2deg(np.arctan2(u, v)) + 180
# Force wdir to proper range (0,360)
wdir = (wdir + 360) % 360

# Data formatting

# Get index from original dataframe
time_index = df_ws.iloc[0:-5]['dt_stop']

# Stack all variables into ndarray
ar = np.array([u,v,w,w_calc,wsp,wdir])
#print(ar.shape)

# Reshape ndarray to have dimensions (time x variables x heights)
ar = np.moveaxis(ar, 1, 0)
#print(ar.shape)

# Construct xarray DataArray with dimensions (time x variables x heights)
da_result = xr.DataArray(ar, coords=[time_index, ['u','v','w','w_calc','wsp','wdir'], range_gates],
                         dims=['time', 'variables', 'height'])

# Format filename using first and last valid timestamps
fname = da_result.indexes['time'][0].strftime('%Y-%m-%d_%H-%M-%S') + '_to_' + \
da_result.indexes['time'][-1].strftime('%Y-%m-%d_%H-%M-%S') + '_DBS_Brise_Reconstructed'
#print(fname)

# Save result dataarray to disk
da_result.to_netcdf(fname + '.nc')

# Extract provided range gate and save as pandas DataFrame
da_result.loc[:,:,args['range_gate']].to_pandas().to_hdf(
    fname + '_' + str(args['range_gate']) + 'm_RG' + '.hdf', 'df', mode='w', complib='blosc')

# Plotting

# Plot 1st range gate (after filtering) CNR & RS and save to file
f, axarr = plt.subplots(2,1, sharex=True, figsize=(10,6))
axarr[0].plot(df_ws[get_col_num(args['range_gate'])+1])
axarr[0].set_title('Radial speed, 1st range gate, after filtering')
axarr[1].plot(df_ws[get_col_num(args['range_gate'])+2])
axarr[1].set_title('CNR, 1st range gate, after filtering')
plt.suptitle('Scenario: ' + str(scn_name) + '\n' + 'Times: ' + start.strftime("%Y-%m-%d %H:%M:%S") + ' to ' +
             end.strftime("%Y-%m-%d %H:%M:%S") + '\n' + 'Length: ' + str(end-start) +
             '\n Elliot Simon, ellsim@dtu.dk', x=0.5, y=.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.8])
plt.savefig(fname + '_cnr_rs_filt_50m.png')

# Plot all reconstruction results and save to file
times = da_result.indexes['time']
f, axarr = plt.subplots(5,1, sharex=True, figsize=(10,14))
axarr[0].plot_date(times, da_result.loc[:,'wsp',args['range_gate']], marker='.')
axarr[0].set_title('lidar DBS wind speed')
axarr[1].plot_date(times, da_result.loc[:,'wdir',args['range_gate']], marker='.')
axarr[1].set_title('lidar DBS wind direction')
axarr[2].plot_date(times, da_result.loc[:,'u',args['range_gate']], marker='.')
axarr[2].set_title('lidar DBS u-component')
axarr[3].plot_date(times, da_result.loc[:,'v',args['range_gate']], marker='.')
axarr[3].set_title('lidar DBS v-component')
axarr[4].plot_date(times, da_result.loc[:,'w',args['range_gate']], marker='.')
axarr[4].set_title('lidar DBS w-component')
formatter = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
f.axes[0].xaxis.set_major_formatter(formatter)
f.autofmt_xdate()
plt.suptitle('Scenario: ' + str(scn_name) + '\n' + 'Times: ' + start.strftime("%Y-%m-%d %H:%M:%S") + ' to ' +
             end.strftime("%Y-%m-%d %H:%M:%S") + '\n' + 'Length: ' + str(end-start) +
             '\n Elliot Simon, ellsim@dtu.dk', x=0.58, y=.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.86])
plt.savefig(fname + '_results_50m.png')
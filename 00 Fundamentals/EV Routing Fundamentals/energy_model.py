
"""
Course: Data-Driven Algorithms in Vehicle Technology

Exercise: Building a simple energy model

@author: Stefan Scheubner

"""

#%% load required modules
import matplotlib.pyplot as plt # for visualization
import numpy as np # for calculation
from scipy.signal import butter, filtfilt # for noise reduction
import pandas as pd # for data analysis

#%% load measurement data

# Load data from CSV
measurement_data_path = os.path.join(os.getcwd(), '02_measurement_data_energy_model.csv')
data_raw = pd.read_csv(measurement_data_path)
vel = data_raw['velocityInMps']
dist = data_raw['distanceInKm']
gradient = data_raw['gradientInPercent']

# Plot
plt.figure(figsize=(6.8, 4.2))
plt.plot(dist, gradient,c='b')
plt.xlabel('Distance in km')
plt.ylabel('gradient in %')
plt.title('Original Sensor Data Gradient')

# Plot
plt.figure(figsize=(6.8, 4.2))
plt.plot(dist, vel,c='r')
plt.xlabel('Distance in km')
plt.ylabel('velocity in m/s')
plt.title('Original Sensor Data Velocity')

#%% PART ONE 
# reduce measurement noise in road grade angle

# filter the measurement data 
# example code taken from https://www.delftstack.com/howto/python/low-pass-filter-python/
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Tune your filter:
order = 5         # order of the filter, e.g. "1"
fs = 100           # sampling frequency in Hz, e.g. "10"
cutoff = 0.1        # cutoff frequency in Hz, e.g. "1"

# filter the data
gradient_filtered = butter_lowpass_filter(gradient, cutoff, fs, order)

# Plot
plt.figure(figsize=(6.8, 4.2))
plt.plot(dist, gradient,c='b')
plt.plot(dist, gradient_filtered,c='r')
plt.xlabel('Distance in km')
plt.ylabel('road grade in %')
plt.title('Filtered Data Gradient')

#%% reduce measurement noise in velocity

# Tune your filter:
order = 5         # order of the filter, e.g. "1"
fs = 100            # sampling frequency in Hz, e.g. "10"
cutoff = 0.5        # cutoff frequency in Hz, e.g. "1"

# filter the data
velocity_filtered = butter_lowpass_filter(vel, cutoff, fs, order)

# Plot
plt.figure(figsize=(6.8, 4.2))
plt.plot(dist, vel,c='b')
plt.plot(dist, velocity_filtered,c='r')
plt.xlabel('Distance in km')
plt.ylabel('velocity in m/s')
plt.title('Filtered Data Velocity')


#%% PART TWO 
# vehicle data
# choose a vehicle you like and type in its parameters or create your own
# Taycan from https://automobil-guru.de/cw-werte-tabelle-stirnflaeche/ and porsche.de

m = 2350                 # vehicle mass in kg
f_r = 0.015              # rolling resistance value 
g = 9.81 #m/s^2       # gravity
c_w = 0.22              # aeordynamic resistance
A = 2.23 #               # frontal area in m^2    
rho = 1.2 #kg/m^3     # air density
Emax_bat = 90          # maximum battery energy in kWh

P_aux = 2000             # power of the auxiliaries in W


#%% driving resistance calculation

# vehicle acceleration derived from velocity
# comment: imagine that without the noise reduction!
acc = np.diff(velocity_filtered)/.01
acc = np.insert(acc, 0, 0)

# Calculation of the driving resistances
F_roll = [f_r*g*m]*len(vel) 
F_aero = np.multiply(velocity_filtered,velocity_filtered)*rho*A*c_w*0.5
F_grade = g*np.sin(np.arctan(gradient_filtered/100))*m
F_acc = m*acc

# Plot the driving resistances
plt.figure(figsize=(6.8, 4.2))
plt.plot(dist, F_roll,c='b',label='F_roll')
plt.plot(dist, F_aero,c='r',label='F_aero')
plt.plot(dist, F_grade,c='c',label='F_grade')
plt.xlabel('Distance in km')
plt.ylabel('Force in N')
plt.legend()
plt.title('Driving Resistances')

#%% energy calculation

eta_PT = 0.7 # efficiency of the whole powertrain (between 0 and 1)

time = np.arange(0,len(vel)*0.01,0.01) # time in s

E_aux = (P_aux*time)/3600/1000 # Energy demand of the auxiliaries in kWh

# Power is Force multiplied by velocity divided by eta
P_res = (F_roll + F_aero + F_grade + F_acc)*velocity_filtered/eta_PT # N*m/s = W
E_tot = np.cumsum(P_res*0.01)/3600/1000 + E_aux #kWh 


# Plot the Power demand
plt.figure(figsize=(6.8, 4.2))
plt.plot(dist, P_res/1000,c='b',label='Resulting Power')
plt.xlabel('Distance in km')
plt.ylabel('Power in kW')
plt.legend()
plt.title('Power Demand')

# Plot the energy consumption
plt.figure(figsize=(6.8, 4.2))
plt.plot(dist, E_aux,c='r',label='E_aux')
plt.plot(dist, E_tot,c='g',label='E_tot')
plt.xlabel('Distance in km')
plt.ylabel('Energy in kWh')
plt.legend()
plt.title('Energy Consumption')

# calculate SoC at the end of the trip
SoC = 1-E_tot[-1]/Emax_bat 

# Print the results
print('----------- Results -----------')
print('SoC at the end of the', round(dist.iloc[-1]), 'km trip is', round(SoC,2), '%')
print('Max. Power demand is', round(max(P_res)/1000), 'kW or roughly',round(max(P_res)/1000*1.36),'horse power')
print('Auxiliaries take',round(E_aux[-1]/E_tot[-1]*100),'% of the total energy')
print('-------------------------------')



# %%

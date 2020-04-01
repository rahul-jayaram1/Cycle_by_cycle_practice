from bycycle.filt import lowpass_filter
from bycycle.features import compute_features

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import scipy.io
import scipy.signal

filename = 'MEG_det_individual_subj_avgs_100_500'
filename = os.path.join('./', filename)
data = sp.io.loadmat(filename)
#srate = data['Fs']
Fs = 600 # sampling rate
#srate = srate[0][0];
signal = data['all_subjs_means']
t = data['plot_time']

signal=signal[4,:]#adjust trial of interest
t=t[0]

tlim = (-.1, .500) #time interval in seconds
tidx = np.logical_and(t>=tlim[0], t<tlim[1])


f_theta = (10,22)
f_lowpass = 100
N_seconds = 0.1
# redefining variable names to follow tutorial

plt.plot(t,signal)
plt.title('raw signal')
#plt.show()


signal_low= lowpass_filter(signal, Fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

#t = np.arange(0, len(signal)/Fs, 1/Fs)
# filename = 'tVec_100_500'
# filename = os.path.join('./', filename)
# t = sp.io.loadmat(filename)
# t = t['plot_time']
# t=t[0]


plt.plot(t[tidx], signal[tidx], '.5')
plt.xlim(tlim)
plt.title('raw signal - trial 1')
plt.show()

plt.plot(t[tidx], signal_low[tidx], 'k')
plt.xlim(tlim)
plt.title('lowpass 100 Hz signal - trial 1')
plt.show()


#localizing peaks and troughs
from bycycle.filt import bandpass_filter
from bycycle.cyclepoints import _fzerorise, _fzerofall, find_extrema

# Narrowband filter signal
N_seconds_theta = 0.02
signal_narrow = bandpass_filter(signal, Fs, f_theta,
                                remove_edge_artifacts=False,
                                N_seconds=N_seconds_theta)

plt.plot(t[tidx], signal[tidx], '.5')
plt.plot(t[tidx], signal_narrow[tidx], 'k')
plt.xlim(tlim)
plt.title('bandpass compared to raw')
plt.show()

#Find rising and falling zerocrossings (narrowband)
zeroriseN = _fzerorise(signal_narrow)
zerofallN = _fzerofall(signal_narrow)

Ps, Ts = find_extrema(signal_low, Fs, f_theta,
    filter_kwargs={'N_seconds':N_seconds_theta})

print(Ps)
print(Ts)

tidxPs = Ps[np.logical_and(Ps>tlim[0]*Fs, Ps<tlim[1]*Fs)]
tidxTs = Ts[np.logical_and(Ts>tlim[0]*Fs, Ts<tlim[1]*Fs)]
#
#plt.figure(figsize=(12, 2))
plt.plot(t[tidx], signal_low[tidx], 'k')
#plt.plot(t[tidx], signal_low[tidx], 'g')
plt.plot(t[tidxPs], signal_low[tidxPs], 'b.', ms=10)
plt.plot(t[tidxTs], signal_low[tidxTs], 'r.', ms=10)
plt.xlim(tlim)
plt.title('lowpass extremas - trial 1')
plt.show()


# from bycycle.filt import bandpass_filter
# bandpass_filter(signal, Fs, (4, 10), N_seconds=.24, plot_frequency_response=True)

from bycycle.cyclepoints import find_zerox
zeroxR, zeroxD = find_zerox(signal_low, Ps, Ts)

tidx = np.logical_and(t>=tlim[0], t<tlim[1])
tidxPs = Ps[np.logical_and(Ps>tlim[0]*Fs, Ps<tlim[1]*Fs)]
tidxTs = Ts[np.logical_and(Ts>tlim[0]*Fs, Ts<tlim[1]*Fs)]
tidxDs = zeroxD[np.logical_and(zeroxD>tlim[0]*Fs, zeroxD<tlim[1]*Fs)]
tidxRs = zeroxR[np.logical_and(zeroxR>tlim[0]*Fs, zeroxR<tlim[1]*Fs)]

plt.plot(t[tidx], signal_narrow[tidx], 'k')
plt.plot(t[tidxDs], signal_narrow[tidxDs], 'm.', ms=10)
plt.plot(t[tidxRs], signal_narrow[tidxRs], 'g.', ms=10)
plt.xlim(tlim)
plt.title('bandpass - rise/decay midpoints - trial 1')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t[tidx], signal_low[tidx], 'k')
plt.plot(t[tidxPs], signal_low[tidxPs], 'b.', ms=10)
plt.plot(t[tidxTs], signal_low[tidxTs], 'r.', ms=10)
plt.plot(t[tidxDs], signal_low[tidxDs], 'm.', ms=10)
plt.plot(t[tidxRs], signal_low[tidxRs], 'g.', ms=10)
plt.xlim(tlim)
plt.title('lowpass - peak/trough and rise/decay midpoints - subject 6')
plt.show()

from bycycle.features import compute_features
df = compute_features(signal_low, Fs, f_theta)
features_mat=df.head() # matrix with all relevant values
print(features_mat)
#np.save('lp_feature_matrix_trial2',features_mat)
print(features_mat['sample_last_trough'])
print(features_mat['sample_next_trough'])
print(features_mat['period'])

#visualizing burst detection settings
from bycycle.burst import plot_burst_detect_params
burst_kwargs = {'amplitude_fraction_threshold': 0,
                'amplitude_consistency_threshold': .2,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .7,
                'N_cycles_min': 3}

df = compute_features(signal_low, Fs, f_theta, burst_detection_kwargs=burst_kwargs)
#plot_burst_detect_params(signal_low, Fs, df, burst_kwargs, tlims=None, figsize=(6,6))

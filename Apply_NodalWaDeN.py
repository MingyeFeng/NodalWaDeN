
#%% Import resources
import os
import numpy as np
from matplotlib import pyplot as plt
import time as timing
import obspy
from utilities2 import waveform_fft, mkdir, downsample_series, bandpass, calsnr
import sys

from torch_tools import * # place this file in the directory
import torch
from torch.utils.data import DataLoader
from obspy.io.sac.sactrace import SACTrace

### Tools control ###
show_fig = 1 # if 1: save the statistics of SNR; if 2: save the comparison figures for each trace
conv_sac = 1 # if 1: convert to SAC

if(len(sys.argv) == 1):
    print("Apply_NodalWaDen.py: denoise nodal teleseismic records -- Mingye, 2023")
    print("USAGE: " ) 
    print("       python Apply_NodalWaDen.py [station name] ")
    print("<eg.>  python Apply_NodalWaDen.py A01 ")
    exit()
#%% Read target waveform for denoising
network='AC'
station=sys.argv[1]
#station='A01'

model_dir="Branch_Encoder_Decoder_LSTM_Model_v2.pth"
os.system("mkdir -p Denoise")
os.system("mkdir -p NoDenoise")
os.system("mkdir -p figs")
data_dir='data/'+station
data_out='Denoise/'+station
mkdir(data_out)
data_ndo='NoDenoise/'+station
mkdir(data_ndo)
figure_dir ='figs/'+network+'_'+station
mkdir(figure_dir)
cut_l = 20 # cut the waveform. in second
cut_r = 80
#bp_left=0.05
#bp_right=5
f_downsample=20
snr_toffset= 3 # signal window begins from 3s before the P wave
snr_timelength= 17 # signal/noise window is 17s long
zero_time = 20

#%%
roundn = 0
f_rep = open(figure_dir+'/ReportSNR_'+network+'_'+station+'.dat','w')
for file_name in os.listdir(data_dir):

    if (len(file_name.split("."))==2 and file_name.split(".")[1]=='EHZ'):
        data_name = file_name.split(".")[0]
        tr = obspy.read(data_dir+'/'+data_name+'.EH?')
        tr.merge(fill_value=0)
        if(tr.count() != 3):
            print(f"{data_name} has no 3-comp traces: {tr.count()}")
            continue
        if(tr[0].stats.npts != tr[1].stats.npts or tr[0].stats.npts != tr[2].stats.npts or tr[1].stats.npts != tr[2].stats.npts):
            print(f"{data_name} three component npts not equal: {tr[0].stats.npts}, {tr[1].stats.npts}, {tr[2].stats.npts}")
            continue
        if(tr[0].stats.delta == 1/f_downsample):
            delta = tr[0].stats.delta
        else:
        #tr.decimate(4)
            delta_raw = tr[0].stats.delta
            time_raw = np.arange(0, len(tr[0].data)-1) * delta_raw
            for i in range(0,3):
                tr[i].data = np.diff(tr[i].data)/delta_raw  # convert from displacement to velocity (optional for your dataset)
                time, tr[i].data, delta = downsample_series(time_raw, tr[i].data, f_downsample)
                tr[i].stats.delta = delta
                #tr[i].data = bandpass(tr[i].data-np.mean(tr[i].data),delta,bp_left,bp_right) # optional
        npts = 2000
    
        zero_index=round(zero_time/delta) # Data is from 20s before P
        signal_b_index = zero_index - round(snr_toffset/delta)
        singal_length = round(snr_timelength/delta)
        t_b_index = round(zero_index-cut_l/delta)
        t_e_index = round(zero_index+cut_r/delta)
        signal_b_index = round(cut_l/delta) - round(snr_toffset/delta)
        singal_length = round(snr_timelength/delta)
        waveform = np.zeros((npts,3))
        snr_raw = np.zeros(3)
        for i in range(3):
            waveform[:, i] = np.array(tr[i].data[t_b_index:t_e_index])
            #waveform[:, i] = bandpass(waveform[:, i]-np.mean(waveform[:, i]),delta,bp_left,bp_right) # optional, feel free to try
            snr_raw[i] = calsnr(waveform[:, i],signal_b_index,singal_length)
    
        data_mean = np.mean(waveform,axis=0)
        data_std = np.std(waveform,axis=0)
        waveform_normalized = (waveform - data_mean) / (data_std + 1e-12)
        waveform_normalized = np.reshape(waveform_normalized[:,np.newaxis,:],(-1, 2000, 3))
    
        ##%% Load data
        waveform_data = WaveformDataset(waveform_normalized,waveform_normalized)
        model = torch.load(model_dir,map_location ='cpu')
        batch_size = 256
        test_iter = DataLoader(waveform_data, batch_size=batch_size, shuffle=False)
    
        ##%% Denoise and store results
        all_output1 = np.zeros(waveform_normalized.shape)
        all_output2 = np.zeros(waveform_normalized.shape)
        model.eval()
        for i, (X,_) in enumerate(test_iter):
            #print('+' * 12 + f'batch{i}' + '+' * 12)
            output1, output2 = model(X)
    
        # output1 is the earthquake signal
        output1 = output1.detach().numpy()
        output1 = np.moveaxis(output1, 1, -1)
        all_output1[(i*batch_size):((i+1)*batch_size), :, :] = output1
        # output2 is the noise signal
        output2 = output2.detach().numpy()
        output2 = np.moveaxis(output2, 1, -1)
        all_output2[(i*batch_size):((i+1)*batch_size), :, :] = output2
    
        waveform_recovered = all_output1*data_std+data_mean
        waveform_recovered = np.reshape(waveform_recovered, (-1,3))
        noise_recovered = all_output2*data_std+data_mean
        noise_recovered = np.reshape(noise_recovered, (-1,3))
        waveform_residual = waveform - waveform_recovered - noise_recovered
    
        tr_raw = tr.copy()
        tr_eq = tr.copy()
        tr_noise = tr.copy()
        tr_residual = tr.copy()
        snr_rec = np.zeros(3)
        for i in range(3):
            tr_raw[i].data = waveform[:,i]
            tr_eq[i].data = waveform_recovered[:,i]
            tr_noise[i].data = noise_recovered[:,i]
            tr_residual[i].data = waveform_residual[:,i]
            snr_rec[i] = calsnr(waveform_recovered[:,i],signal_b_index,singal_length)
        ## Convert to SAC files
        if (conv_sac == 1):
            for j in range(3):
                sac_filename = data_name+'.'+tr_eq[j].stats.channel+'.dn'
                sac = SACTrace.from_obspy_trace(tr_eq[j])
                sac.b = -cut_l
                sac.write(data_out+'/'+sac_filename)
                sac_filename_raw = data_name+'.'+tr_raw[j].stats.channel
                sac_raw = SACTrace.from_obspy_trace(tr_raw[j])
                sac_raw.b = -cut_l
                sac_raw.write(data_ndo+'/'+sac_filename_raw)

        ##%% Show results
        if (show_fig == 2):
            npts_qb = 100/delta # how long would you show in total. This case shows 100 s data
            t = np.arange(0,npts_qb)*delta-20 # 20 means that you want to set the P arrival time at 20 s
            b_index = round(0/delta)
            e_index = round(b_index + npts_qb )
            plt.close("all")
            fig, ax = plt.subplots(6, 3, sharex=False, sharey=True, num=1, figsize=(12, 5)) #16, 8
            #fig.suptitle("(a) Waveforms",fontsize=14,fontweight='bold',)
            #fig.subplots_adjust(top=0.8)
            for i in range(3):
                scaling_factor = np.max(abs(waveform[:,i]))
                ax[i, 0].plot(t, waveform[b_index:e_index,i]/scaling_factor, '-k', label='Raw signal', linewidth=1.5)
                ax[i, 0].plot(t, waveform_recovered[b_index:e_index,i]/scaling_factor, '-r', label='Denoised signal', linewidth=1)
                ax[i, 1].plot(t, waveform_recovered[b_index:e_index,i]/scaling_factor, '-r', label='Denoised signal', linewidth=1)
                ax[i, 2].plot(t, noise_recovered[b_index:e_index,i]/scaling_factor, '-b', label='Predicted noise', linewidth=1)
            ax[0, 0].set_title("(a) Original signal", fontsize=14)
            ax[0, 1].set_title("(b) Predicted earthquake", fontsize=14)
            ax[0, 2].set_title("(c) Predicted noise", fontsize=14)

            titles = ['E', 'N', 'Z']
            for i in range(waveform.shape[1]):
                ax[i, 0].set_ylabel(titles[i])
            for i in range(6):
                for j in range(3):
                    #ax[i, j].axis('off')
                    ax[i, j].xaxis.set_visible(False)
                    ax[i, j].yaxis.set_ticks([])
                    # remove box
                    ax[i, j].spines['right'].set_visible(False)
                    ax[i, j].spines['left'].set_visible(False)
                    ax[i, j].spines['top'].set_visible(False)
                    ax[i, j].spines['bottom'].set_visible(False)
                    ax[i, j].patch.set_alpha(0)
                    #ax[i, j].set_xlim(-50, 150)
                    ax[i, j].set_xlim(-20, 80)
            for j in range(3):
                ax[2, j].spines['bottom'].set_visible(True)
                ax[2, j].xaxis.set_visible(True)
                #ax[2, j].set_xlim(-150, 250)
                ax[2, j].set_xlabel('Time (s)')

            #fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, num=1, figsize=(12, 5)) #16, 8
            #fig.suptitle("(b) Spectrum",fontsize=14,fontweight='bold',)
            #fig.subplots_adjust(top=0.2)
            dt = t[1] - t[0]
            for i in range(3):
                ax[5,i] = plt.subplot2grid((8,3),(5,i),rowspan=3)
                _, spect_raw  = waveform_fft(waveform[b_index:e_index,i]/scaling_factor, dt)
                _, spect_noise = waveform_fft(noise_recovered[b_index:e_index,i]/scaling_factor, dt)
                freq, spect_signal = waveform_fft(waveform_recovered[b_index:e_index,i]/scaling_factor, dt)
                #ax[i].loglog(freq, spect_noisy_signal, '-k', label='X_input', linewidth=1.5)
                ax[5,i].loglog(freq, spect_raw, '-k', label='raw', linewidth=0.5, alpha=1)
                ax[5,i].loglog(freq, spect_signal, '-r', label='signal', linewidth=0.5, alpha=1)
                ax[5,i].loglog(freq, spect_noise, '-',color='b', label='noise', linewidth=0.5, alpha=0.8)
                if i == 0:
                    ax[5,i].set_ylabel('velocity spectra', fontsize=12)
                    ax[5,i].legend(fontsize=8, loc=3)
                ax[5,i].grid(alpha=0.5)
                ax[5,i].set_xlabel('Frequency (Hz)', fontsize=12)
            titles = ['(e) E', '(f) N', '(g) Z']
            for i in range(3):
                ax[5,i].set_title(titles[i], fontsize=14)

            plt.subplots_adjust(bottom=0, right=0.8, top=1)

            #plt.show()
            plt.figure(1)
            plt.savefig(figure_dir + f'/{data_name}.pdf',
                        bbox_inches='tight')
        f_rep.write("%s   %-6.1f %-6.1f %-6.1f %-6.1f %-6.1f %-6.1f %-6.1f %-6.1f \n" % 
            (data_name,snr_raw[0],snr_raw[1],snr_raw[2],np.mean(snr_raw),snr_rec[0],snr_rec[1],snr_rec[2],np.mean(snr_rec)))
        roundn += 1
f_rep.close()

# %%
if (show_fig == 1):
    f_res=open(figure_dir+'/ReportSNR_'+network+'_'+station+'.dat','r')
    data_results=f_res.readlines()
    n = len(data_results)
    snr_before = np.zeros(n)
    snr_before_min = np.zeros(n)
    snr_after = np.zeros(n)
    snr_after_min = np.zeros(n)
    for i in range (0,n):
        data_results[i] = data_results[i].strip('\n')
        snr_before[i] = data_results[i].split()[4]
        snr_after[i] = data_results[i].split()[8]
        snr_before_min[i] = min([data_results[i].split()[1],data_results[i].split()[2],data_results[i].split()[3]])
        snr_after_min[i] = min([data_results[i].split()[5],data_results[i].split()[6],data_results[i].split()[7]])
        if (snr_before[i]>10):
            snr_before[i] = 10
        if (snr_after[i]>10):
            snr_after[i] = 10
        if (snr_before_min[i]>10):
            snr_before_min[i] = 10
        if (snr_after_min[i]>10):
            snr_after_min[i] = 10

        if (snr_before[i]<1):
            snr_before[i] = 1
        if (snr_after[i]<1):
            snr_after[i] = 1
        if (snr_before_min[i]<1):
            snr_before_min[i] = 1
        if (snr_after_min[i]<1):
            snr_after_min[i] = 1
    plt.close("all")
    fig, ax = plt.subplots(1, 2, sharex=False, sharey=True, num=1, figsize=(10, 4), dpi=200)
    ax[0].hist(snr_before,bins=9,edgecolor="k",histtype="step",label='Noisy',density=True,range=(1,10))
    ax[0].hist(snr_after,bins=9,edgecolor="r",histtype="step",label='Denoised',density=True,range=(1,10))
    ax[0].set_xlim(1,10)
    #ax[0].set_ylim(0,1.0)
    ax[0].set_title("Average SNR")
    ax[0].set_xlabel("SNR")
    ax[0].set_ylabel("Density")
    ax[0].legend()
    ax[1].hist(snr_before_min,bins=9,edgecolor="k",histtype="step",label='Noisy',density=True,range=(1,10))
    ax[1].hist(snr_after_min,bins=9,edgecolor="r",histtype="step",label='Denoised',density=True,range=(1,10))
    ax[1].set_xlim(1,10)
    #ax[1].set_ylim(0,1.0)
    ax[1].set_title("Minimum SNR")
    ax[1].set_xlabel("SNR")
    ax[1].set_ylabel("Density")
    ax[1].legend()

    plt.figure(1)
    plt.savefig(figure_dir + f'/Stat_{network}_{station}.pdf',
                bbox_inches='tight')

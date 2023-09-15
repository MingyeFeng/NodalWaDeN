# NodalWaDeN
A convolutional neural network (CNN) model trained for denoising teleseismic short-period records.
NodalWaDeN is developed based on a parental network [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet) developed by [Yin et al. (2023, GJI)](https://doi.org/10.1093/gji/ggac290).

**Usage:**

1. Install [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet) following the instruction there

   - You may want to change the contents in the file "environment.yml" if you encounter errors when installing the env outside macOS platform
   
2. Download [NodalWaDeN](https://github.com/MingyeFeng/NodalWaDeN) and move all files to the main folder of [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet)
3. Activate WaveDecompNet and run NodalWaDeN
   
   ```bash
   conda activate WaveDecompNet
   python Apply_NodalWaDeN.py A01 
   ```
4. You may need to install modules "numpy", "matplotlib", "obspy", "torch", and "h5py" under the WaveDecompNet env if required
   ```bash
   conda install numpy
   conda install matplotlib
   conda install obspy
   conda install torch
   conda install h5py
   ```
   or
   ```bash
   pip install numpy
   pip install matplotlib
   pip install obspy
   pip install torch
   pip install h5py
   ```
**Note:**

- "Apply_NodalWaDen.py" will generate figures about comparison before and after denoising in the folder "./figs", and output SAC files in "./Denoise" and "./NoDenoise" folders.

- NodalWaDeN has been tested on both the macOS and Ubuntu platforms

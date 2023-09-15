# NodalWaDeN
A convolutional neural network (CNN) model trained for denoising teleseismic records.
NodalWaDeN is developed based on a parental network [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet) developed by [Yin et al. (2023, GJI)](https://doi.org/10.1093/gji/ggac290).

**Usage:**

1. Install [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet) following the instruction there

   - You may want to change the contents in the file "environment.yml" if you encounter errors when installing the env outside macOS platform
   
2. Download [NodalWaDeN](https://github.com/MingyeFeng/NodalWaDeN) and move all files to the main folder of [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet)
3. Activate WaveDecompNet and run NodalWaDeN
   
   ```bash
   conda activate WaveDecompNet
   python Apply_NodalWaDen.py A01 
   ```
4. You may need to install modules "numpy", "matplotlib", "obspy", and "torch" under WaveDecompNet env if required
   ```bash
   conda install numpy
   conda install matplotlib
   conda install obspy
   conda install torch
   ```
   or
   ```bash
   pip install numpy
   pip install matplotlib
   pip install obspy
   pip install torch
   ```
> "Apply_NodalWaDen.py" will generate figures about comparison before and after denoising in the folder "./figs/denoise" for each teleseismic trace, and output SAC files in "./Denoise" and "./NoDenoise" folders.

> NodalWaDeN has been tested on both macOS and Ubuntu platforms

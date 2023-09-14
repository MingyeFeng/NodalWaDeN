# NodalWaDeN
A convolutional neural network (CNN) model trained for denoising teleseismic records.
NodalWaDeN is developed based on a parental network [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet) developed by Dr. Jiuxun Yin.

Usage:

1. Install [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet) following the instruction there
2. Download [NodalWaDeN](https://github.com/MingyeFeng/NodalWaDeN) and move all files to the main folder of [WaveDecompNet](https://github.com/yinjiuxun/WaveDecompNet)
   
   You may want to change the contents in the file "environment.yml" if you encounter errors when installing the env
3. Activate WaveDecompNet and run NodalWaDeN
```bash
conda activate WaveDecompNet
python Apply_NodalWaDen.py A01 
```
Apply_NodalWaDen.py will generate figures about comparison before and after denoising in the folder ./figs/denoise for each teleseismic trace, and output SAC files in ./Denoise and ./NoDenoise folders.

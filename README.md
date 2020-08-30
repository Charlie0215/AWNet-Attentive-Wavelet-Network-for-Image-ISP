# PyTorch implementation of AWNet 

This is the official PyTorch implement of AWNet. Our Team, MacAI, recieved a Runner-Up award in the AIM 2020 Learned Image ISP challenge (ECCVWW 2020). The proposed solution can achieve excellent MOS while remaining competetive in numerical result. See more details from our [paper](https://arxiv.org/abs/2008.09228).

## Abstract
As the revolutionary improvement being made on the performance of smartphones over the last decade, mobile photography becomes one of the most common practices among the majority of smartphone users. However, due to the limited size of camera sensors on phone, the photographed image is still visually distinct to the one taken by the digital single-lens reflex (DSLR) camera. To narrow this performance gap, one is to redesign the camera image signal processor (ISP) to improve the image quality. Owing to the rapid rise of deep learning, recent works resort to the deep convolutional neural network (CNN) to develop a sophisticated data-driven ISP that directly maps the phone-captured image to the DSLR-captured one. In this paper, we introduce a novel network that utilizes the attention mechanism and wavelet transform, dubbed AWNet, to tackle this learnable image ISP problem. By adding the wavelet transform, our proposed method enables us to restore favorable image details from RAW information and achieve a larger receptive field while remaining high efficiency in terms of computational cost. The global context block is adopted in our method to learn the non-local color mapping for the generation of appealing RGB images. More importantly, this block alleviates the influence of image misalignment occurred on the provided dataset. Experimental results indicate the advances of our design in both qualitative and quantitative measurements.

## Presentation Video

## Pretrained Models
Download [demosaiced_model](https://drive.google.com/file/d/1RVUHN4GBs-hHpDvdFc4qEy9oGWkHcy3W/view?usp=sharing) and [Raw_model](https://drive.google.com/file/d/1ejXaneEKczHmIRDbesq5V1ExAzTK1Et-/view?usp=sharing) and place into the folder ```./best_weight``` 

## Training
1. Generate pseudo-demosicing images for 3-channel-input model.
``` 
    cd demosaic
    python demosaic.py -data <directory contains your raw images> -save <directory to save your pseudo-demosaicing images>
```
Then, move the resulting folder of demosaicing images under your root dataset directory. Please make sure your dataset structure is the same as what we show in the <em>Training/Validation Dataset Strcuture</em> section.
2. Change configuration in ```config.py``` accordingly and run
```python train_3channel.py``` or ```python train_4channel.py```

## Testing
1. Generate pseudo-demosicing images for 3-channel-input model.
``` 
    cd demosaic
    python demosaic.py -data <directory contains your raw images> -save <directory to save your pseudo-demosaicing images>
```
Then, move the resulting folder of demosaicing images under your root dataset directory.
Please make sure your dataset structure is the same as what we show in the <em>Training/Validation Dataset Strcuture</em> section.
2. To reproduce our final results from testing board, run ```python validation_final.py```
3. To reporduce our full resolution result, run ```python validation_final_fullres.py```.

## Qualitative Results
Full resolution:  
<img alt="" src="/images/qualitative.png" style="display: inline-block;" />

Compare with other state-of-the-arts:  
<div style="text-align: center">
<img alt="" src="/images/qualitative2.png" style="display: inline-block;" />
</div>

## Acknowledgement
We thank the authors of [MWCNN](https://github.com/lpj0/MWCNN.git), [GCNet](https://github.com/xvjiarui/GCNet.git), and [Pytorch_SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim). Part of our code is built upon their modules.



# AWNet

This is the official PyTorch implement of AWNet. Our Team, MacAI, recieved a Runner-Up award in the AIM 2020 Learned Image ISP challenge (ECCVW 2020). The proposed solution can achieve excellent MOS while remaining competetive in numerical result. See more details from our [paper](https://arxiv.org/abs/2008.09228).

## Abstract
As the revolutionary improvement being made on the performance of smartphones over the last decade, mobile photography becomes one of the most common practices among the majority of smartphone users. However, due to the limited size of camera sensors on phone, the photographed image is still visually distinct to the one taken by the digital single-lens reflex (DSLR) camera. To narrow this performance gap, one is to redesign the camera image signal processor (ISP) to improve the image quality. Owing to the rapid rise of deep learning, recent works resort to the deep convolutional neural network (CNN) to develop a sophisticated data-driven ISP that directly maps the phone-captured image to the DSLR-captured one. In this paper, we introduce a novel network that utilizes the attention mechanism and wavelet transform, dubbed AWNet, to tackle this learnable image ISP problem. By adding the wavelet transform, our proposed method enables us to restore favorable image details from RAW information and achieve a larger receptive field while remaining high efficiency in terms of computational cost. The global context block is adopted in our method to learn the non-local color mapping for the generation of appealing RGB images. More importantly, this block alleviates the influence of image misalignment occurred on the provided dataset. Experimental results indicate the advances of our design in both qualitative and quantitative measurements.

## Presentation Video
[![Watch the video](https://img.youtube.com/vi/HlrzVFMUwCQ/0.jpg)](https://youtu.be/HlrzVFMUwCQ)

## Pretrained Models & The ZRR Dataset
1. Download [demosaiced_model](https://drive.google.com/file/d/1uhohG6cYkM_-W4dGLl8yGlo85UMF6KEK/view?usp=sharing) and [Raw_model](https://drive.google.com/file/d/1jBwEm7_zbU55qOlGAVuOAQ8BIwx2g7Fw/view?usp=sharing) and place into the folder ```./best_weight```.
2. Download the ZRR dataset from [here](https://competitions.codalab.org/competitions/24718)

## Environment
The project is built on **Python 3.9**. You can run the following script to setup the venv.
```
script/setup_venv.sh
```

If you want to reproduce our results from the AIM 2020 challenge, please follow the steps in ```old_version``` branch. Though the results are similar, the model definition in ```master``` is slightly different as described in our paper.

## Training
1. Generate pseudo-demosicing images for 3-channel-input model.
```
python script/demosaic.py -s /your/raw/image/path -d /your/saving/path
```
Then, move the resulting folder of demosaicing images under your root dataset directory. Please make sure your dataset structure is the same as what we show in the <em>Training/Validation Dataset Strcuture</em> section.
2. Change configuration in `config.py` accordingly and run

```python train_3channel.py```

or

```python train_4channel.py```

## Testing
1. Generate pseudo-demosicing images for 3-channel-input model.
```
python script/demosaic.py -s /your/raw/image/path -d /your/saving/path
```
Then, move the resulting folder of demosaicing images under your root dataset directory.
Please make sure your dataset structure is the same as what we show in the <em>Training/Validation Dataset Strcuture</em> section.
2. To reproduce our final results from testing board, run ```python validation_final.py```.
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

## Citation
If our work helps your research, please consider to cite our paper:
```
@article{dai2020awnet,
  title={AWNet: Attentive Wavelet Network for Image ISP},
  author={Dai, Linhui and Liu, Xiaohong and Li, Chengqi and Chen, Jun},
  journal={arXiv preprint arXiv:2008.09228},
  year={2020}
}
```

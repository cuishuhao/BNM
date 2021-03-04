# BNM for Unsupervised Open Domain Recognition

## Requirements
The code is implemented with Python(3.6) and Pytorch(1.0.0).

To install the required python packages, run

```
pip install -r requirements.txt
```

## Dataset and Model
Dataset and pretrained models could be calculated and found in [UODTN](https://github.com/junbaoZHUO/UODTN),
while we directly take them.

### Dataset
The source dataset could be found in[here](https://drive.google.com/file/d/1GdDZ1SvEqGin_zeCAGaJn0821vC_PJmc/view?usp=sharing)

For the target dataset, one can download from [here](http://cvml.ist.ac.at/AwA2/). The link is [here](http://cvml.ist.ac.at/AwA2/AwA2-data.zip)

The image list of the dataset should be edit to the path of images in 'data/new_AwA2.txt' and 'data/WEB_3D3_2.txt'

### Model
The models could be calculated in UODTN, while we take the related models as follows:

-["base_net_pretrained_on_I2AwA2_source_only"](https://drive.google.com/file/d/1FiHB8HV8U2Isfx0A6ipWEIaE4q-sekoO/view?usp=sharing) is a trained feature extractor for I2AwA.  

-["awa_50_cls_basic"](https://drive.google.com/file/d/1DLeCpM7-k1xBianFEmc3L6c9526WEha4/view?usp=sharing) contains 50 initial classifiers for AwA2.  

These models should be put in the folder in "model/"

## run
python train_loader.py --gpu_id 0

more options could be found in help


The codes are borrowed from [UODTN](https://github.com/junbaoZHUO/UODTN)

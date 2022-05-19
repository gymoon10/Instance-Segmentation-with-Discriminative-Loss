# Instance-Segmentation-with-Discriminative-Loss

This code is pytorch implementation of 'Semantic Instance Segmentation with a Discriminative Loss Function' (https://arxiv.org/abs/1708.02551)

I slightly updated and revised the code of https://github.com/Wizaron/instance-segmentation-pytorch. The updated content is as follows.

  - revised the Dataset which does not use lmdb

  - added other networks (UNet, UNet with CBAM, DeepLabV3)
     
     - unfortunately semantic & instance segmentation result of DeepLabV3 are very bad

<br/>

## Notification

You should execute `python -m visdom.server` before training

`train3.py` -> `pred_list2.py` -> `evaluation.py`

<br/>


## Results

### CVPPP

#### Scores on validation subset (28 images)

| Model | Loss | Mean SBD | Mean FG Dice | Dic | 
| :----------- | :------------: | ------------: | ------------: | ------------: | 
|ReSeg | 0.1425 | 0.8599 | 0.9669 | 0.7143 | 
|SegNet | 0.1295 | 0.8645 | 0.9695 | 0.6071 | 
|UNet | 0.1137 | 0.8656 | 0.9703 | 0.5713 | 
|UNet-CBAM | 0.8813 | 0.8599 | 0.9708 | 0.7143 | 

<br/>

### Sample Predictions (UNet-CBAM)

input / pred / GT

![image](https://user-images.githubusercontent.com/44194558/169200662-8feca540-3d61-49ff-94b4-818d818fc87f.png)

![image](https://user-images.githubusercontent.com/44194558/169200700-f5678b8c-af7c-478e-ae7a-1449e53f0182.png)

![image](https://user-images.githubusercontent.com/44194558/169200753-7c6901fb-7a3e-44da-8f45-b20a5681e9f4.png)








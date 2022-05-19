# Instance-Segmentation-with-Discriminative-Loss

This code is pytorch implementation of 'Semantic Instance Segmentation with a Discriminative Loss Function' (https://arxiv.org/abs/1708.02551)

I slightly updated and revised the code of https://github.com/Wizaron/instance-segmentation-pytorch which does not match with python3. The updated content is as follows.

  - revised the dataset which does not use lmdb

  - added other networks (UNet, UNet with CBAM, DeepLabV3)
     
     - unfortunately semantic & instance segmentation result of DeepLabV3 are very bad

  - ReSeg with CoordConv(with r) is possible


**My paper 'Leaf Instance Segmentation with Attention Based U-Net & Discriminative Loss' which utilizes this code was submitted for participation at 'Summer Annual conference of IEIE, 2022 (https://conf.theieie.org/2022s/)'**

<br/>

## Notification

- You should execute `python -m visdom.server` before training

- `train3.py` -> `pred_list2.py` -> `evaluation.py`

- I couldn't solve the error of training Stacked Recurrent Hourglass. The training does not proceed from 2 epochs due to errors in the back-propagation process.

<br/>

## UNet-CBAM

### CBAM (Convolutional Block Attention Module)

![image](https://user-images.githubusercontent.com/44194558/169212182-d4867663-f011-4722-8e60-1f0cd9c04ef0.png)

### Architecture

![image](https://user-images.githubusercontent.com/44194558/169212305-980b2299-53c3-48a8-a017-77fbb6a16a6c.png)

 - Instance Counter : predicts normalized # of leaf instances
 - Semantic Head : predicts semantic mask (f.g / b.g)
 - Instance Head : predicts 32 dims embedding space which has to be clustered by K-Means

<br/>

## Results

### CVPPP

#### Scores on validation subset (28 images)

| Model | Loss | Mean SBD | Mean FG Dice | Dic | 
| :----------- | :------------: | ------------: | ------------: | ------------: | 
|ReSeg | 0.1425 | 0.8599 | 0.9669 | 0.7143 | 
|SegNet | 0.1295 | 0.8645 | 0.9695 | 0.6071 | 
|UNet | 0.1137 | 0.8656 | 0.9703 | 0.5713 | 
|UNet-CBAM | 0.1031 | 0.8813 | 0.9708 | 0.7143 | 

<br/>

### Sample Predictions (UNet-CBAM)

input / pred / GT

![image](https://user-images.githubusercontent.com/44194558/169200662-8feca540-3d61-49ff-94b4-818d818fc87f.png)

![image](https://user-images.githubusercontent.com/44194558/169200700-f5678b8c-af7c-478e-ae7a-1449e53f0182.png)

![image](https://user-images.githubusercontent.com/44194558/169200753-7c6901fb-7a3e-44da-8f45-b20a5681e9f4.png)

<br/>

# References

ReNet (used in ReSeg) : https://arxiv.org/abs/1505.00393

ReSeg : https://arxiv.org/abs/1511.07053

SegNet : https://arxiv.org/pdf/1511.00561.pdf

CBAM (Convolutional Block Attention Module) : https://arxiv.org/abs/1807.06521








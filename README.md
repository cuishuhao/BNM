# BNM
code release for ["Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations"](https://arxiv.org/abs/2003.12237) ( CVPR2020 oral)

Old version can be found in [BNM v1](https://github.com/cuishuhao/BNM/tree/BNMv1)

## One-sentence description
We prove in the paper that Batch Nuclear-norm Maximization (BNM) could ensure the prediction discriminability and diversity, which is an effective method under label insufficient situations.

We further optimize BNM by Batch Nuclear-norm Minimization (BNMin) and Fast BNM.

## Application for BNM

### One line code for BNM v1 under Pytorch and Tensorflow

Assume `X` is the prediction matrix. We could calculate BNM loss in both Pytorch and Tensorflow, as follows:
 
-Pytorch

1. Direct calculation (Since there remains direct approach for nuclear-norm)
```
L_BNM = -torch.norm(X,'nuc')
```
2. Calculation by SVD
```
L_BNM = -torch.sum(torch.svd(X, compute_uv=False)[1])
```
-Tensorflow
```
L_BNM = -tf.reduce_sum(tf.svd(X, compute_uv=False))
```

### code for Fast BNM (FBNM) under Pytorch
Assume `X` is the prediction matrix. Then FBNM can be calculated as:
```
list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(X,2),dim=0)), descending=True)
nums = min(X.shape[0],X.shape[1])
L_FBNM = - torch.sum(list_svd[:nums])
```

### Sum of Changes from BNM v1
1. Fast BNM.(By approximation)
2. BNMin.(On the other hand on source domain)
3. Multiple BNM.(Multiple Batch Optimization)
4. Balance domainnet.(New dataset)
5. Semi-supervised DA.(New task)

### Tasks
We apply BNM to domain adaptation (DA) in [DA](DA), unsupervised open domain recognition (UODR) in [UODR](UODR) and semi-supervised learning (SSL) in [SSL](SSL).

Training instructions for DA, UODR and SSL are in the `README.md` in [DA](DA), [UODR](UODR) and [SSL](SSL) respectively.

## Citation
If you use this code for your research, please consider citing:
```
@InProceedings{Cui_2020_CVPR,
author = {Cui, Shuhao and Wang, Shuhui and Zhuo, Junbao and Li, Liang and Huang, Qingming and Tian, Qi},
title = {Towards Discriminability and Diversity: Batch Nuclear-Norm Maximization Under Label Insufficient Situations},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
Supplementary could be found in [Google driver](https://drive.google.com/file/d/15WOL2wFCSYVbPQfZ0OOSwtBXlcvgw8kA/view?usp=sharing)
 and [baidu cloud](https://pan.baidu.com/s/1eZAguvOXUOa0k_sietA8Zg) (z7yt).
 
[量子位](https://zhuanlan.zhihu.com/p/124860496)

## Contact
If you have any problem about our code, feel free to contact
- hassassin1621@gmail.com

or describe your problem in Issues.

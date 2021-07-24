# BNM v1 v2
code release for BNM v1 ["Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations"](https://arxiv.org/abs/2003.12237) ( CVPR2020 oral)
and BNM v2 [Fast Batch Nuclear-norm Maximization and Minimization for Robust Domain Adaptation](https://arxiv.org/abs/2107.06154)(TPAMI under review)

Clean code of BNM v1 can be found in [old version](https://github.com/cuishuhao/BNM/tree/BNMv1)

## One-sentence description
BNM v1: we prove in the paper that Batch Nuclear-norm Maximization (BNM) can ensure the prediction discriminability and diversity, which is an effective method under label insufficient situations.

BNM v2: we further devise Batch Nuclear-norm Minimization (BNMin) and Fast BNM (FBNM) for multiple domain adaptation scenarios.

## Quick start with BNM

### One line code for BNM v1 under Pytorch and Tensorflow

Assume `X` is the prediction matrix. We can calculate BNM loss in both Pytorch and Tensorflow, as follows:
 
-Pytorch

1. Direct calculation (Since there remains direct approach for nuclear-norm in Pytorch)
```
L_BNM = -torch.norm(X,'nuc')
```
2. Calculation with SVD (For S, V and D, only S is useful for calculation of BNM)
```
L_BNM = -torch.sum(torch.svd(X, compute_uv=False)[1])
```
-Tensorflow
```
L_BNM = -tf.reduce_sum(tf.svd(X, compute_uv=False))
```

### code for FBNM under Pytorch
Assume `X` is the prediction matrix. Then FBNM can be calculated as:
```
list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(X,2),dim=0)), descending=True)
nums = min(X.shape[0],X.shape[1])
L_FBNM = - torch.sum(list_svd[:nums])
```

### Sum of Changes from BNM v1 to BNM v2
1. - [x] FBNM.(By approximation)
2. - [ ] BNMin.(On the other hand on source domain)
3. - [ ] Multiple BNM.(Multiple Batch Optimization)
4. - [x] Balance domainnet.(New dataset)
5. - [ ] Semi-supervised DA.(New task)

### Applications
We apply BNM to unsupervised domain adaptation (UDA) in [UDA](UDA), unsupervised open domain recognition (UODR) in [UODR](UODR) and semi-supervised learning (SSL) in [SSL](SSL).

Training instructions for UDA, UODR and SSL are in the `README.md` in [UDA](UDA), [UODR](UODR) and [SSL](SSL) respectively.

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
@article{cui2021fast,
  title={Fast Batch Nuclear-norm Maximization and Minimization for Robust Domain Adaptation},
  author={Cui, Shuhao and Wang, Shuhui and Zhuo, Junbao and Li, Liang and Huang, Qingming and Tian, Qi},
  journal={arXiv preprint arXiv:2107.06154},
  year={2021}
}
```
Supplementary of BNM can be found in [Google driver](https://drive.google.com/file/d/15WOL2wFCSYVbPQfZ0OOSwtBXlcvgw8kA/view?usp=sharing)
 and [baidu cloud](https://pan.baidu.com/s/1eZAguvOXUOa0k_sietA8Zg) (z7yt).
 
[量子位](https://zhuanlan.zhihu.com/p/124860496)

## Contact
If you have any problem about our code, feel free to contact
- hassassin1621@gmail.com

or describe your problem in Issues.

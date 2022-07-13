# PCLossNet
The codes for Learning to Train a Point Cloud Reconstruction Network without Matching

## Environment
* TensorFlow 1.13.1
* Cuda 10.0
* Python 3.6.9
* lmdb 0.98  
* tensorpack 0.10.1
* numpy 1.14.5

## Dataset
The adopted ShapeNet Part dataset is adopted following [FoldingNet](http://www.merl.com/research/license#FoldingNet), while the ModelNet10 and ModelNet40 datasets follow [PointNet](https://github.com/charlesq34/pointnet.git). Other datasets can also be used. Just revise the path in (`getdata.py`).

## Usage

1. Prepartion

```
cd ./tf_ops
bash compile.sh
```

2. Train

For the reconstruction task,
```
Python3 vv_ae.py
```

Note that the paths of data should be edited according to your setting.

3. Test
For the evaluation of reconstruction errors,
```
Python3 vvae_eva.py
```

The trained weight files should be put in (`./ae_files`) to evaluate the sampling performances.


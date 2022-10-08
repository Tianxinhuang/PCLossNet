# PCLossNet
The codes for Learning to Train a Point Cloud Reconstruction Network without Matching[ECCV'22]

## Environment
* TensorFlow 1.13.1
* Cuda 10.0
* Python 3.6.9
* numpy 1.14.5

## Dataset
The adopted ShapeNet Part dataset is adopted following [FoldingNet](http://www.merl.com/research/license#FoldingNet), while the ModelNet10 and ModelNet40 datasets follow [PointNet](https://github.com/charlesq34/pointnet.git). Other datasets can also be used. Just revise the path by the (`--filepath`) parameter when training or evaluating the networks.
The files in (`--filepath`) should be organized as

        <filepath>
        ├── <trainfile1>.h5 
        ├── <trainfile2>.h5
        ├── ...
        ├── train_files.txt
        └── test_files.txt

where the contents in (`train_files.txt`) or (`test_files.txt`) should include the directory of training or testing h5 files, such as:

        train_files.txt
        ├── <trainfile1>.h5
        ├── <trainfile2>.h5
        ├── ...

## Usage

1. Preparation

```
cd ./tf_ops
bash compile.sh
```

2. Train

For the reconstruction task,
```
Python3 vv_ae.py
```

Note that the paths of data should be edited through the (`--filepath`) parameter according to your setting.

3. Test

For the evaluation of reconstruction errors,
```
Python3 vvae_eva.py
```

The trained weight files should be provided by the (`--savepath`) parameter to evaluate the performances.

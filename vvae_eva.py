import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance 
import copy
import random
import sys
import open3d as o3d
from dgcnn import dgcnn_kernel,dgcls_kernel
from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping

from lossnet import sampling,mlp_architecture_ala_iclr_18,local_kernel
from provider import shuffle_points,jitter_point_cloud 

DATA_DIR=getdata.getspdir()
filelist=os.listdir(DATA_DIR)

BATCH_SIZE=16
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def getnormal(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=3))
    normals=np.array(pcd.normals)
    #normals=normals/np.sqrt(np.sum(np.square(normals),axis=-1,keepdims=True))
    result=np.concatenate([np.array(pcd.points),normals],axis=-1)
    return result
def emd_func(pred,gt):
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
    dist = tf.reduce_mean(dist,axis=-1)
    
    cens=tf.reduce_mean(pred,axis=1,keep_dims=True)
    radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=-1))
    #dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    #dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    #print(matched_out,dist)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist)
    return emd_loss
def emd(pred,gt):
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.reduce_sum((pred - matched_out) ** 2,axis=-1)
    dist = tf.reduce_mean(dist,axis=-1)

    cens=tf.reduce_mean(pred,axis=1,keep_dims=True)
    radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=-1))
    #dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    #dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    #print(matched_out,dist)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist)
    return emd_loss

def grouping(xyz,new_xyz, radius, nsample, points, knn=False, use_xyz=True):
    if knn:
        _,idx = tf_grouping.knn_point(nsample, xyz, new_xyz)
    else:
        _,id0 = tf_grouping.knn_point(1, xyz, new_xyz)
        valdist,idx = tf_grouping.knn_point(nsample, xyz, new_xyz)
        idx=tf.where(tf.greater(valdist,radius),tf.tile(id0,[1,1,nsample]),idx)
    grouped_xyz = tf_grouping.group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = tf_grouping.group_point(points, idx) # (batch_size, npoint, nsample, channel)
        #print(grouped_points)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
        grouped_points=grouped_xyz

    return grouped_xyz,grouped_points
def chamfer_max(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1=tf.reduce_mean(dist1)
    dist2=tf.reduce_mean(dist2)
    dist=tf.maximum(dist1, dist2)
    return dist,idx1

def chamfer_local(pcda,pcdb):
    ptnum=pcda.get_shape()[1].value
    knum=pcda.get_shape()[2].value
    pcd1=tf.reshape(pcda,[-1,knum,3])
    pcd2=tf.reshape(pcdb,[-1,knum,3])
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1=tf.reduce_mean(tf.reshape(dist1,[-1,ptnum,knum]),axis=-1)#+0.1*tf.reduce_max(tf.reshape(dist1,[-1,ptnum,knum]),axis=1)#batch*k
    dist2=tf.reduce_mean(tf.reshape(dist2,[-1,ptnum,knum]),axis=-1)#+0.1*tf.reduce_max(tf.reshape(dist2,[-1,ptnum,knum]),axis=1)#batch*k
    dist=tf.reduce_mean(tf.maximum(dist1,dist2))
    return dist,idx1
def multi_chamfer(n,inputpts,output,k=64,r=0.01,use_knn=True):
    in_cen=sampling(n,inputpts,use_type='f')[-1]
    out_cen=sampling(n,output,use_type='f')[-1]
    in_cen=tf.concat([in_cen,out_cen],axis=1)

    out_kneighbor,_=grouping(output,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=use_knn, use_xyz=True)
    in_kneighbor,_=grouping(inputpts,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=use_knn, use_xyz=True)
    local_in=chamfer_local(in_kneighbor,out_kneighbor)[0]

    local_loss=local_in
    return local_loss
def multi_chamfer_func(n,inputpts,output,klist=[16,32,64]):
    #result=[chamfer(inputpts,output)[0]]
    result=[]
    for k in klist:
        result.append(multi_chamfer(n,inputpts,output,k))
    result=tf.add_n(result)/len(klist)
    result=(result+0.1*chamfer_max(inputpts,output)[0])
    return result
#b*n*1
def hausdis(pcd1,pcd2,d=0.12):
    ptnum1=pcd1.get_shape()[1].value
    ptnum2=pcd2.get_shape()[1].value
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)#b*n
    result=tf.reduce_mean((tf.reduce_max(dist1,axis=1)+tf.reduce_max(dist2,axis=1))/2)
    return result
#data:b*n*3
def get_normal(data,cir=True):
    if not cir:
        result=data
        dmax=np.max(result,axis=1,keepdims=True)
        dmin=np.min(result,axis=1,keepdims=True)
        length=np.max((dmax-dmin)/2,axis=-1,keepdims=True)
        center=(dmax+dmin)/2
        result=(result-center)/length
    else:
        cen=np.mean(data,axis=1,keepdims=True)
        rdismat=np.sqrt(np.sum(np.square(data-cen),axis=-1))#b*n
        r=np.max(rdismat,axis=-1,keepdims=True)
        para=1/r
        #print(np.shape(para))
        result=np.expand_dims(para,axis=-1)*(data-cen)#+cen
    return result
def train():
    start=0
    num=2048
    bneck_size=128
    n_pc_points=2048
    bsize=8
    k=1
    mlp=[64,128]
    mlp2=[128,128]
    pointcloud_pl=tf.placeholder(tf.float32,[BATCH_SIZE,n_pc_points,3],name='pointcloud_pl')
    posi_pl=tf.placeholder(tf.float32,[BATCH_SIZE,None,3],name='posi_pl')
    in_pl=tf.placeholder(tf.float32,[1,None,3],name='in_pl')
    out_pl=tf.placeholder(tf.float32,[1,None,3],name='out_pl')

    entype='pn'
    dectype='fc'
    local=dectype in ['lae','lfd']
    global_step=tf.Variable(0,trainable=False)
    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size,mode=dectype)
    with tf.variable_scope('ge'):
        if not local:
            if entype is 'dgcnn':
                word=dgcnn_kernel(pointcloud_pl, is_training=tf.constant(True), bn_decay=None)
            elif entype is 'pn2':
                _,word=local_kernel(pointcloud_pl,local=local,cenlist=None,pooling='max')
            else:
                word=encoder(pointcloud_pl,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],strides=enc_args['strides'],b_norm=enc_args['b_norm'],verbose=enc_args['verbose'])
            out=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_finish=dec_args['b_norm_finish'],verbose=dec_args['verbose'])
            if dectype is 'fd':
                #out=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_finish=dec_args['b_norm_finish'],verbose=dec_args['verbose'])
                out=tf.reshape(out,[-1,45*45,3])
            elif dectype is 'fc':
                #out=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_finish=dec_args['b_norm_finish'],verbose=dec_args['verbose'])
                out=tf.reshape(out,[-1,num,3])
        else:
            encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points//32, bneck_size,3,mode=dectype)
            with tf.variable_scope('ge'):
                cens,feats=local_kernel(pointcloud_pl,local=local,pooling='max')
                cennum=cens.get_shape()[1].value
                outlist=[]
                for i in range(cennum):
                    with tf.variable_scope('dec'+str(i)):
                        outi=tf.expand_dims(cens[:,i,:],axis=1)\
                                +tf.reshape(decoder(feats[:,i,:],layer_sizes=dec_args['layer_sizes'],local=True,b_norm=dec_args['b_norm'],b_norm_finish=dec_args['b_norm_finish'],verbose=dec_args['verbose']),[-1,n_pc_points//cennum,3])
                        outlist.append(outi)
                out=tf.concat(outlist,axis=1)

    chamferloss=multi_chamfer_func(128,pointcloud_pl,out,[4,8,16,32,64])
    cdmloss=hausdis(pointcloud_pl,out)
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        var=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        myvar=[v for v in var if v.name.split(':')[0]!='is_training']
        ivar=[v for v in var if v.name.split(':')[0]=='is_training']
        saver=tf.train.Saver(var_list=myvar)

        saver.restore(sess, tf.train.latest_checkpoint('./ae_files/'))
        sess.run(tf.assign(ivar[0],False))

        mcd_list=[]
        hd_list=[]

        DATA_DIR=getdata.getspdir()
        filelist=os.listdir(DATA_DIR)
        testfiles=getdata.getfile(os.path.join(DATA_DIR,'test_files.txt'))

        for i in range(len(testfiles)):
            testdata = getdata.load_h5(os.path.join(DATA_DIR, testfiles[i]))[:,:,:3]
            testdata=get_normal(testdata,True)
            
            allnum=int(len(testdata)/BATCH_SIZE)*BATCH_SIZE
            batch_num=int(allnum/BATCH_SIZE)
            for batch in range(batch_num):
                start_idx = (batch * BATCH_SIZE) % allnum
                end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
                batch_point=testdata[start_idx:end_idx]
                batch_point=shuffle_points(batch_point)
                
                mcd_list.append(sess.run(chamferloss,feed_dict={pointcloud_pl:batch_point}))
                hd_list.append(sess.run(cdmloss,feed_dict={pointcloud_pl:batch_point}))
        mcderr=100*mean(mcd_list)
        hderr=100*mean(hd_list)
        print(format(mcderr,'.2f'),format(hderr,'.2f'))
        
if __name__=='__main__':
    train()

'''
Created on September 2, 2017

@author: optas
'''
import numpy as np
import tensorflow as tf
import random
from encoders_decoders import  conv2d,encoder_with_convs_and_symmetry, decoder_with_fc_only,decoder_with_folding_only
from tf_ops.sampling import tf_sampling
from pointnet_util import pointnet_sa_module_msg,pointnet_sa_module
def sampling(npoint,xyz,use_type='f'):
    if use_type=='f':
        bnum=tf.shape(xyz)[0]
        idx=tf_sampling.farthest_point_sample(npoint, xyz)
        #new_xyz=tf_sampling.gather_point(xyz,idx)
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    elif use_type=='r':
        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=np.arange(ptnum)
        ptids=tf.random_shuffle(ptids,seed=None)
        ptidsc=ptids[:npoint]
        ptid=tf.cast(tf.tile(tf.reshape(ptidsc,[-1,npoint,1]),[bnum,1,1]),tf.int32)
               
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    return idx,new_xyz
def global_fix(scope,cens,feats,mlp=[128,128],mlp1=[128,128]):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        tensor0=tf.expand_dims(tf.concat([cens,feats],axis=-1),axis=2)
        #tensor0=tf.expand_dims(tf.concat([cens,feats,tf.tile(gfeat,[1,tf.shape(feats)[1],1])],axis=-1),axis=2)
        tensor=tensor0
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('global_ptstate%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
        tensorword=tensor
        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)
        tensor=tf.concat([tf.expand_dims(cens,axis=2),tf.expand_dims(feats,axis=2),tf.tile(tensor,[1,tf.shape(feats)[1],1,1])],axis=-1)
        for i,outchannel in enumerate(mlp1):
            tensor=conv2d('global_ptstate2%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
        tensor=conv2d('global_ptout',tensor,3+feats.get_shape()[-1],[1,1],padding='VALID',activation_func=None)
        newcens=cens
        newfeats=feats+tf.squeeze(tensor[:,:,:,3:],[2])
    tf.add_to_collection('cenex',tf.reduce_mean(tf.abs(tensor[:,:,:,:3])))
    return newcens,newfeats
def local_kernel(l0_xyz,local=True,cenlist=None,pooling='max',it=True):
    l0_points=None
    if cenlist is None:
        cen11,cen22=None,None
    else:
        cen11,cen22=cenlist

    cen1,feat1=pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.2], [16], [[32,32,64]],cens=cen11, is_training=it, bn_decay=None, scope='layer1', use_nchw=False,bn=True,use_knn=True,pooling=pooling)
    cen2,feat2=pointnet_sa_module_msg(cen1, feat1, 128, [0.4], [16], [[64,64,128]],cens=cen22, is_training=it, bn_decay=None, scope='layer2',use_nchw=False,bn=True,use_knn=True,pooling=pooling)
    if not local:
        l3_xyz, rfeat3,_ = pointnet_sa_module(cen2, feat2, npoint=None, radius=None, nsample=None, mlp=[128,256,128], mlp2=None, group_all=True, is_training=it, bn_decay=None, scope='layer3',pooling=pooling)
        return [cen1,cen2],tf.squeeze(rfeat3,[1])
    else:
        cen3,feat3=pointnet_sa_module_msg(cen2, feat2, 32, [0.6], [16], [[128,128,256]], is_training=it, bn_decay=None, scope='layer3',use_nchw=False,bn=True,use_knn=True,pooling=pooling)
        rcen3,rfeat3=global_fix('global3',cen3,feat3,mlp=[256,256],mlp1=[256,256])
        return rcen3,rfeat3
def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, dnum=3, bneck_post_mlp=False,mode='fc'):
    ''' Single class experiments.
    '''
    #if n_pc_points != 2048:
    #    raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    #decoder = decoder_with_fc_only
    #decoder = decoder_with_folding_only

    n_input = [n_pc_points, dnum]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': False,
                    'non_linearity':tf.nn.relu
                    }
    if mode in ['fc','lae']:
        decoder = decoder_with_fc_only
        decoder_args = {'layer_sizes': [256,256, np.prod(n_input)],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': False
                        }
    elif mode=='lfd':
        decoder = decoder_with_folding_only
        decoder_args = {'layer_sizes': [128,128,dnum],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': False,
                        'local':True
                        }
    else:
        decoder = decoder_with_folding_only
        decoder_args = {'layer_sizes': [256,256,dnum],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': False,
                        'local':False
                        }
    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args

def find_diff(inwords,mlp=[64,64]):
    words=tf.expand_dims(inwords,axis=2)
    for i,outchannel in enumerate(mlp):
        words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
        words=tf.nn.relu(words)
    result=tf.reduce_max(words,axis=1)
    return tf.squeeze(result,axis=1)
#input_feat:batch*64*(3+featlen)
#out:batch*64*64*3
def get_auchor_point(scope,input_signal,input_feat,mlp=[64,128],mlp2=[256,256],out_num=64,out_len=3,startcen=None,out_activation=None):
    with tf.variable_scope(scope):
        ftnum=input_feat.get_shape()[1].value
        ptnum=input_signal.get_shape()[1].value
        words=tf.expand_dims(input_feat,axis=2)
        _,startcen=sampling(int(out_num),input_signal,use_type='r')
        words=tf.concat([tf.expand_dims(startcen,axis=2),tf.tile(words,[1,tf.shape(startcen)[1],1,1])],axis=-1)
        for i,outchannel in enumerate(mlp2):
            words=conv2d('start_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
            words=tf.nn.relu(words)
        #wordsfeat=words
        words=tf.reduce_mean(words,axis=1,keepdims=True)
        words=tf.concat([tf.expand_dims(startcen,axis=2),tf.tile(words,[1,tf.shape(startcen)[1],1,1])],axis=-1)
        for i,outchannel in enumerate(mlp):
            words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
            words=tf.nn.relu(words)
        words=conv2d('basic_stateoutg',words,out_len,[1,1],padding='VALID',activation_func=out_activation)
        move=tf.squeeze(words,axis=2)[:,:,:3]
        newcen=tf.expand_dims(move+startcen,axis=1)
        words=tf.concat([newcen,tf.reshape(words[:,:,:,3:],[-1,1,newcen.get_shape()[2].value,1])],axis=-1)
        return words#,move
def get_auchor_fc(scope,input_signal,input_feat,mlp=[64,128],mlp2=[256,256],out_num=64,out_len=3,startcen=None,out_activation=None):
    with tf.variable_scope(scope):
        ptnum=input_feat.get_shape()[1].value
        words=tf.expand_dims(input_feat,axis=2)
        for i,outchannel in enumerate(mlp):
            words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
            words=tf.nn.relu(words)
        words=conv2d('basic_stateoutg',words,out_num*out_len,[1,1],padding='VALID',activation_func=out_activation)
        words=tf.reshape(words,[-1,ptnum,out_num,out_len])
    return words#,words
#input_pts:batch*2048*1*3
#auchor_pts:batch*1*64*3
def auchor_feat_point(input_feat,input_pts,auchor_pts,augment=True):
    auchor_pts,auchor_rs=auchor_pts[:,:,:,:3],auchor_pts[:,:,:,3:]
    vecs=input_pts-auchor_pts
    ptnum=input_pts.get_shape()[1].value
    anum=auchor_pts.get_shape()[2].value
        
    inpts=input_pts
    apts=auchor_pts
    ars=auchor_rs

    dist=tf.sqrt(1e-4+tf.reduce_sum(tf.square(inpts)+\
             tf.square(apts)-2*inpts*apts,axis=-1,keepdims=True))-0.01
    featvec=None
    #if use_gradient:
    rawratio=tf.exp(-dist/(0.01+tf.square(ars)))
    rsum=tf.reduce_sum(rawratio,axis=[1],keepdims=True)
    ratio=rawratio/(1e-8+rsum)
    if augment:
        trnum=inpts.get_shape()[1].value/apts.get_shape()[2].value
    else:
        trnum=1.0
    #print(trnum)
    #assert False
    featvec1=trnum*tf.reduce_sum(inpts*ratio,axis=1)-trnum*(1-1e-8/(1e-8+tf.squeeze(rsum,[1])))*tf.squeeze(apts,[1])
    featvec1=tf.reshape(featvec1,[-1,1,anum,3])
    return dist,featvec1
def local_loss_net(input_signal,all_feat,cennum=128,centype='lnsa',rawmask=None,activation_func=tf.nn.sigmoid,augment=True):
    with tf.variable_scope('local_loss'):
        ptnum=input_signal.get_shape()[1].value
        feat_length=input_signal.get_shape()[-1].value
        gfeat=all_feat
        gfeat=tf.expand_dims(gfeat,axis=1)
        auchor_pts=None
        if rawmask is not None:
            auchor_pts=rawmask

        if auchor_pts is None:
            if centype is 'lnsa':
                gauchor_pts=get_auchor_point('auchor_layerg',input_signal,gfeat,mlp=[128,64],out_num=cennum,out_len=4,startcen=None,out_activation=None)#88
            elif centype is 'lnfc':
                gauchor_pts=get_auchor_fc('auchor_layerg',input_signal,gfeat,mlp=[256,256,256],out_num=128,out_len=4,startcen=None,out_activation=None)
            auchor_pts=gauchor_pts
        dist,feat_vec=auchor_feat_point(gfeat,tf.expand_dims(input_signal,axis=2),auchor_pts,augment=augment)
        word=feat_vec
        reverse_word=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(dist[:,:,:int(cennum*7/8),:],axis=-2),axis=1),axis=1,keepdims=True)\
                +0.1*tf.reduce_mean(tf.reduce_max(tf.square(auchor_pts[:,:,:,3:]),axis=-1),axis=-1)#56
        lr=0.1*tf.reduce_mean(tf.reduce_max(tf.square(auchor_pts[:,:,:,3:]),axis=-1),axis=-1)
        rawmask=auchor_pts
    return word,reverse_word,rawmask,lr
def pclossnet(scope,inpts,out,centype='lnsa',augment=True,BATCH_SIZE=16):
    
    with tf.variable_scope(scope):
        with tf.variable_scope('1ad'):
            gindifval=find_diff(inpts,mlp=[64,128])
        with tf.variable_scope('1ad',reuse=True):
            goutdifval=find_diff(out,mlp=[64,128])
        gdifval=tf.concat([gindifval,goutdifval],axis=-1)
        with tf.variable_scope('2ad'):
            inlocal,rinlocal,maxinlocal,lr=local_loss_net(inpts,gdifval,cennum=128,centype=centype,augment=augment)
        with tf.variable_scope('2ad',reuse=True):
            outlocal,routlocal,maxoutlocal,_=local_loss_net(out,gdifval,rawmask=maxinlocal,centype=centype,augment=augment)
        ginlocal=inlocal
        goutlocal=outlocal
        ginlocals=tf.reshape(ginlocal,[BATCH_SIZE,-1,3])
        goutlocals=tf.reshape(goutlocal,[BATCH_SIZE,-1,3])
        local_loss=tf.reduce_mean(tf.sqrt(1e-6+tf.reduce_sum(tf.reduce_sum(tf.square(ginlocals-goutlocals),axis=[-1]),axis=-1))-0.001)

        rlossin=tf.reduce_mean(rinlocal)
        rlossout=tf.reduce_mean(routlocal)
        sg_loss_e=local_loss

        loss_cons=rlossin+rlossout#////////
        loss_e=sg_loss_e+0.01*rlossout#0.01*tf.reduce_mean(routlocal+lr)
        loss_d_local=-tf.log(local_loss+1e-5)+1*loss_cons#/////////

    return loss_e,loss_d_local


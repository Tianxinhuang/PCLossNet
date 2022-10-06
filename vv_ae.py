import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
import tf_util
import copy
import random
DATA_DIR=getdata.getspdir()

filelist=os.listdir(DATA_DIR)

from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from lossnet import mlp_architecture_ala_iclr_18,pclossnet,local_kernel
from provider import shuffle_data,shuffle_points,rotate_point_cloud,jitter_point_cloud
from dgcnn import dgcnn_kernel,dgcls_kernel
trainfiles=getdata.getfile(os.path.join(DATA_DIR,'train_files.txt'))
#testfiles=getdata.getfile(os.path.join(DATA_DIR,'test_files.txt'))

os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.set_random_seed(1)

def getidpts(pcd,ptid,ptnum):
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(pcd)[0],dtype=tf.int32),[-1,1,1]),[1,ptnum,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    result=tf.gather_nd(pcd,idx)
    return result
def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    pcd12=getidpts(pcd2,idx1,pcd1.get_shape()[1].value)
    pcd21=getidpts(pcd1,idx2,pcd2.get_shape()[1].value)
    dist1=tf.sqrt(tf.reduce_sum(tf.square(pcd1-pcd12),axis=-1))
    dist2=tf.sqrt(tf.reduce_sum(tf.square(pcd2-pcd21),axis=-1))
    dist=(tf.reduce_mean(dist1)+tf.reduce_mean(dist2))/2.0
    return dist
def emd_func(pred,gt):
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
    dist = tf.reduce_mean(dist,axis=-1)
    
    cens=tf.reduce_mean(pred,axis=1,keep_dims=True)
    radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=-1))
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist_norm)
    return emd_loss

def train(args):    
    pointcloud_pl=tf.placeholder(tf.float32,[args.batch_size,args.ptnum,3],name='pointcloud_pl')

    enctype=args.enctype
    dectype=args.dectype
    local=dectype in ['lae','lfd']

    global_step=tf.Variable(0,trainable=False)

    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(args.ptnum, args.bneck,mode=dectype)
    with tf.variable_scope('ge'):
        #Reconstruction Network with global features
        if not local:
            if enctype is 'dgcnn':
                word=dgcnn_kernel(pointcloud_pl, is_training=tf.constant(True), bn_decay=None)
            elif enctype is 'pn2':
                _,word=local_kernel(pointcloud_pl,local=local,cenlist=None,pooling='max')
            else:
                word=encoder(pointcloud_pl,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],strides=enc_args['strides'],b_norm=enc_args['b_norm'],verbose=enc_args['verbose'])
            #set the decoder
            out=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_finish=dec_args['b_norm_finish'],verbose=dec_args['verbose'])
            if dectype is 'fd':
                out=tf.reshape(out,[-1,45*45,3])
            elif dectype is 'ae':
                out=tf.reshape(out,[-1,args.ptnum,3])
        #Reconstruction Network with local features
        else:
            encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(args.ptnum//32, args.bneck,3,mode=dectype)
            with tf.variable_scope('ge'):
                cens,feats=local_kernel(pointcloud_pl,local=local,pooling='max')
                cennum=cens.get_shape()[1].value
                outlist=[]
                for i in range(cennum):
                    with tf.variable_scope('dec'+str(i)):
                        outi=tf.expand_dims(cens[:,i,:],axis=1)\
                                +tf.reshape(decoder(feats[:,i,:],layer_sizes=dec_args['layer_sizes'],local=True,b_norm=dec_args['b_norm'],b_norm_finish=dec_args['b_norm_finish'],verbose=dec_args['verbose']),[-1,args.ptnum//cennum,3])
                        outlist.append(outi)
                out=tf.concat(outlist,axis=1)

    #calculate the loss with PCLossNet
    loss_e,loss_d_local=pclossnet('pcloss',pointcloud_pl,out,BATCH_SIZE=args.batch_size,centype=args.pcltype,augment=(dectype is not 'fd'))

    allvars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    #set the parameters for reconstruction networks and PCLossNet
    varge=[v for v in allvars if 'ge' in v.name]
    varad=[v for v in allvars if '1ad' in v.name or '2ad' in v.name]
    
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    gezhengze=tf.reduce_sum([regularizer(v) for v in varge])
    lde_zhengze=tf.reduce_sum([regularizer(v) for v in varad])

    #set regularization
    loss_e=loss_e+args.ae_reg*gezhengze
    loss_d_local=loss_d_local+args.pc_reg*lde_zhengze

    #set trainstep and visualized losses
    trainstep=[]
    trainstep.append(tf.train.AdamOptimizer(learning_rate=args.ae_lr).minimize(loss_e, global_step=global_step,var_list=varge))
    trainstep.append(tf.train.AdamOptimizer(learning_rate=args.pc_lr).minimize(loss_d_local, global_step=global_step,var_list=varad))
    loss=[loss_e,loss_d_local]

    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        from tflearn import is_training
        is_training(True, session=sess)

        datalist=[]
        for j in range(args.filenum):
            traindata = getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j]))
            datalist.append(traindata)

        for i in range(args.itertime):
            for j in range(args.filenum):
                traindata=datalist[j]
                
                ids=list(range(len(traindata)))
                random.shuffle(ids)
                traindata=traindata[ids,:,:]
                
                allnum=int(len(traindata)/args.batch_size)*args.batch_size
                batch_num=int(allnum/args.batch_size)


                for batch in range(batch_num):
                    start_idx = (batch * args.batch_size) % allnum
                    end_idx=(batch*args.batch_size)%allnum+args.batch_size
                    batch_point=traindata[start_idx:end_idx]
                    feed_dict = {pointcloud_pl: batch_point[:,:args.ptnum,:]}

                    sess.run([trainstep[1]], feed_dict=feed_dict)
                    resi = sess.run([trainstep[0],loss_e], feed_dict=feed_dict)

                    if (batch+1) % args.seestep == 0:
                        print('epoch: %d '%i,'file: %d '%j,'batch: %d' %batch)
                        print('loss: ',resi[-1])
                        
            if (i+1)%args.savestep==0:
                save_path = saver.save(sess, os.path.join(args.savepath,'model'),global_step=i)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--ptnum', type=int, default=2048, help='The number of points')
    parser.add_argument('--bneck', type=int, default=128, help='The size of bottleneck layer in auto-encoder')
    parser.add_argument('--savestep', type=int, default=100, help='The interval to save checkpoint')
    parser.add_argument('--seestep', type=int, default=16, help='The batch interval to see training errors')
    parser.add_argument('--itertime', type=int, default=1000, help='The number of epochs for iteration')
    parser.add_argument('--filenum', type=int, default=6, help='The number of h5files')
    parser.add_argument('--filepath', type=str, default='./data', help='The path of h5 training data')
    parser.add_argument('--savepath', type=str, default='./modelvv_ae/', help='The path of saved checkpoint')
    parser.add_argument('--enctype', type=str, default='pn', help='The type of encoder')
    parser.add_argument('--dectype', type=str, default='ae', help='The type of decoder')
    parser.add_argument('--pcltype', type=str, default='lnsa', help='The type of PCLossNet:LNSA/LNFC')
    parser.add_argument('--ae_reg', type=float, default=0.001, help='The regularization parameter of reconstruction network')
    parser.add_argument('--pc_reg', type=float, default=0.0001, help='The regularization parameter of PCLossNet')
    parser.add_argument('--ae_lr', type=float, default=0.0001, help='The learning rate of reconstruction network')
    parser.add_argument('--pc_lr', type=float, default=0.005, help='The learning rate of PCLossNet')
    args=parser.parse_args()
    train(args) 

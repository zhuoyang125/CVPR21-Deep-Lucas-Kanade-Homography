from fastai.vision import *
from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
import imutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from numpy.linalg import inv

import sys
sys.path.append('../')
from data_read import *
from net import *
import matplotlib.pyplot as plt
import numpy as np

import argparse

import os


parser = argparse.ArgumentParser()


parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="Faces",help='MSCOCO,GoogleMap,GoogleEarth,DayNight')


parser.add_argument('--epoch_load_one', action="store", dest="epoch_load_one", type=int, default=9,help='epoch_load_one')


parser.add_argument('--epoch_load_two', action="store", dest="epoch_load_two", type=int, default=9,help='epoch_load_two')

parser.add_argument('--epoch_load_three', action="store", dest="epoch_load_three", type=int, default=9,help='epoch_load_three')

parser.add_argument('--num_iters', action="store", dest="num_iters", type=int, default=20,help='num_iters')

parser.add_argument('--feature_map_type', action="store", dest="feature_map_type", default='special',help='regular or special')

parser.add_argument('--initial_type', action="store", dest="initial_type", default='vanilla',help='vanilla, simple_net, multi_net')

parser.add_argument('--load_epoch_simplenet', action="store", dest="load_epoch_simplenet", default=40,help='load_epoch_simplenet')

parser.add_argument('--load_epoch_multinet', action="store", dest="load_epoch_multinet", default=[100,100,80],help='load_epoch_multinet')


input_parameters = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create Pytorch model for initial homography estimation
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.fc1 = nn.Linear(128*16*16, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.reshape(-1, 128*16*16)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Make directory for output images
output_dir = 'vis_outputs/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def construct_matrix_regression(batch_size,network_output,network_output_2=[0]):
    extra=tf.ones((batch_size,1))
    predicted_matrix=tf.concat([network_output,extra],axis=-1)
    predicted_matrix=tf.reshape(predicted_matrix,[batch_size,3,3])
    if len(np.shape(network_output_2))>1:
        predicted_matrix_2=tf.concat([network_output_2,extra],axis=-1)
        predicted_matrix_2=tf.reshape(predicted_matrix_2,[batch_size,3,3])
    hh_matrix=[]
    for i in range(batch_size):
        if len(np.shape(network_output_2))>1:
            hh_matrix.append(np.linalg.inv(np.dot(predicted_matrix_2[i,:,:],predicted_matrix[i,:,:])))
        else:
            hh_matrix.append(np.linalg.inv(predicted_matrix[i,:,:]))
        #hh_matrix.append(predicted_matrix[i,:,:])
    
    #return tf.linalg.inv(predicted_matrix+0.0001)
    return np.asarray(hh_matrix)

def initial_motion_COCO():
    # prepare source and target four points
    matrix_list=[]
    for i in range(1):
       
        src_points=[[0,0],[127,0],[127,127],[0,127]]

        tgt_points=[[32,32],[160,32],[160,160],[32,160]]

    
        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)
        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)


def construct_matrix(initial_matrix,scale_factor,batch_size):
    #scale_factor size_now/(size to get matrix)
    initial_matrix=tf.cast(initial_matrix,dtype=tf.float32)
    
    scale_matrix=np.eye(3)*scale_factor
    scale_matrix[2,2]=1.0
    scale_matrix=tf.cast(scale_matrix,dtype=tf.float32)
    scale_matrix_inverse=tf.linalg.inv(scale_matrix)

    scale_matrix=tf.expand_dims(scale_matrix,axis=0)
    scale_matrix=tf.tile(scale_matrix,[batch_size,1,1])

    scale_matrix_inverse=tf.expand_dims(scale_matrix_inverse,axis=0)
    scale_matrix_inverse=tf.tile(scale_matrix_inverse,[batch_size,1,1])

    final_matrix=tf.matmul(tf.matmul(scale_matrix,initial_matrix),scale_matrix_inverse)
    return final_matrix



def average_cornner_error(batch_size,predicted_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127):
    
    four_conner=[[top_left_u,top_left_v,1],[bottom_right_u,top_left_v,1],[bottom_right_u,bottom_right_v,1],[top_left_u,bottom_right_v,1]]
    four_conner=np.asarray(four_conner)
    four_conner=np.transpose(four_conner)
    four_conner=np.expand_dims(four_conner,axis=0)
    four_conner=np.tile(four_conner,[batch_size,1,1]).astype(np.float32)
    
    new_four_points=tf.matmul(predicted_matrix,four_conner)
    
    new_four_points_scale=new_four_points[:,2:,:]
    new_four_points= new_four_points/new_four_points_scale
    
    
    u_predict=new_four_points[:,0,:]
    v_predict=new_four_points[:,1,:]
    
    average_conner=tf.reduce_mean(tf.sqrt(tf.math.pow(u_predict-u_list,2)+tf.math.pow(v_predict-v_list,2)))

    
    
    return average_conner
    


def compute_ssim(img_1,img_2):

    return tf.math.pow((img_1-img_2),2)


def gt_motion_rs(u_list,v_list,batch_size=1):
    # prepare source and target four points
    matrix_list=[]
    for i in range(batch_size):
       
        src_points=[[0,0],[127,0],[127,127],[0,127]]

        #tgt_points=[[32*2+1,32*2+1],[160*2+1,32*2+1],[160*2+1,160*2+1],[32*2+1,160*2+1]]

        tgt_points=np.concatenate([u_list[i:(i+1),:],v_list[i:(i+1),:]],axis=0)
        tgt_points=np.transpose(tgt_points)
        tgt_points=np.expand_dims(tgt_points,axis=1)
       
        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)

        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)

def gt_motion_rs_random_noisy(u_list,v_list,batch_size,lambda_noisy):
    # prepare source and target four points
    matrix_list=[]
    for i in range(batch_size):
       
        src_points=[[0,0],[127,0],[127,127],[0,127]]

        #tgt_points=[[32*2+1,32*2+1],[160*2+1,32*2+1],[160*2+1,160*2+1],[32*2+1,160*2+1]]

        tgt_points=np.concatenate([u_list[i:(i+1),:],v_list[i:(i+1),:]],axis=0)
        tgt_points=np.transpose(tgt_points)
        tgt_points=np.expand_dims(tgt_points,axis=1)
       
        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)
        element_h_matrix=np.reshape(h_matrix,(9,1))
        noisy_matrix=np.zeros((9,1))
        for jj in range(8):
            #if jj!=0 and jj!=4: 
            noisy_matrix[jj,0]=element_h_matrix[jj,0]*lambda_noisy[jj]  
        noisy_matrix=np.reshape(noisy_matrix,(3,3))    
        matrix_list.append(noisy_matrix)
    return np.asarray(matrix_list).astype(np.float32)


def calculate_feature_map(input_tensor):
    bs,height,width,channel=tf.shape(input_tensor)
    path_extracted=tf.image.extract_patches(input_tensor, sizes=(1,3,3,1), strides=(1,1,1,1), rates=(1,1,1,1), padding='SAME')
    path_extracted=tf.reshape(path_extracted,(bs,height,width,channel,9))
    path_extracted_mean=tf.math.reduce_mean(path_extracted,axis=3,keepdims=True)

    #path_extracted_mean=tf.tile(path_extracted_mean,[1,1,1,channel,1])
    path_extracted=path_extracted-path_extracted_mean
    path_extracted_transpose=tf.transpose(path_extracted,(0,1,2,4,3))
    variance_matrix=tf.matmul(path_extracted_transpose,path_extracted)
    
    tracevalue=tf.linalg.trace(variance_matrix)
    row_sum=tf.reduce_sum(variance_matrix,axis=-1)
    max_row_sum=tf.math.reduce_max(row_sum,axis=-1)
    min_row_sum=tf.math.reduce_min(row_sum,axis=-1)
    mimic_ratio=(max_row_sum+min_row_sum)/2.0/tracevalue
    
    return  tf.expand_dims(mimic_ratio,axis=-1)

# Initialize Pytorch model and load weights
ckpt_pth = '/root/CVPR21-Deep-Lucas-Kanade-Homography/model_20230628_104007_127' 
model = Model().to(device)
ckpt = torch.load(ckpt_pth)
model.load_state_dict(ckpt['state_dict'])
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
optimizer.load_state_dict(ckpt['optimizer'])


if input_parameters.feature_map_type=='regular':
    load_path_one='./checkpoints/'+input_parameters.dataset_name+'/level_one_regular/'

    load_path_two='./checkpoints/'+input_parameters.dataset_name+'/level_two_regular/'

    load_path_three='./checkpoints/'+input_parameters.dataset_name+'/level_three_regular/'

    level_one_input=ResNet_first_input(if_regular=True)
    level_one_template=ResNet_first_template(if_regular=True)
    level_two_input=ResNet_second_input(if_regular=True)
    level_two_template=ResNet_second_template(if_regular=True)
    level_three_input=ResNet_third_input(if_regular=True)
    level_three_template=ResNet_third_template(if_regular=True)

elif input_parameters.feature_map_type=='special':

    load_path_one='./checkpoints/'+input_parameters.dataset_name+'/level_one/'

    load_path_two='./checkpoints/'+input_parameters.dataset_name+'/level_two/'

    load_path_three='./checkpoints/'+input_parameters.dataset_name+'/level_three/'

    level_one_input=ResNet_first_input()
    level_one_template=ResNet_first_template()
    level_two_input=ResNet_second_input()
    level_two_template=ResNet_second_template()
    level_three_input=ResNet_third_input()
    level_three_template=ResNet_third_template()


level_one_input.load_weights(load_path_one + 'epoch_'+str(input_parameters.epoch_load_one)+"input_full")

level_one_template.load_weights(load_path_one + 'epoch_'+str(input_parameters.epoch_load_one)+"template_full")

level_two_input.load_weights(load_path_two + 'epoch_'+str(input_parameters.epoch_load_two)+"input_full")

level_two_template.load_weights(load_path_two  + 'epoch_'+str(input_parameters.epoch_load_two)+"template_full")

level_three_input.load_weights(load_path_three + 'epoch_'+str(input_parameters.epoch_load_three)+"input_full")

level_three_template.load_weights(load_path_three  + 'epoch_'+str(input_parameters.epoch_load_three)+"template_full")


if input_parameters.initial_type=='vanilla':
    initial_matrix=initial_motion_COCO()
    initial_matrix=construct_matrix(initial_matrix,scale_factor=0.25,batch_size=1)

if input_parameters.initial_type=='simple_net':
    save_path_regression='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_1/'
    regression_network=Net_first()
    regression_network.load_weights(save_path_regression + 'epoch_'+str(input_parameters.load_epoch_simplenet))

if input_parameters.initial_type=='multi_net':
    save_path_one='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_1/'
    save_path_two='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_2/'
    save_path_three='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_3/'
    regression_network_one=Net_first()
    regression_network_one.load_weights(save_path_one + 'epoch_'+str(input_parameters.load_epoch_multinet[0]))
    regression_network_two=Net_second()
    regression_network_two.load_weights(save_path_two + 'epoch_'+str(input_parameters.load_epoch_multinet[1]))
    regression_network_three=Net_third()
    regression_network_three.load_weights(save_path_three + 'epoch_'+str(input_parameters.load_epoch_multinet[2]))

LK_layer_one=Lucas_Kanade_layer(batch_size=1,height_template=128,width_template=128,num_channels=1)

LK_layer_two=Lucas_Kanade_layer(batch_size=1,height_template=64,width_template=64,num_channels=1)

LK_layer_three=Lucas_Kanade_layer(batch_size=1,height_template=32,width_template=32,num_channels=1)


LK_layer_regression=Lucas_Kanade_layer(batch_size=1,height_template=192,width_template=192,num_channels=3)

if input_parameters.dataset_name=='MSCOCO':
    data_loader_caller=data_loader_MSCOCO('val')

if input_parameters.dataset_name=='GoogleMap':
    data_loader_caller=data_loader_GoogleMap('val')

if input_parameters.dataset_name=='GoogleEarth':
    data_loader_caller=data_loader_GoogleEarth('val')

if input_parameters.dataset_name=='DayNight':
    data_loader_caller=data_loader_DayNight('val')

if input_parameters.dataset_name=='Faces':
    data_loader_caller=data_loader_Faces('val')

total_error = 0.0
fk_loop = input_parameters.num_iters

for iters in range(10000000):
    if not os.path.exists(os.path.join('vis_outputs', str(iters))):
        os.mkdir(os.path.join('vis_outputs', str(iters)))

    input_img, u_list, v_list, template_img = data_loader_caller.data_read_batch(batch_size=1)

    
    if len(np.shape(input_img)) < 2:
        break
    
    if input_parameters.initial_type=='simple_net':
        #input_img_grey=tf.image.rgb_to_grayscale(input_img)
        input_img_grey=input_img
        template_img_new=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)  
        template_img_grey=template_img_new
        #template_img_grey=tf.image.rgb_to_grayscale(template_img_new)    
        network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
        homography_vector=regression_network.call(network_input,training=False)
        extra=tf.ones((1,1))
        initial_matrix=tf.concat([homography_vector,extra],axis=-1)
        initial_matrix=tf.reshape(initial_matrix,[1,3,3])
        cornner_error_pre=average_cornner_error(1,initial_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
        initial_matrix=construct_matrix(initial_matrix,scale_factor=0.25,batch_size=1)

    if input_parameters.initial_type=='multi_net':
        input_img_grey=tf.image.rgb_to_grayscale(input_img)
        template_img_new=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)  
        template_img_grey=tf.image.rgb_to_grayscale(template_img_new)
        network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
        time_a=time.time()
        homography_vector_one=regression_network_one.call(network_input,training=False)
        time_b=time.time()
        inference_time_initial=time_b-time_a

        matrix_one=construct_matrix_regression(1,homography_vector_one)
        template_img_new=LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), matrix_one)
        template_img_grey=tf.image.rgb_to_grayscale(template_img_new) 
        network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
        homography_vector_two=regression_network_two.call(network_input,training=False)
        matrix_two=construct_matrix_regression(1,homography_vector_one,homography_vector_two)
        template_img_new=LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), matrix_two)
        template_img_grey=tf.image.rgb_to_grayscale(template_img_new)  
        network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
        homography_vector_three=regression_network_three.call(network_input,training=False)

        extra=tf.ones((1,1))
        initial_matrix=tf.concat([homography_vector_three,extra],axis=-1)
        initial_matrix=tf.reshape(initial_matrix,[1,3,3])
        initial_matrix=np.dot(initial_matrix[0,:,:], np.linalg.inv(matrix_two[0,:,:]))
        initial_matrix=np.expand_dims(initial_matrix,axis=0)
        cornner_error_pre=average_cornner_error(1,initial_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
        initial_matrix=construct_matrix(initial_matrix,scale_factor=0.25,batch_size=1)

    # Getting Initial Matrix from DeepHomographyEstimator
    input_copy = input_img[0].copy()
    input_copy = input_copy[32:160, 32:160]


    template_copy = template_img[0].copy()
    
    data = np.dstack((input_copy, template_copy))

    data = np.expand_dims(data, axis=0)
    data = (data.astype(float) - 127.5) / 127.5
    data = torch.from_numpy(data)
    data = data.to(device)
    data = data.permute(0,3,1,2).float()

    outputs = model(data)
    outputs = outputs.cpu().detach().numpy()

    outputs = np.reshape(outputs, (4,2))
    outputs *= 32
    
    patch_size = 128
    top_point = (32, 32)
    left_point = (patch_size + 32, 32)
    bottom_point = (patch_size + 32, patch_size + 32)
    right_point = (32, patch_size + 32)
    four_points = np.asarray([top_point, left_point, bottom_point, right_point], dtype=np.float32)
    predicted_four_points = np.add(four_points, outputs).astype(np.float32)
    src_points = [[0,0],[127,0],[127,127],[0,127]]

    src_points = np.reshape(src_points, [4,1,2])
    predicted_four_points = np.reshape(predicted_four_points, [4,1,2])

    h_matrix, status = cv2.findHomography(src_points, predicted_four_points, 0)
    h_matrix = np.asarray([h_matrix]).astype(np.float32)
    h_matrix=construct_matrix(h_matrix,scale_factor=0.25,batch_size=1)
    initial_matrix = h_matrix

#    u_list_coord, v_list_coord = u_list[0], v_list[0]
#    actual_four_points = np.asarray([
#        [u_list_coord[0], v_list_coord[0]],
#        [u_list_coord[1], v_list_coord[1]],
#        [u_list_coord[2], v_list_coord[2]],
#        [u_list_coord[3], v_list_coord[3]]
#        ], dtype=np.float32)
#    print('Actual 4 Points: ', actual_four_points)
#    print('Predicted 4 Points:', predicted_four_points)



    
    input_feature_one=level_one_input.call(input_img,training=False)
    template_feature_one=level_one_template.call(template_img,training=False)

    input_feature_two=level_two_input.call(input_feature_one,training=False)
    template_feature_two=level_two_template.call(template_feature_one,training=False)

    input_feature_three=level_three_input.call(input_feature_two,training=False)
    template_feature_three=level_three_template.call(template_feature_two,training=False)


    if input_parameters.feature_map_type=='regular':
        input_feature_map_one=input_feature_one
        template_feature_map_one=template_feature_one

        input_feature_map_two=input_feature_two
        template_feature_map_two=template_feature_two

        input_feature_map_three=input_feature_three
        template_feature_map_three=template_feature_three

    elif input_parameters.feature_map_type=='special':
                
        input_feature_map_one=calculate_feature_map(input_feature_one)
        template_feature_map_one=calculate_feature_map(template_feature_one)

        input_feature_map_two=calculate_feature_map(input_feature_two)
        template_feature_map_two=calculate_feature_map(template_feature_two)

        input_feature_map_three=calculate_feature_map(input_feature_three)
        template_feature_map_three=calculate_feature_map(template_feature_three)


    input_feature_three_img = np.expand_dims(input_feature_map_three[0,:,:,0].numpy(), axis=-1)
    template_feature_three_img = np.expand_dims(template_feature_map_three[0,:,:,0].numpy(), axis=-1)

    input_feature_two_img = np.expand_dims(input_feature_map_two[0,:,:,0].numpy(), axis=-1)
    template_feature_two_img = np.expand_dims(template_feature_map_two[0,:,:,0].numpy(), axis=-1)

    input_feature_one_img = np.expand_dims(input_feature_map_one[0,:,:,0].numpy(), axis=-1)
    template_feature_one_img = np.expand_dims(template_feature_map_one[0,:,:,0].numpy(), axis=-1)

    updated_matrix=initial_matrix

    lucas_kanade_iter = 0
    tf.keras.utils.save_img(os.path.join('vis_outputs', str(iters), str(lucas_kanade_iter) + '.png'), template_img[0,:,:,:])
    tf.keras.utils.save_img(os.path.join('vis_outputs', str(iters), 'input.png'), input_img[0,:,:,:])
    lucas_kanade_iter += 1

    for j in range(fk_loop):
        try:
            updated_matrix=LK_layer_three.update_matrix(template_feature_map_three, input_feature_map_three, updated_matrix)
            warped_img = cv2.warpPerspective(template_img[0,:,:,:], np.squeeze(updated_matrix.numpy()), (192,192))
            warped_img = np.expand_dims(warped_img, axis=-1)
            tf.keras.utils.save_img(os.path.join('vis_outputs', str(iters), str(lucas_kanade_iter) + '.png'), warped_img)
            lucas_kanade_iter += 1

        except:
            print('s')
        print('Updated Corner Error Layer 1:', float(average_cornner_error(1,updated_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)))
    updated_matrix = construct_matrix(updated_matrix, scale_factor=2.0, batch_size=1)
    for j in range(fk_loop):
        try:
            updated_matrix=LK_layer_two.update_matrix(template_feature_map_two, input_feature_map_two, updated_matrix)
            warped_img = cv2.warpPerspective(template_img[0,:,:,:], np.squeeze(updated_matrix.numpy()), (192,192))
            warped_img = np.expand_dims(warped_img, axis=-1)
            tf.keras.utils.save_img(os.path.join('vis_outputs', str(iters), str(lucas_kanade_iter) + '.png'), warped_img)
            lucas_kanade_iter += 1
        except:
            print('s')
        print('Updated Corner Error Layer 2:', float(average_cornner_error(1,updated_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)))
    updated_matrix = construct_matrix(updated_matrix, scale_factor=2.0, batch_size=1)
    for j in range(fk_loop):
        try:
            updated_matrix=LK_layer_one.update_matrix(template_feature_map_one, input_feature_map_one, updated_matrix)
            warped_img = cv2.warpPerspective(template_img[0,:,:,:], np.squeeze(updated_matrix.numpy()), (192,192))
            warped_img = np.expand_dims(warped_img, axis=-1)
            tf.keras.utils.save_img(os.path.join('vis_outputs', str(iters), str(lucas_kanade_iter) + '.png'), warped_img)
            lucas_kanade_iter += 1
        except:
            print('s')
        print('Updated Corner Error Layer 3:', float(average_cornner_error(1,updated_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)))
    predicted_matrix=updated_matrix
    
    cornner_error=average_cornner_error(1,predicted_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)

    print(float(cornner_error))
    print('=' * 100)
    if iters > 10:
        break





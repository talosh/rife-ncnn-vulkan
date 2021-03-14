import ncnn
import rife_ncnn_vulkan
import cv2
import numpy as np

frame_data1 = cv2.imread('../test_images/scale.exr', cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1]
frame_data2 = cv2.imread('../test_images/scale.exr', cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1]

frame_data11 = (np.transpose(frame_data1, (2,0,1))*255).copy()
frame_data22 = (np.transpose(frame_data2, (2,0,1))*255).copy()
imfloat1 = ncnn.Mat(frame_data11)
imfloat2 = ncnn.Mat(frame_data22)
# imfloat1 = ncnn.Mat(frame_data1)
# imfloat2 = ncnn.Mat(frame_data2)
'''
print (frame_data11.shape)
print (frame_data11)
print (imfloat1)
'''

im1 = ncnn.Mat()
im2 = ncnn.Mat()
out = ncnn.Mat()
rife_ncnn_vulkan.decode_image('../test_images/scale.jpg', im1)
rife_ncnn_vulkan.decode_image('../test_images/scale.jpg', im2)
rife_ncnn_vulkan.decode_image('../test_images/scale.jpg', out)

rife_ncnn_vulkan.create_gpu_instance()
r = rife_ncnn_vulkan.RIFE(0, rife_v2=True)
r.load('../models/rife-v2.4')
r.process(imfloat1, imfloat2, 0.5, out)
del r
rife_ncnn_vulkan.destroy_gpu_instance()

'''
outnp = np.array(out)
print (out)
print (outnp.shape)
print (outnp)
dncnn = ncnn.Mat(outnp)
print (dncnn)
'''

rife_ncnn_vulkan.encode_image('test0.jpg', im1)
rife_ncnn_vulkan.encode_image('test1.jpg', out)
rife_ncnn_vulkan.encode_image('test2.jpg', im2)

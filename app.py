import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array,array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from keras.regularizers import l1
from keras import backend as K
import sys
import time
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.float64(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float64)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def sliceimarrY(image,ndim=16,normalize=True):
    (w,h)=image.size
    imarr2= keras.utils.img_to_array(image)
    w2=w+ndim-w%ndim
    h2=h+ndim-h%ndim
    imarr=np.zeros((h2,w2,3))
    imarr[0:h,0:w]=imarr2[0:h,0:w]
    imarr=rgb2ycbcr(imarr)
    if normalize==True:imarr=imarr/255.0
    boxsize=ndim*ndim*3
    parts=int(w2*h2*3/boxsize)
    arr=np.zeros((parts,ndim,ndim,3))
    p=0
    for y in range(0,h2-ndim+1,ndim):
        for x in range(0,w2-ndim+1,ndim):
            tmp=imarr[y:y+ndim,x:x+ndim]
            arr[p]=tmp.reshape(ndim,ndim,3)
            if p<parts-1:p+=1
    return (h,w,ndim,arr)
def desliceimarrY(arr_container):    
    h=arr_container[0]
    w=arr_container[1]    
    ndim=arr_container[2]
    arr=arr_container[3]
    p=0

    w2=w+ndim-w%ndim
    h2=h+ndim-h%ndim
    imarr=np.zeros((h2,w2,3),dtype="float32")
    for y in range(0,h2-ndim+1,ndim):
        for x in range(0,w2-ndim+1,ndim):
            imarr[y:y+ndim,x:x+ndim]=arr[p].reshape(ndim,ndim,3)
            if p<len(arr)-1:
                p+=1
    imout=np.zeros((h,w,3))
    imout=imarr[0:h,0:w]*255.0
    imout=ycbcr2rgb(imout)
    del imarr
    gc.collect()
    return imout

def LumaExtractFromArr(img):
    Luma=img[:,:,0]*0.299+img[:,:,1]*0.587+img[:,:,2]*0.114
    return Luma
def normalizearr(arr):
    arr[arr[:]>1]=1
    arr[arr[:]<0]=0
    return arr
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
def SSIMMse(y_true,y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    mseval=mse(y_true, y_pred)
    ssim=1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return 0.8*ssim+0.2*mseval # %80 ssim %20 MSE
def mdl_comp(model):
    opt=tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=[SSIMMse], metrics=[PSNR])
    return model

def SplitImageChannels(img):
    #Color channels in img[3]
    return img[3][:,:,:,0],img[3][:,:,:,1],img[3][:,:,:,2]

def MergeImageChannels2Image(InImg,Ch0,Ch1,Ch2):
    arr=np.concatenate((Ch0,Ch1,Ch2),axis=-1)
    #h,w,ndim,color array
    return (InImg[0],InImg[1],InImg[2],arr)
def main():
    print("Jpeg Artifact removing process has started...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #If you want to debug, please comment the code above.
    args = sys.argv[1:]
    if len(args)>0:
        filename=args[0]
    else:
        filename="a.png"
    nslc=128
    model=tf.keras.models.load_model("./model/model.h5",compile=False)
    model=mdl_comp(model)
    p1=time.time_ns()
    lr=tf.keras.utils.load_img(filename)
    lv=sliceimarrY(lr,nslc)
    Yl,Cb,Cr=SplitImageChannels(lv)   
    preY=model.predict(Yl,verbose=0) 
    Cb=Cb.reshape(preY.shape)
    Cr=Cr.reshape(preY.shape)
    #(Yl)Luma change to improved luma preY
    hv=MergeImageChannels2Image(lv,preY,Cb,Cr)
    img=desliceimarrY(hv)
    imgout=tf.keras.utils.array_to_img(img)
    p2=time.time_ns()
    fnsolved=filename.split(".")
    outfilename=f"{fnsolved[0]}(ARI).png"
    imgout.save(outfilename)
    print("Total Elapsed time:"+str(float(p2-p1)/1E9)+"s")
    print("Success")
if __name__=="__main__":
    main()


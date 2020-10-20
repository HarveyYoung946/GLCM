#coding:utf-8
import numpy as np
from  skimage import  data
from  matplotlib import pyplot as plt
import get_glcm
import time
from PIL import  Image

def main():
    pass

if __name__ == '__main__':
    main()

    start = time.time()

    print('...............parameter setting...................')

    nbit = 64 #gray levels
    mi,ma = 0,255 #max gray and min gray
    slide_window = 7
    #step = [2,4,8,16]步长
    #angle = [0，45,90,135]角度
    step = [2]
    angle = [0]
    print('................Load Data..........................')
    image = r"C:\Users\Administrator\Desktop\test.tif"
    img = np.array(Image.open(image))#如果图像有很多通道，则转为灰度图
    print(img.shape)
    img = np.uint8(255.0 * (img - np.min(img))/(np.max(img) - np.min(img)))#归一化
    #print(img.shape)
    h,w = img.shape
    #print(img.shape)
    print('................Calcute GLCM...................')
    glcm = get_glcm.calcu_glcm(img,mi,ma,nbit,slide_window,step,angle)
    print('................Calcute Feature.................')
    #
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = np.zeros((nbit,nbit,h,w),dtype = np.float32)
            glcm_cut = glcm[:,:,i,j,:,:]
            glcm_mean = get_glcm.calcu_glcm_mean(glcm_cut,nbit)
            glcm_variance = get_glcm.calcu_glcm_variance(glcm_cut, nbit)
            glcm_homogeneity = get_glcm.calcu_glcm_homogeneity(glcm_cut, nbit)
            glcm_contrast = get_glcm.calcu_glcm_contrast(glcm_cut, nbit)
            glcm_dissimilarity = get_glcm.calcu_glcm_dissimilarity(glcm_cut, nbit)
            glcm_entropy = get_glcm.calcu_glcm_entropy(glcm_cut,nbit)
            glcm_energy = get_glcm.calcu_glcm_energy(glcm_cut, nbit)
            glcm_correlation = get_glcm.calcu_glcm_correlation(glcm_cut, nbit)
            glcm_Auto_correlation = get_glcm.calcu_glcm_Auto_correlation(glcm_cut, nbit)
    print('................Display and Result.................')
    plt.figure(figsize=(10,4.5))

    font = {'family':'Times New Roman','Weight':'normal','size':12,}

    plt.subplot(2,5,1)
    plt.tick_params(labelbottom=False,labelleft=False)
    plt.axis('off')
    plt.imshow(img,cmap='gray')
    plt.title('Original',font)

    plt.subplot(2, 5, 2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_mean, cmap='gray')
    plt.title('Mean', font)

    plt.subplot(2, 5, 3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_variance, cmap='gray')
    plt.title('Variance', font)

    plt.subplot(2, 5, 4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_homogeneity, cmap='gray')
    plt.title('Homogeneity', font)

    plt.subplot(2, 5, 5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_contrast, cmap='gray')
    plt.title('Contrast', font)

    plt.subplot(2, 5, 6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_dissimilarity, cmap='gray')
    plt.title('Dissimilarity', font)

    plt.subplot(2, 5, 7)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_entropy, cmap='gray')
    plt.title('Entropy', font)

    plt.subplot(2, 5, 8)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_energy, cmap='gray')
    plt.title('Energy', font)

    plt.subplot(2, 5, 9)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_correlation, cmap='gray')
    plt.title('Correlation', font)

    plt.subplot(2, 5, 10)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_Auto_correlation, cmap='gray')
    plt.title('Auto Correction', font)

    plt.tight_layout(pad=0.5)

    plt.show()

    end = time.time()
    print('Code run time:',end-start)

            



import cv2 as cv
import numpy as np

# 使用cv.imread()函数读取图片，任务一
image = cv.imread("/home/david/cv-course/e645a676a5d84bc52030b36202d88a5.jpg")

print(np.ndim(image))#任务二
print(image.shape)#任务二

print(image.shape[0])#任务六

gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)#任务四，转成灰度图

cv.imwrite("11.png",gray)#任务五，保存灰度图

# 使用cv.imshow()函数显示图片，任务三
cv.imshow("22", image)#显示彩色图
cv.imshow("33", gray)#显示灰色图
# 使用cv.waitKey()函数等待键盘输入
cv.waitKey(0) # 0表示无限等待（若没有这个函数，cv.imshow()的窗口就会一闪而过）
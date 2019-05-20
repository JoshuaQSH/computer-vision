from cProfile import Profile
import pstats
import numpy as np
import cv2
import random
import time
import matplotlib.pyplot as plt
defaultNbSamples = 20
defaultReqMatches = 2
defaultRadius = 20
defaultSubsamplingFactor = 16
background = 0
foreground = 255


def Initial_ViBe(gray, samples):

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    width = gray.shape[1]
    print("*****Initlize Time******")
    start1 = time.clock()
    for i in range(height):
        for j in range(width):
            for k in range(defaultNbSamples):
                rand = round(random.uniform(-1, 1))
                r = i + rand
                if r < 0:
                    r = 0
                if r > height:
                    r = height - 1
                rand = round(random.uniform(-1, 1))
                c = j + rand
                if c < 0:
                    c = 0
                if c > width:
                    c = width - 1
                samples[i][j][k] = gray[i][j]
            samples[i][j][defaultNbSamples - 1] = 0
    end = time.clock()
    print(end - start1)
    print("******************************")
    return samples


def update(gray, samples):
    height = gray.shape[0]
    width = gray.shape[1]
    segMat = np.zeros((gray.shape[0], gray.shape[1], 1))
    for i in range(height):
        for j in range(width):
            count = 0
            idex = 0
            dist = 0
            # 遍历每个像素，判断与背景样本的相似程度
            while count < defaultReqMatches and idex < defaultNbSamples:
                dist = abs(gray[i][j] - samples[i][j][idex])
                if dist < defaultRadius:
                    # 统计相似度小于阈值的个数
                    count = count + 1
                idex = idex + 1
            # 大于#min，则认为该点为背景点
            if count >= defaultReqMatches:
                # 判断为背景像素，只有背景点才可以被用来传播和更新存储样本值
                samples[i][j][defaultNbSamples - 1] = 0
                # 输出二值图像用
                segMat[i][j] = background
                rand = round(random.uniform(0, defaultSubsamplingFactor - 1))
                if rand == 0:
                    rand = round(random.uniform(0, defaultNbSamples - 1))
                    samples[i][j][rand] = gray[i][j]
                rand = round(random.uniform(0, defaultSubsamplingFactor - 1))
                if rand == 0:
                    rand = round(random.uniform(-1, 1))
                    iN = i + rand
                    # 以下是防止越界
                    if iN < 0:
                        iN = 0
                    if iN > height:
                        iN = height - 1
                    rand = round(random.uniform(-1, 1))
                    jN = j + rand
                    if jN < 0:
                        jN = 0
                    if jN > width:
                        jN = width - 1
                    rand = round(random.uniform(0, defaultNbSamples - 1))
                    samples[i][j][rand] = gray[i][j]
            else:
                segMat[i][j] = foreground
                samples[i][j][defaultNbSamples -
                              1] = samples[i][j][defaultNbSamples - 1] + 1
                if samples[i][j][defaultNbSamples - 1] > 50:
                    rand = round(random.uniform(0, defaultNbSamples - 1))
                    if rand == 0:
                        rand = round(random.uniform(0, defaultNbSamples - 1))
                        samples[i][j][rand] = gray[i][j]
    return segMat, samples


def main():
    vc = cv2.VideoCapture("nhz.mpg")
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    samples = np.zeros((500, 500, defaultNbSamples))
    while rval:
        rval, frame = vc.read()
        if c == 1:
            # 将输入转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples = Initial_ViBe(gray, samples)
        else:
            # 将输入转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 输出二值图
            (segMat, samples) = update(gray, samples)
            #　转为uint8类型
            segMat = np.array(segMat, np.uint8)
            # 形态学处理模板初始化
            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            # 开运算
            opening = cv2.morphologyEx(segMat, cv2.MORPH_OPEN, kernel1)
            # 形态学处理模板初始化
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            # 闭运算
            closed = cv2.morphologyEx(segMat, cv2.MORPH_CLOSE, kernel2)

            # 寻找轮廓
            contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(0, len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                print(w * h)
                if w * h > 400 and w * h < 10000:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
            cv2.imshow("frame", frame)

            cv2.imwrite("./result/" + str(c) + ".jpg", frame, [int(cv2.IMWRITE_PNG_STRATEGY)])
            k = cv2.waitKey(1)
            if k == ord('q'):
                vc.release()
                cv2.destroyAllWindows()
        c = c + 1


main()
import cv2
import numpy as np
import copy
import os
#png = cv2.imread('target/18.png', cv2.IMREAD_UNCHANGED)
""" 将文件夹中的png图像裁剪，裁去完全透明的行和列
"""

def remove_blank_area(img):
    a = copy.copy(img)


    img_transparency = a[:, :, 3]

    idx0 = np.argmax(img_transparency, axis=1)  # h个列索引
    scores = img_transparency[np.arange(img_transparency.shape[0]), idx0]  # 行最大
    # print(scores)
    idx_h = np.where(scores == 0)

    idx1 = np.argmax(img_transparency, axis=0)  # w个行索引
    scores = img_transparency[idx1, np.arange(img_transparency.shape[1])]  # 列最大
    idx_w = np.where(scores == 0)

    a = np.delete(a, idx_h, 0)
    a = np.delete(a, idx_w, 1)
    return a



if __name__ == '__main__':
    src_folder = 'BigClearTarget'
    dst_folder = '3'
    cnt = 0

    # 确认目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 循环遍历源文件夹中的PNG文件并调整大小
    for filename in os.listdir(src_folder):
        if filename.endswith('.png'):
            # 加载PNG图片
            img = cv2.imread(os.path.join(src_folder, filename), cv2.IMREAD_UNCHANGED)



            # 重新调整大小为指定大小
            try:
                resized_img = remove_blank_area(img)

            except:
                continue

            #resized_img = img
            #cv2.imshow('a',resized_img)
            #cv2.waitKey(0)

            new_filename = str(1000+cnt) + '.png'

            # 将调整后的图像保存到目标文件夹中
            cv2.imwrite(os.path.join(dst_folder, new_filename), resized_img)
            print(cnt)
            cnt += 1






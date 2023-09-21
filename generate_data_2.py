import os
import cv2
import numpy.random as random
import numpy as np
import copy
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

""" generate_dataset = generate_dataset(dir_list列表[‘label-1靶标图片相对于项目绝对地址的相对地址’,...], label_list列表['label-1',...], 
                    'DIV2K_valid_HR'背景图片存储地址, r'D:\pythonproject\generatedataset'项目的绝对地址, 
                    'DIV2K_Grey/valid'图片保存地址, 'DIV2K_Grey/trash'xml文件保存地址)


    从dir_list随机选取相对路径路径与项目绝对地址组合读取png靶标，根据索引对应label_list标签.
    ，从background文件夹读取png或jpg背景，背景均匀划分为四块，将四张target图像分别覆盖在四个区域.
    生成图片保存在生成'DIV2K_Grey/valid'   
    voc格式xml文件保存在'DIV2K_Grey/trash'

"""

def calculate_iou(lis1, lis2):
    # 计算相交区域的左上角和右下角坐标
    xmin1, ymin1, xmax1, ymax1 = lis1
    xmin2, ymin2, xmax2, ymax2 = lis2

    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    # 计算相交区域的宽度和高度
    inter_width = inter_xmax - inter_xmin
    inter_height = inter_ymax - inter_ymin

    # 如果相交区域的宽度或高度小于等于0，表示两个矩形框没有重叠，返回0作为IoU
    if inter_width <= 0 or inter_height <= 0:
        return 0.0
    # 计算相交区域的面积
    inter_area = inter_width * inter_height
    # 计算两个矩形框的面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    # 计算并集的面积
    union_area = area1 + area2 - inter_area
    # 计算IoU
    iou = inter_area / union_area
    return iou

class generate_dataset(object):
    def __init__(self, target_dir_path_list, label_list, background_dir_path,
                 project_abspath, image_save_path, annotation_save_path):

        self.target_dir_path_list = target_dir_path_list
        self.label_list = label_list
        self.background_dir_path = background_dir_path
        self.project_abspath = project_abspath
        self.image_save_path = image_save_path
        self.annotation_save_path = annotation_save_path
        self.classes = len(self.label_list)

        self.target_size_range = [0.1, 0.25]  # 靶标大小的区间，占背景图最短边的比例[0.06, 0.25]
        self.target_filename_list = []
        # self.target_filename_list = [os.path.join(self.target_dir_path, f) for f in os.listdir(self.target_dir_path) if
        #              f.endswith('.png')]
        self.background_filename_list = [os.path.join(self.background_dir_path, f) for f in
                                         os.listdir(self.background_dir_path) if
                                         f.endswith('.png') or f.endswith('.jpg')]

        self.seize_filename()

        self.i = 0

    def seize_filename(self):
        # 获取filrname list 【【】，【】，【】】
        for target_dir in self.target_dir_path_list:
            self.target_filename_list.append([os.path.join(target_dir, f) for f in os.listdir(target_dir) if
                                              f.endswith('.png')])

    def add_alpha_channel(self, img):
        """ 为jpg图像添加alpha通道 """

        b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

        img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
        return img_new

    def merge_img(self, jpg_img, png_img, y1, y2, x1, x2):
        """ 将png透明图像与jpg图像叠加
            y1,y2,x1,x2为叠加位置坐标值
        """
        # cv2.imshow(png_img)
        # cv2.imshow(jpg_img)

        # 判断jpg图像是否已经为4通道
        if jpg_img.shape[2] == 3:
            jpg_img = self.add_alpha_channel(jpg_img)

        '''
        当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
        这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
        '''
        yy1 = 0
        yy2 = png_img.shape[0]
        xx1 = 0
        xx2 = png_img.shape[1]

        if x1 < 0:
            xx1 = -x1
            x1 = 0
        if y1 < 0:
            yy1 = - y1
            y1 = 0
        if x2 > jpg_img.shape[1]:
            xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
            x2 = jpg_img.shape[1]
        if y2 > jpg_img.shape[0]:
            yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
            y2 = jpg_img.shape[0]

        # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
        # if not (png_img.shape[2] == 4):
        #     print(png_img.shape)
        #     print('-------------')
        #     cv2.imshow('qq', png_img)
        #     cv2.waitKey(0)
        alpha_png = png_img[yy1:yy2, xx1:xx2, 3] / 255.0
        alpha_jpg = 1 - alpha_png

        # 开始叠加
        for c in range(0, 3):
            jpg_img[y1:y2, x1:x2, c] = (
                    (alpha_jpg * jpg_img[y1:y2, x1:x2, c]) + (alpha_png * png_img[yy1:yy2, xx1:xx2, c]))

        return jpg_img

    def subarea(self, bg_h, bg_w, target_size):
        """ 划分四个区域
        """
        target_gap = target_size

        # assert(target_size<bg_h//2 and target_size<bg_w//2)

        subarea_dict = {'sub1': [0, bg_w // 2 - target_gap, 0, bg_h // 2 - target_gap],
                        'sub2': [bg_w // 2, bg_w - target_gap, 0, bg_h // 2 - target_gap],
                        'sub3': [0, bg_w // 2 - target_gap, bg_h // 2, bg_h - target_gap],
                        'sub4': [bg_w // 2, bg_w - target_gap, bg_h // 2, bg_h - target_gap]}

        # [xmin,xmax,ymin,ymax]
        return subarea_dict

    def imread_image(self):
        """ 读取图片
        """
        choise_list = []
        label_list = np.array(self.label_list)
        num_images = 5

        choise_list = random.randint(0, self.classes, size=num_images, dtype=int)

        random_images = random.choice(self.target_filename_list[choise_list[0]])
        self.img1 = cv2.imread(random_images, cv2.IMREAD_UNCHANGED)

        random_images = random.choice(self.target_filename_list[choise_list[1]])
        self.img2 = cv2.imread(random_images, cv2.IMREAD_UNCHANGED)

        random_images = random.choice(self.target_filename_list[choise_list[2]])
        self.img3 = cv2.imread(random_images, cv2.IMREAD_UNCHANGED)

        random_images = random.choice(self.target_filename_list[choise_list[3]])
        self.img4 = cv2.imread(random_images, cv2.IMREAD_UNCHANGED)

        random_images = random.choice(self.target_filename_list[choise_list[4]])
        self.img5 = cv2.imread(random_images, cv2.IMREAD_UNCHANGED)

        self.background = cv2.imread(self.background_filename_list[self.i])

        self.img_label = label_list[choise_list]
        self.img_label = self.img_label.tolist()

        # 使用os.path.basename()获取文件名（包含后缀）
        filename_with_extension = os.path.basename(self.background_filename_list[self.i])

        # 使用os.path.splitext()获取文件名和后缀的分割结果
        filename = os.path.splitext(filename_with_extension)[0]

        return filename  # 返回当前读取背景的图片名，

        # cv2.imshow('a', self.background)
        # cv2.waitKey(0)

    def random_resize(self):
        """ target图片随机变换大小返回最长边
        """

        maxlength = 0
        bg_minlength = min(self.background.shape[0], self.background.shape[1])
        target_size_range = [0, 0]
        target_size_range[0] = int(self.target_size_range[0] * bg_minlength)
        target_size_range[1] = int(self.target_size_range[1] * bg_minlength)

        h, w = self.img1.shape[0:2]
        h_a = random.randint(target_size_range[0], target_size_range[1])
        w_a = int(h_a * w / h)
        maxlength = max(h_a, w_a, maxlength)
        self.img1 = cv2.resize(self.img1, dsize=(w_a, h_a))

        h, w = self.img2.shape[0:2]
        h_a = random.randint(target_size_range[0], target_size_range[1])
        w_a = int(h_a * w / h)
        maxlength = max(h_a, w_a, maxlength)
        self.img2 = cv2.resize(self.img2, dsize=(w_a, h_a))

        h, w = self.img3.shape[0:2]
        h_a = random.randint(target_size_range[0], target_size_range[1])
        w_a = int(h_a * w / h)
        maxlength = max(h_a, w_a, maxlength)
        self.img3 = cv2.resize(self.img3, dsize=(w_a, h_a))

        h, w = self.img4.shape[0:2]
        h_a = random.randint(target_size_range[0], target_size_range[1])
        w_a = int(h_a * w / h)
        maxlength = max(h_a, w_a, maxlength)
        self.img4 = cv2.resize(self.img4, dsize=(w_a, h_a))

        h, w = self.img5.shape[0:2]
        h_a = random.randint(target_size_range[0], target_size_range[1])
        w_a = int(h_a * w / h)
        maxlength = max(h_a, w_a, maxlength)
        self.img5 = cv2.resize(self.img5, dsize=(w_a, h_a))

        return maxlength

    def voc_annotation(self, objname_list, filename, location_list):
        """ 根据target插入位置编写voc格式xml文件
        """
        height, width, channel = self.background.shape
        xml_root = ET.Element("annotation")
        ET.SubElement(xml_root, "folder").text = self.image_save_path
        ET.SubElement(xml_root, "filename").text = filename
        ET.SubElement(xml_root, "path").text = os.path.join(self.project_abspath, self.image_save_path,
                                                            filename + '.jpg')

        source_node = ET.SubElement(xml_root, "source")
        ET.SubElement(source_node, "database").text = "Unknown"

        size_node = ET.SubElement(xml_root, "size")
        ET.SubElement(size_node, "width").text = str(width)
        ET.SubElement(size_node, "height").text = str(height)
        ET.SubElement(size_node, "depth").text = str(channel)

        self.test = copy.copy(self.background)

        for index, location in enumerate(location_list):
            x_min, y_min, x_max, y_max = location

            cv2.rectangle(self.test, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

            object_node = ET.SubElement(xml_root, "object")
            ET.SubElement(object_node, "name").text = objname_list[index]
            bndbox_node = ET.SubElement(object_node, "bndbox")  # 添加边界框节点
            ET.SubElement(bndbox_node, "xmin").text = str(x_min)
            ET.SubElement(bndbox_node, "ymin").text = str(y_min)
            ET.SubElement(bndbox_node, "xmax").text = str(x_max)
            ET.SubElement(bndbox_node, "ymax").text = str(y_max)

        xml_file_path = os.path.join(self.annotation_save_path, filename + '.xml')
        tree = ET.ElementTree(xml_root)
        tree.write(xml_file_path, encoding="utf-8")

    def random_set_img(self,location_list,overlap_rate):

        x1, y1, x2, y2 = location_list[random.randint(len(location_list))]
        w1 = x2 - x1
        h1 = y2 - y1
        h2, w2 = self.img5.shape[:2]
        x_in_1 = x1 + overlap_rate * w1 - w2
        y_in_1 = y1 + overlap_rate * h1 - h2
        x_in_2 = x2 - overlap_rate * w1
        y_in_2 = y2 - overlap_rate * h1

        x_ex_1 = x1 - overlap_rate * w1 - w2
        y_ex_1 = y1 - overlap_rate * h1 - h2
        x_ex_2 = x2 + overlap_rate * w1
        y_ex_2 = y2 + overlap_rate * h1

        bg_h, bg_w = self.background.shape[0], self.background.shape[1]

        while True:
            x = random.randint(x_ex_1, x_ex_2)
            y = random.randint(y_ex_1, y_ex_2)
            if x < 0 or x > bg_w - w2 or y < 0 or y > bg_h - h2:
                continue
            if x_in_1 < x < x_in_2 and y_in_1 < y < y_in_2:
                continue

            return x, y, x+w2, y+h2





    def main(self):
        while self.i < len(self.background_filename_list):
            if self.i == 10:
                a = 1
            location_list = []
            background_prefix = self.imread_image()  # 读取target和背景图，返回当前背景图名字前缀

            while not (self.img1.shape[2] == self.img2.shape[2] == self.img3.shape[2] == self.img4.shape[2] == self.img5.shape[2] == 4):
                background_prefix = self.imread_image()

            if not isinstance(self.background, np.ndarray):
                self.i += 1
                continue
            if self.background.shape[0] < 500 and self.background.shape[1] < 500:
                self.i += 1
                continue

            gap = self.random_resize()
            subarea_dict = self.subarea(self.background.shape[0], self.background.shape[1], gap)
            # print(subarea_dict)

            print(self.background.shape)

            # sub1 img1
            x_min, x_max, y_min, y_max = subarea_dict['sub1']
            x1 = random.randint(x_min, x_max)
            y1 = random.randint(y_min, y_max)
            x2 = x1 + self.img1.shape[1]
            y2 = y1 + self.img1.shape[0]
            self.background = self.merge_img(self.background, self.img1, y1, y2, x1, x2)
            self.background = self.background[:, :, 0:3]
            location_list.append([x1, y1, x2, y2])
            # print('img1', self.img1.shape)
            # print("x1,y1,x2,y2:   ", x1, y1, x2, y2)

            # sub2 img2
            x_min, x_max, y_min, y_max = subarea_dict['sub2']
            x1 = random.randint(x_min, x_max)
            y1 = random.randint(y_min, y_max)
            x2 = x1 + self.img2.shape[1]
            y2 = y1 + self.img2.shape[0]
            self.background = self.merge_img(self.background, self.img2, y1, y2, x1, x2)
            self.background = self.background[:, :, 0:3]
            location_list.append([x1, y1, x2, y2])
            # print('img2',self.img2.shape)
            # print("x1,y1,x2,y2:   ", x1, y1, x2, y2)

            # sub3 img3
            x_min, x_max, y_min, y_max = subarea_dict['sub3']
            x1 = random.randint(x_min, x_max)
            y1 = random.randint(y_min, y_max)
            x2 = x1 + self.img3.shape[1]
            y2 = y1 + self.img3.shape[0]
            self.background = self.merge_img(self.background, self.img3, y1, y2, x1, x2)
            self.background = self.background[:, :, 0:3]
            location_list.append([x1, y1, x2, y2])
            # print('img3', self.img3.shape)
            # print("x1,y1,x2,y2:   ", x1, y1, x2, y2)

            # sub4 img4
            x_min, x_max, y_min, y_max = subarea_dict['sub4']
            x1 = random.randint(x_min, x_max)
            y1 = random.randint(y_min, y_max)
            x2 = x1 + self.img4.shape[1]
            y2 = y1 + self.img4.shape[0]
            self.background = self.merge_img(self.background, self.img4, y1, y2, x1, x2)
            self.background = self.background[:, :, 0:3]
            location_list.append([x1, y1, x2, y2])
            # print('img4', self.img4.shape)
            # print("x1,y1,x2,y2:   ", x1, y1, x2, y2)

            #img5 random choice sub space:
            x1, y1, x2, y2 = self.random_set_img(location_list, overlap_rate=0.1)
            self.background = self.merge_img(self.background, self.img5, y1, y2, x1, x2)
            self.background = self.background[:, :, 0:3]
            location_list.append([x1, y1, x2, y2])
            # 考虑一个img5把其他几个遮掉的问题
            lis1,lis2,lis3,lis4,lis5 = location_list
            if calculate_iou(lis1,lis5)>0.08 or calculate_iou(lis2,lis5)>0.08 \
                    or calculate_iou(lis3,lis5)>0.08 or calculate_iou(lis4,lis5)>0.08:
                print('$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$')
                continue

            filename = background_prefix + '_' + str(100000 + self.i)
            image_path = os.path.join(self.image_save_path, filename + '.jpg')

            gray_img = cv2.cvtColor(self.background, cv2.COLOR_RGB2GRAY)
            gray_img = gray_img[:, :, np.newaxis]
            gray_img = np.dstack((gray_img, gray_img, gray_img))
            cv2.imwrite(image_path, gray_img)

            #############################################################

            # # 获取原始图像的高度和宽度
            # height, width = self.background.shape[:2]
            # # 计算进行裁剪后的图像的新高度和新宽度，使其能够被3整除
            # new_height = height - (height % 3)
            # new_width = width - (width % 3)
            # # 对图像进行裁剪
            # self.background = self.background[:new_height, :new_width]
            # cv2.imwrite(image_path,self.background)
            # #LR
            # image_path_lr = os.path.join(r'D:\pythonproject\generatedataset\test\lr', filename + '.jpg')
            # lr_height = new_height // 3
            # lr_width = new_width // 3
            # resized_image = cv2.resize(self.background, (int(lr_width), int(lr_height)), interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite(image_path_lr, resized_image)

            ############################################################
            self.voc_annotation(self.img_label, filename, location_list)
            "=================  test  ===================="
            # cv2.imshow('background',self.background)
            # cv2.imshow('test', self.test)
            # cv2.waitKey(0)
            "============================================="

            self.i += 1


if __name__ == "__main__":
    dir_list = [r'D:\pythonproject\generatedataset\cross_target', r'D:\pythonproject\generatedataset\t_target']
    label_list = ['cross_target', 't_target']
    generate_dataset = generate_dataset(dir_list, label_list, 'background', r'D:\pythonproject\generatedataset',
                                        'VOC2007_C/JPEGimages_',
                                        'VOC2007_C/Annotations_')
    # target_dir_path_list,label_list,background_dir_path,project_abspath,image_save_path,annotation_save_path:

    generate_dataset.main()

    # print(dict)




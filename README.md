# Generate_COCO_VOC_Dataset
背景叠加目标创建voc和coco格式数据集的python代码


#generate_data.py:#
Used to generate data sets in VOC format. Before use, you need to create a folder in the
project directory to put the background picture, and create multiple folders to store the target image
without background in png format (you can use the cut_png.py file to cut off the airspace part of the target
image). Modify the path according to the comments, this file will traverse the background picture folder,
randomly select the target superimposed on the background, and finally converted into a grayscale image and
saved as jpg (you can modify the file and do not convert the grayscale image at the end), the output picture
and xml file will be saved to your custom folder.
用来生成VOC格式的数据集。使用前需要在项目目录下新建一个文件夹放背景图片，新建多个文件夹存放png格式没有背景的目标图片
（可以使用cut_png.py文件把目标图片中的空域部分裁剪掉）。根据注释修改路径，这个文件会遍历背景图片文件夹，随机选择目标叠加到背景上，
最后转成灰度图保存为jpg（可以修改文件最后不进行灰度图转换），输出的图片和xml文件会保存到你自定义的文件夹里。


#generate_data_coco.py:#
The function is the same, only read the coco format marked targets and marked information, and then superimpose
on the background picture
功能一样，只是读取coco格式标注好的目标和标注信息，然后叠加到背景图片里

import os
import random

# 设置路径
train_percent = 0.7
val_percent = 0.2
xml_dir = R"D:\Code\third_party\PaddleDetection-release-2.6\dataset\dog_mouse_other_voc\annotations"
save_dir = R"D:\Code\third_party\PaddleDetection-release-2.6\dataset\dog_mouse_other_voc\ImageSets\Main"

# 确保输出目录存在
os.makedirs(save_dir, exist_ok=True)

# 获取所有标注文件
total_xml = os.listdir(xml_dir)
total_xml = [xml for xml in total_xml if xml.endswith('.xml')]  # 仅保留XML文件

# 打乱列表
random.shuffle(total_xml)

# 划分数据集
num = len(total_xml)
num_train = int(num * train_percent)
num_val = int(num * val_percent)
num_test = num - num_train - num_val

train_list = total_xml[:num_train]
val_list = total_xml[num_train:num_train + num_val]
test_list = total_xml[num_train + num_val:]

# 写入文件
with open(os.path.join(save_dir, "train.txt"), "w") as ftrain, \
     open(os.path.join(save_dir, "val.txt"), "w") as fval, \
     open(os.path.join(save_dir, "test.txt"), "w") as ftest:
    for xml in train_list:
        ftrain.write(xml[:-4] + "\n")
    for xml in val_list:
        fval.write(xml[:-4] + "\n")
    for xml in test_list:
        ftest.write(xml[:-4] + "\n")

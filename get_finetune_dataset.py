import os
import random
import shutil
import numpy as np

def generate_unique_filename(folder, filename):
    name, ext = os.path.splitext(filename)
    count = 1
    new_filename = filename
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{name}_{count}{ext}"
        count += 1
    return new_filename

def copy_images_with_labels(
        image_folder,
        label_folder,
        label_pre_folder,
        label_Esa_folder,
        target_image_folder, 
        target_label_folder,
        target_label_pre_folder,
        target_label_Esa_folder,
        num_images):
    # 获取图像文件夹中的所有图像文件
    image_files = os.listdir(image_folder)


    # 随机选择图像文件
    seed = 800
    np.random.seed(seed)
    selected_images = random.sample(image_files, num_images)

    for image_file in selected_images:
        # 复制选定的图像到目标图像文件夹
        source_image_path = os.path.join(image_folder, image_file)
        target_image_path = os.path.join(target_image_folder, image_file)
        if os.path.exists(target_image_path):
            # 生成新的文件名
            new_image_file = generate_unique_filename(target_image_folder, image_file)
            target_image_path = os.path.join(target_image_folder, new_image_file)
        shutil.copy(source_image_path, target_image_path)

        # 从图像名称中提取标签文件名
        label_file = os.path.splitext(image_file)[0] + '.png'
        source_label_path = os.path.join(label_folder, label_file)
        target_label_path = os.path.join(target_label_folder, label_file)
        if os.path.exists(target_label_path):
            # 生成新的文件名
            new_label_file = generate_unique_filename(target_label_folder, label_file)
            target_label_path = os.path.join(target_label_folder, new_label_file)
        shutil.copy(source_label_path, target_label_path)

        # label_pre_file = os.path.splitext(image_file)[0] + '.png'
        # source_label_pre_path = os.path.join(label_pre_folder, label_pre_file)
        # target_label_pre_path = os.path.join(target_label_pre_folder, label_pre_file)
        # if os.path.exists(target_label_path):
        #     # 生成新的文件名
        #     new_label_pre_file = generate_unique_filename(target_label_pre_path, label_pre_file)
        #     target_label_path = os.path.join(target_label_pre_path, new_label_pre_file)
        # shutil.copy(source_label_pre_path, target_label_pre_path)

        # label_Esa_file = os.path.splitext(image_file)[0] + '.png'
        # source_label_Esa_path = os.path.join(label_Esa_folder, label_Esa_file)
        # target_label_Esa_path = os.path.join(target_label_Esa_folder, label_Esa_file)
        # if os.path.exists(target_label_path):
        #     # 生成新的文件名
        #     new_label_Esa_file = generate_unique_filename(target_label_Esa_path, label_Esa_file)
        #     target_label_path = os.path.join(target_label_Esa_path, new_label_Esa_file)
        # shutil.copy(source_label_Esa_path, target_label_Esa_path)
        # 删除源文件
        # os.remove(source_image_path)
        # os.remove(source_label_path)
        # os.remove(source_label_pre_path)
        # os.remove(source_label_Esa_path)

def Not_e_Make(path):
    if not os.path.exists(path):
        os.makedirs(path)
# 示例用法
num_images = 5 # 选择图像的百分比
image_folder = 'RS_sample\Fine-tune_samples\images\\50'
label_folder = 'RS_sample\Fine-tune_samples\\annotations\\50'

label_pre_folder = 'RS_sample\copy\Hunan_level_18\pre\\test'
label_Esa_folder = 'RS_sample\copy\Hunan_level_18\Esa\\test'

target_image_folder = 'RS_sample\Fine-tune_samples\images\\5'
target_label_folder = 'RS_sample\Fine-tune_samples\\annotations\\5'

target_label_pre_folder = 'RS_sample\copy\Hunan_level_18\pre\\training_all'
target_label_Esa_folder = 'RS_sample\copy\Hunan_level_18\Esa\\training_all'
Not_e_Make(target_image_folder)
Not_e_Make(target_label_folder)
# Not_e_Make(target_label_pre_folder)
# Not_e_Make(target_label_Esa_folder)




copy_images_with_labels(
    image_folder,
    label_folder,
    label_pre_folder,
    label_Esa_folder,
    target_image_folder, 
    target_label_folder,
    target_label_pre_folder,
    target_label_Esa_folder,
    num_images)

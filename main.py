import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from osgeo import gdal,ogr,osr
import os
from learning.miou import IoU #从U-TAEcopy过来的精度评价模块，可以讲边缘部分的放到这里
import json
from alive_progress import alive_bar


def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    '''
    
        return width, height, bands, data, geotrans, proj
    '''
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize 
    #  栅格矩阵的行数
    height = dataset.RasterYSize 
    #  波段数
    bands = dataset.RasterCount 
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

# 用于图像可视化
def Norm_img(img):
    if len(img.shape) == 2:
        min = np.min(img)
        max = np.max(img)
        img = (img - min)/(max-min)*255
    else:
        shape = img.shape[2]
        for i in range(shape):
            min = np.min(img[:,:,i])
            max = np.max(img[:,:,i])
            img[:,:,i] = (img[:,:,i] - min)/(max-min)*255
    img_new = img
    return img_new


def show_anns(img,anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    label = np.zeros(((img.shape[0], img.shape[1], 3)))
    for ann in sorted_anns:
        if ann['area'] == sorted_anns[0]['area'] :
            continue
            
        m = ann['segmentation']
        # img = np.ones((m.shape[0], m.shape[1], 3))
        # color_mask = np.random.random((1, 3)).tolist()[0]
        # for i in range(3):
        #     label[:,:,i][m] = color_mask[i]
        # ax.imshow(np.dstack((img, m*0.35)))
        label[:,:,0][m] = np.random.randint(0, 255)
        label[:,:,1][m] = np.random.randint(0, 255)
        label[:,:,2][m] = np.random.randint(0, 255)
    final_img = (img+label*0.35)
    return final_img

def tif2png(path_Geoinfo,path_save,agg,label = False):
    # path_Geoinfo = 'data/hunan_farmland_A_whole_test/whole_test_final/region_image/3.tif'
    Geo_data = readTif(path_Geoinfo)
    img = Geo_data[-3]
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)
    if label:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.uint8)
        img = Norm_img(img)
    
    if agg:
        for i in range(img.shape[2]):
            img_temp = img[:,:,i]
            # img_temp_2 = cv2.equalizeHist(img_temp)
            clahe = cv2.createCLAHE(clipLimit=1,tileGridSize=(8, 8))
            img_temp_2 = clahe.apply(img_temp)
            img[:,:,i] = img_temp_2
    # path_save = 'data/hunan_farmland_A_whole_test/whole_test_final/region_image/3.png'
    if label:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path_save,img)
    print(img.shape)

    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

        
def whole(sam_checkpoint,image_path,model_type,device,out_path):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    masks = mask_generator.generate(image)
    print(len(masks))
    print(masks[0].keys())
    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    final_img = show_anns(image,masks)
    cv2.imwrite(out_path,final_img)
    # plt.axis('off')
    # plt.savefig('RS_sample\P7_compress_mask_0_demo.png') 

def point_model(sam_checkpoint,image,model_type,device,out_path,input_point,input_label,iter_time):
    

    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    #! one
    # for i in range(iter_time):
    #     clip_size = input_point.shape[0]/iter_time
    #     if i == 0:
    #         input_point_sub = input_point[:int((i+1)*clip_size),:]
    #         input_label_sub = input_label[:int((i+1)*clip_size)]
    #         masks, scores, logits = predictor.predict(
    #         point_coords=input_point_sub,
    #         point_labels=input_label_sub,
    #         multimask_output=False,
    #         )
    #     else:
    #         input_point_sub = input_point[int((i)*clip_size):int((i+1)*clip_size),:]
    #         input_label_sub = input_label[int((i)*clip_size):int((i+1)*clip_size)]
    #         mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    
    #         masks, scores, logits = predictor.predict(
    #         point_coords=input_point_sub,
    #         point_labels=input_label_sub,
    #         mask_input=mask_input[None, :, :],
    #         multimask_output=False,
    #         )
    #! Two
    # for i in range(input_point.shape[0]):
    #     input_point_sub = np.array([[input_point[i,0],input_point[i,1]]])
    #     input_label_sub = np.array([input_label[i]])
    #     if i == 0:
    #         masks, scores, logits = predictor.predict(
    #         point_coords=input_point_sub,
    #         point_labels=input_label_sub,
    #         multimask_output=False,
    #         )
    #     else:
    #         mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    
    #         masks, scores, logits = predictor.predict(
    #         point_coords=input_point_sub,
    #         point_labels=input_label_sub,
    #         mask_input=mask_input[None, :, :],
    #         multimask_output=False,
    #         )
    #! three
    # for i in range(iter_time):
    #     clip_size = input_point.shape[0]/iter_time
    #     if i == 0:
    #         input_point_sub = input_point
    #         input_label_sub = input_label
    #         masks, scores, logits = predictor.predict(
    #         point_coords=input_point_sub,
    #         point_labels=input_label_sub,
    #         multimask_output=False,
    #         )
    #     else:
    #         input_point_sub = input_point
    #         input_label_sub = input_label
    #         mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    
    #         masks, scores, logits = predictor.predict(
    #         point_coords=input_point_sub,
    #         point_labels=input_label_sub,
    #         mask_input=mask_input[None, :, :],
    #         multimask_output=False,
            # )
    # #! Four
    repeat = iter_time
    for i in range(input_point.shape[0]):
        input_point_sub = np.array([[input_point[i,0],input_point[i,1]]])
        input_label_sub = np.array([input_label[i]])
        if i == 0:
            input_point_in = input_point_sub.copy()
            input_label_in = input_label_sub.copy()

            masks, scores, logits = predictor.predict(
            point_coords=input_point_in,
            point_labels=input_label_in,
            multimask_output=False,
            )
            
            for i in range(repeat):
                mask_input = logits[np.argmax(scores), :, :]
                masks, scores, logits = predictor.predict(
                point_coords=input_point_in,
                point_labels=input_label_in,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
                )
                
        else:
            input_point_in = np.concatenate((input_point_in,input_point_sub))
            input_label_in = np.concatenate((input_label_in,input_label_sub))
            # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    
            for i in range(repeat+1):
                mask_input = logits[np.argmax(scores), :, :]
                masks, scores, logits = predictor.predict(
                point_coords=input_point_in,
                point_labels=input_label_in,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
                )

    return masks
    # best_score = np.where(scores == scores.max())
    # masks = masks[best_score[0][0],:,:]

def random_coor(num_points,label_index,seed):
    np.random.seed(seed)
    nonzero_pixels = np.where(label == label_index)
    random_indices = np.random.choice(len(nonzero_pixels[0]), size=num_points, replace=False)
    selected_points = np.column_stack((nonzero_pixels[1][random_indices], nonzero_pixels[0][random_indices]))
    return selected_points

def random_coor_rr(num_points,label_index):
    nonzero_pixels = np.where(label == label_index)
    random_indices = np.random.choice(len(nonzero_pixels[0]), size=num_points, replace=False)
    selected_points = np.column_stack((nonzero_pixels[1][random_indices], nonzero_pixels[0][random_indices]))
    return selected_points

def find_closest_factors(A):
    import math
    root = math.isqrt(A)  # 开方

    # 如果 A 本身能够拆分为两个相差不超过 5 的数，则直接拆分
    for i in range(root, 0, -1):
        if A % i == 0 and abs(i - A//i) <= 5:
            return i, A//i

    closest_product = A  # 最靠近 A 的乘积
    closest_factors = (root, root)  # 最靠近 A 的因子对

    # 在一定范围内对 N 和 M 进行加减操作，找到最靠近 A 的乘积和对应的因子对
    for i in range(1, 6):
        for j in range(1, 6):
            N = root + i
            M = root - j

            if abs(N - M) <= 5:
                product = N * M
                if abs(product - A) < abs(closest_product - A):
                    closest_product = product
                    closest_factors = (N, M)

    # 如果找到的最靠近 A 的乘积与 A 相等，则直接返回 A 的因子对
    if closest_product == A:
        return root, root

    return closest_factors

def get_all_point_union(num_points,image):
    # 获取掩码区域的索引
    N, M = find_closest_factors(num_points)
    
    height, width = image.shape[:2]
    region_height = height // N
    region_width = width // M

    centers = []
    for i in range(N):
        for j in range(M):
            center_x = (j * region_width) + (region_width // 2)
            center_y = (i * region_height) + (region_height // 2)
            centers.append((center_x, center_y))

    return np.array(centers)

def div_point_union(all_point,mask,label_index):
    count = 0
    for i in range(all_point.shape[0]):
        temp_point = all_point[i,:]
        if mask[temp_point[1],temp_point[0]] == label_index:
            if count != 0 :
                selected_point = np.concatenate((selected_point,temp_point))
            else:
                selected_point = temp_point
            count += 1

    if count == 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 提取外接多边形的中心点坐标
        center_points = []
        for contour in contours:
            # 计算外接矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 计算外接矩形的中心点坐标
            center_x = x + w // 2
            center_y = y + h // 2
            center_points.append((center_x, center_y))

        selected_point = np.array(center_points)
    else:
        selected_point = np.reshape(selected_point,(len(selected_point)//2,2))
    return selected_point


def split_array_with_skip(array, num_splits, skip):
    total_elements = array.shape[0]
    elements_per_split = total_elements // num_splits

    split_arrays = []
    for i in range(num_splits):
        if array.shape[0]%num_splits == 0:
            start_index = i
            end_index = start_index + elements_per_split * skip

            split_array = array[start_index:end_index:skip]

        else:
            start_index = i
            end_index = start_index + elements_per_split * skip

            split_array = array[start_index:end_index:skip]
            if i == num_splits-1:
                tile = array.shape[0]%num_splits
                tile_array = array[-tile:]
                split_array = np.concatenate((split_array,tile_array))
        split_arrays.append(split_array)
    return split_arrays

def split_array_with_order(array, num_splits, skip):
    total_elements = array.shape[0]
    elements_per_split = total_elements // num_splits

    split_arrays = []
    for i in range(num_splits):
        start_index = i*elements_per_split
        end_index = (i+1)*elements_per_split

        split_array = array[start_index:end_index]
        if i == num_splits-1:
            split_array = array[start_index:]
        split_arrays.append(split_array)

    return split_arrays

def find_segment_centers(image, mask, num_segments,label_index = 0):
    # 应用掩码
    import skimage.segmentation as seg
    from skimage.measure import regionprops
    masked_image = image.copy()
    masked_image[mask == label_index] = 0
    # cv2.imwrite('mask.png',masked_image)
    # 超像素分割
    # labels = seg.slic(masked_image, n_segments=num_segments)
    scale = 100  # 调整此参数以控制分割的粗细程度
    sigma = 0.5  # 调整此参数以控制分割的平滑度
    min_size = 300
    tt = 3
    # 进行分割
    segments = seg.felzenszwalb(masked_image, scale=scale, sigma=sigma, min_size=min_size)

    # 获取实际区域数目
    num_regions = len(np.unique(segments))
    min_size_min = 1
    min_size_max = 512*512
    min_size = min_size_max

    while min_size_min < min_size_max:
        min_size = (min_size_min + min_size_max) // 2

        segments = seg.felzenszwalb(image, scale=100, sigma=0.5, min_size=min_size)
        segments[mask == label_index] = 0
        segments_count = len(np.unique(segments))

        if segments_count == num_segments+1:
            break
        elif segments_count < num_segments+1:
            min_size_max = min_size
        else:
            min_size_min = min_size + 1
    # 调整分割结果
    # while num_regions != num_segments+1:
    #     if num_regions > num_segments:
    #         min_size *= 1.1
    #     else:
    #         min_size *= 0.9
    #     segments = seg.felzenszwalb(masked_image, scale=scale, sigma=sigma, min_size=int(min_size))
    #     segments[mask == label_index] = 0
    #     num_regions = len(np.unique(segments))
    centers = []
    for region in regionprops(segments):
        if region.label != 0:
            center = region.centroid
            centers.append([center[1],center[0]])

    # 返回符合条件的区域的质心坐标列表
    return np.array(centers).astype(int)

def iter_point(iter_time,
                num_points,
                selected_points_1,
                selected_points_0,
                image_path,
                sam_checkpoint,
                image_ori,
                model_type,
                device,
                ):
    
    for i in range(iter_time):
        if i == 0:
            input_point_ori = np.concatenate((selected_points_1,selected_points_0))
            input_label_ori = np.concatenate((np.ones(num_points),np.zeros(num_points))).astype(np.uint8)
            input_point = input_point_ori.copy()
            input_label = input_label_ori.copy()
        else:
            input_point = np.concatenate((input_point,input_point_ori))
            input_label = np.concatenate((input_label,input_label_ori))
    
    out_path = image_path.split('.')[0]+("_point_%s_iter_%s.png"%(num_points,iter_time))
    # input_point = np.array([[500, 375], [1125, 625], [1000, 600],[500,300],[1600,1000],[1600,200,],[50,400],[1000,1000]])
    # input_label = np.array([1, 1, 1, 1,0,0,0,0])
    masks = point_model(sam_checkpoint,image_ori,model_type,device,out_path,input_point,input_label,iter_time)

    masks = masks.transpose(1,2,0)
    masks = cv2.cvtColor(masks.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    opacity = 0.5
    final_img = image * (1 - opacity) + masks * opacity*255
    final_img = final_img.astype(np.uint8)
    for i in range(len(input_label)):
        point = input_point[i,:]
        if input_label[i] == 1:
            final_img = cv2.circle(final_img, (int(point[0]),int(point[1])), radius=23, color=(0,0,255), thickness=-1)
        else:
            final_img = cv2.circle(final_img, (int(point[0]),int(point[1])), radius=23, color=(255,0,0), thickness=-1)
    cv2.imwrite(out_path,final_img)

def remove_small_object(img_bin):
    from skimage.morphology import remove_small_holes
    from skimage.morphology import remove_small_objects
    img_temp = img_bin > 0
    mask_temp_img = remove_small_holes(img_temp,2000)
    cleaned = remove_small_objects(mask_temp_img,min_size = 2000)
    img_bin[cleaned == True] = 255
    img_bin[cleaned == False] = 0
    return img_bin

def dilate(size,img,iterations):
    k = np.ones((size, size), np.uint8)
    img = cv2.erode(img, k, iterations=iterations)
    # img = remove_small_object(img)
    return img

def erode(size,img,iterations):
    k = np.ones((size, size), np.uint8)
    img = cv2.dilate(img, k, iterations=iterations)
    return img



def postprocess(img_bin):
    size_e_d = 3
    ret,img_bin = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)
    img_bin = erode(size_e_d,img_bin,2)
    img_bin = dilate(size_e_d,img_bin,2)
    img_bin = remove_small_object(img_bin)
    return img_bin

import numpy as np
def find_most_similar(vec, vec_list):
    """
    找到一个向量vec在vec_list中最相似的向量并返回其索引。
    """
    similarities = [np.dot(vec, v) / (np.linalg.norm(vec) * np.linalg.norm(v)) for v in vec_list]
    # return np.argmax(similarities)
    return np.sum(similarities)

def exchange(A,B,image_ori):
    pixels_A = [image_ori[y, x] for x, y in A]
    pixels_A = np.array(pixels_A)
    pixels_B = [image_ori[y, x] for x, y in B]
    pixels_B = np.array(pixels_B)
    sim_A = np.zeros((pixels_A.shape[0]))
    sim_B = np.zeros((pixels_B.shape[0]))
    iter = 0
    while iter<3:
        for i in range(len(A)):
            sim_A[i] = find_most_similar(pixels_A[i], pixels_B)
            sim_B[i] = find_most_similar(pixels_B[i], pixels_A)
        a_index = np.argmax(sim_A)
        b_index = np.argmax(sim_B)
        A_ori,B_ori = A.copy(),B.copy()
        A[a_index],B[b_index] = B_ori[b_index],A_ori[a_index]
        pixels_A_ori,pixels_B_ori = pixels_A.copy(),pixels_B.copy()
        pixels_A[a_index], pixels_B[b_index] = pixels_B_ori[b_index], pixels_A_ori[a_index]
        iter += 1 
    return A,B 
def drop_out(A,B,image_ori):
    pixels_A = [image_ori[y, x] for x, y in A]
    pixels_A = np.array(pixels_A)
    pixels_B = [image_ori[y, x] for x, y in B]
    pixels_B = np.array(pixels_B)
    sim_A = np.zeros((pixels_A.shape[0]))
    sim_B = np.zeros((pixels_B.shape[0]))
    iter = 0
    while iter<2:
        sim_A = np.zeros((len(A)))
        sim_B = np.zeros((len(B)))
        for i in range(len(A)):
            sim_A[i] = find_most_similar(pixels_A[i], pixels_A)
            sim_B[i] = find_most_similar(pixels_B[i], pixels_B)
        a_index = np.argmin(sim_A)
        b_index = np.argmin(sim_B)
        A = np.delete(A,a_index,0)
        B = np.delete(B,b_index,0)
        iter += 1 
    return A,B 

def get_edge(mask):
    _, binary_image = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # path = 'RS_sample\hunan_level18\Predict\\0.png'
    edges = cv2.Canny(binary_image, 30, 150)
    # cv2.imwrite(path,edges)
    # edges = cv2.GaussianBlur(edges, (3, 3), 0)
    # cv2.imwrite(path,edges)
    edges = erode(3,edges,3)
    # cv2.imwrite(path,edges)
    # edges = cv2.GaussianBlur(edges, (3, 3), 0)
    # cv2.imwrite(path,edges)
    edges = dilate(2,edges,2)
    # cv2.imwrite(path,edges)
    # edges[edges!=0] = 255
    edges = cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    edges[:,:,0],edges[:,:,1],edges[:,:,2] = edges[:,:,0]*(170/255),edges[:,:,1]*(123/255),edges[:,:,2]*(29/255)
    # cv2.imwrite(path,edges)
    edges = cv2.blur(edges,(2,2)).astype(np.uint8)
    # cv2.imwrite(path,edges)
    return edges

def get_hint_image(final_img,ori_mask):
    final_img = image.copy()
    ori_mask = np.squeeze(ori_mask)
    for c in range(final_img.shape[2]):
        final_img[:, :, c][~ori_mask] //= 2
    final_img = final_img.astype(np.uint8)
    return final_img
def add_line(image_a,image_b):
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_a, 1, 255, cv2.THRESH_BINARY)
    masked_a = cv2.bitwise_and(image_a, image_a, mask=mask)
    mask_inv = cv2.bitwise_not(mask)
    masked_b = cv2.bitwise_and(image_b, image_b, mask=mask_inv) 
    # mask_edge = cv2.cvtColor(mask_edge.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    result = cv2.add(masked_a, masked_b)
    return result
def get_point_show(input_label,final_img,input_point):
    for i in range(len(input_label)):
        point = input_point[i,:]
        if input_label[i] == 1:
            final_img = cv2.circle(final_img, (int(point[0]),int(point[1])), radius=8, color=(211,219,72), thickness=-1)
        else:
            final_img = cv2.circle(final_img, (int(point[0]),int(point[1])), radius=8, color=(207,199,252), thickness=-1)
    return final_img

def iter_point_normal(iter_time,
                num_points,
                image_path,
                sam_checkpoint,
                image_ori,
                label,
                model_type,
                device,
                image_name,
                path_all,
                point_mode = 'super'
                ):
    # seeds_seed = 800
    # np.random.seed(seeds_seed)
    # seeds = np.random.choice(10000, size=iter_time, replace=False)
    if point_mode == 'random':
        for i in range(iter_time):
            selected_points_1 = random_coor_rr(num_points,label_index=1)
            selected_points_0 = random_coor_rr(num_points,label_index=0)
            # selected_points_1,selected_points_0 = exchange(selected_points_1,selected_points_0,image_ori) #交换策略无效？还是直接扔掉把
            # selected_points_1,selected_points_0 = drop_out(selected_points_1,selected_points_0,image_ori)
            input_point_sub = np.concatenate((selected_points_1,selected_points_0))
            input_label_sub = np.concatenate((np.ones(selected_points_1.shape[0]),np.zeros(selected_points_1.shape[0]))).astype(np.uint8)
            if i == 0:
                input_point = input_point_sub.copy()
                input_label = input_label_sub.copy()
            else:
                input_point = np.concatenate((input_point,input_point_sub))
                input_label = np.concatenate((input_label,input_label_sub))
    elif point_mode == 'union':
        area_size_1 = np.sum(label == 1)
        area_size_0 = np.sum(label == 0)
        area_all = label.shape[0]*label.shape[1]
        point_num_1 =  (num_points*iter_time*(area_all/area_size_1)).astype(np.uint16)
        point_num_0 =  (num_points*iter_time*(area_all/area_size_0)).astype(np.uint16)
        if point_num_1 >= 256:
            point_num_1 = 256
        if point_num_0 >= 256:
            point_num_0 = 256
        selected_points_all_1 = get_all_point_union(point_num_1,image_ori)
        selected_points_all_0 = get_all_point_union(point_num_0,image_ori)

        selected_points_1 = div_point_union(selected_points_all_1,label,label_index = 1)
        selected_points_0 = div_point_union(selected_points_all_0,label,label_index = 0)

        selected_points_1 = split_array_with_skip(selected_points_1, iter_time, iter_time)
        selected_points_0 = split_array_with_skip(selected_points_0, iter_time, iter_time)
        for i in range(iter_time): 
            input_point_sub = np.concatenate((selected_points_1[i],selected_points_0[i]))
            input_label_sub = np.concatenate((np.ones(selected_points_1[i].shape[0]),np.zeros(selected_points_0[i].shape[0]))).astype(np.uint8)
            if i == 0:
                input_point = input_point_sub.copy()
                input_label = input_label_sub.copy()
            else:
                input_point = np.concatenate((input_point,input_point_sub))
                input_label = np.concatenate((input_label,input_label_sub))
        T_S = get_point_show(input_label,image_ori,input_point)
        cv2.imwrite('419_point.png',T_S)

    elif point_mode == 'super':
        # out_path_point = os.path.join(path_all,'Predict_vis_point',point_mode)
        # out_path_point = os.path.join(out_path_point,image_name)
        all_point_num = (num_points*iter_time)
        selected_points_all_1 = find_segment_centers(image_ori, label, all_point_num,label_index = 1)
        selected_points_all_0 = find_segment_centers(image_ori, label, all_point_num,label_index = 0)
        selected_points_1 = split_array_with_skip(selected_points_all_1, iter_time, iter_time)
        selected_points_0 = split_array_with_skip(selected_points_all_0, iter_time, iter_time)
        for i in range(iter_time): 
            input_point_sub = np.concatenate((selected_points_0[i],selected_points_1[i]))
            input_label_sub = np.concatenate((np.ones(selected_points_1[i].shape[0]),np.zeros(selected_points_0[i].shape[0]))).astype(np.uint8)
            if i == 0:
                input_point = input_point_sub.copy()
                input_label = input_label_sub.copy()
            else:
                input_point = np.concatenate((input_point,input_point_sub))
                input_label = np.concatenate((input_label,input_label_sub))
            


    # out_path = image_path.split('.')[0]+("_point_%s_iter_%s.png"%(num_points,iter_time))
    out_path = os.path.join(path_all,'Predict',point_mode)
    out_path_ori = os.path.join(path_all,'Predict_ori',point_mode,str(num_points))
    out_path_point = os.path.join(path_all,'Predict_vis_point',point_mode,str(num_points))

    if not os.path.exists(out_path_point):
        # 创建目录
        os.makedirs(out_path_point)
    else:
        pass
    out_path_point = os.path.join(out_path_point,image_name)
    
    if not os.path.exists(out_path_ori):
        # 创建目录
        os.makedirs(out_path_ori)
    else:
        pass
    
    out_path_ori = os.path.join(out_path_ori,image_name)
    # T_S = get_point_show(input_label,image_ori,input_point)
    # cv2.imwrite(out_path_point,T_S)
    # input_point = np.array([[500, 375], [1125, 625], [1000, 600],[500,300],[1600,1000],[1600,200,],[50,400],[1000,1000]])
    # input_label = np.array([1, 1, 1, 1,0,0,0,0])
    masks = point_model(sam_checkpoint,image_ori,model_type,device,out_path,input_point,input_label,iter_time)
    ori_mask_h = masks.copy()
    masks = masks.transpose(1,2,0)

    
    masks = masks.astype(np.uint8)*255
    masks = postprocess(masks[:,:,0])
    # masks = 
    ori_mask = masks.copy()
    ori_mask[masks == 255] = 1

    # out_path_ori = 'RS_sample\\419_pre.png'
    cv2.imwrite(out_path_ori,masks)

    mask_edge = get_edge(masks)
    # cv2.imwrite(out_path,mask_edge)
    final_img = get_hint_image(image_ori,ori_mask_h)
    # cv2.imwrite(out_path,final_img)
    final_img = add_line(mask_edge,final_img)
    # cv2.imwrite(out_path,final_img)
    final_img = get_point_show(input_label,final_img,input_point)
    # out_path_ori = 'RS_sample\\419_vis.png'
    cv2.imwrite(out_path_point,final_img)
    # out_path = 'RS_sample\\419_p.png'
    # for i in range(len(input_label)):
    #     point = input_point[i,:]
    #     if input_label[i] == 1:
    #         final_img = cv2.circle(image_ori, (int(point[0]),int(point[1])), radius=8, color=(211,219,72), thickness=-1)
    #     else:
    #         final_img = cv2.circle(image_ori, (int(point[0]),int(point[1])), radius=8, color=(207,199,252), thickness=-1)
    # cv2.imwrite(out_path,final_img)
    return ori_mask.astype(np.uint8)



def add_label_show(img,label_show,out_path):
    # label_show[label_show == 0] = 255
    # label_show[label_show != 255] = 0
    label_show = Norm_img(label_show)
    # cv2.imwrite(out_path,label_show)
    if label_show.shape[-1] == 3:
        pass
    else:
        label_show = cv2.cvtColor(label_show.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    opacity = 0.5
    final_img = img * (1 - opacity) + label_show * opacity
    cv2.imwrite(out_path,final_img)

def m2m_label_show(img,label,out_path):
    # label[label==2 ] = 1
    # label[label!=1] = 0
    label = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_GRAY2RGB)*255
    opacity = 0.5
    final_img = img * (1 - opacity) + label * opacity
    cv2.imwrite(out_path,final_img)

def get_metrics(path_all,iou_meter,mode,point_mode,count_improve = 0,sample_num = 0):
    miou, acc ,precision,recall,f1score,conf_matrix = iou_meter.get_miou_acc()
    # mode = 'SAM'
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_IoU".format(mode): miou,
        "{}_precision".format(mode):str(precision),
        "{}_recall".format(mode):str(recall),
        "{}_f1score".format(mode):str(f1score),
        "{}_conf_matrix".format(mode):str(conf_matrix),
        "{}_better_than_label".format(mode):str(count_improve),
        "{}_sample_number".format(mode):str(sample_num)

    }
    write_dir = os.path.join(path_all, 'result',point_mode)
    write_path = os.path.join(write_dir,'metrix_%s'%(mode))
    if  not os.path.exists(write_dir):
        os.makedirs(write_dir)
    with open(write_path, "w") as outfile:
        json.dump(metrics, outfile, indent=4)

def write_to_excel(A, B, C,path_save):
    # 将参数组成数组
    import pandas as pd
    data = [A, B, C]
    if not os.path.exists(path_save):
        # 创建目录
        os.makedirs(path_save)
    else:
        pass
    path_save = os.path.join(path_save,'single.xlsx')
    # 读取现有的Excel文件（如果存在），或创建一个新的DataFrame对象
    try:
        df = pd.read_excel(path_save)
    except FileNotFoundError:
        df = pd.DataFrame()

    # 将数据添加到DataFrame中
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    # 将DataFrame写入Excel文件
    df.to_excel(path_save, index=False)


def print_g_b(Am_record,num,point_mode):
    import heapq
    Miou_all = list(Am_record[1:,1])
    Am_record = Am_record[1:,:]
    re1 = map(Miou_all.index, heapq.nlargest(num, Miou_all))
    re2 = map(Miou_all.index, heapq.nsmallest(num, Miou_all))
    good_out = 'samples:%s,get_%s_best_sample \n '%(Am_record.shape[0],num)
    for name_index in list(re1):
        name_s = Am_record[name_index,2:]
        good_out_temp ='good sample name %s'%(name_s)+' \n '+('Acc : {:.1f},Iou: {:.1f} '.format(Am_record[name_index,0],Am_record[name_index,1]))+' \n '
        good_out = good_out+good_out_temp

    bad_out = 'samples:%s,get_%s_worst_sample'%(Am_record.shape[0],num)
    for name_index in list(re2):
        name_s = Am_record[name_index,2:]
        bad_out_temp = 'bad sample name %s'%(name_s)+'\n'+'Acc : {:.1f},Iou: {:.1f} '.format(Am_record[name_index,0],Am_record[name_index,1])+' \n '
        bad_out = bad_out+bad_out_temp
    write_dir = os.path.join(path_all, 'result',point_mode)
    write_path_good = os.path.join(write_dir,'good_sample.txt')
    write_path_bad = os.path.join(write_dir,'bad_sample.txt')
    file_good = open(write_path_good,'w')
    file_good.write(good_out)
    file_good = open(write_path_bad,'w')
    file_good.write(bad_out)

#结果分析模块，用于查看不同提示精度下的优化情况，目前为全部评估
def Ana_final_result(path_all,point_mode,path_label,path_gt,path_predict,num_points):

    for i in range(11):    
        iou_meter_pred = IoU(
        num_classes=2,
        ignore_index=None,
        cm_device=device,
        )
        iou_meter_label = IoU(
        num_classes=2,
        ignore_index=None,
        cm_device=device,
        )
        range_label = i 
        count_improve = 0
        sample_num = 0
        with alive_bar(len(os.listdir(path_predict))) as bar:
            for image_name in sorted(os.listdir(path_predict)):
                i = 10
                bar()
                iou_meter_pred_single = IoU(
                num_classes=2,
                ignore_index=None,
                cm_device=device,
                )
                iou_meter_label_single = IoU(
                num_classes=2,
                ignore_index=None,
                cm_device=device,
                )
                label_path = os.path.join(path_label,image_name)
                label_gt_path = os.path.join(path_gt,image_name)
                label_predict_path = os.path.join(path_predict,image_name)

                label = cv2.imread(label_path)[:,:,0]
                label = torch.from_numpy(label[np.newaxis,:])

                label_gt = cv2.imread(label_gt_path)[:,:,0]
                label_gt = torch.from_numpy(label_gt[np.newaxis,:])

                label_predict = cv2.imread(label_predict_path)[:,:,0]
                label_predict[label_predict == 255] = 1
                label_predict = torch.from_numpy(label_predict[np.newaxis,:])
                # single acc
                iou_meter_pred_single.add(label_predict.cuda(device), label_gt.cuda(device))
                iou_meter_label_single.add(label.cuda(device), label_gt.cuda(device))
                miou_single_p, acc_single_p ,_,_,_,_ = iou_meter_pred_single.get_miou_acc()
                miou_single_l, acc_single_l ,_,_,_,_ = iou_meter_label_single.get_miou_acc()
                # print(miou_single_p)
                # print(miou_single_l)
                # if miou_single_l - miou_single_p >20:
                #     os.remove(label_path)
                #     os.remove(label_gt_path)
                #     os.remove(label_predict_path)
                #     print(image_name)
                #     continue
                    
                if int(miou_single_l/10) == int(range_label):
                    # continue
                    # all acc
                    iou_meter_pred.add(label_predict.cuda(device), label_gt.cuda(device))
                    iou_meter_label.add(label.cuda(device), label_gt.cuda(device))
                    sample_num += 1
                    if miou_single_p > miou_single_l:
                        count_improve += 1
                if i == 10:
                    iou_meter_pred.add(label_predict.cuda(device), label_gt.cuda(device))
                    iou_meter_label.add(label.cuda(device), label_gt.cuda(device))
                    sample_num += 1
                    # if sample_num == 200:
                    #     break
            if sample_num == 0:
                pass
            else:
                metrics = get_metrics(
                    path_all,
                    iou_meter_pred,
                    mode = 'pred_range_%s_%s'%(range_label/10,num_points),
                    point_mode = point_mode,
                    count_improve = count_improve,
                    sample_num = sample_num )
                metrics = get_metrics(
                    path_all,
                    iou_meter_label,
                    mode = 'label_range_%s_%s'%(range_label/10,num_points),
                    point_mode = point_mode,
                    count_improve = count_improve,
                    sample_num = sample_num )
            break
            
def divdie_the_dataset(path_all,point_mode,path_label,path_gt,path_predict):
    import shutil
    with alive_bar(len(os.listdir(path_predict))) as bar:
        file_count = 0
        for image_name in sorted(os.listdir(path_predict)):
            bar()
            iou_meter_pred_single = IoU(
            num_classes=2,
            ignore_index=None,
            cm_device=device,
            )
            iou_meter_label_single = IoU(
            num_classes=2,
            ignore_index=None,
            cm_device=device,
            )
            label_path = os.path.join(path_label,image_name)
            label_gt_path = os.path.join(path_gt,image_name)
            label_predict_path = os.path.join(path_predict,image_name)

            label = cv2.imread(label_path)[:,:,0]
            label = torch.from_numpy(label[np.newaxis,:])

            label_gt = cv2.imread(label_gt_path)[:,:,0]
            label_gt = torch.from_numpy(label_gt[np.newaxis,:])

            label_predict = cv2.imread(label_predict_path)[:,:,0]
            label_predict[label_predict == 255] = 1
            label_predict = torch.from_numpy(label_predict[np.newaxis,:])
            # single acc
            iou_meter_pred_single.add(label_predict.cuda(device), label_gt.cuda(device))
            iou_meter_label_single.add(label.cuda(device), label_gt.cuda(device))
            miou_single_p, acc_single_p ,_,_,_,_ = iou_meter_pred_single.get_miou_acc()
            miou_single_l, acc_single_l ,_,_,_,_ = iou_meter_label_single.get_miou_acc()
            if miou_single_l < 30:
                if miou_single_l > 10:
                    new_filename = image_name
                    source_folder = os.path.join(path_all,'Predict_vis_point',point_mode)
                    source_file = os.path.join(source_folder,new_filename)
                    destination_folder = os.path.join(path_all,'divid','l_30','Predict_vis_point',point_mode)
                    destination_file = os.path.join(destination_folder, new_filename)
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)
                    shutil.copy2(source_file, destination_file)

                    source_folder = os.path.join(path_all,'label_gt_vis')
                    source_file = os.path.join(source_folder,new_filename)
                    destination_folder = os.path.join(path_all,'divid','l_30','label_gt_vis')
                    destination_file = os.path.join(destination_folder, new_filename)
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)
                    shutil.copy2(source_file, destination_file)

                    source_folder = os.path.join(path_all,'label_vis')
                    source_file = os.path.join(source_folder,new_filename)
                    destination_folder = os.path.join(path_all,'divid','l_30','label_vis')
                    destination_file = os.path.join(destination_folder, new_filename)
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)
                    shutil.copy2(source_file, destination_file)
                    file_count += 1
        # print(miou_single_p)
        # print(miou_single_l)


if __name__ == '__main__':

    sam_checkpoint = "checkpoint\sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    path_image = 'RS_sample\hunan_level18\All_data\image'
    path_all = 'RS_sample\hunan_level18\All_data' 
    point_mode_all = [ 'union']#random # super # union
    label_show = True
    iter_time = 3
    num_points = 11
    Am_record = np.zeros((1,3))
    for point_mode in point_mode_all:
        
        with alive_bar(len(os.listdir(path_image))) as bar:
            
            iou_meter_pred = IoU(
            num_classes=2,
            ignore_index=None,
            cm_device=device,
            )
            iou_meter_label = IoU(
            num_classes=2,
            ignore_index=None,
            cm_device=device
            )
            path_show_gt = os.path.join(path_all,'label_gt_vis')
            path_show_label = os.path.join(path_all,'label_vis')
            if  not os.path.exists(path_show_gt):
                os.makedirs(path_show_gt)

            if  not os.path.exists(path_show_label):
                os.makedirs(path_show_label)

            for image_name in sorted(os.listdir(path_image)):
                # break
                image_name = '419.png'
                bar()
                iou_meter_pred_single = IoU(
                num_classes=2,
                ignore_index=None,
                cm_device=device,
                )
                iou_meter_label_single  = IoU(
                num_classes=2,
                ignore_index=None,
                cm_device=device,
                )


                image_path = os.path.join(path_image,image_name)
                label_path = os.path.join(path_all,'label',image_name)
                label_gt_path = os.path.join(path_all,'label_gt',image_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_ori = image
                label = cv2.imread(label_path)[:,:,0]
                if label.max() == label.min():
                    
                    continue
                label_gt = cv2.imread(label_gt_path)


                # print(set(list(label.flat)))
                # print(set(list(label_gt.flat)))
                if label_show:

                    label_show_image = label.copy()
                    label_gt_show_image = label_gt.copy()
                    add_show_path = os.path.join(path_show_label,image_name)
                    gt_show_path = os.path.join(path_show_gt,image_name)

                    add_label_show(image,label_show_image,add_show_path)
                    add_label_show(image,label_gt_show_image,gt_show_path)
                    # print(set(list(label.flat)))
                temp_done  = os.path.join(path_all,'predict_ori',point_mode,str(num_points)) 
                # if os.path.exists(temp_done):                   
                #     already_done = os.listdir(temp_done)
                #     if image_name in already_done:
                #         continue

                pred = iter_point_normal(iter_time,
                        num_points,
                        image_path,
                        sam_checkpoint,
                        image_ori,
                        label,
                        model_type,
                        device,
                        image_name,
                        path_all,
                        point_mode
                )
                label_gt = label_gt[:,:,0]
                label_gt = label_gt[np.newaxis,:]
                # inverted_label_gt = np.logical_not(label_gt).astype(np.uint8)
                # label_gt = np.concatenate((label_gt,inverted_label_gt),axis = 0)
                label_gt = torch.from_numpy(label_gt)
                label = label[np.newaxis,:]
                label = torch.from_numpy(label)

                # inverted_pred = np.logical_not(pred).astype(np.uint8)
                # pred = np.concatenate((pred,inverted_pred),axis = 0)
                pred = pred[np.newaxis,:]
                pred = torch.from_numpy(pred)

                iou_meter_pred.add(pred.cuda(device), label_gt.cuda(device))
                iou_meter_label.add(label.cuda(device), label_gt.cuda(device))

                iou_meter_pred_single.add(pred.cuda(device), label_gt.cuda(device))
                iou_meter_label_single.add(label.cuda(device), label_gt.cuda(device))
                miou_single, acc_single ,_,_,_,_ = iou_meter_pred_single.get_miou_acc()
                miou_single_l, acc_single_l ,_,_,_,_ = iou_meter_label_single.get_miou_acc()

                # Analysis part
                Am_record_temp = np.zeros((1,3))
                Am_record_temp[0,0],Am_record_temp[0,1],Am_record_temp[0,2] = acc_single,miou_single,image_name.split('.')[0]
                print('image name %s , pred miou %s,label miou %s'%(image_name,miou_single,miou_single_l))
                
                path_save_single = os.path.join(path_all,'result',point_mode)
                # write_to_excel(image_name.split('.')[0], miou_single,miou_single_l,path_save_single)
                Am_record =  np.concatenate((Am_record,Am_record_temp),axis = 0)
                
                # print(conf_matrix)
        metrics = get_metrics(path_all,iou_meter_pred,mode = 'pred',point_mode = point_mode)
        metrics = get_metrics(path_all,iou_meter_label,mode = 'label',point_mode = point_mode)
        print_g_b(Am_record,num = 10,point_mode = point_mode)
        path_label = os.path.join(path_all,'label')
        path_gt = os.path.join(path_all,'label_gt')
        path_predict = os.path.join(path_all,'Predict_ori',point_mode,str(num_points))
        # divdie_the_dataset(path_all,point_mode,path_label,path_gt,path_predict)
        Ana_final_result(path_all,point_mode,path_label,path_gt,path_predict,num_points)
        # Ana_final_result(path_all,point_mode,path_label,path_gt,path_predict)



    # image_path = 'RS_sample\ArcGIS_Img\SA_3_Level_18_agg_2_1.png'
    # label_path = 'RS_sample\CropAdd\Crop_Pre_2_1.png'


    # if image_path.split('.')[-1] == 'tif':
    #     agg = True
    #     if agg:
    #         path_save = image_path.split('.')[0]+'_agg.png'
    #     else:
    #         path_save = image_path.split('.')[0]+'.png'
    #     tif2png(image_path,path_save,agg)
    #     image_path = path_save
    # if label_path.split('.')[-1] == 'tif':
    #     agg = False
    #     if agg:
    #         path_save = label_path.split('.')[0]+'_agg.png'
    #     else:
    #         path_save = label_path.split('.')[0]+'.png'
    #     tif2png(label_path,path_save,agg,label=True)
    #     label_path = path_save
    # # whole(sam_checkpoint,image_path,model_type,device,out_path)



    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_ori = image
    # label = cv2.imread(label_path)[:,:,0]
    # print(set(list(label.flat)))

    # # 10m label
    # print(set(list(label.flat)))
    # label = cv2.resize(label,(int(image.shape[1]),int(image.shape[0]))).astype(np.uint8)
    # print(set(list(label.flat)))
    # add_show_path = label_path.split('.')[0]+"_show.png"
    
    # print(set(list(label.flat)))
    # label_show = label.copy()
    # #! 2m label
    # # label[label==2 ] = 1
    # # label[label!=1] = 0
    # # print(set(list(label.flat)))
    # # label = cv2.resize(label,(int(image.shape[1]),int(image.shape[0]))).astype(np.uint8)
    # # label[label>0] = 1
    # # label_show = label.copy()
    # # m2m_label_show(image,label_show,add_show_path)
    # #! add mode
    # # label[label < 3] = 0
    # # label[label != 0] = 1
    # # label[label == 1] = 2
    # # label[label == 0] = 1
    # # label[label == 2] = 0
    # # add_label_show(image,label_show,add_show_path)
    # #! pre mode(For vis)
    # label_show[label_show == 1] = 2
    # label_show[label_show == 0] = 1
    # label_show[label_show == 2] = 0
    # # label = label_show.copy()
    # add_label_show(image,label_show,add_show_path)
    # print(set(list(label.flat)))



    # # # 点不变，直接重复
    # # iter_time = 1
    # # num_points = 10
    # # selected_points_1 = random_coor(num_points,label_index=1)
    # # selected_points_0 = random_coor(num_points,label_index=0)
    # # for iter_time in range(1,10):
    # #     iter_point(iter_time,
    # #             num_points,
    # #             selected_points_1,
    # #             selected_points_0,
    # #             image_path,
    # #             sam_checkpoint,
    # #             image_ori,
    # #             model_type,
    # #             device,
    # #             )
    # # 每次输入num_points个正点num_points个负点迭代，总点数为iter_time*num_points
    # iter_time = 3
    # num_points = 10
    # iter_point_normal(iter_time,
    #         num_points,
    #         image_path,
    #         sam_checkpoint,
    #         image_ori,
    #         model_type,
    #         device,
    #         )
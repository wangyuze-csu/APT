import numpy as np
import cv2
import numpy as np

def crop_images(big_image, step=256, size=512):
    height, width, channels = big_image.shape
    cropped_images = []
    
    for y in range(0, height-size+1, step):
        for x in range(0, width-size+1, step):
            cropped_images.append(big_image[y:y+size, x:x+size, :])
    
    return cropped_images

def reconstruct_images(cropped_images, big_image_shape, step=256):
    height, width, channels = big_image_shape
    num_images = len(cropped_images)
    size = cropped_images[0].shape[0]
    reconstructed_image = np.zeros((height, width, channels))
    count = np.zeros((height, width))
    
    idx = 0
    for y in range(0, height-size+1, step):
        for x in range(0, width-size+1, step):
            reconstructed_image[y:y+size, x:x+size, :] += cropped_images[idx]
            count[y:y+size, x:x+size] += 1
            idx += 1
    cv2.imwrite('tools\crop_test\%s.png'%("big_ori"),reconstructed_image)
    cv2.imwrite('tools\crop_test\%s.png'%("count_ori"),count[:, :, np.newaxis]*(255/4))
    reconstructed_image /= count[:, :, np.newaxis]
    return reconstructed_image

# 示例使用:
# 假设big_image是一个512x512大小的图像
big_image = cv2.imread('tools\\test1.png')
# 将big_image裁剪成512x512的小图像列表
cropped_images = crop_images(big_image)
count = 0
for sub_image in cropped_images:
    cv2.imwrite('tools\crop_test\%s.png'%(count),sub_image)
    count += 1
# 将裁剪后的小图像列表还原为原始大图像
reconstructed_image = reconstruct_images(cropped_images, big_image.shape)
cv2.imwrite('tools\crop_test\%s.png'%("big"),reconstructed_image)
# 验证是否还原成功（结果应该很接近big_image）
print(np.allclose(big_image, reconstructed_image))
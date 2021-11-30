import numpy as np
import cv2
import torch
from config_logging import LogConfig
import math
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (check_img_size, non_max_suppression)
from utils.torch_utils import time_sync
from PIL import Image, ImageDraw, ImageFont
from kalmanfilter import KalmanFilter

def load_camera_calibration():
    try:
        r = np.load('parameters/calibration.npz')
        mtx = r['arr_0']
        dist = r['arr_1']
    except IOError:
        logging.error('cant find camera calibration data')
        raise Exception()
    return mtx, dist

def load_camera_post_adjust():
    try:
        r = np.load('parameters/post_adjust.npz')
        rotate_matrix = r['arr_0']
        tvec = r['arr_1']
    except IOError:
        logging.error('cant find camera calibration data')
        raise Exception()
    return rotate_matrix, tvec

def load_model(weights_path='models/best.pt'):
    return DetectMultiBackend(weights_path, device='cpu', dnn=False)

def calculate_mapping_coordinate(mtx, dist, rotate_matrix, tvec, bounding_box):
    '''

    '''
    x1, y1, x2, y2, confidence = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], bounding_box[4]
    # 没有检测到目标对象
    if confidence == 0:
        return 0, 0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    target_coordinate = np.array([cx, cy])

    # 畸变校正,转换到相机坐标系,得到(u,v,1)
    target_center_ideal = cv2.undistortPoints(target_coordinate.reshape([1, -1, 2]), mtx, dist)
    target_camera_coodinate = np.append(target_center_ideal[0][0], [1])

    # target的坐标从相机坐标转换到世界坐标
    target_world_coodinate = np.linalg.inv(rotate_matrix).dot((target_camera_coodinate - tvec.reshape(3)))

    # 将相机的坐标原点转换到世界坐标系上
    target_origin_world_coodinate = np.linalg.inv(rotate_matrix).dot((np.array([0, 0, 0.0]) - tvec.reshape(3)))

    # 两点确定直线(x-x0)/(x0-x1) = (y-y0)/(y0-y1) = (z-z0)/(z0-z1)
    # 当z=0时,得到x,y
    delta = target_origin_world_coodinate - target_world_coodinate

    z_world = 0.0
    x_world = (z_world - target_origin_world_coodinate[2]) / delta[2] * delta[0] + target_origin_world_coodinate[0]
    y_world = (z_world - target_origin_world_coodinate[2]) / delta[2] * delta[1] + target_origin_world_coodinate[1]

    return (x_world, y_world)

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def prediction_bounding_box(img, model, img_size=640):
    # 加载模型
    stride, pt, jit = model.stride, model.pt, model.jit
    img_size = check_img_size(img_size, s=stride)  # check image size

    model.model.float()
    # Dataloader
    dataset = LoadImages(img, img_size=img_size, stride=stride, auto=pt and not jit)

    # Run inference
    dt, seen = [0.0, 0.0, 0.0], 0
    for im, im0s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS 非极大值抑制(引入general包)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):
            seen += 1
            im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # x1, y1, x2, y2, confidence
        results = pred[0].numpy().ravel()
        if np.size(results) == 0:
            x1, y1, x2, y2, confidence = 0,0,0,0,0
        else:
            x1, y1, x2, y2, confidence = results[0], results[1], results[2], results[3], results[4]
        return np.array([x1, y1, x2, y2, confidence])

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def draw_bounding_box(image, bounding_box):
    x1, y1, x2, y2, confidence = bounding_box[:5]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
    cv2.circle(image, (cx, cy), 1, (0, 0, 255), 3)
    return image

def calculate_rotate_angle(image, bounding_box, mapping_point):
    mapping_x, mapping_y = mapping_point[0], mapping_point[1]
    if mapping_x < 0 or mapping_y < 0 or mapping_x > width or mapping_y > high:
        return 0.0
    x1, y1, x2, y2 = bounding_box[:4]
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return 0.0
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped_image = image[y1:y2, x1:x2]
    w = (y2 - y1) // 4
    h = (x2 - x1) // 4
    cropped_image2 = image[y1+h:y2-h, x1+w:x2-w]
    gray_image = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2RGBA)
    blur_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    edges = cv2.Canny(blur_image, 10, 100, apertureSize=3)
    lines = cv2.HoughLines(edges,1,np.pi/180, 30)
    if lines is not None:
        line = get_max_houghline(lines)

        rho, theta = line[:]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = x0 + 1000 * (-b)
        y1 = y0 + 1000 * (a)
        x2 = x0 - 1000 * (-b)
        y2 = y0 - 1000 * (a)
        angle = cv2.fastAtan2(float((y2 - y1)), float((x2 - x1)))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.line(cropped_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        if angle is not None:
            return angle
    return 0.0

def get_max_houghline(lines):
    lines1 = lines[:, 0, :]
    index = 0
    max_dist = 0
    max_index = 0
    for rho, theta in lines1[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        dist = (x2 - x1)^2 + (y2 - y1)^2
        if dist > max_dist:
            max_dist = dist
            max_index = index
        index += 1
    return lines1[max_index]

def calculate_additional_angle(line_angles, cur_angle):
    global rotate_flag
    if cur_angle == 0:
        return 0.0
    line_angles.append(cur_angle)

    if len(line_angles) == 1:
        return 0.0
    cur_angle = float(line_angles[-1])
    pre_angle = float(line_angles[-2])
    if len(line_angles) == 2:
        if abs(pre_angle - cur_angle) < 40 or (360-pre_angle) + cur_angle < 40:
            line_angles.remove(line_angles[-1])
            return 0.0
        if cur_angle - pre_angle > 0:
            rotate_flag = False
    if abs(pre_angle - cur_angle) < 40 or (360-pre_angle) + cur_angle < 40:
        line_angles.remove(line_angles[-1])
        return 0.0

    if rotate_flag:
        if pre_angle < cur_angle:
            return ((360.0 - cur_angle) + pre_angle) / 2
        return (pre_angle - cur_angle) / 2
    else:
        if pre_angle > cur_angle:
            return ((360.0 - pre_angle) + cur_angle) / 2
        return (cur_angle - pre_angle) / 2

def calculate_additional_dist(mapping_points, cur_point):
    cur_x, cur_y = float(cur_point[0]), float(cur_point[1])
    if cur_x < 0 or cur_y < 0 or cur_x > width or cur_y > high:
        return 0
    mapping_points.append(cur_point)
    if len(mapping_points) == 1:
        # 不要忘了要调整
        return 0
    pre_point = mapping_points[-2]
    pre_x, pre_y = float(pre_point[0]), float(pre_point[1])
    dist = math.sqrt((pre_x - cur_x)*(pre_x - cur_x) + (pre_y - cur_y)*(pre_y - cur_y))
    if dist > 50 or dist < 1:
        return 0
    return dist

def get_active_and_stop_indexframe(active_frame_index, stop_frame_index, cur_frame_index, cur_point, points):
    cur_x, cur_y = float(cur_point[0]), float(cur_point[1])
    if cur_x > 0 and cur_y > 0 and cur_x < width and cur_y < high and active_frame_index == 0:
        active_frame_index = cur_frame_index
    if cur_x > 0 and cur_y > 0 and cur_x < width and cur_y < high and active_frame_index != 0:
        if len(points) >= 3:
            pre_point = points[-3]
            pre_x, pre_y = float(pre_point[0]), float(pre_point[1])
            dist = math.sqrt((pre_x - cur_x) * (pre_x - cur_x) + (pre_y - cur_y) * (pre_y - cur_y))
            if dist < 2:
                return active_frame_index, stop_frame_index
        stop_frame_index = cur_frame_index
    return active_frame_index, stop_frame_index

def calculate_active_time(active_frame_index, stop_frame_index, frame_rate, cur_point):
    if active_frame_index != 0:
        frame_num = stop_frame_index - active_frame_index
        return frame_num/frame_rate
    return 0.0

def calculate_average_velocity(sum_dist, active_time):
    if sum_dist == 0 or active_time == 0:
        return 0.0
    return sum_dist / active_time / 100

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def draw_info(frame, mapping_point, sum_dist, active_time, velocity, sum_angle):
    mapping_x, mapping_y = mapping_point[0], mapping_point[1]
    x = int(mapping_x)
    y = int(mapping_y)
    frame = cv2ImgAddText(frame, '当前坐标：{} {}'.format(x, y), 30, 30, textColor=(255, 0, 0), textSize=25)
    frame = cv2ImgAddText(frame, '运动距离：{} cm'.format(int(sum_dist)), 30, 60, textColor=(255, 0, 0), textSize=25)
    frame = cv2ImgAddText(frame, '运动时间：{:.2f} s'.format(active_time), 30, 90, textColor=(255, 0, 0), textSize=25)
    frame = cv2ImgAddText(frame, '平均速度：{:.2f} m/s'.format(velocity), 30, 120, textColor=(255, 0, 0), textSize=25)
    frame = cv2ImgAddText(frame, '旋转角度：{}°'.format(int(sum_angle)), 30, 150, textColor=(255, 0, 0), textSize=25)
    return frame

def draw_trajectory(mapping_background_image, cur_point, ratio):
    x, y = int(cur_point[0]), int(cur_point[1])
    y = high - y
    if x < 0 or x > width or y < 0 or y > high:
        return mapping_background_image
    x = int(x) * ratio + 1
    y = int(y) * ratio + 1 + 5 * ratio
    cv2.circle(mapping_background_image, (x , y), 2, (0, 0, 255), 3)
    return mapping_background_image

def draw_prediction_trajectory(image, cur_point, ratio, left_deviation, top_deviation):
    x, y = int(cur_point[0]), int(cur_point[1])
    y = high - y
    if x < 0 or x > width or y < 0 or y > high:
        return image
    x = int(x)
    y = int(y)
    for i in range(1):
        predicted = kf.predict(x, y)
        pre_x, pre_y = predicted[0], predicted[1]
        if pre_x < 0 or pre_x > width or pre_y < 0 or pre_y > high:
            return image
        x, y = pre_x, pre_y
        pre_x = int(pre_x) * ratio + 1
        pre_y = int(pre_y) * ratio + 1 + 5 * ratio
        cv2.circle(image, (pre_x+left_deviation, pre_y+top_deviation), 2, (255, 0, 0), 3)
    return image

# 日志消息
logging = LogConfig()

if __name__ == '__main__':
    # 输入图像
    input = 'data/test9.mp4'
    cap = cv2.VideoCapture(input)
    output = 'result/output.avi'
    model_path = 'models/best.pt'
    background_path = 'data/background.jpg'

    background_image = cv2.imread(background_path)
    background_h, background_w = background_image.shape[:2]
    resize_background_image = cv2.resize(background_image, (917, 332))
    resize_h, resize_w = resize_background_image.shape[:2]
    high = 100
    width = 300

    # load 卡尔曼滤波
    kf = KalmanFilter()

    # 1.加载相机内参和形变畸数
    mtx, dist = load_camera_calibration()

    # 2.加载相机旋转矩阵和偏移矩阵
    rotate_matrix, tvec = load_camera_post_adjust()

    # 3.加载模型
    model = load_model(model_path)



    if cap.isOpened():
        # 显示比例
        proportion = 0.6

        frame_rate = cap.get(5)  # 获取帧率
        frame_num = cap.get(7)

        # 获取视频宽度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取视频高度
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置写入视频的编码格式
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), frame_rate, (frame_width, frame_height + resize_h))

        # 开始活动、 停止活动、 当前帧数
        active_frame_index = 0
        stop_frame_index = 0
        cur_frame_index = 0
        active_flag = True
        global rotate_flag

        # 左右旋转检测标志
        rotate_flag = True

        mapping_points = []
        line_angles = []
        sum_angle = 0.0
        sum_dist = 0
        while True:
            ret, frame = cap.read()
    #
            if ret is False:
                print("Video is played finished")
                break
            # 1. 检测对象bounding_box  x1, y1, x2, y2, confidence
            bounding_box = prediction_bounding_box(frame, model)

            # 2. 计算映射到平面坐标  相机内参、畸变系数、旋转矩阵、偏移量、边界框、每帧图片
            mapping_point = calculate_mapping_coordinate(mtx, dist, rotate_matrix, tvec, bounding_box)

            # 3. 计算当前图片旋转角度
            cur_angle = calculate_rotate_angle(frame, bounding_box, mapping_point)
            # print('度:' + str(cur_angle))

            # 4. 计算总旋转角度
            sum_angle += calculate_additional_angle(line_angles, cur_angle)
            # sum_angle = 0
            # print('sum angle --------------', str(sum_angle))

            # 5. 计算距离总和(cm)
            sum_dist += calculate_additional_dist(mapping_points, mapping_point)
            # print('sum dist --------------', str(sum_dist))

            # 6. 计算运动时间(s)
            cur_frame_index += 1
            # 得到进入场地的帧
            active_frame_index, stop_frame_index = get_active_and_stop_indexframe(active_frame_index, stop_frame_index, cur_frame_index, mapping_point, mapping_points)
            active_time = calculate_active_time(active_frame_index, stop_frame_index, frame_rate, mapping_point)
            # print('cur active time:{}s'.format(active_time))

            # 7. 平均速度----距离/时间
            velocity = calculate_average_velocity(sum_dist, active_time)
            # print('平均速度:{} m/s'.format(velocity))

            # 8. 画出物体bounding_box
            frame = draw_bounding_box(frame, bounding_box)

            # 9. 添加说明信息
            frame = draw_info(frame, mapping_point, sum_dist, active_time, velocity, sum_angle)

            # 10. 记录移动轨迹
            resize_background_image = draw_trajectory(resize_background_image, mapping_point, ratio=3)

            # 11. 拼接图像
            new_frame = np.zeros((frame_height + resize_h, frame_width, 3), np.uint8)
            new_frame[:] = [255, 255, 255]
            # 新帧 视频区域
            new_frame[:frame_height, :frame_width, :] = frame[:,:,:]
            # 新帧 映射区域

            left_deviation = (frame_width - resize_w) // 2
            new_frame[frame_height:, left_deviation:resize_w + left_deviation, :] = resize_background_image[:, :, :]

            # 12. 预测轨迹
            new_frame = draw_prediction_trajectory(new_frame, mapping_point, 3, left_deviation, frame_height)

            # 13. 说明模板
            new_frame = cv2ImgAddText(new_frame, '轨迹记录：', 20, frame_height + 95, textColor=(0, 0, 0), textSize=30)
            new_frame = cv2ImgAddText(new_frame, '与预测', 35, frame_height + 135, textColor=(0, 0, 0), textSize=30)
            info_label_left = 1120
            info_label_top = 850
            cv2.circle(new_frame, (info_label_left, info_label_top), 1, (0, 0, 255), 10)
            new_frame = cv2ImgAddText(new_frame, '轨迹记录', info_label_left + 25, info_label_top - 12, textColor=(0, 0, 0), textSize=24)
            cv2.circle(new_frame, (info_label_left, info_label_top+50), 1, (255, 0, 0), 10)
            new_frame = cv2ImgAddText(new_frame, '轨迹预测', info_label_left + 25, info_label_top - 12+50, textColor=(0, 0, 0), textSize=24)
            print(new_frame.shape[:2])
            out.write(new_frame)
            cv2.namedWindow('detect_obj', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('detect_obj', int(frame_width * proportion), int(frame_height * proportion))
            cv2.imshow('detect_obj', new_frame)
            cv2.waitKey(1)

    out.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
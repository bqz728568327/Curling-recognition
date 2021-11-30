import cv2
import numpy as np
import cv2.aruco as aruco
from config_logging import LogConfig
import os

logging = LogConfig()

def load_camera_calibration():
    try:
        # npzfile = np.load('./calibrateDataMi5.npz')
        mtx = np.array([[1.60709644e+03, 0.00000000e+00, 9.55953541e+02],
                        [0.00000000e+00, 1.50681824e+03, 5.30875983e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float)
        dist = np.array([[ 2.95116787e-02, -1.30555362e-01, -2.64550740e-03,  1.09149435e-04,1.73195597e+00]],dtype=np.float)
    except IOError:
        logging.error('cant find camera calibration data')
        raise Exception()
    return mtx, dist

def get_datum_coordinates(image, mtx, dist):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect aruco markers
    # DICT_6X6_50字典
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    # rejectedImgPoints可以暂时忽略，适用于调试目的和“重新查找”策略
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

    for i in range(len(rvecs)):
        rvec = rvecs[i]
        tvec = tvecs[i]
        aruco.drawAxis(image, mtx, dist, rvec, tvec, 0.1)
    aruco.drawDetectedMarkers(gray_image, corners)
    # cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('detect', 1024, 512)
    cv2.namedWindow('detect1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('detect1', 1024, 512)
    # cv2.imshow("detect", gray_image)
    cv2.imshow("detect1", image)
    cv2.waitKey(0)
    return corners, ids

def camera_estimate_pose(mtx, dist, corners, markerIDs, ref_marker_array):
    '''
        根据基准点的marker，解算相机的旋转向量rvecs和平移向量tvecs，(solvePnP(）实现)
        并将rvecs转换为旋转矩阵输出(通过Rodrigues())
        输入：
            cameraMtx内参矩阵，
            dist畸变系数。
            当前处理的图像帧frame，
            用于定位世界坐标系的参考点refMarkerArray.  py字典类型,需要len(refMarkerArray)>=3, 格式：{ID:[X, Y, Z], ID:[X,Y,Z]..}
            corners, detectMarkers()函數的輸出
            markerIDs, detectMarkers()函數的輸出
        输出：旋转矩阵rMatrix, 平移向量tVecs
    '''
    marker_count = len(ref_marker_array)
    # print(marker_count)
    if marker_count < 4:  # 标志板少于四个
        logging.error('at least 4 pair of points required when estimate camera post')
        raise RuntimeError()

    world_points = []
    image_points = []
    if len(ids) < 4:
        logging.error('at least 4 pair of points required be find')
        raise Exception()

    # 检查是否探测到了所有预期的基准marker
    logging.debug('------detected ref markers----')
    for i in range(len(ids)):
        if ids[i][0] in ref_marker_array:  # 如果是參考點的標志，提取基准点的图像坐标，用于构建solvePnP()的输入
            # print('id:\n ' + str(ids[i][0]))
            # print('cornors: \n ' + str(corners[i][0]))
            world_points.append(ref_marker_array[ids[i][0]])
            image_point = (corners[i][0][0] + corners[i][0][1] + corners[i][0][2] + corners[i][0][3]) // 4
            image_center_point = (corners[i][0][0] + corners[i][0][1] + corners[i][0][2] + corners[i][0][3]) // 4
            image_points.append(image_center_point.tolist())  # 提取基准坐标中心点

    world_points = np.array(world_points)
    image_points = np.array(image_points)
    # print(image_points)
    # print('object_points:\n' + str(world_points))
    # print('image_points:\n' + str(image_points))
    for i in range(len(image_points)):
        center = tuple(image_points[i])
        cv2.circle(frame, (int(center[0]), int(center[1])), 1, (0, 0, 255), 15)
    # cv2.namedWindow('detect2', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('detect2', 1024, 512)
    # cv2.imshow('detect2', frame)


    if len(world_points) >= 4:
        # 至少需要4個點
        retval, rvec, tvec = cv2.solvePnP(world_points, image_points, mtx, dist)
        rotate_matrix, jacobian = cv2.Rodrigues(rvec)
        return rotate_matrix, tvec
    else:
        logging.error('at least 4 points requested')
        raise Exception()

if __name__ == '__main__':
    filename = 'data/test8.mp4'
    filepath = 'parameters/post_adjust.npz'
    # 世界三维坐标
    ref_marker_array = {
        0: [50.0, 0.0, 0.0],
        1: [250.0, 0.0, 0.0],
        2: [50.0, 100.0, 0.0],
        3: [250.0, 100.0, 0.0],
    }

    cap = cv2.VideoCapture(filename)

    # 1.加载相机内参和形变畸数
    mtx, dist = load_camera_calibration()

    if cap.isOpened():

        for i in range(15):
            ret, frame = cap.read()
    #
        if ret is False:
            print("Video is played finished")
            raise Exception
        # 2.
        corners, ids = get_datum_coordinates(frame, mtx, dist)

        # 3.
        rotate_matrix, tvec = camera_estimate_pose(mtx, dist, corners, ids, ref_marker_array)
        if os.path.exists(filepath):
            os.remove(filepath)
        np.savez('parameters/post_adjust.npz', rotate_matrix, tvec)

        print(rotate_matrix)
        print(tvec)
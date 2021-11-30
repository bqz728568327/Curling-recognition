import cv2
import numpy as np
import glob
from config_logging import LogConfig
import os

def camera_calibration(h, w, image_root):
    # 设置阈值
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    filepath = 'parameters/calibration.npz'

    # 世界坐标系下的交界点坐标
    checkerboard_point = np.zeros((w * h, 3), np.float32)
    # 去掉Z轴，记为二维矩阵,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    checkerboard_point[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # 在世界坐标系中的三维点
    checkerboard_points = []
    # 在图像平面的二维点
    image_points = []

    images = glob.glob(image_root)
    print('./Calibration/*.jpg')
    if len(images) == 0:
        logging.error('Not find Target Images')
    for fname in images:
        image = cv2.imread(fname)
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            logging.error('File format appear Error')
            return
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray_image, (w, h), None)
        if ret is True:
            cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
            checkerboard_points.append(checkerboard_point)
            image_points.append(corners)
            logging.info('{} find corners'.format(fname))
            # 将角点在图像上显示
            # cv2.drawChessboardCorners(image, (w, h), corners, ret)
            # cv2.imshow('findCorners', image)
            # cv2.waitKey(2)

        else:
            logging.warning('{} not find corners'.format(fname))
    cv2.destroyAllWindows()

    # # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(checkerboard_points, image_points, gray_image.shape[::-1], None, None)
    print(mtx)
    print(dist)

    # r = np.load('parameters/calibration.npz')
    # mtx1 = r['arr_0']
    # dist1 = r['arr_1']
    # print(mtx1)
    # print(dist1)
    h, w = gray_image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print(newcameramtx)

    if os.path.exists(filepath):
        os.remove(filepath)
    # np.savez('parameters/calibration.npz', mtx, dist)
    np.savez(filepath, newcameramtx, dist)


if __name__ == '__main__':
    # 日志消息
    logging = LogConfig()
    # 棋盘交界点
    h, w = 5, 8
    # 棋盘图片路径
    image_root = 'Calibration/*.jpg'
    # 相机标定
    camera_calibration(h, w, image_root)
import cv2
from cv2 import aruco
import numpy as np
import os
import imageio
import glob

def calib_fisheye(filename, checkerboard = (5,5)):
    #using checkerboard
    video = imageio.get_reader(filename)
    vid_length = video.count_frames() 
    CHECKERBOARD = checkerboard

    if vid_length > 1000:
        pictures = [np.asarray(video.get_data(i)) for i in range(vid_length) if i%5 == 0]
    else:
        pictures = [np.asarray(video.get_data(i)) for i in range(vid_length)]

    #https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    for img in pictures:
        img_shape = img.shape[:2]

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_imm = len(objpoints)
    print("\nnumber of calibration images", N_imm)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    imsize = gray.shape[::-1]
    retval, K, D, rvec, tvec = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        imsize,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))    
    print(retval)
    
    return {"K": K, "D": D, "rvec": rvec, "tvec": tvec, "size": imsize}

def calib_camera_charuco(filename, s):
    #using charuco videos
    video = imageio.get_reader(filename)
    vid_length = video.count_frames() 
    
    ARUCO_DICT = cv2.aruco.Dictionary_get(aruco.DICT_6X6_250)

    board = aruco.CharucoBoard_create(
    	squaresX=5,
    	squaresY=7,
    	squareLength=10, 
    	markerLength=8, 
    	dictionary=ARUCO_DICT)
    
    if vid_length > 1000:
        pictures = [np.asarray(video.get_data(i)) for i in range(vid_length) if i%s == 0]
    else:
        pictures = [np.asarray(video.get_data(i)) for i in range(vid_length)]
        
    allCorners = []
    allIds = []
    decimator = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for img in pictures:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        if len(corners)>0:
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                if len(res2[1]) > 5:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])
        decimator+=1
    print("\nnumber of calibration images:", len(allIds))

    imsize = gray.shape
    
    ret,K,D,rvec,tvec = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
    print(ret)

    return {"K": K, "D": D, "rvec": rvec, "tvec": tvec, "size": imsize}

def calib_camera_checkerboard(filename, s, checkerboard=(5, 5)):
    video = imageio.get_reader(filename)
    vid_length = video.count_frames() 
    CHECKERBOARD = checkerboard

    if vid_length > 1000:
        pictures = [np.asarray(video.get_data(i)) for i in range(vid_length) if i%s == 0]
    else:
        pictures = [np.asarray(video.get_data(i)) for i in range(vid_length)]

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    for img in pictures:
        img_shape = img.shape[:2]

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_imm = len(objpoints)
    print("\nnumber of calibration images", N_imm)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    imsize = gray.shape[::-1]
    retval, K, D, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)
    print(retval)
    
    return {"K": K, "D": D, "rvec": rvec, "tvec": tvec, "size": imsize}
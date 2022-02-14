import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, uic
import sys
import random
from PyQt5.QtWidgets import QComboBox


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('CVDL_HW1.ui', self)                    
        # pass the definition/method, not the return value!
        self.pushButton.clicked.connect(self.findCornerPress) 
        self.pushButton_2.clicked.connect(self.findIntrinsicPress)
        self.pushButton_3.clicked.connect(self.findExtrinsicPress)
        self.pushButton_4.clicked.connect(self.findDistortionPress)
        self.pushButton_5.clicked.connect(self.showResult)
        self.pushButton_6.clicked.connect(lambda: self.showWord(1))
        self.pushButton_7.clicked.connect(lambda: self.showWord(2))
        self.pushButton_8.clicked.connect(self.findDisparityMap)
        self.pushButton_12.clicked.connect(self.CheckDisparityValue)
        self.pushButton_9.clicked.connect(lambda: self.findKeyPoint(1))
        self.pushButton_10.clicked.connect(lambda: self.findKeyPoint(2))
        self.pushButton_11.clicked.connect(self.warpedimage)
        self.show()


#######-------------HOMEWORK 1.1--------------------##########
    def findCornerPress(self):
        print("Find Corner Button Pressed : \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            
            imS = cv2.resize(img, (1024, 1024))                    # Resize image
            cv2.imshow("output", imS)                            # Show image
            cv2.waitKey(0)                                      # Display the image infinitely until any keypress

        cv2.destroyAllWindows()  


#######-------------HOMEWORK 1.2--------------------##########
    def findIntrinsicPress(self):
        # Defining the dimensions of checkerboard
        print("Find Intrinsic Button Pressed : \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)        
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Camera matrix : \n")
        print(mtx)

#######-------------HOMEWORK 1.3--------------------##########
    def findExtrinsicPress(self):
        # Defining the dimensions of checkerboard
        print("Find Extrinsic Button Pressed : \n")
        a = int(self.lineEdit.text())

        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)        
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        R_matrix, _ = cv2.Rodrigues(rvecs[a])
        Rt_matrix = np.concatenate((R_matrix, tvecs[a]), axis=1)

        print("R_Matrix : \n")
        print(R_matrix)
        print("Rt_Matrix : \n")
        print(Rt_matrix)

#######-------------HOMEWORK 1.4--------------------##########
    def findDistortionPress(self):
        # Defining the dimensions of checkerboard
        print("Find Distortion Button Pressed : \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)        
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("dist : \n")
        print(dist)

#######-------------HOMEWORK 1.5--------------------##########
    def showResult(self):
        print("Find Show Result Buttom: \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        images = glob.glob('Q1_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)        
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        for fname in images:
            img = cv2.imread(fname)
            # cv2.resize(img,(1024,1024))
            # cv2.imshow("Distorted image", img)
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,CHECKERBOARD,0,CHECKERBOARD)
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # # crop the image
            # x,y,w,h = roi
            # dst = dst[y:y+h, x:x+w]

            cv2.resize(dst,(1024,1024))
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Distorted image', cv2.WINDOW_NORMAL)
            cv2.imshow("Distorted image", img)
            cv2.imshow("result",dst)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
#######-------------HOMEWORK2--------------------##########
    def draw(img, imgpts):
        
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)

        return img

    def showWord(self,i):
        print("Drawing Word : \n")
        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = [] 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        
        for fname in glob.glob('Q2_Image/*.bmp'):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #corners is the corner find in chessboard
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.

                corners2 = cv2.cornerSubPix(gray, corners, CHECKERBOARD,(-1,-1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.destroyAllWindows()    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        word = self.textEdit.toPlainText()
        if i == 1:
            fs=cv2.FileStorage('./Q2_Image/Q2_Lib/alphabet_lib_onboard.txt',cv2.FILE_STORAGE_READ)
        else:
            fs=cv2.FileStorage('./Q2_Image/Q2_Lib/alphabet_lib_vertical.txt',cv2.FILE_STORAGE_READ)

        ch={}
        ch[0] = np.float32([[7, 5, 0], [7, 5, 0]])
        ch[1] = np.float32([[4, 5, 0], [4, 5, 0]])
        ch[2] = np.float32([[1, 5, 0], [1, 5, 0]])
        ch[3] = np.float32([[7, 2, 0], [7, 2, 0]])
        ch[4] = np.float32([[4, 2, 0], [4, 2, 0]])
        ch[5] = np.float32([[1, 2, 0], [1, 2, 0]])

        word_={}
        for i in range(len(word)):
            word_[i]=np.float32(fs.getNode(word[i]).mat())

        for fname in glob.glob('Q2_Image/*.bmp'):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD ,None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,CHECKERBOARD,(-1,-1),criteria)
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
                # project 3D points to image plane
                	         
                for i in range(len(word)):
                    shape_size = word_[i].shape
                    for j in range(shape_size[0]):
                        imgpts, jac = cv2.projectPoints(word_[i][j]+ch[i], rvecs, tvecs, mtx, dist)
                        img = Ui.draw(img,imgpts)

                imS = cv2.resize(img, (1024, 1024))                    # Resize image
                cv2.imshow('img',imS)
                k = cv2.waitKey(0) & 0xFF
                if k == ord('s'):
                    cv2.imwrite(fname[:6]+'.bmp', imS)
        cv2.destroyAllWindows()

#######-------------HOMEWORK3--------------------##########
    def findDisparityMap(self):

        imgL = cv2.imread('Q3_Image/imL.png',0)
        imgR = cv2.imread('Q3_Image/imR.png',0)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL,imgR)
        print(disparity)
        plt.imshow(disparity,'gray')
        plt.show() 
#######-------------HOMEWORK3-2--------------------##########        
    def CheckDisparityValue(self):
        imgL = cv2.imread('Q3_Image/imL.png',0)
        imgR = cv2.imread('Q3_Image/imR.png',0)

        img_CL = cv2.imread('Q3_Image/imL.png')
        img_CR = cv2.imread('Q3_Image/imR.png')

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        
        disparity = stereo.compute(imgL,imgR)
        disparity_scaled = disparity // 16
        disparity_view = cv2.convertScaleAbs(disparity_scaled)

        concat = np.concatenate((img_CL, img_CR), axis=1)

        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result',concat)

        def draw_circle(event, x, y,flags,param):

            if event == cv2.EVENT_LBUTTONDOWN:
                numpy_concat = np.concatenate((img_CL, img_CR), axis=1)
                x_value = disparity_view[y][x]
                cv2.circle(numpy_concat,(x+2800-x_value,y),10,(0,255,0),-1)
                cv2.imshow('result',numpy_concat)

        cv2.setMouseCallback('result',draw_circle)
        
        while(1):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
#######-------------HOMEWORK4--------------------##########
    def findKeyPoint(self,i):
        # read images
        img1 = cv2.imread('Q4_Image/Shark1.jpg')  
        img2 = cv2.imread('Q4_Image/Shark2.jpg') 

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        MIN_MATCH_COUNT = 10

        #sift
        sift = cv2.xfeatures2d.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = bf.match(descriptors_1,descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance,reverse=True)

        if i == 1 :
            img1_idx = []
            img2_idx = []
            for i in range(200):
                img1_idx.append(keypoints_1[matches[i].queryIdx])
                img2_idx.append(keypoints_2[matches[i].trainIdx])

            #draw keypoint
            imgN1 = cv2.drawKeypoints(img1, keypoints=img1_idx, outImage=img1,  flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            imgN2 = cv2.drawKeypoints(img2, keypoints=img2_idx, outImage=img2,  flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

            numpy_horizontal_concat = np.concatenate((imgN1, imgN2), axis=1)
            cv2.resize(numpy_horizontal_concat,(1024, 1024))
            cv2.imshow('image', numpy_horizontal_concat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else :       
            matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
            img3 = cv2.drawMatches(img1,keypoints_1,img2,keypoints_2,good[:200],None,**draw_params)
            cv2.imshow('image',img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:200], img2, flags=2)
            # cv2.resize(img3,(1024,1024))
            # cv2.imshow('image', img3)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
#######-------------HOMEWORK4-3--------------------##########
    def warpedimage(self):
        img_1 = cv2.imread('Q4_Image/Shark1.jpg')  
        img_2 = cv2.imread('Q4_Image/Shark2.jpg') 

        img1 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()


        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        matches =sorted(matches, key=lambda x: x[0].distance / x[1].distance)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        
        if len(good) >= 4:
            src = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC,7.0)
            dst = cv2.warpPerspective(img1,H,(img2.shape[1] + img1.shape[1], img2.shape[0]))
            dst[0:img2.shape[0],0:img2.shape[1]] = img2
        cv2.resize(dst,(1024,1024))
        cv2.imshow('warped_image',dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
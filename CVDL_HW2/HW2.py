import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QLabel
import cv2.aruco
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize
from PIL import Image

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Homework2.ui', self)                    
        # pass the definition/method, not the return value!
        self.BackgroundSubtraction.clicked.connect(self.BkGrSub)
        self.Preprocessing.clicked.connect(self.Pre_video_Tracking)
        self.VideoTracking.clicked.connect(self.videotracking)
        self.PerspectiveTransform.clicked.connect(self.PerTrans)
        self.ImageReconstruction.clicked.connect(self.ImgReconstruction)
        self.ReconstructionError.clicked.connect(self.RecError)

        self.show()


#######-------------HOMEWORK 4.2--------------------##########
    def RecError(self):
        shape = (400,400,3)
        for jpgfile in glob.glob("Q4_Image/*.jpg"): 
            #img = Image.open(jpgfile) 
            img = cv2.imread(jpgfile)
                        
            a = cv2.imread(jpgfile,0)
            img1 = np.array(img)
            #print(img1.shape)
            
            flatten_images = np.reshape(img1, (img1.shape[0], -1))

            pca = PCA(n_components=25)
            
        
            pipe = make_pipeline(StandardScaler(), pca) 
            transformed_images = pca.fit_transform(flatten_images)
            
            reconstructed_images = pca.inverse_transform(transformed_images)
            reconstructed_images = minmax_scale(reconstructed_images, axis=1)

            reconstructed_images = reconstructed_images.reshape(shape)
            reconstructed_images = reconstructed_images[:,:,0].astype('uint8') #convert to grayscale

            # print("reconstructed_images.shape:", reconstructed_images.shape)
            # print("a.shape ", a.shape)
            result = 0
            for i in range (a.shape[0]): #traverses through height of the image
                for j in range (a.shape[1]): #traverses through width of the image
                   result = result + abs(a[i][j] - reconstructed_images[i][j])
             
            # result = abs(flatten_images.astype('uint8') - flatten_Trans_images.astype('uint8'))
            print("The Reconstruction Error of ", jpgfile ," is : ",result)


#######-------------HOMEWORK 4.1--------------------##########
    def ImgReconstruction(self):
        # data_img =[]
        data_numpy=[]
        for jpgfile in glob.glob("Q4_Image/*.jpg"): 
            img_pre=Image.open(jpgfile) #
            img = img_pre.convert('RGB')
            #img.show()
            img1 = np.array(img)
            #img1 = np.array(img_pre)
            #print(img1.shape)
            # data_img.append(img)
            data_numpy.append(img1)

        out =np.array(data_numpy)
        print(out.shape)

        flatten_images = np.reshape(out, (out.shape[0], -1))
        print("flatten_images.shape:", flatten_images.shape)

   
        # pca = PCA(n_components=20)
        pca = PCA(n_components=25)

        pipe = make_pipeline(StandardScaler(), pca) 
        transformed_images = pca.fit_transform(flatten_images)
        print("transformed_images.shape:", transformed_images.shape)

        reconstructed_images = pca.inverse_transform(transformed_images)
        reconstructed_images = minmax_scale(reconstructed_images, axis=1)

        # shape = (100, 100, 3)
        shape = (400, 400, 3)
        fig, ax = plt.subplots(4, 15, figsize=(14, 8),
                        subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=-0.75, wspace=0.2))
        
        for j in range(15):
            # idx = i * 10 + j
            ax[0, j].imshow(out[j].reshape(shape).astype('uint8'))
            ax[1, j].imshow(reconstructed_images[j].reshape(shape))
            ax[2, j].imshow(out[j+15].reshape(shape).astype('uint8'))
            ax[3, j].imshow(reconstructed_images[j+15].reshape(shape))
        

        # image = Image.fromarray(cv2.cvtColor(out[0].reshape(shape).astype('uint8'),cv2.COLOR_BGR2RGB))
        # image.show()
        #cv2.imshow('test',image) 

        ax[0, 0].set_ylabel('Original')
        ax[1, 0].set_ylabel('Reconstruction')
        ax[2, 0].set_ylabel('Original')
        ax[3, 0].set_ylabel('Reconstruction')
        
        plt.show()
#######-------------HOMEWORK 3--------------------##########

    def PerTrans(self):
        # image or video or other
        detectType = 'video'
        detectPath = 'Q3_Image/perspective_transform.mp4' 
        im_src = cv2.imread("Q3_Image/logo.png")

        outputFile = "ar_out_py.avi"

        if not os.path.isfile(detectPath):
            print(" Input video file ", detectPath, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(detectPath)
        outputFile = detectPath[:-4]+'_ar_out_py.avi'
        print("Storing it as :" , outputFile)
        
        vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28, (round(
            2*cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cv2.waitKey(1) < 0:
            try:
                # get frame from the video
                hasFrame, frame = cap.read()

                # Stop the program if reached end of video
                if not hasFrame:
                    print("Done processing !!!")
                    print("Output file is stored as ", outputFile)
                    cv2. waitKey(3000)
                    break

                # Load the dictionary that was used to generate the markers.
                dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

                # Initialize the detector parameters using default values
                parameters = cv2.aruco.DetectorParameters_create()

                # Detect the markers in the image
                markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
                    frame, dictionary, parameters=parameters)

                #Find id for each markers
                id1 = np.squeeze(np.where(markerIds == 1))
                id2 = np.squeeze(np.where(markerIds == 2))
                id3 = np.squeeze(np.where(markerIds == 3))
                id4 = np.squeeze(np.where(markerIds == 4))
                #Process of perspective transform
                if id1 !=[] and id2 !=[] and id3 !=[] and id4 !=[]:
                    #Get the top-left corner of marker1
                    pt1 = np.squeeze(markerCorners[id1[0]])[0]
                    #Get the top-right corner of marker2
                    pt2 = np.squeeze(markerCorners[id2[0]])[1]
                    #Get the bottom-right corner of marker3
                    pt3 = np.squeeze(markerCorners[id3[0]])[2]
                    #Get the bottom-left corner of marker4
                    pt4 = np.squeeze(markerCorners[id4[0]])[3]
                    #Get coordinates of the corresponding quadrangle vertices in the destination image
                    pts_dst =[[pt1[0],pt1[1]]]
                    pts_dst = pts_dst + [[pt2[0],pt2[1]]]
                    pts_dst = pts_dst + [[pt3[0],pt3[1]]]
                    pts_dst = pts_dst + [[pt4[0],pt4[1]]]
                    #Get coordinates of quadrangle vertices in the source image
                    pts_src = [[0,0], [im_src.shape[1],0], [im_src.shape[1],im_src.shape[0]], [0, im_src.shape[0]]]

                pts_src_m = np.asarray(pts_src)
                pts_dst_m = np.asarray(pts_dst)

                # Calculate Homography
                h, status = cv2.findHomography(pts_src_m, pts_dst_m)

                # Warp source image to destination based on homography
                warped_image = cv2.warpPerspective(
                    im_src, h, (frame.shape[1], frame.shape[0]))

                # Prepare a mask representing region to copy from the warped image into the original frame.
                mask = np.zeros([ frame.shape[0], frame.shape[1]], dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32(
                    [pts_dst_m]), (255, 255, 255), cv2.LINE_AA)

                # Erode the mask to not copy the boundary effects from the warping
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.erode(mask, element, iterations=3)

                # Copy the mask into 3 channels.
                warped_image = warped_image.astype(float)
                mask3 = np.zeros_like(warped_image)
                for i in range(0 , 3):
                    mask3[:, :, i] = mask/255

                # Copy the warped image into the original frame in the mask region.
                warped_image_masked = cv2.multiply(warped_image, mask3)
                frame_masked = cv2.multiply(frame.astype(float), 1-mask3)
                im_out = cv2.add(warped_image_masked, frame_masked)

                # Showing the original image and the new output image side by side
                concatenatedOutput = cv2.hconcat([frame.astype (float), im_out])
                cv2.namedWindow('AR using Aruco markers',0)
                cv2.resizeWindow('AR using Aruco markers', 1200, 500)
                cv2.imshow("AR using Aruco markers",
                        concatenatedOutput.astype(np.uint8))

                # Write the frame with the detection boxes
                vid_writer.write(concatenatedOutput.astype(np.uint8))

            except Exception as inst:
                print(inst)

        cv2.destroyAllWindows()
        if 'vid_writer' in locals():
            vid_writer.release()
            print('Video writer released..')

#######-------------HOMEWORK 2.2 --------------------##########
    # Parameter for video
    def Parameter_video_tracking(self):
        cap = cv2.VideoCapture("Q2_Image/optical_flow.mp4")
        _, first_frame = cap.read()
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_frame_gray = cv2.GaussianBlur(first_frame_gray, (5, 5), 0)

        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 80
        params.maxThreshold = 300
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 55
        # Filter by Circularity
        params.filterByCircularity = True
        params.maxCircularity = 1
        params.minCircularity = 0.8
        # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.9
        # params.maxConvexity = 1
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        detector = cv2.SimpleBlobDetector()

        is_v2 = cv2.__version__.startswith("2.")
        if is_v2:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)
    
        keypoints = detector.detect(first_frame)
        List_point_all = []
        
        for i in range(len(keypoints)):
            x,y = np.int(keypoints[i].pt[0]),np.int(keypoints[i].pt[1])
            sz = np.int(keypoints[i].size)
            list_point = [[x,y]]
            List_point_all.append(list_point)
            if sz > 1:
                sz = np.int(sz/2)
            im_with_keypoints = cv2.rectangle(first_frame, (x-2-sz,y-2-sz), (x+10-sz,y+10-sz), color=(0,0,255), thickness=2)
            im_with_keypoints = cv2.line(im_with_keypoints,(x-2-sz,y),(x+10-sz,y),(0,0,255),1)
            im_with_keypoints = cv2.line(im_with_keypoints,(x,y-2-sz),(x,y+10-sz),(0,0,255),1)
        # cv2.imshow("First Frame", im_with_keypoints)
        # cv2.waitKey(0)
        return im_with_keypoints,List_point_all
        # cap.release()
        # cv2.destroyAllWindows()
#######-------------HOMEWORK 2.1 --------------------##########
    def Pre_video_Tracking(self):
        text =  "Processing 7 blue circles of the 1st frame..."
        print(text)

        im_with_keypoints,List_point_all = self.Parameter_video_tracking()
        cv2.imshow("Pre_tracking", im_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def videotracking(self):
        text =  "Processing video tracking..."
        print(text)

        cap = cv2.VideoCapture("Q2_Image/optical_flow.mp4")

        lk_params = dict( winSize  = (21,21),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors

        ret, old_frame = cap.read()
        old_frame,List_point_all = self.Parameter_video_tracking()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = np.array(List_point_all,np.float32)
    
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while(1):
            ret,frame = cap.read()
            
            if (ret):
                # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                first_frame_blue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                blueLower = (25,50,50)  #130,150,80
                blueUpper = (160,250,250) #250,250,120
                mask1 = cv2.inRange(first_frame_blue, blueLower, blueUpper)
                try:
                    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                except:
                    _,contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                frame_coppy = frame.copy()
                cv2.drawContours(frame_coppy, contours, -1, (255,0,0), 8)
                # cv2.imshow("sds",frame_coppy)
                frame_gray = cv2.cvtColor(frame_coppy, cv2.COLOR_BGR2GRAY)
                # # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
                # # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]
                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    # print(a,b,c,d)
                    mask = cv2.line(mask, (a,b),(c,d), (0,0,255), 2)
                    frame = cv2.rectangle(frame, (int(a)-6,int(b)-6), (int(a)+7,int(b)+7), color=(0,0,255), thickness=-1)
                
                img = cv2.add(frame,mask)
                
                cv2.imshow('frame',img)
                k = cv2.waitKey(20) & 0xff

                if k == 27:
                    break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            else:
                break
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#######-------------HOMEWORK 1--------------------##########
    
    def BkGrSub(self):
        cap =cv2.VideoCapture('Q1_Image/traffic.mp4')

        #frameIds = cap.get(cv2.CAP_PROP_POS_FRAMES)* np.array([i for i in range(1,51)])

        frameIds = np.array([i for i in range(0,50)])
        # Store selected frames in an array
        frames = []
        #print(frameIds)
        #width = 320
        #height = 176
        for fid in frameIds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            frames.append(frame)
        #print(len(frames))

        # Calculate the median, std along the time axis
        medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
        std = np.std(medianFrame, axis=0).astype(dtype=np.uint8)
        print(std)

        # Reset frame number to 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Convert background to grayscale
        grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
        mean, std = cv2.meanStdDev(grayMedianFrame)
        #print(mean)
        print(std)
        # Loop over all frames
        ret = True
        #a = 0
        while(ret):

            # Read frame
            ret, frame = cap.read()
            if ret == True:
                #print(a)
                # Convert current frame to grayscale
                frameG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean2, std2 = cv2.meanStdDev(frameG)
                #print(mean2)
                print(std2)
                # Calculate absolute difference of current frame and 
                # the median frame
                dframe = cv2.absdiff(frameG, grayMedianFrame)
                # Treshold to binarize
                th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
                # Display image
                final = np.hstack([frameG, dframe])
                cv2.imshow('frame', final)
                #a += 1
                k = cv2.waitKey(20)
                if k == 27:
                    break
            else:
                break

        # Release video object
        cap.release()
        cv2.destroyAllWindows()


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

 
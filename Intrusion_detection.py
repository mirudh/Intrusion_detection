import threading
# pip install opencv-python
# pip install streamlit
import cv2
import streamlit as st

st.title('Intrusion Detection')

n = st.number_input("Enter the no.of cameras to be displayed:", step=1)
l = []
i = 0
while i < n:
    cam = st.selectbox("choose the cam to be viewed",
                       ("--<select>--","cam1","cam2","cam3","cam4","cam5"),key=i)
    st.write('You selected cam',cam)
    #print(type(cam))
    l.append(cam)
    i += 1

class myThread(threading.Thread):

    def __init__(self, threadID, rtsp):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.rtsp = rtsp

    def ORB_detector(self, new_image, frame2):
        # Function that compares input image to template
        # It then returns the number of ORB matches between them
        image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
        orb = cv2.ORB_create(1000, 1.2)

        # Detect keypoints of original image
        (kp1, des1) = orb.detectAndCompute(image1, None)

        # Detect keypoints of rotated image
        (kp2, des2) = orb.detectAndCompute(frame2, None)

        # Create matcher
        # Note we're no longer using Flannbased matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Do matching
        matches = bf.match(des1, des2)

        # Sort the matches based on distance.  Least distance
        # is better
        matches = sorted(matches, key=lambda val: val.distance)
        return len(matches)

    def run(self):
        print(self.rtsp)
        cap = cv2.VideoCapture(self.rtsp)

        image_template = cv2.VideoCapture(self.rtsp)
        ret, frame2 = image_template.read()
        for k in l:
            if k == self.threadID:
                cv2.imshow(f'frame{k}', frame2)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                cap = cv2.VideoCapture(self.rtsp)
                ret, frame = cap.read()
            # Get number of ORB matches
            matches = self.ORB_detector(frame, frame2)

            # Display status string showing the current no. of matches
            output_string = "Matches = " + str(matches)
            # cv2.namedWindow(f"frame{self.threadID}", cv2.WINDOW_NORMAL)
            cv2.putText(frame, output_string, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (250, 0, 250), 2)

            # Our threshold to indicate object deteciton
            # For new images or lightening conditions you may need to experiment a bit
            # Note: The ORB detector to get the top 1000 matches, 350 is essentially a min 35% match

            # If matches exceed our threshold then object has been detected
            # if camera facing a human, threshold<250 to be set else 380/550

            if matches < 750:
                # cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
                cv2.putText(frame, 'Motion detected', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 250, 0), 2)
            for j in l:
                if j == self.threadID:
                    cv2.imshow(f'Object Detector using ORB{j}', frame)
                    cv2.waitKey(1)


thread1 = myThread("cam1", "MVI_2919.mov") #Give the link of the rtsp or the video path to be displayed
thread1.start()

thread2 = myThread("cam2", "MVI_2919.mov") #Give the link of the rtsp or the video path to be displayed
thread2.start()

thread3 = myThread("cam3", "MVI_2919.mov") #Give the link of the rtsp or the video path to be displayed
thread3.start()

thread4 = myThread("cam4", "MVI_2919.mov") #Give the link of the rtsp or the video path to be displayed
thread4.start()

thread5 = myThread("cam5", "MVI_2919.mov") #Give the link of the rtsp or the video path to be displayed
thread5.start()





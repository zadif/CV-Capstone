from ultralytics import YOLO
import time
import numpy as np
import cv2 as cv
import logging
import threading
import queue


logger = logging.getLogger(__name__)

class Camera:

    def frameReader(self,queueObj,eventObj):

        while not eventObj.is_set():
            val,frame=self.video.read()
            if val ==False:
                eventObj.set()
                break
            queueObj.put(frame)
            
        return False

    def __init__(self,source,queueSize=5):

        try:
            self.video=cv.VideoCapture(source)
            if self.video.isOpened()==False:
                logging.critical("Video not captured")
            
            self.que=queue.Queue(maxsize=queueSize)
            self.evnt=threading.Event()


        except:
            logging.critical("Video not captured")
    

    
    def start(self):
        self.thred=threading.Thread(target=self.frameReader, args=(self.que,self.evnt ))
        self.thred.start()

    def getFrame(self):
        return self.que.get()
    
    def release(self):
        self.evnt.set()
        self.thred.join()
        self.video.release()


class Detector:
    def __init__(self,modelPath):
        try:
            self.model=YOLO(modelPath)
        except:
            logging.critical("Weights not found")

    def detect(self,frame,conf,device='cpu'):
        results=self.model(frame,device=device, conf=conf)
        return results
    
    def drawBoxes(self,frame,results):
        for box in results[0].boxes:
            lst=box.xyxy[0].tolist()
            conf=box.conf[0].item()
            clas=box.cls[0].item()

            cv.rectangle(frame,(int(lst[2]),int(lst[3])),(int(lst[0]),int(lst[1])),color=(0,255,0),thickness=2)
            cv.putText(frame,"Conf: "+str(round(conf,2)),(int(lst[0]),int(lst[1])),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
      


class Pipeline:

    def __init__(self,source,modelPath):
        self.camera=Camera(source)
        self.detector=Detector(modelPath)
        self.totalTime=0
        self.frameCount=0

    def run(self):
        cv.namedWindow("img",cv.WINDOW_NORMAL)
        cv.resizeWindow("img",1000,800)
        cv.createTrackbar("CONF", "img", 50, 100, lambda x: None)

        self.camera.start()

        while not self.camera.evnt.is_set() or not self.camera.que.empty():
            frame=self.camera.getFrame()
            startTime=time.perf_counter()
            confThreshold = cv.getTrackbarPos("CONF", "img") / 100.0

            results=self.detector.detect(frame, conf=confThreshold)
            self.detector.drawBoxes(frame,results)
            endTime=time.perf_counter()
            secondsPerFrame=endTime-startTime
            self.totalTime += secondsPerFrame
            self.frameCount += 1
            fps=1/secondsPerFrame
            cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv.imshow("img",frame)
            cv.waitKey(1)
            self.camera.que.task_done()
        self.cleanup()

    def cleanup(self):
        self.camera.release()
        cv.destroyAllWindows()

        if self.totalTime > 0:
            logging.info(f"Total processed frames: {self.frameCount}")
            logging.info(f"Average FPS: {self.frameCount / self.totalTime:.2f}")



if __name__ == "__main__":
    pipeline = Pipeline(source=r'C:\Users\zadif\Downloads\test.mp4', modelPath=r'C:\Users\zadif\Desktop\VS CODE\python\CV\yolo\best.pt')
    pipeline.run()
    


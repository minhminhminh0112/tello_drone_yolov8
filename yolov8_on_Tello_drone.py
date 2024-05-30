import os
from ultralytics import YOLO
import cv2
import math 
from djitellopy import Tello
import numpy as np
import csv
import pandas as pd
import time

#SET PARAMETER speed
speed = 100
# IMPORT MODEL WEIGHTS
model = YOLO(r'yolo-Weights/yolov8n.pt')

# READ DETECTABLE CLASSNAMES from a CSV file
classNames = []
with open(r'classNames.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file)
    classNames = next(reader)
    
#SET UP NEW FOLDER to save the images from running drone
main_filepath = r'output_running_drone/photos_positiveVel_bedroom_yellow_light_closer_objects'
counter = 1
folder_name = f'yaw{speed}_{counter}'
while os.path.exists(os.path.join(main_filepath, folder_name)):
    counter += 1
    folder_name = f'yaw{speed}_{counter}'
#create the new folder for run number
os.makedirs(os.path.join(main_filepath, folder_name))
filepath_run = os.path.join(main_filepath, folder_name)
#create 1 folder for raw image, and 1 for image with displayed bounding boxes
image_raw_filepath  = os.path.join(filepath_run, 'raw_images')
image_display_filepath = os.path.join(filepath_run, 'display_images')
os.makedirs(image_raw_filepath)
os.makedirs(image_display_filepath)
print('Folder created')
#os.chdir(filepath) 

# initial a dictionary to represent the table
table = {'frame': [], 'object': [], 'confidence': [], 'x1': [], 'x2': [], 'y1': [], 'y2': [], 'bb_area':[],'speed':[]}

class Drone(Tello):
    def __init__(self):
        Tello.__init__(self)

    def run(self, speed):
        #SET UP drone
        self.connect()
        self.TAKEOFF_TIMEOUT = 0.3
        self.RESPONSE_TIMEOUT = 0.3
        self.set_speed(speed)
        self.streamon()
        battery = self.get_battery()
        print('battery', battery)
        #TAKE OFF and LAND
        self.takeoff()
        self.move_up(30)
        
        for i in range(40):
            analyzed_frame, image = self.analyze_frame()
            self.display_save_output(analyzed_frame, image, frame_no=i) 
            self.send_rc_control(0,0,0,speed) #leftright, forthback,updown,yaw clockwise

        # create a pandas DataFrame from the table then export
        #this is done before landing because there could be a problem by landing and the results could not be saved.
        df = pd.DataFrame(table)
        df.to_csv(os.path.join(filepath_run,'bounding_box_info.csv'), index=False)
        
        time.sleep(3)
        self.land()

    #Read frame, save result of the object detection 
    def analyze_frame(self):
        img = self.get_frame_read().frame
        print('New frame is read')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = model(img, stream=True) #
        return results, img

    def get_coord_object(self, box):
        x1, y1, x2, y2 = box.xyxy[0] 
        return int(x1), int(y1), int(x2), int(y2)
    
    def show_bounding_box(self, img, box):
        # bounding box in cam
        x1, y1, x2, y2  = self.get_coord_object(box)
        org = [x1, y1] # It is the coordinates of the bottom-left corner of the text string in the image. 
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0) #blue
        thickness = 1
        cls = int(box.cls[0])# class name
        #print("Class name -->", classNames[cls])
        confidence = math.ceil((box.conf[0]*100))/100 #round(0)
        cv2.putText(img, f'{classNames[cls]} {confidence}', org, font, fontScale, color, thickness)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,165,0), 3) #the color should be BGR

    def save_coordinate_info(self, box,frame_no):
        x1, y1, x2, y2  = self.get_coord_object(box)
        area = abs(x2 - x1) * abs(y2 - y1)
        cls = int(box.cls[0])# class name
        confidence = math.ceil((box.conf[0]*100))/100 #round(0)
        table['frame'].append(frame_no)
        table['object'].append(classNames[cls])
        table['confidence'].append(confidence)
        table['x1'].append(x1)
        table['x2'].append(x2)
        table['y1'].append(y1)
        table['y2'].append(y2)
        table['bb_area'].append(area)
        table['speed'].append(speed)

    def display_save_output(self,results,img, frame_no):
        cv2.imwrite(os.path.join(image_raw_filepath,f'Frame{frame_no}.png'), img)
        for r in results: # a frame can have many boxes
            for box in r.boxes:
                self.show_bounding_box(img,box)
                self.save_coordinate_info(box, frame_no)
        cv2.imwrite(os.path.join(image_display_filepath ,f'Frame{frame_no}.png'), img)
        #Display all bounding boxes on screen
        cv2.imshow('image', img) #not showing images because of return
        cv2.waitKey(1)

    #end_time = time.time()
    #exe_time = end_time - start_time
    #save values into a table then export csv: table.columns=['frame','object' ,'x1','x2','y1','y2','total_area']
    
   

def main():
    me = Drone()
    me.run(speed)
    print('Finished running programm')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


import cv2
import os 
import time 
import uuid

base_path = './workspace/assets/images/all'  # Base path for image storage

labels = ['a', 'b', 'c', 'd', 'e']
number_imgs = 15

for label in labels:
    folder_path = os.path.join(base_path)  # Path for the specific label folder
    os.makedirs(folder_path, exist_ok=True)  # Ensure the label folder exists
    
    cap = cv2.VideoCapture(0)
    print("Collecting images for label: " + label)
    time.sleep(5)
    
    for index in range(number_imgs):
  
        ret, frame = cap.read()
        
        # Construct the image path including the label-specific folder
        imagename = os.path.join(folder_path, label + '-' + str(uuid.uuid1()) + '.jpg')
        
        # Save the image and display it
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

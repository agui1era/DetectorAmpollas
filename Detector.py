
import cv2  
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
import os
import ftplib
import time

def main():
  model='model.tflite'
  labels='labels.txt'
  img_input='input.jpg'
  img_output='output.jpg'
  limite=0.4
  cantidad=150

  cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
  filename =  img_input

  # Initialize engine.
  engine = DetectionEngine(model)
  labels = dataset_utils.read_label_file(labels) 


  while cv2.waitKey(1) & 0xFF != ord('q'):
    # Open the url image, set stream to True, this will return the stream content.

    ret,frame = cap.read() # return a single frame in variable `frame`
    cv2.imwrite(img_input,frame)   
   
    # Open image.
    img = Image.open(img_input).convert('RGB')
    #Make the new image half the width and half the height of the original image
    img = img.resize((round(img.size[0]*0.2), round(img.size[1]*0.2)))
 
    draw = ImageDraw.Draw(img)
   
    # Run inference.
    objs = engine.detect_with_image(img,
                                    threshold=limite,
                                    keep_aspect_ratio='store_true',
                                    relative_coord=False,
                                    top_k=cantidad)

    # Print and draw detected objects.
    for obj in objs:
      #print('-----------------------------------------')
     
      print('score =', obj.score)
      print(labels[obj.label_id]) 
      box = obj.bounding_box.flatten().tolist()
      #print('box =', box)
      draw.rectangle(box, outline='red')
      
      print('Subiendo imagen')
      with open(img_output, "rb") as file:
          # use FTP's STOR command to upload the file
          ftp.storbinary(f"STOR /www/output.jpg", file)

    if not objs:
      print('No objects detected.')

    # Save image with bounding boxes.
    if img_output:
      img.save(img_output)
    image = cv2.imread(img_output) 
    
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window",image)
    
      #closing all open windows  
  cv2.destroyAllWindows()   
  cap.release()   

if __name__ == '__main__':
  main()
  


import cv2  
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw

def main():
  model='model.tflite'
  labels='labels.txt'
  img_input='input.jpg'
  img_output='output.jpg'
  limite=0.5
  cantidad=150
  ampolla=0

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
    img = img.resize((round(img.size[0]*0.5), round(img.size[1]*0.5)))
 
    draw = ImageDraw.Draw(img)
   
    # Run inference.
    objs = engine.detect_with_image(img,
                                    threshold=limite,
                                    keep_aspect_ratio='store_true',
                                    relative_coord=False,
                                    top_k=cantidad)

    # Print and draw detected objects.
    for obj in objs:
      print('-----------------------------------------')
     
      #print('score =', obj.score)
      if labels[obj.label_id]=='ampolla':
          ampolla=ampolla+1 
      box = obj.bounding_box.flatten().tolist()
      #print('box =', box)
      draw.rectangle(box, outline='red')
    print('Ampollas:'+ str(ampolla))
    if not objs:
      print('No objects detected.')
    ampolla=0
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
  

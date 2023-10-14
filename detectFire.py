from ultralytics import YOLO
import cvzone
import cv2
import math

def drawxy(img,x,y,w,h):
    cv2.line(img, (320,0), (320, 480), (0,255,255), 2) # dọc  y
    cv2.line(img, (0,240), (640, 240), (0,255,255), 2) # ngang x
    
    cv2.putText(img, "x" , (620 , 230), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, "y" , (330 , 20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2, cv2.LINE_AA)
def drawKc(img, x,y,w,h):
    cv2.line(img, (320,y+int(h/2)), (x+int(w/2), y+int(h/2)), (0,255,0), 2) # 
    cv2.line(img, (x+int(w/2), 240), (x+int(w/2), y+int(h/2)), (0,255,0), 2) # 
def kcx(x,y,w,h):
    # Tọa độ của điểm 1
    #x1, y1 = 0, 240
    # Tọa độ của điểm 2
    #x2, y2 = 320, 240 
    # tọa độ điểm 1
    # x1,y1 = 320, y+int(h/2)
    # tọa độ điểm 2
    # x2,y2 = x+int(w/2), y+int(h/2)
    kc = math.sqrt(( (x+int(w/2) - 320) )**2 + ((y+int(h/2)) - (y+int(h/2)))**2)
   
    return kc
def kcy(x,y,w,h):  
    # Tọa độ của điểm 1
    #x1, y1 = 0, 240
    # Tọa độ của điểm 2
    #x2, y2 = 320, 240 
    # tọa độ điểm 1
    #tam_lua = (x+int(w/2), y+int(h/2))
    
    # x1,y1 = x+int(w/2), 240
    # tọa độ điểm 2
    # x2,y2 = x+int(w/2), y+int(h/2)
    #distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    kc = math.sqrt(((x+int(w/2)) - (x+int(w/2)))**2 + ((y+int(h/2)) - 240)**2)
   
    return kc
def main():
    # Running real time from webcam
    cap = cv2.VideoCapture(1)
    model = YOLO('nckh/nckhDetects/best.pt')
    fire_cascade = cv2.CascadeClassifier("nckh/nckhDetects/fire_detection.xml")
    # Reading the classes
    classnames = ['fire']
    
    x1,y1,w1,h1 = 0,0,0,0
    check_fire = False
    x,y,w,h = 0,0,0,0
   
    while True:
        ret,image = cap.read()
        drawxy(image, x, y, w, h)
        #image = cv2.resize(image,(640,480))
        
        result = model(image,stream=True)
        fire = fire_cascade.detectMultiScale(image, 1.15, 1)
        
        s = 0
        for x, y, w, h in fire:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            s = kcx(x, y, w, h)
            # print("kc x:", kcx(x, y, w, h))
            # print("kc y:", kcy(x, y, w, h))
           
        if s != 0:
            drawKc(image, x, y, w, h)
            check_fire = True    
             
        if(check_fire == True): 
            # Getting bbox,confidence and class names informations to work with
            for info in result:
                boxes = info.boxes
        
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    print("condi:", confidence)
                    if confidence > 50:
                        x1,y1,w1,h1 = box.xyxy[0]
                        
                        x1, y1, w1, h1 = int(x1),int(y1),int(w1),int(h1)
                        cv2.rectangle(image,(x1,y1),(w1,h1),(0,0,255),5)
                        w1 = w1 - x1
                        h1 = h1 - y1
 
                        print('tdxywh:',x1,y1,w1,h1)
                        print(kcx(x1,y1,w1,h1)) 
                        cvzone.putTextRect(image, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],scale=1.5,thickness=2) 
                            
                       
                    else:
                        check_fire = False
      
        if(w1 != 0 | h1 != 0):
            drawKc(image,x1,y1,w1,h1)   
            print('kc x:',kcx(x1,y1,w1,h1))
            print('kc y:',kcy(x1,y1,w1,h1))        
            
            
                    
        cv2.imshow('image',image)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Close")
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

from ultralytics import YOLO
import cvzone
import cv2
import math
import time
import serial

def drawKc(img, x,y,w,h):
    cv2.line(img, (320,y+int(h/2)), (x+int(w/2), y+int(h/2)), (0,255,0), 2) # 
    cv2.line(img, (x+int(w/2), 240), (x+int(w/2), y+int(h/2)), (0,255,0), 2) # 
def handlerAndSendToSignal(image, x,y,w,h):
    goc_quay_Px, goc_quay_Py = 0, 0
    direction = "null"
    if x != 0 and y != 0:
        Px = kcx(x, y, w, h)
        Py = kcy(x, y, w, h)
        if(Py > 12 or Px > 12):
            print("PixelX", Px)
            print("PixelY", Py)
            if x > 320 and y > 240:  # Phải/Dưới
                direction = "00"
            elif x > 320 and y < 240:  # Phải/Trên
                direction = "01"
            elif x < 320 and y > 240:  # Trái/Dưới
                direction = "10"
            elif x < 320 and y < 240:  # Trái/Trên
                direction = "11"
            
            goc_quay_Px = int((Px / 3.2) * 10)
            goc_quay_Py = int((Py / 4.3) * 10)
                
            # if(x > 320 or y > 240):
            #     goc_quay_Px = int((Px / 3.2) * 10)
            #     goc_quay_Py = int((Py / 3.2) * 10)
            # elif x < 320 or y < 240:
            #     goc_quay_Px = int((Px / 3.2) * 10)
            #     goc_quay_Py = int((Py / 3.2) * 10)
        elif Px < 12 and Py < 12:
            direction = "OK"
    drawKc(image, x, y, w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)  # ve hinh chu nhat quanh contours
    print(goc_quay_Px,":", goc_quay_Py ,":", direction)
    return str(goc_quay_Px), str(goc_quay_Py), direction

def drawxy(img):
    cv2.line(img, (320,0), (320, 480), (221, 222, 217), 1) # dọc  y
    cv2.line(img, (0,240), (640, 240), (221, 222, 217), 1) # ngang x
    
    cv2.putText(img, "x" , (620 , 230), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, "y" , (330 , 20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 1, cv2.LINE_AA)

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

def YoloDetect(result):
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            if confidence > 70:
                print("FIRE")
                return 1
               
        print("NO FIRE")
        return 0

    #return x1,y1,w1,h1
def main():
    # Running real time from webcam
    ser = serial.Serial("COM3", 9600)
    cap = cv2.VideoCapture(1)
    model = YOLO('nckhUptoGit/nckhDetects/fire.pt')
    fire_cascade = cv2.CascadeClassifier("nckhUptoGit/nckhDetects/fire_detection.xml")
    # Reading the classes
    img = cap.read()
    result = model(img,stream=True)
    x,y,w,h = 0,0,0,0
    
    last_send_time = time.time()
    while True:
        ret,image = cap.read()
        drawxy(image)
        #image = cv2.resize(image,(640,480))
        
            
        
        fire = fire_cascade.detectMultiScale(image, 1.05, 3)
        for x, y, w, h in fire:
            print("ok")
            cv2.rectangle(image, (x - 10, y - 10), (x + (w+10), y + (h+10)), (255, 0, 0), 1)
            
        current_time = time.time()
        if current_time - last_send_time >= 1.5 and  w != 0:  # 0.5 giây = 500ms
            
            x1 = x + w + 20
            y1 = y + h + 20
            image_cut = image[y - 20:y1, x - 20:x1] # cắt ảnh
            image_cut = cv2.addWeighted(image_cut, 1, image_cut, 0, -75) #giảm độ sáng
            height, width, channels = image_cut.shape
            if(height > 0 and width > 0):
                cv2.imshow("cut", image_cut)
                result = model(image_cut,stream=True)
                if YoloDetect(result) == 1:
                    PackageGocPx, PackageGocPy, PackageDirection = handlerAndSendToSignal(image, x,y,h,w)
                    str = PackageGocPx + "\n" + PackageGocPy + "," + PackageDirection + "."  
                    if(PackageDirection != "null" or PackageDirection != "OK"):
                        print("hi", str)
                        ser.write(str.encode())
                        
                        
            x,y,w,h = 0,0,0,0           
            last_send_time = current_time  
                    
                    
        cv2.imshow('image',image)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Close")
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

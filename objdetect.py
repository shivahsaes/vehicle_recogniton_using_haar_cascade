

import cv2

cascade_src = 'cars.xml'
cascade_src1 = 'two_wheeler.xml'
cascade_src2 = 'Bus_front.xml'
cascade_src3 = 'pedestrian.xml'


video_src = 'sam.mp4'

cap = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier(cascade_src)
bike_cascade= cv2.CascadeClassifier(cascade_src1)
bus_cascade= cv2.CascadeClassifier(cascade_src2)
ped_cascade= cv2.CascadeClassifier(cascade_src3)


while True:
    ret, img = cap.read()
   
    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray,1.1, 2)
    bikes= bike_cascade.detectMultiScale(gray,1.19,1)
    buses= bus_cascade.detectMultiScale(gray,1.16,1)
    peds= ped_cascade.detectMultiScale(gray,1.3,2)
    
    


    for (b1x,b1y,b1w,b1h) in buses:
        cv2.rectangle(img,(b1x,b1y),(b1x+b1w,b1y+b1h),(0,150,255),2)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,'bus',(b1x,b1y),font,1,(255,255,255),3,cv2.LINE_8)
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,150),2)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,'car',(x,y),font,1,(255,255,255),3,cv2.LINE_8)
    for (bx,by,bw,bh) in bikes:
        cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(0,0,0),2)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,'two_wheeler',(bx,by),font,1,(255,255,255),3,cv2.LINE_8)
    for (px,py,pw,ph) in peds:
        cv2.rectangle(img,(px,py),(px+pw,py+ph),(0,0,255),2)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,'person',(px,py),font,1,(255,255,255),3,cv2.LINE_8)   
    img=cv2.resize(img,(1000,700))
    cv2.imshow('video', img)
    
   
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()

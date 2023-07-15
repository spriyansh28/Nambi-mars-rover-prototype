import cv2
#CP PLUS 192.168.0.250 admin Teaminferno
cap = cv2.VideoCapture() 
cap.open("rtsp://admin:Teaminferno@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0")

def rescale_frame(frame, percent=75):
    width = 1080
    height = 720
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while(True):
     # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    frame75 = rescale_frame(frame, percent=50)
    cv2.imshow('CAM2',frame75)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture 
cap.release()
cv2.destroyAllWindows()

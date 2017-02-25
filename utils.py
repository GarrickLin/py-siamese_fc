import cv2

clicked = False
P1 = (0, 0)
P2 = (0, 0)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['clicked'] = True
        param['P1'] = (x, y)
        param['P2'] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['clicked']:
            param['P2'] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        param['p2'] = (x, y)
        param['clicked'] = False

class GetRectange:
    def __init__(self):
        self.onClick = False
        
    def getRect(self, img):
        winname = "get rect"       
        param = {}
        param['P1'] = (0, 0)
        param['P2'] = (0, 0)
        param['clicked'] = False
        cv2.imshow(winname, img)
        cv2.setMouseCallback(winname, onMouse, param)
        
        while 1:
            key = cv2.waitKey(1)
            if param['clicked']:
                img_clone = img.copy()
                cv2.rectangle(img_clone, param['P1'], param['P2'], (0, 255, 0))
                cv2.imshow(winname, img_clone)
                self.onClick = True
            if self.onClick and not param['clicked']:
                self.onClick = False
                left = min(param['P1'][0], param['P2'][0])
                right = max(param['P1'][0], param['P2'][0])
                top = min(param['P1'][1], param['P2'][1])
                dowm = max(param['P1'][1], param['P2'][1])
                cv2.destroyWindow(winname)
                h = dowm-top+1
                w = right-left+1
                return (top+h/2., left+w/2.),  (h, w)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    rector = GetRectange()
    while 1:
        ready, frame = cap.read()
        if not ready:
            print "device", device, "is not ready"
        cv2.imshow("frame", frame)        
        key = cv2.waitKey(1)
        if key != -1:
            break    
    rect = rector.getRect(frame)
    print rect
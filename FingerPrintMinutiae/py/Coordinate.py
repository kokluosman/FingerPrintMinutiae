import cv2
  
# Tıklanılan Yerin X,Y Koordinatlarını Gösteren Fonksiyon Tanımlandı.
def click_event(event, x, y, flags, params):
 
    # Sol Tık Mouse
    if event == cv2.EVENT_LBUTTONDOWN:
 
        
        print(x, ' ', y)
 
        # Resim Üzerinde Göster
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

 

if __name__=="__main__":

    # Girilen Resim
    img = cv2.imread('FPMMinutiae.png', 1)
    cv2.imshow('image', img)  
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
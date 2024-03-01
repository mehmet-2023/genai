import cv2
import numpy as np
import pyautogui

# Başlangıç konumları
primary_point = (0, 0)
secondary_point = (0, 0)

# Kamera başlatma
pyautogui.PAUSE = 0
cap = cv2.VideoCapture(1)

# Hareket eşiği ve kaydırma durumu
movement_threshold = 50
angle_threshold = 30  # Derece cinsinden
is_sliding = False

# Kaydırma parametreleri
scroll_speed = 160  # Kaydırma hızı (piksel/saniye)
same_distance_count_threshold = 10  # Aynı uzaklık sayacı eşiği

same_distance_count = 0  # Aynı uzaklık sayacı

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV renk uzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # El rengini belirleme (örneğin, ten rengi)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # El rengi için bir maske oluştur
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Gürültüyü azaltmak için biraz yumuşatma uygula
    mask = cv2.medianBlur(mask, 5)

    # El bölgesini belirleme
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # En büyük konturu seçme
        max_contour = max(contours, key=cv2.contourArea)

        # El bölgesinin merkezini hesapla
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if primary_point == (0, 0):
                # İlk başta ana nokta atanır
                primary_point = (cx, cy)
            else:
                # Hareket algılama
                movement = np.sqrt((cx - primary_point[0]) ** 2 + (cy - primary_point[1]) ** 2)

                if not is_sliding and movement > movement_threshold:
                    is_sliding = True
                    secondary_point = (cx, cy)
                    same_distance_count = 0  # Aynı uzaklık sayacını sıfırla
                    if cx > primary_point[0]:
                        print("Sağa kaydırma")
                        pyautogui.hscroll(scroll_speed)
                    elif cx < primary_point[0]:
                        print("Sola kaydırma")
                        pyautogui.hscroll(-scroll_speed)
                elif is_sliding:
                    # İki nokta arası mesafenin değişimini kontrol et
                    current_distance = np.sqrt((cx - primary_point[0]) ** 2 + (cy - primary_point[1]) ** 2)
                    if abs(current_distance - movement) < movement_threshold:
                        same_distance_count += 1
                    else:
                        same_distance_count = 0

                    # Aynı uzaklık sayacı eşiği aşıldığında işlemi durdur
                    if same_distance_count > same_distance_count_threshold:
                        is_sliding = False

    else:
        # El algılanmadığında noktaları sıfırla
        primary_point = (0, 0)
        secondary_point = (0, 0)

    # Noktaları ve aradaki çizgiyi çizme
    cv2.circle(frame, primary_point, 5, (0, 255, 0), -1)
    if secondary_point != (0, 0):
        cv2.circle(frame, secondary_point, 5, (0, 0, 255), -1)
        cv2.line(frame, primary_point, secondary_point, (0, 0, 255), 2)  # Çizgi

    # Sonuçları gösterme
    cv2.imshow('frame', frame)

    # İki noktanın da üstü kapanırsa alt+f4 tuşlarına basılmasını algıla
    if primary_point == (0, 0) and secondary_point == (0, 0):
        pyautogui.hotkey('alt', 'f4')
        break

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
import cv2

def main():
    # Yüz tanıma sınıflandırıcısını yükleyin
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Kamera bağlantısını başlatın
    cap = cv2.VideoCapture(0)

    # Önceki kişi sayısını saklayın
    prev_num_faces = 0

    while True:
        # Görüntüyü yakalayın
        ret, frame = cap.read()

        # Gri tonlamaya dönüştürün
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Yüzleri algılayın
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Kişi sayısını kontrol edin ve konsola yazdırın
        num_faces = len(faces)
        if num_faces != prev_num_faces:
            if num_faces > 0:
                print(f"Yeni bir kişi algılandı! Toplam kişi sayısı: {num_faces}")
            else:
                print("Kimse algılanmadı!")
        prev_num_faces = num_faces

        # Yüzleri kutu içine alın
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Görüntüyü ekrana gösterin
        cv2.imshow('Yüz Algılama', frame)

        # Çıkış için q tuşuna basılıp basılmadığını kontrol edin
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kamera bağlantısını ve pencereleri kapatın
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

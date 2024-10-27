import cv2
import mediapipe as mp
import numpy as np

# تنظیمات MediaPipe برای تشخیص دست
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


# تابع برای افزایش روشنایی تصویر
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright

image = cv2.imread(r'C:\Users\Ali\PycharmProjects\MachineLearning\1.jpg')

# شروع پردازش ویدیو از وبکم
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # مختصات انگشت اشاره
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # مختصات انگشت شست
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # بررسی فاصله بین انگشت شست و انگشت اشاره (برای مثال اگر انگشت اشاره حرکت کرد)
            thumb_index_distance = np.sqrt(
                (thumb_tip.x - index_finger_tip.x) ** 2 +
                (thumb_tip.y - index_finger_tip.y) ** 2
            )

            # اگر فاصله کمتر از مقدار مشخصی بود، یعنی انگشتان به هم نزدیک‌اند
            if thumb_index_distance < 0.1:
                # افزایش روشنایی تصویر
                bright_image = increase_brightness(image, value=50)

                # نمایش تصویر روشن‌شده
                cv2.imshow("Bright Image", bright_image)
            else:
                # نمایش تصویر اصلی
                cv2.imshow("Bright Image", image)

    # نمایش فریم دوربین
    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

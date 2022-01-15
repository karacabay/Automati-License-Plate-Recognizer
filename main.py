import time
import cv2 as cv
import imutils
import numpy as np
import pytesseract
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import requests


def get_data():
    url = 'http://akilliotopark.pythonanywhere.com/butun-plakalar'
    response = requests.request('GET', url=url)
    response = response.text
    datalist = response.split('---')
    datalist = [s.split('***') for s in datalist]
    plakas = [d[2] for d in datalist]
    return datalist, plakas


cap = cv.VideoCapture(0)
chars = '01234567890ABCDEFGHIJKLMNOPRSTUVYZ'

xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]

datalist, plakas = get_data()


def rect_detector(cnt):
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        (_, _, w, h) = cv.boundingRect(approx)
        if h * 2.5 < w:
            return True
    return False


plaka_list = []

while True:

    fps = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    cv.imwrite('1-OriginalFrame.jpg', frame)
    frame = cv.flip(frame, 1)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    table = np.interp(np.arange(256), xp, fp).astype('uint8')
    frame_gray_contrast = cv.LUT(frame_gray, table)
    blurred = cv.GaussianBlur(frame_gray_contrast, (5, 5), 0)
    th = cv.threshold(frame_gray, 100, 255, cv.THRESH_BINARY)[1]
    dilated = cv.erode(th, np.ones((3, 3)), iterations=3)
    contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(frame)
    cv.drawContours(mask, contours, -1, (0, 255, 0), thickness=2)
    contour_areas = [cv.contourArea(cnt) for cnt in contours]
    sorting_index = np.argsort(contour_areas)
    contours = (np.array(contours))[sorting_index]
    hierarchy = (np.array(hierarchy)[0][sorting_index]).reshape((1, -1, 4))
    contour_areas = [cv.contourArea(cnt) for cnt in contours if cv.contourArea(cnt) < 4000]
    hierarchy = hierarchy[len(contour_areas):]
    contours = contours[len(contour_areas):]
    result = False
    for i, c in enumerate(imutils.grab_contours((contours, hierarchy))):
        if rect_detector(c):
            rect = cv.minAreaRect(contours[i])
            pts1 = np.float32(cv.boxPoints(rect))
            if pts1[3, 0] < pts1[1, 0]:
                pts1 = np.roll(pts1, 1, axis=0)
            pts1_copy = pts1.copy()

            if pts1[0, 0] > 20: pts1[0, 0] -= -30
            if pts1[1, 0] > 20: pts1[1, 0] -= -30
            if pts1[2, 0] < 620: pts1[2, 0] += 15
            if pts1[3, 0] < 620: pts1[3, 0] += 15

            if pts1[0, 1] < 460: pts1[0, 1] += 10
            if pts1[1, 1] > 20: pts1[1, 1] -= 10
            if pts1[2, 1] > 20: pts1[2, 1] -= 10
            if pts1[3, 1] < 460: pts1[3, 1] += 10

            mask = frame.copy()
            for p in pts1:
                cv.circle(mask, np.uint16(p), 4, (0, 0, 255), thickness=-1)

            pts2 = np.float32([[400, 0], [400, 100], [0, 100], [0, 0]])
            matrix = cv.getPerspectiveTransform(pts1, pts2)
            result = cv.warpPerspective(frame, matrix, (400, 100))
            result = cv.flip(result, 0)
            result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            th = cv.threshold(result_gray, 100, 255, cv.THRESH_BINARY)[1]
            dilated = cv.erode(th, np.ones((3, 3)), iterations=1)
            contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key=cv.contourArea)
            mask = np.zeros_like(result_gray)
            cv.drawContours(mask, [cnt], 0, (255), thickness=-1)
            result_gray = cv.bitwise_and(result_gray, result_gray, mask=mask)
            thresh = cv.adaptiveThreshold(result_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 10)
            cv.imwrite('18-Result.png', thresh)

            break

    if type(result) != bool:
        config = f"-c tessedit_char_whitelist={chars}"
        result_plaka = pytesseract.image_to_string('18-Result.png', config=config)
        result_plaka = result_plaka.replace(' ', '').strip()

        if 9 > len(result_plaka) > 6:
            if not result_plaka[:3].isdigit() and result_plaka[:2].isdigit():
                plaka_list.append(result_plaka)

        if len(plaka_list) == 20:
            print(plaka_list)
            sonuc = max(set(plaka_list), key=plaka_list.count)
            print(sonuc)
            info_screen = np.zeros_like(frame)
            plaka_list = []
            if sonuc in plakas:
                index = plakas.index(sonuc)
                name = datalist[index][0]
                surname = datalist[index][1]

                txt1, txt2, txt3 = "Hosgeldiniz", f'{name} {surname}', sonuc
                color = (0, 255, 0)

            else:
                txt1, txt2, txt3 = sonuc, "Giris izniniz", "Bulunmamaktadir!"
                color = (0, 0, 255)

            cv.putText(info_screen, txt1, (170, 100), cv.FONT_HERSHEY_SIMPLEX, 1.3, color, thickness=2)
            cv.putText(info_screen, txt2, (170, 200), cv.FONT_HERSHEY_SIMPLEX, 1.3, color, thickness=2)
            cv.putText(info_screen, txt3, (170, 300), cv.FONT_HERSHEY_SIMPLEX, 1.3, color, thickness=2)

            cv.imwrite("output.jpg", info_screen)
            cv.imshow('Frame', info_screen)
            cv.waitKey(5000)

    cv.putText(frame, f'FPS : {int(1 / (time.time() - fps))}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0),
               thickness=2)
    cv.imshow('Frame', frame)

    if cv.waitKey(1) & 0xff == 27:
        break

cap.release()
cv.destroyAllWindows()

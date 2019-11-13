import cv2 as cv
import numpy as np

base_path = './opencv/data/'
'''使用摄像头
'''
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print('cannot open camera')
#     exit()
# while True:
#     ret, frame = cap.read() # ret标志，确认帧是否正确
#     if not ret:
#         print('cannot receive frame, exitting...')
#         break
#     cv.imshow('frame', frame)
#     if cv.waitKey(1) == ord('q'): # 此处用1的原因在于外面是while True
#         break
# cap.release() # 释放资源
# cv.destroyAllWindows()

'''play video
'''
# cap = cv.VideoCapture(base_path + '1.flv')
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print('cannot receive frame')
#         break
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv.destroyAllWindows()

'''Save video. Ok, it works well.
'''
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter(base_path + 'output.avi', fourcc, 20.0, (640, 480)) # 参数fps, frame size
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('cannot receive frame')
        break
    frame = cv.flip(frame, 0)
    out.write(frame)

    # cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
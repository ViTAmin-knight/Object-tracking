from AIDetector_pytorch import Detector
import imutils
from tracker import update_tracker
import cv2
import torch

print(torch.cuda.is_available())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', device)



def main():
    name = 'demo'
    det = Detector()
    cap = cv2.VideoCapture('D:/pythonproject/Yolov5-Deepsort-main/data/videos/european street.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('fps:', fps)
    t = int(1000/fps)
    videoWriter = None
    all_track_points = []  # 存储所有检测到的轨迹点的列表

    while True:
        _, im = cap.read()
        if im is None:
            break

        # 调用update_tracker函数获取更新后的图像、新人脸和轨迹点
        result, new_faces, track_points = update_tracker(det, im)


        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter('result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        # 将检测到的轨迹点添加到列表中
        all_track_points.extend(track_points)

        # 绘制轨迹线
        for points in all_track_points:
            for i in range(1, len(points)):
                cv2.line(result, points[i - 1][:2], points[i][:2], (255, 0, 0), 2)

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

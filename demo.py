import time

import cv2
import imutils

from AutoDetector import Detector

def create_video_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f'无法创建视频写入器: {output_path}')
    return writer


def main():
    cap = None  # 给cap初始值，避免异常时未赋值
    try:
        name = 'demo'
        start = 0  # 给初始值，避免异常时未赋值
        det = Detector()
        det.init_model(weight='yolo11x-seg.pt', detect_type='segment')
        # det.init_model(weight='yolo11x.pt', detect_type='object')

        cap = cv2.VideoCapture(r'C:\Users\admin\Desktop\test.mp4')
        fps = int(cap.get(5))  # 获取视频帧率
        all_frames = cap.get(7)  # 获取所有帧数量
        # print('fps:', fps)
        if fps == 0:
            fps = 30.0
        t = int(1000 / fps)
        start = time.time()

        videoWriter = None
        frame = 1

        while True:
            inner_start = time.time()
            # try:
            ret, im = cap.read()
            if ret is False or im is None:
                break

            result = det.feedCap(im)
            result = result['frame']
            result = imutils.resize(result, height=500)
            if videoWriter is None:
                videoWriter = create_video_writer(
                    'result.mp4', fps, (result.shape[1], result.shape[0])
                )

            end = time.time()
            print('total time: ', end - inner_start)
            print(frame, '/', all_frames, ' fps: ', frame / (end - start))
            frame += 1

            videoWriter.write(result)
            cv2.imshow(name, result)
            cv2.waitKey(t)

            if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break
            # except Exception as e:
            #     print(e)
            #     break
    finally:
        print('total time: ', time.time() - start)
        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()
    print('total time: ', time.time() - start)
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

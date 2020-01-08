import cv2
from demo_test import process_image
from notebooks import visualization
from time import sleep


class Detertor(object):

    def __init__(self, camera_index=0):

        self.camera_index = camera_index

    def Catch_Video(self, window_name='Detertor'):

        cap = cv2.VideoCapture(self.camera_index)

        if isinstance(self.camera_index, str):
            flag = True
        else:
            flag = False

        if not cap.isOpened():

            raise Exception('Check if the camera is on.')

        while cap.isOpened():

            catch, frame = cap.read()  # 读取每一帧图片

            if not catch:

                raise Exception('Check if the camera is on.')

                break

            rclasses, rscores, rbboxes = process_image(frame)  # 这里传入图片

            labeled_img = visualization.bboxes_draw_on_img(
                frame, rclasses, rscores, rbboxes, visualization.colors_plasma)

            if flag:
                sleep(0.1)

            cv2.imshow(window_name, labeled_img)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        # 释放摄像头

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    detect = Detertor(camera_index=0)

    detect.Catch_Video()

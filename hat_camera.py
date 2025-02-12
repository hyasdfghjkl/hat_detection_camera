import cv2
from ultralytics import YOLO
import time


def video_detection(model_path, source=0, output_size=(1280, 720)):
    """
    实时视频流检测
    :param model_path: 训练好的模型路径
    :param source: 视频源（0-摄像头，或视频文件路径）
    :param output_size: 输出画面尺寸
    """
    # 加载模型
    model = YOLO(model_path)

    # 打开视频源
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, output_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, output_size[1])

    # FPS计算参数
    prev_time = 0
    curr_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 执行推理（启用半精度，优化显存）
        results = model.predict(
            frame,
            imgsz=416,  # 适当降低分辨率
            conf=0.5,
            device=0,  # 使用GPU
            half=True,  # 启用半精度
            verbose=False
        )

        # 绘制检测结果
        annotated_frame = results[0].plot(line_width=2)

        # 计算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 显示FPS
        cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow("Helmet Detection", annotated_frame)

        # 退出检测
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 使用示例
    video_detection(
        model_path="runs/detect/train11/weights/best.pt",
        source="D:/job_hunting/Hat_test/video_no_helmet.mp4",  # 0-默认摄像头，或替换为视频路径
        output_size=(1280, 720)
    )
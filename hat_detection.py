import argparse
from ultralytics import YOLO
import os
import torch

def train_model(data_yaml, model_path, epochs, imgsz, batch, device):
    """
    训练头盔检测模型
    :param data_yaml: 数据集配置文件路径
    :param model_path: 预训练模型路径
    :param epochs: 训练轮数
    :param imgsz: 输入图像大小
    :param batch: 批量大小
    :param device: 使用的设备（cpu 或 gpu）
    """
    # 显存优化预处理
    torch.cuda.empty_cache()  # 清理显存缓存

    # 确保模型路径存在
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，将自动下载预训练模型。")
        model = YOLO("yolov8s.pt")  # 自动下载预训练模型
        model.save(model_path)  # 保存模型到指定路径
    else:
        print(f"使用本地模型文件 {model_path}。")

    # 加载模型
    model = YOLO(model_path)

    # 训练模型
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        # workers=0,  # 启用多线程数据加载
        workers=4,  # 原为0，根据CPU核心数调整
        # lr0=0.01,          # 初始学习率
        # lrf=0.001,          # 最终学习率（cosine衰减）
        lr0=0.001,        # 降低初始学习率
        lrf=0.0005,       # 更平缓的衰减
        # warmup_epochs=5,   # 学习率预热
        warmup_epochs=10, # 延长学习率预热
        # momentum=0.937,    # 动量参数
        momentum=0.9,        # 适当降低动量
        # weight_decay=0.0005, # 权重衰减
        weight_decay=0.001,  # 原0.0005，增强权重衰减
        dropout=0.2,         # 新增dropout正则化
        box=7.5,           # 调整定位损失权重
        cls=0.7,           # 调整分类损失权重
        dfl=1.5,           # 调整分布焦点损失
        # close_mosaic=10,   # 提前关闭Mosaic增强
        close_mosaic=15  # 原为10，增加5个epoch
    )

    print("模型训练完成！")

def predict(model_path, source):
    """
    使用训练好的模型进行预测
    :param model_path: 训练好的模型路径
    :param source: 测试图像路径
    """
    # 加载模型
    model = YOLO(model_path)

    # 进行预测
    results = model(source)

    # 显示结果
    results.show()


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="头盔检测模型训练和预测")
    parser.add_argument("--device", type=str, default="0", help="使用GPU设备编号")
    parser.add_argument("--data", type=str, default="data.yaml", help="数据集配置文件路径")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="预训练模型路径")
    parser.add_argument("--batch", type=int, default=-1, help="批量大小")  # 增大batch size（需根据GPU显存调整）
    parser.add_argument("--imgsz", type=int, default=416, help="输入图像大小")  # 减小输入尺寸
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")  # 合理减少训练轮次
    parser.add_argument("--predict", type=str, default=None, help="测试图像路径（可选）")

    # 解析命令行参数
    args = parser.parse_args()

    # 训练模型
    train_model(
        data_yaml=args.data,
        model_path=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

    # 如果提供了测试图像路径，则进行预测
    if args.predict:
        predict(model_path="runs/detect/train/weights/best.pt", source=args.predict)
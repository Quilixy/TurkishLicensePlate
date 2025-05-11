from ultralytics import YOLO
import multiprocessing
import torchvision
from torchvision.ops import nms


def main():
    model = YOLO('yolo11n.pt')
    model.train(data='dataset/data.yaml', epochs=75, imgsz=640, device=0)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

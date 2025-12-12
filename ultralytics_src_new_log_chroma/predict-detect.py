from ultralytics import YOLO
# Load a model

model = YOLO(r"./yolo11n.pt",task='detect')  # load a pretrained model (recommended for training)

model.predict(source = r"D:\1118临时\spectral ratio copy\spectral ratio copy\model\Huawei_model\ultralytics_src\data\log_chroma", imgsz=640, save=True)


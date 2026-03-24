import os
from ultralytics import YOLO
import numpy as np

def eval():
    dataset_path = '/dataset'
    model_path = '/model/model.pt'
    
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml nao encontrado em {data_yaml}")

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml, 
        split='val',
        batch=16,
        imgsz=640,
        device=0,
        verbose=True
    )
    
    
    print("\nMétricas:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print("\nClassAccuracy (F1 Score):")
    avg_class_acc = np.mean(metrics.box.f1)
    print(f"Acurácia Média: {avg_class_acc:.4f}")


if __name__ == "__main__":
    eval()
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from super_gradients.training import models
from torchvision import transforms as T
from ultralytics import YOLO
from ensemble_boxes import nms
from PIL import Image
import torchvision
import numpy as np
import torch
import cv2
import concurrent.futures


# GIVE IMAGE PATH TO GET ALL THE PREDICTIONS OF MODELS - CHECK output_all_models folder



image_path = r"D:\PROJECTS\OBJECT_DETECTION\data\Predict\DJI_0487.jpg"


#----------------------------------------------------------

def draw_boxes_save(media_path,detections,name):

    image = Image.open(media_path)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Loop through the coordinates and draw bounding boxes
    for coord in detections:
        x_min, y_min, x_max, y_max = map(int, coord)
        cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        label = "plastic"
        cv2.putText(image_cv, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imwrite(name, image_cv)


def run_parallel_functions(image_path):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_rcnn = executor.submit(predict_rcnn, image_path)
        future_yolov8 = executor.submit(predict_yolov8, image_path)
        future_yolonas = executor.submit(predict_yoloNAS, image_path)

        faster_rcnn_preds, rcnn_scores = future_rcnn.result()
        yolov8_preds, yolov8_scores = future_yolov8.result()
        yolovnas_preds, nas_scores = future_yolonas.result()

    return faster_rcnn_preds, rcnn_scores, yolov8_preds, yolov8_scores, yolovnas_preds, nas_scores


def predict_rcnn(image_path,score_threshold=0.9,device='cpu'):
    # Define and load your model architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    model_path = "models/single_class_models/Singleclass_Faster_RCNN_125.pth"
    # Load the saved model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    # Load the model state and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    model.eval()

    # Load and process the image
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
    
    boxes = output[0]['boxes'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()
    labels = output[0]['labels'].cpu().detach().numpy()

    # Filter out boxes with scores less than the threshold
    mask = scores >= score_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]

    
    draw_boxes_save(image_path,filtered_boxes,"output_all_models/fasterRcnn.jpg")

    return filtered_boxes, filtered_scores


def predict_yolov8(image_path):
    
    modelyolo = YOLO('models/single_class_models/YoloV8.onnx')

    results = modelyolo(source=image_path,conf=0.3,iou=0.5)

    for result in results:
        
        boxes = result.boxes.xyxy
        scores = result.boxes.conf

    yolov8_preds = np.array(boxes.tolist())
    yolov8_scores = scores
    print("yolov8",yolov8_preds)

    draw_boxes_save(image_path,yolov8_preds,"output_all_models/YoloV8.jpg")
    return yolov8_preds,yolov8_scores


def predict_yoloNAS(image_path,conf=0.45,iou=0.7):
    
    modelyoloNas = models.get('yolo_nas_l',
                        num_classes=1,
                        checkpoint_path="models/single_class_models/YoloNAS.pth")
    
    # yolonas = best_model.predict(image_path,conf=0.4,iou=0.7).show()
    yolonas = modelyoloNas.predict(image_path,conf=0.45,iou=0.7)

    for result in yolonas:
        yolovnas_preds = result.prediction.bboxes_xyxy
        nas_score = result.prediction.confidence
    draw_boxes_save(image_path,yolovnas_preds,"output_all_models/YoloNAS.jpg")

    return yolovnas_preds,nas_score


def normalize_bounding_boxes(bounding_boxes, image_width, image_height):
    normalized_boxes = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        normalized_x1 = x1 / image_width
        normalized_y1 = y1 / image_height
        normalized_x2 = x2 / image_width
        normalized_y2 = y2 / image_height
        normalized_boxes.append([normalized_x1, normalized_y1, normalized_x2, normalized_y2])
    return normalized_boxes

def denormalize_bounding_boxes(normalized_boxes, image_width, image_height):
    denormalized_boxes = []
    for box in normalized_boxes:
        x1, y1, x2, y2 = box
        denormalized_x1 = x1 * image_width
        denormalized_y1 = y1 * image_height
        denormalized_x2 = x2 * image_width
        denormalized_y2 = y2 * image_height
        denormalized_boxes.append([denormalized_x1, denormalized_y1, denormalized_x2, denormalized_y2])
    return denormalized_boxes


def ensemble_singleClass(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape


    faster_rcnn_preds, rcnn_scores, yolov8_preds, yolov8_scores, yolovnas_preds, nas_scores = run_parallel_functions(image_path)
    
    
    # Normalize bounding boxes for each model
    boxes_list = [normalize_bounding_boxes(preds, image_width, image_height) for preds in [faster_rcnn_preds, yolov8_preds, yolovnas_preds]]

    # Convert scores to lists
    scores_list = [rcnn_scores.tolist(), yolov8_scores.tolist(), nas_scores.tolist()]

    # Initialize labels_list with zeros
    labels_list = [[0] * len(scores) for scores in scores_list]
    
    bbox, conf, lab = nms(boxes_list, scores_list, labels_list, iou_thr= 0.3)
    
    bboxs = denormalize_bounding_boxes(bbox,image_width,image_height)

    draw_boxes_save(image_path,bboxs,"output_all_models/ensembled.jpg")
    return bboxs, conf, lab


ensemble_singleClass(image_path)

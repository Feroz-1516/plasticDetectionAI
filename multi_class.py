from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
import torchvision
import numpy as np
import torch
from PIL import Image


def predict_rcnn_multi(image_path,score_threshold=0.5,device='cpu'):
    # Define and load your model architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)

    model_path = "models/multi_class_models/Multiclass_Faster_RCNN_160.pth"
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
    filtered_lab = labels[mask]
    
    
    # Define the mapping dictionary
    mapping_dict = {
        'Plastic': 0,
        'Plastic Bag': 1,
        'Plastic Bottle': 2,
        'Plastic Cup': 3,
        'Plastic Plate': 4,
        'Plastic Spoon': 5,
        'Plastic Wrapper': 6,
        'Plastic cup': 7,
        'Slippers': 8,
        'Thermocol': 9
    }

    # draw_boxes_on_image(image_path, predicted_boxes, predicted_scores, predicted_labels,save_path)
    filtered_lab = filtered_lab.tolist()
    mapped_names = [key for value in filtered_lab for key, mapped_value in mapping_dict.items() if mapped_value == value]
    mapped_names = np.array(mapped_names)
    
    
    return filtered_boxes, filtered_scores ,mapped_names





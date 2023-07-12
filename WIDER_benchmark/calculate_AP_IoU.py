from shapely.geometry import box
from sklearn.metrics import fbeta_score
import numpy as np

class FileData:
    def __init__(self, name, length, coordinates):
        self.name = name
        self.length = length
        self.coordinates = coordinates

    def __str__(self):
        return f"Name: {self.name}, Length: {self.length}, Coordinates: {self.coordinates}"

def aux_image_index(file_path):
    #Aux function that returns the name and the len of this specific input txt
    images = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines) - 1):
            line = lines[i].strip()
            next_line = lines[i+1].strip()
            if line.endswith('.jpg'):
                images.append((line, int(next_line)))
    return images

def parse_file(file_path):
    #create a list data of File data objects with name, len and coordinates
    data = []
    shift = 2
    images_index = aux_image_index(file_path)
    last_index = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for image_name, image_len in images_index:
            coordinates = []  # Create a new list for each FileData object
            for i in range((last_index+shift), (last_index+image_len+shift)):
                entry = lines[i]
                coordinates.append([int(x) for x in entry.split()[:4]])
            file_data = FileData(image_name, image_len, coordinates)
            last_index = last_index+image_len+shift
            data.append(file_data)
    return data

def aux_match_image(predicted_image, parsed_gt):
    for gt_item in parsed_gt:
        if gt_item.name == predicted_image.name:#same image
            return gt_item
    print("Image not found")
    return None

def calculate_iou(box1, box2):
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    #print("intersection / union", intersection , union)
    iou = intersection/(union+1) 
    return iou

def evaluate(iou_threshold, parsed_gt, parsed_predicted):
    iou_values_list = []

    for predicted_image in parsed_predicted:
        gt_image = aux_match_image(predicted_image, parsed_gt)
        for gt_box_coordinates in gt_image.coordinates:
            gt_box = box(gt_box_coordinates[0], gt_box_coordinates[1],
                         gt_box_coordinates[2] + gt_box_coordinates[0],
                         gt_box_coordinates[3] + gt_box_coordinates[1])
            
            iou_values = []
            for file_box_coordinates in predicted_image.coordinates:
                file_box = box(file_box_coordinates[0], file_box_coordinates[1],
                               file_box_coordinates[2] + file_box_coordinates[0],
                               file_box_coordinates[3] + file_box_coordinates[1])
                iou_values.append(calculate_iou(gt_box, file_box))

            if iou_values:
                iou = max(iou_values)
            else:
                iou = 0.0

            iou_values_list.append(iou)
    y_true = np.ones(len(iou_values_list))
    y_pred = [int(iou > iou_threshold) for iou in iou_values_list]
    ap = fbeta_score(y_true, y_pred, beta=0.5)
    return ap

def main(iou_threshold=0.5, gt=None, predicted=None):
    #print_file_content(gt)
    if gt is None or predicted is None:
        print("Please provide two text files as input.")
        return
    parsed_gt = parse_file(gt)
    print("Ground Truth Parsed Sucessfully")
    parsed_predicted = parse_file(predicted)
    print("Predicted Parsed Sucessfully")
    result = evaluate(iou_threshold, parsed_gt, parsed_predicted)
    print('Average IoU = %s' % str(result))

if __name__ == '__main__':
    gt = "wider_face_val_bbx_gt.txt"
    #predicted = "test.txt"
    predicted = "images/WIDER_Baseline_easy.txt"
    #predicted = "images/WIDER_Baseline_medium.txt"
    #predicted = "images/WIDER_Baseline_hard.txt"
    iou_threshold=0.5
    main(iou_threshold, gt=gt, predicted=predicted)
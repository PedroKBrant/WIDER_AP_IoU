from shapely.geometry import box
from sklearn.metrics import fbeta_score, average_precision_score
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import re
class FileData:
    def __init__(self, name, length, coordinates):
        self.name = name
        self.length = length
        self.coordinates = coordinates
        
    def __str__(self):
        return f"Name: {self.name}, Length: {self.length}, Coordinates: {self.coordinates}"

    
def CalculateAveragePrecision(prec, rec):
    mrec = [0] + rec + [1]
    mpre = [0] + prec + [0]
    mpre.reverse()
    
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    #return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
  
def print_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)

def aux_image_index(file_path):
    #Aux function that returns the name and the len of this specific input txt
    images = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines) - 1):
            line = lines[i].strip()
            next_line = lines[i+1].strip()
            if line.endswith('.jpg') or line.endswith('.png'):
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
        if str(lines).startswith('TIME LIST'):
          return data
        for image_name, image_len in images_index:
            coordinates = []  # Create a new list for each FileData object
            for i in range((last_index+shift), (last_index+image_len+shift)):
                entry = lines[i]
                coordinates.append([float(x) for x in entry.split()[:4]])
            file_data = FileData(image_name, image_len, coordinates)
            last_index = last_index+image_len+shift
            data.append(file_data)
    return data

def aux_match_image(predicted_image, parsed_gt):
    for gt_item in parsed_gt:
        # Split file names by either '/' or '\'
        gt_parts = re.split(r'[\\/]', gt_item.name)
        predicted_parts = re.split(r'[\\/]', predicted_image.name)

        # Extract the prefixes from the file names
        gt_prefix = gt_parts[-1].split('.')[0]
        predicted_prefix = predicted_parts[-1].split('.')[0]

        # Compare the prefixes to check if the images are the same
        if(gt_prefix == predicted_prefix):
            return gt_item
    print("Image not found")
    return None
  
def calculate_TP_FP_FN(gt_image, predicted_image, iou_threshold, width_factor=1.0, height_factor=1.0):
  TP = 0 
  FP= 0 
  FN = 0
  iou_values_list = []
  iou = []
  total_faces = gt_image.length
  pred_faces = predicted_image.length
  for gt_box_coordinates in gt_image.coordinates:#left, top, width, height
    gt_box = box(gt_box_coordinates[0],  gt_box_coordinates[1],
                gt_box_coordinates[2] + gt_box_coordinates[0],
                gt_box_coordinates[3] + gt_box_coordinates[1])
  
    iou_predicteds_to_gt = [0.0]
    for predicted_box_coordinates in predicted_image.coordinates:#left, top, width, height
        predicted_box = box(predicted_box_coordinates[0]*width_factor,  predicted_box_coordinates[1]*height_factor,
                            predicted_box_coordinates[2]*width_factor + predicted_box_coordinates[0]*width_factor,
                            predicted_box_coordinates[3]*height_factor + predicted_box_coordinates[1]*height_factor)
        iou_predicteds_to_gt.append(calculate_iou_face(gt_box, predicted_box))
    #print("trying to find the best match bbox prediction", iou_predicteds_to_gt)
    if(max(iou_predicteds_to_gt) > iou_threshold and max(iou_predicteds_to_gt) > 0.0):
      TP+=1
  FN = total_faces - TP
  FP = pred_faces - TP
  #print("TP, FP, FN",TP, FP, FN)
  return TP, FP, FN 

def calculate_iou_face(box1, box2):
  if box1.intersects(box2):
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection/(union) 
  return 0

def plot_ap(rec, prec, classe, ap, mrec=None, mpre=None, tipo=""):
  plt.plot(rec, prec, label='Precision')
  plt.plot(rec, prec, '*', color='g')
  plt.plot(mrec, mpre, '--r', label=f'Precisão interpolada ({tipo})')
  plt.plot(mrec, mpre, 'X', color='r')
  plt.xlabel('recall')
  plt.ylabel('precision')
  ap_str = "{0:.4f}%".format(ap * 100)
  plt.title(f'Precision x Recall curve \nClass: {str(classe)}, AP: {ap_str}')
  plt.legend(shadow=True)
  plt.grid()
  plt.show()
  
def evaluate(iou_threshold, parsed_gt, parsed_predicted, sizes_gt=None, sizes_predicted=None, plot=False):
    TP_ = 0
    FP_ = 0
    FN_ = 0
    precision = []
    recall = []

    for predicted_image in tqdm(parsed_predicted, desc="Evaluating images"):
        gt_image = aux_match_image(predicted_image, parsed_gt)
        
        # Extract prefix from the file names using os.path.split and split
        gt_prefix = os.path.split(gt_image.name)[1].split('.')[0]
        predicted_prefix = os.path.split(predicted_image.name)[1].split('.')[0]

        # Filter the sizes_gt dictionary based on the extracted prefixes
        filtered_sizes_gt = {key: value for key, value in sizes_gt.items() if key.startswith(gt_prefix)}
        filtered_sizes_predicted = {key: value for key, value in sizes_predicted.items() if key.startswith(predicted_prefix)}

        # Extract the width_gt and height_gt values for gt_image and predicted_image
        width_gt, height_gt = next(iter(filtered_sizes_gt.values()), (None, None))
        width_predicted, height_predicted = next(iter(filtered_sizes_predicted.values()), (None, None))
        #print("gt", width_gt, height_gt)
        #print("predicted", width_predicted, height_predicted)
        width_factor = float(width_gt)/float(width_predicted)
        height_factor = float(height_gt)/float(height_predicted)
        #print("FACTOR",width_factor, height_factor)
        TP, FP, FN = calculate_TP_FP_FN(gt_image, predicted_image, iou_threshold, width_factor, height_factor)
        TP_ += TP
        FP_ += FP
        FN_ += FN

        if (TP_ != 0):
            precision.append(TP_ / (TP_ + FP_))
            recall.append(TP_ / (TP_ + FN_))
        else:
            precision.append(0.0)
            recall.append(0.0)          
    recall, precision = (list(t) for t in zip(*sorted(zip(recall, precision))))
    
    ap, mpre, mrec, ii = CalculateAveragePrecision(precision, recall)
    if plot:
        plot_ap(recall, precision, 1, ap, mrec, mpre, tipo="Todos os pontos")
    return ap

def get_sizes_path(input_path):
    if (input_path == "wider_face_val_bbx_gt.txt"):
        return "WIDER_size.txt"
    # Replace "bbox" with "size" and "_easy", "_medium", "_hard" with "_size"
    output_path = input_path.replace("bbox/", "").replace("_easy", "_size").replace("_medium", "_size").replace("_hard", "_size")
    return output_path

def parse_sizes(file_path):
    sizes_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) >= 3:
                file_name = data[0]
                width = data[1]
                height = data[2]
                sizes_dict[file_name] = (width, height)
    return sizes_dict

def main(iou_threshold=0.5, gt=None, predicted=None, plot=False):
    if gt is None or predicted is None:
        print("Please provide two text files as input.")
        return 0
    print("Calculating AP for: ", predicted)
    parsed_gt = parse_file(gt)
    sizes_gt = parse_sizes(get_sizes_path(gt))
    #print("Ground Truth Parsed Sucessfully")
    parsed_predicted = parse_file(predicted)
    sizes_predicted = parse_sizes(get_sizes_path(predicted))
    #print("Predicted Parsed Sucessfully")
    result = evaluate(iou_threshold, parsed_gt, parsed_predicted, sizes_gt, sizes_predicted, plot)
    print('Average IoU = %s' % str(result))
    return result
        
if __name__ == '__main__':
    gt = "wider_face_val_bbx_gt.txt"
    #predicted = "test.txt"
    predicted = [["bbox/WIDER_easy.txt","bbox/WIDER_medium.txt","bbox/WIDER_hard.txt"],
                 ["bbox/WIDER_DP2_easy.txt","bbox/WIDER_DP2_medium.txt","bbox/WIDER_DP2_hard.txt"],
                 ["bbox/WIDER_CF_easy.txt","bbox/WIDER_CF_medium.txt","bbox/WIDER_CF_hard.txt"],
                 ["bbox/WIDER_CF_DP2_easy.txt","bbox/WIDER_CF_DP2_medium.txt","bbox/WIDER_CF_DP2_hard.txt"],
                 ["bbox/WIDER_DP2_CF_easy.txt","bbox/WIDER_DP2_CF_medium.txt","bbox/WIDER_DP2_CF_hard.txt"]]
    # Determine the dimensions of the 'predicted' list
    num_rows = len(predicted)
    num_cols = len(predicted[0])

    # Create the 'table_AP' with empty lists
    table_AP = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    iou_threshold=0.5
    for i in range(num_rows):
        for j in range(num_cols):
            table_AP[i][j] = main(iou_threshold, gt, predicted[i][j])
            
    for row in table_AP:
        print(row)     
           
    for row in predicted:
        print(row)   
from shapely.geometry import box
from sklearn.metrics import fbeta_score, average_precision_score
import numpy as np
from matplotlib import pyplot as plt

class FileData:
    def __init__(self, name, length, coordinates):
        self.name = name
        self.length = length
        self.coordinates = coordinates
        
    def __str__(self):
        return f"Name: {self.name}, Length: {self.length}, Coordinates: {self.coordinates}"

def plot_Precision_Recall_Curve(precision, recall):
    plt.plot(recall, precision, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()
    
# 11-point interpolated average precision
def ElevenPointInterpolatedAP(rec, prec):
    mrec = []
    [mrec.append(e) for e in rec]
    mpre = []
    [mpre.append(e) for e in prec]
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    print(ap)
    exit()
    return ap
    
def CalculateAveragePrecision(prec, rec):
    print("rec", rec)
    print("prec", prec)
    
    mrec = [0] + rec + [1]
    mpre = [0] + prec + [0]
    mpre.reverse()# TESTE
    
    for i in range(len(mpre) - 1, 0, -1):
        print(mpre[i], mpre[i - 1])
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    print("mrec", mrec)
    print("mpre", mpre)
    #plot_Precision_Recall_Curve(mpre, mrec)
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    print("AP: ", ap)
    return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    #return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
  
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
        if gt_item.name.replace('/', '\\') == predicted_image.name.replace('/', '\\'): #same image
            return gt_item
    print("Image not found")
    return None
  
def calculate_TP_FP_FN(gt_image, predicted_image, iou_threshold):
  TP = 0 
  FP= 0 
  FN = 0
  iou_values_list = []
  iou = []
  total_faces = gt_image.length
  pred_faces = predicted_image.length
  for gt_box_coordinates in gt_image.coordinates:
    gt_box = box(gt_box_coordinates[0],  gt_box_coordinates[1],
                gt_box_coordinates[2] + gt_box_coordinates[0],
                gt_box_coordinates[3] + gt_box_coordinates[1])
  
    iou_predicteds_to_gt = [0.0]
    for predicted_box_coordinates in predicted_image.coordinates:
        predicted_box = box(predicted_box_coordinates[0],  predicted_box_coordinates[1],
                            predicted_box_coordinates[2] + predicted_box_coordinates[0],
                            predicted_box_coordinates[3] + predicted_box_coordinates[1])
        iou_predicteds_to_gt.append(calculate_iou_face(gt_box, predicted_box))
    #print("trying to find the best match bbox prediction", iou_predicteds_to_gt)
    if(max(iou_predicteds_to_gt) > iou_threshold and max(iou_predicteds_to_gt) > 0.0):
      TP+=1
  FN = total_faces - TP
  FP = pred_faces - TP
  #print("return", iou)
  print("TP, FP, FN",TP, FP, FN)
  return TP, FP, FN 
test_bbox = []
anotacoes_preditas_1000 = [
      #{'iscrowd': 0, 'image_id': 1000, 'bbox': [115.16, 152.13, 83.23, 228.41], 'category_id': 1, 'id': 1},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [405.93, 120.42, 37.13, 45.52], 'category_id': 1, 'id': 2, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [265.33, 95.86, 88.92, 315.88], 'category_id': 1, 'id': 3, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [209.23, 174.64, 99.63, 249.08], 'category_id': 1, 'id': 4, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [504.67, 191.95, 135.33, 288.05], 'category_id': 1, 'id': 5, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [410.2, 208.53, 114.9, 271.47], 'category_id': 1, 'id': 6, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [380.85, 159.91, 88.44, 319.25], 'category_id': 1, 'id': 7, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [330.07, 154.25, 83.05, 313.88], 'category_id': 1, 'id': 8, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [410.75, 107.03, 88.09, 121.16], 'category_id': 1, 'id': 9, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [183.4, 121.34, 95.77, 272.75], 'category_id': 1, 'id': 10, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [52.14, 185.12, 59.26, 212.53], 'category_id': 1, 'id': 11, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [349.49, 118.71, 52.64, 41.13], 'category_id': 1, 'id': 12, 'score': 0.99},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [0, 0, 52.64, 41.13], 'category_id': 1, 'id': 13, 'score': 0.6},
      {'iscrowd': 0, 'image_id': 1000, 'bbox': [349.49, 118.71, 52, 30], 'category_id': 1, 'id': 14, 'score': 0.7}]
#'Nestas preditas falta 1 pessoa, possui 11 TP, 1 erro total (predição aleatoria la em 0,0) e 1 que possui um IoU inf
anotacoes_preditas_1353 = [138.58, 198.11, 250, 250]

# aumentei o tamanho do objeto, deveria ser 1 TP (se passar no iou) e 5 FN
anotacoes_preditas_1268 = [[402.34, 205.02, 65.26, 75],
                           [0.0, 209.18, 24.95, 70]]
# 2 FN, 2 tp
anotacoes_preditas_2006 = [313.11, 189.53, 64.8, 67.14, 
                           5.23, 252.99, 45.73, 108.46, 
                           200, 300, 64.8, 67.14]

# 2 TP, 1 FP, 1 FN

def calculate_iou_face(box1, box2):
  if box1.intersects(box2):
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection/(union) 
  return 0

def plot_ap(rec, prec, classe, ap, mrec=None, mpre=None, tipo=""):
  print("Salve")
  plt.plot(rec, prec, label='Precision')
  plt.plot(rec, prec, '*', color='g')
  plt.plot(mrec, mpre, '--r', label=f'Precisão interpolada ({tipo})')
  plt.plot(mrec, mpre, 'X', color='r')
  plt.xlabel('recall')
  plt.ylabel('precision')
  # ap_str = "{0:.2f}%".format(ap * 100)
  ap_str = "{0:.4f}%".format(ap * 100)
  plt.title(f'Precision x Recall curve \nClass: {str(classe)}, AP: {ap_str}')
  plt.legend(shadow=True)
  plt.grid()
  plt.show()
  
def evaluate(iou_threshold, parsed_gt, parsed_predicted):
    total_faces = sum(face.length for face in parsed_gt)
    TP_ = 0
    FP_ = 0
    FN_ = 0
    precision = []
    recall = []
    i=0
    for predicted_image in parsed_predicted:
      gt_image = aux_match_image(predicted_image, parsed_gt)
      #iou_image = calculate_iou_image(gt_image, predicted_image, faces_not_detected)
      TP, FP, FN = calculate_TP_FP_FN(gt_image, predicted_image, iou_threshold)
      TP_+=TP
      FP_+=FP
      FN_+=FN
      fn = [1, 2, 6, 1, 3]
      tp = [11, 2, 0, 2, 1]
      fp = [2, 0, 1, 1, 0]
      '''
      if (TP != 0):
        precision.append(TP/(TP+FP))
        recall.append(TP/(TP+FN))
        #recall.append(TP/float(total_faces))
        '''
      if (TP_ != 0):
        precision.append(TP_/(TP_+FP_))
        recall.append(TP_/(TP_+FN_))
        #recall.append(TP/float(total_faces))
      else:
        precision.append(0.0)
        recall.append(0.0)        
      #if (i == 100):break # TESTE APAGAR DEPOIS
      #else: i+=1
    #print(TP_, FP_, FN_)

    Recall_Dict = [0.9166666666666666,0.5,0,0.6666666666666666,0.25]
    Precision_Dict = [0.8461538461538461,1.0,0,0.6666666666666666,1.0]
    # ordena as litas
    
    recall, precision = (list(t) for t in zip(*sorted(zip(recall, precision))))
    #recall, precision = (list(t) for t in zip(*sorted(zip(Recall_Dict, Precision_Dict))))
    
    #ElevenPointInterpolatedAP(recall, precision)
    #CalculateAveragePrecision(precision, recall)
    ap_, mpre, mrec, ii = CalculateAveragePrecision(precision, recall)
    plot_ap(recall, precision, 1, ap_, mrec, mpre, tipo="Todos os pontos")
    exit()
    return ap

def main(iou_threshold=0.5, gt=None, predicted=None):
    #print_file_content(gt)
    if gt is None or predicted is None:
        print("Please provide two text files as input.")
        return 0
    print("Calculating AP for: ", predicted)
    parsed_gt = parse_file(gt)
    print("Ground Truth Parsed Sucessfully")
    parsed_predicted = parse_file(predicted)
    print("Predicted Parsed Sucessfully")
    result = evaluate(iou_threshold, parsed_gt, parsed_predicted)
    print('Average IoU = %s' % str(result))


        
if __name__ == '__main__':
    gt = "wider_face_val_bbx_gt.txt"
    #predicted = "test.txt"
    predicted = [["bbox/WIDER_easy.txt","bbox/WIDER_medium.txt","bbox/WIDER_hard.txt"],
                 ["bbox/WIDER_DP2_easy.txt","bbox/WIDER_DP2_medium.txt","bbox/WIDER_DP2_hard.txt"],
                 ["bbox/WIDER_CF_easy.txt","bbox/WIDER_CF_medium.txt","bbox/WIDER_CF_hard.txt"],
                 ["bbox/WIDER_CF_DP2_easy.txt","bbox/WIDER_CF_DP2_medium.txt","bbox/WIDER_CF_DP2_hard.txt"],
                 ["bbox/WIDER_DP2_CF_easy.txt","bbox/WIDER_DP2_CF_medium.txt","bbox/WIDER_DP2_CF_hard.txt"]]
    iou_threshold=0.5
    main(iou_threshold, gt, predicted[0][0])
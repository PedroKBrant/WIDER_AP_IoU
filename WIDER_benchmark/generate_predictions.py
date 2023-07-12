import os
import cv2
import face_detection
from tqdm import tqdm


def main( image_folder = "images", difficulty = "easy", output_file = None, detector_type = "DSFDDetector"):
    if output_file is None:
        output_file = f"{image_folder}_{difficulty}.txt"
    print(f'Writing at {output_file}')
    #print(face_detection.available_detectors)
    detector = face_detection.build_detector("DSFDDetector")

    if difficulty == "hard":# TO DO FIX FOR HARD
        difficulty_range = (0,20) 
    elif difficulty == "medium":
        difficulty_range = (21,40)
    else:# difficulty == "easy"
        difficulty_range = (40,61)
    print(f"Files in range{difficulty_range}")

    with open(output_file, "w") as f:
        num_files = sum(len(filtered_files) for _, _, files in os.walk(image_folder) 
                        if len(files) > 0 for filtered_files in [[file for file in files if file.split('_', 1)[0].isdigit() 
                        and int(file.split('_', 1)[0]) in range(difficulty_range[0], difficulty_range[1] + 1)]] if filtered_files)
        pbar = tqdm(total=num_files)
        for root, dirs, files in os.walk(image_folder):
            if len(files) > 0:
                filtered_files = [file for file in files if file.split('_', 1)[0].isdigit() and 
                                int(file.split('_', 1)[0]) in range(difficulty_range[0], difficulty_range[1] + 1)]
                if filtered_files:
                    for filename in filtered_files:
                        pbar.update(1)
                        if filename.endswith((".jpg", ".jpeg", ".png")):
                            # Load the image
                            image_path = os.path.join(root, filename)
                            im = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB

                            # Perform face detection
                            detections = detector.detect(im)

                            # Write the image name and number of detections
                            folder_name = os.path.basename(root)
                            file_path = os.path.join(folder_name, filename)
                            f.write("{}\n".format(file_path))
                            f.write("{}\n".format(len(detections)))

                            # Write the detections to the file
                            for detection in detections:
                                detection[2] = abs(detection[0] - detection[2])
                                detection[3] = abs(detection[1] - detection[3])
                                formatted_detection = [format(value, '.2f') for value in detection]
                                formatted_detection_str = " ".join(formatted_detection)

                                f.write("{}\n".format(formatted_detection_str))
        pbar.close()

    print("Detections saved to {}".format(output_file))


if __name__ == '__main__':
    image_folder = "images/WIDER_Baseline"
    difficulty = "easy"
    output_file = None
    detector_type = "DSFDDetector"
    #detector_type = "RetinaNetResNet50"

    main(image_folder, difficulty, output_file, detector_type)
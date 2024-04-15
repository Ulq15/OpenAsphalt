import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import minmax_scale

def normalize_image(image):
    norm = minmax_scale(image, (0,1))
    return norm 

def get_new_coordinates(old_x, old_y, old_width, old_height, ratio, padding):
    top, left = padding
    new_x = int(old_x * ratio) + left
    new_y = int(old_y * ratio) + top
    new_width = int(old_width * ratio)
    new_height = int(old_height * ratio)
    return new_x, new_y, new_width, new_height

# from https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec 
def resize_with_pad(image, new_shape, padding_color = (0, 0, 0)):
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image, ratio, (top, left)

def process_image(image, target_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image, ratio, padding = resize_with_pad(image, target_size)
    # image = normalize_image(image) # leads to saving blank images so normalization needs to happen in the model
    return image, ratio, padding

def show_bbox(folder_path):
    imagePaths = sorted(list(paths.list_images(folder_path)))
    for imagePath in imagePaths:
        name = imagePath.split("\\")[-1].split(".")[0]
        image = cv2.imread(out_path+name+".jpg")
        with open(folder_path+name+".txt",'r') as file:
            annotation = file.readline().split(",")
            x = int(annotation[1].strip())
            y = int(annotation[2].strip())
            x2 = int(annotation[3].strip())
            y2 = int(annotation[4].strip())
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 1)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def start(in_path, out_path, size):
    # grab all image paths in the input directory
    imagePaths = sorted(list(paths.list_images(in_path)))
    index = 0
    annotations = []
    # loop over all image paths in the input directory
    for imagePath in imagePaths:
        # load the input image from disk and resize it
        image = cv2.imread(imagePath)
        processed_image, ratio, padding = process_image(image, size)
        image_name = imagePath.split("\\")[-1].split(".")[0]
        index += 1
        new_name = "car_" + str(index)
        if cv2.imwrite(filename=str(out_path+new_name+".jpg"), img=processed_image) != True:
           print(image_name +" failed to save!")
        with open(in_path + image_name + ".txt", "r") as file:
            for line in file.readlines():
                annotation = line.split("\t")
                img_name = annotation[0].strip()
                x1 = int(annotation[1].strip())
                y1 = int(annotation[2].strip())
                width = int(annotation[3].strip())
                height = int(annotation[4].strip())
                plate_text = annotation[5].strip()
                x2, y2, new_width, new_height = get_new_coordinates(x1, y1, width, height, ratio, padding)
                new_annotation =f"{new_name}.jpg,{x2},{y2},{x2+new_width},{y2+new_height},{plate_text}\n"
                annotations.append(new_annotation)
                with open(out_path+new_name+".txt", "w") as txt_file:
                    txt_file.write(f"{new_annotation}") 
    with open(out_path +"annotations.csv", "w+") as file:
        if len(file.readlines()) == 0:
            header = "filename,x_min,y_min,x_max,y_max,class\n"
            file.write(header)
        for line in annotations:
            file.write(f"{line}")     
    print("Done!")

def to_csv(path):
    imagePaths = sorted(list(paths.list_images(path)))
    with open(path+"annotations.csv", "a") as file:
        header = "filename,x,y,width,height,class\n"
        file.write(header)
        for imagePath in imagePaths:
            name = imagePath.split("\\")[-1].split(".")[0]
            with open(path+name+".txt",'r') as txt_file:
                annotation = txt_file.readline().replace("\t",",").strip()
                # file.write(f"{name}.jpg,{annotation[1]},{annotation[2]},{annotation[3]},{annotation[4]},{annotation[5]}")
                file.write(f"{annotation}\n")
    

if __name__ == "__main__":
    in_path = ".\\benchmarks\\endtoend\\us\\"
    # out_path/ = ".\\processed_images\\endtoend_us\\"
    size = (768, 768)   # Resize to 768x768 pixels, which is 1.5 * 512, total pixles: 768*768 = 589,824px
    out_path = ".\\processed_images\\data\\"
    # start(in_path, out_path, size)
    show_bbox(out_path)
    # to_csv(out_path)
    
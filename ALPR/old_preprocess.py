import cv2
import numpy as np

from alpr import ALPR, cleanup_text
import imutils
from imutils import paths

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_image(image):
    return image / 255.0

# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(image, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', image) 
  
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = image[y, x, 0] 
        g = image[y, x, 1] 
        r = image[y, x, 2] 
        cv2.putText(image, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', image) 

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
    print(f"Max New = {max(new_shape)}, Max OG = {max(original_shape)}")
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image, ratio, (top, bottom, left, right)

def preprocess_image(image, target_size):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image
    # resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    resized, ratio, padding = resize_with_pad(gray, target_size)
    
    # Normalize pixel values
    normalized = resized / 255.0
    print(f"R = {ratio}, P = {padding}")
    return np.expand_dims(normalized, axis=-1)  # Add channel dimension

if __name__ == "__main__":
    alpr = ALPR(minAR=1, maxAR=2)
    # Example usage
    new_size = (768, 768) # Resize to 768x768 pixels, which is 1.5 * 512
    path = "..\\..\\benchmarks\\endtoend\\us\\"
    image = cv2.imread(path + "car1.jpg")
    preprocessed_image = preprocess_image(image, new_size)  

    cv2.imshow("Image", image)
    cv2.setMouseCallback('Image', click_event) 
    cv2.waitKey(0)
    cv2.imshow("Image", preprocessed_image)
    cv2.setMouseCallback('Image', click_event) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # grab all image paths in the input directory
    # imagePaths = sorted(list(paths.list_images(path)))

    # # loop over all image paths in the input directory
    # for imagePath in imagePaths:
    #     # load the input image from disk and resize it
    #     image = cv2.imread(imagePath)
    #     image = imutils.resize(image, width=600)
    #     # apply automatic license plate recognition
    #     (lpText, lpCnt) = alpr.find_and_ocr(image, psm=7, clearBorder=False)
    #     # only continue if the license plate was successfully OCR'd
    #     if lpText is not None and lpCnt is not None:
    #         # fit a rotated bounding box to the license plate contour and
    #         # draw the bounding box on the license plate
    #         box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
    #         box = box.astype("int")
    #         cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    #         # compute a normal (unrotated) bounding box for the license
    #         # plate and then draw the OCR'd license plate text on the image
    #         (x, y, w, h) = cv2.boundingRect(lpCnt)
    #         cv2.putText(image, cleanup_text(lpText), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    #         # show the output ANPR image
    #         print("[INFO] {}".format(lpText))
    #         cv2.imshow("Output ANPR", image)
    #         cv2.waitKey(0)

    '''
    to get the new coordinates of the plate after resizing the image, 
    1st multiply the old coordinates by the ratio of the new image to the old image, 
    then add the size value of the padding to the x and y coordinates, written as top, bottom, left, right
    depending on the aspect ratio of the original image and the new image, some of the padding values might be 0
    so check if top & bottom, or left & right are 0, if they are, then the new coordinates will be the old coordinates multiplied by the ratio
    '''
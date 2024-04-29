1- save , display image


# Import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import skimage.io as io

# Function to read an image using PIL (Pillow)
def read_image_pil(file_path):
    img = Image.open(file_path)
    return img

# Function to save an image using PIL (Pillow)
def save_image_pil(img, output_path):
    img.save(output_path)

# Function to display an image using PIL (Pillow)
def display_image_pil(img):
    img.show()

# Function to read an image using Matplotlib
def read_image_matplotlib(file_path):
    img = plt.imread(file_path)
    return img

# Function to save an image using Matplotlib
def save_image_matplotlib(img, output_path):
    plt.imsave(output_path, img)

# Function to display an image using Matplotlib
def display_image_matplotlib(img):
    plt.imshow(img)
    plt.show()

# Function to read an image using Scikit Image
def read_image_scikit(file_path):
    img = io.imread(file_path)
    return img

# Function to save an image using Scikit Image
def save_image_scikit(img, output_path):
    io.imsave(output_path, img)

# Function to display an image using Scikit Image
def display_image_scikit(img):
    io.imshow(img)
    io.show()

if __name__ == "__main__":
    # Example usage
    file_path = r"C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp1\\hutao.jpg"
    output_path = r"C:\\Users\\Dell\\Desktop\\BUILDS\\pics\\exp1\\savedimg.jpg"

    # Using PIL (Pillow)
    img_pil = read_image_pil(file_path)
    save_image_pil(img_pil, output_path)
    display_image_pil(img_pil)

    # Using Matplotlib
    img_matplotlib = read_image_matplotlib(file_path)
    save_image_matplotlib(img_matplotlib, output_path)
    display_image_matplotlib(img_matplotlib)

    # # Using Scikit Image
    # img_scikit = read_image_scikit(file_path)
    # save_image_scikit(img_scikit, output_path)
    # display_image_scikit(img_scikit)





2-convert one to another


from PIL import Image

def convert_image(input_path, output_path, output_format):

    try:
        with Image.open(input_path) as img:
            img.save(output_path, format=output_format)
            print(f"Image converted and saved to {output_path} in {output_format} format.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_path = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp2\\hutao.jpg"
    output_path = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp2\\hutao.png"
    output_format = "png"

    convert_image(input_path, output_path, output_format)



3- one space to another


def convert_spaces(input_path, output_path, replacement_char = ' :) '):   
    try:       
        with open(input_path, 'r') as infile:            
            content = infile.read()            
            converted_content = content.replace(' ', replacement_char)            
        with open(output_path, 'w') as outfile:
            outfile.write(converted_content)
        print(f"Spaces converted and saved to {output_path} with replacement character '{replacement_char}'.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage  
    input_path = r"C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp3\\exp3.txt"
    output_path = r"C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp3\\exp3.txt"
    replacement_char = ' :) '
    convert_spaces(input_path, output_path, replacement_char)



4- grayscale


from PIL import Image

def generate_grayscale(input_path, output_path): 
    try: 
        with Image.open(input_path) as img: 
            grayscale_img = img.convert("L") 
            grayscale_img.save(output_path) 
            print(f"Grayscale image generated and saved to {output_path}.") 
    except Exception as e: 
        print(f"Error: {e}")

def generate_negative(input_path, output_path): 
    try: 
        with Image.open(input_path) as grayscale_img: 
            negative_img = Image.eval(grayscale_img, lambda x: 255 - x) 
            negative_img.save(output_path) 
            print(f"Negative image generated and saved to {output_path}.") 
    except Exception as e: 
        print(f"Error: {e}")

if __name__ == "__main__": 
    
    # Example usage 
    input_path = r"C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp4\\hutao.jpg" 
    grayscale_output_path = r"C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp4\\hutao_grayscale.jpg" 
    negative_output_path = r"C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp4\\hutao_negative.jpg" 
    
    generate_grayscale(input_path, grayscale_output_path) 
    generate_negative(grayscale_output_path, negative_output_path) 



5-thresholding


import cv2
import numpy as np

def threshold_image(input_path, output_path, threshold_values):
    try:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        threshold_images = [cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1] for threshold in threshold_values]
        
        for i, threshold_img in enumerate(threshold_images):
            cv2.imshow(f"Threshold{i + 1}", threshold_img)
            cv2.imwrite(output_path.format(i+1), threshold_img)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error:{e}")
        
if __name__ == "__main__":
    input_path = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp5\\hutao.jpg"
    output_path_template = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp5\\thresholded_image{}.jpg"
    threshold_values = [200, 250, 300]
    threshold_image(input_path, output_path_template, threshold_values)



6-cropping, resizing


import cv2 

def crop_image(input_path, output_path, x, y, width, height): 
    try: 
        img = cv2.imread(input_path) 
        cropped_img = img[y:y+height, x:x+width] 
        cv2.imshow("Cropped Image", cropped_img) 
        cv2.imwrite(output_path, cropped_img) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    except Exception as e: 
        print(f"Error: {e}") 

def resize_image(input_path, output_path, new_width, new_height): 
    try: 
        img = cv2.imread(input_path) 
        resized_img = cv2.resize(img, (new_width, new_height)) 
        cv2.imshow("Resized Image", resized_img) 
        cv2.imwrite(output_path, resized_img) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    except Exception as e: 
        print(f"Error: {e}") 

def scale_image(input_path, output_path, scale_factor): 
    try: 
        img = cv2.imread(input_path) 
        scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor) 
        cv2.imshow("Scaled Image", scaled_img) 
        cv2.imwrite(output_path, scaled_img) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    except Exception as e: 
        print(f"Error: {e}") 

def flip_image(input_path, output_path, flip_code): 
    try: 
        img = cv2.imread(input_path) 
        flipped_img = cv2.flip(img, flip_code) 
        cv2.imshow("Flipped Image", flipped_img) 
        cv2.imwrite(output_path, flipped_img) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    except Exception as e: 
        print(f"Error: {e}")

if __name__ == "__main__": 
    # Example usage 
    input_path = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp6\\hutao.jpg" 
    output_path_cropped = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp6\\cropped_hutao.jpg" 
    output_path_resized = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp6\\resized_hutao.jpg" 
    output_path_scaled = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp6\\scaled_hutao.jpg" 
    output_path_flipped = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp6\\flipped_hutao.jpg" 
    # Crop the image 
    crop_image(input_path, output_path_cropped, x=50, y=50, width=300, height=200) 
    # Resize the image 
    resize_image(input_path, output_path_resized, new_width=400, new_height=300) 
    # Scale the image 
    scale_image(input_path, output_path_scaled, scale_factor=0.5) 
    # Flip the image (flip_code: 0 - horizontal, 1 - vertical, -1 - both) 
    flip_image(input_path, output_path_flipped, flip_code=1)



7-histogram


import cv2
import matplotlib.pyplot as plt

def display_image(input_path):
    try:
        img = cv2.imread(input_path)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

def show_image_attributes(input_path):
    try:
        img = cv2.imread(input_path)
        height, width, channels = img.shape
        print(f"Image Dimensions: {width} x {height}")
        print(f"Number of Channels: {channels}")

    except Exception as e:
        print(f"Error: {e}")

def show_image_histogram(input_path):
    try:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(histogram)
        plt.title("Image Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":

    input_path = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp7\\hutao.jpg"
    display_image(input_path)
    show_image_attributes(input_path)
    show_image_histogram(input_path)




8-averaging mask


import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_averaging_filter(input_path, output_path, kernel_size):
    try:
        img = cv2.imread(input_path)
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        filtered_img = cv2.filter2D(img, -1, kernel)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Averaging Filter (Kernel Size {kernel_size})')

        plt.show()

        cv2.imwrite(output_path, filtered_img)

        print(f"Filtered image saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_path = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp8\\hutao.jpg"
    output_path = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp8\\filtered_hutao.jpg"
    kernel_size = 5  

    apply_averaging_filter(input_path, output_path, kernel_size)



9- morphological


import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_morphological_operations(input_path, output_path_prefix):
    try:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5, 5), np.uint8)

        # Erosion
        erosion = cv2.erode(img, kernel, iterations=1)
        cv2.imwrite(f"{output_path_prefix}_erosion.jpg", erosion)

        # Dilation
        dilation = cv2.dilate(img, kernel, iterations=1)
        cv2.imwrite(f"{output_path_prefix}_dilation.jpg", dilation)

        # Opening (erosion followed by dilation)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(f"{output_path_prefix}_opening.jpg", opening)

        # Closing (dilation followed by erosion)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f"{output_path_prefix}_closing.jpg", closing)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 5, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 5, 2)
        plt.imshow(erosion, cmap='gray')
        plt.title('Erosion')

        plt.subplot(1, 5, 3)
        plt.imshow(dilation, cmap='gray')
        plt.title('Dilation')

        plt.subplot(1, 5, 4)
        plt.imshow(opening, cmap='gray')
        plt.title('Opening')

        plt.subplot(1, 5, 5)
        plt.imshow(closing, cmap='gray')
        plt.title('Closing')

        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    
    input_path = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp9\\hutao.jpg"
    output_path_prefix = "C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp9\\processed_hutao.jpg"

    # Apply morphological operations
    apply_morphological_operations(input_path, output_path_prefix)




10-noisy image

import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the input image."""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

def apply_filters(image):
    """Apply different filters to remove noise."""
    # Median Filter
    median_filtered = cv2.medianBlur(image, 5)

    # Bilateral Filter
    bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)

    # Gaussian Filter
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

    return median_filtered, bilateral_filtered, gaussian_filtered

def main():
    # Read a sample image
    input_image = cv2.imread("C:\\Users\Dell\\Desktop\\BUILDS\\pics\\exp10\\hutao.jpg")

    # Add Gaussian noise to the image
    noisy_image = add_gaussian_noise(input_image)

    # Apply different filters
    median_filtered, bilateral_filtered, gaussian_filtered = apply_filters(noisy_image)

    # Display original and filtered images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    plt.title('Noisy Image')

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral Filter')

    plt.show()

    # Display Gaussian Filtered Image
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Filter')
    plt.show()

if __name__ == "__main__":
    main()

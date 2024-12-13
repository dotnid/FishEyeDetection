# pip install opencv-python numpy
import cv2
import numpy as np

def detect_fish_eye(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Check if image is loaded successfully
        if image is None:
            raise Exception("Could not read the image")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            5,
            2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            raise Exception("No contours found")
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw the circle on the original image
        result = image.copy()
        cv2.circle(result, center, radius, (0, 255, 0), 5)
        
        # Crop the eye region
        x1 = max(0, int(x - radius * 1.2))
        y1 = max(0, int(y - radius * 1.2))
        x2 = min(image.shape[1], int(x + radius * 1.2))
        y2 = min(image.shape[0], int(y + radius * 1.2))
        
        eye_crop = image[y1:y2, x1:x2]
        
        # Save results
        cv2.imwrite('detected_eye.jpg', result)
        cv2.imwrite('fish_eye_cropped.jpg', eye_crop)
        
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    image_path = 'ikan.jpg'  # Replace with your image path
    success = detect_fish_eye(image_path)
    if success:
        print("Fish eye detected and cropped successfully!")
    else:
        print("Failed to detect fish eye.")

import cv2

# Read the image
image_path = r'E:\PHD\1.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale image

# Sobel operator
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_abs = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)  # Take absolute value and convert to 8-bit image

# Save Sobel X result
cv2.imwrite('sobel_result.jpg', sobel_abs)

# Scharr operator (similar to Sobel but with a fixed kernel size of 3x3)
scharr_x = cv2.Scharr(image, -1, 1, 0)
scharr_y = cv2.Scharr(image, -1, 0, 1)
scharr_abs = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)

# Save Scharr X result
cv2.imwrite('scharr_result.jpg', scharr_abs)

# Laplacian operator
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)  # Take absolute value and convert to 8-bit image

# Save Laplacian result
cv2.imwrite('laplacian_result.jpg', laplacian_abs)

# Canny edge detector (requires two thresholds)
canny = cv2.Canny(image, 50, 150)

# Save Canny result
cv2.imwrite('canny_result.jpg', canny)
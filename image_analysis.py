# image_analysis.py

import cv2
import numpy as np
from PIL import ImageStat, Image

class ImageAnalyzer:
    def calculate_brightness(self, image):
        """ Calculates the average brightness of an image. """
        img = Image.fromarray(image)
        stat = ImageStat.Stat(img)
        return stat.mean[0]  # Average of R or grayscale

    def calculate_contrast(self, image):
        """ Calculates the contrast of an image. """
        img = Image.fromarray(image)
        stat = ImageStat.Stat(img)
        return stat.stddev[0]  # Standard deviation of R or grayscale

    def calculate_edge_density(self, image):
        """ Calculates the density of edges within an image using Sobel filters. """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edge_img = np.hypot(sobelx, sobely)
        edge_density = np.mean(edge_img)
        return edge_density

    def calculate_dominant_color(self, image):
        """ Calculates the dominant color of an image using k-means clustering. """
        pixels = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, _, centroids = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return centroids[0].astype(int)

# Example usage:
# analyzer = ImageAnalyzer()
# image = cv2.imread('path_to_image.jpg')
# brightness = analyzer.calculate_brightness(image)
# contrast = analyzer.calculate_contrast(image)
# edge_density = analyzer.calculate_edge_density(image)
# dominant_color = analyzer.calculate_dominant_color(image)


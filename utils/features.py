import scipy
from scipy.spatial import KDTree
import numpy as np

def contrast(image):
    r_coeff = 0.2989
    g_coeff = 0.5870
    b_coeff = 0.1140
    gray_image = (image[..., 0] * r_coeff +
                  image[..., 1] * g_coeff +
                  image[..., 2] * b_coeff)
    return gray_image.std()

'''
# Approximate RGB values for each category
color_dict = {
    "#000000": "black",  # Black
    "#FFFFFF": "white",  # White
    "#808080": "gray",   # Gray
    "#FF0000": "red",    # Red
    "#FFA500": "orange", # Orange
    "#FFFF00": "yellow", # Yellow
    "#008000": "green",  # Green
    "#0000FF": "blue",   # Blue
    "#800080": "purple", # Purple
    "#FFC0CB": "pink",   # Pink
    "#A52A2A": "brown"   # Brown
}

'''

# Convert hex to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    # Prepare color dictionary for KDTree
    names = list(color_dict.values())
    rgb_values = [hex_to_rgb(hex_color) for hex_color in color_dict.keys()]

    # Create KDTree
    kdt_db = KDTree(rgb_values)

# Vectorized approach to convert RGB to names
def convert_rgb_to_names(rgb_array):
    _, indices = kdt_db.query(rgb_array)
    return np.array(names)[indices]

def color_distrib(image):

    if image.size == 4096:  # Grayscale image check
        # Optional: Convert grayscale to RGB (simple replication across channels)
        reshaped_image = np.stack((image,)*3, axis=-1).reshape(-1, 3)
    else:
        # Reshape assuming it's already an RGB image
        reshaped_image = np.reshape(image, (-1, 3))


    # Convert RGB values to color names using vectorized operation
    color_names = convert_rgb_to_names(reshaped_image)
    # Calculate the frequency distribution of color names
    unique, counts = np.unique(color_names, return_counts=True)
    color_distribution = dict(zip(unique, counts))
    # Create a distribution array
    distrib = np.array([color_distribution.get(color, 0) for color in names])
    # Normalize the distribution to sum to 1
    distrib_normalized = distrib / np.sum(distrib)
    return distrib_normalized

# Example usage
# Assuming 'image' is a NumPy array of shape (64, 64, 3) representing an image
# distrib = color_distrib(image)
# print(distrib)

import numpy as np

def bbox_area(bbox_data):
    print(bbox_data)
    return (bbox_data[3]-bbox_data[1]) * (bbox_data[2]-bbox_data[0])

def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(
        abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta))
    )
    xmax = np.ceil(max(1, xmax))
    ymax = max(
        abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta))
    )
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb
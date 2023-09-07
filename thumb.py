
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, util, filters, measure, color
from FingerprintImageEnhancer import FingerprintImageEnhancer
from PIL import Image
import fingerprint_enhancer 


def enhance_fingerprint(image):
    image_gray = np.mean(image, axis=-1)
    histogram, bins = np.histogram(image_gray.flatten(), bins=256, range=[0, 256])
    cdf = histogram.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    image_equalized = np.interp(image_gray.flatten(), bins[:-1], cdf_normalized)
    image_equalized = image_equalized.reshape(image_gray.shape).astype(np.uint8)
    return image_equalized

if __name__ == '__main__':
    image_path = "C:\\Users\\durga\\Downloads\\Final_Project\\thumb 1.jpg"
    
    # Load the image
    image = io.imread(image_path)

    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)

    # Apply thresholding
    threshold = filters.threshold_otsu(gray_image)
    binary_image = gray_image < threshold  # Invert the thresholding condition

    # Label connected regions in the binary image
    labeled_image = measure.label(binary_image)

    # Find properties of labeled regions
    region_properties = measure.regionprops(labeled_image)

    # Sort the regions by area in descending order
    region_properties = sorted(region_properties, key=lambda r: r.area, reverse=True)

    # Extract thumbprint region
    thumbprint_region = None
    for region in region_properties:
        if region.major_axis_length > 100 and 0.5 < region.minor_axis_length / region.major_axis_length < 0.9:
            thumbprint_region = region
            break

    # ... (your existing thumbprint extraction and enhancement code)
    
    # Crop the thumbprint region from the original image
    if thumbprint_region:
        min_row, min_col, max_row, max_col = thumbprint_region.bbox
        thumbprint = image[min_row:max_row, min_col:max_col]

        # Enhance and denoise the thumbprint
        enhanced_thumbprint = enhance_fingerprint(thumbprint)
        image_enhancer = FingerprintImageEnhancer()
        denoised_thumbprint = image_enhancer.enhance(enhanced_thumbprint)

    # Save the denoised thumbprint to a file
    # output_path = "thumb_extract_output.jpg"
    output_path = "C:\\Users\\durga\\Downloads\\Final_Project\\output_thumb.jpg"
    io.imsave(output_path, denoised_thumbprint)

    print(f"Denoised thumbprint saved to {output_path}")

    # Display results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(thumbprint, cmap='gray')
    plt.title("Extracted Thumbprint")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(enhanced_thumbprint, cmap='gray')
    plt.title("Enhanced Thumbprint")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_thumbprint, cmap='gray')
    plt.title("Denoised Thumbprint")
    plt.axis('off')

    plt.show()





# --------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, filters, measure, color
# from PIL import Image

# def enhance_fingerprint(image):
#     # Your existing enhancement code here
#     return enhanced_image

# if __name__ == '__main__':
#     image_path = "C:\\Users\\durga\\Downloads\\Final_Project\\thumb 1.jpg"
    
#     # Load the image
#     image = io.imread(image_path)

#     # Convert the image to grayscale
#     gray_image = color.rgb2gray(image)

#     # Apply thresholding
#     threshold = filters.threshold_otsu(gray_image)
#     binary_image = gray_image < threshold

#     # ... (your existing thumbprint extraction code)
    
#     # Enhance and denoise the thumbprint
#     enhanced_thumbprint = enhance_fingerprint(thumbprint)
    
#     # ... (your existing saving and displaying code)

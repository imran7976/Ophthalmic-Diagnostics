import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


original_dataset_root = Path(r"D:\GG\Datasets\Eye Disease Image Dataset\Original Dataset")
enhanced_dataset_root = Path(r"D:\GG\Datasets\Eye Disease Image Dataset\Fourier_Enhanced_Original_Dataset")


def apply_fourier_enhancement(img):
 
    if img is None:
        return None

    # Converting to LAB to separate Light (L) from Color (A, B)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    # Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Applying FOURIER TRANSFORM High-Pass Filter. Converting to float for math operations
    f_img = np.float32(l_clahe)
    
    # Performing DFT
    dft = cv2.dft(f_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Creating a High Pass Filter Mask (Circular)
    rows, cols = l.shape
    crow, ccol = rows // 2, cols // 2
    
    # radius determines how much low-frequency (blur/lighting) to block
    radius = 30 
    
    # Creating a mask with 1s everywhere 
    mask = np.ones((rows, cols, 2), np.uint8)
    # Putting 0s in the center
    center_y, center_x = np.ogrid[:rows, :cols]
    mask_area = (center_y - crow)**2 + (center_x - ccol)**2 <= radius**2
    mask[mask_area] = 0

    # Applying mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the Fourier result to 0-255 range
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    # Blending the Fourier Edges with the CLAHE image
    # This adds the sharp edges back onto the nice contrast image
    # alpha=0.8 (original), beta=0.3 (edges)
    l_final = cv2.addWeighted(l_clahe, 0.8, img_back, 0.3, 0)

    # Merging channels and convert back to BGR
    img_lab_final = cv2.merge((l_final, a, b))
    img_final = cv2.cvtColor(img_lab_final, cv2.COLOR_LAB2BGR)
    
    return img_final

def process_dataset():
    if not original_dataset_root.exists():
        print(f"Error: Source path {original_dataset_root} does not exist.")
        return
    
    print(f"Starting Fourier Preprocessing...")
    print(f"Source: {original_dataset_root}")
    print(f"Destination: {enhanced_dataset_root}")
    print("-" * 50)

    class_folders = [f for f in original_dataset_root.iterdir() if f.is_dir()]

    if not class_folders:
        print("No class folders found in the source directory.")
        return

    for class_folder in class_folders:
        rel_path = class_folder.relative_to(original_dataset_root)
        dest_folder = enhanced_dataset_root / rel_path
        os.makedirs(dest_folder, exist_ok=True)

        images = list(class_folder.glob("*.*")) 
        
        print(f"Processing {class_folder.name} ({len(images)} images)...")

        for img_path in tqdm(images):
            
            img = cv2.imread(str(img_path))
 
            processed_img = apply_fourier_enhancement(img)
            
            if processed_img is not None:
                save_path = dest_folder / img_path.name
                cv2.imwrite(str(save_path), processed_img)

    print("\nProcessing Complete!")

if __name__ == "__main__":
    process_dataset()
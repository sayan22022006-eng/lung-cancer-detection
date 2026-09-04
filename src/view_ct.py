import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# CT folder we already tested
dicom_folder = "/Users/sayanghosh/Desktop/manifest-1787634840820/lidc_idri/LIDC-IDRI-0072/45499/88650"

# Find DICOM files
files = [
    os.path.join(dicom_folder, f)
    for f in os.listdir(dicom_folder)
    if f.lower().endswith(".dcm")
]

# Read all slices
slices = [pydicom.dcmread(file) for file in files]

# Sort slices
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

# Convert to Hounsfield Units
volume = np.stack([
    s.pixel_array.astype(np.float32) * float(s.RescaleSlope)
    + float(s.RescaleIntercept)
    for s in slices
])

print("CT volume shape:", volume.shape)

# Select 3 slices
# Lung window
window_center = -600
window_width = 1500

lower = window_center - window_width / 2
upper = window_center + window_width / 2

volume_windowed = np.clip(volume, lower, upper)

# Normalize to 0-255
volume_windowed = (
    (volume_windowed - lower) / (upper - lower) * 255
).astype(np.uint8)

# Select 3 slices
slice_1 = volume_windowed[75]
slice_2 = volume_windowed[152]
slice_3 = volume_windowed[228]

# Display them
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(slice_1, cmap="gray")
plt.title("Slice 75")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(slice_2, cmap="gray")
plt.title("Slice 152")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(slice_3, cmap="gray")
plt.title("Slice 228")
plt.axis("off")

plt.tight_layout()
plt.show()
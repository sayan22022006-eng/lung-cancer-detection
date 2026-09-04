import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Path to your DICOM folder
dicom_folder = "/Users/sayanghosh/Desktop/manifest-1787634840820/lidc_idri/LIDC-IDRI-0072/45499/88650"

# Get all DICOM files
files = [
    os.path.join(dicom_folder, f)
    for f in os.listdir(dicom_folder)
    if f.lower().endswith(".dcm")
]

# Read all DICOM files
slices = []

for file in files:
    ds = pydicom.dcmread(file)
    slices.append(ds)

print("Number of slices:", len(slices))

# Sort slices by their position
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

# Convert each slice to a NumPy array
volume = np.stack([
    s.pixel_array.astype(np.float32) * float(s.RescaleSlope) + float(s.RescaleIntercept)
    for s in slices
])

print("3D CT volume shape:", volume.shape)

# Display the middle slice
middle = volume.shape[0] // 2

plt.imshow(volume[middle], cmap="gray")
plt.title("Middle CT Slice")
plt.axis("off")
plt.show()
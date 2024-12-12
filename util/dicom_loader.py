import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage

class DicomLoader:
    def __init__(self):
        self.volume_data = None
        self.metadata = {}

    def load_series(self, directory_path):
        """Load a series of DICOM files from a directory."""
        directory = Path(directory_path)
        dicom_files = sorted(list(directory.glob('*.dcm')))

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {directory_path}")

        # Load first slice to get metadata
        first_slice = pydicom.dcmread(str(dicom_files[0]))
        shape = (len(dicom_files), first_slice.Rows, first_slice.Columns)
        self.volume_data = np.zeros(shape, dtype=np.int16)  # Use int16 for raw pixel values

        # Store metadata
        self.metadata = {
            'patient_id': getattr(first_slice, 'PatientID', 'Unknown'),
            'study_date': getattr(first_slice, 'StudyDate', 'Unknown'),
            'pixel_spacing': getattr(first_slice, 'PixelSpacing', [1.0, 1.0]),
            'slice_thickness': getattr(first_slice, 'SliceThickness', 1.0),
        }

        # Load all slices
        for idx, file_path in enumerate(dicom_files):
            slice_data = pydicom.dcmread(str(file_path))
            self.volume_data[idx] = slice_data.pixel_array

        # Normalize volume data to uint8 for rendering
        self.volume_data = self._normalize_to_uint8(self.volume_data)

        print(f"Loaded volume shape: {self.volume_data.shape}")
        print(f"Value range: [{self.volume_data.min()}, {self.volume_data.max()}]")

        return self.volume_data, self.metadata

    def _normalize_to_uint8(self, array):
        """Normalize volume data to uint8 (0-255)."""
        min_val = np.percentile(array, 1)  # Lower percentile for robust normalization
        max_val = np.percentile(array, 99)  # Upper percentile for robust normalization
        normalized = np.clip(array, min_val, max_val)
        normalized = ((normalized - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return normalized
    def apply_threshold(self, lower_threshold, upper_threshold):
        """Apply binary thresholding to the volume"""
        if self.volume_data is None:
            raise ValueError("No volume data loaded")
            
        binary_volume = np.logical_and(
            self.volume_data >= lower_threshold,
            self.volume_data <= upper_threshold
        ).astype(np.uint8) * 255
        
        return binary_volume

    def segment_spine(self, lower_threshold=200, upper_threshold=1500, morphology_iterations=2):
        if self.volume_data is None:
            raise ValueError("No volume data loaded")

        # Apply thresholding
        spine_mask = np.logical_and(
            self.volume_data >= lower_threshold,
            self.volume_data <= upper_threshold
        )

        # Apply morphological operations
        spine_mask = ndimage.binary_opening(spine_mask, iterations=morphology_iterations)
        spine_mask = ndimage.binary_closing(spine_mask, iterations=morphology_iterations)

        # Label connected components
        labeled, num_features = ndimage.label(spine_mask)

        if num_features > 0:
            # Remove small disconnected regions
            sizes = ndimage.sum(spine_mask, labeled, range(1, num_features + 1))
            mask_sizes = sizes > 1000  # Adjust this value based on your data
            remove_pixel = mask_sizes[labeled - 1]
            spine_mask[remove_pixel] = False
        else:
            print("Warning: No structures detected in spine segmentation")

        return spine_mask.astype(np.uint8) * 255

    def apply_segmentation(self, spine_mask):
        """
        Apply the segmentation mask to the volume data.
        
        Args:
        spine_mask (np.array): Binary mask of the segmented spine
        
        Returns:
        np.array: Segmented volume data
        """
        if self.volume_data is None:
            raise ValueError("No volume data loaded")

        return self.volume_data * (spine_mask > 0)
    
    def extract_shell(self, binary_volume, thickness=1):
        """Extract shell from binary volume using morphological operations"""
        
        # Erode the volume
        kernel = np.ones((3, 3, 3))
        eroded = ndimage.binary_erosion(binary_volume, kernel, iterations=thickness)
        
        # Shell is the difference between original and eroded volume
        shell = binary_volume.astype(bool) ^ eroded
        
        return shell.astype(np.uint8) * 255
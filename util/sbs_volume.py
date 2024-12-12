import numpy as np
from collections import defaultdict

class SBSVolume:
    def __init__(self, volume_data):
        """
        Initialize SBS structure from volume data
        
        Args:
            volume_data: 3D numpy array of voxel data
        """
        self.volume_data = volume_data


        self.original_shape = volume_data.shape
        self.slices = {}  # Store non-empty slices
        self.index_array = []  # Store indices of non-empty voxels
        self.attribute_list = []  # Store voxel attributes
        
        # Convert to SBS structure
        self._convert_to_sbs(volume_data)

    def _convert_to_sbs(self, volume_data):
        """Convert volume data to SBS structure"""
        depth, height, width = volume_data.shape
        
        # Process each slice
        for z in range(depth):
            slice_data = volume_data[z]
            non_zero_mask = slice_data > 0
            
            if np.any(non_zero_mask):
                # Store indices of non-zero voxels in this slice
                y_coords, x_coords = np.nonzero(non_zero_mask)
                
                # Create slice entry
                slice_entry = {
                    'indices': list(zip(y_coords, x_coords)),
                    'values': slice_data[non_zero_mask]
                }
                
                self.slices[z] = slice_entry
                
                # Update index array and attribute list
                for y, x in zip(y_coords, x_coords):
                    self.index_array.append((z, y, x))
                    self.attribute_list.append(slice_data[y, x])
        
        # Convert to numpy arrays
        self.index_array = np.array(self.index_array)
        self.attribute_list = np.array(self.attribute_list)

    def get_voxel(self, z, y, x):
        """Get voxel value at given coordinates"""
        if z in self.slices:
            slice_entry = self.slices[z]
            try:
                idx = slice_entry['indices'].index((y, x))
                return slice_entry['values'][idx]
            except ValueError:
                return 0
        return 0

    def get_slice(self, z):
        """Get full slice at given depth"""
        if z not in self.slices:
            return np.zeros((self.original_shape[1], self.original_shape[2]), dtype=np.uint8)
            
        slice_data = np.zeros((self.original_shape[1], self.original_shape[2]), dtype=np.uint8)
        slice_entry = self.slices[z]
        
        for (y, x), value in zip(slice_entry['indices'], slice_entry['values']):
            slice_data[y, x] = value
            
        return slice_data

    def get_shell(self):
        """Extract shell voxels from the volume"""
        shell_indices = []
        shell_values = []
        
        # For each voxel, check if it's on the boundary
        for i, (z, y, x) in enumerate(self.index_array):
            # Check 6-neighborhood
            is_boundary = False
            neighbors = [
                (z-1, y, x), (z+1, y, x),
                (z, y-1, x), (z, y+1, x),
                (z, y, x-1), (z, y, x+1)
            ]
            
            for nz, ny, nx in neighbors:
                if not self._has_voxel(nz, ny, nx):
                    is_boundary = True
                    break
            
            if is_boundary:
                shell_indices.append((z, y, x))
                shell_values.append(self.attribute_list[i])
        
        return np.array(shell_indices), np.array(shell_values)

    def _has_voxel(self, z, y, x):
        """Check if voxel exists at given coordinates"""
        if z in self.slices:
            return (y, x) in self.slices[z]['indices']
        return False

    def get_statistics(self):
        """Get memory usage statistics"""
        original_size = np.prod(self.original_shape)
        compressed_size = len(self.attribute_list)
        
        return {
            'original_voxels': original_size,
            'stored_voxels': compressed_size,
            'compression_ratio': original_size / max(compressed_size, 1),
            'num_slices': len(self.slices)
        }
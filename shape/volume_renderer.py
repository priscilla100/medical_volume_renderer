import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from pydicom import dcmread
from skimage import filters, measure
from scipy.ndimage import binary_closing, binary_opening
from shape import sbs
import scipy.ndimage as ndimage
import glm
import time

class VolumeRenderer:
    def __init__(self, dicom_path, shader):
        self.volume_texture = None
        self.cube_vbo = None
        self.cube_vao = None
        self.shader = shader

        # Load and preprocess DICOM data
        start_time = time.perf_counter()
        self.volume_data = self.load_dicom_series(dicom_path)
        load_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        self.spine_segmentation = self.preprocess_dicom(self.volume_data)
        segmentation_time = time.perf_counter() - start_time

        print(f"Volume Loading Time: {load_time:.2f} seconds")
        print(f"Segmentation Time: {segmentation_time:.2f} seconds")

        self.surface_voxels = None
        self.surface_voxel_texture = None
        self.intermediate_image_texture = None

        self.create_transfer_function()
        self.setup_gradient_textures()

        # Initialize clipping planes
        self.clip_min = np.array([0.0, 0.0, 0.0])  # Min corner of the volume
        self.clip_max = np.array([1.0, 1.0, 1.0])  # Max corner of the volume

        self.setup_spine_segmentation_texture()

        self.camera_position = glm.vec3(0.0, 0.0, 3.0)  # Default camera position
        self.camera_target = glm.vec3(0.0, 0.0, 0.0)  # Default camera target (looking at the center)

        self.lens = {
            'active': False,
            'position': glm.vec2(0.5, 0.5),  # Normalized screen coordinates
            'size': glm.vec2(0.2, 0.2),  # Normalized size
            'magnification': 2.0
        }

        self.setup_textures()

    def update_lens(self, active, position, size, magnification):
        self.shader.use()
        self.shader.setInt("u_lens_active", int(active))
        self.shader.setVec2("u_lens_position", glm.vec2(*position))
        self.shader.setVec2("u_lens_size", glm.vec2(*size))
        self.shader.setFloat("u_lens_magnification", magnification)

    def load_dicom_series(self, dicom_directory):
        """Load and normalize a series of DICOM files."""
        dicom_files = [os.path.join(dicom_directory, f) for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_directory}")

        # Load the first file to get the image dimensions
        first_slice = dcmread(dicom_files[0])
        volume_data = np.zeros((len(dicom_files), *first_slice.pixel_array.shape), dtype=np.float32)

        for i, file_path in enumerate(dicom_files):
            dicom_file = dcmread(file_path)
            volume_data[i] = dicom_file.pixel_array.astype(np.float32)

        # Normalize the entire volume
        volume_data = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min())
        volume_data = (volume_data * 255).astype(np.uint8)

        print(f"DICOM volume shape: {volume_data.shape}")
        return volume_data

    def load_dicom(self, dicom_path):
        """Load and normalize DICOM data."""
        dicom_file = dcmread(dicom_path)
        volume_data = dicom_file.pixel_array.astype(np.float32)
        volume_data = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min())
        volume_data = (volume_data * 255).astype(np.uint8)
        print(f"DICOM volume shape: {volume_data.shape}")
        return volume_data

    def segment_spine(self, volume_data):
        """
        Segment the spine from the 3D DICOM volume_data.
        Args:
            volume_data (numpy array): 3D numpy array representing the DICOM volume.
        Returns:
            spine_segmentation (numpy array): 3D binary array (1 for spine, 0 for background).
        """
        # Apply thresholding to segment spine (Otsu's method for automatic thresholding)
        threshold = filters.threshold_otsu(volume_data)
        binary_segmentation = volume_data > threshold

        # Label connected components and extract the largest connected component (assumed to be the spine)
        labeled, _ = measure.label(binary_segmentation, connectivity=1, return_num=True)
        properties = measure.regionprops(labeled)
        largest_region_label = max(properties, key=lambda prop: prop.area).label

        # Create binary mask for the largest connected component
        spine_segmentation = (labeled == largest_region_label).astype(np.uint8)
        return spine_segmentation

    def refine_segmentation(self, segmentation):
        """
        Apply morphological operations to refine the binary segmentation.
        Args:
            segmentation (numpy array): 3D binary segmentation array.
        Returns:
            refined_segmentation (numpy array): 3D refined binary segmentation array.
        """
        # Apply morphological opening to remove small noise
        opened_segmentation = binary_opening(segmentation, structure=np.ones((3, 3, 3)))

        # Apply morphological closing to fill small holes
        refined_segmentation = binary_closing(opened_segmentation, structure=np.ones((3, 3, 3)))
        return refined_segmentation

    def preprocess_dicom(self, volume_data):
        """
        Preprocess the DICOM volume data by segmenting the spine and refining the segmentation.
        """
        # Segment the spine
        spine_segmentation = self.segment_spine(volume_data)

        # Refine the segmentation
        refined_spine_segmentation = self.refine_segmentation(spine_segmentation)

        # Convert to uint8 and ensure it's compatible with texture creation
        refined_spine_segmentation = (refined_spine_segmentation * 255).astype(np.uint8)
        return refined_spine_segmentation

    def create_transfer_function(self):
        """
        Create a transfer function that specifically highlights spine structures.
        """
        transfer_function = np.zeros((256, 4), dtype=np.float32)
        for i in range(256):
            intensity = i / 255.0
            if intensity < 0.2:
                # Background: Transparent
                transfer_function[i] = [0.0, 0.0, 0.0, 0.0]
            elif 0.2 <= intensity < 0.5:
                transfer_function[i] = [
                    0.6, 0.8, 0.9,  # Adjusted color for spine
                    max(0.5, (intensity - 0.2) * 2.0)  # Ensure sufficient opacity
                ]
            elif 0.5 <= intensity < 0.8:
                # Bone/Dense structures: White with high opacity
                t = (intensity - 0.5) / 0.3
                transfer_function[i] = [
                    0.9 + t * 0.1,  # Transitioning to bright white
                    0.9 + t * 0.1,
                    0.9 + t * 0.1,
                    0.3 + t * 0.3  # Lower opacity
                ]
            else:
                # Very dense structures: Pure white, full opacity
                transfer_function[i] = [1.0, 1.0, 1.0, 1.0]
        return transfer_function

    def setup_textures(self):
        """
        Unified method for setting up volume and transfer function textures.
        """
        # Setup Volume Texture
        self.volume_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.volume_texture)

        # Normalize volume data
        normalized_volume_data = (self.volume_data - self.volume_data.min()) / (self.volume_data.max() - self.volume_data.min())
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, self.volume_data.shape[2], self.volume_data.shape[1], self.volume_data.shape[0], 0, GL_RED, GL_FLOAT, normalized_volume_data)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        # Setup Transfer Function Texture
        self.transfer_function_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.transfer_function_texture)
        transfer_function = self.create_transfer_function()
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, len(transfer_function), 0, GL_RGBA, GL_FLOAT, transfer_function)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)

    def update_transfer_function(self, new_transfer_function):
        glBindTexture(GL_TEXTURE_1D, self.transfer_function_texture)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, len(new_transfer_function), GL_RGBA, GL_FLOAT, new_transfer_function)
        print("Transfer function texture updated.")

    def setup_geometry(self):
        """Create cube geometry for volume rendering."""
        cube_vertices = np.array([
            # Positions        # Texture Coords
            -1, -1, -1,        0, 0, 0,
             1, -1, -1,        1, 0, 0,
             1,  1, -1,        1, 1, 0,
            -1,  1, -1,        0, 1, 0,
            -1, -1,  1,        0, 0, 1,
             1, -1,  1,        1, 0, 1,
             1,  1,  1,        1, 1, 1,
            -1,  1,  1,        0, 1, 1
        ], dtype=np.float32)

        cube_indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front
            4, 5, 6, 6, 7, 4,  # Back
            0, 1, 5, 5, 4, 0,  # Bottom
            2, 3, 7, 7, 6, 2,  # Top
            0, 3, 7, 7, 4, 0,  # Left
            1, 2, 6, 6, 5, 1   # Right
        ], dtype=np.uint32)

        # Generate VAO and VBO
        self.cube_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cube_vao)

        self.cube_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
        glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)

        self.cube_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cube_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)

        # Define vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def update_surface_voxel_texture(self):
        if self.surface_voxels is None:
            self.surface_voxels = sbs.create_surface_voxel_structure(self.volume_data)
            print(f"Surface Voxels create Count: {len(self.surface_voxels)}")

        # Initialize the surface voxel texture if it doesn't exist
        if not hasattr(self, 'surface_voxel_texture') or self.surface_voxel_texture is None:
            self.surface_voxel_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.surface_voxel_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        if self.surface_voxels:
            # Convert surface voxel data to a format suitable for OpenGL texture
            surface_data = np.array([(v[0], v[1], v[2], *v[3]['normal']) for v in self.surface_voxels], dtype=np.float32)

            # Limit the width of the texture to a reasonable size
            max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
            texture_width = min(surface_data.shape[0], max_texture_size)

            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texture_width, 1, 0, GL_RGBA, GL_FLOAT, surface_data[:texture_width])
        else:
            print("Warning: surface_voxels is empty")

    def update_intermediate_image_texture(self, output_image_size):
        if self.surface_voxels is None:
            print("Warning: surface_voxels is None. Cannot create intermediate image.")
            return
        if not hasattr(self, 'intermediate_image_texture') or self.intermediate_image_texture is None:
            print("Creating intermediate image texture.")
            self.intermediate_image_texture = glGenTextures(1)
        intermediate_image = sbs.sbs_projection(self.surface_voxels, output_image_size)
        glBindTexture(GL_TEXTURE_2D, self.intermediate_image_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, output_image_size[0], output_image_size[1], 0, GL_RED, GL_FLOAT, intermediate_image)

    def compute_gradients(self, volume_data):
        """
        Compute gradients for each voxel using a 3x3x3 kernel.
        """
        grad_x = ndimage.sobel(volume_data, axis=0, mode='constant')
        grad_y = ndimage.sobel(volume_data, axis=1, mode='constant')
        grad_z = ndimage.sobel(volume_data, axis=2, mode='constant')
        magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        gradient_field = np.stack((grad_x, grad_y, grad_z), axis=-1)
        return magnitude, gradient_field

    def compute_two_level_gradients(self, volume_data):
        """
        Compute two-level gradients: smooth for base structures and sharp for fine details.
        """
        # Base structure gradients (smoothed)
        smoothed_volume = ndimage.gaussian_filter(volume_data, sigma=1.0)
        base_gradients = np.gradient(smoothed_volume)
        
        # Fine detail gradients (sharpened)
        sharpened_volume = volume_data - smoothed_volume  # Residual highlighting fine details
        detail_gradients = np.gradient(sharpened_volume)
        return base_gradients, detail_gradients

    def setup_gradient_textures(self):
        """
        Create and upload base and detail gradient textures to the GPU.
        """
        base_gradients, detail_gradients = self.compute_two_level_gradients(self.volume_data)
        
        # Ensure base gradients
        if not hasattr(self, 'gradient_texture_base') or self.gradient_texture_base is None:
            self.gradient_texture_base = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.gradient_texture_base)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, 
                    self.volume_data.shape[2], self.volume_data.shape[1], self.volume_data.shape[0], 
                    0, GL_RGB, GL_FLOAT, np.stack(base_gradients, axis=-1))
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Ensure detail gradients
        if not hasattr(self, 'gradient_texture_detail') or self.gradient_texture_detail is None:
            self.gradient_texture_detail = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.gradient_texture_detail)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, 
                    self.volume_data.shape[2], self.volume_data.shape[1], self.volume_data.shape[0], 
                    0, GL_RGB, GL_FLOAT, np.stack(detail_gradients, axis=-1))
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def generate_transfer_function(self, brightness_factor=1.0):
        """
        Generate a simple transfer function dynamically with adjustable brightness.
        """
        transfer_function = np.zeros((256, 4), dtype=np.float32)
        for i in range(256):
            intensity = i / 255.0
            transfer_function[i] = [
                intensity * brightness_factor,  # Red
                intensity * 0.5 * brightness_factor,  # Green
                intensity * 0.2 * brightness_factor,  # Blue
                intensity  # Alpha
            ]
        return transfer_function

    def setup_spine_segmentation_texture(self):
        self.spine_segmentation_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.spine_segmentation_texture)
        glTexImage3D(
            GL_TEXTURE_3D, 0, GL_RED,
            self.spine_segmentation.shape[2],
            self.spine_segmentation.shape[1],
            self.spine_segmentation.shape[0],
            0, GL_RED, GL_UNSIGNED_BYTE, self.spine_segmentation
        )
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def bind_textures(self, shader):
        try:
            # Ensure surface voxel texture is created
            if not hasattr(self, 'surface_voxel_texture') or self.surface_voxel_texture is None:
                self.update_surface_voxel_texture()

            # Bind volume texture
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self.volume_texture)
            shader.setInt("volumeTexture", 0)
            
            # Bind surface voxel texture
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.surface_voxel_texture)
            shader.setInt("surfaceVoxelTexture", 1)

            # Bind intermediate image texture
            glActiveTexture(GL_TEXTURE2)
            if not hasattr(self, 'intermediate_image_texture') or self.intermediate_image_texture is None:
                # Create a dummy texture if not exists
                self.intermediate_image_texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, self.intermediate_image_texture)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT, np.zeros((1, 1), dtype=np.float32))

            glBindTexture(GL_TEXTURE_2D, self.intermediate_image_texture)
            shader.setInt("intermediateImageTexture", 2)

            # Bind transfer function texture
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_1D, self.transfer_function_texture)
            shader.setInt("transferFunction", 3)

            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_3D, self.spine_segmentation_texture)
            self.shader.setInt("spineSegmentationTexture", 2)

            # Bind gradient textures
            if hasattr(self, 'gradient_texture_base') and self.gradient_texture_base is not None:
                glActiveTexture(GL_TEXTURE4)
                glBindTexture(GL_TEXTURE_3D, self.gradient_texture_base)
                shader.setInt("gradientTextureBase", 4)
            else:
                print("Warning: gradient_texture_base not found.")

            if hasattr(self, 'gradient_texture_detail') and self.gradient_texture_detail is not None:
                glActiveTexture(GL_TEXTURE5)
                glBindTexture(GL_TEXTURE_3D, self.gradient_texture_detail)
                shader.setInt("gradientTextureDetail", 5)
            else:
                print("Warning: gradient_texture_detail not found.")

            shader.setFloat("density", 1.5)
            shader.setFloat("brightness", 0.2)
            shader.setFloat("contrast", 1.5)
            shader.setFloat("opacity_multiplier", 0.8)
            shader.setFloat("opacityThreshold", 0.5)  # Adjusted for better visibility
            shader.setFloat("edgeThreshold", 0.2)  # Increased for more pronounced edges
            shader.setVec3("edgeColor", glm.vec3(1.0, 0.9, 0.8))  # Bone-like edge color
            shader.setFloat("ambientStrength", 0.3)
            shader.setFloat("diffuseStrength", 0.7)
            shader.setFloat("specularStrength", 0.4)
            shader.setFloat("shininess", 16.0)
            shader.setVec3("lightDirection", glm.vec3(0.0, -1.0, -1.0))  # Adjusted light direction
            shader.setFloat("stepSize", 0.01)
            shader.setInt("maxSteps", 500)
            ray_direction = glm.normalize(self.camera_target - self.camera_position)
            shader.setVec3("rayDirection", ray_direction)
            # shader.setVec3("spineColor", glm.vec3(1.0, 0.5, 0.5))  # Example: reddish for the spine

            # Set clipping planes
            shader.setVec3("clipMin", self.clip_min)
            shader.setVec3("clipMax", self.clip_max)

            # Set debugging mode
            self.shader.setInt("debugMode", 0)  # 0: Normal rendering, 1: Base gradients, 2: Detail gradients

        except Exception as e:
            print(f"Error in bind_textures: {e}")
            import traceback
            traceback.print_exc()        

    def draw(self):
        try:
            self.shader.use()
            self.bind_textures(self.shader)
            
            glBindVertexArray(self.cube_vao)
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            
        except Exception as e:
            print(f"Error in draw: {e}")
            import traceback
            traceback.print_exc()

    def detect_features(self):
        # Simple edge detection using Sobel filter
        edges = ndimage.sobel(self.volume_data)
        threshold = edges.mean() * 2
        feature_mask = edges > threshold
        return feature_mask

    def update_feature_adaptive_lens(self):
        feature_mask = self.detect_features()
        # Update lens position to center on the largest feature
        labeled, num_features = ndimage.label(feature_mask)
        largest_feature = ndimage.find_objects(labeled)[0]
        center = [(s.start + s.stop) / 2 for s in largest_feature]
        self.lens['position'] = glm.vec2(center[1] / self.volume_data.shape[1], center[0] / self.volume_data.shape[0])

    def cleanup(self):
        """Cleanup resources."""
        if self.volume_texture:
            glDeleteTextures([self.volume_texture])
        if self.cube_vbo:
            glDeleteBuffers(1, [self.cube_vbo])
        if self.cube_vao:
            glDeleteVertexArrays(1, [self.cube_vao])

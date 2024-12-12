import numpy as np
import glm
from OpenGL.GL import *
from OpenGL.GL import shaders

class ShearWarpRenderer:
    def __init__(self, volume_data, width, height):
        self.initialized = False
        self.volume_data = volume_data
        self.volume_dimensions = volume_data.shape
        self.sbs_data = self._convert_to_sbs(volume_data)
        self.volume_texture = None
        self.index_texture = None
        self.attribute_texture = None
        self.gradient_texture = None
        self.width = width
        self.height = height

        if volume_data is None or volume_data.size == 0:
            raise ValueError("Invalid volume data")
    
        print(f"Volume dimensions: {self.volume_dimensions}")
        if any(dim <= 0 for dim in self.volume_dimensions):
            raise ValueError("Invalid volume dimensions")
    

        self.params = {
            'density': 1.0,
            'brightness': 1.0,
            'contrast': 1.0,
            'edge_weight': 0.5
        }
        self.view_matrix = None
        self.projection_matrix = None
        self.shear_matrix = None
        self.warp_matrix = None
        self.init_gl()

    def _convert_to_sbs(self, volume_data):
        """Convert volume data to Slice-Based Binary Shell structure"""
        dims = volume_data.shape
        
        # Add validation
        if len(dims) != 3:
            raise ValueError(f"Volume must be 3D, got shape {dims}")
        
        sbs_data = {
            'index_array': [],
            'attribute_list': [],
            'slice_offsets': np.zeros(dims[0], dtype=np.int32)
        }
            
        # Process each slice
        current_offset = 0
        for z in range(dims[0]):
            slice_data = volume_data[z]
            
            # Find non-empty voxels
            non_empty = np.where(slice_data > 0)
            
            # Store indices
            indices = np.stack([non_empty[0], non_empty[1]], axis=1)
            sbs_data['index_array'].extend(indices)
            
            # Store attributes (voxel values)
            attributes = slice_data[non_empty]
            sbs_data['attribute_list'].extend(attributes)
            
            # Update slice offset
            sbs_data['slice_offsets'][z] = current_offset
            current_offset += len(indices)
        
        # Convert lists to numpy arrays
        sbs_data['index_array'] = np.array(sbs_data['index_array'], dtype=np.int32)
        sbs_data['attribute_list'] = np.array(sbs_data['attribute_list'], dtype=np.uint8)
        
        return sbs_data

    def init_gl(self):
        """Initialize OpenGL textures and validate data for volume rendering."""
        if self.initialized:
            return

        try:
            # Validate volume dimensions
            if len(self.volume_dimensions) != 3:
                raise ValueError(f"Invalid volume dimensions: {self.volume_dimensions}")

            # Get OpenGL maximum texture size
            max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
            if any(dim > max_texture_size for dim in self.volume_dimensions):
                raise ValueError(f"Volume dimensions exceed OpenGL maximum texture size: {max_texture_size}")

            # Optimize data type for memory efficiency
            if self.volume_data.dtype != np.uint8:
                self.volume_data = self.volume_data.astype(np.uint8)
                print("Converted volume data to uint8 for memory optimization.")

            # Generate OpenGL textures
            if self.volume_texture:
                glDeleteTextures([self.volume_texture])  # Clean up old texture
            self.volume_texture = glGenTextures(1)

            # Bind texture and set parameters
            glBindTexture(GL_TEXTURE_3D, self.volume_texture)
            self._setup_3d_texture_parameters()

            # Attempt to initialize 3D texture
            try:
                print(f"Initializing 3D texture with dimensions: {self.volume_dimensions}")
                glTexImage3D(GL_TEXTURE_3D, 0, GL_R8,
                            self.volume_dimensions[2], self.volume_dimensions[1], self.volume_dimensions[0],
                            0, GL_RED, GL_UNSIGNED_BYTE, self.volume_data)
                glGenerateMipmap(GL_TEXTURE_3D)  # Generate mipmaps for better performance
            except Exception as e:
                print(f"Error initializing 3D texture: {e}. Falling back to smaller test volume.")
                # Use a smaller test volume for fallback
                test_volume = self.volume_data[:64, :64, :64]
                glTexImage3D(GL_TEXTURE_3D, 0, GL_R8,
                            test_volume.shape[2], test_volume.shape[1], test_volume.shape[0],
                            0, GL_RED, GL_UNSIGNED_BYTE, test_volume)

            # Initialize SBS textures (Index/Attribute Textures)
            self.index_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self.index_texture)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

            index_array = self.sbs_data['index_array']
            if index_array.size > 0:
                try:
                    tex_depth = len(self.sbs_data['slice_offsets'])
                    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RG32I,
                                max(1, self.volume_dimensions[1]), max(1, self.volume_dimensions[2]), tex_depth,
                                0, GL_RG_INTEGER, GL_INT, index_array)
                except Exception as e:
                    print(f"Error creating index texture: {e}. Skipping.")

            # Mark as initialized
            self.initialized = True
            print("OpenGL textures initialized successfully.")

        except Exception as e:
            print(f"Error in init_gl: {e}")
            self.initialized = False
            # Provide a fallback rendering path
            print("Falling back to standard rendering.")
            self.volume_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_3D, self.volume_texture)
            self._setup_3d_texture_parameters()
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R8,
                        self.volume_dimensions[2], self.volume_dimensions[1], self.volume_dimensions[0],
                        0, GL_RED, GL_UNSIGNED_BYTE, self.volume_data)

    def _setup_3d_texture_parameters(self):
        """Set up common 3D texture parameters"""
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)


    def _calculate_gradients(self):
        # Calculate gradients for fine detail identification
        gradients = np.gradient(self.volume_data)
        self.gradients = np.stack(gradients, axis=-1).astype(np.float32)

    def update_matrices(self, camera_pos, view_matrix, projection_matrix):
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix
        
        view_direction = glm.normalize(camera_pos)
        principal_axis = self._get_principal_axis(view_direction)
        
        self.shear_matrix = self._calculate_shear_matrix(view_direction, principal_axis)
        self.warp_matrix = self._calculate_warp_matrix(view_direction, principal_axis)

    def _get_principal_axis(self, view_direction):
        """Determine principal viewing axis"""
        abs_dir = glm.abs(view_direction)
        if abs_dir.x >= abs_dir.y and abs_dir.x >= abs_dir.z:
            return 'x'
        elif abs_dir.y >= abs_dir.x and abs_dir.y >= abs_dir.z:
            return 'y'
        else:
            return 'z'
            
    def _calculate_shear_matrix(self, view_direction, principal_axis):
        """Calculate shear transformation matrix"""
        shear = glm.mat4(1.0)
        
        if principal_axis == 'x':
            shear[1][0] = -view_direction.y / view_direction.x
            shear[2][0] = -view_direction.z / view_direction.x
        elif principal_axis == 'y':
            shear[0][1] = -view_direction.x / view_direction.y
            shear[2][1] = -view_direction.z / view_direction.y
        else:  # z axis
            shear[0][2] = -view_direction.x / view_direction.z
            shear[1][2] = -view_direction.y / view_direction.z
            
        return shear
        
    def _calculate_warp_matrix(self, view_direction, principal_axis):
        """Calculate 2D warp matrix for final image transformation"""
        # Create perspective matrix for principal axis
        if principal_axis == 'x':
            warp = glm.mat4(1.0)
            warp[1][1] = 1.0 / (1.0 + view_direction.y * view_direction.y)
            warp[2][2] = 1.0 / (1.0 + view_direction.z * view_direction.z)
        elif principal_axis == 'y':
            warp = glm.mat4(1.0)
            warp[0][0] = 1.0 / (1.0 + view_direction.x * view_direction.x)
            warp[2][2] = 1.0 / (1.0 + view_direction.z * view_direction.z)
        else:  # z axis
            warp = glm.mat4(1.0)
            warp[0][0] = 1.0 / (1.0 + view_direction.x * view_direction.x)
            warp[1][1] = 1.0 / (1.0 + view_direction.y * view_direction.y)
            
        return warp
    
    def _render_volume_slices(self, shader):
        # Bind necessary textures and set uniforms
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.volume_texture)
        shader.setInt("volume_texture", 0)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.index_texture)
        shader.setInt("index_texture", 1)
        
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.attribute_texture)
        shader.setInt("attribute_texture", 2)
        
        # Render slices
        num_slices = self.volume_dimensions[0]  # Assuming Z is the first dimension
        for i in range(num_slices):
            # Set slice-specific uniforms
            shader.setFloat("slice_index", i / float(num_slices))
            
            # Render a quad for each slice
            self._render_slice_quad()

    def update_volume_data(self, new_volume_data):
        """Update volume data and recalculate necessary textures."""
        self.volume_data = new_volume_data
        self.volume_dimensions = new_volume_data.shape
        self.sbs_data = self._convert_to_sbs(new_volume_data)
        
        # Update textures
        glBindTexture(GL_TEXTURE_3D, self.volume_texture)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, 
                    self.volume_dimensions[2], self.volume_dimensions[1], self.volume_dimensions[0],
                    0, GL_RED, GL_UNSIGNED_BYTE, self.volume_data)
        
        # Recalculate gradients if needed
        self._calculate_gradients()
        
        # Reinitialize OpenGL settings if necessary
        if not self.initialized:
            self.init_gl()

    def _render_slice_quad(self):
        # Define quad vertices
        vertices = np.array([
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            1.0,  1.0, 0.0,
            -1.0,  1.0, 0.0
        ], dtype=np.float32)
        
        # Create and bind VAO and VBO
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Set vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
        glEnableVertexAttribArray(0)
        
        # Draw quad
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        
        # Clean up
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])

    def render(self, shader):
        if not self.initialized:
            self.init_gl()
        shader.use()
        shader.setMat4("view_matrix", self.view_matrix)
        shader.setMat4("projection_matrix", self.projection_matrix)
        shader.setMat4("shear_matrix", self.shear_matrix)
        shader.setMat4("warp_matrix", self.warp_matrix)
        shader.setFloat("density", self.params['density'])
        shader.setFloat("brightness", self.params['brightness'])
        shader.setFloat("contrast", self.params['contrast'])
        shader.setFloat("edge_weight", self.params['edge_weight'])
        # shader.setInt("tex_width", tex_width)
        # shader.setInt("tex_height", tex_height)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.volume_texture)
        shader.setInt("volume_texture", 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.index_texture)
        shader.setInt("index_texture", 1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.attribute_texture)
        shader.setInt("attribute_texture", 2)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_3D, self.gradient_texture)
        shader.setInt("gradient_texture", 3)

        intermediate_texture = self._render_intermediate_buffer(shader)
        self._apply_warp_transformation(shader, intermediate_texture)

    def _render_fullscreen_quad(self):
        # Define quad vertices for a fullscreen quad
        vertices = np.array([
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            1.0,  1.0, 0.0,
            -1.0,  1.0, 0.0
        ], dtype=np.float32)
        
        # Create and bind VAO and VBO
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Set vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
        glEnableVertexAttribArray(0)
        
        # Draw fullscreen quad
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        
        # Clean up
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
    def _render_intermediate_buffer(self, shader):
        """Render volume to intermediate buffer using shear transformation"""
        # Set up framebuffer for intermediate rendering
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        # Create texture for intermediate result
        intermediate_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, intermediate_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, intermediate_texture, 0)
        
        # Render sheared slices
        self._render_volume_slices(shader)
        
        # Clean up
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [fbo])
        
        return intermediate_texture
    
    def _apply_warp_transformation(self, shader, intermediate_texture):
        """Apply warp transformation to intermediate result"""
        shader.use()
        shader.setInt("intermediate_texture", 0)
        shader.setMat4("warp_matrix", self.warp_matrix)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, intermediate_texture)
        
        # Render full-screen quad with warp transformation
        self._render_fullscreen_quad()
        
        # Clean up
        glDeleteTextures(1, [intermediate_texture])

    def update_parameters(self, **params):
        self.params.update(params)

    def delete(self):
        if self.volume_texture:
            glDeleteTextures([self.volume_texture])
        if self.index_texture:
            glDeleteTextures([self.index_texture])
        if self.attribute_texture:
            glDeleteTextures([self.attribute_texture])
        if self.gradient_texture:
            glDeleteTextures([self.gradient_texture])
        self.initialized = False





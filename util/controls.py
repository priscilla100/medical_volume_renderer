import imgui
import numpy as np
from OpenGL.GL import *
from imgui.integrations.glfw import GlfwRenderer

class VolumeControls:
    def __init__(self, volume, shader_program, window):
        self.volume = volume
        self.shader_program = shader_program
        self.window = window
        self.imgui_context = None
        self.impl = None

        # Control state
        self.show_transfer_function = True
        self.show_edge_detection = True

        # Transfer function editor state
        self.selected_point = -1
        self.tf_canvas_size = (200, 100)

        # Initialize the params dictionary with more attributes
        self.params = {
            'max_steps': 256,
            'step_size': 0.004,
            'density_threshold': 0.01,
            'opacity_multiplier': 0.5,
            'color_map_type': 0,
            'ambient_intensity': 0.3,
            'diffuse_intensity': 0.7,
            'specular_intensity': 0.2,
            'shininess': 32.0,
            'gradient_threshold': 0.5,
            'edge_enhancement': 1.0,
            'color_shift': 0.0,
            'color_saturation': 1.0,
            'shading_mode': 0,
            'rendering_mode': 0,
            'translucency': 0.3,
            'zoom': 1.0,
            'brightness': 0.1, # Default brightness value
            'contrast': 1.2,   # Default contrast value
            'light_positions': [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]],
            'view_position': [0.0, 0.0, 3.0],
            'density': 1.0,
            'edge_weight': 0.5,
            'color_points': [],
            'opacity_points': [],
        }

        self.temp_params = self.params.copy()
        self.params_changed = False

        # Options for dropdowns
        self.color_maps = ["Rainbow", "Grayscale", "Blue-Red", "Green-Magenta", "Plasma"]
        self.shading_modes = ["Phong", "Cel Shading", "PBR-inspired"]
        self.rendering_modes = ["Ray Marching", "Maximum Intensity Projection", "Average Intensity Projection"]

        # Initialize ImGui
        self.init_imgui(window)

    def init_imgui(self, window):
        """Initialize Dear ImGui"""
        imgui.create_context()
        self.impl = GlfwRenderer(window)

    def render(self):
        """Render the control panel"""
        self.params_changed = False # Reset change flag

        imgui.begin("Volume Rendering Controls")

        changed = False

        # Basic parameters for controlling volume rendering attributes.
        _, self.temp_params['brightness'] = imgui.slider_float("Brightness", self.temp_params['brightness'], -1.0, 1.0)
        changed |= _

        _, self.temp_params['contrast'] = imgui.slider_float("Contrast", self.temp_params['contrast'], 0.5, 2.0)
        changed |= _

        _, self.temp_params['edge_weight'] = imgui.slider_float("Edge Weight", self.temp_params['edge_weight'], 0.0, 2.0)
        changed |= _

        if imgui.tree_node("Lighting"):
            _, self.temp_params['ambient_intensity'] = imgui.slider_float("Ambient Intensity", self.temp_params['ambient_intensity'], 0.0, 1.0)
            changed |= _
            
            _, self.temp_params['diffuse_intensity'] = imgui.slider_float("Diffuse Intensity", self.temp_params['diffuse_intensity'], 0.0, 1.0)
            changed |= _
            
            _, self.temp_params['specular_intensity'] = imgui.slider_float("Specular Intensity", self.temp_params['specular_intensity'], 0.0, 1.0)
            changed |= _
            
            _, self.temp_params['shininess'] = imgui.slider_float("Shininess", self.temp_params['shininess'], 1.0, 128.0)
            changed |= _
            
            imgui.tree_pop()

        if changed:
            self.params_changed = True
        
        imgui.end()

    def update_shader_uniforms(self):
        """Update all shader uniforms with current parameter values."""
        shader = self.shader_program
        shader.use()

        # Update shader uniforms
        for param, value in self.params.items():
            if param == 'light_positions':
                for i, light_pos in enumerate(value):
                    shader.setVec3(f"lightPos[{i}]", light_pos)
                    
            elif param == 'view_position':
                shader.setVec3("viewPos", value)

            elif isinstance(value, float):
                shader.setFloat(param, value)

            elif isinstance(value, int):
                shader.setInt(param, value)

    def has_changes(self):
        """Check if parameters have changed."""
        return self.params_changed

    def get_current_params(self):
        """Get current parameter values."""
        return self.temp_params

    def cleanup(self):
       """Cleanup ImGui resources."""
       if self.impl:
           self.impl.shutdown()
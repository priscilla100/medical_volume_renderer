import imgui
import numpy as np
from OpenGL.GL import *
from imgui.integrations.glfw import GlfwRenderer
import glm
from shape.volume_renderer import VolumeRenderer

class VolumeControls:
    def __init__(self, volume, shader_program, window):
        self.volume = volume
        self.shader_program = shader_program
        self.window = window
        
        # Control state
        self.show_transfer_function = True
        self.show_edge_detection = True
        
        # Transfer function editor state
        self.selected_point = -1
        self.tf_canvas_size = (300, 150)  # Increased canvas size
        self.transfer_function_mode = 0  # 0: Color, 1: Opacity
        
        # Initialize the params dictionary with more comprehensive defaults
        self.params = {
            'max_steps': 256,
            'step_size': 0.004,
            'density_threshold': 0.05,
            'color_map_type': 0,
            'ambient_intensity': 0.3,
            'diffuse_intensity': 0.7,
            'specular_intensity': 0.2,
            'shininess': 32.0,
            'gradient_threshold': 0.05,
            'edge_enhancement': 1.0,
            'color_shift': 0.0,
            'color_saturation': 1.0,
            'shading_mode': 0,
            'rendering_mode': 0,
            'translucency': 0.18,
            'zoom': 1.0,
            'view_position': [0.0, 0.0, 3.0],
            'edge_weight': 0.5,
            'spine_color': [0.9, 0.8, 0.7],  # Bone-like color
            'density': 1.0,
            'brightness': 0.1,
            'contrast': 1.20,
            'opacity_multiplier': 1.0,

            # Enhanced transfer function points
            'color_points': [
                (0.0, [0.2, 0.2, 0.8]),    # Start with a deeper blue
                (0.5, [0.5, 0.5, 1.0]),    # Lighter blue in middle
                (1.0, [0.8, 0.8, 1.0])     # Very light blue at end
            ],
            'opacity_points': [
                (0.0, 0.1),  # Start with low opacity
                (0.5, 0.5),  # Increase opacity in middle
                (1.0, 0.8)   # Higher opacity at end
            ]
        }
        
        self.temp_params = self.params.copy()
        self.params_changed = False
        self.lens_active = False
        self.lens_position = [0.5, 0.5]
        self.lens_size = [0.2, 0.2]
        self.lens_magnification = 2.0
        
        # Options for dropdowns with more comprehensive selections
        self.color_maps = [
            "Rainbow", "Grayscale", "Blue-Red", "Green-Magenta", "Plasma", "Viridis", "Inferno"
        ]
        self.shading_modes = [
            "Phong", "Cel Shading", "PBR-inspired", "Toon Shading"  # "Metallic"
        ]
        self.rendering_modes = [
            "Ray Marching", "Maximum Intensity Projection", "Average Intensity Projection", "First Hit Projection", "Ambient Occlusion"
        ]
        self.lens = {
            'active': False,
            'position': glm.vec2(0.5, 0.5),  # Normalized screen coordinates
            'size': glm.vec2(0.2, 0.2),      # Normalized size
            'magnification': 2.0
        }
        
        # Initialize ImGui
        self.init_imgui(window)

    def init_imgui(self, window):
        """Initialize Dear ImGui with enhanced styling"""
        imgui.create_context()
        self.impl = GlfwRenderer(window)
        
        # Advanced ImGui styling
        style = imgui.get_style()
        style.window_rounding = 5.0
        style.frame_rounding = 3.0
        style.grab_rounding = 3.0
        style.alpha = 0.9
        style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.1, 0.1, 0.1, 0.9)
        style.colors[imgui.COLOR_TEXT] = (0.9, 0.9, 0.9, 1.0)
        style.colors[imgui.COLOR_HEADER] = (0.2, 0.2, 0.2, 1.0)
        style.colors[imgui.COLOR_HEADER_ACTIVE] = (0.3, 0.3, 0.3, 1.0)

    def render(self):
        self.params_changed = False
        imgui.begin("Volume Rendering Controls", True)
        
        if imgui.tree_node("Raycasting"):
            changed, self.temp_params['edge_weight'] = imgui.slider_float("Edge Weight", self.temp_params['edge_weight'], 0.0, 1.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['max_steps'] = imgui.slider_int("Max Steps", self.temp_params['max_steps'], 64, 512)
            self.params_changed |= changed
            changed, self.temp_params['step_size'] = imgui.slider_float("Step Size", self.temp_params['step_size'], 0.001, 0.01, format="%.3f")
            self.params_changed |= changed
            imgui.tree_pop()

        if imgui.tree_node("Color Map"):
            changed, self.temp_params['density_threshold'] = imgui.slider_float("Density Threshold", self.temp_params['density_threshold'], -1.0, 1.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['color_shift'] = imgui.slider_float("Color Shift", self.temp_params['color_shift'], -1.0, 1.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['color_saturation'] = imgui.slider_float("Color Saturation", self.temp_params['color_saturation'], -1.0, 2.0, format="%.2f")
            self.params_changed |= changed
            imgui.tree_pop()

        if imgui.tree_node("Lighting"):
            changed, self.temp_params['ambient_intensity'] = imgui.slider_float("Ambient", self.temp_params['ambient_intensity'], -1.0, 1.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['diffuse_intensity'] = imgui.slider_float("Diffuse", self.temp_params['diffuse_intensity'], -1.0, 1.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['specular_intensity'] = imgui.slider_float("Specular", self.temp_params['specular_intensity'], -1.0, 1.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['shininess'] = imgui.slider_float("Shininess", self.temp_params['shininess'], 1.0, 128.0, format="%.1f")
            self.params_changed |= changed
            imgui.tree_pop()

        if imgui.tree_node("Edge Detection"):
            changed, self.temp_params['gradient_threshold'] = imgui.slider_float("Gradient Threshold", self.temp_params['gradient_threshold'], -1.0, 1.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['edge_enhancement'] = imgui.slider_float("Edge Enhancement", self.temp_params['edge_enhancement'], -1.0, 2.0, format="%.2f")
            self.params_changed |= changed
            imgui.tree_pop()

        if imgui.tree_node("Advanced Rendering"):
            changed, self.temp_params['translucency'] = imgui.slider_float("Translucency", self.temp_params['translucency'], -1.0, 1.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['zoom'] = imgui.slider_float("Zoom", self.temp_params['zoom'], 0.5, 2.0, format="%.2f")
            self.params_changed |= changed
            changed, self.temp_params['shading_mode'] = imgui.combo("Shading Mode", self.temp_params['shading_mode'], self.shading_modes)
            self.params_changed |= changed
            changed, self.temp_params['color_map_type'] = imgui.combo("Color Map", self.temp_params['color_map_type'], self.color_maps)
            self.params_changed |= changed
            changed, self.temp_params['rendering_mode'] = imgui.combo("Rendering Mode", self.temp_params['rendering_mode'], self.rendering_modes)
            self.params_changed |= changed
            imgui.tree_pop()

        if imgui.tree_node("Spine Settings"):
            changed, self.temp_params['spine_color'] = imgui.color_edit3("Spine Color", *self.temp_params['spine_color'])
            if changed:
                self.params_changed = True
                self.update_transfer_function()
            imgui.tree_pop()

        imgui.end()

        if self.params_changed:
            self.update_shader_uniforms()
            self.update_lens()

        return self.params_changed

    def update_lens(self):
        self.shader_program.use()
        self.shader_program.setInt("lens_active", int(self.lens['active']))
        self.shader_program.setVec2("lens_position", glm.vec2(*self.lens['position']))
        self.shader_program.setVec2("lens_size", glm.vec2(*self.lens['size']))
        self.shader_program.setFloat("lens_magnification", self.lens['magnification'])

    def _render_color_transfer_function_editor(self):
        """Render color point editor for transfer function"""
        draw_list = imgui.get_window_draw_list()
        canvas_pos = imgui.get_cursor_screen_pos()
        
        # Draw canvas background
        draw_list.add_rect_filled(
            canvas_pos[0], canvas_pos[1],
            canvas_pos[0] + self.tf_canvas_size[0],
            canvas_pos[1] + self.tf_canvas_size[1],
            imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 1.0)
        )
        
        # Draw color gradient
        for i in range(len(self.temp_params['color_points']) - 1):
            x1, color1 = self.temp_params['color_points'][i]
            x2, color2 = self.temp_params['color_points'][i+1]
            
            # Interpolate colors
            gradient = np.linspace(0, 1, 50)
            for j in range(len(gradient) - 1):
                t = gradient[j]
                color = [
                    color1[k] * (1-t) + color2[k] * t 
                    for k in range(3)
                ]
                
                x = canvas_pos[0] + (x1 * (1-t) + x2 * t) * self.tf_canvas_size[0]
                draw_list.add_line(
                    x, canvas_pos[1], 
                    x, canvas_pos[1] + self.tf_canvas_size[1],
                    imgui.get_color_u32_rgba(*color, 1.0)
                )
        
        # Draw control points
        for i, (value, color) in enumerate(self.temp_params['color_points']):
            x = canvas_pos[0] + value * self.tf_canvas_size[0]
            y = canvas_pos[1] + self.tf_canvas_size[1] / 2
            
            # Draw point
            draw_list.add_circle_filled(
                x, y, 6,
                imgui.get_color_u32_rgba(*color, 1.0)
            )
            
            # Color picker for each point
            if imgui.is_mouse_hovering_rect(
                canvas_pos[0] + value * self.tf_canvas_size[0] - 6, 
                canvas_pos[1] + self.tf_canvas_size[1] / 2 - 6,
                canvas_pos[0] + value * self.tf_canvas_size[0] + 6, 
                canvas_pos[1] + self.tf_canvas_size[1] / 2 + 6
            ):
                if imgui.is_mouse_clicked(1):  # Right-click
                    point_color = list(color)
                    imgui.open_popup("Color Edit")
                
                if imgui.begin_popup("Color Edit"):
                    _, point_color = imgui.color_edit3("Color", point_color)
                    
                    # Allow adding or removing points
                    if imgui.button("Remove Point") and len(self.temp_params['color_points']) > 2:
                        del self.temp_params['color_points'][i]
                        imgui.close_current_popup()
                    
                    if imgui.button("Add Point"):
                        # Insert new point midway
                        new_x = min(max(0, value + 0.1), 1)
                        new_color = point_color
                        self.temp_params['color_points'].insert(i+1, (new_x, new_color))
                    
                    if imgui.button("Close"):
                        self.temp_params['color_points'][i] = (value, point_color)
                        imgui.close_current_popup()
                    
                    imgui.end_popup()

    def _render_opacity_transfer_function_editor(self):
        """Render opacity point editor for transfer function"""
        # Similar structure to color editor, but focusing on opacity points
        draw_list = imgui.get_window_draw_list()
        canvas_pos = imgui.get_cursor_screen_pos()
        
        # Draw canvas background
        draw_list.add_rect_filled(
            canvas_pos[0], canvas_pos[1],
            canvas_pos[0] + self.tf_canvas_size[0],
            canvas_pos[1] + self.tf_canvas_size[1],
            imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 1.0)
        )
        
        # Draw opacity gradient
        for i in range(len(self.temp_params['opacity_points']) - 1):
            x1, opacity1 = self.temp_params['opacity_points'][i]
            x2, opacity2 = self.temp_params['opacity_points'][i+1]
            
            gradient = np.linspace(0, 1, 50)
            for j in range(len(gradient) - 1):
                t = gradient[j]
                opacity = opacity1 * (1-t) + opacity2 * t
                
                x = canvas_pos[0] + (x1 * (1-t) + x2 * t) * self.tf_canvas_size[0]
                draw_list.add_line(
                    x, canvas_pos[1], 
                    x, canvas_pos[1] + opacity * self.tf_canvas_size[1],
                    imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
                )
    
    # Similar point interaction logic as color editor...
    def set_shader_uniform(self, key, value):
        self.shader_program.use()
        if isinstance(value, float):
            self.shader_program.setFloat(key, value)
        elif isinstance(value, int):
            self.shader_program.setInt(key, value)
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            self.shader_program.setVec3(key, glm.vec3(*value))
        elif isinstance(value, glm.vec3):
            self.shader_program.setVec3(key, value)
        elif key == 'color_points':
            self.update_color_points(value)  # Call method to update transfer function
        elif key == 'opacity_points':
            self.update_opacity_points(value)  # Call method to update opacity transfer function
        else:
            # Original logic for other parameters
            super().set_shader_uniform(key, value)

    def apply_initial_parameters(self):
        initial_params = self.params
        for key, value in initial_params.items():
            try:
                self.set_shader_uniform(key, value)
            except Exception as e:
                print(f"Error setting uniform {key}: {e}")

    def update_shader_uniforms(self):
        shader = self.shader_program
        shader.use()
        for key, value in self.temp_params.items():
            try:
                if key == 'spine_color':
                    shader.setVec3("spineColor", glm.vec3(*value))
                elif isinstance(value, float):
                    shader.setFloat(key, value)
                elif isinstance(value, int):
                    shader.setInt(key, value)
                elif isinstance(value, (list, tuple)) and len(value) == 3:
                    shader.setVec3(key, glm.vec3(*value))
            except Exception as e:
                print(f"Error setting uniform {key}: {e}")
        # Update lens uniforms
        self.shader_program.setInt("lens_active", int(self.lens['active']))
        self.shader_program.setVec2("lens_position", glm.vec2(*self.lens['position']))
        self.shader_program.setVec2("lens_size", glm.vec2(*self.lens['size']))
        self.shader_program.setFloat("lens_magnification", self.lens['magnification'])

    def create_transfer_function(self):
        tf_data = np.zeros((256, 4), dtype=np.float32)
        # Update color points
        for i in range(len(self.temp_params['color_points']) - 1):
            x1, color1 = self.temp_params['color_points'][i]
            x2, color2 = self.temp_params['color_points'][i + 1]
            start = int(x1 * 255)
            end = int(x2 * 255)
            for j in range(start, end + 1):
                t = (j - start) / (end - start)
                tf_data[j, :3] = [c1 * (1 - t) + c2 * t for c1, c2 in zip(color1, color2)]
                tf_data[j, 3] = 1.0  # Fully opaque
        # Update opacity points
        for i in range(len(self.temp_params['opacity_points']) - 1):
            x1, opacity1 = self.temp_params['opacity_points'][i]
            x2, opacity2 = self.temp_params['opacity_points'][i + 1]
            start = int(x1 * 255)
            end = int(x2 * 255)
            for j in range(start, end + 1):
                t = (j - start) / (end - start)
                tf_data[j, 3] = opacity1 * (1 - t) + opacity2 * t
        return tf_data

    def update_transfer_function(self):
        tf_data = np.zeros((256, 4), dtype=np.float32)
        # Update color points
        for i in range(len(self.temp_params['color_points']) - 1):
            x1, color1 = self.temp_params['color_points'][i]
            x2, color2 = self.temp_params['color_points'][i + 1]
            start = int(x1 * 255)
            end = int(x2 * 255)
            for j in range(start, end + 1):
                t = (j - start) / (end - start)
                tf_data[j, :3] = [c1 * (1 - t) + c2 * t for c1, c2 in zip(color1, color2)]
                tf_data[j, 3] = 1.0  # Set alpha to fully opaque
        # Update opacity points
        for i in range(len(self.temp_params['opacity_points']) - 1):
            x1, opacity1 = self.temp_params['opacity_points'][i]
            x2, opacity2 = self.temp_params['opacity_points'][i + 1]
            start = int(x1 * 255)
            end = int(x2 * 255)
            for j in range(start, end + 1):
                t = (j - start) / (end - start)
                tf_data[j, 3] = opacity1 * (1 - t) + opacity2 * t
        glBindTexture(GL_TEXTURE_1D, self.volume.transfer_function_texture)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGBA, GL_FLOAT, tf_data)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGBA, GL_FLOAT, tf_data)
        print(f"Transfer function updated with {len(self.temp_params['color_points'])} color points and {len(self.temp_params['opacity_points'])} opacity points.")

    def update_color_points(self, color_points):
        # Update the transfer function texture for color points
        tf_data = np.zeros((256, 4), dtype=np.float32)
        for i in range(len(color_points) - 1):
            x1, color1 = color_points[i]
            x2, color2 = color_points[i + 1]
            start = int(x1 * 255)
            end = int(x2 * 255)
            for j in range(start, end + 1):
                t = (j - start) / (end - start)
                tf_data[j, :3] = [c1 * (1 - t) + c2 * t for c1, c2 in zip(color1, color2)]
                tf_data[j, 3] = 1.0  # Set alpha to 1.0
        glBindTexture(GL_TEXTURE_1D, self.volume.transfer_function_texture)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGBA, GL_FLOAT, tf_data)

    def update_opacity_points(self, opacity_points):
        # Update the transfer function texture for opacity points
        tf_data = np.zeros((256, 4), dtype=np.float32)
        for i in range(len(opacity_points) - 1):
            x1, y1 = opacity_points[i]
            x2, y2 = opacity_points[i + 1]
            start = int(x1 * 255)
            end = int(x2 * 255)
            for j in range(start, end + 1):
                t = (j - start) / (end - start)
                tf_data[j, 3] = y1 * (1 - t) + y2 * t
        glBindTexture(GL_TEXTURE_1D, self.volume.transfer_function_texture)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGBA, GL_FLOAT, tf_data)

    def has_changes(self):
        """Check if parameters have changed"""
        return self.params_changed

    def get_current_params(self):
        params = self.params.copy()
        # Debugging: Check all parameter types and values
        for key, value in params.items():
            if key == 'spine_color' or key in ['view_position']:
                if not (isinstance(value, (list, tuple)) and len(value) == 3):
                    print(f"Invalid Vec3 parameter: {key} -> {value}")
            elif key == 'color_points' or key == 'opacity_points':
                if not isinstance(value, list):
                    print(f"Invalid list parameter: {key} -> {value}")
        return params

    def cleanup(self):
        """Cleanup ImGui resources"""
        if self.impl:
            self.impl.shutdown()



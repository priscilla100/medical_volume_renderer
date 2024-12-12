import glfw
from OpenGL.GL import *
from glfw.GLFW import *
import glm
from util import Shader
from shape.volume_renderer import VolumeRenderer
import math
import numpy as np
from util.volume_controls import VolumeControls
import imgui
from imgui.integrations.glfw import GlfwRenderer
import time
import psutil

class App:
    def __init__(self, width=1280, height=1020, title="Medical Volume Renderer", dicom_path="data"):
        self.frame_times = []
        self.process = psutil.Process()
        self.width = width
        self.height = height
        self.title = title
        self.dicom_path = dicom_path
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        # Create GLFW Window
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        # OpenGL Settings
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # Initialize Shader
        self.shader = Shader(
            vert='shader/volume.vert',
            tese=None,
            tesc=None,
            frag='shader/volume.frag'
        )
        self.volume_renderer = VolumeRenderer(dicom_path, self.shader)
        self.volume_renderer.setup_textures()
        self.volume_renderer.setup_geometry()
        self.volume_renderer.spine_segmentation
        
        # Add print statements for debugging
        print("Volume data shape:", self.volume_renderer.volume_data.shape)
        print("Spine segmentation shape:", self.volume_renderer.spine_segmentation.shape)
        print("Volume Texture Shape:", self.volume_renderer.volume_data.shape)
        print("Spine Segmentation Shape:", self.volume_renderer.spine_segmentation.shape)
        
        self.controls = VolumeControls(self.volume_renderer, self.shader, self.window)
        self.controls.apply_initial_parameters()
        
        self.imgui_renderer = GlfwRenderer(self.window)  # Initialize imgui renderer
        
        # Camera and Projection
        self.camera_position = glm.vec3(0, 0, 3)
        self.camera_target = glm.vec3(0, 0, 0)
        self.camera_up = glm.vec3(0, 1, 0)
        self.camera_front = glm.vec3(0.0, 0.0, -1.0)
        self.camera_right = glm.vec3(1.0, 0.0, 0.0)
        self.camera_speed = 0.08
        self.yaw = -90.0
        self.pitch = 0.0
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        
        # Initialize mouse state
        self.first_mouse = True
        self.last_x = self.width / 2
        self.last_y = self.height / 2

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_renderer.process_inputs()  # Process ImGui inputs
            
            # Start new frame for ImGui
            imgui.new_frame()
            self.process_input()
            self.render()
            
            # Render ImGui
            imgui.render()
            self.imgui_renderer.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
        self.cleanup()

    def render(self):
        start_time = time.perf_counter()
        
        # Set default rendering parameters
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view_matrix = glm.lookAt(self.camera_position, self.camera_target, self.camera_up)
        projection_matrix = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 100.0)
        self.shader.use()
        self.shader.setMat4("model", glm.mat4(1.0))
        self.shader.setMat4("view", view_matrix)
        self.shader.setMat4("projection", projection_matrix)
        
        # Set clipping boundaries
        self.shader.setVec3("clipMin", glm.vec3(0.0))
        self.shader.setVec3("clipMax", glm.vec3(1.0))
        
        # Update Shader Parameters from Controls
        if self.controls.params_changed:
            self.controls.update_shader_uniforms()
            self.controls.update_transfer_function()  # Update transfer function
            self.controls.params_changed = False
        
        if self.controls.has_changes():
            params = self.controls.get_current_params()
            print("Updating shader parameters:")
            for key, value in params.items():
                print(f"Setting uniform {key}: {value} (type: {type(value)})")
                try:
                    if key == 'spine_color':
                        spine_color = self.controls.get_current_params()["spine_color"]
                        self.shader.setVec3("spineColor", glm.vec3(*spine_color))
                    else:
                        self.controls.set_shader_uniform(key, value)
                except Exception as e:
                    print(f"Error setting parameter {key}: {e}")
            self.controls.params_changed = False

        # Bind and Draw Volume
        self.volume_renderer.bind_textures(self.shader)
        self.volume_renderer.draw()
        
        # Render Controls Panel
        self.controls.render()
        
        end_time = time.perf_counter()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) >= 60:  # Calculate FPS every 60 frames
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1 / avg_frame_time
            print(f"Average FPS: {fps:.2f}")
            self.frame_times.clear()
        memory_usage = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
        print(f"Memory Usage: {memory_usage:.2f} MB")

    def cleanup(self):
        """Clean up OpenGL resources."""
        self.volume_renderer.cleanup()
        glfw.terminate()

    def process_input(self):
        # Camera movement
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera_position += self.camera_speed * self.camera_front
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera_position -= self.camera_speed * self.camera_front
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera_position -= self.camera_right * self.camera_speed
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera_position += self.camera_right * self.camera_speed

        # Adjust clipping planes
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.volume_renderer.clip_min[2] = min(self.volume_renderer.clip_min[2] + 0.01, 1.0)
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.volume_renderer.clip_min[2] = max(self.volume_renderer.clip_min[2] - 0.01, 0.0)
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.volume_renderer.clip_min[0] = min(self.volume_renderer.clip_min[0] + 0.01, 1.0)
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.volume_renderer.clip_min[0] = max(self.volume_renderer.clip_min[0] - 0.01, 0.0)

        # Adjust transfer function dynamically
        if glfw.get_key(self.window, glfw.KEY_T) == glfw.PRESS:
            new_transfer_function = self.volume_renderer.create_transfer_function()
            self.volume_renderer.update_transfer_function(new_transfer_function)

    def mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        # Clamp pitch to prevent screen flip
        self.pitch = max(-89.0, min(89.0, self.pitch))

        # Update camera front vector
        front = glm.vec3()
        front.x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        front.y = math.sin(glm.radians(self.pitch))
        front.z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.camera_front = glm.normalize(front)
        self.camera_right = glm.normalize(glm.cross(self.camera_front, self.camera_up))
        self.camera_up = glm.normalize(glm.cross(self.camera_right, self.camera_front))

    def cleanup(self):
        """Clean up resources."""
        self.controls.cleanup()
        self.volume_renderer.cleanup()
        self.imgui_renderer.shutdown()  # Shutdown ImGui renderer
        glfw.terminate()

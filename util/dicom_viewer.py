import os
import numpy as np
import pydicom
import moderngl
import numpy as np
from PIL import Image
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
import cv2

class DicomViewer:
    def __init__(self, window_size=(800, 600)):
        self.window_size = window_size
        self.initialize_glfw()
        self.current_slice = 0
        self.window_title = "DICOM Viewer"
        self.brightness = 1.0
        self.contrast = 1.0
        self.edge_detection = False
        self.transfer_function_mode = 0  # 0: grayscale, 1: rainbow, 2: hot
        
    def initialize_glfw(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
            
        # Create window with OpenGL context
        self.window = glfw.create_window(
            self.window_size[0], self.window_size[1], 
            self.window_title, None, None
        )
        
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
            
        glfw.make_context_current(self.window)
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
    def load_dicom_series(self, directory):
        """Load all DICOM files from directory"""
        self.dicom_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.dcm'):
                filepath = os.path.join(directory, filename)
                ds = pydicom.dcmread(filepath)
                self.dicom_files.append(ds)
                
        self.dicom_files.sort(key=lambda x: x.InstanceNumber)
        self.current_slice = 0
        print(f"Loaded {len(self.dicom_files)} DICOM files")
        
    def apply_edge_detection(self, image):
        """Apply Sobel edge detection"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return magnitude.astype(np.uint8)
        
    def apply_transfer_function(self, image):
        """Apply different transfer functions to the image"""
        if self.transfer_function_mode == 0:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif self.transfer_function_mode == 1:  # Rainbow
            colored = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        else:  # Hot
            colored = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
    def process_image(self):
        """Process current DICOM slice with all effects"""
        if not self.dicom_files:
            return None
            
        # Get current slice
        ds = self.dicom_files[self.current_slice]
        image = ds.pixel_array.astype(float)
        
        # Apply brightness and contrast
        image = image * self.contrast + self.brightness
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Apply edge detection if enabled
        if self.edge_detection:
            edges = self.apply_edge_detection(image)
            image = cv2.addWeighted(image, 0.7, edges, 0.3, 0)
            
        # Apply transfer function
        image = self.apply_transfer_function(image)
        
        return image
        
    def render(self):
        """Main render loop"""
        while not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT)
            
            # Process and display current image
            image = self.process_image()
            if image is not None:
                # Convert to OpenGL texture
                height, width = image.shape[:2]
                glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, image)
                
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard inputs"""
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_UP:
                self.brightness += 10
            elif key == glfw.KEY_DOWN:
                self.brightness -= 10
            elif key == glfw.KEY_RIGHT:
                self.contrast += 0.1
            elif key == glfw.KEY_LEFT:
                self.contrast -= 0.1
            elif key == glfw.KEY_E:
                self.edge_detection = not self.edge_detection
            elif key == glfw.KEY_T:
                self.transfer_function_mode = (self.transfer_function_mode + 1) % 3
                
    def scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll for slice navigation"""
        if self.dicom_files:
            self.current_slice = (self.current_slice + int(yoffset)) % len(self.dicom_files)
            
    def cleanup(self):
        """Cleanup resources"""
        glfw.terminate()

# # Example usage
# if __name__ == "__main__":
#     viewer = DicomViewer()
#     # Replace with your DICOM directory path
#     viewer.load_dicom_series("./data")
#     try:
#         viewer.render()
#     finally:
#         viewer.cleanup()
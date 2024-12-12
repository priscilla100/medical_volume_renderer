"""
STOP. You should not modify this file unless you KNOW what you are doing.
"""

from glfw.GLFW import *
from glfw import _GLFWwindow as GLFWwindow


class Window:
    def __init__(self, 
                 width: int, 
                 height: int, 
                 title: str):
        
        glfwInit()
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1)
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
        glfwWindowHint(GLFW_RESIZABLE, True)
        
        self.window: GLFWwindow = glfwCreateWindow(width, height, title, None, None)
        
        if self.window is None:
            glfwTerminate()
            raise RuntimeError("failed to create GLFW window")
    
        glfwMakeContextCurrent(self.window)
    def mouse_drag_event(self, window, x, y):
        pass

    def mouse_scroll_event(self, window, x_offset, y_offset):
        pass

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.render(glfw.get_time(), 0)
            glfw.swap_buffers(self.window)

        glfw.terminate()

    def render(self, time, frame_time):
        pass

    # def __del__(self):
    #     glfwDestroyWindow(self.window)
    #     glfwTerminate()

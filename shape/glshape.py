import numpy as np
from OpenGL.GL import *

class GLShape:
    def __init__(self):
        self.vao = None
        self.vbo = None
        self.shader = None
        self.initialized = False

    def init_gl(self):
        """Initialize OpenGL buffers and vertex arrays"""
        if self.initialized:
            return

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.initialized = True

    def bind(self):
        """Bind VAO and VBO for rendering"""
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

    def unbind(self):
        """Unbind VAO and VBO"""
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # def delete(self):
    #     """Clean up OpenGL resources"""
    #     if self.initialized:
    #         glDeleteVertexArrays(1, [self.vao])
    #         glDeleteBuffers(1, [self.vbo])
    #         self.initialized = False
    def delete(self):
        if hasattr(self, 'vao') and self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
        if hasattr(self, 'vbo') and self.vbo is not None:
            glDeleteBuffers(1, [self.vbo])

    def draw(self):
        """Abstract method for drawing the shape"""
        raise NotImplementedError("Subclasses must implement draw()")

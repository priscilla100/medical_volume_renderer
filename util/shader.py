"""
STOP. You should not modify this file unless you KNOW what you are doing.
"""

from typing import Optional

from OpenGL.GL import *
import glm


class Shader:
    def __init__(self, 
                 vert: str, 
                 tesc: Optional[str], 
                 tese: Optional[str], 
                 frag: str):
        
        self.program: int = 0
        
        # 1. retrieve the vertShader/fragShader source code from filePath
        
        with open(vert, 'r') as fin:
            vertShaderCode: str = fin.read()
        
        if tesc is not None:
            with open(tesc, 'r') as fin:
                tescShaderCode: str = fin.read()
        
        if tese is not None:
            with open(tese, 'r') as fin:
                teseShaderCode: str = fin.read()

        with open(frag, 'r') as fin:
            fragShaderCode: str = fin.read()

        # 2. compile Shader

        # vertex shader
        vertShader: int = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertShader, vertShaderCode)
        glCompileShader(vertShader)
        self.__checkCompileErrors(vertShader, 'VERTEX')

        # tessellation control shader
        if tesc is not None:
            tescShader: int = glCreateShader(GL_TESS_CONTROL_SHADER)
            glShaderSource(tescShader, tescShaderCode)
            glCompileShader(tescShader)
            self.__checkCompileErrors(tescShader, 'TESSELLATION CONTROL')

        # tessellation evaluation shader
        if tese is not None:
            teseShader: int = glCreateShader(GL_TESS_EVALUATION_SHADER)
            glShaderSource(teseShader, teseShaderCode)
            glCompileShader(teseShader)
            self.__checkCompileErrors(teseShader, 'TESSELLATION EVALUATION')

        # fragment shader
        fragShader: int = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragShader, fragShaderCode)
        glCompileShader(fragShader)
        self.__checkCompileErrors(fragShader, 'FRAGMENT')

        # shader program
        self.program: int = glCreateProgram()
        glAttachShader(self.program, vertShader)
        
        if tesc is not None:
            glAttachShader(self.program, tescShader)
        
        if tese is not None:
            glAttachShader(self.program, teseShader)
        
        glAttachShader(self.program, fragShader)
        
        glLinkProgram(self.program)
        self.__checkCompileErrors(self.program, 'PROGRAM')

        # delete the Shader as they're linked into our program now and no longer necessary
        glDeleteShader(vertShader)
        
        if tesc:
            glDeleteShader(tescShader)
        
        if tese:
            glDeleteShader(teseShader)
        
        glDeleteShader(fragShader)
        
    def __del__(self):
        try:
            if self.program and bool(glDeleteProgram):
                glDeleteProgram(self.program)
        except:
            pass  # Ignore errors during cleanup

        
    def use(self) -> None:
        glUseProgram(self.program)
    
    def setBool(self, name: str, val: bool) -> None:
        glUniform1i(glGetUniformLocation(self.program, name), val)
        
    def setInt(self, name: str, val: int) -> None:
        glUniform1i(glGetUniformLocation(self.program, name), val)

    def setFloat(self, name: str, val: float) -> None:
        glUniform1f(glGetUniformLocation(self.program, name), val)
    
    def setVec2(self, name: str, *args) -> None:
        if (len(args) == 1 and type(args[0]) == glm.vec2):
            glUniform2fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(args[0]))
        
        elif (len(args) == 2 and all(map(lambda x: type(x) == float, args))):
            glUniform2f(glGetUniformLocation(self.program, name), *args)
    
    def setVec3(self, name: str, *args) -> None:
        if (len(args) == 1 and type(args[0]) == glm.vec3):
            glUniform3fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(args[0]))
        elif (len(args) == 3 and all(map(lambda x: type(x) == float, args))):
            glUniform3f(glGetUniformLocation(self.program, name), *args)

    def setVec4(self, name: str, *args) -> None:
        if (len(args) == 1 and type(args[0]) == glm.vec4):
            glUniform4fv(glGetUniformLocation(self.program, name), 1, glm.value_ptr(args[0]))
        
        elif (len(args) == 3 and all(map(lambda x: type(x) == float, args))):
            glUniform4f(glGetUniformLocation(self.program, name), *args)
    
    def setMat2(self, name: str, mat: glm.mat2) -> None:
        glUniformMatrix2fv(glGetUniformLocation(self.program, name), 1, GL_FALSE, glm.value_ptr(mat))
    
    def setMat3(self, name: str, mat: glm.mat3) -> None:
        glUniformMatrix3fv(glGetUniformLocation(self.program, name), 1, GL_FALSE, glm.value_ptr(mat))
    
    def setMat4(self, name: str, mat: glm.mat4) -> None:
        glUniformMatrix4fv(glGetUniformLocation(self.program, name), 1, GL_FALSE, glm.value_ptr(mat))
    
    @staticmethod
    def __checkCompileErrors(shader: int, shaderType: str) -> None:
        if (shaderType != 'PROGRAM'):
            success = glGetShaderiv(shader, GL_COMPILE_STATUS)
            if (not success):
                infoLog = glGetShaderInfoLog(shader)
                print('ERROR::SHADER_COMPILATION_ERROR of type: ' + shaderType + '\n' + infoLog.decode() + '\n -- --------------------------------------------------- -- ')
        
        else:
            success = glGetProgramiv(shader, GL_LINK_STATUS)
            if (not success):
                infoLog = glGetProgramInfoLog(shader)
                print('ERROR::PROGRAM_LINKING_ERROR of type: ' + shaderType + '\n' + infoLog.decode() + '\n -- --------------------------------------------------- -- ')


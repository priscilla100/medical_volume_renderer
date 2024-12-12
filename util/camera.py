"""
STOP. You should not modify this file unless you KNOW what you are doing.
"""

from enum import Enum
import copy

import glm


class Camera:
    class Movement(Enum):
        kUp = 0,
        kDown = 1,
        kLeft = 2,
        kRight = 3,
        kForward = 4,
        kBackWard = 5

    # Default camera values
    kDefaultYaw: float = -90.0
    kDefaultPitch: float = 0.0
    kDefaultSpeed: float = 2.5
    kDefaultSensitivity: float = 0.1
    kDefaultZoom: float = 45.0

    def __init__(self, 
                 position: glm.vec3 = glm.vec3(0.0, 0.0, 0.0),
                 up: glm.vec3 = glm.vec3(0.0, 1.0, 0.0),
                 yaw: float = kDefaultYaw,
                 pitch: float = kDefaultPitch):
        
        self.position: glm.vec3 = copy.deepcopy(position)
        self.front: glm.vec3 = glm.vec3()
        self.up: glm.vec3 = copy.deepcopy(up)
        self.right: glm.vec3 = glm.vec3()
        self.worldUp: glm.vec3 = copy.deepcopy(up)
        self.yaw: float = yaw
        self.pitch: float = pitch

        self.movementSpeed: glm.vec3 = copy.deepcopy(self.kDefaultSpeed)
        self.mouseSensitivity: glm.vec3 = copy.deepcopy(self.kDefaultSensitivity)

        self.zoom: glm.vec3 = copy.deepcopy(self.kDefaultZoom)

        self.__updateCameraVectors()

        self.debugCamera: bool = False

        if self.debugCamera:
            print(f'position = {self.position}')
            print(f'   front = {self.front}')
            print(f'   right = {self.right}')
            print(f'      up = {self.up}')
            print(f' worldUp = {self.worldUp}')

    def getViewMatrix(self) -> glm.mat4:
        """
        Returns the view matrix calculated using Euler Angles and the LookAt Matrix.
        """
        return glm.lookAt(self.position, self.position + self.front, self.up)

    
    def processKeyboard(self, 
                        direction: Movement, 
                        deltaTime: float) -> None:
        """
        Processes input received from any keyboard-like input system.
        Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems).
        """
        displacement: float = self.movementSpeed * deltaTime

        if direction == self.Movement.kUp:
            self.position += self.up * displacement
        elif direction == self.Movement.kDown:
            self.position -= self.up * displacement
        elif direction == self.Movement.kLeft:
            self.position -= self.right * displacement
        elif direction == self.Movement.kRight:
            self.position += self.right * displacement
        elif direction == self.Movement.kForward:
            self.position += self.front * displacement
        elif direction == self.Movement.kBackWard:
            self.position -= self.front * displacement
        else:
            raise ValueError(f'unknown Camera.Movement enumerator {direction}')

    def processMouseMovement(self, 
                             xoffset: float, 
                             yoffset: float, 
                             constrainPitch: bool = True) -> None:
        """
        Processes input received from a mouse input system.
        Expects the offset value in both the x and y direction.
        """
        self.yaw += xoffset * self.mouseSensitivity
        self.pitch += yoffset * self.mouseSensitivity

        # Make sure that when pitch is out of bounds, screen doesn't get flipped.
        if constrainPitch:
            if 89.0 < self.pitch:
                self.pitch = 89.0
            
            if self.pitch < -89.0:
                self.pitch = -89.0

        # Update Front, Right and Up Vectors using the updated Euler angles.
        # print(xoffset, yoffset, self.yaw, self.pitch)
        self.__updateCameraVectors()
    
    def processMouseScroll(self, yoffset: float) -> None:
        """
        Processes input received from a mouse scroll-wheel event.
        Only requires inputs on the vertical wheel-axis.
        """
        self.zoom -= yoffset

        if self.zoom < 1.0:
            self.zoom = 1.0

        if 45.0 < self.zoom:
            self.zoom = 45.0

    def __updateCameraVectors(self) -> None:
        """
        Calculates the front vector from the Camera's (updated) Euler angles
        """
        self.front.x = glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        self.front.y = glm.sin(glm.radians(self.pitch))
        self.front.z = glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        self.front = glm.normalize(self.front)

        # Normalize the vectors,
        # because their length gets closer to 0 
        # the more you look up or down which results in slower movement.
        self.right = glm.normalize(glm.cross(self.front, self.worldUp))
        self.up = glm.normalize(glm.cross(self.right, self.front))


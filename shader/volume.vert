#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aTexCoord;

out vec3 TexCoords;
out vec3 WorldPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    TexCoords = aTexCoord;
    WorldPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}

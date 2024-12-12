#version 330 core
in vec3 TexCoords;
out vec4 FragColor;

// Texture Samplers
uniform sampler3D volumeTexture;
uniform sampler1D transferFunction;
uniform sampler3D spineSegmentationTexture;

// Clipping and Basic Rendering Parameters
uniform vec3 clipMin;
uniform vec3 clipMax;
uniform float density;
uniform float brightness;
uniform float contrast;
uniform float opacity_multiplier;
uniform float density_threshold;
uniform int color_map_type;
uniform int debug_mode;

// Ray Marching Parameters
uniform int max_steps;
uniform float step_size;

// Lighting Parameters
uniform vec3 lightPos[4];
uniform vec3 viewPos;
uniform float ambient_intensity;
uniform float diffuse_intensity;
uniform float specular_intensity;
uniform float shininess;

// Advanced Rendering Parameters
uniform float translucency;
uniform float zoom;
uniform int shading_mode;
uniform int rendering_mode;
uniform float edge_weight;
uniform float gradient_threshold;
uniform float edge_enhancement;

// Color manipulation
uniform float color_shift;
uniform float color_saturation;
uniform vec3 spineColor;

// Color point and opacity point uniforms (added for transfer function flexibility)
uniform vec3 color_points[5];     // Up to 5 color points
uniform float color_points_pos[5]; // Corresponding positions
uniform int num_color_points;
uniform float opacity_points[5];   // Opacity values
uniform float opacity_points_pos[5]; // Corresponding positions
uniform int num_opacity_points;

// Advanced utility functions
vec3 applySaturationAndShift(vec3 color, float saturation, float shift) {
    // Convert to HSL
    float maxVal = max(max(color.r, color.g), color.b);
    float minVal = min(min(color.r, color.g), color.b);
    float luminance = (maxVal + minVal) * 0.5;
    
    float sat = 0.0;
    if (maxVal != minVal) {
        float delta = maxVal - minVal;
        sat = delta / (1.0 - abs(2.0 * luminance - 1.0));
    }
    
    // Adjust saturation
    sat *= saturation;
    
    // Apply color shift
    luminance += shift;
    luminance = clamp(luminance, 0.0, 1.0);
    
    return color;
}

vec3 applyColorMap(float value) {
    // Enhanced color map with more interpolation and variety
    switch (color_map_type) {
        case 0: // Rainbow (Smooth interpolation)
            return vec3(
                0.5 + 0.5 * cos(6.28318 * (value + 0.0)),
                0.5 + 0.5 * cos(6.28318 * (value + 0.33)),
                0.5 + 0.5 * cos(6.28318 * (value + 0.67))
            );
        case 1: // Grayscale
            return vec3(value);
        case 2: // Blue-Red
            return vec3(value, 0.0, 1.0 - value);
        case 3: // Green-Magenta
            return vec3(value, 1.0 - value, value);
        case 4: // Plasma
            return vec3(
                value,
                sin(value * 3.14159),
                cos(value * 3.14159)
            );
        case 5: // Viridis
            float r = clamp(0.16 * value + 0.5, 0.0, 1.0);
            float g = clamp(0.14 * value + 0.5, 0.0, 1.0);
            float b = clamp(0.35 * value + 0.5, 0.0, 1.0);
            return vec3(r, g, b);
        case 6: // Inferno
            float r2 = clamp(1.16 * value, 0.0, 1.0);
            float g2 = clamp(0.5 * value, 0.0, 1.0);
            float b2 = clamp(0.2 * value + 0.5, 0.0, 1.0);
            return vec3(r2, g2, b2);
        default:
            return vec3(value);
    }
}

vec3 computeGradient(vec3 texCoord) {
    float eps = gradient_threshold * 0.01;
    return normalize(vec3(
        texture(volumeTexture, texCoord + vec3(eps, 0, 0)).r - texture(volumeTexture, texCoord - vec3(eps, 0, 0)).r,
        texture(volumeTexture, texCoord + vec3(0, eps, 0)).r - texture(volumeTexture, texCoord - vec3(0, eps, 0)).r,
        texture(volumeTexture, texCoord + vec3(0, 0, eps)).r - texture(volumeTexture, texCoord - vec3(0, 0, eps)).r
    ));
}

vec3 computePhongShading(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 baseColor) {
    vec3 ambient = ambient_intensity * baseColor;
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diffuse_intensity * diff * baseColor;
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specular_intensity * spec * vec3(1.0);
    return ambient + diffuse + specular;
}

vec4 interpolateTransferFunction(float intensity) {
    // Custom transfer function interpolation
    vec4 color = vec4(1.0);
    
    // Color interpolation
    for (int i = 0; i < num_color_points - 1; i++) {
        if (intensity >= color_points_pos[i] && intensity <= color_points_pos[i+1]) {
            float t = (intensity - color_points_pos[i]) / 
                      (color_points_pos[i+1] - color_points_pos[i]);
            color.rgb = mix(
                color_points[i], 
                color_points[i+1], 
                t
            );
            break;
        }
    }
    
    // Opacity interpolation
    for (int i = 0; i < num_opacity_points - 1; i++) {
        if (intensity >= opacity_points_pos[i] && intensity <= opacity_points_pos[i+1]) {
            float t = (intensity - opacity_points_pos[i]) / 
                      (opacity_points_pos[i+1] - opacity_points_pos[i]);
            color.a = mix(
                opacity_points[i], 
                opacity_points[i+1], 
                t
            );
            break;
        }
    }
    
    return color;
}
float adjustBrightnessContrast(float intensity, float brightnessValue, float contrastValue) {
    // Adjust contrast: 
    // - When contrast < 1.0, the image becomes less contrasted
    // - When contrast > 1.0, the image becomes more contrasted
    float adjustedIntensity = (intensity - 0.5) * max(contrastValue, 0.0) + 0.5;
    
    // Adjust brightness: 
    // - Positive values brighten the image
    // - Negative values darken the image
    adjustedIntensity += brightnessValue;
    
    return clamp(adjustedIntensity, 0.0, 1.0);
}
vec3 applyShading(int mode, vec3 normal, vec3 viewDir, vec3 lightDir, vec3 baseColor) {
    if (mode == 0) { // Phong Shading
        vec3 ambient = ambient_intensity * baseColor;
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = diffuse_intensity * diff * baseColor;
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = specular_intensity * spec * vec3(1.0);
        return ambient + diffuse + specular;
    } else if (mode == 1) { // Cel Shading
        float celIntensity = floor(dot(normal, lightDir) * 4.0) / 4.0;
        return baseColor * celIntensity;
    } else if (mode == 2) { // PBR-inspired Shading
        float roughness = 0.5;
        float metallic = 0.2;
        vec3 F0 = mix(vec3(0.04), baseColor, metallic);
        vec3 H = normalize(viewDir + lightDir);
        float NdotL = max(dot(normal, lightDir), 0.0);
        float NdotV = max(dot(normal, viewDir), 0.0);
        float NdotH = max(dot(normal, H), 0.0);
        float HdotV = max(dot(H, viewDir), 0.0);
        float D = pow(NdotH, roughness * 2.0);
        float G = NdotL * NdotV / (NdotL + NdotV + 0.001);
        vec3 F = mix(vec3(0.04), baseColor, pow(1.0 - HdotV, 5.0));
        vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.001);
        return baseColor * (1.0 - metallic) + specular * metallic;
    } else if (mode == 3) { // Toon Shading
        float toonShade = smoothstep(0.4, 0.6, dot(normal, lightDir));
        vec3 toonColor = baseColor * toonShade;
        float rimLight = 1.0 - max(dot(normal, viewDir), 0.0);
        rimLight = smoothstep(0.6, 1.0, rimLight);
        return toonColor + rimLight * vec3(1.0);
    }
    return baseColor; // Default
}
void main() {
    vec3 zoomedTexCoords = ((TexCoords - 0.5) * zoom) + 0.5;
    
    if (any(lessThan(zoomedTexCoords, clipMin)) || any(greaterThan(zoomedTexCoords, clipMax))) {
        discard;
    }

    vec4 accumulatedColor = vec4(0.0);
    float transmittance = 1.0;

    for (int i = 0; i < max_steps; i++) {
        vec3 samplingCoord = zoomedTexCoords + (float(i) * step_size * vec3(0, 0, 1));
        
        float intensity = texture(volumeTexture, samplingCoord).r;
        
        // Use the new brightness and contrast adjustment
        intensity = adjustBrightnessContrast(intensity, brightness, contrast);
        
        if (intensity < density_threshold) continue;

        vec4 transferColor = texture(transferFunction, intensity);
        float opacity = transferColor.a * opacity_multiplier * density;
        
        vec3 gradient = computeGradient(samplingCoord);
        
        // Improved edge enhancement
        float edgeMagnitude = length(gradient);
        float edgeWeight = edge_weight * 2.0; // Scale to make more noticeable
        
        vec3 viewDir = normalize(viewPos - samplingCoord);
        vec3 lightDir = normalize(lightPos[0] - samplingCoord);
        
        vec3 shadedColor = applyShading(shading_mode, gradient, viewDir, lightDir, transferColor.rgb);

        // Smarter edge enhancement that doesn't just overlay white
        shadedColor = mix(
            shadedColor, 
            shadedColor * (1.0 + edgeMagnitude * edgeWeight), 
            clamp(edgeMagnitude * edge_enhancement, 0.0, 1.0)
        );
        // Color manipulation
        shadedColor = mix(shadedColor, applyColorMap(intensity), color_saturation);
        shadedColor = clamp(shadedColor + vec3(color_shift), 0.0, 1.0);
        
        // Spine color blending
        vec3 spineSegColor = texture(spineSegmentationTexture, samplingCoord).rgb;
        shadedColor = mix(shadedColor, spineColor, spineSegColor.r);

        if (rendering_mode == 0) { // Ray Marching
            accumulatedColor.rgb += shadedColor * opacity * transmittance;
            transmittance *= (1.0 - opacity);
        } else if (rendering_mode == 1) { // Maximum Intensity Projection
            accumulatedColor = max(accumulatedColor, vec4(shadedColor, opacity));
        } else if (rendering_mode == 2) { // Average Intensity Projection
            accumulatedColor += vec4(shadedColor, opacity) / float(max_steps);
        } else if (rendering_mode == 3) { // First Hit Projection
            if (opacity > 0.1) {
                accumulatedColor = vec4(shadedColor, opacity);
                break;
            }
        } else if (rendering_mode == 4) { // Ambient Occlusion
            float ao = 1.0 - (float(i) / float(max_steps));
            accumulatedColor.rgb += shadedColor * opacity * ao * transmittance;
            transmittance *= (1.0 - opacity);
        }
        
        if (transmittance < 0.01) break;
    }



    FragColor = vec4(accumulatedColor.rgb, 1.0 - transmittance);
    FragColor.rgb = mix(FragColor.rgb, vec3(1.0), translucency);
}
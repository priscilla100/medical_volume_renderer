# Medical Volume Renderer

## Overview

This Medical Volume Renderer is an advanced application designed for visualizing and analyzing 3D medical imaging data, with a focus on DICOM series. It offers a range of sophisticated rendering techniques and interactive controls to enhance the exploration of volumetric medical data.

## Features

- **Volume Rendering**: Utilizes ray marching for high-quality 3D visualization of medical data.
- **Transfer Function Editor**: Customizable color and opacity mapping for highlighting specific structures.
- **Multiple Rendering Modes**:
  - Ray Marching
  - Maximum Intensity Projection (MIP)
  - Average Intensity Projection
  - First Hit Projection
  - Ambient Occlusion
- **Advanced Shading Options**:
  - Phong Shading
  - Cel Shading
  - PBR-inspired Shading
  - Toon Shading
- **Interactive Controls**: Real-time adjustment of rendering parameters.
- **Spine Segmentation**: Automatic detection and highlighting of spine structures.
- **Magic Lens**: Focal point for detailed examination of specific regions.
- **Edge Enhancement**: Improved visibility of structural boundaries.
- **Dynamic Clipping Planes**: For sectional views of the volume.

## Technical Details

### Dependencies

- OpenGL 3.3+
- GLFW
- ImGui
- NumPy
- PyDICOM
- scikit-image
- SciPy

### Key Components

1. **App**: Main application class managing the rendering loop and user interface.
2. **VolumeRenderer**: Handles the core volume rendering logic and texture management.
3. **VolumeControls**: Manages user interface controls for adjusting rendering parameters.
4. **Shader**: Custom GLSL shaders for volume rendering and post-processing effects.

## Installation

1. Ensure you have Python 3.7+ installed.

2. Install required dependencies:
   ```bash
   pip install pyopengl glfw imgui numpy pydicom scikit-image scipy

3. Clone the repository:

   ```bash
    git clone https://github.com/yourusername/medical-volume-renderer.git
4. Navigate to the project directory:
    ```bash
    cd medical-volume-renderer
 
 Usage

Place your DICOM series in the data directory.
Run the application:
-  ```bash
   python main.py
Use the on-screen controls to adjust rendering parameters.
Navigate the volume using mouse and keyboard controls:

- WASD: Move camera
- Mouse: Rotate view
- Arrow keys: Adjust clipping planes


Adjusting the volume control parameters achieves different volume rendering of the spinal imaging.
![PBR+AMBIENT OCCLUSION+GS - Copy](https://github.com/user-attachments/assets/927e794d-1219-4136-916c-f85d4c1354ba)

![EDetMax+maxstep+stepsize+densitythreshold+0 08+colorshift0 04+saturation1 63COPY](https://github.com/user-attachments/assets/471f0fac-d455-4bcf-8d93-f868ac3249e3)

![AIP](https://github.com/user-attachments/assets/50bb2758-a400-434f-adc3-6c247ce97dde)
## Customization
Modify shader/volume.frag to experiment with different rendering techniques.
Adjust default parameters in VolumeControls class for different initial visualizations.
Extend VolumeRenderer class to add support for additional file formats or preprocessing steps.

## Performance Considerations

The application's performance heavily depends on the GPU capabilities.
Large volume datasets may require significant memory. Consider downsampling for smoother performance on less powerful systems.
Adjust the max_steps parameter in the shader for a balance between quality and performance.

## Troubleshooting

If the volume doesn't render, check the DICOM file compatibility and ensure the data directory is correctly set.
For OpenGL-related issues, verify that your graphics drivers are up to date.
If controls are not responsive, check the GLFW event handling in the App class.

## Future Enhancements

- Multi-volume rendering support
- CUDA/OpenCL acceleration for preprocessing steps
- VR/AR integration for immersive visualization
- Machine learning-based automatic segmentation of additional anatomical structures

Contributing
Contributions to improve the Medical Volume Renderer are welcome. Please follow these steps:

Fork the repository
- Create a new branch for your feature
- Commit your changes
- Push to the branch
- Create a new Pull Request

References
1. Kniss, J., et al. "Interactive Texture-Based Volume Rendering for Large Data Sets."
2. Kim, S., et al. "Binary Volume Rendering Using Slice-Based Binary Shell."
3. Zhang, Y., et al. "GPU-Based Visualization of High-Resolution Medical Images."


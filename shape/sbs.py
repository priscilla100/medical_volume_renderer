import numpy as np
from concurrent.futures import ThreadPoolExecutor

def sbs_projection_sheared(sheared_volume, output_image_size):
    """
    Optimized SBS projection with improved memory access patterns.
    """
    if sheared_volume is None:
        print("Error: Sheared volume is None")
        return np.zeros(output_image_size, dtype=np.float32)
    
    intermediate_image = np.zeros(output_image_size, dtype=np.float32)
    depth, height, width = sheared_volume.shape

    for z in range(depth):
        for y in range(width):  # Access along the fastest axis first
            for x in range(height):
                intensity = sheared_volume[z, x, y]
                if intensity > 0:  # Surface voxel
                    pixel_intensity = intensity / 255.0
                    if 0 <= x < output_image_size[0] and 0 <= y < output_image_size[1]:
                        intermediate_image[x, y] = max(intermediate_image[x, y], pixel_intensity)
                    break

    return intermediate_image

# def sbs_projection_sheared(sheared_volume, output_image_size):
#     if sheared_volume is None:
#         print("Error: Sheared volume is None")
#         return np.zeros(output_image_size, dtype=np.float32)
    
#     intermediate_image = np.zeros(output_image_size, dtype=np.float32)
#     depth, height, width = sheared_volume.shape
#     for z in range(depth):
#         for x in range(height):
#             for y in range(width):
#                 intensity = sheared_volume[z, x, y]
#                 if intensity > 0:  # Surface voxel
#                     # Compute intensity-based pixel value
#                     pixel_intensity = intensity / 255.0
#                     if 0 <= x < output_image_size[0] and 0 <= y < output_image_size[1]:
#                         intermediate_image[x, y] = max(intermediate_image[x, y], pixel_intensity)
#                     break
#     return intermediate_image


def create_surface_voxel_structure(volume_data):
    surface_voxels = []
    depth, height, width = volume_data.shape
    volume_data = volume_data.astype(np.float32)

    for z in range(depth):
        for x in range(height):
            for y in range(width):
                if volume_data[z, x, y] > 0:
                    gradient = np.array([
                        volume_data[z, x + 1, y] - volume_data[z, x - 1, y] if 0 < x < height - 1 else 0,
                        volume_data[z, x, y + 1] - volume_data[z, x, y - 1] if 0 < y < width - 1 else 0,
                        volume_data[z + 1, x, y] - volume_data[z - 1, x, y] if 0 < z < depth - 1 else 0
                    ])
                    normal = gradient / (np.linalg.norm(gradient) + 1e-6)
                    surface_voxels.append((z, x, y, {"gradient": gradient, "normal": normal}))
                    break
    print(f"Surface Voxels Count: {len(surface_voxels)}")
    return surface_voxels


def sbs_projection(surface_voxels, output_image_size):
    intermediate_image = np.zeros(output_image_size, dtype=np.float32)

    for voxel_data in surface_voxels:
        z, x, y, attributes = voxel_data
        gradient = attributes["gradient"]
        normal = attributes["normal"]

        pixel_intensity = np.linalg.norm(gradient)

        if 0 <= x < output_image_size[0] and 0 <= y < output_image_size[1]:
            intermediate_image[x, y] = max(intermediate_image[x, y], pixel_intensity)

    return intermediate_image

def shear_warp_transform(volume, shear_matrix, warp_matrix):
    """
    Efficient shear-warp transformation with minimized memory overhead.
    """
    depth, height, width = volume.shape
    coords = np.mgrid[0:depth, 0:height, 0:width].reshape(3, -1).T  # More memory-efficient grid
    coords = np.c_[coords, np.ones(coords.shape[0])]  # Homogeneous coordinates

    sheared_coords = coords[:, :3] @ shear_matrix.T
    warped_coords = sheared_coords @ warp_matrix.T

    transformed_volume = np.zeros_like(volume)

    for coord, value in zip(warped_coords.astype(int), volume.flat):
        z, x, y = coord
        if 0 <= z < depth and 0 <= x < height and 0 <= y < width:
            transformed_volume[z, x, y] = value

    return transformed_volume


# def shear_warp_transform(volume, shear_matrix, warp_matrix):
#     depth, height, width = volume.shape
#     coords = np.array([[z, x, y, 1] for z in range(depth) for x in range(height) for y in range(width)])
    
#     sheared_coords = coords[:, :3].dot(shear_matrix.T)
#     warped_coords = sheared_coords.dot(warp_matrix.T)

#     transformed_volume = np.zeros_like(volume)

#     for coord, value in zip(warped_coords.astype(int), volume.flat):
#         z, x, y = coord
#         if 0 <= z < depth and 0 <= x < height and 0 <= y < width:
#             transformed_volume[int(z), int(x), int(y)] = value

#     return transformed_volume
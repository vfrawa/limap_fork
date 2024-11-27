import numpy as np
from pathlib import Path
import os


import limap.base
import limap.util.io as limapio
import limap.pointsfm as _psfm



# Function to read camera pose from a text file
def read_camera_pose(pose_path: str):
    try:
        with open(pose_path, 'r') as file:
            pose = np.eye(4)  # Initialize as identity matrix (camera to world pose)
            for i in range(4):
                pose[i, :] = np.array([float(x) for x in file.readline().split()])
                
        R = pose[:3, :3].T  # R is world-to-camera rotation (transpose)
        t = -R @ pose[0:3, 3]  # t is world-to-camera translation
        return R, t
    except IOError:
        print(f"Error opening camera pose file: {pose_path}")


# Function to read scene information from a text file
def read_scene_info(info_path: str):
    try:
        with open(info_path, 'r') as file:
            width, height, num_imgs = None, None, None
            K = np.zeros((3, 3))

            for line in file:
                # Split line by '=' and remove whitespace
                parts = line.strip().split('=')
                if len(parts) < 2:
                    continue
                key, value = parts[0].strip(), parts[1].strip()

                if key == 'm_colorWidth':
                    width = int(value)
                elif key == 'm_colorHeight':
                    height = int(value)
                elif key == 'm_calibrationColorIntrinsic':
                    # Parse the intrinsic matrix values from a single line
                    K_values = list(map(float, value.split()))
                    K = np.array(K_values).reshape(4, 4)[:3, :3]
                elif key == 'm_frames.size':
                    num_imgs = int(value)
                    
            return K, width, height, num_imgs
    except IOError:
        print("Error opening scene info file.")

# Function to read image paths and camera poses from directory
def read_images_and_poses(sensordata_dir: Path, images_dir: Path):
    info_path = sensordata_dir / "_info.txt"
    K, width, height, num_imgs = read_scene_info(info_path)

    cameras, camimages, camviews = {}, {}, {}

    cameras[0] = limap.base.Camera("PINHOLE", K, cam_id=0, hw=(height, width)) #all images have the same intrinsic parameters

    for i in range(num_imgs):
        # Format frame number (e.g., "000001" for i = 1 in ScanNet)
        formatted_number = f"{i:06d}"

        # Get image path
        image_filename = f"frame-{formatted_number}.color.jpg"
        image_path = images_dir / image_filename

        # Get pose path
        pose_filename = f"frame-{formatted_number}.pose.txt"
        pose_path = sensordata_dir / pose_filename
        R, t = read_camera_pose(str(pose_path))

        pose = limap.base.CameraPose(R, t)
        camimage = limap.base.CameraImage(cam_id=0, pose=pose, image_name=str(image_path))
        camimages[i] = camimage
        camview = limap.base.CameraView(camera=cameras[0], pose=pose, image_name=str(image_path))
        camviews[i] = camview

    return cameras, camimages, camviews

if __name__ == "__main__":
    scene_dir = Path("/local/home/vfrawa/data/ScanNet/scans/scene0191_00")
    sensordata_dir = scene_dir / "sensorstream"
    images_dir = scene_dir / "sensorstream" #"low_light05_no_noise_no_smaller_contrast"
    output_dir = scene_dir / "formats_shortpath"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cameras, camimages, camviews = read_images_and_poses(sensordata_dir, images_dir)
    imagecols = limap.base.ImageCollection(cameras, camimages)

    limapio.save_npy(output_dir/"limap_imagecols.npy", imagecols.as_dict())

    _psfm.convert_imagecols_to_colmap(imagecols, output_dir/"colmap_format_known_poses")


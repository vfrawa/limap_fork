import numpy as np
from pathlib import Path

import os
import time

import cv2
import torch

import limap.base
import limap.line2d
import limap.util.config
import limap

import pyvelsed

from read_ScanNet import read_images_and_poses



# # Function to read camera pose from a text file
# def read_camera_pose(pose_path: str):
#     try:
#         with open(pose_path, 'r') as file:
#             pose = np.eye(4)  # Initialize as identity matrix (camera to world pose)
#             for i in range(4):
#                 pose[i, :] = np.array([float(x) for x in file.readline().split()])
                
#         R = pose[:3, :3].T  # R is world-to-camera rotation (transpose)
#         t = -R @ pose[0:3, 3]  # t is world-to-camera translation
#         return R, t
#     except IOError:
#         print(f"Error opening camera pose file: {pose_path}")


# # Function to read scene information from a text file
# def read_scene_info(info_path: str):
#     try:
#         with open(info_path, 'r') as file:
#             width, height, num_imgs = None, None, None
#             K = np.zeros((3, 3))

#             for line in file:
#                 # Split line by '=' and remove whitespace
#                 parts = line.strip().split('=')
#                 if len(parts) < 2:
#                     continue
#                 key, value = parts[0].strip(), parts[1].strip()

#                 if key == 'm_colorWidth':
#                     width = int(value)
#                 elif key == 'm_colorHeight':
#                     height = int(value)
#                 elif key == 'm_calibrationColorIntrinsic':
#                     # Parse the intrinsic matrix values from a single line
#                     K_values = list(map(float, value.split()))
#                     K = np.array(K_values).reshape(4, 4)[:3, :3]
#                 elif key == 'm_frames.size':
#                     num_imgs = int(value)
                    
#             return K, width, height, num_imgs
#     except IOError:
#         print("Error opening scene info file.")

# # Function to read image paths and camera poses from directory
# def read_images_and_poses(sensordata_dir: Path, images_dir: Path):
#     info_path = sensordata_dir / "_info.txt"
#     K, width, height, num_imgs = read_scene_info(info_path)

#     cameras, camviews = {}, {}

#     cameras[0] = limap.base.Camera("PINHOLE", K, cam_id=0, hw=(height, width)) #all images have the same intrinsic parameters

#     for i in range(num_imgs):
#         # Format frame number (e.g., "000001" for i = 1 in ScanNet)
#         formatted_number = f"{i:06d}"

#         # Get image path
#         image_filename = f"frame-{formatted_number}.color.jpg"
#         image_path = images_dir / image_filename

#         # Get pose path
#         pose_filename = f"frame-{formatted_number}.pose.txt"
#         pose_path = sensordata_dir / pose_filename
#         R, t = read_camera_pose(str(pose_path))

#         pose = limap.base.CameraPose(R, t)
#         camview = limap.base.CameraView(camera=cameras[0], pose=pose, image_name=str(image_path))
#         camviews[i] = camview

#     return cameras, camviews


def vis_detections(img, segs):
    import copy

    from limap.visualize.vis_utils import draw_segments

    img_draw = copy.deepcopy(img)
    img_draw = draw_segments(img_draw, segs, color=[0, 255, 0])
    return img_draw


def vis_matches(img1, img2, segs1, segs2, matches):
    import copy

    import cv2
    import numpy as np
    import seaborn as sns

    matched_seg1 = segs1[matches[:, 0]]
    matched_seg2 = segs2[matches[:, 1]]
    n_lines = matched_seg1.shape[0]
    colors = sns.color_palette("husl", n_colors=n_lines)
    img1_draw = copy.deepcopy(img1)
    img2_draw = copy.deepcopy(img2)
    for idx in range(n_lines):
        color = np.array(colors[idx]) * 255.0
        color = color.astype(int).tolist()
        cv2.line(
            img1_draw,
            (int(matched_seg1[idx, 0]), int(matched_seg1[idx, 1])),
            (int(matched_seg1[idx, 2]), int(matched_seg1[idx, 3])),
            color,
            4,
        )
        cv2.line(
            img2_draw,
            (int(matched_seg2[idx, 0]), int(matched_seg2[idx, 1])),
            (int(matched_seg2[idx, 2]), int(matched_seg2[idx, 3])),
            color,
            4,
        )
    return img1_draw, img2_draw
    #combined_img = np.concatenate((img1_draw, img2_draw), axis=1)
    #return combined_img

def detect_velsed(view):
    img_gray = view.read_image(set_gray=True)
    vertical_VP = view.matrix()@np.array([0,0,1,0])
    # torch.cuda.synchronize()
    # start = time.time()
    segstages1, _ = pyvelsed.detect(img=img_gray, vanishing_point=vertical_VP)
    # torch.cuda.synchronize()
    # print(f"VELSED detection time: {time.time() - start:.3f}s")
    
    # Use the line length as score
    # Calculate Euclidean distance for each row
    segments = segstages1["detection"]

    distances = np.sqrt((segments[:, 2] - segments[:, 0]) ** 2 + (segments[:, 3] - segments[:, 1]) ** 2)

    # Reshape distances to (num_segs, 1) and concatenate along axis=1
    lines = np.concatenate([segments, distances[:, np.newaxis]], axis=1)

    return lines


if __name__== "__main__":
    scene_dir = Path("/local/home/vfrawa/data/ScanNet/scans/scene0191_00")
    sensordata_dir = scene_dir / "sensorstream"
    images_dir = scene_dir / "low_light05_no_noise_no_smaller_contrast"
    #images_dir = scene_dir / "sensorstream"

    experiment_name = "low_light_pyvelsed_gluestick_every10th_test"
    output_dir = scene_dir / "matching" / experiment_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    cameras, camimages, camviews = read_images_and_poses(sensordata_dir, images_dir)


    detector = limap.line2d.get_detector(
        {"method": "lsd", "skip_exists": False}
    )  # get a line detector
    # extractor = limap.line2d.get_extractor(
    #     {"method": "dense_naive", "skip_exists": False}
    # )  # get a line extractor
    # matcher = limap.line2d.get_matcher(
    #     {
    #         "method": "dense_roma",
    #         "dense_roma": {"mode": "outdoor"},
    #         "one_to_many": False,
    #         "skip_exists": False,
    #         "n_jobs": 1,
    #         "topk": 0,
    #     },
    #     extractor,
    # )  # initiate a line matcher
    extractor = limap.line2d.get_extractor(
        {"method": "wireframe", "skip_exists": False}
    )  # get a line extractor
    matcher = limap.line2d.get_matcher(
        {
            "method": "gluestick",
            "skip_exists": False,
            "n_jobs": 1,
            "topk": 0,
            #"superglue":
                #"weights": "indoor" # ["indoor", "outdoor"] for selecting superglue models
        },
        extractor,
    )  # initiate a line matcher

    # current_dir = os.path.abspath(os.path.dirname(__file__))
    # view1 = limap.base.CameraView(
    #     os.path.join(current_dir, "../third-party/GlueStick/resources/img1.jpg")
    # )  # initiate an limap.base.CameraView instance for detection.
    # view2 = limap.base.CameraView(
    #     os.path.join(current_dir, "../third-party/GlueStick/resources/img2.jpg")
    # )  # initiate an limap.base.CameraView instance for detection.


    num_images = len(camviews)
    step = 10
    for id1 in range(0, num_images, step): #range(0, 120, step):
        view1 = camviews[id1]

        #segs1 = detector.detect(view1)  # detection
        segs1 = detect_velsed(view1)

        desc1 = extractor.extract(view1, segs1)  # description
        img1 = view1.read_image()
        img1_det = vis_detections(img1, segs1)
        #cv2.imwrite(str(output_dir/f"image{id1}_segments.png"), img1_det)

        for id2 in range(id1+step, id1+6*step+1, step):
            if id2>=num_images: break
            
            view2 = camviews[id2]

            #segs2 = detector.detect(view2)  # detection
            segs2 = detect_velsed(view2)

            desc2 = extractor.extract(view2, segs2)  # description
            torch.cuda.synchronize()
            start = time.time()
            matches = matcher.match_pair(desc1, desc2)  # matching
            torch.cuda.synchronize()
            print(f"Matching time: {time.time() - start:.3f}s")


            img2 = view2.read_image()

            img2_det = vis_detections(img2, segs2)
            #cv2.imwrite(str(output_dir/f"image{id2}_segments.png"), img2_det)

            img1_draw, img2_draw = vis_matches(img1, img2, segs1, segs2, matches)
            #cv2.imwrite(str(output_dir/f"image{id1}_{id2}_matches{id1}.png"), img1_draw)
            #cv2.imwrite(str(output_dir/f"image{id1}_{id2}_matches{id2}.png"), img2_draw)
            combined_det = np.concatenate((img1_det, img2_det), axis=1)
            combined_match = np.concatenate((img1_draw, img2_draw), axis=1)
            combined_imgs = np.concatenate((combined_det, combined_match), axis=0)
            cv2.imwrite(str(output_dir/f"image{id1}_{id2}_matches.png"), combined_imgs)

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from runners.hypersim.Hypersim import Hypersim
from runners.hypersim.loader import read_scene_hypersim

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.runners
import limap.util.config as cfgutils

import limap.util.io as limapio
from limap.pointsfm.model_converter import convert_imagecols_to_colmap
from pathlib import Path


#we simply save the imagecols to a file and convert to colmap
def run_scene_hypersim(cfg, dataset, scene_id, cam_id=0):

    imagecols = read_scene_hypersim(
        cfg, dataset, scene_id, cam_id=cam_id, load_depth=False
    )
    scene_dir = Path(cfg["data_dir"]) / scene_id
    output_dir = scene_dir / "formats_from_hypersim_limap"
    limapio.save_npy(output_dir/"limap_imagecols.npy", imagecols.as_dict())

    convert_imagecols_to_colmap(imagecols, output_dir/"empty_colmap_model_known_poses")


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(description="fit and merge 3d lines")
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/fitnmerge/hypersim.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/fitnmerge/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "--npyfolder",
        type=str,
        default=None,
        help="folder to load precomputed results",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-sid"] = "--scene_id"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = args.npyfolder
    if cfg["folder_to_load"] is None:
        cfg["folder_to_load"] = os.path.join(
            "precomputed", "hypersim", cfg["scene_id"]
        )
    return cfg


def main():
    cfg = parse_config()
    cfg["data_dir"] = "/local/home/vfrawa/data/hypersim_limap"
    dataset = Hypersim(cfg["data_dir"])
    run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"])


if __name__ == "__main__":
    main()

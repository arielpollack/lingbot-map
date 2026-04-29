# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GCT Visualization Module

This module provides visualization utilities for 3D reconstruction results:
- PointCloudViewer: Interactive point cloud viewer with camera visualization
- viser_wrapper: Quick visualization wrapper for predictions
- predictions_to_glb: Export predictions to GLB 3D format
- Colorization and utility functions

Usage:
    from lingbot_map.vis import PointCloudViewer, viser_wrapper, predictions_to_glb

    # Interactive visualization
    viewer = PointCloudViewer(pred_dict=predictions, port=8080)
    viewer.run()

    # Quick visualization
    viser_wrapper(predictions, port=8080)

    # Export to GLB
    scene = predictions_to_glb(predictions)
    scene.export("output.glb")
"""

import importlib

__all__ = [
    # Main viewer
    "PointCloudViewer",
    # Quick visualization
    "viser_wrapper",
    # GLB export
    "predictions_to_glb",
    # Utilities
    "CameraState",
    "colorize",
    "colorize_np",
    "get_vertical_colorbar",
    # Sky segmentation
    "apply_sky_segmentation",
    "segment_sky",
    "download_skyseg_model",
    "load_or_create_sky_masks",
]

_ATTR_TO_MODULE = {
    "PointCloudViewer": "lingbot_map.vis.point_cloud_viewer",
    "viser_wrapper": "lingbot_map.vis.viser_wrapper",
    "CameraState": "lingbot_map.vis.utils",
    "colorize": "lingbot_map.vis.utils",
    "colorize_np": "lingbot_map.vis.utils",
    "get_vertical_colorbar": "lingbot_map.vis.utils",
    "apply_sky_segmentation": "lingbot_map.vis.sky_segmentation",
    "segment_sky": "lingbot_map.vis.sky_segmentation",
    "download_skyseg_model": "lingbot_map.vis.sky_segmentation",
    "load_or_create_sky_masks": "lingbot_map.vis.sky_segmentation",
    "predictions_to_glb": "lingbot_map.vis.glb_export",
}


def __getattr__(name: str):
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'lingbot_map.vis' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

import cv2
import math
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from itertools import product
from pathlib import Path

# sonar default parameters
from dataclasses import dataclass

@dataclass
class SonarConfig:
    SONAR_RANGE_M: float
    SONAR_MAX_RANGE_M: float
    SONAR_MIN_RANGE_M: float
    SONAR_H_APERTURE_DEG: float
    SONAR_V_APERTURE_DEG: float
    SONAR_BEAM_BIN_DEG: float
    SONAR_IMAGE_HEIGHT: int
    SONAR_IMAGE_WIDTH: int

    @property
    def SONAR_SCALING_FACTOR(self):
        return self.SONAR_IMAGE_HEIGHT / self.SONAR_RANGE_M

    @property
    def SONAR_TIP_PIXEL_COORD(self):
        return (
            self.SONAR_IMAGE_WIDTH / 2 - 1,
            self.SONAR_IMAGE_HEIGHT - 1
        )


class SonarParameters:

    LOW_FREQUENCY = SonarConfig(
        SONAR_RANGE_M=30.0,
        SONAR_MAX_RANGE_M=5,
        SONAR_MIN_RANGE_M=1,
        SONAR_H_APERTURE_DEG=40.0,
        SONAR_V_APERTURE_DEG=20.0,
        SONAR_BEAM_BIN_DEG=0.6,
        SONAR_IMAGE_HEIGHT=976,
        SONAR_IMAGE_WIDTH=668,
    )

    HIGH_FREQUENCY = SonarConfig(
        SONAR_RANGE_M=5.0,
        SONAR_MAX_RANGE_M=4,
        SONAR_MIN_RANGE_M=1,
        SONAR_H_APERTURE_DEG=130.0,
        SONAR_V_APERTURE_DEG=20.0,
        SONAR_BEAM_BIN_DEG=0.4,
        SONAR_IMAGE_HEIGHT=960,
        SONAR_IMAGE_WIDTH=1732,
    )

def get_sonar_fan_mask(
        sonar_params = SonarParameters.HIGH_FREQUENCY
        ):
    """
    Returns the sonar fan shape mask using the sonar image shape, horizontal aperture, and tip coordinates.
    """

    h = sonar_params.SONAR_IMAGE_HEIGHT
    w = sonar_params.SONAR_IMAGE_WIDTH
    fov_deg = sonar_params.SONAR_H_APERTURE_DEG
    sonar_tip_coord = sonar_params.SONAR_TIP_PIXEL_COORD

    cx, cy = sonar_tip_coord

    theta_max = np.deg2rad((fov_deg - 5) / 2)

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    dx = xs - cx
    dy = cy - ys

    # polar coordinates
    theta = np.arctan2(dx, dy + 1e-8)
    R = np.sqrt(dx**2 + dy**2)

    # range limits
    R_max = h * 0.98   # slightly smaller than image height
    R_min = h * 0.02   # small inner cutoff

    mask = (
        (np.abs(theta) <= theta_max) &
        (R <= R_max) &
        (R >= R_min) &
        (dy > 0)
    )

    return mask.astype(np.uint8)

def sonar_pixel_to_physical(
    pt_uv,
    sonar_params=SonarParameters.HIGH_FREQUENCY
    ):
    """
    Convert a sonar image pixel to approximate polar sonar coordinates.

    Returns:
        rho  = range in meters
        theta = azimuth angle in radians
    """

    scaling_factor = sonar_params.SONAR_SCALING_FACTOR
    tip_coord = sonar_params.SONAR_TIP_PIXEL_COORD

    u, v = float(pt_uv[0]), float(pt_uv[1])
    x0, y0 = tip_coord

    xp = (u - x0) / scaling_factor
    yp = (y0 - v) / scaling_factor
    
    rho = np.sqrt(xp**2 + yp**2)
    theta = np.arctan2(xp, yp)
    
    return rho, theta

def sonar_keypoint_polar(
    kpts,
    sonar_params = SonarParameters.HIGH_FREQUENCY
    ):
    """
    Compute sonar polar coordinates for keypoints.

    Returns:
        rho      : metric range
        theta  : radians
    """

    tip = sonar_params.SONAR_TIP_PIXEL_COORD
    s = sonar_params.SONAR_SCALING_FACTOR
    
    kpts = np.asarray(kpts, dtype=np.float64)

    x0, y0 = tip

    dx = kpts[:, 0] - x0
    dy = y0 - kpts[:, 1]

    rho = np.sqrt(dx**2 + dy**2) / (s + 1e-12)
    theta = np.arctan2(dx, dy + 1e-12)

    return rho, theta

def euler_zyx_rotation(alpha, beta, gamma):
    """
    Paper convention:
        R = Rx(alpha) @ Ry(beta) @ Rz(gamma)
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rx = np.array([[1, 0, 0],
                   [0, ca, -sa],
                   [0, sa, ca]], dtype=np.float64)
    Ry = np.array([[cb, 0, sb],
                   [0, 1, 0],
                   [-sb, 0, cb]], dtype=np.float64)
    Rz = np.array([[cg, -sg, 0],
                   [sg, cg, 0],
                   [0, 0, 1]], dtype=np.float64)
    return Rx @ Ry @ Rz  

def sonar_point_to_camera_arc(
    sonar_pt_xy,
    camera_shape,
    R,
    t,
    f,
    n_elev_samples,
    sonar_params=SonarParameters.HIGH_FREQUENCY
    ):
    """
    Project one sonar image point to an arc of camera image points by sampling the unknown elevation angle.
    """

    vert_aperture_deg = sonar_params.SONAR_V_APERTURE_DEG

    rho, theta = sonar_pixel_to_physical(sonar_pt_xy, sonar_params=sonar_params)

    phi_min = -np.deg2rad(vert_aperture_deg / 2.0)
    phi_max =  np.deg2rad(vert_aperture_deg / 2.0)
    phis = np.linspace(phi_min, phi_max, n_elev_samples)

    # 3D points in sonar frame
    xs = np.full((n_elev_samples,), rho * np.sin(theta) * np.cos(phis))
    ys = np.full((n_elev_samples,), rho * np.cos(theta) * np.cos(phis))
    zs = rho * np.sin(phis)
    P_s = np.stack([xs, ys, zs], axis=1)  # (N, 3)  

    # Transform to camera frame
    P_c = (R @ P_s.T).T + t.reshape(1, 3)
    
    Z = P_c[:, 2]

    # Keep points in front of the camera
    valid = Z > 1e-6
    P_c = P_c[valid]
    
    if len(P_c) == 0:
        return np.empty((0, 2), dtype=np.float32), rho, theta

    h, w = camera_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    u = f * (P_c[:, 0] / P_c[:, 2]) + cx
    v = f * (P_c[:, 1] / P_c[:, 2]) + cy
    
    pts = np.stack([u, v], axis=1).astype(np.float32)

    return pts, rho, theta

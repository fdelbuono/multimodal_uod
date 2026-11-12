import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import time

from calibration.calibration_config import Calibration
from calibration.sonar_camera_sensor_models import *
from calibration.calibration_pipeline import *

class FusionParameters:
    CLASS_NAMES = ["rov", "plant", "animal", "trash"]
    NUM_CLASSES = 4

    IOU_MATCH_THRESHOLD = 0.40
    UNCERTAINTY_THRESHOLD = 0.30
    CONF_FALLBACK = 0.25


def load_image(image):
    """Load an image path or validate an already-loaded OpenCV image (BGR)."""
    if isinstance(image, (str, Path)):
        path = str(image)
        loaded = cv2.imread(path)
        if loaded is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        image = loaded

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a file path or a NumPy array.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return image


def get_sonar_configuration(sonar_img):
    """Get sonar geometry from image dimensions."""
    h, w = sonar_img.shape[:2]

    if h == 960:
        sonar_p = SonarParameters.HIGH_FREQUENCY

    elif h == 976:
        sonar_p = SonarParameters.LOW_FREQUENCY

    return sonar_p


def load_calibration(alpha, beta, gamma, tx, ty, tz, focal_length):
    R = euler_zyx_rotation(alpha, beta, gamma)
    T = np.array([tx, ty, tz], dtype=np.float64)
    return R, T, focal_length


def box_iou(box_a, box_b):
    """IoU for xyxy bounding boxes."""
    ax0, ay0, ax1, ay1 = map(float, box_a)
    bx0, by0, bx1, by1 = map(float, box_b)

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def clip_box_xyxy(box, width, height):
    x0, y0, x1, y1 = map(float, box)
    return np.array([
        np.clip(x0, 0, width - 1),
        np.clip(y0, 0, height - 1),
        np.clip(x1, 0, width - 1),
        np.clip(y1, 0, height - 1),
    ], dtype=np.float32)


def yolo_results_to_detections(results):
    """Convert a single-image Ultralytics result into detection dictionaries."""
    detections = []
    if not results:
        return detections

    boxes = results[0].boxes
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()

    for box, cls_id, conf in zip(xyxy, classes, confidences):
        cls_id = int(cls_id)
        if cls_id < 0 or cls_id >= FusionParameters.NUM_CLASSES:
            continue
        detections.append({
            "xyxy": box.astype(np.float32),
            "cls": cls_id,
            "conf": float(conf),
        })
    return detections


def sonar_point_to_camera_uv(rho, theta, phi, R, T, f, camera_shape):
    """Project one 3D sonar point into the camera image."""
    P_s = np.array([
        rho * np.sin(theta) * np.cos(phi),
        rho * np.cos(theta) * np.cos(phi),
        rho * np.sin(phi),
    ], dtype=np.float64)

    P_c = R @ P_s + T
    z = P_c[2]

    # A point behind/on the camera plane cannot be projected meaningfully.
    if not np.all(np.isfinite(P_c)) or z <= 1e-6:
        return None, P_s, P_c

    h, w = camera_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    u = f * P_c[0] / z + cx
    v = f * P_c[1] / z + cy
    uv = np.array([u, v], dtype=np.float32)

    if not np.all(np.isfinite(uv)):
        return None, P_s, P_c
    return uv, P_s, P_c


def confidence_to_opinion(cls_id, confidence, num_classes=FusionParameters.NUM_CLASSES):
    """
    Convert one detector's class/confidence into a valid subjective-logic opinion.

    b[class] = confidence, other class beliefs = 0, u = 1-confidence.
    This treats detector confidence as belief mass, not as a calibrated probability.
    """
    confidence = float(np.clip(confidence, 0.0, 1.0))
    belief = np.zeros(num_classes, dtype=np.float64)
    belief[int(cls_id)] = confidence
    uncertainty = 1.0 - confidence
    return belief, uncertainty


def fuse_subjective_logic(belief_a, uncertainty_a, belief_b, uncertainty_b):
    """Fuse two valid subjective-logic opinions using the consensus rule."""
    b_a = np.asarray(belief_a, dtype=np.float64)
    b_b = np.asarray(belief_b, dtype=np.float64)
    u_a = float(uncertainty_a)
    u_b = float(uncertainty_b)

    if b_a.shape != b_b.shape:
        raise ValueError("Belief vectors must have the same shape.")
    if (np.any(b_a < 0) or np.any(b_b < 0)
            or u_a < 0 or u_b < 0
            or not np.isclose(b_a.sum() + u_a, 1.0, atol=1e-5)
            or not np.isclose(b_b.sum() + u_b, 1.0, atol=1e-5)):
        raise ValueError("Each opinion must have non-negative masses summing to 1.")

    # Conflict between the two opinions.
    conflict = float(np.sum(b_a[:, None] * b_b[None, :]) - np.sum(b_a * b_b))
    denominator = 1.0 - conflict
    if denominator <= 1e-8:
        # Complete conflict: no well-defined consensus. Return an uninformative opinion.
        return np.zeros_like(b_a), 1.0

    belief_f = (b_a * b_b + b_a * u_b + b_b * u_a) / denominator
    uncertainty_f = (u_a * u_b) / denominator

    # Small numerical errors can occur with floating-point arithmetic.
    belief_f = np.clip(belief_f, 0.0, 1.0)
    uncertainty_f = float(np.clip(uncertainty_f, 0.0, 1.0))
    total = belief_f.sum() + uncertainty_f
    if total > 0 and not np.isclose(total, 1.0, atol=1e-5):
        belief_f /= total
        uncertainty_f /= total
    return belief_f, uncertainty_f

def confidence_to_opinion(cls, confidence, num_classes=FusionParameters.NUM_CLASSES):
    """
    Convert a detector confidence into a simple subjective-logic opinion.

    Returns:
        belief: belief mass for each class
        uncertainty: remaining uncertainty
    """
    belief = np.zeros(num_classes, dtype=np.float32)
    belief[int(cls)] = float(confidence)

    uncertainty = 1.0 - float(confidence)

    return belief, uncertainty

def fuse_one_detection(det, source):
    """Create a standard output record for a single-sensor detection."""
    belief, uncertainty = confidence_to_opinion(det["cls"], det["conf"])
    return {
        "class_belief": belief.astype(np.float32),
        "cls": int(det["cls"]),
        "class_name": FusionParameters.CLASS_NAMES[int(det["cls"])],
        "uncertainty": float(uncertainty),
        "conf": float(det["conf"]),
        "camera_box": det["xyxy"].copy() if source == "camera" else None,
        "sonar_box": det["xyxy"].copy() if source == "sonar" else None,
        "source": source,
        "matched_iou": None,
    }


def fuse_matched_detection(sonar_det, camera_det, iou):
    """Fuse class evidence from one matched sonar-camera detection pair."""
    b_s, u_s = confidence_to_opinion(sonar_det["cls"], sonar_det["conf"])
    b_c, u_c = confidence_to_opinion(camera_det["cls"], camera_det["conf"])
    b_f, u_f = fuse_subjective_logic(b_s, u_s, b_c, u_c)

    pred_cls = int(np.argmax(b_f)) if np.sum(b_f) > 0 else int(
        camera_det["cls"] if camera_det["conf"] >= sonar_det["conf"]
        else sonar_det["cls"]
    )

    return {
        "class_belief": b_f.astype(np.float32),
        "cls": pred_cls,
        "class_name": FusionParameters.CLASS_NAMES[pred_cls],
        "uncertainty": float(u_f),
        "conf": float(np.max(b_f)) if len(b_f) else 0.0,
        "camera_box": camera_det["xyxy"].copy(),
        "sonar_box": sonar_det["xyxy"].copy(),
        "source": "camera+sonar",
        "matched_iou": float(iou),
    }

def sonar_box_to_camera_box(
    sonar_box_xyxy,
    R,
    T,
    f,
    camera_shape,
    sonar_params,
):
    """
    Convert a sonar 2D box into a camera-space 2D box by sampling the
    corners of the sonar frustum and using the vertical aperture as the
    projected height cue.

    Returns:
        camera_box_xyxy, projected_points
    """

    sonar_image_height =  sonar_params.SONAR_IMAGE_HEIGHT
    sonar_image_width =  sonar_params.SONAR_IMAGE_WIDTH
    vert_aperture_deg = sonar_params.SONAR_V_APERTURE_DEG

    
    x0, y0, x1, y1 = sonar_box_xyxy.astype(float)
    h_s, w_s = sonar_image_height, sonar_image_width

    rho0, theta0 = sonar_pixel_to_physical([x0,y1], sonar_params=sonar_params)
    rho1, theta1 = sonar_pixel_to_physical([x1,y1], sonar_params=sonar_params)

    phi_min = -np.deg2rad(vert_aperture_deg / 2.0)
    phi_max =  np.deg2rad(vert_aperture_deg / 2.0)
    
    projected_pts = []
    P_s = []
    P_c = []
    
    for rho, theta in [
        (rho0, theta0),
        (rho1, theta1),
    ]:
        for phi in [phi_min, phi_max]:
            uv, ps, pc = sonar_point_to_camera_uv(
                rho, theta, phi,
                R, T, f, camera_shape
            )

            P_s.append(ps)
            P_c.append(pc)
    
            if uv is not None and np.all(np.isfinite(uv)):
                projected_pts.append(uv)

    if len(projected_pts) == 0:
        return None, np.empty((0, 2), dtype=np.float32)

    projected_pts = np.stack(projected_pts, axis=0).astype(np.float32)

    x_min, y_min = projected_pts.min(axis=0)
    x_max, y_max = projected_pts.max(axis=0)

    cam_h, cam_w = camera_shape[:2]
    camera_box = clip_box_xyxy([x_min, y_min, x_max, y_max], cam_w, cam_h)

    return camera_box, projected_pts, P_s, P_c


def project_sonar_detections(camera_img, sonar_img, sonar_dets, R, T, f, sonar_params):
    """Project sonar detections into camera image coordinates."""
    
    camera_img = load_image(camera_img)
    sonar_img = load_image(sonar_img)

    projected_sonar_dets = []

    for sonar_idx, sdet in enumerate(sonar_dets):
        cam_box, proj_pts, ps, pc = sonar_box_to_camera_box(
            sdet["xyxy"],
            R=R,
            T=T,
            f=f,
            camera_shape=camera_img.shape,
            sonar_params=sonar_params
        )

        if cam_box is None:
            continue

        projected_sonar_dets.append({
            **sdet,
            "xyxy": cam_box,
            "projected_pts": proj_pts,
            "P_s": ps,
            "P_c": pc,
            "source": "sonar->camera",
            "sonar_idx": sonar_idx,
        })

    return projected_sonar_dets


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: [x1, y1, x2, y2] in pixel coordinates.

    Returns:
        IoU between 0.0 and 1.0.
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * \
            max(0.0, float(box1[3]) - float(box1[1]))

    area2 = max(0.0, float(box2[2]) - float(box2[0])) * \
            max(0.0, float(box2[3]) - float(box2[1]))

    union_area = area1 + area2 - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def match_detections(camera_detections, sonar_detections, iou_threshold=FusionParameters.IOU_MATCH_THRESHOLD):
    """
    Greedily match camera and projected sonar detections using IoU.

    Expected detection dictionaries contain:
      - 'xyxy': bounding box [x1, y1, x2, y2]
      - 'cls': predicted class ID

    Returns:
      matches: list of (camera_index, sonar_index, iou)
      unmatched_camera: list of unmatched camera indices
      unmatched_sonar: list of unmatched sonar indices
    """
    candidates = []

    for cam_idx, cam_det in enumerate(camera_detections):
        for sonar_idx, sonar_det in enumerate(sonar_detections):
            if cam_det["cls"] != sonar_det["cls"]:
                continue

            iou = calculate_iou(
                cam_det["xyxy"],
                sonar_det["xyxy"]
            )

            if iou >= iou_threshold:
                candidates.append((iou, cam_idx, sonar_idx))

    # Highest-IoU matches first; each detection can be matched only once.
    candidates.sort(reverse=True)

    matches = []
    used_camera = set()
    used_sonar = set()

    for iou, cam_idx, sonar_idx in candidates:
        if cam_idx in used_camera or sonar_idx in used_sonar:
            continue

        matches.append((cam_idx, sonar_idx, iou))
        used_camera.add(cam_idx)
        used_sonar.add(sonar_idx)

    unmatched_camera = [
        i for i in range(len(camera_detections))
        if i not in used_camera
    ]

    unmatched_sonar = [
        i for i in range(len(sonar_detections))
        if i not in used_sonar
    ]

    return matches, unmatched_camera, unmatched_sonar


def draw_fused_detections(camera_img, detections, show_single_sensor=True):
    """Draw final fused detections on a copy of the camera image."""
    out = camera_img.copy()

    for det in detections:
        source = det["source"]
        if source == "sonar" or det["camera_box"] is None:
            # A sonar-only detection is shown only if a projected box is available.
            if not show_single_sensor or det.get("projected_camera_box") is None:
                continue
            box = det["projected_camera_box"]
        else:
            box = det["camera_box"]

        x0, y0, x1, y1 = np.asarray(box).astype(int)
        if source == "camera+sonar":
            color = (255, 255, 0)
        elif source == "camera":
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)

        label = (
            f'{det["source"]}: {det["class_name"]} '
            f'b={det["conf"]:.2f} u={det["uncertainty"]:.2f}'
        )
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            out, label, (x0, max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        )
    return out

def draw_boxes(img, detections, class_names=FusionParameters.CLASS_NAMES, color=(0, 255, 0), label_prefix=""):
    """
    Draw detections on a BGR image.

    Args:
        img (str or np.ndarray): Image path or already loaded BGR image.
        detections (list): List of detection dictionaries.
    """
    # Load image if a path is provided
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img}")

    out = img.copy()

    for det in detections:
        x0, y0, x1, y1 = det["xyxy"].astype(int)
        cls_id = det["cls"]
        conf = det["conf"]

        label = f"{label_prefix}{class_names[cls_id]} {conf:.2f}"

        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            out,
            label,
            (x0, max(0, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return out

def bgr_to_rgb(img):

    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img}")
            
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def plot_pair_overview(camera_img, sonar_img, camera_dets, sonar_dets, title="Raw detector outputs"):
    """
    Show sonar detections and camera detections side by side.
    """
    cam_vis = draw_boxes(camera_img, camera_dets, color=(0, 255, 0), label_prefix="cam: ")
    sonar_vis = draw_boxes(sonar_img, sonar_dets, color=(255, 255, 0), label_prefix="sonar: ")

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(bgr_to_rgb(sonar_vis))
    ax[0].set_title("Sonar detections")
    ax[0].axis("off")

    ax[1].imshow(bgr_to_rgb(cam_vis))
    ax[1].set_title("Camera detections")
    ax[1].axis("off")

    plt.suptitle(title)
    plt.savefig("plot_pair_overview.jpg")
    plt.show()

def plot_camera_projection_overlay(
    camera_img,
    camera_dets=None,
    projected_sonar_dets=None,
    title="Camera overlay",
):
    """Show camera detections and projected sonar detections."""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(bgr_to_rgb(camera_img))

    # Camera detections
    if camera_dets is not None:
        for det in camera_dets:
            x0, y0, x1, y1 = det["xyxy"]
            cls_id = int(det.get("macro_cls", det["cls"]))
            conf = det["conf"]

            rect = plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                linewidth=2,
                edgecolor="lime",
            )
            ax.add_patch(rect)

            ax.text(
                x0, max(0, y0 - 6),
                f"cam: {FusionParameters.CLASS_NAMES[cls_id]} {conf:.2f}",
                color="lime",
                fontsize=10,
                bbox=dict(
                    facecolor="black", alpha=0.4,
                    pad=1, edgecolor="none"
                ),
            )

    # Projected sonar detections
    colors = plt.cm.tab10.colors
    order = [0, 1, 3, 2, 0]

    if projected_sonar_dets is not None:
        for i, det in enumerate(projected_sonar_dets):
            color = colors[i % len(colors)]
            x0, y0, x1, y1 = det["xyxy"]
            cls_id = int(det["cls"])
            conf = det["conf"]

            # Projected bounding box
            rect = plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                linewidth=1.5,
                edgecolor=color,
                linestyle="--",
            )
            ax.add_patch(rect)

            # Projected sonar quadrilateral
            pts = det.get("projected_pts")

            if pts is not None and len(pts) == 4:
                pts = np.asarray(pts)
                ax.plot(
                    pts[order, 0],
                    pts[order, 1],
                    "-o",
                    color=color,
                    linewidth=2,
                    markersize=4,
                )

            # Optional second projected quadrilateral, if available
            pts2 = det.get("projected_pts_1")

            if pts2 is not None and len(pts2) == 4:
                pts2 = np.asarray(pts2)
                ax.plot(
                    pts2[order, 0],
                    pts2[order, 1],
                    "--o",
                    color=color,
                    linewidth=2,
                    markersize=4,
                )

            ax.text(
                x0,
                min(camera_img.shape[0] - 5, y1 + 15),
                f"sonar->cam: {FusionParameters.CLASS_NAMES[cls_id]} {conf:.2f}",
                color=color,
                fontsize=9,
                bbox=dict(
                    facecolor="black", alpha=0.4,
                    pad=1, edgecolor="none"
                ),
            )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("plot_camera_projection_overlay.jpg")
    plt.show()


class CameraSonarFusion:
    def __init__(
        self,
        camera_model_path,
        sonar_model_path,
        device,
        calibration,
        iou_threshold=FusionParameters.IOU_MATCH_THRESHOLD,
        uncertainty_threshold=FusionParameters.UNCERTAINTY_THRESHOLD,
    ):
        self.camera_model = YOLO(str(camera_model_path))
        self.sonar_model = YOLO(str(sonar_model_path))
        self.device = device
        self.iou_threshold = float(iou_threshold)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.alpha = calibration["alpha"]
        self.beta = calibration["beta"]
        self.gamma = calibration["gamma"]
        self.tx = calibration["tx"]
        self.ty = calibration["ty"]
        self.tz = calibration["tz"]
        self.focal_length = calibration["f"]
        self.R, self.T, self.f = load_calibration(self.alpha, self.beta, self.gamma, self.tx, self.ty, self.tz, self.focal_length)

    def process(self, camera_image, sonar_image, visualize=False):
        camera_img = load_image(camera_image)
        sonar_img = load_image(sonar_image)

        sonar_params = get_sonar_configuration(sonar_img)

        camera_results = self.camera_model.predict(
            source=camera_img, device=self.device, verbose=False
        )
        sonar_results = self.sonar_model.predict(
            source=sonar_img, device=self.device, verbose=False
        )

        camera_dets = yolo_results_to_detections(camera_results)
        sonar_dets = yolo_results_to_detections(sonar_results)

        if visualize:
            plot_pair_overview(
                camera_img,
                sonar_img,
                camera_dets,
                sonar_dets,
                title="Stage 1: Raw detector outputs",
            )

        projected_sonar_dets = project_sonar_detections(
            camera_img,
            sonar_img,
            sonar_dets,
            self.R,
            self.T,
            self.f,
            sonar_params,
        )
        
        if visualize:
            plot_camera_projection_overlay(
                camera_img,
                camera_dets=camera_dets,
                projected_sonar_dets=projected_sonar_dets,
                title="Stage 2: Sonar detections projected onto camera",
            )

        # Match with original sonar indices preserved.
        matches, unmatched_camera_indices, unmatched_projected_indices = (
            match_detections(
                camera_dets,
                projected_sonar_dets,
                self.iou_threshold,
            )
        )
        
        # Convert projected-list indices to original sonar detection indices.
        matched_sonar_indices = {
            projected_sonar_dets[p_idx]["sonar_idx"]
            for _, p_idx, _ in matches
        }
        
        matched_camera_indices = {
            c_idx for c_idx, _, _ in matches
        }
        
        projected_by_sonar_idx = {
            det["sonar_idx"]: det
            for det in projected_sonar_dets
        }
        
        fused = []
        
        # Matched camera-sonar detections.
        for c_idx, p_idx, iou in matches:
            s_idx = projected_sonar_dets[p_idx]["sonar_idx"]
        
            fused.append(
                fuse_matched_detection(
                    sonar_dets[s_idx],
                    camera_dets[c_idx],
                    iou,
                )
            )

        # Filter by uncertainty. A detector-only opinion has u = 1 - confidence.
        fused = [
            det for det in fused
            if np.isfinite(det["uncertainty"])
            and det["uncertainty"] <= self.uncertainty_threshold
        ]

        # Unmatched sonar detections: preserve sonar-only evidence.
        # If projection succeeded, retain its projected camera-space box for display.
        for s_idx, sonar_det in enumerate(sonar_dets):
            if s_idx in matched_sonar_indices:
                continue
            item = fuse_one_detection(sonar_det, "sonar")
            projected = projected_by_sonar_idx.get(s_idx)
            item["projected_camera_box"] = (
                projected["xyxy"].copy() if projected is not None else None
            )
            fused.append(item)

        # Unmatched camera detections: preserve camera-only evidence.
        for c_idx, camera_det in enumerate(camera_dets):
            if c_idx in matched_camera_indices:
                continue
            fused.append(fuse_one_detection(camera_det, "camera"))

        filtered = fused

        # Ensure the final class label agrees with the fused belief vector.
        for det in filtered:
            belief = np.asarray(det["class_belief"])
            if belief.sum() > 0:
                det["cls"] = int(np.argmax(belief))
                det["class_name"] = FusionParameters.CLASS_NAMES[det["cls"]]

        visualization = draw_fused_detections(camera_img, filtered)

        if visualize:
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("fusion_output.jpg")
            plt.show()

        return {
            "final_detections": filtered,
            "all_fused_detections_before_filtering": fused,
            "camera_detections": camera_dets,
            "sonar_detections": sonar_dets,
            "projected_sonar_detections": projected_sonar_dets,
            "matches": matches,
            "camera_image": camera_img,
            "sonar_image": sonar_img,
            "visualization": visualization,
            "sonar_configuration": sonar_params,
        }

def load_yolo_labels(label_path, image_shape):
    """
    Return ground-truth boxes in absolute xyxy coordinates.

    Each YOLO label line:
        cls x_center y_center width height
    """
    h, w = image_shape[:2]
    boxes = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing ground-truth label: {label_path}"
        )

    for line in label_path.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        cls_id, xc, yc, bw, bh = map(float, values)

        x0 = (xc - bw / 2) * w
        y0 = (yc - bh / 2) * h
        x1 = (xc + bw / 2) * w
        y1 = (yc + bh / 2) * h

        boxes.append({
            "cls": int(cls_id),
            "xyxy": np.array([x0, y0, x1, y1], dtype=float),
        })

    return boxes


def get_evaluable_predictions(final_detections):
    """
    Use camera-coordinate boxes for evaluation.

    Matched detections: camera_box
    Camera-only detections: camera_box
    Sonar-only detections: projected_camera_box, if available

    Sonar-only detections without a valid camera projection
    cannot be evaluated against camera-image labels.
    """
    predictions = []

    for det in final_detections:
        box = det.get("camera_box")

        if box is None:
            box = det.get("projected_camera_box")

        if box is None:
            continue

        box = np.asarray(box, dtype=float)

        if box.shape != (4,) or not np.all(np.isfinite(box)):
            continue

        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        predictions.append({
            "cls": int(det["cls"]),
            "conf": float(det["conf"]),
            "xyxy": box,
        })

    return predictions


def evaluation_iou(box_a, box_b):
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    intersection = max(0, x1 - x0) * max(0, y1 - y0)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def calculate_ap(predictions, ground_truth, class_id, iou_threshold):
    """
    Predictions and ground_truth are dictionaries keyed by image ID.

    Uses confidence-ranked detections and 101-point interpolated AP.
    """
    gt_by_image = {}
    total_gt = 0

    for image_id, gt_list in ground_truth.items():
        boxes = [
            gt["xyxy"]
            for gt in gt_list
            if gt["cls"] == class_id
        ]
        gt_by_image[image_id] = boxes
        total_gt += len(boxes)

    if total_gt == 0:
        return None

    # A ground-truth object can be matched only once per threshold.
    matched = {
        image_id: np.zeros(len(boxes), dtype=bool)
        for image_id, boxes in gt_by_image.items()
    }

    class_predictions = []

    for image_id, pred_list in predictions.items():
        for pred in pred_list:
            if pred["cls"] == class_id:
                class_predictions.append((image_id, pred))

    class_predictions.sort(
        key=lambda item: item[1]["conf"],
        reverse=True,
    )

    tp = np.zeros(len(class_predictions))
    fp = np.zeros(len(class_predictions))

    for i, (image_id, pred) in enumerate(class_predictions):
        gt_boxes = gt_by_image.get(image_id, [])

        if not gt_boxes:
            fp[i] = 1
            continue

        ious = np.array([
            evaluation_iou(pred["xyxy"], gt_box)
            for gt_box in gt_boxes
        ])

        best_idx = int(np.argmax(ious))
        best_iou = ious[best_idx]

        if (
            best_iou >= iou_threshold
            and not matched[image_id][best_idx]
        ):
            tp[i] = 1
            matched[image_id][best_idx] = True
        else:
            fp[i] = 1

    if len(class_predictions) == 0:
        return 0.0

    tp_cumulative = np.cumsum(tp)
    fp_cumulative = np.cumsum(fp)

    recall = tp_cumulative / total_gt
    precision = tp_cumulative / np.maximum(
        tp_cumulative + fp_cumulative, 1e-12
    )

    # 101-point interpolated average precision.
    recall_levels = np.linspace(0, 1, 101)

    interpolated_precision = [
        np.max(precision[recall >= level])
        if np.any(recall >= level)
        else 0.0
        for level in recall_levels
    ]

    return float(np.mean(interpolated_precision))



if __name__ == "__main__":

    calibration = Calibration("calibration/calibration_croatia.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    camera_dir = Path("dataset/test_croatia/camera-test/images")
    sonar_dir = Path("dataset/test_croatia/sonar-test/images")

    cam_img = list(camera_dir.glob("*.jpg"))
    sonar_img = list(sonar_dir.glob("*.jpg"))


    fusion = CameraSonarFusion(
        camera_model_path="unimodal/trained_models/YOLO26nCAMERA-all.pt",
        sonar_model_path="unimodal/trained_models/YOLO26nSONAR-all.pt",
        device=device,
        calibration=calibration.params
    )

    results = fusion.process(
        camera_image=cam_img[90],       # image array or path
        sonar_image=sonar_img[90],       # image array or path
        visualize=True,
    )

    final_detections = results["final_detections"]
    print(final_detections)

    # Evaluation
    
    label_dir = Path("dataset/test_croatia/camera-test/labels")

    cam_img = sorted(camera_dir.glob("*.jpg"))
    sonar_img = sorted(sonar_dir.glob("*.jpg"))

    if len(cam_img) != len(sonar_img):
        raise ValueError(
            f"Image counts differ: {len(cam_img)} camera images, "
            f"{len(sonar_img)} sonar images. Match pairs explicitly."
        )

    all_predictions = {}
    all_ground_truth = {}
    inference_times = []

    for camera_path, sonar_path in tqdm(
        zip(cam_img, sonar_img),
        total=min(len(cam_img), len(sonar_img)),
        desc="Evaluating fusion",
        unit="image",
    ):
        # Assumes the sorted image lists are correctly paired.
        image_id = camera_path.stem
        label_path = label_dir / f"{image_id}.txt"

        camera_image = cv2.imread(str(camera_path))

        if camera_image is None:
            print(f"Skipping unreadable image: {camera_path}")
            continue

        # Ground truth is in camera image coordinates.
        gt = load_yolo_labels(label_path, camera_image.shape)

        ## Synchronize GPU before timing, if CUDA is being used.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        result = fusion.process(
            camera_image=str(camera_path),
            sonar_image=str(sonar_path),
            visualize=False,
        )
        
        # Wait for GPU operations to finish before stopping the timer.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = time.perf_counter() - start_time
        inference_times.append(inference_time)
        
        pred = get_evaluable_predictions(
            result["final_detections"]
        )

        all_ground_truth[image_id] = gt
        all_predictions[image_id] = pred

        tqdm.write(
            f"{image_id}: {len(pred)} predictions, "
            f"{len(gt)} ground-truth objects"
        )

    iou_thresholds = np.arange(0.50, 0.951, 0.05)
    class_ids = range(len(FusionParameters.CLASS_NAMES))

    ap_by_threshold = {}

    for threshold in iou_thresholds:
        per_class_ap = {}

        for cls_id in class_ids:
            ap = calculate_ap(
                all_predictions,
                all_ground_truth,
                cls_id,
                float(threshold),
            )

            if ap is not None:
                per_class_ap[cls_id] = ap

        ap_by_threshold[round(float(threshold), 2)] = per_class_ap


    ap50_values = list(ap_by_threshold[0.5].values())

    map50 = (
        float(np.mean(ap50_values))
        if ap50_values else float("nan")
    )

    all_ap_values = [
        ap
        for class_ap in ap_by_threshold.values()
        for ap in class_ap.values()
    ]

    valid_classes = [
        cls_id for cls_id in class_ids
        if any(
            gt["cls"] == cls_id
            for gt_list in all_ground_truth.values()
            for gt in gt_list
        )
    ]

    class_aps_5095 = [
        ap_by_threshold[round(float(t), 2)][cls_id]
        for t in iou_thresholds
        for cls_id in valid_classes
        if cls_id in ap_by_threshold[round(float(t), 2)]
    ]

    map5095 = (
        float(np.mean(class_aps_5095))
        if class_aps_5095 else float("nan")
    )

    print("\n" + "=" * 55)
    print("CAMERA-SONAR FUSION EVALUATION")
    print("=" * 55)
    print(f"Images evaluated: {len(all_predictions)}")
    print(f"mAP@0.5:          {map50:.4f}")
    print(f"mAP@0.5:0.95:     {map5095:.4f}")

    print("\nAP@0.5 per class:")
    for cls_id, ap in ap_by_threshold[0.5].items():
        print(f"  {FusionParameters.CLASS_NAMES[cls_id]}: {ap:.4f}")

    print("\nAP per class at each IoU threshold:")
    for threshold, per_class_ap in ap_by_threshold.items():
        values = list(per_class_ap.values())
        mean_ap = np.mean(values) if values else float("nan")
        print(f"  IoU {threshold:.2f}: mAP = {mean_ap:.4f}")

    if inference_times:
        average_inference_time = np.mean(inference_times)
        total_inference_time = np.sum(inference_times)
        average_fps = 1.0 / average_inference_time

        print("\n" + "=" * 55)
        print("INFERENCE SPEED")
        print("=" * 55)
        print(f"Timed image pairs:              {len(inference_times)}")
        print(f"Total inference time:           {total_inference_time:.2f} s")
        print(f"Average time per image pair:    {average_inference_time * 1000:.2f} ms")
        print(f"Average inference speed:        {average_fps:.2f} FPS")
    else:
        print("\nNo inference timings available.")
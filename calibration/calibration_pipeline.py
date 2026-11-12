import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from itertools import product
from tqdm import tqdm
from calibration.sonar_camera_sensor_models import SonarParameters as sonar_parameters
from calibration.sonar_camera_sensor_models import sonar_keypoint_polar, get_sonar_fan_mask, sonar_pixel_to_physical, euler_zyx_rotation, sonar_point_to_camera_arc
from calibration.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from calibration.SuperGluePretrainedNetwork.models.superglue import SuperGlue

class CalibrationParameters:
    SONAR_MIN_FEATURES = 10      # minimum number of sonar features to calibrate
    N_ELEV_SAMPLES = 10          # number of elevation samples in each sonar projection arc
    SONAR_MAX_FEATURES = 30      # maximum number of sonar features to use for calibration (for computational reasons)

def load_static_keyframes(path, seconds_per_keyframe=1):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, round(fps * seconds_per_keyframe))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def load_static_keyframes_sonar(path, 
                                sonar_params=sonar_parameters.HIGH_FREQUENCY,
                                seconds_per_keyframe=1,
                                ):

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, round(fps * seconds_per_keyframe))
    crop_size_w =sonar_params.SONAR_IMAGE_WIDTH
    crop_size_h =sonar_params.SONAR_IMAGE_HEIGHT

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2
            frame_cropped = frame[y_start:y_start+crop_size_h, x_start:x_start+crop_size_w]
            frames.append(frame_cropped)
        idx += 1
    cap.release()
    return frames

def bgr_to_gray(frame):
    if len(frame.shape) == 2:
        # already grayscale
        return frame
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def low_pass_filter_fft(sonar_image, cutoff_ratio=0.8):
    """
    Applies a low-pass filter on the sonar image using its Fourier transform.
    """

    if len(sonar_image.shape) == 2:
        gray = sonar_image
    else :
        gray = bgr_to_gray(sonar_image)
        
    gray = gray.astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    hy = max(1, int(h * cutoff_ratio / 2))
    hx = max(1, int(w * cutoff_ratio / 2))
    mask[cy-hy:cy+hy, cx-hx:cx+hx] = 1.0

    filtered = np.fft.ifft2(np.fft.ifftshift(dft_shift * mask)).real
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    return filtered.astype(np.uint8)

def visualize_sonar_denoising(sonar_image, denoised):
    """
    Plot original sonar image and FFT-denoised image side-by-side.
    """
    gray = bgr_to_gray(sonar_image)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].imshow(gray, cmap="gray")
    ax[0].set_title("Original Sonar")
    ax[0].axis("off")

    ax[1].imshow(denoised, cmap="gray")
    ax[1].set_title("FFT Low-Pass Filtered")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def extract_sp_sonar(img, sonar_params=sonar_parameters.HIGH_FREQUENCY):
    """
    Sonar-specific SuperPoint extraction:
    - grayscale conversion
    - mask the sonar fan shape
    - run SuperPoint
    """
    
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    mask = get_sonar_fan_mask(sonar_params)

    gray_masked = gray.copy()
    gray_masked[mask == 0] = 0

    inp = torch.from_numpy(gray_masked / 255.0).float()[None, None].to(device)

    with torch.no_grad():
        pred = superpoint_sonar({"image": inp})

    return pred


def filter_sonar_predictions(
    pred,
    sonar_params = sonar_parameters.HIGH_FREQUENCY,
    remove_behind=True,
    ):
    """
    Filter sonar SuperPoint predictions by:
    - angle
    - minimum range
    - maximum range
    - optionally keeping only the closest prediction
      within each sonar angular beam

    Filtering is applied consistently to:
    - keypoints
    - descriptors
    - scores
    """

    max_theta_deg = sonar_params.SONAR_H_APERTURE_DEG
    min_range = sonar_params.SONAR_MIN_RANGE_M
    max_range =sonar_params.SONAR_MAX_RANGE_M
    tip = sonar_params.SONAR_TIP_PIXEL_COORD
    s = sonar_params.SONAR_SCALING_FACTOR
    beam_bin_deg = sonar_params.SONAR_BEAM_BIN_DEG

    max_theta = np.deg2rad(max_theta_deg - 30) / 2

    kpts = pred["keypoints"][0]
    desc = pred["descriptors"][0]
    scores = pred["scores"][0]

    if torch.is_tensor(kpts):
        kpts_np = kpts.detach().cpu().numpy()
    else:
        kpts_np = np.asarray(kpts)

    R, theta = sonar_keypoint_polar(
        kpts_np,
        sonar_params=sonar_params
    )

    keep = np.abs(theta) <= max_theta
    keep &= R >= min_range

    if max_range is not None:
        keep &= R <= max_range

    if remove_behind:

        valid_indices = np.where(keep)[0]

        if len(valid_indices) > 0:

            theta_valid = theta[valid_indices]
            R_valid = R[valid_indices]

            theta_deg = np.rad2deg(theta_valid)

            beam_indices = np.floor(
                theta_deg / beam_bin_deg
            ).astype(int)

            closest_indices = []

            for beam in np.unique(beam_indices):

                points_in_beam = np.where(
                    beam_indices == beam
                )[0]

                # R is distance from sonar
                closest_local = points_in_beam[
                    np.argmin(R_valid[points_in_beam])
                ]

                closest_indices.append(
                    valid_indices[closest_local]
                )

            keep = np.zeros(len(kpts_np), dtype=bool)

            keep[closest_indices] = True

    if torch.is_tensor(kpts):

        keep_t = torch.from_numpy(keep).to(kpts.device)

        kpts_f = kpts[keep_t]
        desc_f = desc[:, keep_t]
        scores_f = scores[keep_t]

    else:

        kpts_f = kpts[keep]
        desc_f = desc[:, keep]
        scores_f = scores[keep]

    return {
        "keypoints": [kpts_f],
        "descriptors": [desc_f],
        "scores": [scores_f]
    }


def match_sg_sonar(pred0, pred1, img0, img1, sonar_params, filter_angle=True):

    if img0.ndim == 3:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if filter_angle:
        pred0 = filter_sonar_predictions(pred0, sonar_params)
        pred1 = filter_sonar_predictions(pred1, sonar_params)

    kpts0 = pred0['keypoints'][0]
    kpts1 = pred1['keypoints'][0]
    desc0 = pred0['descriptors'][0]
    desc1 = pred1['descriptors'][0]
    scores0 = pred0['scores'][0]
    scores1 = pred1['scores'][0]

    if isinstance(kpts0, np.ndarray):
        kpts0 = torch.from_numpy(kpts0).float().to(device)
        desc0 = torch.from_numpy(desc0).float().to(device)
        scores0 = torch.from_numpy(scores0).float().to(device)

    data = {
        'keypoints0': kpts0.unsqueeze(0),
        'keypoints1': kpts1.unsqueeze(0),
        'descriptors0': desc0.unsqueeze(0),
        'descriptors1': desc1.unsqueeze(0),
        'scores0': scores0.unsqueeze(0),
        'scores1': scores1.unsqueeze(0),
        'image0': torch.from_numpy(img0).float()[None,None].to(device),
        'image1': torch.from_numpy(img1).float()[None,None].to(device),
    }

    with torch.no_grad():
        pred = superglue_sonar(data)

    matches = pred['matches0'][0]
    conf = pred['matching_scores0'][0]

    valid = matches > -1

    idx0 = torch.arange(len(matches), device=matches.device)[valid]
    idx1 = matches[valid]

    k0 = kpts0[idx0].cpu().numpy()
    k1 = kpts1[idx1].cpu().numpy()
    d0 = desc0[:, idx0].cpu().numpy()
    d1 = desc1[:, idx1].cpu().numpy()
    c  = conf[valid].cpu().numpy()

    return k0, k1, d0, d1, c

def plot_superpoint_keypoints(img, pred, title="SuperPoint keypoints"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kpts = pred['keypoints'][0].cpu().numpy()
    
    plt.figure(figsize=(7,7))
    plt.imshow(img_rgb)
    plt.scatter(kpts[:,0], kpts[:,1], s=1, c="red", alpha=0.7)
    plt.title(f"{title}  (count={len(kpts)})")
    plt.axis("off")
    plt.savefig("plot_superpoint_keypoints.jpg")
    plt.show()

def plot_superglue_matches(img0, img1, k0, k1, title="SuperGlue matches"):
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_rgb
    canvas[:h1, w0:] = img1_rgb
    plt.figure(figsize=(14, 7))
    plt.imshow(canvas)
    for pt0, pt1 in zip(k0, k1):
        x0, y0 = pt0
        x1, y1 = pt1
        plt.plot([x0, x1 + w0], [y0, y1], 'red', linewidth=1)
    plt.title(f"{title}  (matched={len(k0)})")
    plt.axis("off")
    plt.savefig("plot_superglue_matches.jpg")
    plt.show()


def plot_superglue_matches_colored(img0, img1, k0, k1, title="Colored matches"):

    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_rgb
    canvas[:h1, w0:] = img1_rgb

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas)

    n_matches = len(k0)

    colors = plt.cm.hsv(np.linspace(0, 1, n_matches))

    for (pt0, pt1, c) in zip(k0, k1, colors):
        x0, y0 = pt0
        x1, y1 = pt1

        plt.scatter(x0, y0, color=c, s=3)
        plt.scatter(x1 + w0, y1, color=c, s=3)

    plt.title(f"{title} (matched={n_matches})")
    plt.axis("off")
    plt.savefig("plot_superglue_matches_colored.jpg")
    plt.show()


def plot_sonar_arc_projection(
    sonar_img,
    camera_img,
    sonar_pt,
    arc_pts,
    start_end="start",
    filename="plot_sonar_arc_projection"
):
    """
    Visualize:
    - sonar point on sonar image
    - projected arc on camera image
    """

    if start_end =="start":
        color_line="blue"

    else:
        color_line="red"

    if len(sonar_img.shape) == 2:
        sonar_rgb = cv2.cvtColor(sonar_img, cv2.COLOR_GRAY2RGB)
    else:
        sonar_rgb = cv2.cvtColor(sonar_img, cv2.COLOR_BGR2RGB)

    if len(camera_img.shape) == 2:
        camera_rgb = cv2.cvtColor(camera_img, cv2.COLOR_GRAY2RGB)
    else:
        camera_rgb = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    ax[0].imshow(sonar_rgb)

    ax[0].scatter(
        sonar_pt[0],
        sonar_pt[1],
        c="red",
        s=80,
        marker="o",
    )

    ax[0].set_title(
        "Sonar Point"
    )

    ax[0].axis("off")

    ax[1].imshow(camera_rgb)

    if len(arc_pts) > 0:

        ax[1].plot(
            arc_pts[:, 0],
            arc_pts[:, -1],
            color=color_line
        )


    ax[1].set_title("Projected Sonar Arc")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{filename}.jpg")
    plt.show()


def track_points_lk(prev_gray, next_gray, points_xy):
    """
    Track 2D points from prev_gray to next_gray using Lucas–Kanade.
    Returns:
        new_points_xy, status_mask, error
    """

    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float32)

    p0 = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, winSize=(16, 16), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    if p1 is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float32)

    return p1.reshape(-1, 2), st.reshape(-1), err.reshape(-1)


def pairwise_distances(A, B):
    # Euclidean distances between two point clouds A (m,2) and B (n,2).
    if len(A) == 0 or len(B) == 0:
        return np.empty((len(A), len(B)), dtype=np.float32)

    return np.linalg.norm(A - B, axis=1)


def plot_tracked_points_colored(
    prev_img,
    next_img,
    prev_pts,
    next_pts,
    status=None,
    title="Lucas-Kanade Tracking",
):
    """
    Visualize tracked point correspondences with matching colors.
    """

    if len(prev_img.shape) == 2:
        prev_rgb = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2RGB)
    else:
        prev_rgb = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)

    if len(next_img.shape) == 2:
        next_rgb = cv2.cvtColor(next_img, cv2.COLOR_GRAY2RGB)
    else:
        next_rgb = cv2.cvtColor(next_img, cv2.COLOR_BGR2RGB)

    h0, w0 = prev_rgb.shape[:2]
    h1, w1 = next_rgb.shape[:2]

    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)

    canvas[:h0, :w0] = prev_rgb
    canvas[:h1, w0:] = next_rgb

    if status is not None:
        valid = status.astype(bool)
        prev_pts = prev_pts[valid]
        next_pts = next_pts[valid]

    n_tracks = len(prev_pts)

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)

    colors = plt.cm.hsv(np.linspace(0, 1, max(n_tracks, 1)))

    for (p0, p1, c) in zip(prev_pts, next_pts, colors):

        x0, y0 = p0
        x1, y1 = p1

        plt.scatter(
            x0,
            y0,
            color=c,
            s=10,
        )

        plt.scatter(
            x1 + w0,
            y1,
            color=c,
            s=10,
        )

    plt.title(f"{title} (tracks={n_tracks + 1})")
    plt.axis("off")
    plt.savefig("plot_tracked_points_colored.jpg")
    plt.show()


def plot_arc_tracking_debug(
    camera_img,
    start_arc,
    end_arc,
    estimated_end_arc,
    status_mask=None,
):
    """
    Plot:
    - start arc
    - estimated end arc
    - end arc

    all on the SAME camera image.
    """

    if len(camera_img.shape) == 2:
        img_rgb = cv2.cvtColor(camera_img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)

    if len(start_arc) > 0:
        plt.scatter(
            start_arc[:, 0],
            start_arc[:, 1],
            c="cyan",
            s=5,
            label="Start arc",
        )

    if len(end_arc) > 0:
        plt.scatter(
            end_arc[:, 0],
            end_arc[:, 1],
            c="lime",
            s=5,
            label="End arc",
        )

    if status_mask is not None:
        valid = status_mask.astype(bool)
        estimated_end_arc = estimated_end_arc[valid]

    if len(estimated_end_arc) > 0:
        plt.scatter(
            estimated_end_arc[:, 0],
            estimated_end_arc[:, 1],
            c="red",
            s=5,
            label="Estimated end arc",
        )

    plt.title("Arc Tracking Debug")
    plt.legend()
    plt.axis("off")
    plt.savefig("plot_arc_tracking_debug.jpg")
    plt.show()


def arc_score(start_arc, end_arc, estimated_end_arc, status_mask, eps=1e-9):
    """
    Score based on the point with the largest displacement
    between start_arc and end_arc.

    Lower is better.

    d_max: largest displacement between corresponding points
           in start_arc and end_arc.
    d:      distance between the corresponding points in
            estimated_end_arc and end_arc at that same index.

    score = abs(d_max - d) / d_max
    """
    start_arc = np.asarray(start_arc, dtype=float)
    end_arc = np.asarray(end_arc, dtype=float)
    estimated_end_arc = np.asarray(estimated_end_arc, dtype=float)

    status_mask = np.asarray(status_mask, dtype=bool)

    start_arc = start_arc[status_mask]
    end_arc = end_arc[status_mask]
    estimated_end_arc = estimated_end_arc[status_mask]

    if (len(start_arc) == 0 or
        len(end_arc) == 0 or
        len(estimated_end_arc) == 0):
        return np.inf

    if not (len(start_arc) == len(end_arc) == len(estimated_end_arc)):
        raise ValueError("All arcs must have the same number of corresponding points.")

    # Displacement between corresponding start and end points
    start_end_distances = np.linalg.norm(start_arc - end_arc, axis=1)
    start_estimated_end_distances = np.linalg.norm(start_arc - estimated_end_arc, axis=1)


    # Index of the point with the largest displacement
    max_idx = np.argmax(start_estimated_end_distances)

    d_max = start_end_distances[max_idx]

    # Distance between the estimated and actual end point
    # at the index of maximum displacement
    d = np.linalg.norm(
        estimated_end_arc[max_idx] - end_arc[max_idx]
    )

    if d_max < eps:
        return 0.0 if d < eps else np.inf

    return abs(d_max - d) / d_max


def get_sonar_start_end_points(    
    sonar_i,
    sonar_ip1,
    min_features=CalibrationParameters.SONAR_MIN_FEATURES,
    max_sonar_points=CalibrationParameters.SONAR_MAX_FEATURES,
    sonar_params = sonar_parameters.HIGH_FREQUENCY
):
    
    if len(sonar_i.shape) == 2:
        sonar_i_gray = sonar_i
    else :
        sonar_i_gray = bgr_to_gray(sonar_i)
        
    if len(sonar_ip1.shape) == 2:
        sonar_ip1_gray = sonar_ip1
    else :
        sonar_ip1_gray = bgr_to_gray(sonar_ip1)

    sonar_i_filt = low_pass_filter_fft(sonar_i_gray)
    sonar_ip1_filt = low_pass_filter_fft(sonar_ip1_gray)

    pred_i = extract_sp_sonar(sonar_i_filt, sonar_params)
    pred_ip1 = extract_sp_sonar(sonar_ip1_filt, sonar_params)
    
    start_sonar_points, end_sonar_points, _, _, _ = match_sg_sonar(
        pred_i,
        pred_ip1,
        sonar_i_filt,
        sonar_ip1_filt,
        sonar_params
    )
    
    if len(start_sonar_points) < min_features:
        return None, None

    # Keep only a subset for speed
    if len(start_sonar_points) > max_sonar_points:
        idx = np.linspace(0, len(start_sonar_points) - 1, max_sonar_points).astype(int)
        start_sonar_points = start_sonar_points[idx]
        end_sonar_points = end_sonar_points[idx]

    return start_sonar_points, end_sonar_points


def evaluate_frame_pair(
    start_sonar_points,
    end_sonar_points,
    cam_i,
    cam_ip1,
    R,
    t,
    f,
    sonar_params=sonar_parameters.HIGH_FREQUENCY,
    n_elev_samples=CalibrationParameters.N_ELEV_SAMPLES
):
    """
    Score one synchronized sonar/camera frame pair.
    """

    vert_aperture_deg = sonar_params.SONAR_V_APERTURE_DEG

    camera_scores = []
    per_point_debug = []

    for i, sp0, sp1 in zip(range(0,len(start_sonar_points)), start_sonar_points, end_sonar_points):

        start_arc, _, _ = sonar_point_to_camera_arc(
            sp0,
            cam_i.shape,
            R=R,
            t=t,
            f=f,
            n_elev_samples=n_elev_samples,
            sonar_params=sonar_params
        )
        end_arc, _, _ = sonar_point_to_camera_arc(
            sp1,
            cam_ip1.shape,
            R=R,
            t=t,
            f=f,
            n_elev_samples=n_elev_samples,
            sonar_params=sonar_params
        )


        if len(start_arc) < 4 or len(end_arc) < 4:
            continue


        est_end_arc, st_cam, _ = track_points_lk(cam_i, cam_ip1, start_arc)

        st_cam = np.asarray(st_cam).reshape(-1).astype(bool)

        # Check whether all arrays have matching lengths.
        if not (
            len(start_arc) == len(end_arc) == len(est_end_arc) == len(st_cam)
        ):
            print(
                "Mismatched lengths:",
                f"start_arc={len(start_arc)},",
                f"end_arc={len(end_arc)},",
                f"est_end_arc={len(est_end_arc)},",
                f"status={len(st_cam)}",
            )
            continue
        
        if np.count_nonzero(st_cam) < 4:
            continue
        
        s = arc_score(start_arc, end_arc, est_end_arc, st_cam)

        if np.isfinite(s):
            camera_scores.append(s)
            per_point_debug.append((sp0, sp1, start_arc, end_arc, est_end_arc, s))

    if not camera_scores:
        return np.inf, {"reason": "no_valid_scores"}

    return float(np.mean(camera_scores)), {
        "reason": "ok",
        "n_valid_points": len(camera_scores),
        "mean_score": float(np.mean(camera_scores)),
        "per_point_debug": per_point_debug,
    }

@dataclass
class SearchSpace:
    tx: np.ndarray
    ty: np.ndarray
    tz: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    f: np.ndarray

def exhaustive_search(
    camera_frames,
    sonar_frames,
    search_space,
    sonar_params=sonar_parameters.HIGH_FREQUENCY,
    n_pairs=6,
):
    """
    Run a coarse-to-fine search.
    search_spaces: list[SearchSpace] from coarse to fine.
    n_pairs: number of frame pairs to check.
    """

    best = {"score": np.inf, "params": None, "details": None}

    total_candidates = (
        len(search_space.tx)
        * len(search_space.ty)
        * len(search_space.tz)
        * len(search_space.alpha)
        * len(search_space.beta)
        * len(search_space.gamma)
        * len(search_space.f)
    )
    
    candidates = product(
        search_space.tx,
        search_space.ty,
        search_space.tz,
        search_space.alpha,
        search_space.beta,
        search_space.gamma,
        search_space.f,
    )
    
    valid_pairs = []
    
    n_frames = min(len(camera_frames), len(sonar_frames))
    
    for k in range(n_frames - 1):
        if len(valid_pairs) >= n_pairs:
            break
            
        start_sonar_pts, end_sonar_pts = get_sonar_start_end_points(sonar_frames[k], sonar_frames[k + 1])

        if start_sonar_pts is None or end_sonar_pts is None:
            print(f"Skipping frame pair {k}: too few sonar points.")
            continue
            
        valid_pairs.append((k, start_sonar_pts, end_sonar_pts))

    
    for tx, ty, tz, alpha, beta, gamma, f in tqdm(
        candidates,
        total=total_candidates,
        desc="Exhaustive search",
        unit="candidate",
    ):
        R = euler_zyx_rotation(alpha, beta, gamma)
        T = np.array([tx, ty, tz], dtype=np.float64)

        scores = []
        pair_details = []
        
        for k, start_s_pts, end_s_pts in valid_pairs:
            
            s, details = evaluate_frame_pair(
                start_s_pts,
                end_s_pts,
                camera_frames[k],
                camera_frames[k + 1],
                R=R,
                t=T,
                f=f,
                sonar_params=sonar_params
            )

        
            if np.isfinite(s):
                scores.append(s)
                pair_details.append({
                    "frame_pair": (k, k + 1),
                    **details,
                })

        if not scores:
            continue
        
        mean_score = float(np.mean(scores))
        print("\nmean score: ", mean_score)
        if mean_score < best["score"]:
            best = {
                "score": mean_score,
                "params": {
                    "tx": tx,
                    "ty": ty,
                    "tz": tz,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "f": f,
                },
            }
            print("\nNew best:", best["score"], best["params"])

    return best

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    superpoint_sonar = SuperPoint({
        'nms_radius': 3,
        'keypoint_threshold': 0.005,
        'max_keypoints': 8192,
        'resize': 1600,
        'remove_borders': 5
    }).to(device).eval()

    superglue_sonar = SuperGlue({
        'weights': 'outdoor',
        'match_threshold': 0.75,
    }).to(device).eval()

    sonar_p = sonar_parameters.HIGH_FREQUENCY

    camera_frames = load_static_keyframes("dataset/2026_02_25_12_00_46_cropped.mp4")
    sonar_frames = load_static_keyframes_sonar("dataset/20260225_120049_cropped.avi", sonar_params=sonar_p)

    print("Camera frames:", len(camera_frames))
    print("Sonar frames:", len(sonar_frames))
    print("Camera frame shape:", camera_frames[0].shape)
    print("Sonar frame shape:", sonar_frames[0].shape)

    coarse = SearchSpace(
    tx=np.array([-0.05, 0.00, 0.05]),
    ty=np.array([-0.10, -0.15, -0.20]),
    tz=np.array([-0.05, 0.0, 0.05]),
    alpha=np.deg2rad(np.array([80, 85, 90])),
    beta=np.deg2rad(np.array([-2.5, 0.0, 2.5])),
    gamma=np.deg2rad(np.array([-2.5, 0.0, 2.5])),
    f=np.array([1000, 1100, 1200, 1300]),
)

    best_coarse = exhaustive_search(camera_frames, sonar_frames, coarse, sonar_p)
    print("Best parameters found (coarse search):", best_coarse)

    p = best_coarse["params"]

    fine = SearchSpace(
        tx=np.linspace(p["tx"] - 0.025, p["tx"] + 0.025, 3),
        ty=np.linspace(p["ty"] - 0.05, p["ty"] + 0.05, 5),
        tz=np.linspace(p["tz"] - 0.025, p["tz"] + 0.025, 3),

        alpha=np.deg2rad(np.linspace(
            np.rad2deg(p["alpha"]) - 5,
            np.rad2deg(p["alpha"]) + 5,
            5,
        )),
        beta=np.deg2rad(np.linspace(
            np.rad2deg(p["beta"]) - 1,
            np.rad2deg(p["beta"]) + 1,
            3,
        )),
        gamma=np.deg2rad(np.linspace(
            np.rad2deg(p["gamma"]) - 1,
            np.rad2deg(p["gamma"]) + 1,
            3,
        )),

        f=np.linspace(p["f"] - 50, p["f"] + 50, 5),
    )

    best_fine = exhaustive_search(camera_frames, sonar_frames, fine, sonar_p)
    print("Best parameters found (fine search):", best_fine)
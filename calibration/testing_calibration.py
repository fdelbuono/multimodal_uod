import json
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from calibration.calibration_config import Calibration
from calibration.sonar_camera_sensor_models import *
from calibration.calibration_pipeline import *


def calculate_pixel_error(p_s, p_c, calibration, sonar_parameters):

    alpha = calibration["alpha"]
    beta = calibration["beta"]
    gamma = calibration["gamma"]
    tx = calibration["tx"]
    ty = calibration["ty"]
    tz = calibration["tz"]
    focal_length = calibration["f"]

    scaling_factor = sonar_parameters.SONAR_SCALING_FACTOR
    tip_coord = sonar_parameters.SONAR_TIP_PIXEL_COORD
    vert_aperture_deg = sonar_parameters.SONAR_V_APERTURE_DEG

    cx = 960
    cy = 540

    n_elev_samples = CalibrationParameters.N_ELEV_SAMPLES

    rho, theta = sonar_pixel_to_physical(p_s, scaling_factor, tip_coord)

    R = euler_zyx_rotation(alpha, beta, gamma)

    t = np.array([tx, ty, tz])

    phi_min = -np.deg2rad(vert_aperture_deg / 2.0)
    phi_max =  np.deg2rad(vert_aperture_deg / 2.0)
    phis = np.linspace(phi_min, phi_max, n_elev_samples)

    xs = np.full((n_elev_samples,), rho * np.cos(phis) * np.sin(theta))
    ys = np.full((n_elev_samples,), rho * np.cos(phis) * np.cos(theta))
    zs = rho * np.sin(phis)
    P_s = np.stack([xs, ys, zs], axis=1)

    P_c = (R @ P_s.T).T + t.reshape(1, 3)

     # Project the 3D sonar points into the camera image
    u_cal = focal_length * (P_c[:, 0] / P_c[:, 2]) + cx
    v_cal = focal_length * (P_c[:, 1] / P_c[:, 2]) + cy

    projected_points = np.column_stack((u_cal, v_cal))
    p_c = np.asarray(p_c, dtype=float)

    # Euclidean distance from the observed camera point
    # to every projected point
    distances = np.linalg.norm(projected_points - p_c, axis=1)

    # Shortest distance
    error = np.min(distances)

    return error

def calculate_mean_pixel_error(pts_sonar, pts_camera, calibration, sonar_parameters):

    errors = []

    for (p_s, p_c) in zip(pts_sonar, pts_camera):
        error = calculate_pixel_error(p_s, p_c, calibration, sonar_parameters)

        errors.append(float(error))

    errors = np.asarray(errors)

    return {
        "mean": np.mean(errors),
        "median": np.median(errors),
        "std": np.std(errors),
        "max": np.max(errors),
        "errors": errors,
    }




if __name__ == "__main__":

    calibration_dir = "./calibration"
    output_dir = "./calibration/calibration_results_croatia"

    os.makedirs(output_dir, exist_ok=True)

    # sonar_parameters = SonarParameters.HIGH_FREQUENCY
    sonar_parameters = SonarParameters.LOW_FREQUENCY

    # pts_sonar = [
    #     (803, 482), (848, 476), (812, 528), (854, 523),
    #     (782, 466), (826, 447), (795, 512), (838, 491),
    #     (809, 448), 
    #     # (942, 420), 
    #     (834, 487), (875, 458),
    #     (870, 583), (908, 601), (851, 626), (891, 645),
    #     (922, 577), (957, 605), (893, 613), (927, 643),
    #     (971, 630), (995, 669), (931, 656), (959, 695),
    #     (918, 519), (985, 528), (931, 569), (973, 575)
    # ]

    # pts_sonar = [
    #     (865, 519), (871, 565), (829, 527), (819, 524),
    #     (879, 538), (897, 581), (851, 599), (835, 555),
    #     (888, 542), (919, 581), (879, 609), (852, 571),
    #     (972, 636), (949, 677), (907, 654), (930, 613),
    #     (946, 667), (917, 702), (880, 673), (908, 637),
    #     (982, 756), (941, 781), (917, 744), (956, 719),
    #     (988, 484), (979, 530), (923, 525), (942, 478)
    #     ]

    # pts_camera = [
    #     (673, 292), (871, 283), (650, 365), (889, 360),
    #     (621, 301), (810, 281), (644, 373), (879, 347),
    #     (725, 282), 
    #     # (883, 256), 
    #     (834, 340), (1007, 305),
    #     (928, 335), (1173, 371), (809, 434), (1122, 492),
    #     (1085, 348), (1326, 411), (923, 433), (1210, 523),
    #     (1245, 370), (1490, 477), (1029, 450), (1308, 600),
    #     (981, 296), (1197, 310), (959, 376), (1230, 392)
    # ]

    # pts_camera = [
    #     (908, 303), (944, 385), (678, 395), (701, 314),
    #     (950, 322), (1078, 406), (811, 436), (737, 343),
    #     (993, 322), (1171, 397), (956, 454), (798, 364),
    #     (1321, 452), (1303, 640), (885, 545), (1013, 395),
    #     (1328, 471), (1168, 647), (833, 508), (1041, 384),
    #     (1600, 656), (1342, 905), (976, 611), (1264, 477),
    #     (1171, 299), (1197, 369), (959, 356), (979, 291)
    #     ]

    pts_sonar = [
        (405, 645),
        (440, 565),
        (507, 492),
        (456, 557),
        (337, 503),
        (235, 619),
        (325, 619),
        (281, 660),
        (332, 543),
        (422, 461),
        (246, 468),
        (407, 505),
        (312, 544),
        (470, 337),
        (377, 395)
    ]

    pts_camera = [
        (1121, 340),
        (1612, 194),
        (1780, 124),
        (1634, 326),
        (1185, 141),
        (161, 290),
        (1017, 228),
        (540, 467),
        (1034, 279),
        (1320, 144),
        (731, 169),
        (1138, 643),
        (852, 798),
        (1197, 402),
        (950, 478)
    ]

    calibration_files = sorted(
        glob.glob(os.path.join(calibration_dir, "*.json"))
    )

    if not calibration_files:
        raise FileNotFoundError(
            f"No calibration JSON files found in {calibration_dir}"
        )

    print(f"Found {len(calibration_files)} calibration files.")

    # Store each calibration's results
    all_results = []
    all_errors = []
    all_scores = []


    for filepath in calibration_files:

        filename = os.path.basename(filepath)
        calibration_name = Path(filepath).stem

        print(f"\nEvaluating: {filename}")

        try:
            calibration = Calibration(filepath)

            results = calculate_mean_pixel_error(
                pts_sonar,
                pts_camera,
                calibration.params,
                sonar_parameters
            )

            errors = np.asarray(results["errors"], dtype=float)
            errors = errors[np.isfinite(errors)]

            if len(errors) == 0:
                print("No valid errors. Skipping.")
                continue

            # Calibration score
            score = calibration.score

            # Support either a property or a method
            if callable(score):
                score = score()

            score = float(score)

            # Statistics for this calibration
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)

            all_errors.extend(errors.tolist())
            all_scores.append(score)

            all_results.append({
                "name": calibration_name,
                "score": score,
                "mean": mean_error,
                "median": median_error,
                "std": std_error,
                "max": max_error,
                "errors": errors,
                "n": len(errors),
            })

            print(f"Score:        {score:.4f}")
            print(f"Mean error:   {mean_error:.2f} px")
            print(f"Median error: {median_error:.2f} px")
            print(f"Std error:    {std_error:.2f} px")
            print(f"Maximum:      {max_error:.2f} px")


            fig, ax = plt.subplots(figsize=(7, 4))

            ax.boxplot(
                errors,
                vert=False,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "green",
                    "markeredgecolor": "green",
                },
                medianprops={
                    "color": "black",
                    "linewidth": 2,
                },
            )

            ax.set_xlabel("Pixel error (px)")
            ax.set_title(
                f"Calibration Error Spread\n{calibration_name}"
            )
            ax.grid(axis="x", alpha=0.3)

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"{calibration_name}_boxplot.png"
                ),
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(fig)


            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(
                np.arange(1, len(errors) + 1),
                errors,
                linestyle="None",
                marker="o",
                color="steelblue",
                label="Individual errors",
            )

            ax.axhline(
                mean_error,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_error:.2f} px",
            )

            ax.axhline(
                median_error,
                color="black",
                linestyle="-.",
                label=f"Median: {median_error:.2f} px",
            )

            ax.set_xlabel("Correspondence index")
            ax.set_ylabel("Pixel error (px)")
            ax.set_title(
                f"Calibration Error per Matched Point\n"
                f"{calibration_name}"
            )
            ax.legend()
            ax.grid(alpha=0.3)

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"{calibration_name}_correspondence_errors.png"
                ),
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(fig)

        except Exception as exc:
            print(f"Failed to evaluate {filename}: {exc}")


    if not all_results:
        raise RuntimeError("No calibrations were evaluated successfully.")

    pooled_errors = np.asarray(all_errors, dtype=float)
    scores = np.asarray(all_scores, dtype=float)

    print("\n" + "=" * 60)
    print("OVERALL CALIBRATION SUMMARY")
    print("=" * 60)

    print(f"Successful calibrations: {len(all_results)}")
    print(f"Total correspondences:   {len(pooled_errors)}")

    print("\nCalibration scores:")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Median score: {np.median(scores):.4f}")
    print(f"Std score: {np.std(scores):.4f}")

    print("\nPooled pixel errors (all correspondences):")
    print(f"Mean error:   {np.mean(pooled_errors):.2f} px")
    print(f"Median error: {np.median(pooled_errors):.2f} px")
    print(f"Std error:    {np.std(pooled_errors):.2f} px")
    print(f"Maximum error: {np.max(pooled_errors):.2f} px")

    print("\nAverage of individual calibration statistics:")
    print(
        f"Mean of means:   "
        f"{np.mean([r['mean'] for r in all_results]):.2f} px"
    )
    print(
        f"Mean of medians: "
        f"{np.mean([r['median'] for r in all_results]):.2f} px"
    )
    print(
        f"Mean of stds:    "
        f"{np.mean([r['std'] for r in all_results]):.2f} px"
    )

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(all_results) * 0.45))
    )

    ax.boxplot(
        [r["errors"] for r in all_results],
        vert=False,
        tick_labels=[r["name"] for r in all_results],
        showmeans=True,
    )

    ax.set_xlabel("Pixel error (px)")
    ax.set_title("Pixel Error Distribution Across Calibrations")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "all_calibrations_boxplot.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)


    names = [r["name"] for r in all_results]
    means = [r["mean"] for r in all_results]

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(names))
    bars = ax.bar(x, means)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Mean pixel error (px)")
    ax.set_title("Mean Reprojection Error per Calibration")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "calibration_mean_errors.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Score comparison
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x, scores)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Calibration score")
    ax.set_title("Calibration Score Comparison")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "calibration_scores.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)


    import csv

    csv_path = os.path.join(output_dir, "calibration_summary.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name", "score", "mean", "median",
                "std", "max", "n"
            ]
        )
        writer.writeheader()

        for r in all_results:
            writer.writerow({
                "name": r["name"],
                "score": r["score"],
                "mean": r["mean"],
                "median": r["median"],
                "std": r["std"],
                "max": r["max"],
                "n": r["n"],
            })

    print(f"\nGraphs and summary saved to: {output_dir}")
    print(f"CSV summary: {csv_path}")
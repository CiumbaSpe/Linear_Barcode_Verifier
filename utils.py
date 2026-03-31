import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_patch_maps(
    image,
    energy_map,
    coherence_map,
    theta_map,
    stride,
    patch_size,
    energy_percentile=80,
    scale=4,
):
    energy_vis = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    coherence_vis = (255 * np.clip(coherence_map, 0.0, 1.0)).astype(np.uint8)

    energy_vis_big = cv2.resize(
        energy_vis,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    coherence_vis_big = cv2.resize(
        coherence_vis,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    energy_thr = np.percentile(energy_map, energy_percentile)
    half_len = patch_size // 3

    n_y, n_x = energy_map.shape
    for i in range(n_y):
        for j in range(n_x):
            if energy_map[i, j] >= energy_thr:
                cy = i * stride + patch_size // 2
                cx = j * stride + patch_size // 2

                angle = theta_map[i, j]
                dx = int(np.cos(angle) * half_len)
                dy = int(np.sin(angle) * half_len)

                # Draw on top of the original image, a short green segment for each patch whose energy is high enough.
                pt1 = (cx - dx, cy - dy)
                pt2 = (cx + dx, cy + dy)
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 1)

    cv2.imshow("Energy map", energy_vis_big)
    cv2.imshow("Coherence map", coherence_vis_big)
    cv2.imshow("Orientation overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_candidate_mask(image, candidate_mask, stride, patch_size, scale=4):
    mask_vis = (255 * candidate_mask.astype(np.uint8)).astype(np.uint8)
    mask_vis_big = cv2.resize(
        mask_vis,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    base_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlap_count = np.zeros(image.shape[:2], dtype=np.float32)
    n_y, n_x = candidate_mask.shape

    for i in range(n_y):
        for j in range(n_x):
            if not candidate_mask[i, j]:
                continue

            y0 = i * stride
            x0 = j * stride
            y1 = min(y0 + patch_size, image.shape[0])
            x1 = min(x0 + patch_size, image.shape[1])

            overlap_count[y0:y1, x0:x1] += 1.0

    if overlap_count.max() > 0:
        overlap_vis = cv2.normalize(overlap_count, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        overlap_heatmap = cv2.applyColorMap(overlap_vis, cv2.COLORMAP_TURBO)
        overlap_overlay = cv2.addWeighted(base_image, 0.45, overlap_heatmap, 0.55, 0.0)
    else:
        overlap_vis = np.zeros(image.shape[:2], dtype=np.uint8)
        overlap_overlay = base_image.copy()

    cv2.imshow("Candidate mask", mask_vis_big)
    cv2.imshow("Candidate overlap overlay", overlap_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_best_cluster_mask(best_cluster_mask, scale=0.5):
    mask_vis = best_cluster_mask
    if best_cluster_mask.dtype != np.uint8:
        mask_vis = best_cluster_mask.astype(np.uint8)

    resized_mask = cv2.resize(
        mask_vis,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.imshow("Best cluster mask", resized_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_image(image): 
    cv2.imshow("Rotated image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_profile(profile):
    plt.figure(figsize=(12, 4))
    plt.plot(profile, color="black")
    plt.title("Central scanline intensity profile")
    plt.xlabel("x")
    plt.ylabel("intensity")
    plt.grid(True)
    plt.show()


def visualize_best_cluster_bbox(image, box):
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [np.int32(box)], -1, (0, 255, 0), 2)

    cv2.imshow("Best cluster bbox", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

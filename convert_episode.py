import tensorflow as tf
import numpy as np
import os
from skimage.draw import polygon
import imageio.v2 as imageio


# === Utility ===
def rotate_points(cx, cy, yaw, length, width):
    """Return corner points of a rotated rectangle."""
    dx = length / 2
    dy = width / 2
    corners = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = corners @ rot.T
    return rotated + np.array([cx, cy])

def make_circle_kernel(radius, value=0.4):
    """Create a small filled circle kernel (for lane dots)."""
    d = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((d, d), dtype=np.float32)
    kernel[mask] = value
    return kernel

def rasterize_scene(x_center, y_center, extent, lane_centers, agent_boxes, H=256, W=256, lane_radius=8):
    """
    Fast rasterization with circle stamping for lane centers.
    """
    img = np.zeros((H, W), dtype=np.float32)

    # Coordinate transform (world -> pixel)
    scale = H / (2 * extent)
    def world_to_pix(x, y):
        px = (x - (x_center - extent)) * scale
        py = (y - (y_center - extent)) * scale
        return int(round(px)), int(round(H - py))

    # Precompute circle kernel
    circle_kernel = make_circle_kernel(lane_radius, value=0.1)
    kr, kc = circle_kernel.shape

    # Rasterize lane centers as circle stamps
    for (lx, ly) in lane_centers:
        px, py = world_to_pix(lx, ly)
        r0, r1 = max(0, py - lane_radius), min(H, py + lane_radius + 1)
        c0, c1 = max(0, px - lane_radius), min(W, px + lane_radius + 1)
        kr0, kr1 = r0 - (py - lane_radius), kr - ((py + lane_radius + 1) - r1)
        kc0, kc1 = c0 - (px - lane_radius), kc - ((px + lane_radius + 1) - c1)
        img[r0:r1, c0:c1] = np.maximum(
            img[r0:r1, c0:c1],
            circle_kernel[kr0:kr1, kc0:kc1]
        )

    # Rasterize agent boxes (still polygons, but usually fewer)
    from skimage.draw import polygon
    for poly in agent_boxes:
        px, py = zip(*[world_to_pix(x, y) for x, y in poly])
        rr, cc = polygon(py, px, img.shape)
        img[rr, cc] = 1.0

    return img

# === Core Transformation ===
def tfrecord_to_images(tfrecord_path, output_base_dir="renders", H=800, W=800):
    """
    Optimized rasterization version: processes *all* episodes in one TFRecord.
    """
    import tensorflow as tf
    import numpy as np
    import os, imageio

    # Output dir (folder named after the tfrecord file)
    file_name = os.path.basename(tfrecord_path).replace(".tfrecord", "")
    output_dir = os.path.join(output_base_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load all examples from TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    for example_idx, raw_example in enumerate(dataset):

        # Create subfolder for this episode
        ep_dir = os.path.join(output_dir, str(example_idx))
        os.makedirs(ep_dir, exist_ok=True)

        example = tf.train.Example()
        example.ParseFromString(raw_example.numpy())

        # --- same parsing and rasterization code as before ---
        # Roadgraph
        roadgraph_xyz = np.array(example.features.feature['roadgraph_samples/xyz'].float_list.value).reshape(-1, 3)
        roadgraph_valid = np.array(example.features.feature['roadgraph_samples/valid'].int64_list.value)
        roadgraph_type = np.array(example.features.feature['roadgraph_samples/type'].int64_list.value)

        # Agents
        num_agents = len(example.features.feature['state/current/x'].float_list.value)
        def get_feature(key, shape=None):
            vals = np.array(example.features.feature[key].float_list.value)
            return vals.reshape(shape) if shape else vals

        def get_feature_int(key, shape=None):
            vals = np.array(example.features.feature[key].int64_list.value)
            return vals.reshape(shape) if shape else vals

        state_x_past = get_feature('state/past/x', (num_agents, 10))
        state_y_past = get_feature('state/past/y', (num_agents, 10))
        state_yaw_past = get_feature('state/past/bbox_yaw', (num_agents, 10))
        state_valid_past = get_feature_int('state/past/valid', (num_agents, 10))

        state_x_cur = get_feature('state/current/x')
        state_y_cur = get_feature('state/current/y')
        state_yaw_cur = get_feature('state/current/bbox_yaw')
        state_valid_cur = get_feature_int('state/current/valid')

        state_x_fut = get_feature('state/future/x', (num_agents, 80))
        state_y_fut = get_feature('state/future/y', (num_agents, 80))
        state_yaw_fut = get_feature('state/future/bbox_yaw', (num_agents, 80))
        state_valid_fut = get_feature_int('state/future/valid', (num_agents, 80))

        state_length = get_feature('state/current/length')
        state_width = get_feature('state/current/width')

        # Build timeline
        dt = 0.1
        all_x = np.hstack([state_x_past, state_x_cur[:, None], state_x_fut])
        all_y = np.hstack([state_y_past, state_y_cur[:, None], state_y_fut])
        all_yaw = np.hstack([state_yaw_past, state_yaw_cur[:, None], state_yaw_fut])
        all_valid = np.hstack([state_valid_past, state_valid_cur[:, None], state_valid_fut])

        # Scene extent
        valid_mask_global = all_valid > 0
        if not valid_mask_global.any():
            continue
        x_vals_global = all_x[valid_mask_global]
        y_vals_global = all_y[valid_mask_global]
        x_center = (x_vals_global.min() + x_vals_global.max()) / 2
        y_center = (y_vals_global.min() + y_vals_global.max()) / 2
        extent = max(x_vals_global.max() - x_vals_global.min(),
                     y_vals_global.max() - y_vals_global.min()) / 2
        extent *= 1.1

        # Sliding window
        window_size = int(1.5 / dt)
        delta_t = int(1.0 / dt)
        stride = int(0.5 / dt)

        for start in range(0, all_x.shape[1] - window_size + 1, delta_t):
            timesteps = [start, start + stride, start + 2*stride]

            # Skip if no valid agent across all timesteps
            valid_agents = np.array([all_valid[i, timesteps].all() for i in range(num_agents)])
            if not valid_agents.any():
                continue

            channels = []
            for t in timesteps:
                # Lane centers in viewport
                mask_rg = (roadgraph_xyz[:,0] >= x_center-extent) & \
                          (roadgraph_xyz[:,0] <= x_center+extent) & \
                          (roadgraph_xyz[:,1] >= y_center-extent) & \
                          (roadgraph_xyz[:,1] <= y_center+extent)
                lane_center_mask = (roadgraph_type == 2) & (roadgraph_valid > 0) & mask_rg
                lane_centers = roadgraph_xyz[lane_center_mask, :2]

                # Agent boxes
                agent_boxes = []
                for i in range(num_agents):
                    if t >= all_x.shape[1] or all_valid[i, t] == 0:
                        continue
                    x, y, yaw = all_x[i, t], all_y[i, t], all_yaw[i, t]
                    if not (x_center-extent <= x <= x_center+extent and y_center-extent <= y <= y_center+extent):
                        continue
                    poly = rotate_points(x, y, yaw, state_length[i], state_width[i])
                    agent_boxes.append(poly)

                # Rasterize one channel
                img_gray = rasterize_scene(x_center, y_center, extent, lane_centers, agent_boxes, H=H, W=W)
                channels.append(img_gray)

            # Merge into RGB
            rgb_img = np.stack(channels, axis=2)

            # Save into this episode’s folder
            out_path = os.path.join(ep_dir, f"win{start:03d}.png")
            imageio.imwrite(out_path, (rgb_img * 255).astype(np.uint8))


def main():
    tfrecord_path = "test/training_tfexample.tfrecord-00605-of-01000"
    tfrecord_to_images(tfrecord_path, output_base_dir="converted_dataset")

if __name__ == "__main__":
    main()


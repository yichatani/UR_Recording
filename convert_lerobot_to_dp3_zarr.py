import os
import zarr
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from termcolor import cprint

# =========================
# Config
# =========================
DATA_ROOT = "/home/ani/UR_Recording/data/pour_water"
REPO_ID = "ani/sid"
SAVE_ZARR_PATH = "./lerobot_dp3.zarr"

IMG_SIZE = 84
USE_DUMMY_POINT_CLOUD = True
NUM_PC_POINTS = 1024

# =========================
# Utils
# =========================
def chw_to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    return x.numpy().astype(np.uint8)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    img = torchvision.transforms.functional.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.permute(1, 2, 0)
    return img.numpy().astype(np.uint8)

def make_dummy_point_cloud(num_points=1024):
    xyz = np.random.uniform(-0.5, 0.5, size=(num_points, 3))
    rgb = np.random.uniform(0, 1, size=(num_points, 3))
    return np.concatenate([xyz, rgb], axis=-1)

# =========================
# Load dataset
# =========================
ds = LeRobotDataset(repo_id=REPO_ID, root=DATA_ROOT)

ep_col = np.asarray(ds.hf_dataset["episode_index"])
fr_col = np.asarray(ds.hf_dataset["frame_index"])

unique_eps = np.unique(ep_col)

# =========================
# Storage buffers
# =========================
img_arrays = []
depth_arrays = []
pc_arrays = []
state_arrays = []
action_arrays = []
episode_ends = []

total_count = 0

# =========================
# Main loop
# =========================
for ep_id in unique_eps:
    idxs = np.where(ep_col == ep_id)[0]
    idxs = idxs[np.argsort(fr_col[idxs])]

    cprint(f"Processing episode {ep_id}, len={len(idxs)}", "green")

    for i in tqdm(idxs):
        sample = ds[int(i)]

        # ---------- Image ----------
        rgb = chw_to_hwc_uint8(sample["rgb_global"])
        rgb = preprocess_image(rgb)
        img_arrays.append(rgb)

        # ---------- Depth ----------
        if "depth_wrist" in sample:
            depth = sample["depth_wrist"].detach().cpu().numpy()
            depth = torch.from_numpy(depth).unsqueeze(0)
            depth = torchvision.transforms.functional.resize(depth, (IMG_SIZE, IMG_SIZE))
            depth_arrays.append(depth.squeeze(0).numpy())
        else:
            depth_arrays.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))

        # ---------- Point Cloud ----------
        if USE_DUMMY_POINT_CLOUD:
            pc = make_dummy_point_cloud(NUM_PC_POINTS)
        else:
            raise NotImplementedError("Depth → point cloud not implemented yet")
        pc_arrays.append(pc)

        # ---------- State / Action ----------
        state = sample["observation.state"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        action = sample["action"].detach().cpu().numpy().astype(np.float32).reshape(-1)

        state_arrays.append(state)
        action_arrays.append(action)

        total_count += 1

    episode_ends.append(total_count)

# =========================
# Stack
# =========================
img_arrays = np.stack(img_arrays, axis=0)
depth_arrays = np.stack(depth_arrays, axis=0)
pc_arrays = np.stack(pc_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
episode_ends = np.array(episode_ends, dtype=np.int64)

# =========================
# Write zarr
# =========================
if os.path.exists(SAVE_ZARR_PATH):
    cprint(f"Overwriting {SAVE_ZARR_PATH}", "red")
    os.system(f"rm -rf {SAVE_ZARR_PATH}")

zarr_root = zarr.group(SAVE_ZARR_PATH)
zarr_data = zarr_root.create_group("data")
zarr_meta = zarr_root.create_group("meta")

compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

zarr_data.create_dataset(
    "img", img_arrays,
    chunks=(100, IMG_SIZE, IMG_SIZE, 3),
    dtype="uint8",
    compressor=compressor
)

zarr_data.create_dataset(
    "depth", depth_arrays,
    chunks=(100, IMG_SIZE, IMG_SIZE),
    dtype="float32",
    compressor=compressor
)

zarr_data.create_dataset(
    "point_cloud", pc_arrays,
    chunks=(100, NUM_PC_POINTS, 6),
    dtype="float64",
    compressor=compressor
)

zarr_data.create_dataset(
    "state", state_arrays,
    chunks=(100, state_arrays.shape[1]),
    dtype="float32",
    compressor=compressor
)

zarr_data.create_dataset(
    "action", action_arrays,
    chunks=(100, action_arrays.shape[1]),
    dtype="float32",
    compressor=compressor
)

zarr_meta.create_dataset(
    "episode_ends", episode_ends,
    chunks=(100,),
    dtype="int64",
    compressor=compressor
)

# =========================
# Summary
# =========================
cprint("Saved DP3 zarr dataset:", "green")
cprint(f"img: {img_arrays.shape}", "green")
cprint(f"depth: {depth_arrays.shape}", "green")
cprint(f"pc: {pc_arrays.shape}", "green")
cprint(f"state: {state_arrays.shape}", "green")
cprint(f"action: {action_arrays.shape}", "green")
cprint(f"episode_ends: {episode_ends}", "green")

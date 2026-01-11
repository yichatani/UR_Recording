import rerun as rr
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA_ROOT = "/home/ani/UR_Recording/data/single_pick_place"
REPO_ID = "ani/sid"
EP_ID = 0

ds = LeRobotDataset(repo_id=REPO_ID, root=DATA_ROOT)

# --- 找到 EP_ID 对应的所有全局帧索引，并按 frame_index 排序 ---
ep_col = np.asarray(ds.hf_dataset["episode_index"])
fr_col = np.asarray(ds.hf_dataset["frame_index"])

idxs = np.where(ep_col == EP_ID)[0]
idxs = idxs[np.argsort(fr_col[idxs])]

print(f"Episode {EP_ID}: {len(idxs)} frames")

def chw_to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)  # CHW -> HWC
    return x.numpy()

# 你的 state/action 维度名字（7维）
names = ["x", "y", "z", "rx", "ry", "rz", "width"]

rr.init(f"lerobot_vis_ep{EP_ID}", spawn=True)

for global_i in idxs:
    sample = ds[int(global_i)]

    frame_i = int(sample["frame_index"].item()) if hasattr(sample["frame_index"], "item") else int(sample["frame_index"])
    rr.set_time("frame", sequence=frame_i)

    # --- 相机 ---
    rgb_wrist = chw_to_hwc_uint8(sample["rgb_wrist"])
    rgb_global = chw_to_hwc_uint8(sample["rgb_global"])
    rr.log("cam/wrist", rr.Image(rgb_wrist))
    rr.log("cam/global", rr.Image(rgb_global))

    # --- state: 7个标量 ---
    st = sample["observation.state"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    for i, n in enumerate(names):
        rr.log(f"state/{n}", rr.Scalars(float(st[i])))

    # --- action: 7个标量 ---
    ac = sample["action"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    for i, n in enumerate(names):
        rr.log(f"action/{n}", rr.Scalars(float(ac[i])))

    # --- 深度图（可选）---
    if "depth_wrist" in sample:
        depth = sample["depth_wrist"].detach().cpu().numpy().astype(np.float32)
        rr.log("depth/wrist", rr.Image(depth))
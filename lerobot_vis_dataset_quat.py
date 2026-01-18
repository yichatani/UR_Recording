import rerun as rr
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA_ROOT = "/home/ani/UR_Recording/data/sid_banana_picking"
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

# ✅ 更新为 8 维：四元数 (qx, qy, qz, qw) 代替旋转向量
names = ["x", "y", "z", "qx", "qy", "qz", "qw", "width"]

rr.init(f"lerobot_vis_ep{EP_ID}", spawn=True)

for wrist_i in idxs:
    sample = ds[int(wrist_i)]

    frame_i = int(sample["frame_index"].item()) if hasattr(sample["frame_index"], "item") else int(sample["frame_index"])
    rr.set_time("frame", sequence=frame_i)

    # --- 相机 ---
    rgb_wrist = chw_to_hwc_uint8(sample["rgb_wrist"])
    rr.log("cam/wrist", rr.Image(rgb_wrist))
    if "rgb_global" in sample:
        rgb_global = chw_to_hwc_uint8(sample["rgb_global"])
        rr.log("cam/global", rr.Image(rgb_global))
    # rgb_global = chw_to_hwc_uint8(sample["rgb_global"])
    # rr.log("cam/global", rr.Image(rgb_global))

    # --- state: 8 个标量 ---
    st = sample["observation.state"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    for i, n in enumerate(names):
        rr.log(f"state/{n}", rr.Scalars(float(st[i])))

    # --- action: 8 个标量 ---
    ac = sample["action"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    for i, n in enumerate(names):
        rr.log(f"action/{n}", rr.Scalars(float(ac[i])))

    # ✅ 可选：直接可视化 3D 位姿 (Rerun 原生支持)
    # state 的前 7 维是 [x, y, z, qx, qy, qz, qw]
    pos = st[:3]
    quat_xyzw = st[3:7]  # Rerun 用 xyzw 顺序
    rr.log("robot/eef_state", rr.Transform3D(translation=pos, rotation=quat_xyzw))
    
    # action 同理
    pos_act = ac[:3]
    quat_act = ac[3:7]
    rr.log("robot/eef_action", rr.Transform3D(translation=pos_act, rotation=quat_act))

    # --- 深度图（可选）---
    if "depth_wrist" in sample:
        depth = sample["depth_wrist"].detach().cpu().numpy().astype(np.float32)
        rr.log("depth/wrist", rr.Image(depth))
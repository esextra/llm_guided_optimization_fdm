import math
import itertools
from typing import Dict, Tuple, List

import numpy as np


                                                                

def Rx(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s,  c]])


def Ry(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def Rz(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def prusa_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    return Rz(rz) @ (Ry(ry) @ Rx(rx))


def rotation_matrix_to_axis_angle(R: np.ndarray, tol: float = 1e-8):
    tr = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    theta = math.acos(cos_theta)

    if abs(theta) < tol:
        return np.array([1.0, 0.0, 0.0]), 0.0

    denom = 2.0 * math.sin(theta)
    if abs(denom) < tol:
        vals, vecs = np.linalg.eig(R)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        axis = np.real(vecs[:, idx])
        axis /= np.linalg.norm(axis)
        return axis, theta

    ux = (R[2, 1] - R[1, 2]) / denom
    uy = (R[0, 2] - R[2, 0]) / denom
    uz = (R[1, 0] - R[0, 1]) / denom
    axis = np.array([ux, uy, uz], dtype=float)
    n = np.linalg.norm(axis)
    if n < tol:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis /= n
    return axis, theta

def _greedy_rotation_order_from_lut(lut_keys: np.ndarray, lut_vals: np.ndarray) -> np.ndarray:
       
    if lut_keys.size == 0:
        return np.zeros((0,), dtype=int)

                                                       
                                                                   
                                             
    ry = lut_vals[:, 0].astype(float)
    rx = lut_vals[:, 1].astype(float)
    rz = lut_vals[:, 2].astype(float)

    N = lut_vals.shape[0]
    R_all = np.empty((N, 3, 3), dtype=float)
    for i in range(N):
        R_all[i] = prusa_rotation_matrix(rx[i], ry[i], rz[i])

                                               
                                                                                  
    F = R_all.reshape(N, 9)

                                                                                
    start_candidates = np.where((lut_vals == 0).all(axis=1))[0]
    cur = int(start_candidates[0]) if start_candidates.size > 0 else 0

    mask = np.ones(N, dtype=bool)
    mask[cur] = False
    perm = np.empty(N, dtype=int)
    perm[0] = cur

    for t in range(1, N):
                                                            
        tr = F @ F[cur]        
        cos_theta = (tr - 1.0) * 0.5
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta[~mask] = np.inf

        nxt = int(np.argmin(theta))
        perm[t] = nxt
        mask[nxt] = False
        cur = nxt

    return perm


def _print_lut_preview(lut_keys: np.ndarray, lut_vals: np.ndarray, max_rows: int = 10) -> None:
       
    if lut_keys.size == 0:
        print("[preview] LUT is empty.")
        return

    N = int(lut_keys.shape[0])
    m = min(int(max_rows), N)

    print(f"[preview] First {m} LUT entries (final order):")
    prev_R = None
    for i in range(m):
        up_idx = int(lut_keys[i, 0])
        yaw_deg = int(lut_keys[i, 1])
        ry = int(lut_vals[i, 0])
        rx = int(lut_vals[i, 1])
        rz = int(lut_vals[i, 2])

        R = prusa_rotation_matrix(rx, ry, rz)
        step_deg = 0.0
        if prev_R is not None:
            R_rel = R @ prev_R.T
            _, ang = rotation_matrix_to_axis_angle(R_rel)
            step_deg = float(abs(math.degrees(ang)))
        prev_R = R

        print(f"  {i:4d}: pair=({up_idx:3d},{yaw_deg:3d})  angles(ry,rx,rz)=({ry:3d},{rx:3d},{rz:3d})  step_deg={step_deg:7.3f}")


def enumerate_unique_orientations(
    rx_values: List[int],
    ry_values: List[int],
    rz_values: List[int],
    up_axis: str = "z",
    R_round_decimals: int = 8,
    up_round_decimals: int = 8,
):
    axis_vecs = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    if up_axis.lower() not in axis_vecs:
        raise ValueError(f"up_axis must be one of 'x','y','z', got '{up_axis}'")
    up_basis = axis_vecs[up_axis.lower()]

                                                     
    orientation_list = []
    R_key_to_index: Dict[Tuple[float, ...], int] = {}

    for rx_deg, ry_deg, rz_deg in itertools.product(rx_values, ry_values, rz_values):
        R = prusa_rotation_matrix(rx_deg, ry_deg, rz_deg)
        R_key = tuple(np.round(R.flatten(), R_round_decimals))
        if R_key in R_key_to_index:
            continue
        R_key_to_index[R_key] = len(orientation_list)
        up_vec = R @ up_basis
        up_vec = up_vec / np.linalg.norm(up_vec)
        orientation_list.append(
            {"rx": int(rx_deg), "ry": int(ry_deg), "rz": int(rz_deg),
             "R": R, "up_vec": up_vec}
        )

    print(f"Unique orientations: {len(orientation_list)}")

                                     
    up_key_to_idx: Dict[Tuple[float, float, float], int] = {}
    num_ups = 0
    for ori in orientation_list:
        up_key = tuple(np.round(ori["up_vec"], up_round_decimals))
        if up_key not in up_key_to_idx:
            up_key_to_idx[up_key] = num_ups
            num_ups += 1
        ori["up_idx"] = up_key_to_idx[up_key]
        ori["up_key"] = up_key

    print(f"Distinct up directions: {num_ups}")

    clusters: Dict[int, List[dict]] = {}
    for ori in orientation_list:
        clusters.setdefault(ori["up_idx"], []).append(ori)

                                        
    orientation_lut: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    yaw_values = set()

    for up_idx, clist in clusters.items():
        canonical = clist[0]
        R_c = canonical["R"]
        up_vec = canonical["up_vec"]

        for ori in clist:
            R = ori["R"]
            R_rel = R @ R_c.T
            axis, angle = rotation_matrix_to_axis_angle(R_rel)

            if np.dot(axis, up_vec) < 0.0:
                axis = -axis
                angle = -angle

            yaw_deg = int(round(math.degrees(angle))) % 360
            ori["yaw_deg"] = yaw_deg
            yaw_values.add(yaw_deg)

            key = (up_idx, yaw_deg)
            if key in orientation_lut:
                continue
                                                  
            orientation_lut[key] = (ori["ry"], ori["rx"], ori["rz"])

    orientation_ups = list(range(num_ups))
    orientation_yaws_deg = sorted(yaw_values)

    print(f"Distinct yaw values: {len(orientation_yaws_deg)}")
    print(f"LUT size: {len(orientation_lut)}")

    return orientation_ups, orientation_yaws_deg, orientation_lut


def save_orientation_tables_npz(
    path: str,
    rx_values: List[int],
    ry_values: List[int],
    rz_values: List[int],
    up_axis: str = "z",
) -> None:
    ups, yaws, lut = enumerate_unique_orientations(
        rx_values=rx_values,
        ry_values=ry_values,
        rz_values=rz_values,
        up_axis=up_axis,
    )

    lut_keys = np.array(list(lut.keys()), dtype=int)                                     
    lut_vals = np.array(list(lut.values()), dtype=int)                              

                                                                               
    perm = _greedy_rotation_order_from_lut(lut_keys, lut_vals)
    lut_keys = lut_keys[perm]
    lut_vals = lut_vals[perm]
    orient_pairs = lut_keys                                                     

    _print_lut_preview(lut_keys, lut_vals, max_rows=10)
    
    np.savez(
        path,
        ups=np.array(ups, dtype=int),
        yaws=np.array(yaws, dtype=int),
        lut_keys=lut_keys,
        lut_vals=lut_vals,
        orient_pairs=orient_pairs,
    )
    print(f"Saved orientation tables to: {path}")


if __name__ == "__main__":
                    
    rx_vals = list(range(0, 360, 30))
    ry_vals = list(range(0, 360, 30))
    rz_vals = list(range(0, 360, 30))

    save_orientation_tables_npz(
        path="orientation_tables_prusa_30deg.npz",
        rx_values=rx_vals,
        ry_values=ry_vals,
        rz_values=rz_vals,
        up_axis="z",
    )
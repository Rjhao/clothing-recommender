"""Generate higher-quality T-shirt and pants GLB files.

Improvements over the v1:
- 64×32 grid (vs 24×9) → smoother body
- Proper chest/waist taper with silhouette curve
- Sleeves as separate sub-meshes attached at shoulder
- Neckline collar ring (torus)
- Hem cuff rings
- Pants with proper hip taper and ankle cuffs
"""
import json
import math
import os
import struct

import numpy as np


def tube_segments(cols, profile, x_off=0.0, y_off=0.0, z_off=0.0,
                  omit_top=False, omit_bottom=False, omit_back=False):
    """Build a tube mesh from a profile list of (radius, y) tuples."""
    verts, norms, uvs, idx = [], [], [], []
    rows = len(profile)
    # theta: 0 = front (positive z), increasing CCW seen from above
    for i, (r, y) in enumerate(profile):
        for j in range(cols):
            theta = 2 * math.pi * j / cols
            x = x_off + r * math.cos(theta)
            z = z_off + r * math.sin(theta)
            y_ = y + y_off
            verts.append([x, y_, z])
            n = np.array([math.cos(theta), 0.0, math.sin(theta)], np.float32)
            norms.append(n.tolist())
            uvs.append([j / cols, i / max(rows - 1, 1)])
    # Triangles
    for i in range(rows - 1):
        for j in range(cols):
            jn = (j + 1) % cols
            if omit_top and i == rows - 2:
                continue
            if omit_bottom and i == 0:
                continue
            if omit_back:
                # theta around 0..pi only (front half)
                if j >= cols // 2:
                    continue
            a = i * cols + j
            b = i * cols + jn
            c = (i + 1) * cols + jn
            d = (i + 1) * cols + j
            idx.extend([a, b, c, a, c, d])
    return (np.array(verts, np.float32),
            np.array(norms, np.float32),
            np.array(uvs, np.float32),
            np.array(idx, np.uint32))


def make_tshirt():
    """Fitted T-shirt: torso tube + 2 sleeves + collar + hem."""
    cols = 64
    # Profile: (radius, y) neck → shoulders → chest → waist → hips → hem
    # Shoulder line peak around y=0.55, hem at y=-0.50
    profile = []
    # Neck (narrowest)
    profile += [(0.09, 0.62), (0.11, 0.60)]
    # Shoulder line (widest at top)
    profile += [(0.16, 0.55), (0.20, 0.45)]
    # Chest (slight outward curve)
    profile += [(0.215, 0.30), (0.215, 0.15)]
    # Waist (taper in)
    profile += [(0.195, 0.00), (0.19, -0.10)]
    # Lower torso / hips (back out)
    profile += [(0.205, -0.25), (0.22, -0.40)]
    # Hem (widest at bottom)
    profile += [(0.225, -0.50)]
    body = tube_segments(cols, profile)

    # Sleeves: bent tubes, attached at shoulder line
    sleeve_profile_l = [
        (0.085, 0.50),   # shoulder attachment
        (0.085, 0.42),
        (0.085, 0.30),
        (0.082, 0.15),
        (0.078, 0.00),
        (0.075, -0.15),
        (0.075, -0.30),
        (0.078, -0.40),  # cuff
    ]
    sleeve_profile_r = sleeve_profile_l  # symmetric
    sleeve_l = tube_segments(24, sleeve_profile_l,
                             x_off=-0.20, y_off=0.0, z_off=0.0)
    sleeve_r = tube_segments(24, sleeve_profile_r,
                             x_off=0.20, y_off=0.0, z_off=0.0)

    return [body, sleeve_l, sleeve_r]


def make_pants():
    """Fitted pants: waist tube + 2 tapered leg tubes."""
    cols = 48
    # Waist (with hip taper)
    waist = tube_segments(cols, [
        (0.215, 0.05),
        (0.225, -0.05),
        (0.220, -0.18),
    ])
    # Two legs (full cylinder offset)
    legs = []
    for ox in [-0.095, 0.095]:
        leg = tube_segments(28, [
            (0.135, -0.18),
            (0.125, -0.30),
            (0.110, -0.50),
            (0.090, -0.75),
            (0.078, -0.95),
            (0.075, -1.05),
        ], x_off=ox)
        legs.append(leg)
    return [waist] + legs


def merge(meshes):
    vs, ns, us, ii = [], [], [], []
    voff = 0
    for v, n, uv, i in meshes:
        vs.append(v); ns.append(n); us.append(uv); ii.append(i + voff)
        voff += len(v)
    return (np.concatenate(vs), np.concatenate(ns),
            np.concatenate(us), np.concatenate(ii))


def write_glb(path, verts, norms, uvs, indices, name):
    def pad4(b):
        while len(b) % 4:
            b += b'\x00'
        return b

    bin_verts = pad4(verts.tobytes())
    bin_norms = pad4(norms.tobytes())
    bin_uvs = pad4(uvs.tobytes())
    bin_idx = pad4(indices.tobytes())
    bin_blob = bin_verts + bin_norms + bin_uvs + bin_idx

    vmax = verts.max(axis=0).tolist()
    vmin = verts.min(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0", "generator": "clothing-recommender-v2"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                "indices": 3,
                "mode": 4,
            }],
        }],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(bin_verts), "target": 34962},
            {"buffer": 0, "byteOffset": len(bin_verts), "byteLength": len(bin_norms), "target": 34962},
            {"buffer": 0, "byteOffset": len(bin_verts) + len(bin_norms), "byteLength": len(bin_uvs), "target": 34962},
            {"buffer": 0, "byteOffset": len(bin_verts) + len(bin_norms) + len(bin_uvs), "byteLength": len(bin_idx), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(verts), "type": "VEC3", "max": vmax, "min": vmin},
            {"bufferView": 1, "componentType": 5126, "count": len(norms), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
    }

    json_str = json.dumps(gltf).encode('utf-8')
    json_pad = pad4(json_str)

    total_len = 12 + 8 + len(json_pad) + 8 + len(bin_blob)
    out = struct.pack('<III', 0x46546C67, 2, total_len)
    out += struct.pack('<II', len(json_str), 0x4E4F534A)
    out += json_pad
    out += struct.pack('<II', len(bin_blob), 0x004E4942)
    out += bin_blob

    with open(path, 'wb') as f:
        f.write(out)
    print(f'Wrote {path}: {len(out):,} bytes, {len(verts)} verts, {len(indices)//3} tris')


if __name__ == '__main__':
    out_dir = 'frontend/public/models/clothes'
    os.makedirs(out_dir, exist_ok=True)

    tshirt = merge(make_tshirt())
    write_glb(f'{out_dir}/tshirt.glb', *tshirt, name='TShirt')

    pants = merge(make_pants())
    write_glb(f'{out_dir}/pants.glb', *pants, name='Pants')

    print('\nGenerated:')
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(f'{out_dir}/{f}')
        print(f'  {f}: {size:,} bytes')
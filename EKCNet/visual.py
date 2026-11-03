
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


COLOR_MAP = {
    0: [255, 255, 255],      # 背景 -> bai
    2: [0, 255, 0],    # WT -> 绿色
    3: [0, 0, 255],    # ET -> 蓝色
    1: [255, 0, 0],  # TC -> 红色

}

def apply_color_map(label_slice, color_map=COLOR_MAP):


    h, w = label_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for val, color in color_map.items():
        mask = (label_slice == val)
        rgb[mask] = color

    slice_data = np.rot90(rgb, k=1)

    rgb = np.flipud(slice_data)


    return rgb

def view_nifti(file_path, slice_idx=None):

    img = nib.load(file_path)
    data = img.get_fdata()


    if data.ndim == 4:
        data = data[:, :, :, 0]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape {data.shape}")

    D,H, W  = data.shape
    print(f"Data shape: {data.shape}")


    if slice_idx is None:
        slice_idx = [70, 90, 98]
    elif isinstance(slice_idx, int):
        slice_idx = [slice_idx, H // 2, W // 2]
    else:
        if len(slice_idx) != 3:
            raise ValueError("slice_idx must be an integer or a list of 3 integers")


    axial = data[slice_idx[0],:, : ]      # 轴向 (横断面) 形状 (H, W)
    coronal = data[:, slice_idx[1], :]    # 冠状 形状 (H, D)
    sagittal = data[ :, :,slice_idx[2]]   # 矢状 形状 (W, D) ？注意原代码写的是 data[slice_idx[2], :, :]，但通常应为 (W, D) 或 (D, W)


    axial_rgb = apply_color_map(axial)
    coronal_rgb = apply_color_map(coronal)
    sagittal_rgb = apply_color_map(sagittal)


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    axes[0].imshow(axial_rgb, origin='lower')
    axes[0].set_title(f'Axial (D={slice_idx[0]}/{D})')
    axes[0].axis('off')

    # 冠状
    axes[1].imshow(coronal_rgb, origin='lower')
    axes[1].set_title(f'Coronal (H={slice_idx[1]}/{H})')
    axes[1].axis('off')

    # 矢状
    axes[2].imshow(sagittal_rgb, origin='lower')
    axes[2].set_title(f'Sagittal (W={slice_idx[2]}/{W})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


    unique_vals = np.unique(data)
    print(f"Unique values in segmentation: {unique_vals}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View NIfTI file (3D) with color mapping')
    parser.add_argument('file', type=str, help='Path to .nii.gz file')
    parser.add_argument('--slice', type=int, nargs='*', default=None,
                        help='Slice indices for three views (e.g., --slice 50 100 80)')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        exit(1)


    if args.slice is None:
        slice_idx = None
    else:
        if len(args.slice) == 1:
            slice_idx = args.slice[0]
        elif len(args.slice) == 3:
            slice_idx = args.slice
        else:
            print("Error: --slice must have 1 or 3 integers.")
            exit(1)

    view_nifti(args.file, slice_idx)


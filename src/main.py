from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from src.utilities.database import Database


def interpolate_spline(wavelengths_in, intensity_in, wavelengths_out):
    if intensity_in.ndim == 1:
        spline = scipy.interpolate.CubicSpline(wavelengths_in, intensity_in)
        intensity_out = spline(wavelengths_out)
    elif intensity_in.ndim == 2:
        intensity_out = np.empty((len(wavelengths_out), intensity_in.shape[1]))
        for ii in range(intensity_out.shape[1]):
            spline = scipy.interpolate.CubicSpline(wavelengths_in, intensity_in[:, ii])
            intensity_out[:, ii] = spline(wavelengths_out)
    else:
        raise ValueError("Dimensions are inconsistent")
    return intensity_out


def load_dataset(
        image_label: str,
        crop_y: tuple[int, int] = None,
        crop_x: tuple[int, int] = None,
        max_dim: int = None,
        wavelengths_limits: tuple[float, float] = (300, 825),
):
    database = Database.create()
    image_id = None
    for acquisition in database.acquisitions:
        if acquisition.dataset_id == image_label:
            image_id = acquisition.id
    if image_id is None:
        return ValueError("Image id not available")

    wavelengths = database.wavelengths(image_id=image_id)
    idx_min = np.argwhere(wavelengths > wavelengths_limits[0])[0][0]
    idx_max = np.argwhere(wavelengths > wavelengths_limits[1])[0][0]

    image = database.image(image_id=image_id)
    with image.open() as image_opened:
        shape = image_opened.shape
        max_dim = max_dim if max_dim is not None else np.maximum(shape[0], shape[1])
        max_dim_x = np.minimum(shape[1], max_dim)
        max_dim_y = np.minimum(shape[0], max_dim)
    cut_x = slice(crop_x[0], crop_x[1]) if crop_x is not None else slice(0, max_dim_x)
    cut_y = slice(crop_y[0], crop_y[1]) if crop_y is not None else slice(0, max_dim_y)
    with image.open() as image_opened:
        numpy_array = image_opened[cut_y, cut_x, :, :][:, :, :, 0]
    numpy_array_cut = numpy_array[:, :, idx_min:idx_max]
    wavelengths_cut = wavelengths[idx_min:idx_max]
    return numpy_array, numpy_array_cut, wavelengths, wavelengths_cut


def xyz_to_rgb(xyz):
    conversion_matrix = np.array(
        [
            [+3.2406255, -1.5372080, -0.4986286],
            [-0.96899307, +1.8757561, +0.0415175],
            [+0.0557101, -0.2040211, +1.0569959],
        ]
    )
    out = xyz @ conversion_matrix.T
    out = np.clip(out, 0, 1)
    return out


def normalize_image(image):
    min_image = np.min(image)
    max_image = np.max(image)
    return (image - min_image) / (max_image - min_image)


def gamma_correction(rgb):
    out = np.empty_like(rgb)
    mask = rgb <= 0.0031308
    out[mask] = 12.92 * rgb[mask]
    out[~mask] = 1.055 * rgb[~mask] ** 0.416 - 0.055
    return out


def save_arrays(rgb, numpy_array, wavelengths, output_path: Path, label: str) -> None:
    save_path = output_path / label
    save_path.mkdir(exist_ok=False)
    np.save(f"{save_path / 'rgb.npy'}", rgb)
    np.save(f"{save_path / 'hsi.npy'}", numpy_array)
    np.save(f"{save_path / 'wavelengths.npy'}", wavelengths)


def generate_rgb_array(
    image_label: str,
    crop_x: tuple[int, int] = None,
    crop_y: tuple[int, int] = None,
):

    cmf = "./data/illuminants/ciexyzjv.csv"
    illuminant = "./data/illuminants/Illuminantd65.csv"

    project_path = Path(__file__).resolve().parents[1]
    numpy_array, numpy_array_cut, wavelengths, wavelengths_cut = load_dataset(
        image_label=image_label, crop_x=crop_x, crop_y=crop_y,
    )

    df_cmf = pd.read_csv(project_path / cmf)
    cmf_wavelengths = np.array(df_cmf.iloc[:, 0])
    cmf_xyz = np.array(df_cmf.iloc[:, 1:])
    cmf_xyz_interp = interpolate_spline(
        wavelengths_in=cmf_wavelengths,
        wavelengths_out=wavelengths_cut,
        intensity_in=cmf_xyz,
    )

    df_ill = pd.read_csv(project_path / illuminant)
    ill_wavelengths = np.array(df_ill.iloc[:, 0])
    ill_intensity = np.array(df_ill.iloc[:, 1])
    ill_intensity_interp = interpolate_spline(
        wavelengths_in=ill_wavelengths,
        wavelengths_out=wavelengths_cut,
        intensity_in=ill_intensity,
    )

    reflectances = normalize_image(numpy_array_cut)
    radiances = reflectances * ill_intensity_interp[np.newaxis, np.newaxis, :]
    scale_factor = np.sum(ill_intensity_interp * cmf_xyz_interp[:, 1])
    radiances = radiances / scale_factor

    xyz = radiances @ cmf_xyz_interp
    xyz = normalize_image(xyz)
    rgb = xyz_to_rgb(xyz)
    rgb_corrected = gamma_correction(rgb)

    save_arrays(
        rgb=rgb_corrected,
        numpy_array=numpy_array,
        wavelengths=wavelengths,
        output_path=project_path / "data/outputs",
        label=image_label,
    )

    fig, ax = plt.subplots()
    ax.imshow(gamma_correction(rgb))
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate a curve to new samples and save to a folder")
    parser.add_argument("--i", type=str, nargs="+", help="Image identifier", default="pavia")
    parser.add_argument("--x", type=tuple[int, int], nargs="+", help="Cropping interval in the horizontal direction", default=None)
    parser.add_argument("--y", type=tuple[int, int], nargs="+", help="Cropping interval in the vertical direction", default=None)
    args = parser.parse_args()

    generate_rgb_array(image_id=args.i, crop_x=args.x, crop_y=args.y)


if __name__ == "__main__":
    main()

import numpy as np

from .warp_image import warp_images
from itertools import product

def _get_regular_grid_new_ours(image, points_per_dim):
    nrows, ncols = image.shape[0], image.shape[1]
    res = []
    if points_per_dim == 1:
        indexes = [nrows//2, nrows//2]
        res.append(indexes)
    elif points_per_dim == 2:
        indexes = [(nrows//4)*1, (nrows//4)*3]
        for x in product(indexes, repeat=2):
            res.append(x)
    elif points_per_dim == 3:
        indexes = [(nrows//6)*1, (nrows//6)*3, (nrows//6)*5]
        for x in product(indexes, repeat=2):
            res.append(x)
    elif points_per_dim == 4:
        indexes = [(nrows//8) * 1, (nrows//8) * 3, (nrows//8) * 5, (nrows//8) * 7]
        for x in product(indexes, repeat=2):
            res.append(x)
    elif points_per_dim == 5:
        indexes = [(nrows // 10) * 1, (nrows // 10) * 3, (nrows // 10) * 5, (nrows // 10) * 7, (nrows // 10) * 9]
        for x in product(indexes, repeat=2):
            res.append(x)

    return np.array(res)

def _get_regular_grid(image, points_per_dim):
    nrows, ncols = image.shape[0], image.shape[1]
    rows = np.linspace(0, nrows, points_per_dim)
    cols = np.linspace(0, ncols, points_per_dim)
    rows, cols = np.meshgrid(rows, cols)
    return np.dstack([cols.flat, rows.flat])[0]


def _generate_random_vectors(image, src_points, scale):
    dst_pts = src_points + np.random.uniform(-scale, scale, src_points.shape)
    return dst_pts


def _thin_plate_spline_warp(image, src_points, dst_points, keep_corners=True):
    width, height = image.shape[:2]
    if keep_corners:
        corner_points = np.array(
            [[0, 0], [0, width], [height, 0], [height, width]])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))
    out = warp_images(src_points, dst_points,
                      np.moveaxis(image, 2, 0),
                      (0, 0, width - 1, height - 1))
    return np.moveaxis(np.array(out), 0, 2)


def tps_warp(image, points_per_dim, scale):
    width, height = image.shape[:2]
    src = _get_regular_grid(image, points_per_dim=points_per_dim)
    dst = _generate_random_vectors(image, src, scale=scale*width)
    out = _thin_plate_spline_warp(image, src, dst)
    return out

def tps_warp_2(image, dst, src):
    out = _thin_plate_spline_warp(image, src, dst)
    return out
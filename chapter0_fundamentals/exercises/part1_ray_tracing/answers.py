import sys
from pathlib import Path

import torch as t
import typeguard
from einops import rearrange, repeat, reduce
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from matplotlib.pyplot import imshow
from torch import Tensor
import plotly.express as px

import tests
from utils import render_lines_with_plotly

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part1_ray_tracing', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (render_lines_with_plotly,
                                     setup_widget_fig_ray,
                                     setup_widget_fig_triangle)


def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    out = t.zeros((num_pixels, 2, 3))
    out[:,1,0] = 1
    t.linspace(-y_limit, y_limit, num_pixels, out=out[:,1,1])
    return out
rays1d = make_rays_1d(9, 10.0)

# fig = render_lines_with_plotly(rays1d)


def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''

    A = t.stack((ray[1,:2], segment[0,:2] - segment[1,:2]), 1)
    try:
        sol = t.linalg.solve(A, segment[0,:2])
    except t._C._LinAlgError:
        return False
    return sol[0] > 0 and sol[1] >= 0 and sol[1] <= 1


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)


def intersect_rays_1d(rays: Float[t.Tensor, "nrays 2 3"], segments: Float[t.Tensor, "nsegments 2 3"]) -> Bool[t.Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    nrays = rays.shape[0]
    nsegments = segments.shape[0]

    As = t.stack((repeat(rays[:,1,:2], "nrays n -> (nsegments nrays) n", nsegments=nsegments), repeat(segments[:,0,:2] - segments[:,1,:2], "nsegments n -> (nsegments nrays) n", nrays = nrays)), 2)

    dets = t.linalg.det(As)
    mask = t.isclose(dets, t.zeros_like(dets))
    As[mask,...] = t.eye(2)

    try:
        sols = t.linalg.solve(As, repeat(segments[:,0,:2], "nsegments n -> (nsegments nrays) n", nrays = nrays))
    except t._C._LinAlgError:
        return False
    sols = rearrange(sols, "(nsegments nrays) n -> nsegments nrays n", nsegments=nsegments, nrays=nrays)
    mask_rearranged = rearrange(mask, "(nsegments nrays) -> nsegments nrays", nsegments=nsegments, nrays=nrays)

    return t.any(((sols[...,0] > 0) & (sols[...,1] >= 0) & (sols[...,1] <= 1) & ~mask_rearranged), dim=0) 

tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    num_rays = num_pixels_y * num_pixels_z
    out = t.zeros((num_rays, 2, 3))
    out[:,1,0] = 1
    out[:,1,1] = repeat(t.linspace(-y_limit, y_limit, num_pixels_y), "num_pixels_y -> (num_pixels_y num_pixels_z) ", num_pixels_z=num_pixels_z)
    out[:,1,2] = repeat(t.linspace(-y_limit, y_limit, num_pixels_z), "num_pixels_z  -> (num_pixels_y num_pixels_z) ", num_pixels_y=num_pixels_y)
    return out



rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
# render_lines_with_plotly(rays_2d)


Point = Float[Tensor, "points=3"]

@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''

    M = t.stack((-D, B - A, C - A), -1)
    b_vec = O - A

    dets = t.linalg.det(M)
    mask = t.isclose(dets, t.zeros_like(dets))
    M[mask,...] = t.eye(3)

    sol = t.linalg.solve(M, b_vec)

    s, u, v = sol
    ret = ((s >= 0) & (u + v <= 1) & (u >= 0) & (v >= 0) & (~mask)).item()
    return ret


tests.test_triangle_ray_intersects(triangle_ray_intersects)



def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    A, B, C = repeat(triangle, "trianglePoints dims -> nrays trianglePoints dims", nrays = rays.shape[0]).unbind(1)
    O, D = rays.unbind(1)
    M = t.stack((-D, B - A, C - A), -1)
    b_vec = O - A

    dets = t.linalg.det(M)
    mask = t.isclose(dets, t.zeros_like(dets))
    M[mask,...] = t.eye(3)

    sol = t.linalg.solve(M, b_vec)

    s, u, v = sol.unbind(1)
    ret = (s >= 0) & (u + v <= 1) & (u >= 0) & (v >= 0) & (~mask)
    return ret


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 40
y_limit = z_limit = 0.5

# # Plot triangle & rays
# test_triangle = t.stack([A, B, C], dim=0)
# rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
# triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
# render_lines_with_plotly(rays2d, triangle_lines)

# # Calculate and display intersections
# intersects = raytrace_triangle(rays2d, test_triangle)
# img = intersects.reshape(num_pixels_y, num_pixels_z).int()
# imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")


with open(section_dir / "pikachu.pt", "rb") as f:
    triangles = t.load(f)



def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    ntriangles = triangles.shape[0]
    nrays = rays.shape[0]
    A, B, C = repeat(triangles, "ntriangles trianglePoints dims -> (nrays ntriangles) trianglePoints dims", nrays = nrays).unbind(1)
    O, D = repeat(rays, "nrays rayPoints dims -> (nrays ntriangles) rayPoints dims", ntriangles=ntriangles).unbind(1)
    M = t.stack((-D, B - A, C - A), -1)
    b_vec = O - A


    dets = t.linalg.det(M)
    mask = t.isclose(dets, t.zeros_like(dets))
    M[mask,...] = t.eye(3)

    sol = rearrange(t.linalg.solve(M, b_vec), "(nrays ntriangles) ndims -> nrays ntriangles ndims", ntriangles=ntriangles, nrays=nrays)
    
    m = rearrange(mask, "(nrays ntriangles) -> nrays ntriangles", ntriangles=ntriangles, nrays=nrays)
    s, u, v = sol.unbind(2)
    intersects = t.any((s >= 0) & (u + v <= 1) & (u >= 0) & (v >= 0) & (~m), dim=1)

    ret = reduce(s, "nrays ntriangles -> nrays", "min")
    ret[~intersects] = float("inf")

    # ret[intersects] = t.min(s, dim=0)[intersects]
    return ret


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]): 
    fig.layout.annotations[i]['text'] = text
fig.show()


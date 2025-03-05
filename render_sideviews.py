# Basic imports
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj
from pathlib import Path
import cv2
import sam # sam.py
import time
from tqdm import tqdm

# Library imports
from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.meshes import TexturedPhotogrammetryMesh
# from geograypher.meshes.derived_meshes import TexturedPhotogrammetryMeshChunked
from skimage.transform import resize


## Path constants
# Where all the data is stored
PROJECT_FOLDER = Path("/headless/persistent/timm/hawaii_automate_metashape/multiview-labeling/example_hawaii").resolve()
# Where to save vis data
VIS_FOLDER = Path("/headless/geograypher-vis").resolve()
# Where to cache results
CACHE_FOLDER = Path("/headless/geograypher-cache").resolve()

NULL_TEXTURE_INT_VALUE = 255
# Name of the column to use in the example data # Render data from this column in the geofile to each image
LABEL_COLUMN_NAME = "species_observed"

## Parameters to control the outputs
# The image is downsampled to this fraction for accelerated rendering
RENDER_IMAGE_SCALE = 0.1
# Portions of the mesh within this distance of the labels are used for rendering
MESH_BUFFER_RADIUS_METER = 5
# Cameras within this radius of the annotations are used for sideviews
CAMERAS_BUFFER_RADIUS_METERS = 50
# Downsample target
DOWNSAMPLE_TARGET = 0.25

## Define the inputs
# The automate-metashape run name and timestamp
RUN_NAME = "run-001_20240930T1143"
# The mesh exported from Metashape
MESH_FILE = Path(PROJECT_FOLDER, "exports", f"{RUN_NAME}_model_local.ply")
# The camera file exported from Metashape
CAMERAS_FILE = Path(PROJECT_FOLDER, "exports", f"{RUN_NAME}_cameras.xml")
# The image folder used to create the Metashape project
IMAGE_FOLDER = Path(PROJECT_FOLDER, "photos")

## Define the intermediate results
# # Processed geo file
# LABELED_MESH_FILE = Path(PROJECT_FOLDER, "intermediate","labeled_mesh.ply")
# Where to save the rendering label images
# RENDERED_LABELS_FOLDER = Path(PROJECT_FOLDER, "intermediate","rendered_labels",)

# Create camera set
CAMERA_SET = MetashapeCameraSet(CAMERAS_FILE, IMAGE_FOLDER)


def ortho_mask(cropped_ortho, bbox, bottomleft, geo_transform):
    """
    cropped_ortho (array): portion of orthomosaic cropped to only bounding box of interest (and extra 20 pixel allowance on each side)
    bbox (tuple): (x0,y0,x1,y1) geographic coordinates of bounding box
    bottomleft (tuple): (x0,y0) image coordinates of bottom left of bounding box
    geo_transform (object): self.ortho_raster.transform object
   
    segments the given orthomosaic to mask out the tree

    returns (
        largest_contour (object): cv2.contour object of tree,
        mask_roi (geodataframe): gdp object with 'geometry' column containing the mask polygon,
    )
    """
    start_time = time.time()

    ### segment to find region of interest
    ortho_mask = sam.mask(cropped_ortho)
    ortho_mask = ortho_mask[20:-20,20:-20,0]
    # morphological processing to fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    ortho_mask = cv2.morphologyEx(ortho_mask, cv2.MORPH_CLOSE, kernel)
    ortho_mask = cv2.morphologyEx(ortho_mask, cv2.MORPH_OPEN, kernel)
    # find contours
    contours, hierarchy = cv2.findContours(ortho_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001*cv2.arcLength(largest_contour,True)
    largest_contour = cv2.approxPolyDP(largest_contour,epsilon,True)
    ortho_contour = cv2.drawContours(cropped_ortho.copy(), [largest_contour], -1, (0,255,0), 3)
    # find og img coords (offset by bottom left) and convert to geo
    x0,y0 = bottomleft
    geo_polygon = []
    for i,[[x,y]] in enumerate(largest_contour):
        geo_polygon.append(geo_transform*(x+x0,y+y0))
    # create new geo mask
    mask_roi = gpd.GeoDataFrame({
        'observed_tree_id': [0,],
        'species_observed': ['unknown',],
        'geometry': [Polygon(geo_polygon),],
    }, crs='EPSG:4326', geometry='geometry')

    # # bbox roi
    # long0,lat0,long1,lat1 = bbox
    # bbox_roi = gpd.GeoDataFrame({
    #     'observed_tree_id': [0,],
    #     'species_observed': ['unknown',],
    #     'geometry': [Polygon(((long0,lat0),(long1,lat0),(long1,lat1),(long0,lat1))),],
    # }, crs='EPSG:4326', geometry='geometry')

    end_time = time.time()
    print('masking took', end_time-start_time, 'seconds')
    return largest_contour,mask_roi


def sideviews(mask_roi):
    """
    mask_roi (geodataframe): gpd object with 'geometry' column containing the mask polygon

    generates sideviews given the segmented topview mask

    returns an array of sideviews (images)
    """
    start_time = time.time()

    mesh = TexturedPhotogrammetryMesh(
        MESH_FILE,
        downsample_target=DOWNSAMPLE_TARGET,
        texture=mask_roi,
        ROI=mask_roi,
        ROI_buffer_meters=MESH_BUFFER_RADIUS_METER,
        texture_column_name=LABEL_COLUMN_NAME,
        transform_filename=CAMERAS_FILE,
    )
    print(f'{time.time()-start_time} seconds to generate mesh') # usually around 1 second

    # Extract cameras near the training data
    camera_subset = CAMERA_SET.get_subset_ROI(
        ROI=mask_roi, buffer_radius=CAMERAS_BUFFER_RADIUS_METERS
    )

    # Generate sideview masks
    render_gen = mesh.render_flat(
        cameras=camera_subset,
        batch_size=1,
        render_img_scale=RENDER_IMAGE_SCALE,
    )

    # loop through masks (from render_gen) to output a usable image
    out = []
    for i,rendered in enumerate(tqdm(render_gen, total=len(camera_subset), desc="Computing renders",)):
        if i%2==0:
            continue
        native_size = camera_subset[i].get_image_size()
        rendered = resize(rendered,native_size,order=(0 if mesh.is_discrete_texture() else 1),)
        if rendered.ndim == 3:
            rendered = rendered[..., :3]
        invalid = np.logical_or.reduce([rendered < 0, rendered > 255, ~np.isfinite(rendered)])
        rendered[invalid] = NULL_TEXTURE_INT_VALUE
        rendered = rendered.astype(np.uint8)
        mask = np.all(rendered != NULL_TEXTURE_INT_VALUE, axis=-1)
        if np.any(mask):
            # match mask to original images and generate cropped and greyed out images
            filename = camera_subset.get_image_filename(i, absolute=False)
            y_indices, x_indices = np.where(mask)
            y_min, y_max, x_min, x_max = y_indices.min(), y_indices.max(), x_indices.min(), x_indices.max()
            mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]
            rgb_cropped = cv2.imread(str(Path(IMAGE_FOLDER, filename)))[y_min:y_max+1, x_min:x_max+1]
            # cv2.imwrite('debug/rgb_cropped.png',rgb_cropped)
            # rgb_mask = sam.mask(rgb_cropped)
            # cv2.imwrite('debug/rgb_mask.png',rgb_mask)
            # rgb_masked = 0.5*rgb_cropped + 0.5*rgb_mask/255*rgb_cropped
            mask_3d = np.stack([mask_cropped] * 3, axis=-1)
            rgb_masked = np.where(mask_3d, rgb_cropped, rgb_cropped // 2)

            # add contours from the mask to the image
            contours, _ = cv2.findContours(mask_cropped.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.001 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            cv2.drawContours(rgb_masked, [largest_contour], -1, (0, 0, 255), 3)
            out.append(rgb_masked)

            # # limit to 10
            # if i>=10:
            #     break
    
    end_time = time.time()
    print(f'rendering {i} sideviews took {end_time-start_time} seconds')
    return out


# ## DEBUG -- dirt road
# import rasterio
# x0,y0,x1,y1 = 5816,6299,6111,6508
# geo_bbox = (-155.28370436698305, 19.4468668556957, -155.28364268130306, 19.446825399082698)
# ortho_raster = rasterio.open('example_hawaii/exports/orthomosaic.tif')
# raw_ortho = ortho_raster.read().transpose(1, 2, 0)[:, :, :3][:, :, [2, 1, 0]]
# img_contour,mask_roi = ortho_mask(raw_ortho[y0-20:y1+20,x0-20:x1+20,:], geo_bbox, (x0,y0), ortho_raster.transform)
# multiviews = sideviews(mask_roi)
# cv2.imwrite('debug/sideview_0.png',multiviews[0])
# cv2.imwrite('debug/sideview_1.png',multiviews[1])
# cv2.imwrite('debug/sideview_2.png',multiviews[2])
# cv2.imwrite('debug/sideview_3.png',multiviews[3])
# cv2.imwrite('debug/sideview_4.png',multiviews[4])
# cv2.imwrite('debug/sideview_5.png',multiviews[5])
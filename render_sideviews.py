# Basic imports
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import shutil
import os
import cv2
from shapely.geometry import Polygon
import sam
import time
import tempfile as tmp

# Library imports
from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.meshes.derived_meshes import TexturedPhotogrammetryMeshChunked
from geograypher.utils.visualization import show_segmentation_labels

import pyproj

## Path constants
# Where all the data is stored
PROJECT_FOLDER = Path("/headless/persistent/timm/hawaii_automate_metashape").resolve()
# Where to save vis data
VIS_FOLDER = Path("/headless/geograypher-vis").resolve()
# Where to cache results
CACHE_FOLDER = Path("/headless/geograypher-cache").resolve()

VERT_ID = "vert_ID"
CLASS_ID_KEY = "class_ID"
INSTANCE_ID_KEY = "instance_ID"
PRED_CLASS_ID_KEY = "pred_class_ID"
CLASS_NAMES_KEY = "class_names"
RATIO_3D_2D_KEY = "ratio_3d_2d"
NULL_TEXTURE_INT_VALUE = 255
LAT_LON_CRS = pyproj.CRS.from_epsg(4326)
EARTH_CENTERED_EARTH_FIXED_CRS = pyproj.CRS.from_epsg(4978)

# Name of the column to use in the example data # Render data from this column in the geofile to each image
LABEL_COLUMN_NAME = "species_observed"
# The mapping between integer class IDs and string class names
IDS_TO_LABELS = {
    0: "unknown",
    1: "A",
    2: "B",
}

## set constants

## Parameters to control the outputs
# Repeat the labeling process
RETEXTURE = True
# Points less than this height above the DTM are considered ground
# Something is off about the elevation between the mesh and the DTM, this should be a threshold in meters above ground
HEIGHT_ABOVE_GROUND_THRESH = -float("inf")
# The image is downsampled to this fraction for accelerated rendering
RENDER_IMAGE_SCALE = 0.2 # 0.1 shows no views?
# Portions of the mesh within this distance of the labels are used for rendering
MESH_BUFFER_RADIUS_METER = 5
# Cameras within this radius of the annotations are used for training
CAMERAS_BUFFER_RADIUS_METERS = 50
# Downsample target
DOWNSAMPLE_TARGET = 0.25

## Define the inputs
# The automate-metashape run name and timestamp
RUN_NAME = "run-001_20240930T1143"
# The file containing geospatial labels # The input labels
LABELS_FILE = Path(PROJECT_FOLDER, "labels.gpkg")
# The mesh exported from Metashape
MESH_FILE = Path(PROJECT_FOLDER, "exports", f"{RUN_NAME}_model_local.ply")
# The camera file exported from Metashape
CAMERAS_FILE = Path(PROJECT_FOLDER, "exports", f"{RUN_NAME}_cameras.xml")
# The digital elevation map exported by Metashape
DTM_FILE = Path(PROJECT_FOLDER, "exports", f"{RUN_NAME}_dsm-mesh.tif")
# The image folder used to create the Metashape project
IMAGE_FOLDER = Path(PROJECT_FOLDER, "photos")

## Outputs
PREDICTED_VECTOR_LABELS_FILE = Path(PROJECT_FOLDER, "outputs", "predicted_labels.geojson")

EXAMPLE_INTRINSICS = {
    "f": 1000,
    "cx": 0,
    "cy": 0,
    "image_width": 800,
    "image_height": 600,
    "distortion_params": {},
}

def ortho_mask(cropped_ortho, bbox, bottomleft, geo_transform):
    start_time = time.time()

    ### segment to find region of interest
    ortho_mask = sam.mask(cropped_ortho)
    ortho_mask = ortho_mask[20:-20,20:-20,0]
    cv2.imwrite('test_mask_before.png',ortho_mask)
    # morphological processing to fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    ortho_mask = cv2.morphologyEx(ortho_mask, cv2.MORPH_CLOSE, kernel)
    ortho_mask = cv2.morphologyEx(ortho_mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('test_mask_after.png',ortho_mask)
    # find contours
    contours, hierarchy = cv2.findContours(ortho_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001*cv2.arcLength(largest_contour,True)
    largest_contour = cv2.approxPolyDP(largest_contour,epsilon,True)
    # largest_contour = cv2.approxPolyDP(largest_contour,32,True)
    ###
    # print(largest_contour.dtype)
    # print(largest_contour[0].dtype)
    # print(largest_contour[0][0].dtype)
    # return
    ###
    ortho_contour = cv2.drawContours(cropped_ortho.copy(), [largest_contour], -1, (0,255,0), 3)
    cv2.imwrite('test_ortho_contours.png',ortho_contour)
    # find og img coords (offset by top left) and convert to geo
    x0,y0 = bottomleft
    geo_polygon = []
    for i,[[x,y]] in enumerate(largest_contour):
        geo_transform*(x+x0,y+y0)
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
    start_time = time.time()
    intermediate_folder = tmp.TemporaryDirectory().name
    rendered_labels_folder = Path(intermediate_folder,"rendered_labels",)
    labeled_mesh_file = Path(intermediate_folder,"labeled_mesh.ply")
    cache_folder = Path(intermediate_folder,"cache",)

    # mesh
    MESH_BUFFER_RADIUS_METER = 5
    # Create a labeled version of the mesh from the field data
    # if not present or requested
    if not Path(labeled_mesh_file).is_file() or RETEXTURE:
        # Load the downsampled mesh and apply the texture from the vector file
        mesh = TexturedPhotogrammetryMesh(
            MESH_FILE,
            downsample_target=DOWNSAMPLE_TARGET,
            texture=mask_roi,
            ROI=mask_roi,
            ROI_buffer_meters=MESH_BUFFER_RADIUS_METER,
            IDs_to_labels=IDS_TO_LABELS,
            texture_column_name=LABEL_COLUMN_NAME,
            transform_filename=CAMERAS_FILE,
        )
        # mesh = TexturedPhotogrammetryMeshChunked(
        #     MESH_FILE,
        #     downsample_target=DOWNSAMPLE_TARGET,
        #     texture=mask_roi,
        #     ROI=mask_roi,
        #     ROI_buffer_meters=MESH_BUFFER_RADIUS_METER,
        #     IDs_to_labels=IDS_TO_LABELS,
        #     texture_column_name=LABEL_COLUMN_NAME,
        #     transform_filename=CAMERAS_FILE,
        # )
        # Get the vertex textures from the mesh
        texture_verts = mesh.get_texture(
            request_vertex_texture=True, try_verts_faces_conversion=False
        )
        mesh.label_ground_class(
            DTM_file=DTM_FILE,
            height_above_ground_threshold=HEIGHT_ABOVE_GROUND_THRESH,
            only_label_existing_labels=True,
            ground_class_name="entire",
            ground_ID=np.nan, # This means that no ground label will be included
            set_mesh_texture=True,
        )

        mesh.save_mesh(labeled_mesh_file, save_vert_texture=True)
    else:
        mesh = TexturedPhotogrammetryMesh(
            labeled_mesh_file, transform_filename=CAMERAS_FILE
        )
        # mesh = TexturedPhotogrammetryMeshChunked(
        #     labeled_mesh_file, transform_filename=CAMERAS_FILE
        # )

    ## cameras set
    # Create camera set
    camera_set = MetashapeCameraSet(CAMERAS_FILE, IMAGE_FOLDER)
    # Extract cameras near the training data
    training_camera_set = camera_set.get_subset_ROI(
        ROI=mask_roi, buffer_radius=CAMERAS_BUFFER_RADIUS_METERS
    )
    # # Show the camera set
    # training_camera_set.vis(force_xvfb=True, frustum_scale=0.5)

    # ## visualize mesh
    # mesh.vis(camera_set=training_camera_set, force_xvfb=True)

    ## save renders
    # np._set_promotion_state("legacy")  # required to prevent overflow errors
    # mesh.save_renders(
    #     camera_set=training_camera_set,
    #     render_image_scale=RENDER_IMAGE_SCALE,
    #     save_native_resolution=False, # upsamples mask to og
    #     output_folder=rendered_labels_folder,
    # )
    # TODO put in memory
    ## render_flat
    masks = mesh.render_flat(
        cameras=training_camera_set,
        batch_size=4,
        render_img_scale=0.25,
        save_to_cache=True,
        cache_folder=cache_folder # will create if not
        )

    # i=0
    # multiviews=[]
    # for file in os.listdir(Path(rendered_labels_folder,'100MEDIA')):
    #     mask = 255-cv2.imread(Path(rendered_labels_folder,'100MEDIA',file))
    #     if mask.max()!=0: # the object of interest actually exists in the photo
    #         print(file)
    #         y_indices, x_indices, _ = np.nonzero(mask)
    #         mask_cropped = mask[min(y_indices):max(y_indices),min(x_indices):max(x_indices),:]
    #         rgb_cropped = cv2.imread(Path(IMAGE_FOLDER,'100MEDIA',file[:-3]+'JPG'))[min(y_indices):max(y_indices),min(x_indices):max(x_indices),:]
    #         rgb_masked = 0.5*mask_cropped/255*rgb_cropped + 0.5*rgb_cropped
    #         contours, hierarchy = cv2.findContours(mask_cropped[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         largest_contour = max(contours, key=cv2.contourArea)
    #         epsilon = 0.001*cv2.arcLength(largest_contour,True)
    #         largest_contour = cv2.approxPolyDP(largest_contour,epsilon,True)
    #         masked_contour = cv2.drawContours(rgb_masked, [largest_contour], -1, (0,0,255), 3)
    #         multiviews.append(masked_contour)
    #         i+=1
            # if i>=10:
            #     break

    print(i, 'views')
    end_time = time.time()
    print('rendering sideviews took', end_time-start_time, 'seconds')
    if len(multiviews) < 2:
        pass
    return multiviews
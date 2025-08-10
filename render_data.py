# This script is run as a blender script attatched to demo_scene.blend and is coordinate-specific to the scene

# Generates aliased and supersample resolution frames as training data

import bpy, os
import mathutils, random

####################
BASE_W = 960
BASE_H = 540
ALIAS_DIR = "//renders/540/alias/"
SSAA_DIR  = "//renders/540/ssaa16x/"
N_FRAMES = 1024
SKIP_EXISTING = False
####################

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = BASE_W
scene.render.resolution_y = BASE_H
scene.render.image_settings.file_format = 'PNG'
scene.cycles.samples = 256

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

ensure_dir(bpy.path.abspath(ALIAS_DIR))
ensure_dir(bpy.path.abspath(SSAA_DIR))

def get_xy_bounds():
    return (-1.7, 1.7, -1.6, 1.6), mathutils.Vector((0, 0, 0))

bounds, target = get_xy_bounds()

comp_prefix = os.path.splitext(os.path.basename(bpy.data.filepath))[0] or "scene"
cam = scene.camera

for i in range(N_FRAMES):
    cam_x = random.uniform(bounds[0]- 0.3, bounds[1] + 0.3)
    cam_y = random.uniform(bounds[2] - 0.3, bounds[3] + 0.3)
    cam_z = random.uniform(1.0, 1.5)
    cam.location = mathutils.Vector((cam_x, cam_y, cam_z))

    target_x = random.uniform(bounds[0], bounds[1])
    target_y = random.uniform(bounds[2], bounds[3])
    target_z = random.uniform(0.65, 0.9)
    look_at = mathutils.Vector((target_x, target_y, target_z))

    direction = look_at - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    print(f"[Frame {i}] Cam @ {cam.location} → {look_at}")

    # ALIASED
    alias_path = bpy.path.abspath(ALIAS_DIR + f"{comp_prefix}_frame{i:04d}_alias.png")
    if not SKIP_EXISTING or not os.path.exists(alias_path):
        scene.render.resolution_percentage = 100
        scene.render.filepath = alias_path
        bpy.ops.render.render(write_still=True)

    # SS
    ssaa_path = bpy.path.abspath(SSAA_DIR + f"{comp_prefix}_frame{i:04d}_ssaa16x.png")
    if not SKIP_EXISTING or not os.path.exists(ssaa_path):
        scene.render.resolution_percentage = 400
        scene.render.filepath = ssaa_path
        bpy.ops.render.render(write_still=True)

scene.render.resolution_percentage = 100

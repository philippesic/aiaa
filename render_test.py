
# This script is run as a blender script attatched to demo_scene.blend and is coordinate-specific to the scene

# Generates an aliased frame to test the model

import os
import random
import sys

def Test():
    import bpy, mathutils

    args = sys.argv
    if '--' in args:
        idx = args.index('--')
    args = args[idx + 1:] 
    
    BASE_W = int(args[0])
    BASE_H = int(args[1])
    OUTPUT_DIR = f"//output/{BASE_H}/test/"
    N_FRAMES = 1
    SKIP_EXISTING = False

    BASE_W = int(BASE_W)
    BASE_H = int(BASE_H)
    N_FRAMES = int(N_FRAMES)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = BASE_W
    scene.render.resolution_y = BASE_H
    scene.render.image_settings.file_format = 'PNG'
    scene.cycles.samples = 256

    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    ensure_dir(bpy.path.abspath(OUTPUT_DIR))

    def get_xy_bounds():
        return (-1.7, 1.7, -1.6, 1.6), mathutils.Vector((0, 0, 0))

    bounds, target = get_xy_bounds()

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
        alias_path = bpy.path.abspath(OUTPUT_DIR + f"test.png")
        if not SKIP_EXISTING or not os.path.exists(alias_path):
            scene.render.resolution_percentage = 100
            scene.render.filepath = alias_path
            bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    Test()
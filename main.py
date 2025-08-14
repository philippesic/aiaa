import inference, model, downscale
from inference import Infer
from model import train
from downscale import Downscale
import render_data, render_test
import subprocess, os, re, sys

def main():
    SELECTION = int(input("Select Operation:\n 1: Generate Training Data\n 2: Generate Test Image\n 3: Train Model\n 4: Run Inference\n"))
    BASE_W = str(input("Target Resolution Width:\n"))
    BASE_H = str(input("Target Resolution Height:\n"))

    match SELECTION:
        case 1:
            N_FRAMES = input("Number of training Images:\n")
            SS_TARGET = input("Supersampled Resolution (EG 4 for 4x resolution):\n")

            ver = re.search(r'(\d+\.\d+)', "Blender 4.5")
            if ver:
                version_number = ver.group(1)

            blender_path = f"C:/Program Files/Blender Foundation/Blender {version_number}/blender.exe"
            script_path = "./render_data.py"

            if os.path.exists(blender_path):
                print("Blender Installation Found")
            else:
                root = Tk()
                root.withdraw()

                file_path = filedialog.askopenfilename(
                title="Select a Blender Installation",
                filetypes=[("Executable", "*.exe")]
                )

            command = [
                blender_path,
                "./demo_scene.blend",
                "--background",
                "--python", script_path,
                "--",
                str(BASE_W),
                str(BASE_H),
                str(N_FRAMES),
                str(SS_TARGET),
            ]

            subprocess.run(command)

            downscale.Downscale(BASE_W, BASE_H, SS_TARGET)
        
        case 2:
            ver = re.search(r'(\d+\.\d+)', "Blender 4.5")
            if ver:
                version_number = ver.group(1)

            blender_path = f"C:/Program Files/Blender Foundation/Blender {version_number}/blender.exe"
            script_path = "./render_test.py"


            if os.path.exists(blender_path):
                print("Blender Installation Found")
            else:
                root = Tk()
                root.withdraw()

                file_path = filedialog.askopenfilename(
                title="Select a Blender Installation",
                filetypes=[("Executable", "*.exe")]
                )

            command = [
                blender_path,
                "./demo_scene.blend",
                "--background",
                "--python", script_path,
                "--",
                str(BASE_W),
                str(BASE_H)
            ]

            subprocess.run(command)

        case 3:
            CONV = input("Number of Convolution Layers (Default 6):\n")
            EPOCH = input("Number of Epochs to Train (Default 20):\n")
            CHANNEL = input("Number of Channels (Default 64):\n")
            train(BASE_W, BASE_H, CONV, EPOCH, CHANNEL)
        
        case 4:
            Infer(BASE_W, BASE_H)


if __name__ == "__main__":
    main()


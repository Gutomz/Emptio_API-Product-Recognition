import os
import re
import shutil
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--root-path', help='Full Input Path')
parser.add_argument('-s', '--save-path', help='Full Save Path')
parser.add_argument('-w', '--new-width', help='New image width')
args = parser.parse_args()

# The new width for images in the ROOT_PATH folder.
NEW_WIDTH = args.new_width or 32
# Program running path
RUN_PATH = os.getcwd()
# Path to the folder you want Python to find and resize images
ROOT_PATH = args.root_path or os.path.join(RUN_PATH, "images")
# Path to the folder you want Python to save the resized images
SAVE_PATH = args.save_path or os.path.join(RUN_PATH, "model\\images")

MAX_IMAGES = 20


def calc_new_height(width, height, new_width):
    return round(new_width * height / width)

def resize(root, save_root, file, new_width, new_img_name):
    original_img_path = os.path.join(root, file)
    new_img_path = os.path.join(save_root, new_img_name)

    try:
        new_width = int(new_width)
    except:
        raise TypeError(
            f'-w, --new-width or NEW_WIDTH must be a number. Sent "{NEW_WIDTH}".')

    pillow_img = Image.open(original_img_path)

    width, height = pillow_img.size
    new_height = calc_new_height(width, height, new_width)

    new_img = pillow_img.resize((new_width, new_width), Image.ADAPTIVE)

    if not(os.path.exists(save_root)):
        os.makedirs(save_root)

    try:
        new_img.save(
            new_img_path,
            optimize=True,
            quality=50,
            exif=pillow_img.info.get('exif')
        )
    except:
        try:
            new_img.save(
                new_img_path,
                optimize=True,
                quality=50,
            )
        except:
            print(f'Could not convert "{original_img_path}".')

    print(f'Saved at {new_img_path}')


def is_image(extension):
    extension_lowercase = extension.lower()
    return bool(re.search(r'^\.(jpe?g)$', extension_lowercase))


def files_checks(root, dir, file, index):
    filename, extension = os.path.splitext(file)

    if not is_image(extension):
        return 0

    if index > MAX_IMAGES:
        return 0

    new_img_name = f"converted_{filename}{extension}"
    save_path = os.path.join(SAVE_PATH, dir)
    root_path = os.path.join(root, dir)

    resize(root=root_path, save_root=save_path, file=file,
           new_width=NEW_WIDTH, new_img_name=new_img_name)

    return 1


def files_loop(root, dir, files):
    index = 1

    for file in files:
        index += files_checks(root, dir, file, index)


def dir_loop(root_folder, dirs):
    for dir in dirs:
        dir_path = os.path.join(root_folder, dir)

        for _root, _dirs, files in os.walk(dir_path):
            files_loop(root_folder, dir, files)


def main(root_folder):
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH, ignore_errors=True)

    os.mkdir(SAVE_PATH)

    for root, dirs, files in os.walk(root_folder):
        # dir_loop(root_folder, dirs)
        files_loop(root, "", files)


if __name__ == '__main__':
    main(ROOT_PATH)

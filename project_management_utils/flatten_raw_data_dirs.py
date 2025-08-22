import os
import shutil
import click

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

@click.command()
@click.argument('src_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('dst_dir', type=click.Path(file_okay=False))
@click.option('--move', is_flag=True, help='Move files instead of copying')
def flatten_and_rename_images(src_dir, dst_dir, move):
    """
    Flatten image files from subdirectories of SRC_DIR into DST_DIR,
    renaming them as <subdir>--<filename>.
    """

    os.makedirs(dst_dir, exist_ok=True)

    for subdir_name in os.listdir(src_dir):
        subdir_path = os.path.join(src_dir, subdir_name)

        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    src_file = os.path.join(subdir_path, filename)
                    new_filename = f"{subdir_name}--{filename}"
                    dst_file = os.path.join(dst_dir, new_filename)

                    if move:
                        shutil.move(src_file, dst_file)
                        action = "Moved"
                    else:
                        shutil.copy2(src_file, dst_file)
                        action = "Copied"

                    click.echo(f"{action}: {src_file} -> {dst_file}")

if __name__ == '__main__':
    flatten_and_rename_images()

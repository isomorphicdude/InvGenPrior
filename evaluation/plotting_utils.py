"""Contains utility functions for plotting and visualization."""

import os
from PIL import Image, ImageSequence, ImageDraw, ImageFont
from IPython.display import Image as IPImage, display


def make_gif_gmm(img_dir, save_path, duration=1000):
    """
    Creates a GIF from images in a directory following the naming pattern "GMM_i.png".
    """
    files = os.listdir(img_dir)

    images = [file for file in files if file.startswith(f"GMM")]

    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    frames = [Image.open(os.path.join(img_dir, image)) for image in images]
     
    
    if frames:
        frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=duration, optimize=False)
    else:
        raise FileNotFoundError("No images found matching the pattern.")

    
    


def make_gif(img_dir,
             batch,
             i,
             save_path,
             view=True,
             duration=1000,
             font_size=40,
             title_height=40,
             name='pgdm'):
    """
    Creates a GIF from images in a directory following the naming pattern "batch_i_time.png".
    """
    files = os.listdir(img_dir)
    # print(files)

    # Filter files that match the pattern
    images = [file for file in files if file.startswith(f"{batch}_{i}_time")]

    # print(images)
    # Sort images by the time component in the filename
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Open images, add titles, and append them to a list
    frames = []
    for image in images:
        frame = Image.open(os.path.join(img_dir, image))
        frame_width, frame_height = frame.size

        # Create a new image with extra space for the title
        new_frame = Image.new('RGB', (frame_width, frame_height + title_height), 'black')
        new_frame.paste(frame, (0, title_height))

        draw = ImageDraw.Draw(new_frame)
        # Extract the time from the filename
        time = int(image.split('_')[-1].split('.')[0])
        # Draw the time above the image
        draw.text((10, 10), f"{name} Time: {time}", font = ImageFont.load_default(),
                  fill="white")

        frames.append(new_frame)

    # Save as GIF
    if frames:
        frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=duration, optimize=False)
    else:
        raise FileNotFoundError("No images found matching the pattern.")


    if view:
        display(IPImage(filename=save_path))

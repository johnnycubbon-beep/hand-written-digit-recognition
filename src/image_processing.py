from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

def _load_digit_pixels(filepath, invert=None, preserve_aspect_ratio=True):
    filepath = Path(filepath)

    with Image.open(filepath) as source_image:
        image = source_image.convert("L")

    if invert is None:
        invert = filepath.suffix.lower() != ".png"
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    if invert:
        pixels = 1.0 - pixels

    image = Image.fromarray(np.uint8(np.round(pixels * 255.0)))
    if preserve_aspect_ratio:
        image = ImageOps.contain(image, (28, 28), method=Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), color=0)
        canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
        image = canvas
    else:
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    x = (np.asarray(image, dtype=np.float32) / 255.0).reshape(784, 1)
    return x


def load_digit(filepath, invert=None, preserve_aspect_ratio=True):
    """Load one labelled image as a normalized (784, 1) input and integer label.

    If ``invert`` is None, PNG files are treated as light digits on a dark
    background and other formats as dark digits on a light background. Set it
    explicitly when a file does not follow that convention.
    """
    filepath = Path(filepath)
    label = int(filepath.stem[-1])
    x = _load_digit_pixels(filepath, invert, preserve_aspect_ratio)

    return (x, label)


def load_unlabelled_digit(filepath, invert=None, preserve_aspect_ratio=True):
    """Load an image without reading a label from its filename."""
    return _load_digit_pixels(filepath, invert, preserve_aspect_ratio)

def load_digit_folder(folder_path, invert=None, preserve_aspect_ratio=True):
    folder_path = Path(folder_path)
    data = []
    for filepath in sorted(folder_path.iterdir()):
        if filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            data.append(
                load_digit(
                    filepath,
                    invert=invert,
                    preserve_aspect_ratio=preserve_aspect_ratio,
                )
            )
    return data

def translate_data(data, vertical_shift, horizontal_shift, side_length=28):

    x, y = data

    if len(x) != side_length**2:
             raise ValueError("Input data does not have a length compatible with the grid side length.")
         
    matrix = np.reshape(x, (side_length, side_length))
    shifted_data = np.roll(matrix, shift=vertical_shift, axis=0)
    shifted_data = np.roll(shifted_data, shift=horizontal_shift, axis=1)

    return (np.reshape(shifted_data, (-1,1)), y)

     

def translate_dataset(dataset, vertical_shift, horizontal_shift, side_length=28):
        """
        Inputs  
        data: List of tuples (x,y) where x is a one dimensional numpy array of length side_length^2 and y is the label for the digit.
        vertical_shift: An integer which indicates how far the image is translated downwards (negative is upwards)
        horizontal_shift: An integer which indicates how far the image is translated downwards
        side_length: An integer, the side length of the matrix of pixel activations.
    
        Returns shifted_data of the same shape as data which corresponds to the matrix shifted up or down
        """
        new_dataset = []
        for data in dataset:
             new_data = translate_data(data=data, vertical_shift=vertical_shift, horizontal_shift=horizontal_shift, side_length=side_length)
             new_dataset.append(new_data)
        
        return new_dataset

def centre_data(data, side_length=28):
    """
    Inputs  
    data: Tuple (x,y) where x is a 1 dimensional numpy array of length side_length^2 and y is the true label
    side_length: An integer, the side length of the matrix of pixel activations.

    Returns data for an digit image whose weighted centre of mass is as close as possible to
    the centre of the frame, which is a tuple with a different numpy array as the first entry but the same label.
    """
    x, y = data
    if len(x) != side_length**2:
            raise ValueError("Input data does not have a length compatible with the grid side length.")
        
    matrix = np.reshape(x, (side_length, side_length))
    total_mass = float(np.sum(matrix))
    if total_mass == 0:
        # A blank image has no centre of mass; return it unchanged.
        return (np.array(x, copy=True).reshape(-1, 1), y)

    row_coordinates, column_coordinates = np.indices(matrix.shape)
    centre_of_mass = np.array([
        np.sum(row_coordinates * matrix) / total_mass,
        np.sum(column_coordinates * matrix) / total_mass,
    ])
    target_centre = np.array([(side_length - 1) / 2, (side_length - 1) / 2])
    vertical_shift, horizontal_shift = np.rint(target_centre - centre_of_mass).astype(int)

    return translate_data(
        data=data,
        vertical_shift=vertical_shift,
        horizontal_shift=horizontal_shift,
        side_length=side_length,
    )


def centre_dataset(dataset, side_length=28):
    """
    Inputs  
    dataset: Lists of tuples (x,y) where x is a 1 dimensional numpy array of length side_length^2 and y is the true label
    side_length: An integer, the side length of the matrix of pixel activations.

    Returns list of tuples (x, y) one for each digit image whose weighted centre of mass is as close as possible to
    the centre of the frame, which is a tuple with a different numpy array as the first entry but the same label.
    """
    return [centre_data(data, side_length) for data in dataset]



     




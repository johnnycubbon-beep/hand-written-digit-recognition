"""Collect labelled drawings or run a blind test on handwritten digits."""

import re
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw


SIZE = 280
BRUSH_RADIUS = 10
LAYER_SIZES = [784, 16, 16, 10]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
LABELLED_OUTPUT = PROJECT_ROOT / "data" / "raw" / "better_wacom_digits"
BLIND_OUTPUT = PROJECT_ROOT / "data" / "raw" / "unlabelled_digit_runs"
MODEL_FILE = PROJECT_ROOT / "data" / "processed" / "centred_trained_network.npz"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import image_processing
import neural_network_nielsen


def choose_mode():
    root = tk.Tk()
    root.title("Digit test")
    root.resizable(False, False)
    tk.Label(root, text="Choose what you want to do:").pack(padx=16, pady=(14, 8))

    choice = {"mode": None}

    def select(mode):
        choice["mode"] = mode
        root.destroy()

    options = (
        ("Draw labelled digits (save in numbered folders)", "labelled"),
        ("Draw anonymous digits (blind test)", "blind_draw"),
        ("Choose a folder of anonymous digits", "blind_folder"),
    )
    for text, mode in options:
        tk.Button(root, text=text, command=lambda selected=mode: select(selected)).pack(
            fill="x", padx=14, pady=4
        )

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return choice["mode"]


def load_network():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Trained network not found: {MODEL_FILE}")
    network = neural_network_nielsen.Network(sizes=LAYER_SIZES)
    network.load(MODEL_FILE)
    return network


def predict_images(network, image_files):
    predictions = []
    for filepath in image_files:
        pixels = image_processing.load_unlabelled_digit(filepath)
        centered_input, _ = image_processing.centre_data((pixels, None))
        output = network.feedforward(centered_input)
        predictions.append(int(np.argmax(output)))
    return predictions


def reveal_labels(root, image_files, predictions):
    guess_lines = [
        f"{filepath.name}: guessed {prediction}"
        for filepath, prediction in zip(image_files, predictions)
    ]
    guesses_text = "\n".join(guess_lines)
    print("\nModel guesses (in filename order):")
    print(guesses_text)
    messagebox.showinfo(
        "Model guesses",
        guesses_text + "\n\nNow enter the actual digits in the same order.",
        parent=root,
    )

    actual_text = simpledialog.askstring(
        "Reveal the labels",
        f"Enter {len(image_files)} actual digits in filename order, separated by spaces or commas.",
        parent=root,
    )
    if not actual_text:
        return

    actual_tokens = [token for token in re.split(r"[\s,]+", actual_text.strip()) if token]
    valid_labels = (
        len(actual_tokens) == len(predictions)
        and all(token in {str(digit) for digit in range(10)} for token in actual_tokens)
    )
    if not valid_labels:
        messagebox.showerror(
            "Invalid labels",
            f"Enter exactly {len(predictions)} digits from 0 to 9, in filename order.",
            parent=root,
        )
        return

    actual_labels = [int(token) for token in actual_tokens]
    result_lines = [
        f"{filepath.name}: guessed {prediction}, actual {actual}"
        f" ({'correct' if prediction == actual else 'incorrect'})"
        for filepath, prediction, actual in zip(image_files, predictions, actual_labels)
    ]
    accuracy = sum(
        prediction == actual
        for prediction, actual in zip(predictions, actual_labels)
    ) / len(predictions)
    report = "\n".join(result_lines) + f"\n\nAccuracy: {accuracy:.1%}"
    print("\nBlind-test results:")
    print(report)
    messagebox.showinfo("Blind-test results", report, parent=root)


def draw_digits(blind_test):
    root = tk.Tk()
    if blind_test:
        network = load_network()
        root.withdraw()
        while True:
            collection_name = simpledialog.askstring(
                "Name this collection",
                "Enter a folder name for these anonymous digits (for example, Johnny_Digits):",
                parent=root,
            )
            if collection_name is None:
                root.destroy()
                return
            collection_name = collection_name.strip()
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", collection_name):
                break
            messagebox.showerror(
                "Invalid folder name",
                "Use letters, numbers, underscores, or hyphens; start with a letter or number.",
                parent=root,
            )

        output_folder = BLIND_OUTPUT / collection_name
        output_folder.mkdir(parents=True, exist_ok=True)
        existing_numbers = []
        for filepath in output_folder.glob("digit_*.png"):
            match = re.fullmatch(r"digit_(\d+)", filepath.stem)
            if match:
                existing_numbers.append(int(match.group(1)))
        first_number = max(existing_numbers, default=0) + 1
        root.deiconify()
    else:
        network = None
        output_folder = LABELLED_OUTPUT

    root.title("Draw Digits" if not blind_test else "Blind Digit Test")
    instruction = (
        "Draw one digit at a time. Save each drawing without entering its label."
        if blind_test
        else "Draw one digit at a time. Save each digit in its numbered folder."
    )
    tk.Label(root, text=instruction).pack(padx=10, pady=(10, 4))
    root.protocol("WM_DELETE_WINDOW", root.quit)

    canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="black")
    canvas.pack(padx=10, pady=6)

    image = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(image)
    has_drawing = False
    next_number = first_number if blind_test else 1

    def paint(event):
        nonlocal has_drawing
        has_drawing = True
        x, y = event.x, event.y
        radius = BRUSH_RADIUS
        canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="white",
            outline="white",
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    def clear():
        nonlocal has_drawing
        canvas.delete("all")
        draw.rectangle((0, 0, SIZE, SIZE), fill=0)
        has_drawing = False

    def save():
        nonlocal next_number
        if not has_drawing:
            return

        if blind_test:
            filename = output_folder / f"digit_{next_number:03d}.png"
        else:
            label = simpledialog.askinteger(
                "Label digit",
                "Which digit did you draw? (0-9)",
                parent=root,
                minvalue=0,
                maxvalue=9,
            )
            if label is None:
                return
            label_folder = output_folder / str(label)
            label_folder.mkdir(parents=True, exist_ok=True)
            existing = list(label_folder.glob("*.png"))
            filename = label_folder / f"{len(existing) + 1}_{label}.png"

        image.save(filename)
        print(f"Saved drawing: {filename}")
        next_number += 1
        clear()

    canvas.bind("<B1-Motion>", paint)
    buttons = tk.Frame(root)
    buttons.pack(pady=(0, 10))
    tk.Button(buttons, text="Save drawing", command=save).pack(side="left", padx=4)
    tk.Button(buttons, text="Clear", command=clear).pack(side="left", padx=4)
    tk.Button(buttons, text="Finish", command=root.quit).pack(side="left", padx=4)

    if blind_test:
        print(f"Anonymous drawings will be saved in: {output_folder}")
    root.mainloop()
    root.withdraw()

    if blind_test:
        image_files = sorted(
            filepath
            for filepath in output_folder.glob("digit_*.png")
            if (match := re.fullmatch(r"digit_(\d+)", filepath.stem))
            and int(match.group(1)) >= first_number
        )
    else:
        image_files = []
    if blind_test and image_files:
        predictions = predict_images(network, image_files)
        reveal_labels(root, image_files, predictions)
    elif blind_test:
        print("No drawings were saved.")

    root.destroy()


def classify_folder():
    network = load_network()
    root = tk.Tk()
    root.withdraw()
    folder_name = filedialog.askdirectory(
        title="Choose a folder containing anonymous digit images",
        parent=root,
    )
    if not folder_name:
        root.destroy()
        return

    folder = Path(folder_name)
    image_files = sorted(
        filepath
        for filepath in folder.iterdir()
        if filepath.is_file() and filepath.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_files:
        messagebox.showinfo("No images found", "That folder contains no supported images.", parent=root)
        root.destroy()
        return

    predictions = predict_images(network, image_files)
    reveal_labels(root, image_files, predictions)
    root.destroy()


def main():
    mode = choose_mode()
    if mode == "labelled":
        draw_digits(blind_test=False)
    elif mode == "blind_draw":
        draw_digits(blind_test=True)
    elif mode == "blind_folder":
        classify_folder()


if __name__ == "__main__":
    main()

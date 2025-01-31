import os
import csv
import random
from PIL import Image, ImageDraw

def generate_dataset(
    output_dir="C:\Data",
    csv_filename="trainingData.csv",
    num_samples=1000,
    image_size=(64, 64)
):
    """
    Generate a dataset of 64x64 images of various shapes, colors, and sizes,
    and store the text-image pairs for later training.
    
    :param output_dir: directory to store the generated images
    :param csv_filename: name of the CSV file to store image paths & descriptions
    :param num_samples: number of images to generate
    :param image_size: tuple (width, height) for the output images
    """
    shapes = ["circle", "rectangle", "triangle"]
    colors = ["red", "green", "blue", "yellow"]
    sizes = ["small", "medium", "large"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["image_path", "text_description"])
        
        for i in range(num_samples):
            shape = random.choice(shapes)
            color = random.choice(colors)
            size_label = random.choice(sizes)
            
            img = Image.new("RGB", image_size, color="white")
            draw = ImageDraw.Draw(img)
            
            # Determine bounding box size based on size label
            # (These are rough heuristics; feel free to adjust)
            if size_label == "small":
                shape_size = 16
            elif size_label == "medium":
                shape_size = 32
            else:  # "large"
                shape_size = 48
            
            # Coordinates to center the shape
            w, h = image_size
            left = (w - shape_size) // 2
            top = (h - shape_size) // 2
            right = left + shape_size
            bottom = top + shape_size
            
            # Draw the chosen shape
            if shape == "circle":
                draw.ellipse([left, top, right, bottom], fill=color, outline=color)
            elif shape == "rectangle":
                draw.rectangle([left, top, right, bottom], fill=color, outline=color)
            else:  # "triangle"
                # Coordinates of an isosceles triangle centered in the image
                x_center = w // 2
                # top vertex, bottom-left, bottom-right
                triangle_coords = [
                    (x_center, top),
                    (left, bottom),
                    (right, bottom)
                ]
                draw.polygon(triangle_coords, fill=color, outline=color)
            
            # Construct text description
            # Example: "a small red circle" or "a large blue triangle"
            text_description = f"a {size_label} {color} {shape}"
            
            # Save image
            image_filename = f"image_{i:05d}.png"
            image_path = os.path.join(output_dir, image_filename)
            img.save(image_path)
            
            # Write row to CSV: [relative_path, text_description]
            writer.writerow([image_filename, text_description])
    
    print(f"Dataset generation complete! {num_samples} images saved to '{output_dir}'.")
    print(f"Metadata stored in '{csv_path}'.")

if __name__ == "__main__":
    # Example usage
    generate_dataset(
        output_dir="C:\Data",
        csv_filename="trainingData.csv",
        num_samples=2000,     # Adjust as needed
        image_size=(64, 64)
    )

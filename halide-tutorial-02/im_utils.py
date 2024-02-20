from PIL import Image
import numpy as np

def save_bird():
    rgba_image = Image.open("bird.png")
    rgb_image = rgba_image.convert('RGB')
    np.save("bird.npy", np.array(rgb_image))

def load_bird():
    img = Image.fromarray(np.load("brighter.npy"))
    img.save("bird_brightened.png")

# save_bird()
load_bird()
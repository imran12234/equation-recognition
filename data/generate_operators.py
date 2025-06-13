from PIL import Image, ImageDraw, ImageFont
import os

symbols = ['+', '-', '*', '/', '=']
output_dir = "data/operators"
os.makedirs(output_dir, exist_ok=True)
try:
    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 24)  # MacOS
except:
    font = ImageFont.load_default()  # fallback if font not found
for s in symbols:
    img = Image.new("L", (28, 28), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((6, 0), s, fill=0, font=font)
    
    # Make filename safe
    safe_name = s.replace('*', 'mul').replace('/', 'div') \
                 .replace('+', 'plus').replace('-', 'minus').replace('=', 'eq')
    img.save(f"{output_dir}/{safe_name}.png")


print("✅ Saved operator images to", output_dir)

from PIL import Image
import os
import random

DIGITS_PATH = "data/digits"
OPERATORS_PATH = "data/operators"
OUTPUT_PATH = "data/equations"
os.makedirs(OUTPUT_PATH, exist_ok=True)

op_map = {
    '+': 'plus.png',
    '-': 'minus.png',
    '*': 'mul.png',
    '/': 'div.png',
    '=': 'eq.png'
}

filename_op_map = {
    '+': 'plus',
    '-': 'minus',
    '*': 'mul',
    '/': 'div',
    '=': 'eq'
}

digit_images = [os.path.join(DIGITS_PATH, f) for f in os.listdir(DIGITS_PATH)]

def load_digit(d):
    matches = [img for img in digit_images if img.startswith(f"{DIGITS_PATH}/{d}_")]
    return Image.open(random.choice(matches))

def load_symbol(sym):
    return Image.open(f"{OPERATORS_PATH}/{op_map[sym]}")

def make_equation_image(a, op, b, c, idx):
    parts = [
        load_digit(a),
        load_symbol(op),
        load_digit(b),
        load_symbol('='),
        load_digit(c)
    ]
    combined = Image.new("L", (28 * 5, 28), color=255)
    for i, img in enumerate(parts):
        combined.paste(img, (i * 28, 0))
    
    # Word-based filename
    op_word = filename_op_map[op]
    eq_name = f"{a}{op_word}{b}eq{c}_{idx}.png"
    combined.save(f"{OUTPUT_PATH}/{eq_name}")

# Generate 4000 examples
for i in range(4000):
    a = random.randint(0, 9)
    b = random.randint(0, 9)
    op = random.choice(['+', '-', '*'])
    try:
        c = eval(f"{a}{op}{b}")
        if 0 <= c <= 9:
            make_equation_image(a, op, b, c, i)
    except:
        continue

print("✅ Equations saved in word format.")

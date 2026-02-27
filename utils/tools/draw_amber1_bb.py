from PIL import Image, ImageDraw

image_path = "/leonardo/home/usertrain/a08trc0u/Vision_Challenge/data/amber_disc/images/AMBER_1.jpg"
out_path = "/leonardo/home/usertrain/a08trc0u/Vision_Challenge/utils/tools/AMBER_1_with_bb.jpg"

# Detection result box: [x1, y1, x2, y2]
box = [1.7896, -0.12668, 373.09, 220.44]
label = "blue sky"
score = 0.6474

img = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)

w, h = img.size
x1, y1, x2, y2 = box
# Clamp to image bounds and convert to ints for drawing
x1 = max(0, min(w - 1, int(round(x1))))
y1 = max(0, min(h - 1, int(round(y1))))
x2 = max(0, min(w - 1, int(round(x2))))
y2 = max(0, min(h - 1, int(round(y2))))

# Draw bbox
draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)

# Draw simple text
text = f"{label} {score:.3f}"
draw.rectangle([x1, y1, min(w - 1, x1 + 180), min(h - 1, y1 + 24)], fill=(255, 0, 0))
draw.text((x1 + 4, y1 + 4), text, fill=(255, 255, 255))

img.save(out_path)
print(out_path)

from pathlib import Path
from PIL import Image

result = list(Path("../capstone-mlnd-data/seedlings-data").glob('**/*.png'))

largest_width = 0
largest_height = 0
largest_area = 0

smallest_width = 1000000 
smallest_height = 1000000
smallest_area = 1000000

for filename in result:
	with Image.open(filename) as image:
		width, height = image.size
		area = width * height
		if area > largest_area:
			largest_area = area
			largest_width = width
			largest_height = height
		if area < smallest_area:
			smallest_area = area
			smallest_width = width
			smallest_height = height

print(f"smallest {smallest_width} X {smallest_height} - largest {largest_width} X {largest_height}")

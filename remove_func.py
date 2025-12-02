import os

path = 'd:/fupan/fupan.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove lines 510 to 1163 (1-based)
# Index 509 to 1162
# Keep 0-508 and 1163-end
new_lines = lines[:509] + lines[1163:]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

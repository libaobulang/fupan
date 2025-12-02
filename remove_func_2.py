import os

path = 'd:/fupan/fupan.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove lines 73 to 219 (1-based)
# Index 72 to 218
# Keep 0-72 and 219-end
new_lines = lines[:72] + lines[219:]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

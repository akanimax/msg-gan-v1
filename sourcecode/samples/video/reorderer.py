from shutil import copy
import os

file_path = "../Celeba/3/4_x_4"
files = os.listdir(file_path)
factor = 2
to_dir = "reordered"

count = 1
ot_count = 1

for j in range(8):
  for k in range(9000):
    file_name = "gen" + "_" + str(j) + "_" + str(k) + ".png"
    if file_name in files:
      if ot_count % factor == 0:
        copy(os.path.join(file_path, file_name), os.path.join(to_dir, str(count) + ".png"))
        count += 1
      ot_count += 1


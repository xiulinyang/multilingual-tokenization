import os
import re
import shutil

# Path to model folder
model_dir = "AR_50000_41"

# Get all checkpoint folders
checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
# Extract numbers and sort
checkpoints_sorted = sorted(checkpoints, key=lambda x: int(re.findall(r"\d+", x)[0]))

print(checkpoints_sorted)
# Rename
for chk_id, ckpt in enumerate(checkpoints_sorted):
    i=chk_id+1
    i = str(i)
    src = os.path.join(model_dir, ckpt)
    dst = os.path.join(model_dir, f"epoch_{i}")
    print(f"Renaming {ckpt} -> epoch_{i}")
    shutil.move(src, dst)

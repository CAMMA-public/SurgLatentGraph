import os
import pandas as pd
import shutil

# Paths

# Update these paths as needed
csv_path = './cholec80_CVS.csv'  # Assuming the CSV is in the current working directory
frames_root = './data/Cholec80'  # Relative to workspace root
output_root = './data/Cholec80_CVS_test'

df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    video_id = str(row['video']).zfill(2)  # e.g., 2 -> '02'
    start_minute = int(row['initial_minute'])
    start_second = int(row['initial_second'])
    end_minute = int(row['final_minute'])
    end_second = int(row['final_second'])

    # Calculate start and end frame numbers (1 fps, so frame = minute*60 + second)
    start_frame = start_minute * 60 + start_second
    end_frame = end_minute * 60 + end_second

    for frame_num in range(start_frame, end_frame + 1):
        frame_name = f"{frame_num:08d}.jpg"
        src = os.path.join(frames_root, video_id, frame_name)
        dst_dir = os.path.join(output_root, video_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, frame_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Missing: {src}")
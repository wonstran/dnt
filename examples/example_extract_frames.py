import os

from dnt.label.labeler import Labeler

input_path = "/mnt/d/videos/benesch/24-120161 (Tampa)/Cam 1 (EB view)"
out_path = "/mnt/d/videos/benesch/24-120161 (Tampa)/frames/eb"
videos = os.listdir(input_path)

labeler = Labeler()
for video in videos:
    labeler.export_frames(
        input_video=os.path.join(input_path, video), frames=[0], output_path=out_path, prefix=os.path.splitext(video)[0]
    )
    print(os.path.join(input_path, video))

print("okay")

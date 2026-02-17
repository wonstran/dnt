import pandas as pd

from dnt.label.labeler import Labeler

video_file = "/mnt/d/videos/hfst/Standard_SCU7WH_2022-09-13_1600.004.mp4"
ouput_file = "/mnt/d/videos/hfst/labels/shapes_v2.mp4"

points = [
    [256, 228],
    [408, 337],
    [334, 212],
    [515, 297],
    [403, 197],
    [596, 268],
    [455, 187],
    [651, 247],
    [507, 174],
    [693, 231],
]

lines = [[256, 228, 408, 337], [334, 212, 515, 297], [403, 197, 596, 268], [455, 187, 651, 247], [507, 174, 693, 231]]

polygons = [[25, 70], [25, 160], [110, 200], [200, 160], [200, 70], [110, 20]]

polygons2 = [(25, 70), (25, 160), (110, 200), (200, 160), (200, 70), (110, 20)]

results = []
for i in range(1000):
    results.append([i, "polylines", polygons2, (0, 255, 0), 1, 2, ""])

df = pd.DataFrame(results, columns=["frame", "type", "coords", "color", "size", "thick", "desc"])

labeler = Labeler()
labeler.draw(video_file, output_video=ouput_file, draws=df)

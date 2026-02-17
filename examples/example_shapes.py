from dnt.label import Labeler

video_file = "/mnt/d/videos/hfst/Standard_SCU7WH_2022-09-13_1600.004.mp4"
ouput_file = "/mnt/d/videos/hfst/labels/shapes.mp4"

points = [[256, 228],[408, 337], 
          [334, 212],[515, 297],
          [403, 197],[596, 268],
          [455, 187],[651, 247],
          [507, 174],[693, 231]]

lines = [[256, 228, 408, 337], 
          [334, 212, 515, 297],
          [403, 197, 596, 268],
          [455, 187, 651, 247],
          [507, 174, 693, 231]]

polygons = [[[256, 228], [408, 337], [515, 297], [334, 212]],
          [[334, 212], [515, 297], [596, 268], [403, 197]],
          [[403, 197], [596, 268], [651, 247], [455, 187]],
          [[455, 187], [651, 247], [693, 231], [507, 174]]]

labeler = Labeler()
labeler.draw_shapes(input_video=video_file, output_video=ouput_file, points = points, polygons=polygons, 
                    point_color=(0,0,255), polygon_color=(0, 255, 0))


import pyrealsense2 as rs

#create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

try:
    while True:
        #create a pipeline object. This object configures the streaming camera and owns its handles
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        #print a simple text-bases representation of the image, by breaking it into 10x20 pixel regions and approximating the
        coverage = [0]*64
        for y in xrange(480):
            for x in xrange(640):
                dist = depth.get_distance(x, y)
                if 0 < dist and dist < 1:
                    coverage[x/10] += 1

            if y%20 is 19:
                line = ""
                for c in coverage:
                    line += " .:nhBXWW"[c/255]
                coverage = [0]*64
                print(line)

finally:
    pipeline.stop()

import sys
import os

import cv2

def main(): 

    if(len(sys.argv) < 2): 
        print("call this thing with file containing paths to frames")
        exit()

    with open(sys.argv[1]) as f: 
        dirs = f.readlines()
        print("comparing: " + str(dirs))

        nhf = len(dirs)+1
        if(len(sys.argv) > 2): 
            nhf = int(sys.argv[2])
        out_file = str(sys.argv[1]) + ".avi"
        if(os.path.exists(out_file)): 
            os.remove(out_file)
        img_shape = None
        i = 0
        writer = None
        while True: 
            frames = []
            layers = []
            if(i%5==0): 
                print("Frame: " + str(i))
            x = 0
            gt_count = 0
            display = 0
            pdiri = ""
            for diri in dirs: 
                diri = diri.strip()
                if diri[:3] == "DV:": #deep voxels
                    diri = diri[3:]
                    framediri = os.path.join(diri, f"img_{i:05d}.png")
                elif diri[:3] == "DD:": #DeeoVoxels Down sample
                    diri = diri[3:]
                    framediri = os.path.join(diri, str(i) + "_rgb.png")
                    gt_count += 1
                elif diri[:3] == "GT:": 
                    diri = diri[3:]
                    framediri = os.path.join(diri, str(i) + "_rgb_target.png")
                    gt_count += 1
                elif diri == "GT": 
                    framediri = os.path.join(pdiri, str(i) + "_rgb_target.png")
                    gt_count += 1
                else: 
                    framediri = os.path.join(diri, str(i) + "_rgb_fake.png")
                    pdiri = diri
               # print(framediri)
                if (not os.path.exists(framediri)):
                    print("Ended at: " + str(framediri))
                    writer.release()
                    exit()
                im = cv2.imread(framediri)
                if not img_shape: 
                    img_shape = im.shape[:2]
                    print ("Shape: " + str(img_shape))
                elif not im.shape[:2] == img_shape:                 
                    im = cv2.resize(im, img_shape)

                if(x==0): 
                    display = im
                elif x == nhf-1: 
                    display = cv2.hconcat([display, im])
                    layers.append(display)
                    x = -1
                else: 
                    display = cv2.hconcat([display, im])
                x+=1
            
            if gt_count <1 : 
                im = cv2.imread(os.path.join(diri, str(i) + "_rgb_target.png"))
                display = cv2.hconcat([display, im])
                layers.append(display)
            if len(layers)>1: 
                display = cv2.vconcat(layers)

            if(writer == None): 
                dims = display.shape[1], display.shape[0]
                print("Dims: " + str(dims))
                print(len(layers))
                writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"XVID"), 15.0, dims, True)

            writer.write(display)
            i+=1

if __name__ == "__main__": 
    main()

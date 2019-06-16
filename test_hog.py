import os
import sys
import glob

import dlib

if len(sys.argv) != 2:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py ../examples/faces")
    exit()
faces_folder = sys.argv[1]


options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4
options.be_verbose = True


training_xml_path = os.path.join(faces_folder, "training.xml")
testing_xml_path = os.path.join(faces_folder, "testing.xml")
detector = dlib.simple_object_detector("detector.svm")

win_det = dlib.image_window()
win_det.set_image(detector)

print("Showing detections on the images in the faces folder...")
win = dlib.image_window()
test_files = ['data/test.png']
for f in test_files:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

detector1 = dlib.fhog_object_detector("detector.svm")
detector2 = dlib.fhog_object_detector("detector.svm") 
detectors = [detector1, detector2]
image = dlib.load_rgb_image(faces_folder + '/test.png')
[boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=1, adjust_threshold=0.0)
for i in range(len(boxes)):
    print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))

sys.exit()
images = [dlib.load_rgb_image(faces_folder + '/test.png'),
          dlib.load_rgb_image(faces_folder + '/test.png')]
boxes_img1 = ([dlib.rectangle(left=329, top=78, right=437, bottom=186),
               dlib.rectangle(left=224, top=95, right=314, bottom=185),
               dlib.rectangle(left=125, top=65, right=214, bottom=155)])
boxes_img2 = ([dlib.rectangle(left=154, top=46, right=228, bottom=121),
               dlib.rectangle(left=266, top=280, right=328, bottom=342)])
boxes = [boxes_img1, boxes_img2]

detector2 = dlib.train_simple_object_detector(images, boxes, options)

win_det.set_image(detector2)
dlib.hit_enter_to_continue()

print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(images, boxes, detector2)))


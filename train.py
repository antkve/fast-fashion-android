import os
import sys
import glob

import dlib

if len(sys.argv) != 2:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_shape_predictor.py ../examples/faces")
    exit()
faces_folder = sys.argv[1]

options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

training_xml_path = os.path.join(faces_folder, "training.xml")
dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)

print("\nTraining accuracy: {}".format(
    dlib.test_shape_predictor(training_xml_path, "predictor.dat")))
testing_xml_path = os.path.join(faces_folder, "testing.xml")
print("Testing accuracy: {}".format(
    dlib.test_shape_predictor(testing_xml_path, "predictor.dat")))

predictor = dlib.shape_predictor("predictor.dat")
detector = dlib.get_frontal_face_detector()

print("Showing detections and predictions on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        win.add_overlay(shape)

    win.add_overlay(dets)
    dlib.hit_enter_to_continue()



# character_rec
download training data from : http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

# preparing training dataset
In `transfer_dataset.py`

1.set number of validation images  _NUM_VALIDATION

2.set image size _IMAGE_SIZE 

3.set dataset_dir to your dataset directory. The structure should be like /path/class1/xxx.png /path/class2/xxx.png 
or else you have to change some code.

4.it will generate xxx.tfrecord. you can change the name pattern in function `_get_dataset_filename`

# training 
In `train.py`

1.set FLAGS, usually set depth multiplier, checkpoint_dir, num_classes, and image_size. training_set_size.
FLAGS.dataset_dir is where you store *.tfrecord

2.run train.py

3.notice that though the network may seem to converge after 2k steps, however it should run at least 10,000 steps to get a reasonable result.

4.in `preprocessing_factory`, function `distorted_bounding_box_crop`, change area_range and min_objected_covered to strengthen the ability of size invariance.

# evaluation
In `eval.py`

1.set the same thing as training.

2.run `eval.py` (in some operating systems you may have to point out the checkpoint you use by setting model_dir as like model.ckpt-12345 or it will report bug.

# test with your own image
In `test.py`

1.set FLAGS.test_pic as the test_image, when you run the image, it will show you the image and print the character it predict. Click any key to recognize the next one.


# small labeling tools
in `split_char.py`

if you have a large image with a lot of characters and you want to sperate them to individual images.

1.run the main function, change the input image in main function, and press the corresponding key. If you see A just press "a" and it will save image to directory "A".

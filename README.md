# RevisorVision_test
 My test task for CV Engineer
First we open video using OpenCV and safe to one folder every 15th image from it.
Second step is to crop and resize images by given coordinates and shape.
Next, we are sorting every image from this dataset by the rule "If person is on the camera at this moment - it has to be 'At work' group",otherwise 'Absent'.
I didn't take into account other people for my classification, because it was only one's person working place. So if there was somebody else, I classificated this images like 'Absent'.

Next step is to divide our dataset to the train/valid and test data flow. I used TensorFlow flow_from_directory() function because it automatically transfer different directories like different calsses which is convenient.

Using Matplotlib I checked whether the split was performed okay or not.

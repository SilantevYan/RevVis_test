# RevisorVision_test
 My test task for CV Engineer
 
First we open video using OpenCV and safe to one folder every 15th image from it.
Second step is to crop and resize images by given coordinates and shape.
Next, we are sorting every image from this dataset by the rule "If person is on the camera at this moment - it has to be 'At work' group",otherwise 'Absent'.
I didn't take into account other people for my classification, because it was only one's person working place. So if there was somebody else, I classificated this images like 'Absent'.

Next step is to divide our dataset to the train/valid and test data flow. I used TensorFlow flow_from_directory() function because it automatically transfer different directories like different calsses which is convenient.

Using Matplotlib I checked whether the split was performed okay or not.
![RevisorVision](https://user-images.githubusercontent.com/96116349/188224891-eaa07f4f-15be-4af2-bddc-c3ce947dc4ae.jpg)
The split is alright.
Then I tried simple neural network (Sequential) with some Conv2D/Activation/MaxPooling2D layers. In the end I used Flatten/Dense layers and Dropout to lower overfitting of the network.
This model performs quite well with the val accuracy at 0.97-0.98 when it stopped training at 9th epoch.
I saved pretrained model for the further usage. This model also perform well with test data.
Here are some metrics:
![RevisorVision_model predictions](https://user-images.githubusercontent.com/96116349/188226528-ead655cc-3bb8-4b7e-ae1b-ecc627495352.jpg)


# IIT_ML_m22AI606
# For Q2
To build a robust CNN model for this dataset, we need to follow these steps:

Data Preprocessing: We need to preprocess the dataset by scaling the pixel values between 0 and 1, and split it into training and testing sets. We can also perform data augmentation to increase the size of our training data and reduce overfitting.

Model Architecture: We can use a simple CNN architecture consisting of multiple Convolutional layers followed by MaxPooling layers and then a few dense layers for classification. We can experiment with different hyperparameters such as number of filters, kernel size, and activation functions to get the best performance.

Model Training: We can train our model on the training set and validate it on the validation set. We can use different optimization algorithms such as Adam, RMSprop, or SGD and different loss functions such as categorical cross-entropy or binary cross-entropy.

Model Evaluation: Once the model is trained, we can evaluate its performance on the test set and calculate metrics such as accuracy, precision, recall, and F1 score. We can also visualize the performance using confusion matrix and classification report.

#for q3

Task 1: Dataset Preparation

Download the dataset from the given drive link and extract it.
The dataset consists of 1750 chart images of different types (Line, Dot Line, Horizontal Bar, Vertical Bar, and Pie chart) and a CSV file containing corresponding labels for the images.
Split the dataset into training and validation sets in an appropriate ratio (e.g., 80% for training and 20% for validation) using train_test_split() function from scikit-learn library.
Load the images and their corresponding labels using OpenCV library.
Preprocess the images by resizing them to a fixed size, converting them to grayscale, and scaling their pixel values between 0 and 1.

Task 2: Two-Layer CNN Implementation

Define a two-layer CNN architecture with Conv2D and MaxPooling2D layers followed by Flatten and Dense layers for classification.
Compile the model with an appropriate optimizer (e.g., Adam), loss function (e.g., categorical cross-entropy), and evaluation metric (e.g., accuracy).
Train the model on the training set and evaluate its performance on the validation set.
Calculate accuracy, loss, and plot the obtained loss using matplotlib library.
Observe the training and validation accuracy and loss curves to check for overfitting.

Task 3: Finetune a Pretrained Network

Load a pretrained network (e.g., AlexNet) using keras library and freeze its convolutional layers.
Add a few new layers (e.g., Flatten and Dense layers) for classification.
Train the model on the training set and evaluate its performance on the validation set.
Calculate accuracy, loss, and plot the obtained loss.
Compare the performance of the finetuned network with the two-layer CNN.

# CrashCal

## Objective

Use computer vision to train convolution neural networks in accurately classifying vehicle damage for efficient claims triage.

## Use Case

The rapidly growing automobile industry is highly supportive of the auto insurance market, that is also growing equally fast. However, till now, this industry has been solely based on conventional ways of filing manual repair claims. In case of any unfortunate accident, the claims for the car damage need to be filed manually. It needs an on-site visit by an inspector to inspect vehicles physically for assessing damage and obtaining a cost estimate. This also opens the window to incorrect settlement due to human error. Also, it would make the whole process of doing so much more convenient with the help of machine learning and remote usage, increasing productivity for both sides of the damage-insurance carrier and customer satisfaction.

While the technology is yet to achieve the highest possible levels of accuracy, above is a proof of concept for the application of Deep Learning and Computer Vision in automating damage assessments by building and training Convolution Neural Networks.

## Solution

This system could most easily be automated by developing a Convolutional Neural Network model capable of accepting images from the user and determining the location and severity of the damage. All this requires the model to pass through many checks; first and foremost, the model should make sure the given image is of a car, and then it must make sure the vehicle is indeed damaged. These are the gate checks before the analysis begins. The damage check will start once all the gate checks have been validated. The model will predict the location of damage in front, side, or rear and the severity of such damage as minor, moderate, or severe.

The model accepts an input image from the user and processes it across 4 stages:

1. Validates that given image is of a car.

2. Validates that the car is damaged.

3. Finds location of damage as front, rear or side

4. Determines severity of damage as minor, moderate or severe

5. Obtain a cost estimate

The model can also further be imporved to:

1. Send assessment to insurance carrier

2. Print documentation

## Challenges

1. Computer Vision is still a field of research and not developed enough to handle the quality images of modular phone cameras. Angle, lighting, and resolution are factors that may cause a huge disturbance in image classification.

2. The claims for the settlement of car insurance need to be just about perfect so that the customer is not frauded in this whole process. Such models will need to be trained on some huge datasets, which are very hard to obtain.

3. Running such heavy datasets to ensure complete accuracy would bring forth a hardware restriction; therefore, storing, training, and deploying such heavy data through the cloud would take up very expensive architecture.

4. While the computer can avoid human errors, there are often situations that would require such a model to flag for human intervention.

5. Systems running on the Cloud, especially those dealing with monetary data, are also highly susceptible to cyber risks and need to have highly structured frameworks to ensure customer data security.

6. There does need to be some degree of manual control and filter to prevent flooding with fraudulent insurance claims.

## Model Architecture and Pipeline

Our system architecture is built around the following modules:

1. User Input: User submits image of a car.

2. Gate 1: Checks to ensure the submitted image is a car or not.

3. Gate 2: Checks to ensure the submitted image of car is damaged or not.

4. Location Assessment: Tests image against the pre-trained model to locate damage.

5. Severity Assessment: Tests image against pre-trained models to determine the severity of damage.

6. Price Prediction Assessment: Tests image against pre-trained models to predict the price of damage part.

7. Results: The results are sent back to the user and third party.

## Tools and Frameworks Used

Data Set Collection:

1. Google Images – data source

2. Kaggle Image Dataset – data source

3. Import.io – online web data scraper

Model Development:

1. TensorFlow and Keras – Deep Learning Library

2. NumPy – Scientific numerical calculations library

3. Scikit-learn – Machine learning algorithms tools

Web Development:

1. Flask – Python web framework

2. Bootstrap – HTML, CSS, JavaScript framework

Development Environment:

1. PyCharm IDE – Python program development environment

2. Jupyter Notebooks – web application for interactive data science and scientific computing

3. Anaconda Virtual Environments – python virtual environment application

Libraries Used:

1. numpy

2. pandas

3. matplotlib

4. sklearn

5. seaborn

6. pickle

## Improving The Model

1. With a wider range of data set featuring multiple components of the car, the model can also be trained to identify what components are damaged, also classifying the varying degree of damage of each.

2. With a highly expansive dataset containing the make, model, year of the car and the possible cost estimates for the varying degrees of damage, the model can also predict the value for the user, before he submits the more advanced and detailed assessment for evaluation.

3. Using more secure and durable hardware, the entire system can be built on the Cloud to run remotely and from the user’s cellular device itself.

4. The application can also be updated to recommend the user of policies pertaining to the specific accounts and other insurance benefits.

## Demo

<img src="/Demo.jpeg" width ="900" height ="507"/>

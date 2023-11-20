# GRLib: An Open-Source Hand Gesture Detection and Recognition Python Library
This repository houses the GRLib, a Python library for hand gesture detection and recognition. GRLib is capable of identifying both static and dynamic hand gestures using a combination of image augmentations, MediaPipe Hands and other algorithms.

## Installation
`pip install grlib`

## Usage
Use the scripts provided to detect and recognize hand gestures in real-time using an RGB camera.
Detailed instructions and examples are included in the script headers.

## Structure
 * `examples/`: Sample scripts demonstrating the library's usage.
 * `src/grlib/feature_extraction/`: Classes to process the images.
 * `src/grlib/load_data/`: Classes to load the datasets.
 * `src/grlib/dynamic_detector.py`: The class for dynamic gesture recognition.

## Sample results
Using this library, the authors produce the following results in comparison to [MediaPipe Solution](https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md):

ASL            |  HaGRID | Kenyan sign language
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://github.com/mikhail-vlasenko/grlib/assets/27450370/0fe5cf62-f94c-477c-9de5-0a9e1a9b48f1) | ![image](https://github.com/mikhail-vlasenko/grlib/assets/27450370/244f858a-fc27-4433-86a9-c560c9e8543f) | ![image](https://github.com/mikhail-vlasenko/grlib/assets/27450370/474125dd-d12d-4955-b2ee-7e7f94cc5028)

## Contributing
Contributions to GRLib are welcome and we will actively review PRs and answer issues.

## Authors
Mikhail Vlasenko and Jan Warchocki

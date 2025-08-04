---

# Convolutional Neural Networks Course Notebooks

This repository contains Jupyter notebooks from the ["Convolutional Neural Networks" course](https://www.coursera.org/learn/convolutional-neural-networks) offered by deeplearning.ai on Coursera. The notebooks were prepared by the deeplearning.ai team, and I have completed the exercises within them as part of my learning journey in this course.

## Contents

The repository currently includes the following notebooks:

- **Autonomous_driving_application_Car_detection.ipynb**: This notebook focuses on object detection using the YOLO (You Only Look Once) model. It covers:
  - Detecting cars in images from a car detection dataset.
  - Implementing non-max suppression to improve detection accuracy.
  - Calculating intersection over union (IoU) for evaluating overlapping boxes.
  - Working with bounding boxes for image annotation.

- **Image_segmentation_Unet_v2.ipynb**: This notebook is about semantic image segmentation using the U-Net architecture. It includes:
  - Building a U-Net model for precise pixel-wise classification.
  - Understanding the differences between a regular CNN and U-Net, including skip connections.
  - Applying the model to the CARLA self-driving car dataset.
  - Using sparse categorical crossentropy for pixel-wise prediction.

*Additional supporting files, such as datasets, model weights, and utility scripts, will be added as they are completed and uploaded.*

## How to Use

To run these notebooks, you will need to have Jupyter installed on your machine or use an online platform like Google Colab. Ensure you have the following libraries installed:

- TensorFlow
- NumPy
- Matplotlib
- Pandas
- Other dependencies as specified in the notebooks (e.g., `imageio`, `PIL`, or specific utilities like `yad2k` for the YOLO notebook)

You can install the core dependencies using:

```bash
pip install tensorflow numpy matplotlib pandas
```

Place the notebooks and any supporting files in the same directory, then open the notebooks in Jupyter or Colab to execute them. For specific instructions, refer to the notebooks' code cells, which include import statements and preprocessing steps.

## Note on Solutions

The notebooks in this repository contain completed exercises with my solutions. If you are currently taking the course, I recommend attempting the exercises on your own before reviewing the solutions to maximize your learning experience.

## Prerequisites

To fully benefit from these notebooks, it is recommended to have:

- A basic understanding of machine learning concepts, particularly neural networks and convolutional neural networks.
- Proficiency in Python programming.
- Familiarity with TensorFlow or other deep learning frameworks.

## Dataset Information

The datasets used in these notebooks are provided as part of the course materials:
- For the car detection notebook, the dataset is provided by [drive.ai](https://www.drive.ai/) under a Creative Commons Attribution 4.0 International License.
- For the image segmentation notebook, the CARLA self-driving car dataset is used, sourced from the course materials.

## License and Copyright

The notebooks and course materials are copyrighted by deeplearning.ai. This repository is intended for educational purposes only, showcasing my completion of the course exercises. Please respect the intellectual property rights of the original creators and refer to the official course for the complete, unmodified materials.

## Course Information

For more details about the course, including its syllabus and enrollment options, visit the [Convolutional Neural Networks course page on Coursera](https://www.coursera.org/learn/convolutional-neural-networks).

---

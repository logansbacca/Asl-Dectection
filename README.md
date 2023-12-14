# Custom Sign Detector using OpenCV and TensorFlow

A custom sign detector built using OpenCV and TensorFlow for real-time translation of American Sign Language gestures utilizing the Tensorflow Object Detection API.

## Getting Started

These instructions will guide you to set up the project on your local machine for development and testing purposes.

### Prerequisites

- Conda 23.11.0
- Python 3.9.13

### Installing

To install the necessary dependencies, follow these steps:

1. Create a virtual environment with Conda:

    ```bash
    conda create -n myenv python=3.9.13
    ```

2. Activate the created virtual environment:

    ```bash
    conda activate myenv
    ```

3. Install required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Running

Follow these steps to run the project:

1. Run the capture script:

    ```bash
    python3 capture.py
    ```

2. Move images from individual folders to the "All" folder.

3. In the label image folder, run:

    ```bash
    python3 LABELIMAG.PY
    ```

4. Label the images.

5. From the "ts" folder, run:

    ```bash
    python3 setup.py
    ```

6. Paste the training command inside your terminal to begin training the model.

7. Once the training has completed, run:

    ```bash
    python3 test.py
    ```

## License

This project is licensed under the GNU General Public License v3.0 License.

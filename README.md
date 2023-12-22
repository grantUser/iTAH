# iTAH

This Python script uses a pre-trained deep learning model to perform real-time head detection using your webcam. The script displays the webcam feed in a window and updates the window title based on whether a head is detected or not.

## Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Keras (`pip install keras`)

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/grantUser/iTAH.git
    cd iTAH
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:

    ```bash
    python main.py
    ```

4. Press `Esc` key to exit the application.

## Configuration

- The script uses a pre-trained Keras model (`head.h5`) for head detection.
- The labels for the classes are stored in a file (`labels.txt`).

## Customization

- You can modify the window size by changing `window_width` and `window_height` in the script.
- Adjust the paths to the model and label files if they are not in the same directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# OpenCV Zoo Model Cycling Demo

This project demonstrates a cycling demo of various computer vision models using OpenCV. The demo allows users to switch between different models and pause/resume the display using keyboard inputs. The models have been sampled from github.com/opencv/opencv_zoo

## Prerequisites

- Python 3.x
- OpenCV 4.10.0 or later
- Required Python packages (install using `requirements.txt` if available)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd opencv_zoo_demo
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. Controls:
   - **Space**: Toggle between pause and resume.
   - **ESC**: Quit the demo.
   - **Right Arrow**: Switch to the next model.
   - **Left Arrow**: Switch to the previous model.

3. The demo will automatically switch models every 15 seconds if not paused.

## Models

The demo supports the following models:
- Handpose Detection
- Edge Detection
- Object Detection
- Facial Expression Recognition
- Object Tracking

## Customization

- You can customize the models by editing the `model_modes.txt` file. Each line should contain a model name, and lines starting with `#` are treated as comments.

## Troubleshooting

- Ensure your OpenCV version is 4.10.0 or later.

## License

This project is licensed under the Apache-2.0 license. Each model has its own license. Checkout github.com/opencv/opencv_zoo for details.
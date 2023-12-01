# Car Counter using YOLO

This project implements a car counter using the YOLO (You Only Look Once) real-time object detection system.

## Getting Started

### Prerequisites

To run this project, you'll need to have Python installed on your local machine.
Note: I used python 3.9.12

### Installation

1. Clone the repository
    ```
    git clone https://github.com/A-Alviento/car-counter-yolo
    ```
   
2. Navigate into the cloned repository directory
    ```
    cd car-counter-yolo
    ```
   
3. A virtual environment is recommended to keep dependencies required by different projects separate. 

4. Install the required dependencies
    ```
    pip install -r requirements.txt
    ```

## Running the Tests

After installing the dependencies, you can verify the installation by running the following scripts:

- To verify if everything is working fine:
    ```
    python testing-yolo.py
    ```

- To test the model with webcam:
    ```
    python testing-yolo-webcam.py
    ```
  
## Run the Project

After verifying the installation, you can now run the main project script:
    ```
    python car-counter.py
    ```

## Acknowledgements

* This project is inspired by and follows the tutorial from [this YouTube video](https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=90s).

# SyncNet Evaluation (Lip Sync) Pipeline for Talking Head videos

To evaluate the lip-sync quality of Talking-Head videos using SyncNet, which measures the synchronization between the generated videos and the reference audio, follow these steps: 

## Setup Instructions

1. **Clone the SyncNet Repository:**

    First, clone the SyncNet repository from GitHub:

    ```bash
    git clone https://github.com/joonson/syncnet_python.git
    ```

2. **Set up the environment:**

    It is recommended to set up a separate virtual environment for running the evaluation to avoid conflicts between SyncNet and other projects. To install the necessary dependencies and download the pre-trained models, follow these steps:

    ```bash
    cd syncnet_python
    pip install -r requirements.txt
    sh download_model.sh
    ```

3. **Prepare the Video Data:**

    Organize your video data by placing all the `.mp4` files in a single directory, following this folder structure:

    ```
    video data root (Folder containing all videos)
    ├── All .mp4 files
    ```

    Example path: `/path/to/video/data/root`

## Running the Evaluation Scripts

1. **Navigate to the Evaluation Directory:**

    Move back to the `syncnet_python` directory:

    ```bash
    cd /home/host/pegah/evl/syncnet_python
    ```

2. **Copy Evaluation Scripts:**

    Copy the provided evaluation scripts to the cloned repository:

    ```

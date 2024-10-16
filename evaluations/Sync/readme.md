```markdown
# SyncNet Evaluation (Lip Sync) Pipeline for Talking Head Videos

This pipeline evaluates the lip-sync quality of Talking-Head videos using SyncNet. The evaluation measures synchronization between generated videos and reference audio.

## Setup Instructions

1. **Clone the SyncNet Repository:**

    First, clone the SyncNet repository from GitHub:

    ```bash
    git clone https://github.com/joonson/syncnet_python.git
    ```

2. **Set Up the Environment:**

    It is recommended to create a virtual environment to isolate dependencies:

    ```bash
    cd syncnet_python
    pip install -r requirements.txt
    sh download_model.sh
    ```

3. **Copy Wav2Lip Evaluation Scripts:**

    Copy the evaluation scripts from the [scores_LSE folder](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation/scores_LSE) (include all `.py` and `.sh` files) of the Wav2Lip repository into the `syncnet_python/` directory of your cloned repository.

4. **Prepare the Video Data:**

    Organize all your video files in a single directory with `.mp4` files structured as follows:

    ```
    video_data_root/
    ├── video1.mp4
    ├── video2.mp4
    └── video3.mp4
    ```

    Example path: `/path/to/video/data/root`

## Running the Evaluation Scripts

1. **Navigate to the Evaluation Directory:**

    Move to the `syncnet_python` directory:

    ```bash
    cd /path/to/syncnet_python
    ```

2. **Run the Evaluation Pipeline:**

    Use the following Python script to evaluate all `.mp4` files in the specified directory. The script runs the pipeline for each video and calculates lip-sync scores.

    ```python
    import os
    import subprocess

    # Define the directory containing your video files
    video_dir = r'/path/to/video/data/root'

    # Initialize or clear the output file
    with open('all_scores.txt', 'w') as f:
        pass  # This will clear the file

    # Get the list of all .mp4 files in the directory
    filenames = [file for file in os.listdir(video_dir) if file.endswith('.mp4')]

    # Loop through each file and run the necessary Python scripts
    for eachfile in filenames:
        videofile_path = os.path.join(video_dir, eachfile)
        
        # Run the pipeline script
        subprocess.run(['python', 'run_pipeline.py', '--videofile', videofile_path, '--reference', 'wav2lip', '--data_dir', 'tmp_dir'])

        # Capture the output of the scoring script
        result = subprocess.run(
            ['python', 'calculate_scores_real_videos.py', '--videofile', videofile_path, '--reference', 'wav2lip', '--data_dir', 'tmp_dir'],
            capture_output=True, text=True
        )

        # Print the output for debugging
        print(f"Output for {eachfile}:")
        print(result.stdout)

        # Write the results to the file
        with open('all_scores.txt', 'a') as f:
            f.write(f'Video: {eachfile}\n')
            f.write(result.stdout)
            f.write('-' * 50 + '\n')
    ```

3. **Results:**

    The scores for each video will be saved in the `all_scores.txt` file. Each score represents the lip-sync accuracy for the corresponding video.

## Notes

- Make sure to install all necessary dependencies listed in the `requirements.txt` file.
- The script assumes that the `run_pipeline.py` and `calculate_scores_real_videos.py` scripts are set up correctly in the SyncNet directory.
- Ensure that the reference model (e.g., Wav2Lip) is correctly referenced during execution.
```

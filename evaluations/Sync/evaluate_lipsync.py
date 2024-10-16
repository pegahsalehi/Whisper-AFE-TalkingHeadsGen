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

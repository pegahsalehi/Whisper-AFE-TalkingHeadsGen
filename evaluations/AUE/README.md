# Action Units Error (AUE)

The `.csv` file should contain the facial landmarks detected using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). Installing OpenFace from scratch can be difficult due to dependency issues, so it's recommended to use Docker for a quicker setup. Follow these steps to get started with OpenFace using Docker:

- Ensure that Docker is installed on your machine before proceeding.

## Steps

1. **Run the OpenFace Docker container:**

    ```bash
    docker run -it --rm algebr/openface:latest
    ```

2. **Find the container ID by executing the following command in a separate terminal:**

    ```bash
    docker ps
    ```

    *(For example, the container ID might be `e604790827a8`)*

3. **Transfer any videos you want to the desired folder:**

4. **Edit the `VIDEO_DIR` and `CONTAINER_ID` in the `facelandmark.sh` file, then run the script:**

    ```bash
    sh facelandmark.sh
    ```

5. **The outputs will be saved as `[your_video_name].csv` in the `processed` directory.**

6. **Run the `aue.py` file to calculate the AU error for both lower and upper AUs:**

    ```bash
    python aue.py
    ```

---



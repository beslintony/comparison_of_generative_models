**Creating TFRecords from ImageNet Dataset**

## Prerequisites

- **Download Data:**
  - Obtain the ImageNet dataset from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) [1].
  
- **Validation Data Preparation:**
  - Utilize the `valprep.sh` script from the [ImageNet2012-download](https://github.com/DoranLyong/ImageNet2012-download) repository to prepare validation data.
    ```bash
    cd /path/to/imagenet/ILSVRC/Data/CLS-LOC/val/
    wget https://raw.githubusercontent.com/DoranLyong/ImageNet2012-download/main/valprep.sh
    chmod +x valprep.sh
    ./valprep.sh
    ```
    Replace `/path/to/imagenet/ILSVRC/Data/CLS-LOC/val/` with the path to the validation images from the dataset.
  
## TFRecord Creation

- **Install Required Packages:**
  - Make sure you have TensorFlow installed.
    ```bash
    pip install tensorflow tqdm
    ```

- **Modify Configuration:**
  - Open `tf-train.py` and `tf-val.py` in a text editor.
  - Adjust the image size and provide correct paths and adapt it as you wish.

- **Run the Script:**
  - Execute the script to create TFRecord files for training and validation.
    ```bash
    python tf-train.py
    python tf-val.py
    ```

- **Use proper path:**
  - After executing both the scripts to create TFRecord files for training and validation, create a folder in the root called `imagenet` and store both `train_data.tfrecord` and `val_data.tfrecord` files.
    ```bash
    mkdir imagenet
    mv /path/to/imagenet/ILSVRC/some_where/train_data.tfrecord /imagenet/train_data.tfrecord
    mv /path/to/imagenet/ILSVRC/some_where/val_data.tfrecord /imagenet/val_data.tfrecord
    ```

  Replace /path/to/imagenet/ILSVRC/some_where/` with the path to the created tfrecord files.

## Important Notes

- Ensure you have sufficient disk space as the ImageNet dataset is large.

- Validate that the ImageNet dataset is properly organized in the specified directories.

- Adjust the image size to match the requirements of your model.

- Verify TensorFlow installation and compatibility.

- For any issues, refer to the TensorFlow documentation or community forums.

## Reference

  - [1] ImageNet Object Localization Challenge. (2018).
    ```
    @misc{imagenet-object-localization-challenge,
        author = {Addison Howard, Eunbyung Park, Wendy Kan},
        title = {ImageNet Object Localization Challenge},
        publisher = {Kaggle},
        year = {2018},
        url = {https://kaggle.com/competitions/imagenet-object-localization-challenge}
    }
    ```


# Instructions for Downloading and Processing Libero Data

## Setting up Libero
1. Install Libero following [these](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master?tab=readme-ov-file#installtion) instructions -- no need to install torch, torchvision and torchaudio so ignore that line. You should be able to do it on top of an existing octo environment (it worked for me).

2. Download the data as described here: https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master?tab=readme-ov-file#datasets

## Converting Libero data into RLDS format
1. The folders ```rlds_dataset_builder/libero90``` and ```rlds_dataset_builder/libero90_horizon``` correspond to trajectory and horizon level datasets respectively.

2. Change the ```DATA_DIR``` in the file ```rlds_dataset_builder/libero90/libero90.py``` to the directory where the original datasets are downloaded.

3. Run the conversion script,
    ```
    cd rlds_dataset_builder/libero90
    tfds build --overwrite
    ```
    The datasets would be written to ```~/tensorflow_datasets/``` in the format acceptable by octo. 

## Evaluating Octo on Libero
Will add instructions soon.
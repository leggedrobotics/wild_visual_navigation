# Dataset generation

1. Dataset is configured in **wild_visual_navigation/utils/dataset_info.py**
2. Run extract_images_and_labels.py (can be also done for multiple mission with **extract_all.py**)
   - Currently this is done by storing a binary mask and images as .pt files (maybe change to png for storage)
3. Validate if images and labels are generated correctly with **validate_extract_images_and_labels.py**
   - This script will remove images if no label is available and the other way around
   - The stopping early times should be quite small within the seconds
4. Create lists with the training and train/val/test images: **create_train_val_test_lists.py**
5. Convert the correct .pt files to .png such that you can upload them for the test set to segments.ai **convert_test_images_for_labelling.py**
6. Label them online
7. Fetch the results using **download_bitmaps_from_segments_ai.py**
8. Extract the features segments and graph from the image **extract_features_for_dataset**
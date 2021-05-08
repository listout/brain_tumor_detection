# Brain Tumor Detection

## Data Augmentation

Since we have **Data Imbalance** i.e. we have 253 images with 155 belonging to "yes" class and 98 belonging to "no" class, we are using a *Data Augmentation*. We take a MRI image and perform various image enhancements such as rotate, mirror and flip to get more number of images

The `ImageDataGenerator` will generate batches of tensor images data from real-time data augmentation. The data will be looped over.

The function `data_augmentation` will take:

- `file_dir`: the directory of the images
- `num_samples`: number of augmented data that will be generated per original image
- `save_to`: directory to save the augmented images to

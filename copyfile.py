import tempfile
from distutils.dir_util import copy_tree

# copy files in temporary directory
# OS independent
temp_dataset_dir = tempfile.gettempdir() + '/dataset'
print('Temporary Dataset dir:', temp_dataset_dir)

# dataset source directory
dataset_source_directory = 'dataset'

# training and testing temporary directory
training_dataset_directory = temp_dataset_dir + '/training/'
testing_dataset_directory = temp_dataset_dir + '/testing/'
print('Training Dataset directory', training_dataset_directory)
print('Testing Dataset directory', testing_dataset_directory)

# copy the dataset to temporary directory
copy_tree(dataset_source_directory, training_dataset_directory)
copy_tree(dataset_source_directory, testing_dataset_directory)

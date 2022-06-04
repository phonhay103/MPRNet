from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest

def get_training_data(rgb_dir, img_options):
    return DataLoaderTrain(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    return DataLoaderTest(rgb_dir, img_options)

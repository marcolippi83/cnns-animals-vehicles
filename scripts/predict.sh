PYTHON=python3.5

test_images="../data/test_images"

# Predictions of densenet201 fine-tuned on the binary task animals vs. vehicles
out_file="out_densenet201.txt"
$PYTHON densenet201_test.py ../lists/LIST_TEST_IMAGES_WITH_LABELS.txt $test_images ../models/densenet201_vehicles_animals.h5 ../models/X_train_stats.txt $out_file

# Predictions of resnet50 fine-tuned on the binary task animals vs. vehicles
out_file="out_resnet50.txt"
$PYTHON resnet50_test.py ../lists/LIST_TEST_IMAGES_WITH_LABELS.txt $test_images/ ../models/resnet50_vehicles_animals.h5 ../models/X_train_stats.txt $out_file



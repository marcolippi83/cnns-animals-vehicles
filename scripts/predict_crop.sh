PYTHON=python3.5

test_images="../data/test_images"

# Predictions of densenet201 fine-tuned on the binary task animals vs. vehicles
out_file="out_densenet201_crop.txt"
$PYTHON densenet201_test.py ../lists/LIST_TEST_IMAGES_WITH_LABELS.txt $test_images densenet201_vehicles_animals_crop.h5 X_train_stats_densenet201_crop.txt $out_file

# Predictions of resnet50 fine-tuned on the binary task animals vs. vehicles
out_file="out_resnet50_crop.txt"
$PYTHON resnet50_test.py ../lists/LIST_TEST_IMAGES_WITH_LABELS.txt $test_images resnet50_vehicles_animals_crop.h5 X_train_stats_resnet50_crop.txt $out_file



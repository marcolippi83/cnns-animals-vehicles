python3.5 densenet201_finetune.py ../lists/LIST_TRAIN_CROP_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/crop/ .
mv densenet201_vehicles_animals.h5 densenet201_vehicles_animals_crop.h5
mv X_train_stats_densenet201.txt X_train_stats_densenet201_crop.txt

python3.5 resnet50_finetune.py ../lists/LIST_TRAIN_CROP_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/crop/ .
mv resnet50_vehicles_animals.h5 resnet50_vehicles_animals_crop.h5
mv X_train_stats_resnet50.txt X_train_stats_resnet50_crop.txt


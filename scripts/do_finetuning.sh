# Fine-tuning densenet201 with 3,000 regular images
python3.5 densenet201_finetune.py ../lists/LIST_TRAIN_ALL_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/ . densenet201_vehicles_animals.h5 X_train_stats_densenet201.txt

# Fine-tuning resnet50 with 3,000 regular images
python3.5 resnet50_finetune.py ../lists/LIST_TRAIN_ALL_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/ . resnet50_vehicles_animals.h5 X_train_stats_resnet50.txt

# Fine-tuning densenet201 with 30,000 HPF images
python3.5 densenet201_finetune.py ../lists/LIST_TRAIN_HPF_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/HPF/ . densenet201_vehicles_animals_hpf.h5 X_train_stats_densenet201_hpf.txt

# Fine-tuning resnet50 with 30,000 HPF images
python3.5 resnet50_finetune.py ../lists/LIST_TRAIN_HPF_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/HPF/ . resnet50_vehicles_animals_hpf.h5 X_train_stats_resnet50_hpf.txt

# Fine-tuning densenet201 with 30,000 HPF images
python3.5 densenet201_finetune.py ../lists/LIST_TRAIN_HPF_SMALL_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/HPF/ . densenet201_vehicles_animals_hpf_small.h5 X_train_stats_densenet201_hpf_small.txt

# Fine-tuning resnet50 with 3,000 HPF images
python3.5 resnet50_finetune.py ../lists/LIST_TRAIN_HPF_SMALL_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/HPF/ . resnet50_vehicles_animals_hpf_small.h5 X_train_stats_resnet50_hpf_small.txt

# Fine-tuning densenet201 with 30,000 crop images
python3.5 densenet201_finetune.py ../lists/LIST_TRAIN_CROP_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/crop/ . densenet201_vehicles_animals_crop.h5 X_train_stats_densenet201_crop.txt

# Fine-tuning resnet50 with 30,000 crop images
python3.5 resnet50_finetune.py ../lists/LIST_TRAIN_CROP_WITH_LABELS.txt ~/datasets/ImageNet/AnimalsVehicles/crop/ . resnet50_vehicles_animals_crop.h5 X_train_stats_resnet50_crop.txt





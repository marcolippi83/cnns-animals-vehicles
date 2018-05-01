test_images="../data/test_images"
out_file="out_original_resnet50.txt"
python3.5 resnet50_original.py $test_images > $out_file

test_images="../data/test_images"
out_file="out_original_densenet201.txt"
python3.5 densenet201_original.py $test_images > $out_file


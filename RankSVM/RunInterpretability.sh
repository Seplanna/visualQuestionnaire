data_set=../CycleGAN_shoes/Toy/shoes_boots_heels_white_black/
result_dir=../CycleGAN_shoes/Toy/My_interpretability/
svm=../CycleGAN_shoes/Toy/My/

python PCA.py --My=True --data_path=$data_set --n_features_embedding=3 --result_dir=$result_dir --svm=$svm


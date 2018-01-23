step=$1

python BuildTheNextSetForClustering.py --create_real_pairs=True --n_features_embedding=10 --step=$step --result_dir=../CycleGAN_shoes/Experiment/

python svmRank.py $step

step1=$(($step+1))
step_step1=$(($step*2))
step_step1+=_
step_step1+=$(($step*2+1))

python BuildTheNextSetForClustering.py --check_ranking_for_the_clusters=True --svm_model=../CycleGAN_shoes/Experiment/My_ --result_dir=../CycleGAN_shoes/Experiment/$step_step1/ --data_set=../CycleGAN_shoes/Experiment/ --step=$step1 --n_features_embedding=$step

python BuildTheNextSetForClustering.py --take_pictures_from_first_bin=True --data_path=../CycleGAN_shoes/Experiment/$step_step1/ --result_dir=../CycleGAN_shoes/Experiment/$step_step1/result/ --n_bins=3 --step=$step --n_features_embedding=10

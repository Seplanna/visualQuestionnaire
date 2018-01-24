#1.Get 5000 random pictures
#2.PCA with GIST
#3.PCA with Embedding
#4.Mine

#python runEvaluation.py --save_random_examples=True --data_set=../CycleGAN_shoes/DATA_ALL_LATENT1/ --result_dir=../CycleGAN_shoes/Random_5000/ --n_examples=5000
python PCA.py --PCA_GIST=True --data_path=../CycleGAN_shoes/Random_5000/ --n_features_embedding=960 --new_features_embedding=5 --result_dir=../CycleGAN_shoes/PCA_5000/
#python PCA.py --PCA_AutoEncoder=True --data_path=../CycleGAN_shoes/Random_5000/ --n_features_embedding=10 --new_features_embedding=5 --result_dir=../CycleGAN_shoes/AutoEncoder_5000/

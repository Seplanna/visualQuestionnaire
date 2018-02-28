#1. Recieve








if FLAGS.take_pictures_from_first_bin:
    print("AAAAAAA")
    data_path=FLAGS.data_path
    result_dir=FLAGS.result_dir
    step = FLAGS.step

    n_bins=FLAGS.n_bins
    bins = DevideFeatureIntoBins(data_path, n_bins, 0)
    data = []
    for root, subdirs, files in os.walk(data_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg'):
                data += [os.path.join(root, f)]
    print(len(data))
    os.makedirs(result_dir + "_0/")
    for d in data:
        basename = os.path.basename(d)
        basename = basename.split("_")
        if (float(basename[0]) < bins[0]):
            subprocess.call(['cp', d, result_dir + "_0/" + "_".join(basename[1:])])

    os.makedirs(result_dir + "_1/")
    for d in data:
        basename = os.path.basename(d)
        basename = basename.split("_")
        if (float(basename[0]) > bins[1]):
            subprocess.call(['cp', d, result_dir + "_1/" + "_".join(basename[1:])])

    images_indexes1, clusters1 = ClusterData(GetData(result_dir + "_0/"), FLAGS.n_features_embedding, 20, result_dir + "_0_test/", FLAGS.step)
    images_indexes2, clusters2 = ClusterData(GetData(result_dir + "_1/"), FLAGS.n_features_embedding, 20, result_dir + "_1_test/", FLAGS.step)
    print("VARIETION")
    print(np.mean(clusters1.cluster_centers_, axis=0), np.mean(clusters2.cluster_centers_, axis=0))
    print(np.linalg.norm(clusters1.cluster_centers_ - np.mean(clusters1.cluster_centers_, axis=0)),
          np.linalg.norm(clusters2.cluster_centers_ - np.mean(clusters2.cluster_centers_, axis=0)))
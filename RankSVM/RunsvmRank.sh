for i in {1..5}
do
  mkdir result$i
  python svmRank.py $i
done

AFGBStream


AFGBStream/
├── DPC.py
├── FuzzyClustering_no_random.py
├── GBSW.py  # Core algorithm implementation
├── datasets/
├── granular_ball.py
├── HyperballClustering.py
├── main.py
└──MicroCluster.py



Requirements

python == 3.7
scikit-learn == 1.4.2
scipy == 1.13.1
numpy == 1.26.4
pandas == 2.2.2


Start Examples

python main.py --dataset RBF3_40000.csv  --window_size 300 --step_size=250 --lam 0.2 --threshold 0.5 

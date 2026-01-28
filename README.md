# AFGBStream

## Project Structure

```
AFGBStream/
├── DPC.py
├── FuzzyClustering_no_random.py
├── GBSW.py  # Core algorithm implementation
├── datasets/
├── granular_ball.py
├── HyperballClustering.py
├── main.py
└──MicroCluster.py
```

## Requirements

```bash
python == 3.7
scikit-learn == 1.4.2
scipy == 1.13.1
numpy == 1.26.4
pandas == 2.2.2
```

## Start Examples

```bash
python main.py --dataset RBF3_40000.csv  --window_size 300 --step_size=250 --lam 0.2 --threshold 0.5
```

For more related researches, please visit my homepage: https://dumingjing.github.io/. For data and discussion, please message Mingjing Du (杜明晶@江苏师范大学): [dumj@jsnu.edu.cn](mailto:dumj@jsnu.edu.cn).

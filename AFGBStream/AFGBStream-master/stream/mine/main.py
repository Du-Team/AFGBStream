import pandas as pd
from GBSW import start
if __name__ == '__main__':
    dataset = "Benchmark1_5500.csv"
    dataset_path = ""
    csv1 = pd.read_csv(dataset_path + dataset, header=None)
    plot_evaluate_flag =False#
    start(csv1, dataset, plot_evaluate_flag)


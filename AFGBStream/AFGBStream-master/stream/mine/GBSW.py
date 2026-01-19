

import sys
from sklearn.decomposition import PCA
from HyperballClustering import *
from MicroCluster import MicroBall
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from DPC import get_cluster_DPC
from granular_ball import GranularBall

from collections import deque




class MBStream:


    def __init__(self, data, dataset, plot_evaluate_flag, lam, threshold, window_size=300, step_size=250):
        self.datasetName = dataset
        self.data = self.normalized(data)
        self.window_size = window_size
        self.step_size = step_size
        self.data_window = deque(maxlen=window_size)
        self.lam = lam
        self.threshold = threshold
        self.trueLabel = list(map(int, data.values[:, -1]))
        self.micro_balls=[]
        self.micro_balls = self.init_v1(plot_evaluate_flag)


    def normalized(self, data):
        labels = data.values[:, -1].reshape(-1, 1)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # 数据缩放
        value_to_normalize = data.values[:, :-1]
        if value_to_normalize.shape[1] > 2:
            pca = PCA(n_components=2)
            value_to_normalize = pca.fit_transform(value_to_normalize)
        value_to_normalize = min_max_scaler.fit_transform(value_to_normalize)
        value_to_normalize = np.nan_to_num(value_to_normalize, nan=0.0, posinf=0.0, neginf=0.0)
        normalized_values = min_max_scaler.fit_transform(value_to_normalize)
        normalized_data = np.hstack((normalized_values, labels))
        value = normalized_data
        return value

    def init_v1(self, plot_evaluate_flag):
        current_window_data = []
        for x in range(0, self.window_size):
            current_window_data.append(self.data[x])
            self.data_window.append(self.data[x])  # 加入滑动窗口
        clusters, gb_list, gb_dict = self.connect_ball_DPC(self.micro_balls,np.array(current_window_data))
        init_mb_list = []
        for data in gb_list:
            mb = MicroBall(data, None)
            mb.init_weight(1, len(data))
            init_mb_list.append(mb)

        if plot_evaluate_flag:
            gb_plot(gb_dict, [], 1)

        return init_mb_list


    def get_nearest_micro_ball(self, sample, micro_balls):
        smallest_distance = sys.float_info.max
        nearest_micro_ball = None
        nearest_micro_ball_index = -1
        for i, micro_ball in enumerate(micro_balls):
            current_distance = np.linalg.norm(micro_ball.center - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_ball = micro_ball
                nearest_micro_ball_index = i
        if nearest_micro_ball is None:
            smallest_distance = None  # 设置为 None，表示未找到

        return nearest_micro_ball_index, nearest_micro_ball, smallest_distance


    def fit_predict(self, plot_evaluate_flag):
        window_index = 1
        for s in range(self.window_size, len(self.data) + 1, self.step_size):
            window_index += 1
            old_data = self.data[s - self.window_size:s - self.step_size]
            if len(old_data) > 0:
                self.micro_balls = [mb for mb in self.micro_balls if len(mb.data) > 0]
                for mb in self.micro_balls:
                    mb.center = mb.get_center()
                    mb.radius = mb.get_radius()
                    mb.num = mb.get_num()
                    mb.DM = mb.get_dm()

                    mb.add_weight_info(window_index, len(mb.data))
                    mb.remove_data_from_weight_info(window_index, len(old_data), lam=self.lam)
                    if mb not in self.micro_balls:
                        self.micro_balls.append(mb)
            new_samples = self.data[s - self.step_size:s]
            new_samples_with_time = np.column_stack(
                (new_samples[:, :-1], np.full((new_samples.shape[0], 1), window_index)))
            self.data_window.extend(new_samples)
            current_window_data = self.data_window
            gb_list_temp = [np.array(new_samples_with_time)]
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = division_2_2(gb_list_temp)
                ball_number_new = len(gb_list_temp)
                if ball_number_new == ball_number_old:
                    break
            radius = [get_radius(gb_data[:, :-1]) for gb_data in gb_list_temp if len(gb_data) >= 2]
            radius_median = np.median(radius)
            radius_mean = np.mean(radius)
            radius = min(radius_median, radius_mean)
            max_iterations = 100
            iteration = 0
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = minimum_ball(gb_list_temp, radius)  # 缩小粒球
                ball_number_new = len(gb_list_temp)
                if ball_number_new == ball_number_old or iteration >= max_iterations:
                    break
                iteration += 1
            # 将新粒球与现有微球进行融合
            for obj in gb_list_temp:
                if len(obj) == 1:
                    continue
                gb = GranularBall(obj)
                if not self.micro_balls:
                    print("self.micro_balls is None")
                nearest_micro_ball_index, nearest_micro_ball, smallest_distance = \
                    self.get_nearest_micro_ball(gb.center, self.micro_balls)
                centers = [mb.center for mb in self.micro_balls]
                if (
                        centers is not None and
                        isinstance(centers, list) and
                        len(centers) > 0 and
                        nearest_micro_ball is not None and
                        nearest_micro_ball.center is not None and
                        isinstance(nearest_micro_ball.center, np.ndarray) and
                        nearest_micro_ball.center.size > 0
                ):

                    IRC = min((np.linalg.norm(center - nearest_micro_ball.center) for center in centers
                               if center is not nearest_micro_ball.center))
                    if smallest_distance + gb.radius >= IRC and smallest_distance + gb.radius > nearest_micro_ball.radius and smallest_distance + nearest_micro_ball.radius > gb.radius:
                        mb = MicroBall(gb.data, label=None)
                        mb.init_weight(window_index, len(gb.data))
                        self.micro_balls.append(mb)
                    else:
                        if isinstance(nearest_micro_ball, MicroBall):
                            insert = nearest_micro_ball.insert_ball(gb, 1)
                            if not insert:
                                continue
                            else:
                                del self.micro_balls[nearest_micro_ball_index]
                                self.micro_balls.extend(insert)
                else:
                    if(len(obj.data)>2):
                        self.micro_balls.append(MicroBall(obj,obj[:, 2]))

            temp = []
            for mb in self.micro_balls:
                mb.update_weight(window_index, self.lam)
                if mb.weight > self.threshold:
                    temp.append(mb)
                else:
                    mb.data = np.array([])
            self.micro_balls = temp

            # 进行聚类和结果展示
            clusters, gb_list, gb_dict = self.connect_ball_DPC( self.micro_balls,np.array(current_window_data))

            if plot_evaluate_flag:
                gb_plot(gb_dict, [], window_index)


    def connect_ball_DPC(self, current_window_data):
        self.micro_balls
        radius = []  # 汇总所有粒球半径
        gb_list_temp = []

        if self.micro_balls:
            for mb in self.micro_balls:
                if len(mb.data) >= 2:
                    radius.append(mb.radius)
                    gb_list_temp.extend(mb.data)

            filtered_data = []
            for sample in current_window_data:
                if not any(np.array_equal(sample, mb_data) for mb_data in gb_list_temp):
                    filtered_data.append(sample)
            # 转回 numpy 数组
            filtered_data = np.array(filtered_data)
            print(1)
        else:
            filtered_data= current_window_data

        if filtered_data is not None:
            gb_list_temp = [filtered_data]
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = division_2_2(gb_list_temp)
                ball_number_new = len(gb_list_temp)
                if ball_number_new == ball_number_old:
                    break
            for gb_data in gb_list_temp:
                if len(gb_data) >= 2:
                    radius.append(get_radius(gb_data[:, :-1]))
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius = max(radius_median, radius_mean)
        while 1:
            ball_number_old = len(gb_list_temp)
            gb_list_temp = normalized_ball(gb_list_temp, radius)
            ball_number_new = len(gb_list_temp)
            if ball_number_new == ball_number_old:
                break
        gb_center_list = []
        # noise
        gb_list_temp_no_noise = []
        for gb in gb_list_temp:
            # noise
            if len(gb) > 1:
                gb_center_list.append(gb[:, :-1].mean(0))
                gb_list_temp_no_noise.append(gb)
        gb_center = np.array(gb_center_list)
        # noise
        gb_list_temp = gb_list_temp_no_noise
        if len(gb_center) >= 2:
            clusters_label, n = get_cluster_DPC(gb_center)
        else:
            clusters_label = [-1] * len(gb_list_temp)
            n = 0
        clusters = {}
        gb_dict = {}
        for i in range(0, len(gb_list_temp)):
            gb_dict[i] = GB(gb_list_temp[i], clusters_label[i])
            if clusters_label[i] in clusters.keys():
                clusters[clusters_label[i]] = np.append(clusters[clusters_label[i]], gb_list_temp[i], axis=0)
            else:
                clusters[clusters_label[i]] = gb_list_temp[i]
        return clusters, gb_list_temp, gb_dict

# 自定义参数
def start(data, dataset_name, plot_evaluate_flag):
    M = MBStream(data, dataset_name, plot_evaluate_flag, lam=0.2, threshold=0.5)
    M.fit_predict(plot_evaluate_flag)
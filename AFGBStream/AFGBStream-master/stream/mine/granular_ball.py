

class GranularBall:
    def __init__(self, data):
        self.data = data
        if len(self.data) == 0:
            self.center = 0  # 如果数据为空，中心为0
            self.radius = 0  # 半径为0
        else:
            self.center = self.data[:, :-1].mean(0)  # 计算中心
            self.radius = self.get_radius()  # 计算半径

    def get_radius(self):
        if len(self.data) <= 1:
            return 0  # 如果数据为空或只有一个样本，返回半径0
        return max(((self.data[:, :-1] - self.center) ** 2).sum(axis=1) ** 0.5)

import torch.nn as nn
import torch

def centerloss(feature, label, lambda_):

    # 生成分类中心
    a = torch.tensor([[0, 0], [-1, -3], [-2, -2], [-3, 0], [-2, 3], [0, 3.5], [2, 3], [3, 0], [2, -2], [1, -3]])  # 十个中心坐标
    center = nn.Parameter(a, requires_grad=True).cuda()

    # 通过label筛选出标签相同的类别
    center_class = center.index_select(dim=0, index=label.long())

    # 生成直方图，统计每个类别中个数
    count = torch.histc(label, bins=int(max(label).item()+1), min=0, max=int(max(label).item()))

    # 每个类别对应上其类中元素个数
    count_class = count.index_select(dim=0, index=label.long())

    # 计算损失
    loss = lambda_/2*(torch.mean(torch.div(torch.sum(torch.pow(feature-center_class, 2), dim=1), count_class)))

    return loss


if __name__ == "__main__":
    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32).cuda()
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32).cuda()
    centerloss(data, label, 2)
    # print(data[label == 1])

    a = torch.tensor([[0, 0], [-1, -3], [-2, -2], [-3, 0], [-2, 3], [0, 3.5], [2, 3], [3, 0], [-2, -2], [1, -3]])

import numpy as np

def generate_matrix(lower_bound, upper_bound, total_sum=766):
    size = lower_bound.shape[0]
    matrix = np.zeros((size, size), dtype=int)

    for i in range(size):
        # 计算每个非对角线元素的比例范围
        proportions = []
        for j in range(size):
            if i != j:
                min_prop = lower_bound[i][j] / total_sum
                max_prop = upper_bound[i][j] / total_sum
                proportions.append((min_prop, max_prop))
        
        # 随机生成符合比例范围的比例值
        random_proportions = [np.random.uniform(min_prop, max_prop) for min_prop, max_prop in proportions]
        
        # 归一化比例值使得总和为1
        sum_proportions = sum(random_proportions)
        normalized_proportions = [prop / sum_proportions for prop in random_proportions]
        
        # 按照归一化的比例分配元素值
        k = 0
        remaining_sum = total_sum
        for j in range(size):
            if i != j:
                if j == size - 1 and i != size - 1:
                    matrix[i][j] = remaining_sum
                elif j == size - 2 and i == size - 1:
                    matrix[i][j] = remaining_sum
                else:
                    value = int(normalized_proportions[k] * total_sum)
                    matrix[i][j] = max(lower_bound[i][j], min(value, upper_bound[i][j]))
                    remaining_sum -= matrix[i][j]
                    k += 1

    # 确保对角线为0
    for i in range(size):
        matrix[i][i] = 0

    return matrix

def test_metrix():

    # 检查每个元素是否在上下限范围内
    valid = True
    for i in range(len(lower_bound)):
        for j in range(len(lower_bound[i])):
            if matrix[i][j] < lower_bound[i][j] or matrix[i][j] > upper_bound[i][j]:
                valid = False
                print(f"矩阵元素[{i}][{j}] = {matrix[i][j]} 不在范围 [{lower_bound[i][j]}, {upper_bound[i][j]}] 内")

    return valid

def print_sum_of_every_row(matrix):
    for i in range(matrix.shape[0]):
        print(f"第{i}行的和为{sum(matrix[i])}")

if __name__ == "__main__":
    # 输入两个矩阵，分别为下限和上限
    lower_bound = np.array([
        [0, 176, 176, 260], 
        [176, 0, 260, 176], 
        [176, 260, 0, 176], 
        [260, 176, 176, 0]
    ])

    upper_bound = np.array([
        [0, 264, 264, 390], 
        [264, 0, 390, 264], 
        [264, 390, 0, 264], 
        [390, 264, 264, 0]
    ])

    # 设置随机种子
    np.random.seed(None)

    # 生成满足条件的矩阵
    matrix = generate_matrix(lower_bound, upper_bound)

    while not test_metrix():
        matrix = generate_matrix(lower_bound, upper_bound)
    
    print(matrix)
    print_sum_of_every_row(matrix)
    

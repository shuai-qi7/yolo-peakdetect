DATA_LENGTH = 1024  # 序列的长度

GRID_LENGTH = [8, 16, 32]  # 小中大网格长度

# 小中大网格数量
GRID_NUMBER = [DATA_LENGTH // 8, DATA_LENGTH // 16, DATA_LENGTH // 32]

PEAK_HEIGHT_MAX = 1  # 峰的最大高度

PEAK_WIDTH_MAX = 256

PEAK_NUM_MAX = 16  # 最多的峰个数

REG_MAX = 15  # box分割方式0——7对应0——PEAK_SIZE_MAX/0——PEAK_HEIGHT_MAX

if __name__ == "__main__":
    print("DATA_LENGTH:", DATA_LENGTH)
    print("GRID_LENGTH:", GRID_LENGTH)
    print("GRID_NUMBER:", GRID_NUMBER)

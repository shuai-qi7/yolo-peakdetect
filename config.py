DATA_LENGTH = 1024  # Length of the sequence

GRID_LENGTH = [8, 16, 32]  # Small, medium and large grid lengths

# Number of small, medium and large grids
GRID_NUMBER = [DATA_LENGTH // 8, DATA_LENGTH // 16, DATA_LENGTH // 32]

PEAK_HEIGHT_MAX = 1  # Maximum height of peaks

PEAK_WIDTH_MAX = 256

PEAK_NUM_MAX = 16  # Maximum number of peaks

REG_MAX = 15  # Box segmentation mode 0——7 corresponds to 0——PEAK_SIZE_MAX/0——PEAK_HEIGHT_MAX

if __name__ == "__main__":
    print("DATA_LENGTH:", DATA_LENGTH)
    print("GRID_LENGTH:", GRID_LENGTH)
    print("GRID_NUMBER:", GRID_NUMBER)
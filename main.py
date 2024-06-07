from trainer import MultiModalTrainer
from config import CONFIG_DICT
import utils as U
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    U.set_seed(2024)
    time = datetime.now().strftime("%Y年%m月%d日%H时%M分%S秒")

    # MultiModalTrainer(CONFIG_DICT[dataset], time).experiment()
    MultiModalTrainer(CONFIG_DICT[dataset], time).train()

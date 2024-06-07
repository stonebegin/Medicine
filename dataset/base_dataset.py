import os
import json
from typing import *
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import utils as U


class BaseDataset:
    def __init__(self, config: Dict, modal: str) -> None:
        r"""
        :config 配置文件
        :modal 模态名称
        """
        self.dataset = config["dataset"]
        self.dataset_dir = config["dataset_dir"]
        self.num_workers = config["num_workers"]

        # data
        self.__y__ = {"failure_type": [], "root_cause": []}
        self.__X__ = []

        self.services = config["services"].split(" ")
        self.instances = config["instances"].split(" ")
        self.failures = config["failures"].split(" ")

        self.__data_path__ = os.path.join(
            config["save_dir"], f"{self.dataset}_{modal}_tmp.json"
        )
        self.__config__ = config
        U.check(self.__data_path__)

    @property
    def X(self):
        return self.__X__

    @property
    def y(self):
        return self.__y__[self.__config__["label_type"]]

    def get_dataset_path(self) -> str:
        return self.__data_path__

    def load_from_tmp(self) -> Dict:
        r"""
        :return 你自己附带的appendix字典信息
        """
        with open(
            self.__data_path__,
            "r",
            encoding="utf8",
        ) as r:
            obj = json.load(r)
        # load data
        self.__X__ = obj["X"]
        self.__y__ = obj["y"]
        return obj["appendix"]

    def save_to_tmp(self, appendix: Dict = None):
        r"""
        :appendix 你要存放的额外字典信息
        """
        if appendix is None:
            appendix = {}
        with open(
            self.__data_path__,
            "w",
            encoding="utf8",
        ) as w:
            json.dump(
                {"X": self.__X__, "y": self.__y__, "appendix": appendix},
                w,
            )

    def load(self):
        r"""
        需要实现
        """
        raise NotImplementedError

    def __get_files__(self, root: str, keyword: str):
        r"""
        :root: 搜索的根目录
        :keyword：文件含有的关键词
        :return 含有关键字的文件列表['file1', 'file2']
        """
        files = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.find(keyword) != -1:
                    files.append(os.path.join(dirpath, filename))

        return files

    def __load_df__(self, file_list: List[str], is_json=False):
        r"""
        :file_list 同一 columns 的csv文件列表
        :is_json 是否为json文件
        :return Dataframe
        """
        if is_json:
            dfs = [
                pd.read_json(file, keep_default_dates=False) for file in tqdm(file_list)
            ]
        else:
            dfs = [pd.read_csv(file) for file in tqdm(file_list)]
        return pd.concat(dfs)

    def __add_by_date__(self, files, dates):
        r"""
        :files 一堆含有时间信息的文件列表
        :dates 日期
        :return 返回一个按dates排的二维列表
        >>> files = ["05-02/demo.csv", "05-03/demo.csv"]
        >>> dates = ["05-02", "05-03"]
        >>> rt = self.__add_by_date(files, dates)
        >>> # output: rt = [["05-02/demo.csv"], ["05-03/demo.csv"]]
        """
        _files = [[] for _ in dates]
        for index, date in enumerate(dates):
            for file in files:
                if file.find(date) != -1:
                    _files[index].append(file)
        return _files

    def __getitem__(self, index: int):
        return self.__X__[index], self.__y__[self.__config__["label_type"]][index]

    def __len__(self):
        return len(self.__X__)

    def __desc_date__(self, ts) -> str:
        r"""
        :ts 时间戳
        :return 时间描述
        """
        return datetime.fromtimestamp(ts).strftime("%Y年%m月%d日%H:%M:%S")

    def data_argument(self, X, y):
        r"""
        数据增强，自己实现
        """
        return X, y

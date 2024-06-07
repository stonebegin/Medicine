import json
from transformers import BertTokenizer, BertModel
import torch
from .drain3.file_persistence import FilePersistence
from .drain3.template_miner import TemplateMiner
from .drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
from tqdm import tqdm
from typing import *
import utils as U
from .base_dataset import BaseDataset
import re
import numpy as np
import random


class BertEncoder:
    def __init__(self, config) -> None:
        self._bert_tokenizer = BertTokenizer.from_pretrained(config["tokenizer_path"])
        self._bert_model = BertModel.from_pretrained(config["model_path"])
        self.cache = {}

    def __call__(self, sentence, no_wordpiece=False) -> torch.Any:
        r"""
        return list(len=768)
        """
        if self.cache.get(sentence, None) is None:
            if no_wordpiece:
                words = sentence.split(" ")
                words = [
                    word for word in words if word in self._bert_tokenizer.vocab.keys()
                ]
                sentence = " ".join(words)
            inputs = self._bert_tokenizer(
                sentence, truncation=True, return_tensors="pt", max_length=512
            )
            outputs = self._bert_model(**inputs)

            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(dim=1)
            self.cache[sentence] = embedding[0].tolist()
            return embedding[0].tolist()
        else:
            return self.cache[sentence]


class DrainProcesser:
    def __init__(self, config) -> None:
        r"""
        config: {
            "save_path": "path/to",
            "drain_config_path": "path/to"
        }
        """
        self._drain_config_path = config["drain_config_path"]
        U.check(config["drain_save_path"])
        persistence = FilePersistence(config["drain_save_path"])
        miner_config = TemplateMinerConfig()
        miner_config.load(config["drain_config_path"])
        self._template_miner = TemplateMiner(persistence, config=miner_config)

    def __call__(self, sentence) -> str:
        line = str(sentence).strip()
        result = self._template_miner.add_log_message(line)
        return result["template_mined"]


class LogDataset(BaseDataset):
    def __init__(self, config) -> None:
        r"""
        X: [sample_num, seq, n_model]
        """
        super().__init__(config, "log")
        self.sample_interval = self.__config__["sample_interval"]
        self._drain = DrainProcesser(config["drain_config"])
        self._encoder = BertEncoder(config["bert_config"])

    def __add_sample__(self, st_time, cnts, log_df, label):
        cnt_of_log = {}
        seqs = []
        for cnt in range(cnts):
            lst_time = st_time + self.sample_interval * cnt
            led_time = lst_time + self.sample_interval
            sample_df = log_df.query(
                f"timestamp >= {lst_time} & timestamp < {led_time}"
            )
            template_list = []
            for log in sample_df["message"].tolist():
                template = self._drain(log)
                if cnt_of_log.get(template, None) is None:
                    cnt_of_log[template] = [0] * cnts
                cnt_of_log[template][cnt] += 1
                template_list.append(template)
            seqs.append(list(set(template_list)))
        # with open(f"log_tmp/{st_time}.json", "w", encoding="utf8") as w:
        #     json.dump(cnt_of_log, w)
        wei_of_log = {}
        total_gap = 0.00001
        for template, cnt_list in cnt_of_log.items():
            cnt_list = np.array(cnt_list)
            cnt_list = np.log(cnt_list + 0.00001)
            cnt_list = np.abs([0] + np.diff(cnt_list))
            gap = cnt_list.max() - cnt_list.mean()
            wei_of_log[template] = gap
            total_gap += gap
        new_seq = []
        for seq in seqs:
            repr = np.zeros((768,))
            for template in seq:
                repr += (
                    wei_of_log[template] * np.array(self._encoder(template)) / total_gap
                )
            new_seq.append(repr.tolist())
        """
        wei_of_log = {}
        for template, cnt_list in cnt_of_log.items():
            gap = np.log(np.max(cnt_list) + 1) - np.log(np.median(cnt_list) + 1)
            if gap == 0:
                wei_of_log[template] = (1) / np.exp(1)
            else:
                wei_of_log[template] = (gap) / np.exp(np.median(cnt_list))
        new_seq = []
        for seq in seqs:
            repr = np.zeros((768,))
            for template in seq:
                repr += wei_of_log[template] * np.array(self._encoder(template))
            new_seq.append(repr.tolist())
        """
        self.__X__.append(new_seq)
        self.__y__["failure_type"].append(label["failure_type"])
        self.__y__["root_cause"].append(label["root_cause"])

    def __load__(self, log_df: pd.DataFrame, groundtruth_df: pd.DataFrame):
        r"""
        :log_df       : [timestamp, message]
        :groundtruth_df   : [st_time, ed_time, failure_type, root_cause]
        """
        log_columns = log_df.columns.tolist()
        assert "timestamp" in log_columns, "log_df requires `timestamp`"
        assert "message" in log_columns, "log_df requires `message`"
        log_df = log_df.sort_values(by="timestamp")
        log_df["label"] = 0
        anomaly_columns = groundtruth_df.columns.tolist()
        assert "st_time" in anomaly_columns, "groundtruth_df requires `st_time`"
        assert "ed_time" in anomaly_columns, "groundtruth_df requires `ed_time`"
        assert (
            "failure_type" in anomaly_columns
        ), "groundtruth_df requires `failure_type`"
        assert "root_cause" in anomaly_columns, "groundtruth_df requires `root_cause`"
        st_time = log_df.head(1)["timestamp"].item()
        ed_time = log_df.tail(1)["timestamp"].item()

        process_bar = tqdm(
            total=len(groundtruth_df),
            desc=f"process {self.__desc_date__(st_time)} ~ {self.__desc_date__(ed_time)}",
        )
        for index, case in groundtruth_df.iterrows():
            in_condition = log_df.loc[
                (log_df["timestamp"] >= case["st_time"])
                & (log_df["timestamp"] <= case["ed_time"]),
                :,
            ]
            cnts = int((case["ed_time"] - case["st_time"]) / self.sample_interval)
            self.__add_sample__(
                case["st_time"],
                cnts,
                in_condition,
                {
                    "failure_type": self.failures.index(case["failure_type"]),
                    "root_cause": self.services.index(case["root_cause"]),
                },
            )
            process_bar.update(1)
        process_bar.close()

class Aiops22Log(LogDataset):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.ANOMALY_DICT = {
            "k8s容器网络延迟": "network",
            "k8s容器写io负载": "io",
            "k8s容器读io负载": "io",
            "k8s容器cpu负载": "cpu",
            "k8s容器网络资源包重复发送": "network",
            "k8s容器进程中止": "process",
            "k8s容器网络丢包": "network",
            "k8s容器内存负载": "memory",
            "k8s容器网络资源包损坏": "network",
        }

    def __load_groundtruth_df__(self, file_list):
        groundtruth_df = self.__load_df__(file_list, is_json=True)
        groundtruth_df = groundtruth_df.query("level != 'node'")
        groundtruth_df.loc[:, "cmdb_id"] = groundtruth_df["cmdb_id"].apply(
            lambda x: re.sub(r"\d?-\d", "", x)
        )
        groundtruth_df = groundtruth_df.rename(
            columns={"timestamp": "st_time", "cmdb_id": "root_cause"}
        )
        duration = 600
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df.loc[
            :, ["st_time", "ed_time", "failure_type", "root_cause"]
        ]

    def __load_log_df__(self, file_list):
        # read log
        log_df = self.__load_df__(file_list)
        log_df = log_df.rename(columns={"value": "message"})
        return log_df.loc[:, ["timestamp", "message"]]

    def load(self):
        # read groundtruth
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth-")
        log_files = self.__get_files__(self.dataset_dir, "-log-service")
        dates = [
            "2022-05-01",
            "2022-05-03",
            "2022-05-05",
            "2022-05-07",
            "2022-05-09",
        ]
        groundtruths = self.__add_by_date__(groundtruth_files, dates)
        logs = self.__add_by_date__(log_files, dates)

        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")
            groundtruth_df = self.__load_groundtruth_df__(groundtruths[index])
            log_df = self.__load_log_df__(logs[index])
            self.__load__(log_df, groundtruth_df)


class PlatformLog(LogDataset):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.ANOMALY_DICT = {
            "cpu anomaly": "cpu",
            "http/grpc request abscence": "http/grpc",
            "http/grpc requestdelay": "http/grpc",
            "memory overload": "memory",
            "network delay": "network",
            "network loss": "network",
            "pod anomaly": "pod_failure",
        }

    def __load_groundtruth_df__(self, file_list):
        groundtruth_df = self.__load_df__(file_list).rename(
            columns={
                "故障类型": "failure_type",
                "对应服务": "cmdb_id",
                "起始时间戳": "st_time",
                "截止时间戳": "ed_time",
                "持续时间": "duration",
            }
        )

        def meta_transfer(item):
            if item.find("(") != -1:
                item = eval(item)
                item = item[0]
            return item

        groundtruth_df.loc[:, "cmdb_id"] = groundtruth_df["cmdb_id"].apply(
            meta_transfer
        )
        groundtruth_df = groundtruth_df.rename(columns={"cmdb_id": "root_cause"})
        duration = 600
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df.loc[
            :, ["st_time", "ed_time", "failure_type", "root_cause"]
        ]

    def __load_log_df__(self, file_list):
        # read log
        log_df = self.__load_df__(file_list)
        log_df.loc[:, "timestamp"] = log_df["timestamp"].apply(lambda x: int(x))
        return log_df.loc[:, ["timestamp", "message"]]

    def load(self):
        groundtruth_files = self.__get_files__(self.dataset_dir, "ground_truth")
        log_files = self.__get_files__(self.dataset_dir, "log")
        dates = [
            # "2024-03-20",
            # "2024-03-21",
            "2024-03-22",
            "2024-03-23",
            "2024-03-24",
        ]
        groundtruths = self.__add_by_date__(groundtruth_files, dates)
        logs = self.__add_by_date__(log_files, dates)

        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")
            groundtruth_df = self.__load_groundtruth_df__(groundtruths[index])
            log_df = self.__load_log_df__(logs[index])
            self.__load__(log_df, groundtruth_df)


class GaiaLog(LogDataset):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.ANOMALY_DICT = {
            "[memory_anomalies]": "memory",
            "[normal memory freed label]": "memory",
            "[access permission denied exception]": "access",
            "[login failure]": "login",
            "[file moving program]": "file",
        }

    def __load_groundtruth_df__(self, file_list):
        groundtruth_df = self.__load_df__(file_list).rename(
            columns={
                "anomaly_type": "failure_type",
            }
        )

        groundtruth_df = groundtruth_df.rename(columns={"service": "root_cause"})
        duration = 600
        from datetime import datetime

        groundtruth_df["st_time"] = groundtruth_df["st_time"].apply(
            lambda x: datetime.strptime(
                x.split(".")[0], "%Y-%m-%d %H:%M:%S"
            ).timestamp()
        )
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df.loc[
            :, ["st_time", "ed_time", "failure_type", "root_cause"]
        ]

    def __load_log_df__(self, file_list):
        # read log
        # ,datetime,service,message
        from datetime import datetime

        log_df = self.__load_df__(file_list)

        def meta_ts(x):
            try:
                # exist `nan` message
                return datetime.strptime(
                    x.split(",")[0], "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            except:
                return 0

        log_df.loc[:, "timestamp"] = log_df["message"].apply(meta_ts)
        log_df["message"] = log_df["message"].apply(
            lambda x: "|".join(str(x).split("|")[1:])
        )
        log_df = log_df.rename(columns={"service": "cmdb_id"})
        return log_df.loc[:, ["timestamp", "message"]]

    def load(self):
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth.csv")
        log_files = self.__get_files__(self.dataset_dir, "log.csv")
        dates = [
            "2021-07-04",
            "2021-07-05",
            "2021-07-06",
            "2021-07-07",
            "2021-07-08",
            "2021-07-09",
            "2021-07-10",
            "2021-07-11",
            "2021-07-12",
            "2021-07-13",
            "2021-07-14",
            "2021-07-15",
            "2021-07-16",
            "2021-07-17",
            "2021-07-18",
            "2021-07-20",
            "2021-07-21",
            "2021-07-22",
            "2021-07-23",
            "2021-07-24",
            "2021-07-25",
            "2021-07-26",
            "2021-07-27",
            "2021-07-28",
            "2021-07-29",
            "2021-07-30",
            "2021-07-31",
        ]
        groundtruths = self.__add_by_date__(groundtruth_files, dates)
        logs = self.__add_by_date__(log_files, dates)

        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")
            groundtruth_df = self.__load_groundtruth_df__(groundtruths[index])
            log_df = self.__load_log_df__(logs[index])
            self.__load__(log_df, groundtruth_df)

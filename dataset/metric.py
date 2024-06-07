import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import re
import utils as U
from .base_dataset import BaseDataset


def normalize_and_diff(df, column):
    try:
        ts = df[column].tolist()
        nts = np.array(ts)
        nts = (nts - nts.mean()) / (nts.std() + 0.00001)
        nts = [0] + np.diff(nts).tolist()
        df[column] = nts
        return df
    except Exception as e:
        print(repr(e))


def meta_load_metric(
    metric_df,
    instances,
    cnts,
    kpi_list,
    st_time,
    sample_interval,
):
    try:
        kpi_dict = {kpi_name: index for index, kpi_name in enumerate(kpi_list)}
        metric = []
        for instance in instances:
            ins_ts = []
            ins_df = metric_df.query(f"cmdb_id == '{instance}'")
            for cnt in range(cnts):
                kpi_features = [0] * len(kpi_list)
                lst_time = st_time + sample_interval * cnt
                led_time = lst_time + sample_interval
                sample_df = ins_df.query(
                    f"timestamp >= {lst_time} & timestamp < {led_time}"
                )
                if len(sample_df) != 0:
                    for kpi_name, kpi_group in sample_df.groupby(by="kpi_name"):
                        if kpi_dict.get(kpi_name, None) is None:
                            continue
                        else:
                            kpi_features[kpi_dict[kpi_name]] = kpi_group["value"].mean()
                ins_ts.append(kpi_features)
            metric.append(ins_ts)
    except Exception as e:
        print(repr(e))
    return metric


class MetricDataset(BaseDataset):
    def __init__(self, config) -> None:
        super().__init__(config, "metric")
        self.sample_interval = config["sample_interval"]

        # feature
        self.kpi_num = 0
        self.kpi_list = []

    def load_from_tmp(self):
        appendix = super().load_from_tmp()
        self.kpi_num = appendix["kpi_num"]
        self.kpi_list = appendix["kpi_list"]

    def save_to_tmp(self):
        super().save_to_tmp({"kpi_num": self.kpi_num, "kpi_list": self.kpi_list})

    def __get_kpi_list__(self, metric_files):
        metric_df = self.__load_df__(metric_files)
        self.kpi_list = list(set(metric_df["kpi_name"].tolist()))
        self.kpi_list.sort()
        self.kpi_num = len(self.kpi_list)

    def __load_labels__(self, gt_df: pd.DataFrame):
        r"""
        gt_df:
            columns = ["failure_type", "root_cause(service level)"]
        """
        failure_type = []
        root_cause = []
        for _, case in gt_df.iterrows():
            root_cause.append(self.services.index(case["root_cause"]))
            failure_type.append(self.failures.index(case["failure_type"]))
        self.__y__["failure_type"].extend(failure_type)
        self.__y__["root_cause"].extend(root_cause)

    def __load_metric__(self, gt_df: pd.DataFrame, metric_df: pd.DataFrame):
        r"""
        gt_df:
            columns = ["st_time(10)", "ed_time(10)", "failure_type", "root_cause"]
        metric_df:
            columns = ["timestamp(10)", "cmdb_id", "kpi_name", "value"]
        """
        U.notice("Load metric")
        metric_df = metric_df.set_index("timestamp")
        pool = Pool(min(self.num_workers, len(gt_df)))
        scheduler = tqdm(total=len(gt_df), desc="dispatch")
        tasks = []
        for index, case in gt_df.iterrows():
            st_time = case["st_time"]
            ed_time = case["ed_time"]
            cnts = int((ed_time - st_time) / self.sample_interval)
            tmp_metric_df = metric_df.query(
                f"timestamp >= {st_time} & timestamp < {ed_time}"
            )
            """
            process metric
            """
            task = pool.apply_async(
                meta_load_metric,
                (
                    tmp_metric_df,
                    self.instances,
                    cnts,
                    self.kpi_list,
                    st_time,
                    self.sample_interval,
                ),
            )
            tasks.append(task)
            scheduler.update(1)
        pool.close()
        scheduler.close()
        scheduler = tqdm(total=len(tasks), desc="aggregate")
        for task in tasks:
            self.__X__.append(task.get())
            scheduler.update(1)
        scheduler.close()

    def get_feature(self):
        return self.kpi_num

    def rebuild(self, metric_df):
        U.notice("Rebuild ts data")
        pool = Pool(self.num_workers)
        scheduler = tqdm(total=len(metric_df), desc="dispatch")
        tasks = []
        for _, cmdb_group in metric_df.groupby("cmdb_id"):
            for _, kpi_group in cmdb_group.groupby("kpi_name"):
                task = pool.apply_async(normalize_and_diff, (kpi_group, "value"))
                tasks.append(task)
                scheduler.update(len(kpi_group))
        pool.close()
        scheduler.close()
        scheduler = tqdm(total=len(metric_df), desc="aggregate")
        dfs = []
        for task in tasks:
            df = task.get()
            dfs.append(df)
            scheduler.update(len(df))
        scheduler.close()
        return pd.concat(dfs)


class Aiops22Metric(MetricDataset):
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

    def __load_metric_df__(self, file_list):
        metric_df = self.__load_df__(file_list)
        metric_df["cmdb_id"] = metric_df["cmdb_id"].apply(lambda x: x.split(".")[1])
        metric_df = self.rebuild(metric_df)
        return metric_df.loc[:, ["timestamp", "cmdb_id", "kpi_name", "value"]]

    def load(self):
        # read run_table
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth-")
        metric_files = self.__get_files__(self.dataset_dir, "kpi_container")
        self.__get_kpi_list__(metric_files)
        dates = [
            "2022-05-01",
            "2022-05-03",
            "2022-05-05",
            "2022-05-07",
            "2022-05-09",
        ]
        groundtruths = self.__add_by_date__(groundtruth_files, dates)
        metrics = self.__add_by_date__(metric_files, dates)

        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")
            gt_df = self.__load_groundtruth_df__(groundtruths[index])
            metric_df = self.__load_metric_df__(metrics[index])
            self.__load_labels__(gt_df)
            self.__load_metric__(gt_df, metric_df)


class PlatformMetric(MetricDataset):
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

    def __load_metric_df__(self, file_list):
        metric_df = self.__load_df__(file_list)

        def meta_transfer(name):
            name = name.split(".")[1]
            if name.find("redis-cart") != -1:
                return "redis-cart"
            return name.split("-")[0]

        metric_df["cmdb_id"] = metric_df["cmdb_id"].apply(meta_transfer)
        metric_df = self.rebuild(metric_df)
        return metric_df.loc[:, ["timestamp", "cmdb_id", "kpi_name", "value"]]

    def load(self):
        groundtruth_files = self.__get_files__(self.dataset_dir, "ground_truth")
        metric_files = self.__get_files__(self.dataset_dir, "kpi_container")
        self.__get_kpi_list__(metric_files)
        dates = [
            # "2024-03-20",
            # "2024-03-21",
            "2024-03-22",
            "2024-03-23",
            "2024-03-24",
        ]
        groundtruths = self.__add_by_date__(groundtruth_files, dates)
        metrics = self.__add_by_date__(metric_files, dates)
        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")
            gt_df = self.__load_groundtruth_df__(groundtruths[index])
            metric_df = self.__load_metric_df__(metrics[index])
            self.__load_labels__(gt_df)
            self.__load_metric__(gt_df, metric_df)


class GaiaMetric(MetricDataset):
    def __init__(self, config: dict) -> None:
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

    def __load_metric_df__(self, file_list):
        metric_df = self.__load_df__(file_list)
        metric_df["cmdb_id"] = metric_df["cmdb_id"].apply(lambda x: x.split(".")[-1])

        def meta_transfer(x):
            if len(str(int(x))) != 10:
                x = int(x / 1000)
            return x

        metric_df["timestamp"] = metric_df["timestamp"].apply(meta_transfer)
        metric_df = self.rebuild(metric_df)
        return metric_df.loc[:, ["timestamp", "cmdb_id", "kpi_name", "value"]]

    def load(self):
        # read run_table
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth.csv")
        metric_files = self.__get_files__(self.dataset_dir, "docker_")
        self.__get_kpi_list__(metric_files)
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
        metrics = self.__add_by_date__(metric_files, dates)
        for index, date in enumerate(dates):
            U.notice(f"Loading... {date}")
            gt_df = self.__load_groundtruth_df__(groundtruths[index])
            metric_df = self.__load_metric_df__(metrics[index])
            self.__load_labels__(gt_df)
            self.__load_metric__(gt_df, metric_df)

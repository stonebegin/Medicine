from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from .base_dataset import BaseDataset
import re


class TraceDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config, "trace")
        self.instance_list = []
        self.invoke_list = []
        self.topo = [[], []]

    def load_from_tmp(self):
        appendix = super().load_from_tmp()
        self.invoke_list = appendix["invoke_list"]

    def save_to_tmp(self):
        super().save_to_tmp({"invoke_list": self.invoke_list})

    def __load__(self, traces, groundtruths):
        total = 0

        all_trace_df = self.__load_trace_df__([t[0] for t in traces])
        self.invoke_list = list(set(all_trace_df["invoke_link"].tolist()))
        self.invoke_list.sort()
        self.instance_list = list(
            set(all_trace_df["cmdb_id"].tolist()).union(
                set(all_trace_df["ccmdb_id"].tolist())
            )
        )
        self.instance_list.sort()

        for index, date in enumerate(self.dates):
            data = []

            # 读取 data
            trace_df = self.__load_trace_df__(traces[index])
            gt_df = self.__load_groundtruth_df__(groundtruths[index])
            total += len(gt_df)

            # print(len(self.invoke_list))
            # self.failures = list(set(gt_df["failure_type"].tolist()))
            # self.failures.sort()

            # 获取ref
            ref_mean = {}
            ref_std = {}
            grouped_df = trace_df.groupby("invoke_link")
            for invoke, group_data in grouped_df:
                durations = group_data["duration"].values
                # durations = np.diff(durations)
                ref_mean[invoke] = np.mean(durations)
                ref_std[invoke] = np.std(durations)

            cnt_of_invoke = {}
            for invoke in self.invoke_list:
                cnt_of_invoke[invoke] = [0 for _ in range(len(gt_df))]

            # 异常检测
            for index, row in tqdm(
                gt_df.iterrows(), total=len(gt_df), desc=f"{date} Trace Detecting"
            ):
                window = [0 for _ in range(len(self.invoke_list))]
                cnt_list = [0 for _ in range(len(self.invoke_list))]
                start = row["st_time"]
                end = row["ed_time"]
                tmp_df = trace_df.query(f"timestamp > {start} & timestamp < {end}")

                # 没有数据的情况（不应该发生）
                if len(tmp_df) == 0:
                    data.append(window)
                    self.__y__["failure_type"].append(
                        int(self.failures.index(row["failure_type"]))
                    )
                    print(f"Warning! no trace data in {start}~{end}")
                    continue

                # 处理
                grouped_df = tmp_df.groupby("invoke_link")

                for invoke, group_data in grouped_df:
                    durations = group_data["duration"].values
                    mean = ref_mean[invoke]
                    std_dev = ref_std[invoke]
                    if std_dev == 0:
                        continue
                    z_scores = abs((durations - mean) / std_dev)
                    anomalies = int(sum(z_scores > 3))
                    cnt_of_invoke[invoke][index] = anomalies

                    if anomalies == 0:
                        window[self.invoke_list.index(invoke)] = 0
                    else:
                        window[self.invoke_list.index(invoke)] = np.mean(
                            z_scores[z_scores > 3]
                        )

                data.append(window)
                self.__y__["failure_type"].append(
                    int(self.failures.index(row["failure_type"]))
                )
                # self.__y__['root_cause'].append(
                #     self.instances.index(row['cmdb_id']))

            total_gap = 0.00001
            wei_of_invoke = {}
            for invoke, cnt_list in cnt_of_invoke.items():
                cnt_list = np.array(cnt_list)
                cnt_list = (cnt_list - cnt_list.min()) / (
                    cnt_list.max() - cnt_list.min() + 0.00001
                )
                cnt_list = np.log(cnt_list + 1)
                cnt_list = np.abs([0] + np.diff(cnt_list))
                gap = cnt_list.max() - np.mean(cnt_list)
                wei_of_invoke[invoke] = gap
                total_gap += gap
                # print(f'cnt_list:{cnt_list}')
                # print(f'weight: {gap}')

            new_data = []
            for window in data:
                for i, value in enumerate(window):
                    window[i] = wei_of_invoke[self.invoke_list[i]] * value / total_gap
                new_data.append(window)

            self.__X__.extend(new_data)

        # print(self.invoke_list)
        count = 0
        for invoke in self.invoke_list:
            start = invoke.split("_")[0]
            end = invoke.split("_")[1]
            n1 = self.instance_list.index(start)
            n2 = self.instance_list.index(end)
            self.topo[0].append(n1)
            self.topo[1].append(n2)

            if n1 != n2:
                count += 1

        print(
            f"topo: {self.topo}, num_nodes: {len(self.instance_list)}, num_edges: {count}"
        )

    def get_feature(self):
        return len(self.invoke_list)


class Aiops22Trace(TraceDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dates = [
            "2022-05-01",
            "2022-05-03",
            "2022-05-05",
            "2022-05-07",
            "2022-05-09",
        ]
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
        groundtruth_df = groundtruth_df.rename(columns={"timestamp": "st_time"})
        duration = 600
        groundtruth_df["ed_time"] = groundtruth_df["st_time"] + duration
        groundtruth_df["st_time"] = groundtruth_df["st_time"] - duration
        groundtruth_df = groundtruth_df.reset_index(drop=True)
        groundtruth_df.loc[:, "failure_type"] = groundtruth_df["failure_type"].apply(
            lambda x: self.ANOMALY_DICT[x]
        )
        return groundtruth_df

    def __load_trace_df__(self, file_list):
        # 读取 data
        trace_df = self.__load_df__(file_list)

        # 处理 span
        trace_df["timestamp"] = trace_df["timestamp"].apply(lambda x: int(x / 1000))
        trace_df = trace_df.rename(columns={"parent_span": "parent_id"})

        # 父子拼接
        meta_df = trace_df[["parent_id", "cmdb_id"]].rename(
            columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
        )
        trace_df = pd.merge(trace_df, meta_df, on="span_id")

        # 划分 span
        trace_df = trace_df.set_index("timestamp")
        trace_df = trace_df.sort_index()
        trace_df["invoke_link"] = trace_df["cmdb_id"] + "_" + trace_df["ccmdb_id"]
        return trace_df

    def load(self):
        trace_files = self.__get_files__(self.dataset_dir, "trace_jaeger-span")
        traces = self.__add_by_date__(trace_files, self.dates)
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth-")
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)

        self.__load__(traces, groundtruths)


class PlatformTrace(TraceDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dates = [
            # "2024-03-21",
            "2024-03-22",
            "2024-03-23",
            "2024-03-24",
        ]
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
        return groundtruth_df

    def __load_trace_df__(self, file_list):
        # 读取 data
        trace_df = self.__load_df__(file_list)

        # 处理 span
        trace_df["timestamp"] = trace_df["timestamp"].apply(lambda x: int(x / 1e6))
        trace_df = trace_df.rename(columns={"parent_span": "parent_id"})

        # 父子拼接
        meta_df = trace_df[["parent_id", "cmdb_id"]].rename(
            columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
        )
        trace_df = pd.merge(trace_df, meta_df, on="span_id")

        # 划分 span
        trace_df = trace_df.set_index("timestamp")
        trace_df = trace_df.sort_index()
        trace_df["invoke_link"] = trace_df["cmdb_id"] + "_" + trace_df["ccmdb_id"]
        return trace_df

    def load(self):
        trace_files = self.__get_files__(self.dataset_dir, "trace")
        traces = self.__add_by_date__(trace_files, self.dates)
        groundtruth_files = self.__get_files__(self.dataset_dir, "ground_truth")
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)

        self.__load__(traces, groundtruths)


class GaiaTrace(TraceDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dates = [
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
                "instance": "cmdb_id",
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
        groundtruth_df["st_time"] = groundtruth_df["st_time"].apply(
            lambda x: datetime.strptime(
                x.split(".")[0], "%Y-%m-%d %H:%M:%S"
            ).timestamp()
        )
        groundtruth_df["ed_time"] = groundtruth_df["ed_time"].apply(
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
        return groundtruth_df

    def __load_trace_df__(self, file_list):
        # 读取 data
        trace_df = self.__load_df__(file_list)

        # 处理 span
        trace_df = trace_df.rename(columns={"service_name": "cmdb_id"})

        # 父子拼接
        meta_df = trace_df[["parent_id", "cmdb_id"]].rename(
            columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
        )
        trace_df = pd.merge(trace_df, meta_df, on="span_id")

        # 划分 span
        trace_df = trace_df.set_index("timestamp")
        trace_df = trace_df.sort_index()
        trace_df["duration"] = trace_df["ed_time"] - trace_df["st_time"]
        trace_df["invoke_link"] = trace_df["cmdb_id"] + "_" + trace_df["ccmdb_id"]
        return trace_df

    def load(self):
        trace_files = self.__get_files__(self.dataset_dir, "trace")
        traces = self.__add_by_date__(trace_files, self.dates)
        groundtruth_files = self.__get_files__(self.dataset_dir, "groundtruth")
        groundtruths = self.__add_by_date__(groundtruth_files, self.dates)

        self.__load__(traces, groundtruths)

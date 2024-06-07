from sklearn.model_selection import train_test_split
import torch
import utils as U
import os
import math
import time
from typing import *
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from model.fusion import AdaFusion, ExperimentModel
from dataset import LOG_DATASET, METRIC_DATASET, TRACE_DATASET
import random


class MultiModalTrainer:
    def __init__(self, config, time) -> None:
        U.set_logger(config, time)
        self.__config__ = config
        self.__device__ = U.get_device(config)
        self.__log_data__ = self.__load_dataset__(
            LOG_DATASET[config["dataset"]](config)
        )
        self.__metric_data__ = self.__load_dataset__(
            METRIC_DATASET[config["dataset"]](config)
        )
        self.__trace_data__ = self.__load_dataset__(
            TRACE_DATASET[config["dataset"]](config)
        )

    def __create_model__(self):
        return AdaFusion(
            self.__metric_data__.get_feature(),
            self.__trace_data__.get_feature(),
            len(self.__log_data__.instances),
            self.__config__["max_len"],
            self.__config__["d_model"],
            self.__config__["nhead"],
            self.__config__["d_ff"],
            self.__config__["layer_num"],
            self.__config__["dropout"],
            self.__config__["num_class"],
            self.__device__,
        )

    def __create_experiment_model__(self):
        return ExperimentModel(
            self.__metric_data__.get_feature(),
            self.__trace_data__.get_feature(),
            len(self.__log_data__.instances),
            self.__config__["max_len"],
            self.__config__["d_model"],
            self.__config__["nhead"],
            self.__config__["d_ff"],
            self.__config__["layer_num"],
            self.__config__["dropout"],
            self.__config__["num_class"],
            self.__device__,
        )

    def __load_dataset__(self, dataset):
        if self.__config__["use_tmp"] and os.path.exists(dataset.get_dataset_path()):
            print(f"Use: cached dataset")
            dataset.load_from_tmp()
        else:
            dataset.load()
            dataset.save_to_tmp()
        return dataset

    def __softmax__(self, X):
        X_exp = X.exp()
        partition = X_exp.sum()
        return X_exp / partition

    def __normalization__(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def __sigmoid__(self, data):
        fz = []
        for num in data:
            fz.append(1 / (1 + math.exp(-num)))
        return fz

    def experiment(self):
        self.__experiment_train__("all")
        self.__experiment_train__("log")
        self.__experiment_train__("metric")
        self.__experiment_train__("trace")

    def __experiment_train__(self, modal):
        logger = logging.getLogger(self.__config__["log_name"])
        logger.info(f"Single modal training [{modal}]")
        print(f"Single modal training [{modal}]")
        model = self.__create_experiment_model__()
        train_loader, eval_loader, test_loader = self.__get_loader__()
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=self.__config__["lr"],
            # momentum=0.9,
            weight_decay=self.__config__["weight_decay"],
        )
        model.set_use_modal(modal)
        model.to(self.__device__)
        best_model = (torch.inf, None)
        entropy = torch.nn.CrossEntropyLoss()
        record_loss = []
        for epoch in tqdm(range(self.__config__["epochs"]), desc="Training"):
            model.train()
            avg_loss = 0
            for inputs, labels in train_loader:
                inputs = [_inputs.to(self.__device__) for _inputs in inputs]
                labels = labels.to(self.__device__)
                optim.zero_grad()
                outputs = model(inputs)
                loss = entropy(outputs, labels)
                loss.backward()
                avg_loss += loss.item()
                optim.step()

            avg_loss /= len(train_loader)
            record_loss.append(avg_loss)
            logger.info(f"epoch: {epoch: 3}, training loss: {avg_loss}")
            best_model = self.__experiment_eval__(
                model, eval_loader, best_model, modal
            )
        import json
        with open(
            os.path.join(
                "result", self.__config__["dataset"], f"single_train_{modal}_loss.json"
            ),
            "w",
            encoding="utf8",
        ) as w:
            json.dump(
                {
                    "loss": record_loss,
                    "epoch": list(range(1, self.__config__["epochs"] + 1)),
                },
                w,
            )
        model.load_state_dict(best_model[1])
        return self.__experiment_test__(model, test_loader, modal)

    def train(self):
        logger = logging.getLogger(self.__config__["log_name"])
        model = self.__create_model__()
        train_loader, eval_loader, test_loader = self.__get_loader__()


        if self.__config__["optim"] == "AdamW":
            optim = torch.optim.AdamW(
                model.parameters(),
                lr=self.__config__["lr"],
                weight_decay=self.__config__["weight_decay"],
            )
        elif self.__config__["optim"] == "SGD":
            optim = torch.optim.SGD(
                model.parameters(),
                lr=self.__config__["lr"],
                momentum=0.9,
                weight_decay=self.__config__["weight_decay"],
            )

        model.to(self.__device__)
        best_model = (torch.inf, None)
        entropy = torch.nn.CrossEntropyLoss()
        alpha = 0.5
        beta = 0.3
        st_time = time.time()
        for epoch in tqdm(range(self.__config__["epochs"]), desc="Training"):
            # Multimodal Adaptive Optimization
            model.train()
            if epoch < 1 or epoch > 80:
                avg_loss = 0
                train_m_loss = 0
                train_l_loss = 0
                train_t_loss = 0
                for inputs, labels in train_loader:
                    inputs = [_inputs.to(self.__device__) for _inputs in inputs]
                    labels = labels.to(self.__device__)
                    optim.zero_grad()
                    feat_m, feat_l, feat_t, outputs = model(inputs)

                    out_m = (
                        torch.mm(feat_m, model.concat_fusion.linear_x_out.weight)
                        + model.concat_fusion.linear_x_out.bias / 2
                    )
                    out_m = (
                        torch.mm(
                            out_m, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                        )
                        + model.concat_fusion.clf.bias / 2
                    )

                    out_l = (
                        torch.mm(feat_l, model.concat_fusion.linear_y_out.weight)
                        + model.concat_fusion.linear_y_out.bias / 2
                    )
                    out_l = (
                        torch.mm(
                            out_l, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                        )
                        + model.concat_fusion.clf.bias / 2
                    )

                    out_t = (
                        torch.mm(feat_t, model.concat_fusion.linear_z_out.weight)
                        + model.concat_fusion.linear_z_out.bias / 2
                    )
                    out_t = (
                        torch.mm(
                            out_t, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                        )
                        + model.concat_fusion.clf.bias / 2
                    )

                    loss = entropy(outputs, labels)
                    loss_m = entropy(out_m, labels)
                    loss_l = entropy(out_l, labels)
                    loss_t = entropy(out_t, labels)
                    theta_ratio = self.__sigmoid__(
                        np.array(
                            [
                                loss_m.detach().numpy(),
                                loss_l.detach().numpy(),
                                loss_t.detach().numpy(),
                            ]
                        )
                    )
                    loss = (
                        loss
                        + theta_ratio[0] * loss_m
                        + theta_ratio[1] * loss_l
                        + theta_ratio[2] * loss_t
                    )
                    loss.backward()

                    avg_loss += loss.item()
                    train_m_loss += loss_m.item()
                    train_l_loss += loss_l.item()
                    train_t_loss += loss_t.item()
                    optim.step()

                avg_loss /= len(train_loader)
                train_m_loss /= len(train_loader)
                train_l_loss /= len(train_loader)
                train_t_loss /= len(train_loader)
                logger.info(f"epoch: {epoch: 3}, training loss: {avg_loss}")
                best_model, eval_m_loss, eval_l_loss, eval_t_loss = self.__eval__(
                    model, eval_loader, best_model
                )
            else:
                avg_loss = 0
                cur_m_loss = 0
                cur_l_loss = 0
                cur_t_loss = 0
                for inputs, labels in train_loader:
                    inputs = [_inputs.to(self.__device__) for _inputs in inputs]
                    labels = labels.to(self.__device__)
                    optim.zero_grad()
                    feat_m, feat_l, feat_t, outputs = model(inputs)

                    out_m = (
                        torch.mm(feat_m, model.concat_fusion.linear_x_out.weight)
                        + model.concat_fusion.linear_x_out.bias / 2
                    )
                    out_m = (
                        torch.mm(
                            out_m, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                        )
                        + model.concat_fusion.clf.bias / 2
                    )

                    out_l = (
                        torch.mm(feat_l, model.concat_fusion.linear_y_out.weight)
                        + model.concat_fusion.linear_y_out.bias / 2
                    )
                    out_l = (
                        torch.mm(
                            out_l, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                        )
                        + model.concat_fusion.clf.bias / 2
                    )

                    out_t = (
                        torch.mm(feat_t, model.concat_fusion.linear_z_out.weight)
                        + model.concat_fusion.linear_z_out.bias / 2
                    )
                    out_t = (
                        torch.mm(
                            out_t, torch.transpose(model.concat_fusion.clf.weight, 0, 1)
                        )
                        + model.concat_fusion.clf.bias / 2
                    )

                    loss = entropy(outputs, labels)
                    loss_m = entropy(out_m, labels)
                    loss_l = entropy(out_l, labels)
                    loss_t = entropy(out_t, labels)
                    loss = loss + loss_m + loss_l + loss_t
                    loss.backward()

                    avg_loss += loss.item()
                    cur_m_loss += loss_m.item()
                    cur_l_loss += loss_l.item()
                    cur_t_loss += loss_t.item()

                    optim.step()

                avg_loss /= len(train_loader)
                cur_m_loss /= len(train_loader)
                cur_l_loss /= len(train_loader)
                cur_t_loss /= len(train_loader)
                logger.info(f"epoch: {epoch: 3}, training loss: {avg_loss}")
                best_model, cur_eval_m_loss, cur_eval_l_loss, cur_eval_t_loss = (
                    self.__eval__(model, eval_loader, best_model)
                )

                # Gradient optimization
                minus_eval_m = (
                    (eval_m_loss - cur_eval_m_loss)
                    if eval_m_loss > cur_eval_m_loss
                    else 0.0001
                )
                minus_train_m = (
                    (train_m_loss - cur_m_loss) if train_m_loss > cur_m_loss else 0.0001
                )
                minus_eval_l = (
                    (eval_l_loss - cur_eval_l_loss)
                    if eval_l_loss > cur_eval_l_loss
                    else 0.0001
                )
                minus_train_l = (
                    (train_l_loss - cur_l_loss) if train_l_loss > cur_l_loss else 0.0001
                )
                minus_eval_t = (
                    (eval_t_loss - cur_eval_t_loss)
                    if eval_t_loss > cur_eval_t_loss
                    else 0.0001
                )
                minus_train_t = (
                    (train_t_loss - cur_t_loss) if train_t_loss > cur_t_loss else 0.0001
                )
                ratio_m = minus_eval_m / minus_train_m
                ratio_l = minus_eval_l / minus_train_l
                ratio_t = minus_eval_t / minus_train_t
                # add_m = minus_eval_m + minus_train_m
                # add_l = minus_eval_l + minus_train_l
                # add_t = minus_eval_t + minus_train_t
                theta_ratio = self.__sigmoid__(np.array([ratio_m, ratio_l, ratio_t]))
                value_ratio, index_ratio = torch.sort(torch.tensor(theta_ratio))
                coeffs = [1, 1, 1]
                coeffs[index_ratio[0]] = 1 - alpha * value_ratio[0]  # 抑制
                coeffs[index_ratio[-1]] = 1 + beta * value_ratio[-1] # 鼓励

                for name, parms in model.named_parameters():
                    layer = str(name).split(".")[0]
                    if "metric" in layer:
                        # parms.grad *= coeffs[0]
                        parms.grad = parms.grad * coeffs[0] + torch.zeros_like(
                            parms.grad
                        ).normal_(0, parms.grad.std().item() + 1e-8)
                    if "log" in layer:
                        # parms.grad *= coeffs[1]
                        parms.grad = parms.grad * coeffs[1] + torch.zeros_like(
                            parms.grad
                        ).normal_(0, parms.grad.std().item() + 1e-8)
                    if "trace" in layer:
                        # parms.grad *= coeffs[2]
                        parms.grad = parms.grad * coeffs[2] + torch.zeros_like(
                            parms.grad
                        ).normal_(0, parms.grad.std().item() + 1e-8)
                optim.step()

                train_m_loss = cur_m_loss
                train_l_loss = cur_l_loss
                train_t_loss = cur_t_loss
                eval_m_loss = cur_eval_m_loss
                eval_l_loss = cur_eval_l_loss
                eval_t_loss = cur_eval_t_loss

        training_time = (time.time() - st_time)
        logger.info(f"Training time={training_time: .6}, #Case={len(train_loader)}")
        model.load_state_dict(best_model[1])
        return self.__test__(model, test_loader)

    def __experiment_test__(self, model: ExperimentModel, test_loader, modal):
        logger = logging.getLogger(self.__config__["log_name"])
        logger.info(f"Tesing under {modal}")
        model.set_use_modal(modal)
        model.eval()
        all_labels = []
        all_preds = []
        test_loss = 0
        for inputs, labels in test_loader:
            inputs = [_inputs.to(self.__device__) for _inputs in inputs]
            labels = labels.to(self.__device__)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            test_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        test_loss /= len(test_loader)
        logger.info(f"Testing loss={test_loss: .6}")
        print(f"Testing loss={test_loss: .6}")
        return (
            all_preds,
            all_labels,
            {
                "macro": self.__metric__(all_labels, all_preds, "macro"),
                "weighted": self.__metric__(all_labels, all_preds, "weighted"),
            },
        )

    def __test__(self, model, test_loader):
        logger = logging.getLogger(self.__config__["log_name"])
        model.eval()
        all_labels = []
        all_preds = []
        test_loss = 0
        st_time = time.time()
        for inputs, labels in test_loader:
            inputs = [_inputs.to(self.__device__) for _inputs in inputs]
            labels = labels.to(self.__device__)
            _, _, _, outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            test_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        tesing_time = (time.time() - st_time)
        logger.info(f"Testing time={tesing_time: .6}, #Case={len(test_loader)}")
        test_loss /= len(test_loader)
        logger.info(f"Testing loss={test_loss: .6}")
        print(f"Testing loss={test_loss: .6}")
        return (
            all_preds,
            all_labels,
            {
                "macro": self.__metric__(all_labels, all_preds, "macro"),
                "weighted": self.__metric__(all_labels, all_preds, "weighted"),
            },
        )

    def __experiment_eval__(self, model: ExperimentModel, eval_loader, best_model, modal):
        logger = logging.getLogger(self.__config__["log_name"])
        model.set_use_modal(modal)
        model.eval()
        eval_loss = 0
        for inputs, labels in eval_loader:
            inputs = [_inputs.to(self.__device__) for _inputs in inputs]
            labels = labels.to(self.__device__)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            eval_loss += loss.item()
        eval_loss /= len(eval_loader)
        if eval_loss <= best_model[0]:
            logger.info(f"Reduce from {best_model[0]: .6f} -> {eval_loss: .6f}")
            best_model = (eval_loss, model.state_dict())
        return best_model

    def __eval__(self, model, eval_loader, best_model):
        logger = logging.getLogger(self.__config__["log_name"])
        model.eval()
        eval_loss = 0
        eval_m_loss = 0
        eval_l_loss = 0
        eval_t_loss = 0
        for inputs, labels in eval_loader:
            inputs = [_inputs.to(self.__device__) for _inputs in inputs]
            labels = labels.to(self.__device__)
            feat_m, feat_l, feat_t, outputs = model(inputs)

            out_m = (
                torch.mm(feat_m, model.concat_fusion.linear_x_out.weight)
                + model.concat_fusion.linear_x_out.bias / 2
            )
            out_m = (
                torch.mm(out_m, torch.transpose(model.concat_fusion.clf.weight, 0, 1))
                + model.concat_fusion.clf.bias / 2
            )

            out_l = (
                torch.mm(feat_l, model.concat_fusion.linear_y_out.weight)
                + model.concat_fusion.linear_y_out.bias / 2
            )
            out_l = (
                torch.mm(out_l, torch.transpose(model.concat_fusion.clf.weight, 0, 1))
                + model.concat_fusion.clf.bias / 2
            )

            out_t = (
                torch.mm(feat_t, model.concat_fusion.linear_z_out.weight)
                + model.concat_fusion.linear_z_out.bias / 2
            )
            out_t = (
                torch.mm(out_t, torch.transpose(model.concat_fusion.clf.weight, 0, 1))
                + model.concat_fusion.clf.bias / 2
            )

            labels = labels.clone().detach()
            outputs = outputs.clone().detach()
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss_m = torch.nn.functional.cross_entropy(out_m, labels)
            loss_l = torch.nn.functional.cross_entropy(out_l, labels)
            loss_t = torch.nn.functional.cross_entropy(out_t, labels)
            eval_loss += loss.item()
            eval_m_loss += loss_m.item()
            eval_l_loss += loss_l.item()
            eval_t_loss += loss_t.item()
        eval_loss /= len(eval_loader)
        eval_m_loss /= len(eval_loader)
        eval_l_loss /= len(eval_loader)
        eval_t_loss /= len(eval_loader)
        if eval_loss <= best_model[0]:
            logger.info(f"Reduce from {best_model[0]: .6f} -> {eval_loss: .6f}")
            best_model = (eval_loss, model.state_dict())
        return best_model, eval_m_loss, eval_l_loss, eval_t_loss

    def __metric__(
        self,
        y_true,
        y_pred,
        method: Literal["micro", "macro", "samples", "weighted"] = "weighted",
        to_stdout=True,
        to_logger=True,
    ):
        logger = logging.getLogger(self.__config__["log_name"])
        precision = precision_score(
            y_true,
            y_pred,
            average=method,
            zero_division=0,
        )
        recall = recall_score(y_true, y_pred, average=method, zero_division=0)
        f1score = f1_score(y_true, y_pred, average=method)
        if to_logger:
            logger.info(f"{method} precision:{precision}")
            logger.info(f"{method} recall   :{recall}")
            logger.info(f"{method} f1score  :{f1score}")
        if to_stdout:
            print(f"{method} precision:{precision}")
            print(f"{method} recall   :{recall}")
            print(f"{method} f1score  :{f1score}")
        return [precision, recall, f1score]

    def __get_loader__(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        X_train, X_test, y_train, y_test = train_test_split(
            range(len(self.__log_data__)),
            self.__log_data__.y,
            test_size=0.2,
            # stratify=self.__log_data__.y,
        )
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            # stratify=y_train
        )
        data = list(
            zip(
                self.__log_data__.X,
                self.__metric_data__.X,
                self.__trace_data__.X,
                self.__log_data__.y,
            )
        )
        train_data = []
        for index in X_train:
            train_data.append(data[index])
        train_data = self.__data_enhance__(train_data)
        print("The number of train samples: ", len(train_data))
        print("The number of eval  samples: ", len(X_eval))
        print("The number of test  samples: ", len(X_test))
        return (
            DataLoader(
                train_data,
                batch_size=self.__config__["batch_size"],
                shuffle=True,
                collate_fn=self.__collat_fn__,
            ),
            DataLoader(
                data,
                batch_size=self.__config__["batch_size"],
                sampler=X_eval,
                collate_fn=self.__collat_fn__,
            ),
            DataLoader(
                data,
                batch_size=self.__config__["batch_size"],
                sampler=X_test,
                collate_fn=self.__collat_fn__,
            ),
        )

    def __collat_fn__(self, batch):
        log = []
        metric = []
        trace = []
        y = []
        for _log, _metric, _trace, _y in batch:
            log.append(_log)
            metric.append(_metric)
            trace.append(_trace)
            y.append(_y)
        return (
            (
                torch.tensor(log, dtype=torch.float),
                torch.tensor(metric, dtype=torch.float),
                torch.tensor(trace, dtype=torch.float),
            ),
            torch.tensor(y, dtype=torch.long),
        )

    def __data_enhance__(self, data):
        y_dict = {}
        for index, sample in enumerate(data):
            if y_dict.get(sample[3], None) is None:
                y_dict[sample[3]] = []
            y_dict[sample[3]].append(index)
        enhance_num = max([len(val) for val in y_dict.values()])
        scheduler = tqdm(total=enhance_num * len(y_dict.keys()), desc="Data enhancing")
        new_data = []
        for _, indices in y_dict.items():
            cnt = len(indices)
            scheduler.update(cnt)
            while cnt < enhance_num:
                new_data.append(self.__fake__(indices, data))
                cnt += 1
                scheduler.update(1)
        scheduler.close()
        data.extend(new_data)
        return data

    def __fake__(self, indices, data):
        choices = random.choices(indices, k=2)
        sample1 = data[choices[0]]
        sample2 = data[choices[1]]
        return (
            random.choice([sample1[0], sample2[0]]),
            random.choice([sample1[1], sample2[1]]),
            random.choice([sample1[2], sample2[2]]),
            sample1[3],
        )

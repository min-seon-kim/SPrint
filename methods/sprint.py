import logging
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.finetune import Finetune
from utils.data_loader import cutmix_data, ImageDataset
from utils.compute_forgetting import compute_forgetting_statistics, sort_examples_by_forgetting

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i
            

class SPLLoss(nn.NLLLoss):
    def __init__(self, *args, n_samples=0, device='cuda', **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)
        self.device = device  
        self.threshold = 0.1
        self.growing_factor = 1.4
        self.v = torch.zeros(n_samples, device=self.device).int()

    def forward(self, input: Tensor, target: Tensor, index: Tensor, flag: Tensor) -> Tensor:
        super_loss = F.nll_loss(input, target, reduction="none")
        # logger.info(super_loss, super_loss.shape)
        v = self.spl_loss(super_loss, flag)
        # logger.info(index.long(), index.shape)
        # logger.info(v, v.shape)
        self.v.index_copy_(0, index.long(), v)
        return (super_loss * v.float()).mean()

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss, flag):
        v = super_loss < self.threshold
        v = v.int()
        if flag is not None:
            v[flag.to(torch.bool)] = 1
        return v
            
            
class SPrint(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.n_epoch = kwargs["n_epoch"]
        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.exp_env = kwargs["stream_env"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "reservoir"
        
        # forgetting statistics
        self.example_stats = {}
        self.memory_indices_set = set()
        
    def get_dataloader(self, batch_size, n_worker, train_list, test_list):
        # Loader
        train_loader = None
        test_loader = None
        if train_list is not None and len(train_list) > 0:
            train_dataset = ImageDataset(
                pd.DataFrame(train_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            # drop last becasue of BatchNorm1D in IcarlNet
            train_loader = DataLoader(
                train_dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=False,
            )

        if test_list is not None:
            test_dataset = ImageDataset(
                pd.DataFrame(test_list),
                dataset=self.dataset,
                transform=self.test_transform,
            )
            test_loader = DataLoader(
                test_dataset, shuffle=False, batch_size=batch_size, num_workers=n_worker
            )

        return train_loader, test_loader, train_dataset, test_dataset
    
    def inverse_class_probability_sampling(self, samples):
        class_counts = {}
        total_count = len(samples)

        # Count the number of samples for each class
        for sample in samples:
            class_label = sample['label']
            
            if class_label not in class_counts:
                class_counts[class_label] = 0
            
            class_counts[class_label] += 1

        # Calculate the inclusion probability for each class
        class_probabilities = {}
        for class_label, count in class_counts.items():
            class_probabilities[class_label] = 1 - (count / total_count)

        # Normalize the probabilities
        total_probability = sum(class_probabilities.values())
        for class_label in class_probabilities:
            class_probabilities[class_label] /= total_probability

        # Perform the modified reservoir sampling
        for i, sample in enumerate(samples):
            class_label = sample['label']
            inclusion_probability = class_probabilities[class_label]

            if len(self.memory_list) < self.memory_size:
                self.memory_list.append(sample)
            else:
                j = np.random.rand()
                if j < inclusion_probability:
                    replace_index = np.random.randint(len(self.memory_list))
                    self.memory_list[replace_index] = sample
    
    def create_combined_dataset(self, A, B, batch_size):
        combined_dataset = []
        A_index, B_index = 0, 0
        cur_mem_size = len(self.memory_list)
        mem_per_batch = int(cur_mem_size // (len(self.streamed_list) // int(self.batch_size)))
        logger.info(f"{mem_per_batch} and {type(mem_per_batch)}")
        
        while A_index < len(A) or B_index < len(B):
            # A에서 데이터 선택
            A_end = min(A_index + batch_size - mem_per_batch, len(A))  # 변경됨
            batch_from_A = A[A_index:A_end]
            A_index = A_end  # 변경됨
            
            # B에서 데이터 선택
            B_end = min(B_index + mem_per_batch, len(B))
            batch_from_B = B[B_index:B_end]
            B_index = B_end  # 변경됨
            
            # 배치 합치기
            batch = np.concatenate((batch_from_A, batch_from_B))
            combined_dataset.extend(batch)
        
        return combined_dataset
    
    
    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=0):
        # Initialize forgetting statistics
        self.example_stats = {}
        
        # train_list == streamed_list + memory_list in SPrint
        train_list = self.streamed_list
        test_list = self.test_list
        random.shuffle(train_list)
        
        # Combined stream data with memory data
        if cur_iter >= 1:
            train_list = self.create_combined_dataset(train_list, self.memory_list, batch_size)
        
        # Configuring a batch with streamed and memory data equally.
        train_loader, test_loader, train_dataset, test_dataset = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")
        
        if cur_iter >= 1:
            logger.info("Use Self-Paced Loss Function")
            self.criterion = SPLLoss(n_samples=len(self.streamed_list + self.memory_list), device=self.device).cuda()
        # Get indicies of samples in memory 
        self.memory_indices_set = set([item['original_index'] for item in self.memory_list])
        
        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        self.model = self.model.to(self.device)
        for epoch in range(n_epoch):
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()

            train_loss, train_acc = self._train(cur_iter=cur_iter, train_loader=train_loader, 
                                                optimizer=self.optimizer, criterion=self.criterion)
            eval_dict = self.evaluation(
                cur_iter=cur_iter, test_loader=test_loader, criterion=self.criterion
            )
            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )
            
            # Increase loss threshold
            if cur_iter >= 1:
                self.criterion.increase_threshold()
            
            best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict
    

    def update_model(self, x, y, cur_iter, criterion, optimizer, index_tensor, flag):
        optimizer.zero_grad()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        
        if cur_iter >= 1:
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)
                output = torch.log_softmax(logit, dim=1)
                loss = lam * criterion(output, labels_a, index_tensor, flag) + (1 - lam) * criterion(
                    output, labels_b, index_tensor, flag
                )
            else:
                logit = self.model(x)
                output = torch.log_softmax(logit, dim=1)
                loss = criterion(output, y, index_tensor, flag)
        else:
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)
                loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                    logit, labels_b
                )
            else:
                logit = self.model(x)
                loss = criterion(logit, y)
        
        _, predicted = torch.max(logit.data, 1)
        _, preds = logit.topk(self.topk, 1, True, True)

        acc = predicted == y
        
        loss.backward()
        optimizer.step()
        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0), acc


    def _train(
        self, cur_iter, train_loader, optimizer, criterion
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        label_list = []
        
        self.model.train()
        
        if train_loader is None:
            raise NotImplementedError("None of dataloder is valid")

        for batch_idx, data in enumerate(train_loader):
            batch_inds = data['original_index'].tolist()
            x = data["image"]
            y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)
            
            label_list.extend(y.cpu().tolist())
            
            index_tensor = torch.tensor(list(range(batch_idx*self.batch_size, 
                                            min(batch_idx*self.batch_size+self.batch_size, 
                                            len(self.streamed_list + self.memory_list)))), dtype=torch.int64).cuda()
            flag = torch.tensor([ind in self.memory_indices_set for ind in batch_inds], dtype=torch.float32).cuda()

            l, c, d, acc = self.update_model(x, y, cur_iter, criterion, optimizer, index_tensor, flag)
            total_loss += l
            correct += c
            num_data += d
            
            for j, index in enumerate(batch_inds):
                index_in_original_dataset = index
                index_stats = self.example_stats.get(index_in_original_dataset, [])
                index_stats.append(acc[j].sum().item())
                self.example_stats[index_in_original_dataset] = index_stats

        n_batches = len(train_loader)

        if cur_iter >= 1:
            # How many samples has been chosen by self-paced loss function
            easy_counts = {}
            hard_counts = {}

            for lb, weight in zip(label_list, criterion.v):
                if weight >= 1:
                    if lb in easy_counts:
                        easy_counts[lb] += 1
                    else:
                        easy_counts[lb] = 1
                else:
                    if lb in hard_counts:
                        hard_counts[lb] += 1
                    else:
                        hard_counts[lb] = 1

            # Combine results
            results = {}
            for lb in set(label_list):
                results[lb] = {"Easy": easy_counts.get(lb, 0), "Hard": hard_counts.get(lb, 0)}
            logger.info(f'Lambda threshold is {self.criterion.threshold}')
            logger.info(f'\n{results}')

        return total_loss / n_batches, correct / num_data


    def order_examples_by_forgetting(self, example_stats, cur_iter):
        presentations_needed_to_learn, unlearned_per_presentation, first_learned = compute_forgetting_statistics(example_stats, self.n_epoch)
        # logger.info(f'presentations_needed_to_learn is {presentations_needed_to_learn}')
    
        # Initialize lists to collect forgetting stastics per example across multiple training runs
        unlearned_per_presentation_all, first_learned_all = [], []

        unlearned_per_presentation_all.append(unlearned_per_presentation)
        first_learned_all.append(first_learned)

        # Sort examples by forgetting counts in ascending order, over one or more training runs
        ordered_examples, ordered_values = sort_examples_by_forgetting(unlearned_per_presentation_all, first_learned_all, self.n_epoch)
        # logger.info(f'ordered_examples is {ordered_examples}')
        # logger.info(f'ordered_values is {ordered_values}')

        forgetting_dictionary = {k: v for k, v in zip(ordered_examples, ordered_values)}
        # paced_dict = {k: forgetting_dictionary[k] for k in forgetting_dictionary if presentations_needed_to_learn[k] <= 30} # 11, 12, 13

        # logger.warning(f"Forgetting threshold is ({cur_iter})")
        forgettable_inds = [key for key, value in forgetting_dictionary.items() if value >= 1]
        # unforgettable_inds = [key for key, value in forgetting_dictionary.items() if value == 0]
        
        forgettable_list = [item for item in self.streamed_list if item['original_index'] in forgettable_inds]
        # unforgettable_list = [item for item in self.streamed_list if item['original_index'] in unforgettable_inds]

        # unforgettable list 도 메모리에 추가?
        
        return forgettable_list, forgetting_dictionary


    def get_memory_forgetting_average(self, forgettable_dict):
        forgetting_values = []
        for sample in self.memory_list:
            original_index = sample['original_index']
            forgetting_value = forgettable_dict.get(original_index, 0)
            forgetting_values.append(forgetting_value)
        
        if len(forgetting_values) > 0:
            avg_forgetting = sum(forgetting_values) / len(forgetting_values)
        else:
            avg_forgetting = 0.0
        
        logger.info(f"Average forgettings in memory: {avg_forgetting:.2f}")
        return avg_forgetting


    def update_memory(self, cur_iter, num_class=None):
        if num_class is None:
            num_class = self.num_learning_class
            
        paced_list, forgettable_dict = self.order_examples_by_forgetting(self.example_stats, cur_iter)
        logger.info(f"Unforgettable samples : {len(paced_list)}")
        paced_df = pd.DataFrame(paced_list)
        logger.info(f"\n{paced_df.klass.value_counts(sort=True)}")

        if not self.already_mem_update:
            logger.info(f"Update memory over {num_class} classes by {self.mem_manage}")
            candidates = paced_list + self.memory_list
            if len(candidates) <= self.memory_size:
                self.memory_list = candidates
                self.seen = len(candidates)
                logger.warning("Candidates < Memory size")
            else:
                if self.mem_manage == "random":
                    self.memory_list = self.rnd_sampling(candidates)
                elif self.mem_manage == "reservoir":
                    self.reservoir_sampling(paced_list)
                elif self.mem_manage == "prototype":
                    self.memory_list = self.mean_feature_sampling(
                        exemplars=self.memory_list,
                        samples=paced_list,
                        num_class=num_class,
                    )
                elif self.mem_manage == "class_balanced":
                    self.memory_list = self.equal_class_sampling(
                            candidates, num_class
                        )
                elif self.mem_manage == "uncertainty":
                    if cur_iter == 0:
                        self.memory_list = self.equal_class_sampling(
                            candidates, num_class
                        )
                    else:
                        self.memory_list = self.uncertainty_sampling(
                            candidates,
                            num_class=num_class,
                        )
                else:
                    logger.error("Not implemented memory management")
                    raise NotImplementedError

            assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")
            
            # 메모리 업데이트 후 평균 forgetting 값 계산
            self.get_memory_forgetting_average(forgettable_dict)

            # memory update happens only once per task iteratin.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")


    def evaluation(self, cur_iter, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x)
                
                index_tensor = torch.tensor(list(range(i*self.batch_size, 
                                min(i*self.batch_size+self.batch_size, 
                                len(self.test_list)))), dtype=torch.int64).cuda()
                
                if cur_iter >= 1:
                    loss = criterion(logit, y, index_tensor, None)
                else:
                    loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret
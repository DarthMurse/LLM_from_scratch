from tokenizer import BPETokenizer
from dataclasses import dataclass
import mmap
import torch
from torch.utils.data import Dataset
from multiprocessing import Process, Queue, cpu_count

@dataclass
class DataConfig:
    seq_len: int = 257
    batch_size: int = 64

class TextDatasetBuilder:
    def __init__(self, file, save_path, tokenizer: BPETokenizer, config: Dataset):
        self.file_path = file
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.config = config

    def _single_tokenize(self, s, i, queue, cpu_num):
        print(f"Process {i} starts!")
        current_idx = 0
        string = []
        while current_idx >= 0 and current_idx < len(s):
            next_idx = s.find(self.tokenizer.eos_id, current_idx)
            if next_idx != -1:
                next_idx += len(self.tokenizer.eos_id)
            string += self.tokenizer.encode(s[current_idx: next_idx])
            current_idx = next_idx
            if i == cpu_num - 1:
                print(f"Process {i}: {current_idx / len(s) * 100} %", end='\r', flush=True)
        queue.put(string)
        print(f"Process {i} finishes!")

    def tokenize(self, s):
        result = []
        queue = Queue()
        cpu_num = cpu_count() // 2
        size = len(s)
        split_point = [0]
        for i in range(1, cpu_num):
            split_point.append(s.find(self.tokenizer.eos_id, int(i * size / cpu_num)) + len(self.tokenizer.eos_id))
        split_point.append(size)
        file_list = [s[split_point[i]: split_point[i+1]] for i in range(cpu_num)]
        print("File splitting complete!")

        processes = [Process(target=self._single_tokenize, args=(file_list[i], i, queue, cpu_num)) for i in range(cpu_num)]
        for i in range(cpu_num):
            processes[i].start()

        for i in range(cpu_num):
            result += queue.get()
        for i in range(cpu_num):
            processes[i].join()
        print("Tokenization complete!")
        return result

    def build(self):
        with open(self.file_path, "r+b") as f:
            mm = f.read()
            print("File loading complete!")
            result = self.tokenize(mm)
            row_len = int(len(result) / self.config.seq_len)
            cut_len = row_len * self.config.seq_len
            result = torch.LongTensor(result[: cut_len])
            result = result.reshape(row_len, self.config.seq_len)
            print(result.shape)
            torch.save(result, self.save_path)

class TextDataset(Dataset):
    def __init__(self, load_path, config: DataConfig):
        super().__init__()
        self.config = config
        self.data = torch.load(load_path)
        print(self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, 0: self.config.seq_len-1], self.data[idx, 1: self.config.seq_len]

if __name__ == "__main__":
    config = DataConfig(256+1, 64)
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    data_file = ["../data/TinyStoriesV2-GPT4-valid.txt", "../data/TinyStoriesV2-GPT4-train.txt"]
    save_path = ["../data/tiny_story_valid.pth", "../data/tiny_story_train.pth"]
    '''
    builder1 = TextDatasetBuilder(data_file[0], save_path[0], tokenizer, config)
    builder1.build()
    builder2 = TextDatasetBuilder(data_file[1], save_path[1], tokenizer, config)
    builder2.build()
    '''
    dataset1 = TextDataset(save_path[0], config)
    print(len(dataset1))
    #dataset2 = TextDataset(save_path[1], config)
    #print(len(dataset2))
    x, y = dataset1[190]
    print(tokenizer.decode(x.tolist()))
    print(tokenizer.decode(y.tolist()))

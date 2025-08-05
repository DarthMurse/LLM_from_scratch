from typing import List, Tuple, Dict, Set
import mmap
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Process, Queue
import time

class BPETokenizer:
    def __init__(self, merge_num=30000):
        self.merge_num = merge_num
        self.vocab = {}
        self.inverse_vocab = {}
        self.merge_list = []
        self.merge_rank = {}
        self.special_vocab = {b'<|endoftext|>': 0}
        self.vocab_size = self.merge_num + 256 + len(self.special_vocab.keys())
        self._init_byte_vocab()

    def _init_byte_vocab(self):
        special_token_num = len(self.special_vocab.keys())
        for i in range(256):
            self.vocab[bytes([i])] = i + special_token_num
            self.inverse_vocab[i + special_token_num] = bytes([i])
        for i in self.special_vocab.keys():
            self.vocab[i] = self.special_vocab[i]
            self.inverse_vocab[self.special_vocab[i]] = i

    def encode(self, s):
        pre_tokenized_list, _ = self.pretokenize(s)
        result = []
        for word in pre_tokenized_list:
            if len(word) > 1:
                result = result + self._merge_word(word)
            else:
                result = result + word
        return result

    def _merge_word(self, w):
        # w is a sequence of integers
        pair_rank = []
        word = w
        for i in range(len(word)-1):
            current_pair = tuple(word[i: i+2])
            if current_pair in self.merge_rank:
                pair_rank.append(self.merge_rank[current_pair])
            else:
                pair_rank.append(self.merge_num+1)

        min_rank = min(pair_rank)
        while min_rank <= self.merge_num:
            index = pair_rank.index(min_rank)
            word[index] = len(self.special_vocab) + 256 + min_rank
            for i in range(index+1, len(word)-1):
                word[i] = word[i+1]
            word.pop(-1)
            if index > 0:
                new_pair = tuple(word[index-1: index+1])
                if new_pair in self.merge_rank:
                    pair_rank[index-1] = self.merge_rank[new_pair]
                else:
                    pair_rank[index-1] = self.merge_num + 1
            if index+1 < len(word):
                new_pair = tuple(word[index: index+2])
                if new_pair in self.merge_rank:
                    pair_rank[index] = self.merge_rank[new_pair]
                else:
                    pair_rank[index] = self.merge_num + 1
            for i in range(index+1, len(word)-1):
                pair_rank[i] = pair_rank[i+1]
            pair_rank.pop(-1)
            if len(pair_rank) > 0:
                min_rank = min(pair_rank)
            else:
                min_rank = self.merge_num + 1

        return word

    def decode(self, int_array):
        result = b''
        for i in int_array:
            result = result + self.inverse_vocab[i]
        return result

    def pretokenize(self, s):
        # Split the text by space, enter and special tokens
        # The splitted words start with space, enter, letter, or a single special token
        # The returned words are converted to base tokens and special tokens
        i = 0
        pre_tokenized_list = []
        counter = {}
        split_dict = {b' ': self.vocab[b' '], b'\n': self.vocab[b'\n']} | self.special_vocab
    
        current_index = 0
        next_start, next_end = self._get_next_special(s, current_index)
        while next_start != -1:
            if current_index == next_start:
                num = [split_dict[s[next_start: next_end]]]
                pre_tokenized_list.append(num)
                if tuple(num) in counter:
                    counter[tuple(num)] += 1
                else:
                    counter[tuple(num)] = 1
                current_index = next_end 
            else:
                word = []
                for i in range(current_index, next_start):
                    word.append(self.vocab[s[i: i+1]])
                pre_tokenized_list.append(word)
                if tuple(word) in counter:
                    counter[tuple(word)] += 1
                else:
                    counter[tuple(word)] = 1
                current_index = next_start
            next_start, next_end = self._get_next_special(s, current_index)
        # address the ending.
        word = []
        for i in range(current_index, len(s)):
            word.append(self.vocab[s[i: i+1]])
        if len(word) > 0:
            pre_tokenized_list.append(word)
            if tuple(word) in counter:
                counter[tuple(word)] += 1
            else:
                counter[tuple(word)] = 1

        return pre_tokenized_list, counter

    def pretokenize_multiprocess(self, s, queue):
        _, counter = self.pretokenize(s)
        queue.put(counter)

    def _get_next_special(self, s, start):
        special_list = list(self.special_vocab.keys())
        blank_list = [b' ', b'\n']
        total_list = special_list + blank_list 
        next_start = -1
        next_end = -1
        for key in total_list:
            next_index = s.find(key, start)
            if next_index != -1:
                if next_start == -1:
                    next_start = next_index
                    next_end = next_start + len(key)
                elif next_index < next_start:
                    next_start = next_index 
                    next_end = next_start + len(key)
        return next_start, next_end

    def save(self, path):
        vocab_for_save = {k.decode("utf-8", errors="backslashreplace"): v for k, v in self.vocab.items()}
        special_vocab_for_save = {k.decode("utf-8", errors="backslashreplace"): v for k, v in self.special_vocab.items()}
        data = {
                "vocab_size": self.vocab_size,
                "special_vocab": special_vocab_for_save,
                "vocab": vocab_for_save,
                "merge_list": self.merge_list
                }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab = {k.encode("utf-8"): v for k, v in data['vocab'].items()}
            self.vocab_size = data['vocab_size']
            self.special_vocab = {k.encode("utf-8"): v for k, v in data['special_vocab'].items()}
            self.merge_list = data['merge_list']
            self.merge_num = len(self.merge_list)
            for key in self.vocab.keys():
                self.inverse_vocab[self.vocab[key]] = key 
            for i in range(len(self.merge_list)):
                self.merge_rank[tuple(self.merge_list[i])] = i

    def train(self, txt_file, core_num=100):
        with open(txt_file, 'r+b') as f:
            print("Reading txt file ...")
            #mm = mmap.mmap(f.fileno(), 0)
            mm = f.read() # Assume the file is small enough
            print("Pretokenizing context ...")
            # Prepare multi-process file
            size = len(mm)
            split_token = list(self.special_vocab.keys()) + [b' ', b'\n']
            split_point = [0]
            for i in range(1, core_num):
                pos = int(i / core_num * size)
                next_split = []
                for token in split_token:
                    token_pos = mm.find(token, pos)
                    if token_pos != -1:
                        next_split.append(token_pos)
                if len(next_split) > 0:
                    next_pos = min(next_split)
                else:
                    next_pos = pos
                split_point.append(next_pos)
            split_point.append(size)
            file_list = [mm[split_point[i]: split_point[i+1]] for i in range(core_num)]
            # Start multi-processing pretokenization
            queue = Queue(maxsize=core_num)
            processes = [Process(target=self.pretokenize_multiprocess, args=(file_list[i], queue)) for i in range(core_num)]
            for i in range(core_num):
                processes[i].start()

            word_count = {}
            for i in range(core_num):
                sub_counter = queue.get()
                for key, value in sub_counter.items():
                    if key in word_count:
                        word_count[key] += value
                    else:
                        word_count[key] = value

            for i in range(core_num):
                processes[i].join()

            #_, word_count = self.pretokenize(mm)
            print("Pretokenizing complete!")
            pair_count = {}
            pair_pos = {}
            
            for word in word_count.keys():
                for i in range(len(word) - 1):
                    pair = word[i: i+2]
                    if pair in pair_count:
                        pair_count[pair] += word_count[word]
                    else:
                        pair_count[pair] = word_count[word]
                    
                    if pair in pair_pos:
                        pair_pos[pair].add(word)
                    else:
                        pair_pos[pair] = {word}

            # Start merging
            print("Start merging ...")
            for i in tqdm(range(len(self.vocab), len(self.vocab) + self.merge_num)):
                # Add new token to vocab
                max_pair = max(pair_count, key=pair_count.get)
                #print(i, max_pair)
                self.merge_list.append(max_pair)
                new_token = self.inverse_vocab[max_pair[0]] + self.inverse_vocab[max_pair[1]]
                self.vocab[new_token] = i
                self.inverse_vocab[i] = new_token

                word_update_list = pair_pos.pop(max_pair)
                for word in word_update_list:
                    for j in range(0, len(word)-1):
                        if word[j: j+2] == max_pair:
                            if j-1 >= 0:
                                if word[j-1: j+1] != max_pair:
                                    pair_count[word[j-1: j+1]] -= word_count[word]
                                if pair_count[word[j-1: j+1]] == 0:
                                    pair_count.pop(word[j-1: j+1])
                                    pair_pos.pop(word[j-1: j+1])
                            if j+2 < len(word):
                                if j+4 > len(word) or word[j+2: j+4] != max_pair:
                                    if word[j+1: j+3] != max_pair:
                                        pair_count[word[j+1: j+3]] -= word_count[word]
                                else:
                                    double_new_token = tuple([i, i])
                                    if double_new_token in pair_count:
                                        pair_count[double_new_token] += word_count[word]
                                    else:
                                        pair_count[double_new_token] = word_count[word]
                                if pair_count[word[j+1: j+3]] == 0:
                                    pair_count.pop(word[j+1: j+3])
                                    pair_pos.pop(word[j+1: j+3])

                        if word[j: j+2] in pair_pos and word in pair_pos[word[j: j+2]]:
                            pair_pos[word[j: j+2]].remove(word)

                    merged_word = []
                    j = 0
                    while j < len(word):
                        if j != len(word)-1 and tuple(word[j: j+2]) == max_pair:
                            merged_word.append(i)
                            j += 1
                        else:
                            merged_word.append(word[j])
                        j += 1
                    merged_word = tuple(merged_word)

                    for j in range(len(merged_word)-1):
                        if merged_word[j: j+2] in pair_pos:
                            pair_pos[merged_word[j: j+2]].add(merged_word)
                            pair_count[merged_word[j: j+2]] += word_count[word]
                        else:
                            pair_pos[merged_word[j: j+2]] = {merged_word}
                            pair_count[merged_word[j: j+2]] = word_count[word]
                    word_count[merged_word] = word_count.pop(word)

                pair_count.pop(max_pair)

            for i in range(len(self.merge_list)):
                self.merge_rank[tuple(self.merge_list[i])] = i

if __name__ == "__main__":
    tokenizer = BPETokenizer(30000)
    start = time.time()
    tokenizer.train("../data/TinyStoriesV2-GPT4-valid.txt")
    duration = time.time() - start
    print(f"Total {duration} seconds past.")
    tokenizer.save("tokenizer.json")
    tokenizer.load("tokenizer.json")
    string = "Hello, I am a cuda programmer who is very good at optimizing performance.\n<|endoftext|>"
    bstring = string.encode('utf-8')
    int_array = tokenizer.encode(bstring)
    print(tokenizer.vocab[b' '], tokenizer.vocab[b'\n'])
    print(int_array)
    string = tokenizer.decode(int_array)
    print(string)

import torch

from model import ModelConfig, Transformer, softmax
from tokenizer import BPETokenizer
from dataset import TextDataset, DataConfig

def generate(model, input_ids, eos_id, temperature=0.9, top_p=0.9, max_new_token=200):
    # Assuming decoding a single batch of input_ids
    # input_ids: torch.LongTensor [1, S]
    out_token = []
    out = model(input_ids, inference=True)
    print("prefilling complete!")
    logits = out[0, -1] # [vocab_size]
    prob = softmax(logits / temperature)
    new_prob = top_p_transform(prob, top_p)
    index = torch.multinomial(prob, 1)
    out_token.append(index.item())
    while index != eos_id and len(out_token) < max_new_token:
        print(f"{len(out_token)} tokens generated.")
        new_token = torch.LongTensor([index]).to(input_ids.device).unsqueeze(0)
        out = model(new_token, inference=True)
        logits = out[0, -1] # [vocab_size]
        prob = softmax(logits / temperature)
        prob = top_p_transform(prob, top_p)
        index = torch.multinomial(prob, 1)
        out_token.append(index.item())
    return out_token

def top_p_transform(prob, top_p):
    values, indices = torch.sort(prob, dim=-1, descending=True)
    aux1 = torch.concat([torch.zeros([1], device=values.device), values], dim=-1)
    aux2 = torch.concat([values, torch.zeros([1], device=values.device)], dim=-1)
    cumsum = torch.cumsum(aux1, dim=-1)
    new_prob = torch.where(cumsum < top_p, aux2, 0)
    new_prob = new_prob[:-1]
    new_prob = new_prob / new_prob.sum()
    reorder_index = torch.argsort(indices)
    return new_prob[reorder_index]

def test(model, input_ids, tokenizer):
    logits = model(input_ids)
    logits = softmax(logits[0])
    out_token = []
    for i in range(logits.shape[0]):
        token = torch.multinomial(logits[i], 1)
        out_token.append(token.item())
    print(tokenizer.decode(input_ids[0].cpu().tolist()))
    print(tokenizer.decode(out_token))

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    eos_id = tokenizer.vocab[tokenizer.eos_id]
    config = ModelConfig()
    model = Transformer(config)
    ckpt = torch.load("../out/TinyStories_22M/iter_054682.ckpt")
    model.load_state_dict(ckpt["model"])
    model = model.to("cuda:1")
    '''
    data_config = DataConfig()
    dataset = TextDataset("../data/tiny_story_valid.pth", data_config)
    x, y = dataset[0]
    x = x.unsqueeze(0).to("cuda:1")
    test(model, x, tokenizer)
    '''
    string = "It is a lovely day and "
    input_ids = tokenizer.encode(string.encode("utf-8"))
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to("cuda:1")
    output_ids = generate(model, input_ids, eos_id)
    out_string = tokenizer.decode(output_ids)
    print(string)
    print(out_string)
    #'''

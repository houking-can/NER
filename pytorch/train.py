from utils import get_mask
import os
import codecs
import torch
from torch.nn.utils import clip_grad_norm_
from metrics import get_entities_bio, f1_score, classification_report


def train_model(dataloader, model, optimizer, batch_num, writer, use_gpu=False):
    model.train()
    for batch in dataloader:
        batch_num += 1
        model.zero_grad()
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        char_inputs = batch['char']
        char_inputs = char_inputs[word_perm_idx]
        char_dim = char_inputs.size(-1)
        char_inputs = char_inputs.contiguous().view(-1, char_dim)
        if use_gpu:
            batch_text = batch_text.cuda()
            batch_label = batch_label.cuda()
            char_inputs = char_inputs.cuda()
        mask = get_mask(batch_text)
        loss = model.neg_log_likelihood_loss(batch_text, seq_length, char_inputs, batch_label, mask)
        writer.add_scalar('loss', loss, batch_num)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    return batch_num


def evaluate(dataloader, model, word_vocab, label_vocab, output_path, prefix, use_gpu=False):
    model.eval()
    prediction = []
    trues_list = []
    preds_list = []
    for batch in dataloader:
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        char_inputs = batch['char']
        char_inputs = char_inputs[word_perm_idx]
        char_dim = char_inputs.size(-1)
        char_inputs = char_inputs.contiguous().view(-1, char_dim)
        if use_gpu:
            batch_text = batch_text.cuda()
            batch_label = batch_label.cuda()
            char_inputs = char_inputs.cuda()
        mask = get_mask(batch_text)
        with torch.no_grad():
            tag_seq = model(batch_text, seq_length, char_inputs, batch_label, mask)

        for line_tesor, labels_tensor, predicts_tensor in zip(batch_text, batch_label, tag_seq):
            for word_tensor, label_tensor, predict_tensor in zip(line_tesor, labels_tensor, predicts_tensor):
                if word_tensor.item() == 0:
                    break
                line = [word_vocab.id_to_word(word_tensor.item()), label_vocab.id_to_label(label_tensor.item()),
                        label_vocab.id_to_label(predict_tensor.item())]
                trues_list.append(line[1])
                preds_list.append(line[2])
                prediction.append(' '.join(line))
            prediction.append('')

    true_entities = get_entities_bio(trues_list)
    pred_entities = get_entities_bio(preds_list)
    print(len(trues_list), len(preds_list), len(prediction))

    results = {
        "f1": f1_score(true_entities, pred_entities),
        'report': classification_report(true_entities, pred_entities)
    }

    with open(os.path.join(output_path, '%s_pred.txt' % prefix), 'w', encoding='utf-8') as f:
        f.write('\n'.join(prediction))

    with open(os.path.join(output_path, '%s_score.txt' % prefix), "a") as writer:
        writer.write("***** Eval results {} *****\n".format(prefix))
        for key in sorted(results.keys()):
            if key == 'report_dict':
                continue
            writer.write("{} = {}\n".format(key, str(results[key])))

    return results["f1"]

import torch.optim
from matplotlib import pyplot as plt
import datasetloader
from model_modu import Model
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import argparse
import transformers
import random
import time

transformers.logging.set_verbosity_error()
parser = argparse.ArgumentParser(description="Train a sequence classifier - via Transformers")
parser.add_argument("--BERT", type=bool, help="You are using BERT embeddings", default=False)
parser.add_argument("--RoBERTa", type=bool, help="You are using RoBERTa embeddings", default=True)
parser.add_argument("--DistilBERT", type=bool, help="You are using DistilBERTa embeddings", default=False)
parser.add_argument("--train_data", type=str, help="this is the training file", default=datasetloader.gold_function()[0]) #10700
parser.add_argument("--test_data", type=str, help="this is the test file", default=datasetloader.gold_function()[1]) #1339
parser.add_argument("--val_data", type=str, help="this is the val file", default=datasetloader.gold_function()[2])

parser.add_argument("--epoch", type=int, help="this is the number of epochs", default=10)
parser.add_argument("--hiddens", type=int, help="this is the LSTM hidden_size", default=100)
parser.add_argument("--batch_size", type=int, help="number of samples in each iteration", default=10)
parser.add_argument("--lr", type=float, help="this is learning rate value", default=0.005)    # best performance 0.000005, 10 epoches
parser.add_argument("--num_labels", type=int, help="this is the total number of labels", default=5)
parser.add_argument("--max_length", type=int, help="this is maximum length of an utterance", default=200)
parser.add_argument("--lstm", type=bool, help="You are using LSTMs ", default=False)

parser.add_argument("--L1_reg", type=bool, help="L1 regularizer", default=False)
parser.add_argument("--L2_reg", type=bool, help="L2 regularizer", default=False)
parser.add_argument("--drop_out", type=bool, help="implement a dropout to the model output", default=True)
parser.add_argument("--L1_lambda", type=int, help="Lambda value used for regularization", default=0.01)
parser.add_argument("--L2_lambda", type=int, help="Lambda value used for regularization", default=0.1)
parser.add_argument("--dr", type=int, help="P value used for Dropout", default=0.3)

args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoder_model = BertModel.from_pretrained("bert-base-uncased")
embedding_dim = encoder_model.config.hidden_size
model_object = Model(embedding_dim,args.num_labels,args.max_length, args.hiddens, args.dr)
loss = CrossEntropyLoss()
if args.L2_reg: # between 0.0001 and 0.1; penalizes the sum of the squares of the model parameters
    optimizer = torch.optim.Adam(model_object.parameters(), lr=args.lr, weight_decay= args.L2_lambda)
else:
    optimizer = torch.optim.Adam(model_object.parameters(), lr=args.lr)
label_map = {'Joyful': 0, 'Scared': 1, 'Sad': 2, 'Neutral': 3, 'Excited': 4}


loss_records = []
def train_classifier():
    text, label = args.train_data
    text = text[:500]
    label = label[:500]
    patience = 0
    curr_loss = 0
    for epoch_indx in range(args.epoch):
        prev_loss = curr_loss
        epoch_loss = 0
        acc_epoch_record = []
        loss_epoch_record = []
        text,label = randomize(text,label)
        for batch_indx in tqdm(range(0, len(text), args.batch_size), desc= f"TRAINING DATASET: {epoch_indx+1}/{args.epoch}"):
            correct = 0
            batch_encoding = tokenizer.batch_encode_plus(
                text[batch_indx:batch_indx + args.batch_size], padding='max_length', truncation=True,
                max_length=args.max_length, return_tensors='pt', text_pair=True, token_type_ids=True)
            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding['attention_mask']
            out = encoder_model(input_ids)
            embeddings = out.last_hidden_state
            predicted = model_object.forward(embeddings, attention_mask, args.lstm)
            gold_label_tensor = torch.tensor(label[batch_indx:batch_indx + args.batch_size])
            preds = torch.argmax(predicted, dim=1)
            for indx in range(len(gold_label_tensor)):
                if preds[indx] == gold_label_tensor[indx]:
                    correct += 1
            acc_epoch_record.append(correct/args.batch_size)
            loss_value = loss(predicted, gold_label_tensor)
            if args.L1_reg: # 0.01 and 0.5, since L1 regularization tends to be more aggressive at shrinking the weights towards zero.
                for param in model_object.parameters(): # penalizes the sum of the absolute values of the model parameters
                    loss_value += torch.sum(torch.abs(param)) * args.L1_lambda
            loss_epoch_record.append(loss_value.item())
            epoch_loss += loss_value.item()
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Ave TRAIN acc: Epoch {epoch_indx + 1}: {sum(acc_epoch_record)/len(acc_epoch_record)}")
        print(f"Ave TRAIN loss: Epoch {epoch_indx + 1}: {sum(loss_epoch_record)/len(loss_epoch_record)}")
        loss_records.append(epoch_loss)
        eval_txt, eval_lbl = args.val_data
        eval_txt = eval_txt[:100]
        eval_lbl = eval_lbl[:100]
        eval_txt, eval_lbl = randomize(eval_txt, eval_lbl)
        acc,ave_loss = evaluation(model_object, eval_txt, eval_lbl)
        print(f"Ave DEV acc: Epoch {epoch_indx+1}: {acc}")
        print(f"Ave DEV loss: Epoch {epoch_indx + 1}: {ave_loss}")
        curr_loss = ave_loss
        if curr_loss >= prev_loss: # Implementing Early Stopping
            patience+=1
            if patience>2:
                test_txt, test_lbl = args.test_data
                test_txt = test_txt[:100]
                test_lbl = test_lbl[:100]
                acc, ave_loss = evaluation(model_object, test_txt, test_lbl)
                print(f"Ave TEST acc: {acc}")
                print(f"Ave TEST loss: Epoch {epoch_indx + 1}: {ave_loss}")
                return
        else:
            patience=0
    test_txt, test_lbl = args.test_data
    test_txt = test_txt[:100]
    test_lbl = test_lbl[:100]
    acc, ave_loss = evaluation(model_object, test_txt, test_lbl )
    print(f"Ave TEST acc: {acc}")
    print(f"Ave TEST loss: Epoch {epoch_indx + 1}: {ave_loss}")

def evaluation(model,text,label):
    correct = 0
    total = len(label)
    ave_loss_epoch = []
    with torch.no_grad():
        for batch_indx in range(0, len(label), args.batch_size):
        # for batch_indx in tqdm(range(0, len(label), args.batch_size), desc=f" DEV/TEST DATASET: "):
            batch_encoding = tokenizer.batch_encode_plus(
                text[batch_indx:batch_indx + args.batch_size], padding='max_length',max_length=args.max_length, truncation=True, return_tensors='pt', text_pair=True)
            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding['attention_mask']
            out = encoder_model(input_ids)
            embeddings = out.last_hidden_state
            predicted = model.forward(embeddings, attention_mask, args.lstm)
            gold_label_list = label[batch_indx:batch_indx + args.batch_size]
            gold_label_tensor = torch.tensor(gold_label_list)
            loss_value = loss(predicted, gold_label_tensor)
            ave_loss_epoch.append(loss_value)
            preds = torch.argmax(predicted, dim=1)
            for indx in range(len(gold_label_list)):
                if preds[indx] == gold_label_list[indx]:
                    correct += 1
        return correct / total, sum(ave_loss_epoch)/len(ave_loss_epoch)

def plotting(records):
    batchList = [i for i in range(len(records))]
    plt.plot(batchList, records, linewidth=5, label="Loss variation")
    plt.xlabel("Batch", color="green", size=20)
    plt.ylabel("Loss", color="green", size=20)
    plt.title("Progress Line for BERT Model", size=20)
    plt.grid()
    plt.show()

def randomize(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    shuffled_list1, shuffled_list2 = zip(*combined)
    return shuffled_list1,shuffled_list2

if __name__ == '__main__':
    start_time = time.time()
    train_classifier()
    plotting(loss_records)
    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S", time.gmtime(seconds)))


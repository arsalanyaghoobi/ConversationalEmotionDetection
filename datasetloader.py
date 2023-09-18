import json
import pickle
import random
from collections import defaultdict
from collections import defaultdict,Counter
from sklearn.model_selection import train_test_split

label_map = {'Joyful': 0, 'Scared': 1, 'Sad': 2, 'Neutral': 3, 'Excited': 4}
def IEMOCAP():
    path = r"dataset\IEMOCAP\archive\IEMOCAP_features.pkl"
    videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(open(path, 'rb'), encoding='latin1')
    IDs = videoIDs.keys()
    sequence_labels = [videoLabels[id] for id in IDs]
    sequence = [videoSentence[id] for id in IDs]
    result = list(zip(sequence, sequence_labels))
    GOLD_list = []
    for touple_index in range(len(result)):
        target_tuple = result[touple_index]
        text = target_tuple[0]
        label = target_tuple[1]
        final = list(zip(text, label))
        GOLD_list.append(final)

    NEW_GOLD_LIST = []
    for tuples in GOLD_list:
        new_tuples = []
        for (txt, lbl) in tuples:
            if lbl == 3 or lbl == 5 or lbl == 1 :
                lbl = "Sad"
            elif lbl == 0:
                lbl = "Joyful"
            elif lbl == 2:
                lbl = "Neutral"
            elif lbl == 4:
                lbl = "Excited"
            new_tuples.append((txt, lbl))
        NEW_GOLD_LIST.append(new_tuples)
    return NEW_GOLD_LIST


def dailyDialogue():
    BRONZE_LIST = []
    text_path = r"dataset\dailyDialogue\ijcnlp_dailydialog\ijcnlp_dailydialog\dialogues_text.txt"
    emotion_path = r"dataset\dailyDialogue\ijcnlp_dailydialog\ijcnlp_dailydialog\dialogues_emotion.txt"
    with open(text_path, encoding="UTF-8")as text:
        text = text.readlines()
    with open(emotion_path, encoding="UTF-8") as emotion:
        emotions = emotion.readlines()

    final_result = list(zip(text, emotions))
    for txt, lbls in final_result:
        text_split = txt.split("__eou__")
        lbls_split = [int(x) for x in lbls.split()]
        BRONZE_LIST.append(list(zip(text_split, lbls_split)))

    GOLD_LIST=[]
    for conversation in BRONZE_LIST:
        my_list = []
        for utterance in conversation:
            label_index = utterance[1]
            sequence = utterance[0]
            if label_index==5 or label_index==2 or label_index==1 :
                label_index= "Sad"
                my_list.append((sequence, label_index))
            elif label_index==4:
                label_index = "Joyful"
                my_list.append((sequence, label_index))
            elif label_index==3:
                label_index = "Scared"
                my_list.append((sequence, label_index))
            elif label_index==0:
                label_index = "Neutral"
                my_list.append((sequence, label_index))
            elif label_index==6:
                label_index = "Excited"
                my_list.append((sequence, label_index))
            else:
                my_list.append((sequence, label_index))
        GOLD_LIST.append(my_list)
    return GOLD_LIST


def friends():
    path = r"dataset\friends-corpus\utterances.jsonl"
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    id_text_label_dict = defaultdict(list)
    id_set = set()
    for file in json_list:
        each_conv = json.loads(file)
        id = each_conv["id"][:-5]
        text = each_conv["text"]
        if len(text)!=0:
            emotion = each_conv["meta"]["emotion"]
            if emotion==None:
                emotion="None"
                tple = (text, emotion)
            else:
                tple =(text,emotion[0])
            id_text_label_dict[id].append(tple)
            id_set.add(id)

    Total_conv = []
    for ids in id_set:
        text_label_tuples = id_text_label_dict[ids]
        id_conv = []
        for (text, label) in text_label_tuples:
            if label == "Mad":
                label = "Sad"
                my_tuple = (text,label)
            elif label == "Peaceful" or label == "Powerful":
                label = "Neutral"
                my_tuple = (text, label)
            else:
                my_tuple = (text,label)
            id_conv.append(my_tuple)
        Total_conv.append(id_conv)

    LAST_CONV = []
    for conv in Total_conv:
        Non_None_convs = []
        if conv[0][1] != "None":
            Non_None_convs.append((conv[0][0], conv[0][1]))
        for indx in range(1,len(conv)):
            curr_conv = conv[indx]
            prev_conv = conv[indx-1]
            if curr_conv[1] != "None" and prev_conv[1] != "None":
                curr_text = curr_conv[0]
                curr_lbl = curr_conv[1]
                Non_None_convs.append((curr_text,curr_lbl))
        if len(Non_None_convs)>0:
            LAST_CONV.append(Non_None_convs)
    return LAST_CONV


def data_split(dataset):
    train, non_train = train_test_split(dataset, train_size=0.8, shuffle=False, random_state=46)
    test, val = train_test_split(non_train, train_size=0.5, shuffle=False, random_state=46)
    # print(len(train),len(test),len(val))
    return train, test, val


def equalizer(data):
    # 1700
    text = []
    label = []
    label_count = {'Joyful': 0, 'Scared': 0, 'Sad': 0, 'Neutral': 0, 'Excited': 0}
    # {'Joyful': 14951, 'Scared': 1678, 'Sad': 8241, 'Neutral': 81268, 'Excited': 2714}
    for conv in data:
        for indx in range(1, len(conv)):
            current_utterance = conv[indx][0]
            current_utterance_label = conv[indx][1]
            prev_utterance = conv[indx - 1][0]
            prev_utterance_label = conv[indx - 1][1]
            if label_count[current_utterance_label] < 3000:
                label_count[current_utterance_label] += 1
                final_utterance = (prev_utterance_label + " " + prev_utterance, current_utterance)
                # final_utterance = (prev_utterance, current_utterance)
                # final_utterance = (prev_utterance_label + " " + prev_utterance, prev_utterance_label + " " +current_utterance)
                text.append(final_utterance)
                label.append(label_map[current_utterance_label])
            # label_count[current_utterance_label] += 1
            # final_utterance = (prev_utterance_label + " " + prev_utterance, current_utterance)
            # text.append(final_utterance)
            # label.append(label_map[current_utterance_label])
    list_for_shuffle = list(zip(text, label))
    random.shuffle(list_for_shuffle)
    text, label = zip(*list_for_shuffle)
    return text, label


def gold_function():
    overal_dataset = []
    dataset_1 = IEMOCAP()
    dataset_2 = dailyDialogue()
    dataset_3 = friends()
    for conv in dataset_1:
        overal_dataset.append(conv)
    for conv in dataset_2:
        overal_dataset.append(conv)
    for conv in dataset_3:
        overal_dataset.append(conv)

    random.shuffle(overal_dataset)
    random.shuffle(overal_dataset)
    random.shuffle(overal_dataset)
    text, label = equalizer(overal_dataset)
    text_train, text_test, text_val = data_split(text)
    label_train, label_test, label_val = data_split(label)
    gold_train = (text_train, label_train)
    gold_test = (text_test, label_test)
    gold_val = (text_val, label_val)
    return gold_train, gold_test, gold_val


#
#
# if __name__ == '__main__':
#     gold_function()



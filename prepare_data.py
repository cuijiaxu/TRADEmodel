import random
import copy
import json
import re,pprint
import pickle
from config import *
from collections import Counter
from collections import OrderedDict
from fix_label import *
from embeddings import GloveEmbedding, KazumaCharEmbedding

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

class TradeDataset:
    def __init__(self):
        with open('data.json') as f:
            self.data = json.load(f)

        self.train_data = self.data[1000:]
        self.test_data = self.data[:1000]
        self.batch_id = 0
        #prepare_data_seq(True, args['task'], False, batch_size=int(32))


    def get_batch(self, batch_size=20):
        if self.batch_id+batch_size>=len(self.train_data):
            self.batch_id = 0
            random.shuffle(self.train_data)
        batch_data = self.train_data[self.batch_id:self.batch_id + batch_size]
        batch_source = copy.deepcopy(batch_data)
        batch_target = copy.deepcopy(batch_data)
        self.batch_id = self.batch_id + batch_size
        return batch_source, batch_target

    def get_test_data(self):
        batch_data = self.test_data
        batch_source = copy.deepcopy(batch_data)
        batch_target = copy.deepcopy(batch_data)
        return batch_source, batch_target

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split():
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
                for v in value.split():
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS

def prepare_data_seq(training, task="dst", sequicity=0, batch_size=100):
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    file_train = 'data/train_dials.json'
    file_dev = 'data/dev_dials.json'
    file_test = 'data/test_dials.json'
    # Create saving folder
    if args['path']:
        folder_name = args['path'].rsplit('/', 2)[0] + '/'
    else:
        folder_name = 'save/{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/'
    print("folder_name", folder_name)
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
    # load domain-slot pairs from ontology
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    gating_dict = {"ptr":0, "dontcare":1, "none":2}
    # Vocabulary
    lang, mem_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')
    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

    if training:
        pair_train, train_max_len, slot_train = read_langs(file_train, gating_dict, ALL_SLOTS, "train", lang, mem_lang, sequicity, training)   
        #print(pair_train)   
        nb_train_vocab = lang.n_words
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        if os.path.exists(folder_name+lang_name) and os.path.exists(folder_name+mem_lang_name):
            print("[Info] Loading saved lang files...")
            with open(folder_name+lang_name, 'rb') as handle: 
                lang = pickle.load(handle)
            with open(folder_name+mem_lang_name, 'rb') as handle: 
                mem_lang = pickle.load(handle)
        else:
            print("[Info] Dumping lang files...")
            with open(folder_name+lang_name, 'wb') as handle: 
                pickle.dump(lang, handle)
            with open(folder_name+mem_lang_name, 'wb') as handle: 
                pickle.dump(mem_lang, handle)
        emb_dump_path = 'data/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    else:
        #print('not training')
        with open(folder_name+lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        

    # max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    # print("Read %s pairs train" % len(pair_train))
    # print("Read %s pairs dev" % len(pair_dev))
    # print("Read %s pairs test" % len(pair_test))  
    # print("Vocab_size: %s " % lang.n_words)
    # print("Vocab_size Training %s" % nb_train_vocab )
    # print("Vocab_size Belief %s" % mem_lang.n_words )
    # print("Max. length of dialog words for RNN: %s " % max_word)
    # print("USE_CUDA={}".format(USE_CUDA))

    # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    # print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
    # print(SLOTS_LIST[2])
    # print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
    # print(SLOTS_LIST[3])
    # LANG = [lang, mem_lang]
    # return train, dev, test, test_4d, LANG, SLOTS_LIST, gating_dict, nb_train_vocab

def read_langs(file_name, gating_dict, SLOTS, dataset, lang, mem_lang, sequicity, training, max_line = None):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {} 
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first 
        for dial_dict in dials:
            if (args["all_vocab"] or dataset=="train") and training:
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')
        # determine training data ratio, default is 100%
        if training and dataset=="train" and args["data_ratio"]!=100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials)*0.01*args["data_ratio"])]
        
        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
               (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]): 
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                turn_uttr_strip = turn_uttr.strip()
                dialog_history +=  (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                source_text = dialog_history.strip()
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

                if (args["all_vocab"] or dataset=="train") and training:
                    mem_lang.index_words(turn_belief_dict, 'belief')

                class_label, generate_y, slot_mask, gating_label  = [], [], [], []
                start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys(): 
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])
                
                data_detail = {
                    "ID":dial_dict["dialogue_idx"], 
                    "domains":dial_dict["domains"], 
                    "turn_domain":turn_domain,
                    "turn_id":turn_id, 
                    "dialog_history":source_text, 
                    "turn_belief":turn_belief_list,
                    "gating_label":gating_label, 
                    "turn_uttr":turn_uttr_strip, 
                    'generate_y':generate_y
                    }
                data.append(data_detail)
                
                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split()) #len(source_text) = 2 * num(system_transcript+transcript)
            
            #??????
            cnt_lin += 1
            if(max_line and cnt_lin>=max_line):
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len-1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    print("domain_counter", domain_counter)
    return data, max_resp_len, slot_temp


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)






if __name__ == '__main__':
    dataset = TradeDataset()
    pprint.pprint(dataset.get_batch())

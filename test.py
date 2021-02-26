from tensorflow.keras.models import load_model
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk import pos_tag
from wordfreq import zipf_frequency
import numpy as np
import pickle

model_cwi = load_model('model_CWI_full.h5')

with open('train.pickle', 'rb') as f:
    pickle1 = pickle.load(f)
word2index = pickle1['word2index']  
sent_max_length  = pickle1['sent_max_length']

# Now, let´s define some useful functions in order to use the CWI with some out of samples sentences
# Function for clean the data and remove non characters symbols
stop_words_ = set(stopwords.words('english'))
def cleaner(word):
  #Remove links
  word = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 
                '', word, flags=re.MULTILINE)
  word = re.sub('[\W]', ' ', word)
  word = re.sub('[^a-zA-Z]', ' ', word)
  return word.lower().strip()

# Function for to create the padded sequence
def process_input(input_text):
  input_text = cleaner(input_text)
  clean_text = []
  index_list =[]
  input_token = []
  index_list_zipf = []
  for i, word in enumerate(input_text.split()):
    if word in word2index:
      clean_text.append(word)
      input_token.append(word2index[word])
    else:
      index_list.append(i)
  input_padded = pad_sequences(maxlen=sent_max_length, sequences=[input_token], padding="post", value=0)
  return input_padded, index_list, len(clean_text)

def complete_missing_word(pred_binary, index_list, len_list):
  list_cwi_predictions = list(pred_binary[0][:len_list])
  for i in index_list:
    list_cwi_predictions.insert(i, 0)
  return list_cwi_predictions

# Second part: The Candidates generation and selection using BERT
# Load the BERT model for masked languge


bert_model = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertForMaskedLM.from_pretrained(bert_model)
model.eval()  

zipf_frequency('stop', 'en')
zipf_frequency('thwart', 'en')

# Now the function to get the candidates out of BERT (MLM):
def get_bert_candidates(input_text, list_cwi_predictions, numb_predictions_displayed = 10):
  list_candidates_bert = []
  for word,pred  in zip(input_text.split(), list_cwi_predictions):
    if (pred and (pos_tag([word])[0][1] in ['NNS', 'NN', 'VBP', 'RB', 'VBG','VBD' ]))  or (zipf_frequency(word, 'en')) <3.1:
      replace_word_mask = input_text.replace(word, '[MASK]')
      text = f'[CLS]{replace_word_mask} [SEP] {input_text} [SEP] '
      tokenized_text = tokenizer.tokenize(text)
      masked_index = [i for i, x in enumerate(tokenized_text) if x == '[MASK]'][0]
      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
      segments_ids = [0]*len(tokenized_text)
      tokens_tensor = torch.tensor([indexed_tokens])
      segments_tensors = torch.tensor([segments_ids])
      # Predict all tokens
      with torch.no_grad():
          outputs = model(tokens_tensor, token_type_ids=segments_tensors)
          predictions = outputs[0][0][masked_index]
      predicted_ids = torch.argsort(predictions, descending=True)[:numb_predictions_displayed]
      predicted_tokens = tokenizer.convert_ids_to_tokens(list(predicted_ids))
      list_candidates_bert.append((word, predicted_tokens))
  return list_candidates_bert



# Simplifying new sentences:
# Given a list of new sentences with complex words:
list_texts = [ 
 'The Risk That Students Could Arrive at School With the Coronavirus As schools grapple with how to reopen, new estimates show that large parts of the country would probably see infected students if classrooms opened now.',
 'How a photograph of a young man cradling his dying friend sent me on a journey across India.',
 'Pro-democracy parties, which had hoped to ride widespread discontent to big gains, saw the yearlong delay as an attempt to thwart them.',
 'Night after night, calm gave way to chaos. See what happened between the protesters and the federal agents.',
 'Contact Tracing Is Failing in Many States. Here is Why. Inadequate testing and protracted delays in producing results have crippled tracking and hampered efforts to contain major outbreaks.',
 'After an initial decrease in the youth detention population, the rate of release has slowed, and the gap between white youth and Black youth has grown.'
 'A laboratory experiment hints at some of the ways the virus might elude antibody treatments. Combining therapies could help, experts said.',
 'Though I may not be here with you, I urge you to answer the highest calling of your heart and stand up for what you truly believe.',
 'The research does not prove that infected children are contagious, but it should influence the debate about reopening schools, some experts said.',
 'Dropping antibody counts are not a sign that our immune system is failing against the coronavirus, nor an omen that we can not develop a viable vaccine.',
 'The Senate majority leader has said he will not approve a stimulus package without a “liability shield,” but top White House officials say they do not see it as essential.',
 'Campaign efforts to refocus come as the president continues to push divisive messages that have frustrated his own party.'
]   

# We apply the simplifier to see how it is performing:

for input_text in list_texts:
  new_text = input_text
  input_padded, index_list, len_list = process_input(input_text)
  pred_cwi = model_cwi.predict(input_padded)
  pred_cwi_binary = np.argmax(pred_cwi, axis = 2)
  complete_cwi_predictions = complete_missing_word(pred_cwi_binary, index_list, len_list)
  bert_candidates =   get_bert_candidates(input_text, complete_cwi_predictions)
  for word_to_replace, l_candidates in bert_candidates:
    tuples_word_zipf = []
    for w in l_candidates:
      if w.isalpha():
        tuples_word_zipf.append((w, zipf_frequency(w, 'en')))
    tuples_word_zipf = sorted(tuples_word_zipf, key = lambda x: x[1], reverse=True)
    new_text = re.sub(word_to_replace, tuples_word_zipf[0][0], new_text) 
  print("Original text: ", input_text )
  print("Simplified text:", new_text, "\n")
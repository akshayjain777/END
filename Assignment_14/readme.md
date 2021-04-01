# Transformer Based Model Python Code Generator

Capstone project is to write a transformer-based model that can write python code (with proper whitespace indentations).

## What is Transformer Network?

The transformer model is based entirely on the attention mechanism and completely gets
rid of recurrence. The transformer uses a special type of attention mechanism called self-attention.

The transformer consists of an encoder-decoder architecture. We feed the input sentence (source
sentence) to the encoder. The encoder learns the representation of the input sentence and
sends the representation to the decoder. The decoder receives the representation learned by
the encoder as input and generates the output sentence (target sentence).

To process a sentence we need these 3 steps:

1. Word embeddings of the input sentence are computed simultaneously.
2. Positional encodings are then applied to each embedding resulting in word vectors that also include positional information.
3. The word vectors are passed to the first encoder block.

## Transformer Encoder



Each block consists of the following layers in the same order:

1. A multi-head self-attention layer to find correlations between each word
2. A [normalization](https://theaisummer.com/normalization/) layer
3. A residual connection around the previous two sublayers
4. A linear layer
5. A second normalization layer
6. A second residual connection


### Self-attention mechanism

Consider the following sentence: A dog ate the meat because it was hungry

By reading the above statement, we can easily understand pronoun “it” relates to “dog” rather that meat. But how a model automatically understand this?

Model takes representation of each such word and then relate with all order it to understand which other words are strongly related to it. So, that’s how it will understand.

## Transformer decoder

1. The output sequence is fed in its entirety and word embeddings are computed
2. Positional encoding are again applied
3. And the vectors are passed to the first Decoder block

Each decoder block includes:

1. A **Masked** multi-head self-attention layer
2. A normalization layer followed by a residual connection
3. A new multi-head attention layer (known as **Encoder-Decoder attention**)
4. A second normalization layer and a residual connection
5. A linear layer and a third residual connection


## Output of Transformer

The output probabilities predict the next token in the output sentence. How? In essence, we assign a probability to each word in the French language and we simply keep the one with the highest score.



## Dataset

You can find the dataset [here (Links to an external site.)](https://github.com/piyushjain220/TSAI/blob/main/NLP/Capstone/english_python_data.txt). There are some 4600+ examples of English text to python code. 



## Data Preparation/preprocessing Strategy

Input file is quite messy and hence there are some of the areas where data specific additional conditions been added to address them.

I have developed and used following function to read from the text file shared as part of this capstone project.

Salient points of my data processing and Input file preparations are

- Any lines been started with # has been marked as Question
- Subsequent lines after Questions been marked as Answer for the corresponding question
- Have ensure Each line of Python code is separated by newline
- There are some questions "#24. Python Program to Find Numbers Divisible by Another Number" and have written custom logic to remove #24 by adding string checking of # and digit validation to strip off 24.
- Above logic has given me a two list and they are Question and Answers

```
datas = []
temp = {'input':'','output':''}
with open ('english_python_data.txt', 'r') as openfile:
  lines = openfile.readlines ()
  #print (line)
for line in lines:
  #print (line)
  try:
    if line[0] == '#' or line[1] == '#':
      datas.append (temp)
      temp = {'input':'','output':''}
      temp['input'] = re.sub('^[\d]+', '', line.replace ('\n', '').replace('#','').rstrip().lstrip()).rstrip().lstrip()
    else:
      temp['output'] += line.replace ('\n', '<new_line>').replace (' ', '<space>')
  except:
    pass

n = len (datas)
t = int (n * 0.8)
v = int (n * 0.9)
train = datas
val = datas[t:v]
test = datas[v:]

counter = 0
with open ('train.csv', 'w') as openfile:
  for element in train:
    if element['input'] == '' or element['output'] == '':
      continue
    openfile.write ('"' + element['input'] + '","' + element['output'] + '"\n')
with open ('val.csv', 'w') as openfile:
  for element in val:
    if element['input'] == '' or element['output'] == '':
      continue
    openfile.write ('"' + element['input'] + '","' + element['output'] + '"\n')
with open ('test.csv', 'w') as openfile:
  for element in test:
    if element['input'] == '' or element['output'] == '':
      continue
    openfile.write ('"' + element['input'] + '","' + element['output'] + '"\n')

```

Further data process has following logic

- My two list have now data
  - Length of Question:4850
  - Length of Answer:4850
- Have then converted the list into dataframe by following code and then saved the file as CSV because my plan is to use pytorch tabulardataset function with CSV file extension

```
import pandas as pd
df_Question = pd.DataFrame(Question, columns =['Question']) 
df_Answer = pd.DataFrame(Answer,columns =['Answer']) 
frames = [df_Question, df_Answer]
combined_question_answer = pd.concat(frames,axis=1)
```

- I have set max_length as 500 and removed approx 540 record as their length were more.

```
combined_question_answer_df = combined_question_answer[combined_question_answer['AnswerLen'] < 495] 
```

## Embedding Strategy

I did my experimentation on Tokenizer and Embedding.  All my  embedding strategy are shared below. Have decided to use separate tokenizer for Question and Answers. Question is standard English text and standard spacy tokenizer works great while Answer is having python code which requires special charecter handling.

```
def tokenize_de (line):
  split_points = ['<space>', ':', '<new_line>', '(', ')', '[', ']', '>', '<', '=', "'", '"', '.', '%', ',', '+', '-', '*', '\t', '{', '}', '/']

  words = []
  temp = ''
  n = len(line)
  i = 0

  while i < n:
    #print (i, line[i], temp, words)
    if line[i:i+7] == '<space>':
      if temp != '':
        words.append (temp)
        temp = ''
      words.append ('<space>')
      i = i + 7

    elif line[i:i+10] == '<new_line>':
      if temp != '':
        words.append (temp)
        temp = ''
      words.append ('<new_line>')
      i = i + 10

    elif line[i] in split_points:
      if temp != '':
        words.append (temp)
        temp = ''
      words.append (line[i])
      i += 1
    else:
      temp += line[i]
      i += 1
  return words

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
```

```
SRC = Field(tokenize = tokenize_en, 
            eos_token = '<eos>',
            init_token = '<sos>',
            batch_first = True)

TRG  = Field(tokenize = tokenize_de, 
            eos_token = '<eos>',
            init_token = '<sos>',
            batch_first = True)

fields = [("Question", SRC), ("Answer", TRG)]
```


## Metrices

I tried few other loss function but then stayed back with Cross Entropy as it has given me better result.

```
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
```

## Final Model

### Model with TEXT.build_vocab(train_data, min_freq = 1) including custom tokenizer for Python Code

Have kept min_freq to 1 and this has helped me to get rid of <unk>. Model output has been kept as default one with 3 Encoder and Decoder layer. I have also used custom tokenizer for Python code to ensure special characters' are handled properly.

```
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
```

Model PPL value on test data looks to be best among all my other model experimentation.

```
| Test Loss: 0.709 | Test PPL:   2.032 |
```

Python Code Generated also looks better.

```
Question: Write a function to calculate Volume of Hexagonal Pyramid
Source Python:
def volumeHexagonal ( a , b , h ) : 
     return a * b * h


Target Python:
def cal_area_ellipse ( minor , major ) : 
     pi = 3 . 14 
     return pi * ( minor * major ) 
#########################################################################################################
#########################################################################################################
Question: write a python program to multiply two list with list comprehensive
Source Python:
l1 = [ 1 , 2 , 3 ] 
 l2 = [ 4 , 5 , 6 ] 
 print ( [ x * y for x in l1 for y in l2 ] )


Target Python:
aList = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] 
 aList =   [ x * x for x in aList ] 
 print ( aList ) 
#########################################################################################################
#########################################################################################################
Question: Write a Python Program to Find the Intersection of Two Lists
Source Python:
def main ( alist , blist ) : 
     def intersection ( a , b ) : 
         return list ( set ( a ) & set ( b ) ) 
     return intersection ( alist , blist )


Target Python:
list1 = [ 1 , 2 , 3 , 4 , 5 ] 
 list2 = [ 5 , 6 , 7 , 8 , 7 ] 
 final = [ a * b for a in list1 for a in list2 ] 
 print ( f " Product of every pair of numbers from two lists : { final } " ) 
#########################################################################################################
#########################################################################################################
Question: write a program to print count of number of unique matching characters in a pair of strings
Source Python:
str1 = " ababccd12@ " 
 str2 = " bb123cca1@ " 
 matched_chars = set ( str1 ) & set ( str2 ) 
 print ( " No . of matching characters are : " + str ( len ( matched_chars ) ) )


Target Python:
word = " Hello World " 
 check = word . isdigit ( ) 
 print ( f " String contains digits ? : { check } " ) 
#########################################################################################################
#########################################################################################################
Question: Write a function to calculate median of a list of numbers given
Source Python:
def median ( pool ) : 
     copy = sorted ( pool ) 
     size = len ( copy ) 
     if size % 2 = = 1 : 
         return copy [ int ( ( size - 1 ) / 2 ) ] 
     else : 
         return ( copy [ int ( size / 2 - 1 ) ] + copy [ int ( size / 2 ) ] ) / 2


Target Python:
def median ( arr ) : 
     return list ( arr ) / len ( arr ) 
    return arr = [ 0 : - 1 ] 
    else : 
     return arr = [ 1 : : : : - 2 ] 
 a = [ 2 , 3 ] 
 a = [ 4 , 5 , 10 , 10 , 10 , 10 , 10 , 20 ] 
 a = [ 4 , 2 , 5 ] 
 a = [ 5 , 10 , 10 , 20 , 20 , 20 ] 
 a = [ 5 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 ] 
 a = [ 12 ] 
 a = [ 13 , 15 , 20 , 15 , 15 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 15 ] 
 a = [ 15 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 ] 
 def finder ( a . insert ( a ) 
 a . insert ( a ) 
 a . insert ( a , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 , 20 ] 
 return a . insert ( a . insert ( a ) 
 def removeValue ( a ) 
 def removeValue ( a . insert ( a ) 
 def removeValue ( a ) 
 def removeValue ( a ) 
 def removeValue ( a . insert ( a ) 
 return a . insert ( a . insert ( a ) 
 def removeValue ( a . insert ( a ) 
 a . insert ( a ) 
 a . insert ( a ) 
 a . insert ( a . insert ( a ) 
 a . insert ( a ) 
 a ) 
 a . insert ( a . insert ( a
#########################################################################################################
#########################################################################################################
Question: Given a list slice it into a 3 equal chunks and revert each list
Source Python:
sampleList = [ 11 , 45 , 8 , 23 , 14 , 12 , 78 , 45 , 89 ] 
 length = len ( sampleList ) 
 chunkSize   = int ( length / 3 ) 
 start = 0 
 end = chunkSize 
 for i in range ( 1 , 4 , 1 ) : 
   indexes = slice ( start , end , 1 ) 
   listChunk = sampleList [ indexes ] 
   mylist = [ i for i in listChunk ] 
   print ( " After reversing it " , mylist ) 
   start = end 
   if ( i ! = 2 ) : 
     end + = chunkSize 
   else : 
     end + = length - chunkSize


Target Python:
import numpy as np 
 A = np . array ( [ [ [ [ 1 , 2 , 3 ] , [ 5 , [ 5 , 7 , 8 ] , [ 5 , [ 1 , [ 6 , 7 , 7 , 7 , 8 ] , [ 5 ] , [ 5 , [ 7 , [ 5 , 8 , 7 , 8 , 7 , 8 ] ] ] ] ] , [ [ [ [ 0 ] ] ] ] ] 
 for i in range ( len ( A ) ) : 
     for j in range ( A [ i ] ) : 
         row = row + ' | ' 
     for row in range ( row + ' ) ) : 
         row [ row + ' | ' 
     for row in range ( row [ row [ row ] [ row ] [ row ] [ row ] [ row ] [ row ] [ row ] [ row ] for row ] [ row ] for row in row ] [ row ] [ row ] [ row ] [ row ] [ row ] ] for row in row ] for row in row ] [ row ] [ row ] [ row ] [ row ] [ row ] for row in row ] [ row ] [ row ] 
 print ( row in row ] for row in row ] 
 print ( row in row ] ) 
#########################################################################################################
#########################################################################################################
Question: write a python function to count ' a 's in the repetition of a given string ' n ' times .
Source Python:
def repeated_string ( s , n ) : 
     return s . count ( ' a ' ) * ( n / / len ( s ) ) + s [ : n % len ( s ) ] . count ( ' a ' )


Target Python:
def rev_sentence ( sentence ) : 
 
      words = sentence . split ( ' ) 
 
      reverse_sentence = ' .join ( reversed ( reversed ( words ) ) 
 
 
 
      return reverse_sentence 
#########################################################################################################
#########################################################################################################
Question: Write a Python Program to Check if a Number is a Perfect Number
Source Python:
def perfect_no_check ( n ) : 
     sum1 = 0 
     for i in range ( 1 , n ) : 
         if ( n % i = = 0 ) : 
             sum1 = sum1 + i 
     if ( sum1 = = n ) : 
         return True 
     else : 
         return False


Target Python:
num = int ( input ( " Enter a number : " ) ) 
 if num > 1 : 
    print ( " Positive number " ) 
 elif num = = = 0 : 
    print ( " Zero " ) 
 else : 
    print ( " Negative number " ) 
 else : 
    print ( " Negative number " ) 
#########################################################################################################
#########################################################################################################
Question: Write a function to calculate the total resistance of resistances in series in a given list
Source Python:
def cal_total_res_in_series ( res_list : list ) - > float : 
     return sum ( res_list )


Target Python:
def cal_total_res_in_parallel ( res_list : list ) - > float : 
     return sum ( [ 1 / r for r in res_list ] ) 
#########################################################################################################
#########################################################################################################
Question: Write a Python program to check number either positive , negative or zero
Source Python:
num = int ( input ( " Enter Integer Number : " ) ) 
 if num = = 0 : 
 print ( " Zero Entered " ) 
 elif num > 0 : 
 print ( " Positive Number Entered " ) 
 elif num < 0 : 
 print ( " Negative Number Entered " )


Target Python:
a = 60 
 b = 13 
 c = a ^ b 
 print ( " XOR " , c ) 
#########################################################################################################
#########################################################################################################
Question: Write a python program to print the uncommon elements in List
Source Python:

 test_list1 = [ [ 1 , 2 ] , [ 3 , 4 ] , [ 5 , 6 ] ] 
 test_list2 = [ [ 3 , 4 ] , [ 5 , 7 ] , [ 1 , 2 ] ] 
 
 res_list = [ ] 
 for i in test_list1 : 
     if i not in test_list2 : 
         res_list . append ( i ) 
 for i in test_list2 : 
     if i not in test_list1 : 
         res_list . append ( i ) 
 
 print ( " The uncommon of two lists is : " + str ( res_list ) )


Target Python:
list1 = [ 1 , 2 , 3 , 4 , 5 ] 
 list2 = [ 5 , 6 , 7 , 8 , 7 ] 
 final = [ i ] 
 print ( f " pair of two lists : { final } " ) 
#########################################################################################################
#########################################################################################################
Question: printing result
Source Python:
print ( " The filtered tuple : " + str ( res ) )


Target Python:
print ( " The extracted words : " + str ( res ) ) 
#########################################################################################################
#########################################################################################################
Question: write a program to convert key - values list to flat dictionary
Source Python:
from itertools import product 
 test_dict = { ' month ' : [ 1 , 2 , 3 ] , 
              ' name ' : [ ' Jan ' , ' Feb ' , ' March ' ] } 
 
 print ( " The original dictionary is : " + str ( test_dict ) ) 
 
 res = dict ( zip ( test_dict [ ' month ' ] , test_dict [ ' name ' ] ) ) 
 print ( " Flattened dictionary : " + str ( res ) )


Target Python:
tuplex = ( ' , ' , ' r ' , ' , ' r ' , ' e ' , ' , ' , ' r ' , ' , ' r ' , ' e ' , ' , ' , ' , ' r ' , ' , ' e ' , ' , ' i ' , ' , ' , ' o ' , ' , ' e ' , ' , ' e ' , ' e ' , ' e ' , ' e ' i ' , ' , ' o ' , ' , ' , ' e ' e ' e ' , ' , ' , ' e ' e ' i ' o ' , ' , ' u ' .join ( tuplex ) 
#########################################################################################################
#########################################################################################################
Question: Write a function to return the volume of a hemi sphere
Source Python:
def cal_hemisphere_volume ( radius : float ) - > float : 
     pi = 3 . 14 
     return ( 2 / 3 ) * pi * ( radius * * 3 )


Target Python:
def cal_area_hemisphere ( radius ) : 
     pi = 3 . 14 
     return 2 * pi * pi * pi * radius * height 
#########################################################################################################
#########################################################################################################
Question: write Python3 code to demonstrate working of   Sort tuple list by Nth element of tuple   using sort ( ) + lambda
Source Python:
test_list = [ ( 4 , 5 , 1 ) , ( 6 , 1 , 5 ) , ( 7 , 4 , 2 ) , ( 6 , 2 , 4 ) ] 
 print ( " The original list is : " + str ( test_list ) ) 
 N = 1 
 test_list . sort ( key = lambda x : x [ N ] ) 
 print ( " List after sorting tuple by Nth index sort : " + str ( test_list ) )


Target Python:
test_list = [ ( ' Geeks ' , ' ) , ( ' ) , ( ' , ' ) , ( ' ) , ( " The original list is : " + str ( test_list ) ) 
 res = [ ] 
 for sub in test_list if res : 
     res . append ( sub ) 
 print ( " The list is : " + str ( res ) ) 
#########################################################################################################
#########################################################################################################
Question: Write a Python function to check whether a person is eligible for voting or not based on their age
Source Python:
def vote_eligibility ( age ) : 
 	 if age > = 18 : 
 	      status = " Eligible " 
 	 else : 
 	      status = " Not Eligible " 
 	 return status


Target Python:
def isPalindrome ( s ) : 
     return s = s [ : - 1 ] 
#########################################################################################################
#########################################################################################################
Question: Write a python program to swap tuple elements in list of tuples . Print the output .
Source Python:
test_list = [ ( 3 , 4 ) , ( 6 , 5 ) , ( 7 , 8 ) ] 
 
 res = [ ( sub [ 1 ] , sub [ 0 ] ) for sub in test_list ] 
 
 print ( " The swapped tuple list is : " + str ( res ) )


Target Python:
test_list = [ ( ' gfg ' , ' is ' , ' ) , ( ' , ' ) , ( ' , ' ) , ( ' ) , ( ' x ' , ' ) , ( ' ) ] 
 print ( " The original list is : " + str ( test_list ) ) 
 res = [ ] 
 for sub in test_list if res [ 0 ] 
 print ( " The list after conversion is : " + str ( res ) ) 
#########################################################################################################
#########################################################################################################
Question: Convert two lists into a dictionary
Source Python:
ItemId = [ 54 , 65 , 76 ] 
 names = [ " Hard Disk " , " Laptop " , " RAM " ] 
 itemDictionary = dict ( zip ( ItemId , names ) ) 
 print ( itemDictionary )


Target Python:
ItemId = [ 54 , 65 , 76 ] 
 names = [ " Hard Disk " , " RAM " ] 
 itemDictionary = dict ( zip ( ItemId , names ) ) 
 print ( itemDictionary ) 
#########################################################################################################
#########################################################################################################
Question: Write a program that calculates and prints the value according to the given formula : Q = Square root of [ ( 2 * C * D)/H ]
Source Python:
import math 
 c = 50 
 h = 30 
 value = [ ] 
 items = [ x for x in raw_input ( ) . split ( ' , ' ) ] 
 for d in items : 
     value . append ( str ( int ( round ( math . sqrt ( 2 * c * float ( d ) / h ) ) ) ) ) 
 print ' , ' .join ( value )


Target Python:
num1 = 12 
 num2 = 12 
 num3 = 14 
 print ( f ' Product : { product } ' ) 
#########################################################################################################
#########################################################################################################
Question: create a tuple
Source Python:
tuplex = ( 2 , 4 , 3 , 5 , 4 , 6 , 7 , 8 , 6 , 1 )


Target Python:
NA 
#########################################################################################################
#########################################################################################################
Question: In[102 ] :
Source Python:
NA


Target Python:
NA 
#########################################################################################################
#########################################################################################################
Question: Counting total Upper Case in a string
Source Python:
str1 = " abc4234AFde " 
 digitCount = 0 
 for i in range ( 0 , len ( str1 ) ) : 
   char = str1 [ i ] 
   if ( char . upper ( ) ) : 
     digitCount + = 1 
 print ( ' Number total Upper Case : ' , digitCount )


Target Python:
word = " Hello World " 
 check = word . isdigit ( ) 
 print ( f " String contains digits ? : { check } " ) 
#########################################################################################################
#########################################################################################################
Question: Write a python function which wil return True if list parenthesis used in a input expression is valid , False otherwise
Source Python:
def isValid ( s ) : 
     stack = [ ] 
     mapping = { ' ) ' : ' ( ' , ' } ' : ' { ' , ' ] ' : ' [ ' } 
     for char in s : 
         if char in mapping : 
             if not stack : 
                 return False 
             top = stack . pop ( ) 
             if mapping [ char ] ! = top : 
                 return False 
         else : 
             stack . append ( char ) 
     return not stack


Target Python:
def load_pickle_data ( pickle_file ) : 
   import pickle 
   with open ( pickle_file , ' rb ' ) as f : 
       data = pickle . load ( f ) 
   return data 
#########################################################################################################
#########################################################################################################
Question: The consequences of modifying a list when looping through it
Source Python:
a = [ 1 , 2 , 3 , 4 , 5 ] 
 for i in a : 
     if not i % 2 : 
         a . remove ( i ) 
 print ( a ) 
 b = [ 2 , 4 , 5 , 6 ] 
 for i in b : 
      if not i % 2 : 
          b . remove ( i ) 
 print ( b )


Target Python:
my_list = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] 
 print ( " The original list is : " + str ( my_list ) ) 
 res = list ( map ( lambda x : x [ x [ i ] , my_list ) ) 
#########################################################################################################
#########################################################################################################
Question: Write a function to remove punctuation from the string
Source Python:
def r_punc ( ) : 
     test_str = " end , is best : for ! Nlp ; " 
     print ( " The original string is : " + test_str ) 
     punc = r ' ! ( ) - [ ] { } ; : \ , < > . / ? @#$ % ^ & * _ ~ ' 
     for ele in test_str : 
         if ele in punc : 
             test_str = test_str . replace ( ele , " " ) 
     print ( " The string after punctuation filter : " + test_str )


Target Python:
def r_punc ( ) : 
     test_str = ' ' ' ' ' ' ' ' .join ( test_str . split ( ) 
     print ( " The original string is : " + test_str ) 
 htness_4 
#########################################################################################

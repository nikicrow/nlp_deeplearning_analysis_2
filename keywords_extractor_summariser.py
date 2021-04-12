import pandas as pd
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

survey_data = pd.read_excel('data/raw_data.xlsx')

columns_with_open_responses = ['Q02', 'Q04', 'Q06', 'Q08', 'Q10', 'Q12', 'Q14', 'Q16', 'Q18']
raw_data = survey_data[columns_with_open_responses]

current_column = 'Q04'
current_data = raw_data[current_column]
current_data = current_data.dropna()
def clean_up(data):
    return_data = []
    for line in data:
        new_line = ""
        for word in line.split(" "):
            if "\n" in word:
                new_word = word.replace("\n"," ")
            else:
                new_word = word
            new_line = new_line + " " + new_word
        return_data.append(new_line)
    return return_data

current_data = clean_up(current_data)
all_data_string = " ".join([i for e in current_data for i in e.split(" ")])

text_summarised = summarize(all_data_string, word_count=200)

text_keywords = keywords(all_data_string,split=True, words=20,lemmatize=True)
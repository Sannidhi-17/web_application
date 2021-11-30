# Import all the libraries
import shutil

import pandas as pd
import numpy as np
import os
import PyPDF2
import sys
import re
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from nltk.tokenize import RegexpTokenizer
import libvoikko
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import gensim
from itertools import combinations
#from nltk import flatten
#from . import forms
#import string

# Define the path for directory
#os.chdir('../trailproject/media/')
os.chdir('E:/Internship_finland/web_application/internship/trailproject/media/')

# Pre-define stp_words for the Finnish language
stop_en = stopwords.words('finnish')


# Adding and removing the stop_words from the pre-define list of stop_words
class Stopwords:
    def __init__(self, add_stopwords='', remove_stopwords=''):
        self.add_stopwords = add_stopwords
        self.remove_stopwords = remove_stopwords
        self.stop_en = stop_en

    def adding_stopwords(self):
        additional = self.stop_en
        additional.append('ok')
        b = int(len(additional))
        x = b - 236
        x = int(x)
        additional.pop(x)
        if self.add_stopwords != '':
            self.add_stopwords = self.add_stopwords.lower()
            additional.append(self.add_stopwords)
        b = int(len(additional))
        x = b - 235
        if self.remove_stopwords != '':
            self.remove_stopwords = self.remove_stopwords.lower()
            additional.remove(self.remove_stopwords)
        additional.sort()
        return additional


# Collect the Excel or PDF file file and convert into the .csv file
class generate_csvfile:
    def __init__(self, file_name, sheet_name=None):
        self.file_name = file_name
        self.sheet_name = sheet_name

    def emptyDir(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def convert_file(self):
        a = os.path.splitext(self.file_name)[0]
        f_name = a + '.csv'
        if self.file_name.endswith('.xlsx'):
            read_file = pd.read_excel(self.file_name, sheet_name=self.sheet_name)
            for col in read_file.columns:
                col1 = col
            print(read_file.values.tolist())
            x = read_file[col1].values.tolist()

            text = ''
            for i in x:
                # print(i)
                text = text + ' ' + i
            print("=============")

            print(text)
            print("-----------------------")
            j = text.split(". ")
            print(j)

            df = pd.DataFrame(j)
            print(df)
            df.to_csv(f_name)
            #read_file.to_csv(f_name, index=None, header=True, encoding='utf-8-sig')
            file = pd.read_csv(f_name, encoding='utf-8-sig', error_bad_lines=False, names=['documents'])


        elif self.file_name.endswith('.pdf'):
            pdf_file = self.file_name
            read_pdf = PyPDF2.PdfFileReader(pdf_file)
            number_of_pages = read_pdf.getNumPages()
            df1 = pd.DataFrame()
            # df1 = df1.append(pd.DataFrame(row, columns=['documents']),ignore_index=True)
            for page_number in range(number_of_pages):
                # use xrange in Py2
                page = read_pdf.getPage(page_number).extractText().split('.')
                row = pd.Series(page)
                df1 = df1.append(pd.DataFrame(row, columns=['documents']), ignore_index=True)
                # df1 = df1.replace('\n',' ', regex=True)
                df1.documents = df1.documents.str.replace("\n-", "")
                df1.documents = df1.documents.str.replace("- \n", '')
                df1.documents = df1.documents.str.replace("-", '')
                df1.documents = df1.documents.str.replace("*", '')
                df1.documents = df1.documents.str.replace(".com", "/com", case=False)
                df1.to_excel('checking.xlsx')
                df1.documents = df1.documents.str.replace('[ˆ,†,˛,˝,˝,˜,š,˘,‹,ˇ,›,•,Ł,˙,",(,),<,>,{,},œ,”,”,•,*,*]', '')
                df1.documents = df1.documents.str.replace('[:,\n, @,˚,”]', ' ')
                # .documents = re.sub('([a-z])-(?=[a-z])', r'\1', df1.documents)
                # df1.documents = df1.documents.str.replace('[^\w-\w]', '\w\w')
                df1.documents = df1.documents.str.replace("ﬂ", "", case=False)
                df1['documents'].replace('', np.nan, inplace=True)
                df1.dropna(subset=['documents'], inplace=True)
                df1.to_csv(f_name, index=None, encoding='utf-8-sig')
                df1.to_excel('pdf to excel.xlsx')
                file = pd.read_csv(f_name, encoding='utf-8-sig', sep=',', error_bad_lines=False, names=['documents'])
                # print(file.head())
        else:
            # return
            sys.exit()

        return file


# Apply pre-processing on the text, generate word cloud, count the coherence score and plot the coherence score
class Automate_topic_modeling:

    def __init__(self, dataframe, additional=stop_en):
        self.dataframe = dataframe
        self.stop_words = additional
        self.config = {
            "tfidf": {
                "sublinear_tf": True,
                "ngram_range": (1, 1),
                "max_features": 10000,
            },
            "nmf": {
                "init": "nndsvd",
                "alpha": 0,
                "random_state": 42,
            }
        }

        # Remove the numbers from the file

    def remove_numbers(self):
        self.dataframe['documents'] = self.dataframe['documents'].str.replace('\d+', '')

        return

        # Apply preprocessing

    def pre_processing(self):

        initial_df = self.dataframe
        # initial_df = str(initial_df)
        initial_df['Index'] = np.arange(1, len(initial_df) + 1)
        initial_df = initial_df[['Index', 'documents']]
        initial_df['documents'] = initial_df['documents'].astype(str)
        new_df = pd.DataFrame(initial_df, index=initial_df.Index).stack()
        # new_df = pd.DataFrame(initial_df.documents.str.split('[.?!,]').tolist(), index=initial_df.Index).stack()
        new_df = new_df.reset_index([0, 'Index'])
        new_df.columns = ['Index', 'documents']
        new_df['documents'] = new_df['documents'].str.replace('[œ,Œ]', '-')
        new_df['documents'] = new_df['documents'].str.replace('ƒ⁄ﬁﬁ⁄', '')
        new_df['documents'] = new_df['documents'].str.replace('*', '')
        new_df['documents'] = new_df['documents'].str.lstrip()

        # # Remove empty row
        new_df['documents'].replace('', np.nan, inplace=True)
        new_df.dropna(subset=['documents'], inplace=True)
        # new_df.to_excel('checking.xlsx')
        # Capitalize the first letter
        new_df['documents'] = new_df['documents'].map(lambda x: x[0].upper() + x[1:])
        # new_df.to_excel('checking_upper.xlsx')
        # Converting into lower case
        # new_df['documents1'] = new_df.documents.map(lambda x: x.lower())
        new_df['documents1'] = new_df['documents'].str.replace(
            '[-,:,/,(,),",;,>,<,?,_,\n,❤,\t,??,ӻ,كw,큞,ԃ,ˮ,ĭ,ﬁﬁ,ﬂ,•,*,.,!]',
            '')
        # new_df['documents1'] = new_df['documents1'].str.replace('[^\w]', '')
        # new_df['documents1'] = new_df['documents1'].str.replace('[^\s]', ' ')
        new_df['documents1'] = new_df['documents1'].str.lstrip()
        # remove empty strings
        new_df['new_col'] = new_df['documents1'].astype(str).str[0]
        # new_df['documents1'] = new_df['documents1'].str.replace('[^\w]', '')
        # new_df['documents1'] = new_df['documents1'].str.replace('[^\s]', ' ')
        nan_value = float("NaN")
        # Convert NaN values to empty string
        new_df.replace("", nan_value, inplace=True)
        new_df.dropna(subset=["new_col"], inplace=True)
        new_df.drop('new_col', inplace=True, axis=1)
        # Convert articles ino the tokens

        new_df['docuemnt_tokens'] = new_df.documents.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))

        # Apply Lemmatization (Voikko)
        # os.add_dll_directory(r'C:\Voikko')
        C = libvoikko.Voikko(u"fi")
        # C.setLibrarySearchPath("C:\Voikko")

        # Apply lemmatizations to the words
        def lemmatize_text(text):
            bf_list = []
            for w in text:
                voikko_dict = C.analyze(w)
                if voikko_dict:
                    bf_word = voikko_dict[0]['BASEFORM']
                else:
                    bf_word = w
                bf_list.append(bf_word)
            return bf_list

        new_df['lemmatized'] = new_df.docuemnt_tokens.apply(lemmatize_text)
        # new_df['documents'] = new_df['documents'].map(lambda x: [t for t in x if t not in self.stop_words])
        # stop_en = stopwords.words('finnish')
        new_df['article'] = new_df.docuemnt_tokens.map(lambda x: [t for t in x if t not in self.stop_words])
        # make sure the datatype of column 'article_removed_stop_words' is string
        new_df['article'] = new_df['article'].astype(str)
        new_df['article'] = new_df['article'].apply(eval).apply(' '.join)
        new_df['Index'] = np.arange(1, len(new_df['article']) + 1)
        new_df.to_excel('../static/assets/text_preprocessing.xlsx')
        self.stop_words.sort()
        return new_df, self.stop_words

        # Generate the wordcloud

    # def generate_wordcloud(self, new_df):
    #     text = new_df['article'].tolist()
    #     wordcloud = WordCloud(background_color='black').generate(" ".join(text))
    #     # Open a plot of the generated image.
    #     plt.figure(figsize=(40, 20))
    #     plt.imshow(wordcloud)
    #     plt.axis("off")
    #     plt.switch_backend('agg')
    #     plt.margins(0, 0)
    #     # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     # # #plt.savefig('../static/images/wordCloud.png')
    #     #transperant = True,,
    #     plt.tight_layout(pad=0)
    #     plt.tight_layout(pad=0)
    #     plt.savefig('../static/images/wordCloud.png', bbox_inches = 'tight')
    #     return
    #     # plt.switch_backend('agg')
    #     # plt.savefig('../static/images/wordCloud.png')
    #     # return

    def generate_wordcloud(self, new_df):
        text = new_df['article'].tolist()
        wordcloud = WordCloud(background_color='black').generate(" ".join(text))
        cm = 1 / 2.54
        plt.figure(figsize=(21 *cm, 9*cm), facecolor='k')
        plt.tight_layout(pad=0)
        plt.imshow(wordcloud)

        plt.axis('off')
        plt.rcParams['figure.dpi'] = 500
        plt.rcParams['savefig.dpi'] = 100
        plt.savefig('../static/images/wordCloud.png', bbox_inches = 'tight')
        return

        # # Count the tf-idf matrix

    def tfidf_matrix(self, new_df):
        new_df['documents1'] = new_df['documents1'].str.replace('[^\w\s]', '')
        raw_documents = new_df['documents1'].tolist()
        self.count = new_df['documents1'].count()
        for i in range(len(raw_documents)):
            raw_documents[i] = raw_documents[i].lower()

        self.vectorizer = TfidfVectorizer(**self.config["tfidf"],
                                          stop_words=self.stop_words)
        A = self.vectorizer.fit_transform(raw_documents)
        terms = self.vectorizer.get_feature_names()

        return terms, A, raw_documents

        # Apply NMF-Model  model

    def nmf_model(self, A):
        kmin, kmax = 1, 1
        topic_models = []
        if self.count <= 100:
            kmin, kmax = 3, 10
        elif self.count <= 101 & self.count <= 500:
            kmin, kmax = 5, 15
        elif self.count > 501 & self.count <= 1000:
            kmin, kmax = 10, 30
        elif self.count > 1001 & self.count <= 3000:
            kmin, kmax = 10, 50
        else:
            kmin, kmax = 15, 70
        # try each value of k
        for k in range(kmin, kmax + 1):
            # run NMF
            model = decomposition.NMF(n_components=k, **self.config["nmf"])
            W = model.fit_transform(A)
            H = model.components_
            # store for later
            topic_models.append((k, W, H))
        return topic_models, H, W,

        # Building the word-to-vector dictionary

    def build_w2c(self, raw_documents):
        docgen = TokenGenerator(raw_documents, self.stop_words)
        new_list = []
        for each in docgen.documents:
            new_list.append(each.split(" "))
        new_list = [string for string in new_list if string != ""]
        # Build the word2vec model
        self.w2v_model = gensim.models.Word2Vec(vector_size=500, min_count=0.0005, sg=1)
        self.w2v_model.build_vocab(corpus_iterable=new_list)
        return self.w2v_model

        # Find the get descriptor

    def get_descriptor(self, all_terms, H, topic_index, top):
        # reverse sort the values to sort the indices
        top_indices = np.argsort(H[topic_index, :])[::-1]
        # now get the terms corresponding to the top-ranked indices
        top_terms = []
        for term_index in top_indices[0:top]:
            top_terms.append(all_terms[term_index])
        return top_terms

        # Calculate the coherence score

    def get_coherence(self, terms, topic_models):
        k_values = []
        coherences = []
        dict = {}
        for (k, W, H) in topic_models:
            # Get all of the topic descriptors - the term_rankings, based on top 10 terms
            term_rankings = []
            for topic_index in range(k):
                term_rankings.append(self.get_descriptor(terms, H, topic_index, 10))
            # Now calculate the coherence based on our Word2vec model
            k_values.append(k)
            coherences.append(self.calculate_coherence(term_rankings))
            dict[k] = coherences[-1]
        newDict = {}
        # Iterate over all the items in dictionary and filter items which has even keys
        for (key, value) in dict.items():
            # Check if key is even then add pair to new dictionary
            if key % 2 == 0:
                newDict[key] = value

        max_key = max(newDict, key=newDict.get)
        return term_rankings, max_key, newDict

        # Calculate coherence score

    def calculate_coherence(self, term_rankings):
        overall_coherence = 0.0
        for topic_index in range(len(term_rankings)):
            # check each pair of terms
            pair_scores = []
            for pair in combinations(term_rankings[topic_index], 2):
                pair_scores.append(self.w2v_model.wv.similarity(pair[0], pair[1]))
            # get the mean for all pairs in this topic
            topic_score = sum(pair_scores) / len(pair_scores)
            overall_coherence += topic_score
        # get the mean score across all topics
        return overall_coherence / len(term_rankings)

        # Plot the coherence score

    def plot_the_coherence_graph(pself, newDict):
        plt.figure(figsize=(15, 5))
        # create the line plot
        k_values = list(newDict.keys())
        coherences = list(newDict.values())
        # create the line plot
        plt.plot(k_values, coherences)
        plt.xticks(k_values, rotation=90)
        plt.xlabel("Number of Topics")
        plt.ylabel("Mean Coherence")
        # add the points
        plt.scatter(k_values, coherences)
        # find and annotate the maximum point on the plot
        ymax = max(coherences)
        xpos = coherences.index(ymax)
        best_k = k_values[xpos]
        plt.annotate("k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=10)
        # show the plot
        # plt.show()

        plt.savefig('../static/images/coherenc_graph.png')
        return


# Using in the generating dictionary
class TokenGenerator:
    def __init__(self, documents, stopwords):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile(r"(?u)\b\w\w+\b")

    def __iter__(self):
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall(doc):
                if tok in self.stopwords:
                    tokens.append("<stopword>")
                elif len(tok) >= 3:
                    tokens.append(tok)
            yield tokens


# Analyse the number topic numbers and visualize them
class topic_modeling:

    def __init__(self, tfidf, topic_numbers, feature_names, new_df):
        self.topic_word_2 = None
        self.tfidf = tfidf
        self.topic_numbers = topic_numbers
        self.feature_names = feature_names
        self.new_df = new_df
        self.config = {
            "nmf": {
                "init": "nndsvd",
                "alpha": 0,
                "random_state": 42
            }
        }
        self.model = decomposition.NMF(n_components=self.topic_numbers, **self.config["nmf"])

        # Again count the NMF model on the specific number of topics

    def nmf_modeling(self):
        # apply the model and extract the two factor matrices
        W = self.model.fit_transform(self.tfidf)
        H = self.model.components_
        return H, W

        # Top 10 words dataframe topic_word

    def moveFiles(self, source_dir, target_dir):
        file_names = os.listdir(source_dir)
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), target_dir)


    def display_topics(self, no_top_words):
        col1 = 'topic'
        col2 = 'top_ten_words'
        dct = {col1: [], col2: []}
        for topic_idx, topic in enumerate(self.model.components_):
            dct[col1].append(int(topic_idx) + 1)
            dct[col2].append(", ".join([self.feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            x = pd.DataFrame.from_dict(dct)

        print(x)
        return x

        # Top 3 keywords themes per topic

    def top_three_keywords(self):
        # Themes top 3 keywords dataframe top_words1
        no_top_words = 3
        topic_word_3 = self.display_topics(no_top_words)
        topic_word_3['Themes'] = topic_word_3.top_ten_words.str.title()
        topic_word_1 = topic_word_3.loc[:, ['topic', 'Themes']]
        return topic_word_1

        # Find the responses which lies into which number of topics

    def documents_per_topic(self, W, topic_word_1):
        df2 = pd.DataFrame({'topic': W.argmax(axis=1),
                            'documents': self.new_df['documents']},
                           columns=['topic', 'documents'])
        df2['documents'] = df2['documents'].apply(str).str.replace('\n', ' ')
        df2['documents'] = df2['documents'].apply(str).str.replace("/km", "€/km", case=False)
        df2['documents'] = df2['documents'].apply(str).str.replace(" gmail", "@gmail", case=False)
        # df2.to_excel('topic_number_with_responses.xlsx')
        # df2.to_pickle('topic.pkl')

        no_top_words = 10
        topic_word = self.display_topics(10)
        topic_word['documents'] = ''
        for i in range(self.topic_numbers):
            df3 = df2[df2['topic'] == i]
            x1 = df3['documents'].tolist()
            topic_word.iat[i, topic_word.columns.get_loc('documents')] = x1
            # i += 1
        topic_word_merge = pd.merge(topic_word_1, topic_word, on='topic')  # Merge two different dataframes and
        topic_word_merge['Length'] = topic_word_merge['documents'].str.len()
        return df2, topic_word_merge

        # Removve the topic after merging the topics

    def remove_row(self, topic_word_merge):
        topic_word_merge['Length'] = topic_word_merge['documents'].str.len()
        topic_word_2 = topic_word_merge[topic_word_merge.Length != 0]
        topic_word_2['topic'] = np.arange(1, len(topic_word_2) + 1)
        topic_word_2 = topic_word_2.reset_index(drop=True)
        return topic_word_2

        # Frequency Plot

    def frequency_plot(self, topic_word_2):
        cm = 1 / 2.54
        plt.figure(figsize=(15 * cm, 10.5 * cm))
        sns.set(font_scale=1)
        splot = sns.barplot(x="topic", y="Length", data=topic_word_2, color='blue')
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.0f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 10),
                           textcoords='offset points',
                           fontsize=10)
        plt.xlabel("Topics", size=10)
        plt.ylabel("Frequency", size=10)

        # path = self.create_directory
        plt.savefig('../static/images/Frequency_of_topic.png', transparent=True)

        return topic_word_2

        # Percentage Plot

    def percentage_plot(self, topic_word_2):
        topic_word_2['percent'] = (topic_word_2['Length'] / topic_word_2['Length'].sum()) * 100
        topic_word_2['percent'] = topic_word_2['percent'].astype(int)
        cm = 1 / 2.54
        plt.figure(figsize=(15
                            * cm, 10.5 * cm))
        sns.set(font_scale=1)
        splot = sns.barplot(x="topic", y="percent", data=topic_word_2, color='blue')

        for p in splot.patches:
            splot.annotate('{:0.0f}%'.format(p.get_height()),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 10),
                           textcoords='offset points',
                           fontsize=10)

        # without_hue(splot, df.percentage_of_each_topic)
        plt.xlabel("Topics", size=10)
        plt.ylabel("percentage", size=10)
        #plt.set_size_inches(5.1, 5.1)
        # path = self.create_directory
        plt.savefig( '../static/images/percentage_of_topic.png', transparent=True)

        return

    # def percentage_plot(self, topic_word_2):
    #     topic_word_2['percent'] = (topic_word_2['Length'] / topic_word_2['Length'].sum()) * 100
    #     topic_word_2['percent'] = topic_word_2['percent'].astype(int)
    #     #plt.figure(figsize=(30, 15))
    #     sns.set(font_scale=1)
    #     splot = sns.barplot(x="topic", y="percent", data=topic_word_2, color='blue')
    #     #plt.rcParams['figure.figsize'] = [5, 20]
    #     plt.gcf().set_size_inches(6.7, 4)
    #     for ax in plt.gcf().axes:
    #         l = ax.get_xlabel()
    #         ax.set_xlabel(l, fontsize=15)
    #     for p in splot.patches:
    #         splot.annotate('{:0.0f}%'.format(p.get_height()),
    #                        (p.get_x() + p.get_width() / 2., p.get_height()),
    #                        ha='center', va='center',
    #                        xytext=(0, 10),
    #                        textcoords='offset points',
    #                        fontsize=10)
    #
    #     # without_hue(splot, df.percentage_of_each_topic)
    #     plt.xlabel("Topics", size=10)
    #     plt.ylabel("percentage", size=10)
    #     # path = self.create_directory
    #     plt.savefig( '../static/images/percentage_of_topic.png', transparent=True, figsize=(15, 5))
    #
    #     return

        # Top five responses per topic

    def get_top_snippets(self, all_snippets, W, topic_index, top):
        top = int(top)
        top_indices = np.argsort(W[:, topic_index])[::-1]
        # now get the snippets corresponding to the top-ranked indices
        top_snippets = []
        for doc_index in top_indices[0:top]:
            top_snippets.append(all_snippets[doc_index])

        return top_snippets

        # Create final csv file and save as excel file

    def final_output(self, topic_word_2, W, top=10):
        snippets = self.new_df['documents'].tolist()
        topic_word_2['responses'] = ''
        topic_word_2['xl'] = ''


        print(self.topic_numbers)
        for i in range(0, self.topic_numbers):
            if topic_word_2['Length'][i] <= 10:
                print(topic_word_2['documents'][i])
                y2 =  topic_word_2['documents'][i]
                topic_word_2.at[i, 'responses'] = y2

            else:
                y1 = self.get_top_snippets(snippets, W, i, top)
                topic_word_2.at[i, 'responses'] = y1
        print(topic_word_2)
        themes_keywords = topic_word_2


        self.topic_word_2 = topic_word_2
        topic_word_2.to_csv(
                '../static/history/topic_word_2.csv',
                index=False)
        themes_keywords = themes_keywords.rename(
            columns={'topic': 'Topic_id', 'Themes': 'Themes', 'top_ten_words': 'Top_ten_words',
                     'responses': 'Top_responses',
                     'documents': 'Documents',
                     'Length': 'Frequency_of_each_topic',
                     'percent': 'Percentage_of_each_topic'})
        themes_keywords = themes_keywords[['Topic_id', 'Themes', 'Top_ten_words', 'Top_responses', 'Documents', 'Frequency_of_each_topic',
                         'Percentage_of_each_topic']]
        themes_keywords_ = themes_keywords[['Topic_id', 'Themes', 'Top_ten_words', 'Top_responses', 'Documents', 'Frequency_of_each_topic', 'Percentage_of_each_topic']]
        themes_keywords_[' '] = ''
        themes_keywords_[' '] = ''
        themes_keywords_[' '] = ''
        themes_keywords_ = themes_keywords_.rename(
            columns = {'Topic_id': "Topic Id", "Top_ten_words": "Top Ten Words", 'Top_responses':'Top Responses',
                       'Frequency_of_each_topic': 'Frequency Of Each Topic', 'Percentage_of_each_topic':'Percentage Of Each Topic' }
        )
        themes = themes_keywords_[['Topic Id', 'Themes', 'Top Ten Words', 'Top Responses',' ', 'Documents', ' ', 'Frequency Of Each Topic', ' ', 'Percentage Of Each Topic']]
        if len(os.listdir('../static/history')) == 0:
            themes_keywords.to_csv(
                '../static/history/backup.csv',
                index=False)
            themes.to_excel(
                '../static/history/backup.xlsx')

        themes_keywords.to_csv(
            '../static/assets/themes_keywords.csv',
            index=False)
        themes.to_excel(
            '../static/assets/themes_keywords.xlsx')
        return themes_keywords

        # For automatic subplot

    def printPFsInPairs(self):
        prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if self.topic_numbers in prime_numbers:
            self.topic_numbers = self.topic_numbers + 1
        for i in range(1, int(pow(self.topic_numbers, 1 / 2)) + 1):
            if self.topic_numbers % i == 0:
                self.n = i
                self.m = int(self.topic_numbers / i)

        # Keywords plot

    # def plot_top_words(self, n_top_words, title):
    #
    #     fig, axes = plt.subplots(self.m, self.n, figsize=(40, 40), tight_layout=True)
    #     axes = axes.flatten()
    #     for topic_idx, topic in enumerate(self.model.components_):
    #         top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
    #         top_features = [self.feature_names[i] for i in top_features_ind]
    #         weights = topic[top_features_ind]
    #         ax = axes[topic_idx]
    #         ax.barh(top_features, weights, height=0.7)
    #         ax.set_title(f'Topic {topic_idx + 1}',
    #                      fontdict={'fontsize': 50})
    #         ax.invert_yaxis()
    #         ax.tick_params(labelsize=50)
    #         for i in 'top right left'.split():
    #             ax.spines[i].set_visible(False)
    #         fig.savefig('../static/images/keywords.png',
    #                     transparent=True)
    #     return
    ## changes fontsize and set_size_inches
    def plot_top_words(self, n_top_words, title):
        # figsize=(40, 40),
        fig, axes = plt.subplots(self.m, self.n,figsize=(40, 40), tight_layout=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            ax = axes[topic_idx]
            #linewidth = 5, height=1
            ax.barh(top_features, weights, height = 1)
            ax.set_title(f'Topic {topic_idx + 1}',
                             fontdict={'fontsize': 10})
            ax.invert_yaxis()
            ax.tick_params(labelsize=10)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.set_size_inches(5.9, 5.9)

            fig.savefig('../static/images/keywords.png',
                            transparent=True)

        return

        # Merge the different topics

    def fit_transform_merge(self, feature_1, feature_2, W_1, H_1):

        feature_1 = int(feature_1) - 1
        feature_2 = int(feature_2) - 1
        W = np.copy(W_1)
        H = np.copy(H_1)
        # merge (addition) column values of W
        W[:, feature_1] = W[:, feature_1] + W[:, feature_2]
        w = np.delete(W, feature_2, 1)
        # merge (addition) row values of H
        H[feature_1, :] = H[feature_1, :] + H[feature_2, :]
        self.model.components_ = np.delete(H, feature_2, 0)
        self.topic_numbers -= 1
        return w, H

        # Rename the topics

    def rename_topic(self, topic_number, new_name, themes_keywords):

        x = int(topic_number)
        y = themes_keywords.loc[themes_keywords['Topic_id'] == x]['Themes'].values[0]
        themes_keywords['Themes'] = themes_keywords['Themes'].replace(to_replace=y, value=new_name)
        themes_keywords.to_csv(
            '../static/assets/themes_keywords.csv',
            index=False)
        themes_keywords.to_excel(
            '../static/assets/themes_keywords.xlsx')
        return themes_keywords

        # Remove the Keywords from the topic

    def split_topic(self, topic_number):
        x = int(topic_number)
        # self.topic_word_2 = self.topic_word_2.set_index("topic")
        # self.topic_word_2 = self.topic_word_2.drop([x])
        self.topic_word_2.to_csv('../static/history/aaaa.csv')
        self.frequency_plot(self.topic_word_2)
        self.percentage_plot(self.topic_word_2)
        self.topic_numbers = self.topic_numbers - 1
        self.printPFsInPairs(self)
        self.plot_top_words(self, 3, 'Per topic top 3 keywords')
        return



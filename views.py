# Import all the libraries
from django.shortcuts import render, redirect
from django.shortcuts import HttpResponse
from django.core.files.storage import FileSystemStorage
from basic.code_NMF import generate_csvfile
from basic.code_NMF import Automate_topic_modeling
from basic.code_NMF import TokenGenerator
from basic.code_NMF import topic_modeling
from basic.code_NMF import Stopwords
from . import forms
import pandas as pd
from .forms import KeyWordDeletionForm
from .forms import SplitTopic
from .models import Topic, Keywords

# Content to pass data on the web-page
new_dict = {'_A': None, '_terms': None, 'final': None, 'theme_keywords': None}
content = {'max_key': None, 'output_file_name': None, 'columns': None, 'rows': None, 'id': None, 'top10': None,
           'additional_stopwords': None, 'file': None, 'form': None, 'new_stopword': None, 'top_responses': None}
trial = {'W': None, 'H': None}
#form = {'additional_stopwords': None}


# Recieve the first data like file, analysis name, sheet number and apply basic analysis
def upload(request):
    global content
    max_key = int

    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        file_name = uploaded_file.name
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)

        form_1 = forms.SheetName(request.POST)
        if form_1.is_valid():
            sheet_name = form_1.cleaned_data['sheetname']
            output_file_name = form_1.cleaned_data['output_file_name']
            checkbox = request.POST.get('vehicle1')
            file_1 = generate_csvfile(file_name, sheet_name)
            file_1.emptyDir(path='../static/assets')
            file_1.emptyDir(path='../static/images')
            file_1.emptyDir(path='../static/history')
            final_csv = file_1.convert_file()
            content['file'] = final_csv
            # New class
            if (final_csv.empty == False):
                x = Automate_topic_modeling(final_csv)
                if checkbox == 'on':
                    x.remove_numbers()

                new_dict['final'], stop_words = x.pre_processing()
                content['additional_stopwords'] = stop_words
                x.generate_wordcloud(new_dict['final'])
                new_dict['_terms'], new_dict['_A'], _raw_documents = x.tfidf_matrix(new_dict['final'])
                topic_models, H, W = x.nmf_model(new_dict['_A'])
                x.build_w2c(_raw_documents)
                term_rankings, max_key, newDict = x.get_coherence(new_dict['_terms'], topic_models)
                content['max_key'] = max_key
                plot = x.plot_the_coherence_graph(newDict)
                m = topic_modeling(new_dict['_A'], max_key, new_dict['_terms'], new_dict['final'])
                H1, W1 = m.nmf_modeling()
                trial['H'] = H1
                trial['W'] = W1
                df = m.top_three_keywords()
                _df2, _topic_word = m.documents_per_topic(W1, df)
                m.frequency_plot(_topic_word)
                m.percentage_plot(_topic_word)
                theme_keywords = m.final_output(_topic_word, W1)
                m.printPFsInPairs()
                m.plot_top_words(3, 'Per topic top 3 keywords')
                new_dict['theme_keywords'] = theme_keywords
                # content = {'max_key': max_key, 'output_file_name': output_file_name}
                content['output_file_name'] = output_file_name
                finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
                content['columns'] = finalTableData.columns
                content['rows'] = finalTableData.to_dict('records')
                content['id'] = finalTableData['Topic_id'].tolist()
                content['top10'] = finalTableData['Top_ten_words'].tolist()
                content['top_responses'] = 10


            else:
                print('ADD CORRECT FILE')
        else:
            output_file_name = form_1.cleaned_data['output_file_name']
            checkbox = request.POST.get('vehicle1')
            file_1 = generate_csvfile(file_name)
            file_1.emptyDir(path='../static/assets')
            file_1.emptyDir(path='../static/images')
            file_1.emptyDir(path='../static/history')
            final_csv = file_1.convert_file()
            content['file'] = final_csv
            # New class
            if (final_csv.empty == False):
                x = Automate_topic_modeling(final_csv)
                if checkbox == 'on':
                    x.remove_numbers()
                new_dict['final'], stop_words = x.pre_processing()
                content['additional_stopwords'] = stop_words
                x.generate_wordcloud(new_dict['final'])
                new_dict['_terms'], new_dict['_A'], _raw_documents = x.tfidf_matrix(new_dict['final'])
                topic_models, H, W = x.nmf_model(new_dict['_A'])
                x.build_w2c(_raw_documents)
                term_rankings, max_key, newDict = x.get_coherence(new_dict['_terms'], topic_models)
                plot = x.plot_the_coherence_graph(newDict)
                m = topic_modeling(new_dict['_A'], max_key, new_dict['_terms'], new_dict['final'])
                H1, W1 = m.nmf_modeling()
                trial['H'] = H1
                trial['W'] = W1
                df = m.top_three_keywords()
                _df2, _topic_word = m.documents_per_topic(W1, df)
                m.frequency_plot(_topic_word)
                m.percentage_plot(_topic_word)
                theme_keywords = m.final_output(_topic_word, W1)
                m.printPFsInPairs()
                m.plot_top_words(3, 'Per topic top 3 keywords')
                new_dict['theme_keywords'] = theme_keywords
                # content = {'max_key': max_key, 'output_file_name': output_file_name}
                content['max_key'] = max_key
                content['output_file_name'] = output_file_name
                finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
                content['columns'] = finalTableData.columns
                content['rows'] = finalTableData.to_dict('records')
                content['id'] = finalTableData['Topic_id'].tolist()
                content['top10'] = finalTableData['Top_ten_words'].tolist()
                content['top_responses'] = 10

            else:
                print('ADD CORRECT FILE')

    return render(request, 'basic.html', content)

#
# # Archieve page
# def archieve(request):
#     return render(request, 'archieve.html', content)
#

# Change topic and apply re-analysis
def change_topic(request):
    form = forms.ChangeTopic()
    if request.method == "POST":
        form = forms.ChangeTopic(request.POST)

        if form.is_valid():
            new_topic_number = form.cleaned_data['change_topic']
            topic_number = int(new_topic_number)
            m = topic_modeling(new_dict['_A'], topic_number, new_dict['_terms'], new_dict['final'])
            H1, W1 = m.nmf_modeling()
            df = m.top_three_keywords()
            _df2, _topic_word = m.documents_per_topic(W1, df)
            m.frequency_plot(_topic_word)
            m.percentage_plot(_topic_word)
            output = m.final_output(_topic_word, W1)
            m.printPFsInPairs()
            m.plot_top_words(3, 'Per topic top 3 keywords')
            content["max_key"] = topic_number
            finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
            content['columns'] = finalTableData.columns
            content['rows'] = finalTableData.to_dict('records')
            content['top_responses'] = 10
    return render(request, 'basic.html', {'rows': content['rows'], 'columns': content['columns'], 'max_key':content['max_key'], 'top_responses': content['top_responses']})


# Add new stopwords and re-analysis
def addstopwords(request):
    form_2 = forms.Add_stopwords(request.POST)
    if request.method == "POST":
        if form_2.is_valid():
            new_stopwords = form_2.cleaned_data['add_stopwords']
            content['new_stopwords'] = new_stopwords
            n_class = Stopwords(add_stopwords=new_stopwords)
            additional = n_class.adding_stopwords()
            content['additional_stopwords'] = additional
            x = Automate_topic_modeling(content['file'], additional)
            new_dict['final'], stop_words = x.pre_processing()
            x.generate_wordcloud(new_dict['final'])
            new_dict['_terms'], new_dict['_A'], _raw_documents = x.tfidf_matrix(new_dict['final'])
            # topic_models, H, W = x.nmf_model(new_dict['_A'])
            # x.build_w2c(_raw_documents)
            m = topic_modeling(new_dict['_A'], content['max_key'], new_dict['_terms'], new_dict['final'])
            H1, W1 = m.nmf_modeling()
            trial['H'] = H1
            trial['W'] = W1
            df = m.top_three_keywords()
            _df2, _topic_word = m.documents_per_topic(W1, df)
            m.frequency_plot(_topic_word)
            m.percentage_plot(_topic_word)
            theme_keywords = m.final_output(_topic_word, W1)
            m.printPFsInPairs()
            m.plot_top_words(3, 'Per topic top 3 keywords')
            new_dict['theme_keywords'] = theme_keywords
            finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
            content['columns'] = finalTableData.columns
            content['rows'] = finalTableData.to_dict('records')
            content['id'] = finalTableData['Topic_id'].tolist()
            content['top10'] = finalTableData['Top_ten_words'].tolist()
            topic_number = content['max_key']
            content['max_key'] = topic_number
            content['top_responses'] = 10
    return render(request, 'basic.html', content)


# Remove stop_words from pre-defined list
def remove_stopwords(request):
    form_2 = forms.Remove_stopwords(request.POST)
    if request.method == "POST":
        if form_2.is_valid():
            new_stopwords = form_2.cleaned_data['remove_stopwords']
            n_class = Stopwords(remove_stopwords=new_stopwords)
            additional = n_class.adding_stopwords()
            content['additional_stopwords'] = additional
            x = Automate_topic_modeling(content['file'], additional)
            new_dict['final'], stop_words = x.pre_processing()
            x.generate_wordcloud(new_dict['final'])
            new_dict['_terms'], new_dict['_A'], _raw_documents = x.tfidf_matrix(new_dict['final'])
            # topic_models, H, W = x.nmf_model(new_dict['_A'])
            # x.build_w2c(_raw_documents)
            # term_rankings, max_key, newDict = x.get_coherence(new_dict['_terms'], topic_models)
            # plot = x.plot_the_coherence_graph(newDict)

            m = topic_modeling(new_dict['_A'], content['max_key'], new_dict['_terms'], new_dict['final'])
            H1, W1 = m.nmf_modeling()
            trial['H'] = H1
            trial['W'] = W1
            df = m.top_three_keywords()
            _df2, _topic_word = m.documents_per_topic(W1, df)
            m.frequency_plot(_topic_word)
            m.percentage_plot(_topic_word)
            theme_keywords = m.final_output(_topic_word, W1)
            m.printPFsInPairs()
            m.plot_top_words(3, 'Per topic top 3 keywords')
            new_dict['theme_keywords'] = theme_keywords
            finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
            content['columns'] = finalTableData.columns
            content['rows'] = finalTableData.to_dict('records')
            content['id'] = finalTableData['Topic_id'].tolist()
            content['top10'] = finalTableData['Top_ten_words'].tolist()
            topic_number = content['max_key']
            content['max_key'] = topic_number
            content['top_responses'] = 10
    return render(request, 'basic.html', content)


# Rename topic
def rename_topic(request):
    form = forms.RenameTopic(request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            topic_number = form.cleaned_data['topic_number']
            name = form.cleaned_data['name']
            m = topic_modeling(new_dict['_A'], content['max_key'], new_dict['_terms'], new_dict['final'])
            m.rename_topic(topic_number, name, new_dict['theme_keywords'])
            finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
            content['columns'] = finalTableData.columns
            content['rows'] = finalTableData.to_dict('records')
            topic_number = content['max_key']
            content['max_key'] = topic_number
            content['top_responses'] = 10
    return render(request, 'basic.html', content)


# Merge two or more topics
def merge_topics(request):
    form = forms.MergeTopic(request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            topic_number_1 = form.cleaned_data['topic_number_1']
            topic_number_2 = form.cleaned_data['topic_number_2']
            m = topic_modeling(new_dict['_A'], content['max_key'], new_dict['_terms'], new_dict['final'])
            m.nmf_modeling();
            w, h = m.fit_transform_merge(topic_number_1, topic_number_2, trial['W'], trial['H'])
            df = m.top_three_keywords()
            _df2, _topic_word = m.documents_per_topic(w, df)
            new_df = m.remove_row(_topic_word)
            m.frequency_plot(new_df)
            m.percentage_plot(new_df)
            m.final_output(new_df, w)
            m.printPFsInPairs()
            m.plot_top_words(3, 'Per topic top 3 keywords')
            finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
            content['columns'] = finalTableData.columns
            content['rows'] = finalTableData.to_dict('records')
            topic_number = content['max_key']
            content['max_key'] = topic_number
            content['top_responses'] = 10
    return render(request, 'basic.html', content)


# Delete keywords per topic
def delete_keywords(request):
    form = KeyWordDeletionForm(request.POST or None)
    if request.method == 'POST':
        print(form)
        if form.is_valid():
            new_stopwords = form.cleaned_data['delete_keyword']
            n_class = Stopwords(add_stopwords=new_stopwords)
            additional = n_class.adding_stopwords()
            content['additional_stopwords'] = additional
            x = Automate_topic_modeling(content['file'], additional)
            new_dict['final'], stop_words = x.pre_processing()
            x.generate_wordcloud(new_dict['final'])
            new_dict['_terms'], new_dict['_A'], _raw_documents = x.tfidf_matrix(new_dict['final'])
            # topic_models, H, W = x.nmf_model(new_dict['_A'])
            # x.build_w2c(_raw_documents)
            # term_rankings, max_key, newDict = x.get_coherence(new_dict['_terms'], topic_models)
            # plot = x.plot_the_coherence_graph(newDict)
            #
            m = topic_modeling(new_dict['_A'], content['max_key'], new_dict['_terms'], new_dict['final'])
            H1, W1 = m.nmf_modeling()
            trial['H'] = H1
            trial['W'] = W1
            # print(W)
            df = m.top_three_keywords()
            _df2, _topic_word = m.documents_per_topic(W1, df)
            m.frequency_plot(_topic_word)
            m.percentage_plot(_topic_word)
            theme_keywords = m.final_output(_topic_word, W1)
            m.printPFsInPairs()
            m.plot_top_words(3, 'Per topic top 3 keywords')
            new_dict['theme_keywords'] = theme_keywords
            finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
            content['columns'] = finalTableData.columns
            content['rows'] = finalTableData.to_dict('records')
            content['id'] = finalTableData['Topic_id'].tolist()
            content['top10'] = finalTableData['Top_ten_words'].tolist()
            topic_number = content['max_key']
            content['max_key'] = topic_number
            content['top_responses'] = 10
    return render(request, 'basic.html', content)


# Split topic
def split_topic(request):
    splitForm = forms.SplitTopic(request.POST or None)
    if request.method == 'POST':
        if splitForm.is_valid():
            topic_number = splitForm.cleaned_data['topic_number']
            m = topic_modeling(new_dict['_A'], content['max_key'], new_dict['_terms'], new_dict['final'])
            H1, W1 = m.nmf_modeling()
            trial['H'] = H1
            trial['W'] = W1
            # print(W)
            df = m.top_three_keywords()
            _df2, _topic_word = m.documents_per_topic(W1, df)
            theme_keywords = m.final_output(_topic_word, W1)
            new_dict['theme_keywords'] = theme_keywords
            m.split_topic(topic_number)
            topic_number = content['max_key']
            content['max_key'] = topic_number
            content['top_responses'] = 10
    return render(request, 'basic.html')


# AJAX
def load_keywords(request):
    topic_id = request.GET.get('topic_id')
    keywords = Keywords.objects.filter(topic_id=topic_id).all()
    return render(request, 'basic.html', {'keywords': keywords})


# Choose the top responses and display
def top_responses(request):
    form = forms.TopResponses(request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            top_responses = form.cleaned_data['top_responses']
            m = topic_modeling(new_dict['_A'], content['max_key'], new_dict['_terms'], new_dict['final'])
            H1, W1 = m.nmf_modeling()
            df = m.top_three_keywords()
            _df2, _topic_word = m.documents_per_topic(W1, df)
            m.frequency_plot(_topic_word)
            m.percentage_plot(_topic_word)
            output = m.final_output(_topic_word, W1, top=top_responses)
            finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
            content['columns'] = finalTableData.columns
            content['rows'] = finalTableData.to_dict('records')
            topic_number = content['max_key']
            content['max_key'] = topic_number
            content['top_responses'] = top_responses
    return render(request, 'basic.html', content)


# Download the excel file
def down_file(request):
    with open('../static/assets/themes_keywords.xlsx', 'rb') as model_excel:
        result = model_excel.read()
    response = HttpResponse(result)
    file_name = content['output_file_name'] + '.xlsx'
    response['Content-Disposition'] = f'attachment; filename={file_name}'
    return response


# Download WordCloud
def down_file_wordcloud(request):
    with open('../static/images/wordCloud.png', 'rb') as model_excel:
        result = model_excel.read()
    response = HttpResponse(result)
    response['Content-Disposition'] = 'attachment; filename=Wordcloud.png'
    return response


# Download Frequency graph
def down_file_Frequency_of_topics(request):
    with open('../static/images/Frequency_of_topic.png', 'rb') as model_excel:
        result = model_excel.read()
    response = HttpResponse(result)
    response['Content-Disposition'] = 'attachment; filename=Frequency_of_topic.png'
    return response


# Download the Keyword graph
def down_file_top_keywords(request):
    with open('../static/images/keywords.png', 'rb') as model_excel:
        result = model_excel.read()
    response = HttpResponse(result)
    response['Content-Disposition'] = 'attachment; filename=keywords.png'
    return response


# Download the Percentage graph
def down_file_Percentage_of_Topics(request):
    with open('../static/images/percentage_of_topic.png', 'rb') as model_excel:
        result = model_excel.read()
    response = HttpResponse(result)
    response['Content-Disposition'] = 'attachment; filename=percentage_of_topic.png'
    return response

# def undo(request):
#     file = pd.read_csv('../static/history/backup.csv', encoding='utf-8-sig', sep=',', error_bad_lines=False, names=['documents'])
#     x = Automate_topic_modeling(file)
#     new_dict['final'], stop_words = x.pre_processing()
#     x.generate_wordcloud(new_dict['final'])
#     new_dict['_terms'], new_dict['_A'], _raw_documents = x.tfidf_matrix(new_dict['final'])
#     m = topic_modeling(new_dict['_A'], content['max_key'], new_dict['_terms'], new_dict['final'])
#     H1, W1 = m.nmf_modeling()
#     trial['H'] = H1
#     trial['W'] = W1
#     # print(W)
#     df = m.top_three_keywords()
#     _df2, _topic_word = m.documents_per_topic(W1, df)
#     m.frequency_plot(_topic_word)
#     m.percentage_plot(_topic_word)
#     theme_keywords = m.final_output(_topic_word, W1)
#     m.printPFsInPairs()
#     m.plot_top_words(3, 'Per topic top 3 keywords')
#     new_dict['theme_keywords'] = theme_keywords
#     finalTableData = pd.read_csv('../static/assets/themes_keywords.csv')
#     content['columns'] = finalTableData.columns
#     content['rows'] = finalTableData.to_dict('records')
#     content['id'] = finalTableData['Topic_id'].tolist()
#     content['top10'] = finalTableData['Top_ten_words'].tolist()
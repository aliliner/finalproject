import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components
import shapely.geometry
import folium
import re

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from streamlit_folium import folium_static
from pyvis.network import Network

df     = pd.read_csv('winedata.csv')
df4    = pd.read_csv('codes_cntr.csv')
df11   = pd.read_csv('wineries.csv')
df_msk = pd.read_csv('msk_data.csv')

with st.echo(code_location='below'):
    # parse simplewine.ru using the Scrappy library in the file smpl.py
    # results in the winedata.csv file

    # the product pages in the catalog are similar, but poorly standardized
    # need to check the coincidence of the articles and alternative articles

    df_check = df
    df_check = df_check.where(df_check['code'] != df_check['code_check']).dropna()
    len(df_check.index)
    # len(df.index) is 0
    # the articles matched -> it's ok to delete code_check column

    df0 = df
    df0 = df0.drop(['code_check'], axis=1)
    df0 = df0.set_index('title')
    df0 = df0.groupby(['code'], as_index=True)['value'].agg(dict)
    df0 = df0.to_dict()

    df1 = pd.DataFrame(df0).transpose()
    df1.columns = [elem.replace(":", "") for elem in df1.columns]

    old_country_list = ['Франция', 'Италия', 'Испания', 'Германия', 'Австрия', 'Португалия', 'Греция']
    column_list = ['Цена', 'Рейтинг', 'Страна', 'Старый свет', 'Возраст', 'Объем', 'Крепость', 'Потенциал мин',
                   'Виноград #']

    df2 = df1[['price', 'rating', 'Вино', 'Сахар', 'Страна, регион', 'Виноград', 'Крепость', 'Объем', 'Регион', 'Год',
               'Потенциал хранения', 'Аппелласьон', 'Категория', 'Декантация']]

    df2 = df2.rename(columns={'price': 'Цена',
                              'rating': 'Рейтинг'
                              }
                     )

    df2 = df2[df2['Потенциал хранения'].notnull()]
    df2 = df2[df2['Крепость'].notnull()]
    df2 = df2[df2['Рейтинг'].notnull()]
    df2 = df2[df2['Объем'].notnull()]
    df2 = df2[df2['Вино'].notnull()]
    df2 = df2[df2['Цена'].notnull()]
    df2 = df2[df2['Год'].notnull()]

    df2 = df2[df2['Рейтинг'] != '0.0']

    df2[['Страна', 'Регион_2']] = df2['Страна, регион'].str.split('##', expand=True)
    df2 = df2.drop(['Регион_2'], axis=1)

    df2['Цена'] = df2.apply(lambda row: float(re.findall(r'(.*)( ?)(₽)', row['Цена'])[0][0].replace(' ', '')), axis=1,
                            result_type='expand')
    df2['Объем'] = df2.apply(lambda row: float(re.findall(r'(.*)( [А-Яа-я]*)', row['Объем'])[0][0]), axis=1,
                             result_type='expand')
    df2['Рейтинг'] = df2.apply(lambda row: float(row['Рейтинг']), axis=1, result_type='expand')
    df2['Крепость'] = df2.apply(lambda row: float(re.findall(r'(.*)(%[А-Яа-я]*)', row['Крепость'])[0][0]), axis=1,
                                result_type='expand')

    df2['Виноград #'] = df2.apply(lambda row: len(row['Виноград'].split('##')), axis=1, result_type='expand')
    df2['Потенциал мин'] = df2.apply(
        lambda row: int(re.findall(r'([0-9]{1,4})( |-|$|[А-Яа-я])', row['Потенциал хранения'])[0][0]), axis=1,
        result_type='expand')
    df2['Старый свет'] = df2.apply(lambda row: int(row['Страна'] in old_country_list), axis=1, result_type='expand')

    df2['Год'] = df2['Год'].astype(int)
    df2['Возраст'] = 2022 - df2['Год']

    df2 = df2[
        ['Цена', 'Рейтинг', 'Вино', 'Сахар', 'Страна', 'Регион', 'Старый свет', 'Год', 'Возраст', 'Объем', 'Крепость',
         'Потенциал мин', 'Виноград #', 'Аппелласьон', 'Категория', 'Декантация']]

    df2 = df2.rename(columns={'Цена': 'Price',
                              'Рейтинг': 'Rating',
                              'Вино': 'Colour',
                              'Сахар': 'Sweetness',
                              'Страна': 'Country',
                              'Регион': 'Region',
                              'Старый свет': 'PartOfTheWorld',
                              'Год': 'Year',
                              'Возраст': 'Maturity',
                              'Виноград #': 'GrapeSortNumber',
                              'Объем': 'Volume',
                              'Крепость': 'ABV',
                              'Потенциал хранения': 'PotentialStorage',
                              'Потенциал мин': 'MinPotentialStorage',
                              'Аппелласьон': 'Appellation',
                              'Категория': 'Category',
                              'Декантация': 'Decantation'
                              }
                     )

    df3 = pd.DataFrame()
    df3 = df2
    df3['PartOfTheWorld'] = np.where(df3['PartOfTheWorld'] == 1, 'Old World', 'New World')
    # now the table is more informative and readable

    st.title("Истина в вине!")
    st.markdown("""
         _Это финальный проект по курсу «Наука о данных», 2021-2022 учебный год_. 
            """)

    st.markdown("""
          Вино - один из самых благородных алкогольных напитков с большой историей. На самом деле, под этим словом скрывается
          бесконечно большой спектр вкусов и ароматов, которые зависят от особенностей производства и состава вина. Вино не только приносит 
          гастрономическое удовольствие, но и делает вечер незабываемым. Не стоит забывать и о том, что когда вы пьете вино, вы 
          прикасаетесь к особой культуре, а вернее будет сказать культурам разных народов.  
            """)
    st.markdown("""
          SimpleWine является одной из ведущих фирм-импортеров вина, а так же крупным винным ритейлером и дистрибьютором. Помимо 
          собственной розницы, они также работают с частными и корпоративными клиентами, компаниями. В их винотеках можно не только 
          купить бутылочку вина, но и принять участие в дегустации или просто отдохнуть в фирменном ресторане. У них еще есть 
          собственное приложение (лично советую, оно удобное и красивое).
            """)
    st.markdown("""
          Культура потребления вина мне, мягко говоря, не чужда, поэтому SimpleWine давно стал частью моей жизни. И вот пришло
          время делать проект, и тема нашлась сама собой. На сайте simplewine.ru более 5 тысяч уникальных вин, на сранице каждого
          из которых есть перечень индивидуальных характеристик. Показалось, что это может стать хорошей базой для нашего мини-исследования. Начнем 
          погружение в мир вин!
            """)

    st.markdown("""
          Проект будет разделен на несколько частей (назовем из главами), в начале каждой из которых будут перечислены задействованные инструменты. Внутри 
          частей будет описание того, как были получены результаты, а также сама интерпретация результатов.
            """)

    st.subheader("Глава 1. Поиск информации")
    st.markdown("""
          В этой главе:
          
          - сложные технологии веб-скреппинга на основе Scrapy (более 50 строк кода)
          
          - нетривиальная обработка результатов скреппинга с помощью pandas
          
          - регулярные выражения
            """)

    st.markdown("""
          В каталоге Simplewine вы можете найти несколько тысяч неповторимых вин. Однако веб-скреппинг данного каталога
          является очень сложной задачей. Для создания полноценного датасета надо зайти на персональную страницу каждого
          вина из каталога и собрать информацию из нескольких разных мест (условно можем назвать их "контейнерами"). Однако, 
          особую сложность представляет тот факт, что страницы вин хоть и выглядят одинаковыми, назвать их стандартизированными нельзя.
          В перечне характеристик могут отсутствовать некоторые позиции, которые присутствуют у других вин (например, не указана
          крепость или регион производства). Также сами перечни характеристик могут отличаться в последовательности позиций (например, 
          у одного вина сначала указан сорт винограда, а потом крепость, а у другого наоборот, сначала крепость, а потом сорт винограда).
          Это основные, но не последние сложности, с которыми мы столкнемся.
            """)
    st.markdown("""
          Сам код парсинга вы можете найти в файле smpl.py, где с помощью библиотеки **Scrapy** осуществляется веб-скреппинг 
          каталога вин SimpleWine. Хочу сразу отметить, что BeautifulSoup использовался только как вспомогательный инструмент
          для сбора информации внутри "контейнеров" с характеристиками, в то время как само извлечение информации с персональных страниц
           вин и сбор данных с сайта каталога был осуществлен силами библиотеки Scrapy. 
            """)

    st.markdown("""
          В силу структурированности, но остутствия стандартизации персональных страниц вин из каталога был получен файл winedata.csv:
            """)

    st.dataframe(df)

    st.markdown("""
          Как упоминалось выше, страницы не были стандартизированы. В связи с этим с персональных страниц вин также собирался проверочный артикул,
          который показал бы нам, если бы вдруг появились исключения-ошибки в результате скреппинга. В коде вы можете найти проверку (и 
          что таких ошибок нет).
                """)

    st.markdown("""
          После нетривиального использования возможностей **Pandas** и применения **регулярных выражений** был получен DataFrame, с которым уже можно полноценно работать (более подробно
          о работе с изначальным DataFrame вы можете узнать в коде):
    """)

    st.dataframe(df3)

    st.markdown("""
          Строки кода уже подошли к первой сотне, но это только начало.
            """)

    st.subheader("Глава 2. Анализ информации")
    st.markdown("""
          В этой главе:
    
          - много сложных визуализаций
    
          - продолжение нетривиальной обработки результатов скреппинга с помощью pandas
    
          - интерактивные графы, раскрывающие структуру каталога (его географический аспект)
            """)

    st.markdown("""
          Как упоминалось выше, SimpleWine является одним из ведущих ритейлеров и дистрибьюторов импортеров вина в Россию. Откуда привозят вино? Давайте
          посмотрим в целом на географическое разнообразие вин из каталога:
    """)

    # to make a visualization on the map, it is necessary to obtain a special country code

    list_cnt = sorted(df2['Country'].unique().tolist())
    df5 = df2
    df5 = df5.groupby(['Country'])['Country'].count()

    num_cnt = []

    for i in list_cnt:
        num_cnt.append(df5[i])

    df6 = pd.DataFrame()
    df6['Country'] = list_cnt
    df6['WineNumber'] = num_cnt

    df6 = pd.merge(df6, df4, on="Country")
    df6['Wine'] = 'Wine'

    map1 = px.choropleth(df6, color='Wine',
                         locations="Alpha-3",
                         hover_name="Country",
                         height=450,

                         color_discrete_sequence=px.colors.diverging.RdYlGn,

                         title="The origin of wines from the Simplewine catalog",
                         labels={"WineNumber": "unique wine bottles",
                                 "Alpha-3": "code of country"
                                 }
                         )
    map1.update_geos(fitbounds="locations")

    map1.update_layout(
        margin=dict(t=50, b=0, l=0, r=0),
        title_font=dict(size=20),
        showlegend=False,
        height=425,
        width=750
    )
    st.plotly_chart(map1)

    st.markdown("""
          Из этой карты видно, что в каталоге SimpleWine собраны вина со всего света. В их винотеках есть вина со всех материков,
          кроме Антарктиды. Однако, кроме как о географическом разнообразии производителей вин, этой карте больше не о чем нам рассказать. 
          Интересно взглянуть на контраст между производителями вин:
    """)
    colors = []

    uni_bot_sum = str(df6['WineNumber'].sum()) + ' unique bottles'
    fig1 = go.Figure(data=[go.Pie(labels=df6['Country'],
                                  values=df6['WineNumber']
                                  )
                           ]
                     )
    color_discrete_sequence = ['#EAA148', '#D4514A', '#DF7832', '#CDAF95', '#FFA3A3', '#D64242', '#9A2121',
                               '#D11F3F', '#FFA583', '#D11F3F', '#FFEFD5', '#FF615D', '#FFA583', '#DAD8C0', '#FFDAB9',
                               '#EAF0F1',
                               '#F4E3AD', '#E58A6B', '#FF6666', '#EECBAD']

    fig1.update_traces(hole=.7, hoverinfo="label+percent+name",
                       marker=dict(colors=color_discrete_sequence))
    fig1.update_layout(
        margin=dict(t=50, b=0, l=0, r=0),
        title_text="The contrast in origin of wines from the Simplewine catalog",
        annotations=[dict(text=uni_bot_sum, x=0.5, y=0.5, font_size=23, showarrow=False)])
    st.plotly_chart(fig1)

    st.markdown("""
    На графике выше мы видим яркий контраст в соотношении количества уникальных бутылок вина среди стран-производителей вин,
     у которых закупает продукцию SimpleWine (напомню, что уникальные бутылки вина - это, по сути, артикулы, то есть уникальные позиции).
     В начале проекта я говорила о наличии более чем 5 тысяч позиций в каталоге вин, но сейчас на графике отмечено лишь 3308 уникальных вин.
     Почему так? Ответ в коде. Изначальный датасет пришлось очистить от винных напитков, портвейнов, и других позиций, не соответствующих
     заявленной теме (а цель (истина) заключалась именно в вине). Также, к сожалению, пришлось очистить датасет от вин, о котоых не было
     достаточной для дальнейшего исследования информации. Тем не менее, мы уже можем сделать некоторые выводы.
    """)

    st.markdown("""
    Благодаря второму графику, мы видим, что сразу после вывода о разнообразии и широком спектре стран-производителей вин в каталоге
    SimpleWine, мы должны подчеркнуть неравномерность этого разнообразия. В каталоге вин наблюдается сильный перекос в сторону 
    Европы, а если быть точнее, то Франции, Италии и Испании. Эти три страны обеспечивают более чем 75% каталога вин. Это не 
    является чем-то плохим или неожиданным, так как Франция, Италия и Испания входят в так называемый 'Старый свет' (у вин это понятие 
    не эквивалентно географическому пониманию этого термина). Исторически сложилось так, что эти страны являются крупнейшими производителями 
    вин, так что их доминирующее присутствие в каталоге не должно пугать. 
    """)

    st.markdown("""
    Мы поговорили о винах из каталога SimpleWine на уровне континентов и стран. Но, у вин играет важную роль не только страна происхождения,
    но и регион производства. Именно регион дает гарантию вкуса и качества. Если идти дальше, то в мире вин есть такое понятие, как
    аппелласьон. Это слово означает, что винодельческая территория и обязательные для неё условия выращивания винограда и способы 
    производства вина закреплены законодательно на международном уровне. По своей сути аппелласьон это один из самых главных
    показателей качества, который основывается на гораздо более узком (чем понятие 'страна') территориальном признаке. 
    """)

    st.markdown("""
    Представить спектр регионов или аппелласьонов на карте в данном проекте не представляется возможным, но мы можем посмотреть
    на структуру каталога SimpleWine, создав граф с вершинами в Частях мира (Старый/Новый свет), странах, и исходящих из них 
    регионов. Таким образом мы увидим структуру и распределение между регионами среди представленных в каталоге вин.
    """)

    df3.reset_index(inplace=True)
    df7 = df3[['index', 'Region', 'Country', 'PartOfTheWorld', 'Appellation']]

    df7.rename(columns={'index': 'Code'})
    df7.dropna(subset=["Region"], inplace=True)
    df7.dropna(subset=["Appellation"], inplace=True)

    level = st.radio(
        "Мне интересна структура ",
        ("Регионов", "Аппелласьонов"))

    if level == "Регионов":
        X = nx.Graph()

        X = nx.from_pandas_edgelist(df7, source='Country', target='Region')
        X = nx.from_pandas_edgelist(df7, source='PartOfTheWorld', target='Country')

        X.add_node('Wine', color='#960018', size=10)
        X.add_edges_from([('Wine', 'Old World'), ('Wine', 'New World')], color='#C41E3A')

        X.add_edges_from(zip(df7['PartOfTheWorld'], df7['Country']), color='#C41E3A')
        X.add_edges_from(zip(df7['Country'], df7['Region']), color='#C41E3A')

        wine_net = Network('550px', '550px')
        wine_net.from_nx(X)

        wine_net.save_graph(r'C:\Users\lizzie\Desktop\pyvis_graph.html')

        HtmlFile = open(r'C:\Users\lizzie\Desktop\pyvis_graph.html', 'r', encoding='utf-8')

        components.html(HtmlFile.read(), height=550)

    else:
        X = nx.Graph()

        X = nx.from_pandas_edgelist(df7, source='Country', target='Appellation')
        X = nx.from_pandas_edgelist(df7, source='PartOfTheWorld', target='Country')

        X.add_node('Wine', color='#960018', size=10)
        X.add_edges_from([('Wine', 'Old World'), ('Wine', 'New World')], color='#C41E3A')

        X.add_edges_from(zip(df7['PartOfTheWorld'], df7['Country']), color='#C41E3A')
        X.add_edges_from(zip(df7['Country'], df7['Appellation']), color='#C41E3A')

        wine_net = Network('550px', '550px')
        wine_net.from_nx(X)

        wine_net.save_graph(r'C:\Users\lizzie\Desktop\pyvis_graph.html')

        HtmlFile = open(r'C:\Users\lizzie\Desktop\pyvis_graph.html', 'r', encoding='utf-8')

        components.html(HtmlFile.read(), height=550)

    st.markdown("""
    Напомню, что при каждом запуске граф может немного отличаться (это нормально, так как положения точек не закреплены). Чтобы
    увидеть названия **приблизьте** с помощью мышки (названия были убраны изначально, чтобы не нагружать граф). В ценктре находится 
    стартовая точка - 'Вино', а далее идет деление на Часть света (напомню, что тут подразумевается винное понятие), Страну и Регион/Аппелласьон (на ваш выбор).
    Точки и граф интерактивны, вы можете **переставлять вершины** с помощью мышки.
    """)

    st.markdown("""
    Итак, на этих графах можно увидеть структуру каталога SimpleWine относительно регионов/аппелласьонов, на ваш выбор.
    В принципе, мы видим, что SimpleWine держит очень высокую планку относительно разнообразия и качества своей продукции 
    (особенно это видно по графу с аппелласьонами). Так же в очередной раз можно заметить, что Италия и Франция выделяются
    среди других стран (только тут уже по количеству апелласьонов). Такой большой спектр аппелласьонов в этих двух странах не
    должен пугать, так как это результат исторических и климатических предпосылок. Однако, правильнее будет сделать вывод о том,
    что в каталоге SimpleWine представлено сильно больше вин из Франции, Италии и Испании, однако этот перекос сглаживается 
    большим спектром аппелласьонов (из которых SimpleWine импортирует вина). Таким образом поддерживается разнообразие и многогранность вин в каталоге.
    Напомню, что аппелласьон означает качество и уникальность.
    Если бы в каталоге количество аппелласьонов Франции, Италии и Испании не было бы таким большим (при условии что эти три страны 
    обеспечивают более чем 75% каталога), то это означало бы, что большая часть поставляемой продукции однотипна и однородна.
    Иными словами, это означало бы скудный ассортимент, но благодаря графам видно, что это не так. 
    """)

    st.markdown("""
    Теперь давайте отойдем от территориальной привязки и посмотрим на ценовое распределение вин в каталоге (так как у SimpleWine
    есть экстра-дорогие вина, цены которых под миллион рублей и выше, мы установим верхнюю планку в 30к рублей (только для информативности), но
    вы можете поменять ее значение):
    """)

    max_price = st.slider('Максимальная цена:', 1000, 500000, 30000)

    df8 = df3[df3['Price'] < max_price]
    fig2 = px.histogram(df8, x="Price", labels={'Price': 'Price of wine'},
                        title='Histogram of wine prices',
                        color_discrete_sequence=px.colors.diverging.RdYlGn
                        )
    fig2.add_trace(go.Histogram(opacity=0.75))
    st.plotly_chart(fig2)

    st.markdown("""
    Из данной гистограммы видно, что значительная часть вин имеет цену не превышающую 5к рублей. Также, благодаря слайдеру, мы можем наблюдать 
    широкий диапозон цен в каталоге SimpleWine. Каждый надет то, что ему по душе и по карману :)
    """)

    st.markdown("""
    Одним из показателей вина является потенциал хранения, то есть срок, в рамках которого вино готово к употреблению. Посмотрим на его
    распределение:
    """)

    df9 = df3
    fig3 = px.histogram(df9, x="MinPotentialStorage", labels={'MinPotentialStorage': 'Potential storage, years'},
                        title='Histogram of wine potential storage',
                        color_discrete_sequence=px.colors.diverging.RdYlGn,
                        nbins=25
                        )
    fig3.add_trace(go.Histogram(opacity=0.75))
    st.plotly_chart(fig3)

    st.markdown("""
    Как можно заметить, потенциал хранения некоторых вин измеряется десятками лет. К слову, с годами вино растет в цене, что позволяет рассматривать его
    как инвестиционный инструмент (ну или как предмет для коллекционирования, если вам это интересно).
    """)

    st.markdown("""
    Еще интересно проверить, как распределены вина в каталоге SimpleWine относительно Потенциала хранения, Возраста и Крепости.
    На графике ниже мы можете видеть это распределенеие с учетом цен вин. Чем больше окружность - тем больше стоит отмеченная на графике 
    позиция из каталога вин. Это график наглядно показывает, что высокий ценник на вино не означает более высокий градус напитка.
    Как раз таки наоборот, чем дешевле вино, тем более вероятно что оно будет крепче (по сравнению с другими винами).
    """)

    tr_w = px.scatter_ternary(df3, a="MinPotentialStorage", b="Maturity", c="ABV",
                              hover_name="Country",
                              size="Price",
                              color_discrete_sequence=["darkred"],
                              size_max=40,
                              opacity=0.5,
                              title="Distribution of wines in catalog relative to the Potential Storage, ABV and Maturity",
                              labels={"MinPotentialStorage": "Потенциал Хранения",
                                      "Maturity": "Возраст",
                                      "ABV": "Крепость"}
                              )
    tr_w.update_layout(
        title_font=dict(size=20),
        font_size=15,
        height=600,
    )
    st.plotly_chart(tr_w)

    st.markdown("""
    Еще из графика выше можно сделать вывод о том, что более дорогие вина имеют больший возраст и/или потенциал хранения (что как 
    раз иллюстрирует мои утверждения в этой главе).
    """)

    st.markdown("""
    Теперь, когда мы изучили датасет, пора двигаться дальше.
    """)

    st.subheader("Глава 3. Цена вина")
    st.markdown("""
          В этой главе:
    
          - машинное обучение (2 модели) + немного математики (numpy)
    
          - предсказание цены вина на основе регрессий
          
          - немного визуализаций
    
            """)

    st.markdown("""
    Уже получен и изучен датасет, уже есть представление о SimpleWine и его продукции, но все еще чего-то не хватает. 
    Не хватает предстакателя цены вина! С помощью машинного обучения было построено 2 модели, одну из которых мы будем использовать
    для предсказания цены вина по заданным параметрам. 
    """)

    st.markdown("""
    Начнем с регрессии, которая достаточно хорошо описывает ценообразование вин (согласно статистическим показателям, которые были посчитаны
    с помощью numpy и метода наименьших квадратов). Данная модель основывалась на следующих показателях: Регион, Апелласьон, Часть Света,
    Крепость, Потенциал хранения и Рейтинг (в таком порядке и указаны в итоговой табличке переменные).
    """)

    appellasion_list = ['Bolgheri Sassicaia DOC', 'Pauillac AOC', 'Cote Rotie AOC', 'Pessac-Leognan AOC', 'Priorat DOC',
                        'Ribera del Duero DO', 'Chianti Classico DOCG',
                        'Napa Valley AVA', 'Pomerol AOC', "Val d'Arno di Sopra DOC", 'Recioto della Valpolicella DOC',
                        'Margaux AOC', 'Saint-Estephe AOC', 'Meursault AOC', 'Toscana IGT', 'Rutherford AVA',
                        'Saint-Julien AOC',
                        'Sauternes AOC', 'Montrachet Grand Cru AOC', 'Valle del Aconcagua', 'Santa Cruz Mountains AVA',
                        'Chambertin Clos de Beze Grand Cru AOC',
                        'Barolo DOCG', 'Romanee Saint-Vivant Grand Cru AOC', 'Bolgheri Superiore DOC',
                        'Saint-Emilion Grand Cru AOC', 'Pago de Arinzano DO', 'Willamette Valley AVA',
                        'Puligny-Montrachet 1-er Cru AOC',
                        'Puligny-Montrachet AOC', 'Beaune 1-er Cru AOC', 'Brunello di Montalcino DOCG',
                        'Corton-Charlemagne Grand Cru AOC', 'Venezie IGT',
                        'Valpolicella Classico DOC', 'Chambertin Grand Cru AOC', 'Haut-Medoc AOC',
                        'Amarone della Valpolicella DOCG', 'Chablis Grand Cru AOC', 'Lujan de Cuyo',
                        'Puente Alto', 'Marche IGT', 'Nuits-Saint-Georges 1-er Cru AOC', 'Clos-de-Tart Grand Cru AOC',
                        'Clos des Lambrays Grand Cru AOC', 'Gevrey-Chambertin AOC',
                        'Carmignano DOCG', 'Colli della Toscana Centrale IGT', 'Echezeaux Grand Cru AOC',
                        'Vigneti delle Dolomiti IGT', 'Bourgogne AOC', 'Bordeaux AOC', 'Bolgheri DOC',
                        'Gevrey-Chambertin 1-er Cru AOC', 'Volnay 1-er Cru AOC', 'Alto Adige Terlano DOC', 'Veneto IGT',
                        'Chassagne-Montrachet 1-er Cru AOC'
                        ]

    category_list = ['Denominazione di Origine Controllata (DOC)', '1-er Grand Cru Classe',
                     "Appellation d'Origine Controlee (AOC)", 'Denominacion de Origen Calificada (DOCa)',
                     'Grand Cru Classe',
                     'Denominacion de Origen (DO)', 'Denominazione di Origine Controllata e Garantita (DOCG)',
                     'American Viticultural Areas (AVA)', '3-eme Grand Cru Classe',
                     'Indicazione Geografica Tipica (IGT)',
                     '5-eme Grand Cru Classe', '4-eme Grand Cru Classe', '2-eme Grand Cru Classe',
                     '1-er Grand Cru Superieur', 'Australian Geographical Indications (GI)',
                     '1-er Grand Cru Classe "B"', 'Denominacion de Pago', 'Vino d’Italia', 'Indicacion Geografica (IG)'
                     ]

    region_list = ['Аконкагуа', 'Апулия', 'Бордо', 'Бургундия', 'Венето', 'Долина Роны', 'Иудейские Холмы',
                   'Калифорния', 'Кастилья и Леон', 'Каталония', 'Ломбардия', 'Марке', 'Мендоса', 'Наварра',
                   'Нижняя Австрия', 'Орегон', 'Пьемонт', 'Риоха', 'Сицилия', 'Токай', 'Тоскана',
                   'Трентино-Альто Адидже', 'Центральная Долина', 'Южная Австралия'
                   ]

    old_country_list = ['Франция', 'Италия', 'Испания', 'Германия', 'Австрия', 'Португалия', 'Греция']
    decantation_list = ['декантация', 'аэрация']

    df10 = df2
    df10['PartOfTheWorld'] = df10.apply(lambda row: int(row['Country'] in old_country_list), axis=1,
                                        result_type='expand')
    df10['Decantation'] = df10.apply(lambda row: int(row['Decantation'] in decantation_list), axis=1,
                                     result_type='expand')
    df10['Appellation'] = df10.apply(lambda row: int(row['Appellation'] in appellasion_list), axis=1,
                                     result_type='expand')
    df10['Category'] = df10.apply(lambda row: int(row['Category'] in category_list), axis=1, result_type='expand')
    df10['Region'] = df10.apply(lambda row: int(row['Region'] in region_list), axis=1, result_type='expand')

    df10 = df10[df10['Price'] < 30000]
    df10 = df10[
        ['Price', 'Rating', 'Region', 'PartOfTheWorld', 'Appellation', 'Volume', 'ABV', 'GrapeSortNumber', 'Maturity',
         'MinPotentialStorage']]
    a = df10['Price'].to_list()

    b = [df10['Region'].to_list(),
         df10['Appellation'].to_list(),
         df10['PartOfTheWorld'].to_list(),
         df10['ABV'].to_list(),
         df10['MinPotentialStorage'].to_list(),
         df10['Rating'].to_list()
         ]


    def regr(x, y):
        ones_y = np.ones(len(y[0]))
        Ys = sm.add_constant(np.column_stack((y[0], ones_y)))

        for elem in y[1:]:
            Ys = sm.add_constant(np.column_stack((elem, Ys)))

        lin_regr_mod = sm.OLS(x, Ys)
        results = lin_regr_mod.fit()

        return results


    st.text(regr(a, b).summary())

    st.markdown("""
    Предсказывать цены мы будем с помощью следующей регрессии, а у этой будет интересно изучить результаты. Согласно этой модели
    большую роль в ценообразовании играет регион, апелласьон и потенциал хранения вина. Как упоминалось выше, данные показатели действительно 
    являются гарантами качества вина и изысканности вкуса. (Мое утверждение о роли в ценоообразовании основывается на значении коэффициентов
    (столбец coef) переменных xi, где i = {1, 2, ..., 6}). По остальным частям данной модели можно сказать, что результаты довольно достоверны
    (R-squared) и значимость переменных соответствует тому, что были получены релевантные результаты.
    """)

    st.markdown("""
    Эта регрессия хорошо описывает саму концепцию ценообразования вин. Найденные коэффициенты можно использовать чтобы предсказать цену интересующего
    вина. Однако предстказывать цены мы будем с помощью Алгоритма Ближайшего Соседа (код в main.py), для чего построим следующую регрессию.""")

    st.markdown("""
    Итак, перейдем к Алгоритму Ближайшего Соседа. Мы будем строить регрессию на основании поиска 10 ближайших соседей. Для начала сделаем базовую вещь 
    (построим предсказание самого DataFrame (того, на котором проходило обучение)) и покажем, что данный Алгоритм можно назвать репрезентативным для поиска цены вина.
    """)

    knn = KNeighborsRegressor(n_neighbors=10)

    var_list = ['PartOfTheWorld', 'Appellation', 'Volume', 'ABV', 'GrapeSortNumber', 'Maturity', 'MinPotentialStorage']
    knn.fit(df10[var_list], df10[['Price']])

    df10.plot.scatter(x='ABV', y='Price', color='#C70039', alpha=0.7)
    plt.plot(df10['ABV'], knn.predict(df10[var_list]), 'o', color='#FFC300', alpha=0.1, lw=1)
    st.pyplot()

    st.markdown("""
    Теперь будем учить и тестировать модель на разных DataFrame, для чего разобьем первоначальный с помощью train_test_split.
    """)

    train_df, test_df = train_test_split(df10)
    knn1 = KNeighborsRegressor(n_neighbors=10)

    knn1.fit(train_df[var_list], train_df[['Price']])
    test_df.plot.scatter(x='ABV', y='Price', color='#C70039', alpha=0.7)
    plt.plot(test_df['ABV'], knn1.predict(test_df[var_list]), 'o', color='#FFC300', alpha=0.1, lw=1)
    st.pyplot()

    st.markdown("""
    Как мы видим, предсказания довольно точны. Давайте для интереса проверим таккже предскажания на основании Возраста и Потенциального хранения вина.
    """)

    col1, col2 = st.columns(2)

    with col1:
        train_df, test_df = train_test_split(df10)
        knn2 = KNeighborsRegressor(n_neighbors=10)

        knn2.fit(train_df[var_list], train_df[['Price']])
        test_df.plot.scatter(x='Maturity', y='Price', color='#C70039', alpha=0.7)
        plt.plot(test_df['Maturity'], knn2.predict(test_df[var_list]), 'o', color='#FFC300', alpha=0.1, lw=1)
        st.pyplot()

    with col2:
        train_df, test_df = train_test_split(df10)
        knn3 = KNeighborsRegressor(n_neighbors=10)

        knn3.fit(train_df[var_list], train_df[['Price']])
        test_df.plot.scatter(x='MinPotentialStorage', y='Price', color='#C70039', alpha=0.7)
        plt.plot(test_df['MinPotentialStorage'], knn3.predict(test_df[var_list]), 'o', color='#FFC300', alpha=0.1, lw=1)
        st.pyplot()

    st.markdown("""
    Как мы видим, относительно этих трех переменных (характеристик вина) можно построить релевантные предсказания цены. Однако,
    в самом начале было установлено органичение на максимальную цену вина. Это было сделано с целью исключения выбосов и устранения перекосов
    из-за наличия экстра-дорогих вин.
    """)

    st.markdown("""
    Узнайте о предположении (на основе регрессии), сколько будет стоить вино из Старого света с аппелласьоном, при интересующем вас Возрасте.
    """)

    age = st.number_input('Возраст вина', min_value=1, step=1, value=4)
    wine_age = age

    potst = st.number_input('Потенциальное хранение', min_value=1, step=1, value=3)
    wine_pst = potst

    abv = st.number_input('Крепость', min_value=7, max_value=16, step=1, value=12)
    wine_abv = abv

    df12 = pd.DataFrame(columns=var_list)
    df12.loc[0] = [1, 1, 0.75, wine_abv, 1, wine_age, wine_pst]
    price = knn3.predict(df12)
    st.text(str(price[0][0]) + ' руб.')

    st.subheader("Глава 4. Винотеки")
    st.markdown("""
          В этой главе:
    
          - работа с REST API (JSON)
    
          - работа с геоданными с помощью библиотек geopandas, shapely и folium
          
          - еще немного визуализаций
    
            """)

    st.markdown("""
    Теперь, когда у вас есть представление как о винах из каталога SimpleWine, так и о их стоимости, пора отправляться в их фирменную
    винотеку. Для этого нам надо с помощью REST API получить данные о их расположении. Хочу обратить ваше внимание: с помощью
    REST API был получен JSON файл, из которого был собран обычный DataFrame (wineries.csv), а в дальнейшем уже и GeoDataFrame. Начало этого 
    пункта можно найти в файле wineries_start.py (это было сделано отдельно с целью ускорения работы приложения).
    """)

    st.markdown("""
    Итак, в результате работы с REST API (JSON) был получен следующий DataFrame: 
    """)

    st.dataframe(df11)

    # part of code is in wineries_start.py

    gdf1 = gpd.GeoDataFrame(df11, geometry=gpd.points_from_xy(df11['coordinates_lon'], df11['coordinates_lat']))

    # Moscow geojson and work with it is in wineries_start.ry

    for i in range(125):
        df_msk['poly'][i] = shapely.wkt.loads(df_msk['poly'][i])

    gdf2 = gpd.GeoDataFrame(df_msk, geometry='poly')

    gdf3 = gdf1.sjoin(gdf2, how="right", predicate='intersects')
    gdf3 = gpd.GeoDataFrame(gdf3[['id', 'name', 'properties.name', 'properties.description',
                                  'properties.CompanyMetaData.address', 'properties.CompanyMetaData.Hours.text',
                                  'coordinates_lon', 'coordinates_lat']],
                            geometry=gpd.points_from_xy(gdf3['coordinates_lon'], gdf3['coordinates_lat']))

    st.markdown("""
    Из изначального датафрейма видно, что подавляющее большинство фирменных винотек сосредоточено на территории Москвы.
    Поэтому был найден файл Moscow.geojson, чтобы определить распределение винотек SimpleWine в столице (распределение по России 
    или ЦФО не было бы информативным. (Подчеркну, что это не файл из домашней работы, так как он включает в себя большее
    количество административных единиц (единственное, он не включает Новомосковский АО и Троицкий АО, так как они были включены относительно
    недавно). Поэтому тут другой алгоритм работы (в коде видно отличия), другие сложности и другой подход).
    """)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    wineries_counts = gdf3['name'].value_counts()
    fig, ax = plt.subplots(figsize=(13, 13))
    map1 = gdf2.set_index('name').assign(wineries_counts=wineries_counts)
    map1.plot(column=wineries_counts, legend=True, cmap='Reds', edgecolor='black')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.markdown("""
    На данном графике видно, что практически все винотеки SimpleWine сосредоточены в центре Москвы. Если вы все же решили забежать 
    в винотеку SimpleWine - я оставлю вам интерактивную карту с отметками: 
    """)

    gdf3 = gdf3[gdf3['properties.name'].notnull()]

    map_wineries = folium.Map([55.75215, 37.61819], zoom_start=11)
    folium.TileLayer('cartodbpositron').add_to(map_wineries)

    for ind, row in gdf3.iterrows():
        folium.CircleMarker([row.coordinates_lat, row.coordinates_lon],
                            radius=10, color='darkred').add_to(map_wineries)

    folium_static(map_wineries)

    st.markdown("""
    Спасибо за внимание!

    И хорошего тебе лета!!!
    
    P.S. следите за здоровьем!!!!
    """)

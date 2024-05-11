from pyalex import Works, Authors, Sources, Institutions,Topic, Concepts, Funders
import pyalex,openai,ast,pandas as pd, networkx as nx, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
import igraph as ig,leidenalg
import matplotlib.patheffects as patheffects
import random,time,itertools,tqdm,collections
from sklearn.feature_extraction.text import TfidfVectorizer
from networkx.drawing.nx_agraph import graphviz_layout
from adjustText import adjust_text
import os
import requests
from retry import retry

os.getcwd()

def get_journal_name_original(citing):
    '''
    citnig(列)：['primary_location']列しか取れない

    '''
    try:
        return citing['source']['display_name']
    except:
        return None


def get_journal_cited(cited):
    '''
    cited(列):primary_locations列しかとれない
    '''
    try:
        return cited["primary_location"]["source"]['display_name']
    except:
        return 
    
def fetch_all_results(url):
    url += url+"&per-page=200"
    results = []

    # 初回のリクエストでカーソルを取得
    response = requests.get(url + "&cursor=*")
    data = response.json()

    # レスポンスから結果を取得
    if "results" in data:
        results += data["results"]

    # ページング情報を更新
    while "meta" in data and "next_cursor" in data["meta"]:
        next_cursor = data["meta"]["next_cursor"]
        response = requests.get(url + f"&cursor={next_cursor}")
        data = response.json()

        # レスポンスから結果を取得
        if "results" in data:
            results += data["results"]

    return results

    
from pyalex import config

config.max_retries = 3  # 最大再試行数を 3 回に設定
config.retry_backoff_factor = 1  # 再試行間隔を 0.1 秒に設定
config.retry_http_codes = [429, 443, 500, 503]  # 再試行を行う HTTP エラーコードを指定



def get_journal_name(article_raw):
    '''
    citnig(列)：['primary_location']列しか取れない

    '''
    try:
        return article_raw['source']['display_name']
    except:
        return None

def get_topic(topic_raw,class_):
    '''
    topic()：['primary_location']列しか取れない
    class(str):domain,field,subfieldから選ぶ
    '''
    if class_ == "domain":
        try:
          return topic_raw['domain']["display_name"]

        except:
          return None

    elif class_ == "field":
        try:
          return topic_raw['field']["display_name"]

        except:
          return None

    elif class_ == "subfield":

        try:
         return topic_raw['subfield']["display_name"]

        except:
          return None

cols = ["id",'cited_by_api_url',"publication_year","referenced_works","cited_by_count","primary_topic","primary_location","abstract_inverted_index"]

source_ids = [
    "https://openalex.org/sources/s202144432", # Science, technology & society
    "https://openalex.org/sources/s2181421",   # Science, technology & human values
    "https://openalex.org/sources/s68632876",  # Social studies of science
    "https://openalex.org/sources/s104981805", # East Asian science, technology and society
    "https://openalex.org/sources/s124907306", # Public understanding of science
    "https://openalex.org/sources/s30803906",  # JCOM, journal of science communication
    "https://openalex.org/sources/s4210212669",# Science communication
    "https://openalex.org/sources/s190143089", # Science as culture
    "https://openalex.org/sources/s16793705",  # Research Evaluation
    "https://openalex.org/sources/s9731383",   # Research policy(4800)
    "https://openalex.org/sources/s68632876",  # Science and public policy
    "https://openalex.org/sources/s148561398"  # scientometorics
]

# LLM関連の2016年以降の論文を抽出。クエリは改善の余地あり
THRESHOLD = 3
pyalexObj = Works().filter(locations={"source":{"id":"|".join(source_ids)}}, cited_by_count=f">{THRESHOLD-1}",has_references="true").select(cols)

###

os.chdir("/Users/katetsukenkyuushitsu/citingSTS")


field_count = pyalexObj.count()
print(field_count)
it=0
pager = pyalexObj.paginate(per_page=100,n_max=None)
for page in pager:
    it += 1
    df = pd.DataFrame(page)
    df.to_json(f"citingSTS{it}.json", orient="records")
    arr = []
    print(it)
    
os.getcwd()
os.chdir("/Users/katetsukenkyuushitsu/citingSTS/citing_add_ver")
    


import requests
import pandas as pd
import time

def fetch_all_results(url):
    url += "&per-page=200"
    results = []

    # リトライ回数
    retries = 3

    for _ in range(retries):
        try:
            # 初回のリクエストでカーソルを取得
            response = requests.get(url + "&cursor=*")
            data = response.json()

            # レスポンスから結果を取得
            if "results" in data:
                results += data["results"]

            # ページング情報を更新
            while "meta" in data and "next_cursor" in data["meta"]:
                next_cursor = data["meta"]["next_cursor"]
                response = requests.get(url + f"&cursor={next_cursor}")
                data = response.json()

                # レスポンスから結果を取得
                if "results" in data:
                    results += data["results"]

            # エラーが起きなかった場合はループを抜ける
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            print("リトライします...")
            time.sleep(5)
            continue

    return results

# STSを引用している論文を追加

for i in range(1, 160):
    path = f"/Users/katetsukenkyuushitsu/citingSTS/citingSTS{i}.json"
    df = pd.read_json(path, orient="records")
    df = df.dropna(subset=['cited_by_api_url', "primary_location"])
    df["citing_ori"] = df["cited_by_api_url"].apply(fetch_all_results)
    df.to_json(f"0501citing_add_{i}.json", orient="records")
    print(f"{i}番目が終わりました")

#全てのデータフレームを結合
# 重すぎるから10ずつにしましょう


all_df = []
for i in range(1, 160, 10):
    path = f"/Users/katetsukenkyuushitsu/citingSTS/citing_add_ver/0501citing_add_{i}.json"
    df = pd.read_json(path, orient="records")
    all_df.append(df)
    print(f"{i}番目が終わりました")

merged_df = pd.concat(all_df, ignore_index=True)

    
        
import pandas as pd

# データを結合するための空のリストを作成します
all_df = []

# 10ファイルずつのグループでデータを読み込んで結合します
for i in range(1, 160, 10):
    group_df = []
    for j in range(i, min(i + 10, 160)):  # 10ファイルずつ処理する
        path = f"/Users/katetsukenkyuushitsu/citingSTS/citing_add_ver/0501citing_add_{j}.json"
        df = pd.read_json(path, orient="records")
        group_df.append(df)
        print(f"{j}番目が終わりました")
    
    # 10ファイルのデータを結合します
    merged_group_df = pd.concat(group_df, ignore_index=True)
    
    # ローカルに結合されたデータを保存します
    save_path = f"/Users/katetsukenkyuushitsu/citingSTS/citing_add_ver/merged_data_{i}.json"
    merged_group_df.to_json(save_path, orient="records")
    print(f"{i}から{i+9}までのデータを保存しました")










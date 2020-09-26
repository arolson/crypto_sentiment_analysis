from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk import download
from urllib.request import urlopen
from urllib.request import Request
from dateutil.parser import parse
from bs4 import BeautifulSoup
import pandas as pd
import ssl
import json
import re
import os

pd.set_option('display.expand_frame_repr', False)
# nltk.download('vader_lexicon')

def search(q):
    try:
        key = os.getenv("CRYPTOCONTROL_API")
        url = "https://cryptocontrol.io/api/v1/public/news/coin/{}?key={}".format(q, key)
        articles = []
        scontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(request, context=scontext)
        results = json.loads(response.read())

        # Map the results any similarly related articles
        for result in results:
            title = re.sub(r"^\s+|\s+$", "", result['title'].replace("\n", "")).lower()
            url = result['url']
            date = formatDate(result['publishedAt'])
            text = getText(url)
            article = {"title": title, "url": url, "date": date, "text": text}
            articles.append(article)

            similarArticles = result['similarArticles']
            for similar in similarArticles:
                title = re.sub(r"^\s+|\s+$", "", similar['title'].replace("\n", "")).lower()
                url = similar['url']
                date = formatDate(similar['publishedAt'])
                text = getText(url)
                article = {"title": title, "url": url, "date": date, "text": text}
                articles.append(article)

        df = pd.DataFrame(articles)
    except Exception as e:
        print(str(e))
        return pd.DataFrame()
    return df


def formatDate(published):
    d = parse(published).strftime("%m/%d/%Y")
    date = parse(d)
    return date


def getText(url):
    # Fetch the body of the article
    scontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
    response = urlopen(request, context=scontext)
    html = BeautifulSoup(response, features="lxml")
    text = re.sub(r"^\s+|\s+$", "", html.text.replace("\n", "")).lower()

    return text


if __name__ == "__main__":
    df = search("bitcoin")
    # df.to_csv("data.csv", index=False)
    # df = pd.read_csv("data.csv")
    # df.sort_values(by=["date"])

    scores = []
    for title in df.title:
        score = SIA().polarity_scores(title)
        score['title'] = title
        scores.append(score)
    df['vader_score_title'] = pd.DataFrame(scores)['compound']

    scores = []
    for text in df.text:
        score = SIA().polarity_scores(text)
        score['text'] = text
        scores.append(score)
    df['vader_score_text'] = pd.DataFrame(scores)['compound']

    scores = []
    for index, row in df.iterrows():
        text = row["title"] + " " + row['text']
        score = SIA().polarity_scores(text)
        score['text'] = text
        scores.append(score)
    df['vader_score_combined'] = pd.DataFrame(scores)['compound']

    # Average the scores by date
    data = df.groupby(['date']).mean()
    print(data)

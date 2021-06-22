from bs4 import BeautifulSoup
import pandas as pd
import re
import sys


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    df = pd.read_csv(input_file,
                        sep=",", 
                        names=['id', 'body', 'samenvatting', 'article_type', 'article_category'])
    def textify(raw_text):
        if (raw_text is not None) and (isinstance(raw_text, str)):
            soup = BeautifulSoup(raw_text)
            txt = soup.getText()
            txt = re.sub(r'\[\[.*\]\]','', txt)
            txt = re.sub(r'\s{2,}',' ', txt)
            txt = txt.strip()
            return txt
        else:
            return None
    df['body_txt'] = df.body.apply(textify)
    df['samenvatting_txt'] = df.samenvatting.apply(textify)
    df[['id', 'body_txt', 'samenvatting_txt', 'article_type', 'article_category']]\
            .to_feather(output_file)
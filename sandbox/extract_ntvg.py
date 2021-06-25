from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import argparse
from getpass import getpass
from tqdm import tqdm
from pathlib import Path

def textify(raw_text):
    if (raw_text is not None) and (isinstance(raw_text, str)):
        soup = BeautifulSoup(raw_text, features='lxml')
        txt = soup.getText()
        txt = re.sub(r'\[\[.*\]\]','', txt)
        txt = re.sub(r'\s{2,}',' ', txt)
        txt = txt.strip()
        return txt
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing input for the NtvG extractor')
    parser.add_argument('--in', dest='input_file', type=str, default=None)
    parser.add_argument('--out_folder', dest='output_folder', type=str, default="./out")
    parser.add_argument('--source', dest='source', help='Cleaning settings', type=str, default="sql")
    args = parser.parse_args()

    #################
    ## READ ########
    #################
    # from csv
    if args.source == 'file':
        assert args.input_file is not None, "Input error: you need to provide the path to the input file"
        df = pd.read_csv(args.input_file,
                            sep=",", 
                            names=['id', 'body', 'samenvatting', 'article_type', 'article_category'])
    # from sql
    if args.source == 'sql':
        load_dotenv()
        user = os.getenv('MYSQL_USER')
        host = os.getenv('MYSQL_HOST')
        port = os.getenv('MYSQL_PORT')
        database = os.getenv('MYSQL_DATABASE')
        password= getpass(prompt="Please provide SQL pass: ")

        # Create the connection
        connection_string = f'mysql://{user}:{password}@{host}:{port}/{database}'
        connection = create_engine(connection_string)

        # do sql alchemy stuffs
        query = """SELECT DISTINCT field_data_body.entity_id,
       field_data_body.body_value,
       db_summary.field_summary_value,
       db_category.name_category,
       db_vakgebied.name_vakgebied
        FROM field_data_body
                LEFT JOIN
        (SELECT entity_id as ent_id2, field_summary_value
            FROM field_data_field_summary) as db_summary
                ON field_data_body.entity_id = db_summary.ent_id2
                    LEFT JOIN
        (SELECT entity_id as ent_id3, field_category_tid, taxonomy.name as name_category
            FROM field_data_field_category
                LEFT JOIN
                (SELECT tid, name FROM taxonomy_term_data) as taxonomy
                ON field_category_tid=taxonomy.tid
            ) as db_category
                ON field_data_body.entity_id = db_category.ent_id3
                    LEFT JOIN
        (SELECT entity_id as ent_id4, field_vakgebied_tid, taxonomy.name name_vakgebied
            FROM field_data_field_vakgebied
                LEFT JOIN
                (SELECT tid, name FROM taxonomy_term_data) as taxonomy
                ON field_vakgebied_tid=taxonomy.tid
            ) as db_vakgebied
                ON field_data_body.entity_id = db_vakgebied.ent_id4
        """
        df = pd.read_sql_query(query, con=connection)
        df.columns = ['id', 'body', 'samenvatting', 'article_type', 'article_category']

    df['body_txt'] = df.body.apply(textify)
    df['samenvatting_txt'] = df.samenvatting.apply(textify)

    #################
    ## WRITE ########
    #################
    # to feather 
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    df[['id', 'body_txt', 'samenvatting_txt', 'article_type', 'article_category']]\
            .to_feather(os.path.join(args.output_folder, "ntvg_articles.feather"))

    # to individual files per category
    new_folder = os.path.join(args.output_folder, 'category')
    Path(new_folder).mkdir(parents=True, exist_ok=True)
    for _cat in df.article_category.unique():
        if _cat is not None:
            dfc = df[df.article_category==_cat]
            dfc.fillna('', inplace=True)
            with open(os.path.join(new_folder, re.sub(r"[^\w]","_", _cat)+".txt"), 'w', encoding='utf-8') as fw:
                for txt in tqdm(dfc[['body_txt', 'samenvatting_txt']].sum(axis=1)):
                    fw.write(txt+"\n")

    # to individual files per article type
    new_folder = os.path.join(args.output_folder, 'article_type')
    Path(new_folder).mkdir(parents=True, exist_ok=True)
    for _type in df.article_type.unique():
        if _type is not None:
            dfc = df[df.article_type==_type]
            dfc.fillna('', inplace=True)
            with open(os.path.join(new_folder, re.sub(r"[^\w]","_", _type)+".txt"), 'w', encoding='utf-8') as fw:
                for txt in tqdm(dfc[['body_txt', 'samenvatting_txt']].sum(axis=1)):
                    fw.write(txt+"\n")
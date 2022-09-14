import pymysql
import pandas as pd

con = pymysql.connect(host='localhost', user='root', password='aidml2021', db='mhsupport')
cur = con.cursor()

sql = "SELECT * FROM basedata WHERE es_score IS NOT NULL"
cur.execute(sql)

rows = cur.fetchall()
con.close()

basedata = pd.DataFrame(rows, columns=['post_key', 'post_text', 'comment_key', 'comment_text', 'is_score', 'es_score'])
basedata.to_csv('./basedata.csv', index=False)

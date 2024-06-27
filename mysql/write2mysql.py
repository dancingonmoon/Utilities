import pandas as pd
import mysql.connector

# 读取Excel文件
excel_path = "L:\丁翊弘\高考\浙江省2023年普通高校招生普通类第一段平行投档分数线表.xlsx"
df = pd.read_excel(excel_path)

# MySQL数据库配置
config = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'your_database'
}

# 连接MySQL数据库
connection = mysql.connector.connect(**config)
cursor = connection.cursor()

# 获取Excel表格的列名
columns = df.columns.values.tolist()

# 创建MySQL表（如果需要）
create_table_query = f"CREATE TABLE {df.index.name} ({', '.join([f'{col} VARCHAR(255)' for col in columns])});"
cursor.execute(create_table_query)

# 将数据写入MySQL表
for index, row in df.iterrows():
    insert_query = f"INSERT INTO {df.index.name} ({', '.join(columns)}) VALUES ({', '.join(['%s' for _ in range(len(columns))])});"
    cursor.execute(insert_query, tuple(row))

# 提交事务
connection.commit()

# 关闭连接
cursor.close()
connection.close()

# df.itertuples()
import pandas as pd
import mysql.connector
import configparser

def config_read(
    config_path, section="DingTalkAPP_chatGLM", option1="Client_ID", option2=None
):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value


# 读取Excel文件
excel_path = "L:\丁翊弘\高考\浙江省2023年普通高校招生普通类第一段平行投档分数线表.xlsx"

df = pd.read_excel(excel_path)

# MySQL数据库配置
config_path = "E:/Python_WorkSpace/config/mysql.ini"
user, pw = config_read(config_path,section='MySQL', option1='user', option2='password')
database_name = 'gaokao_'
config = {
    'host': 'localhost',
    'user': user,
    'password': pw,
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

df.to_sql()
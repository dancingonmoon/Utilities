import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import configparser
import openpyxl

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



def connect_mysql(host='localhost', user:str='root', password=None, database:str=None):
    """
    连接MySQL数据库, 当错误,则返回错误值,退出程序;
    return: connection
    """
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        cursor = connection.cursor()
        if database is not None:
            cursor.execute(f"USE {database}") # 通过USE database 是否错误,来判断是否已经存在指定的database;
            cursor.close()
        return connection
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"Database '{database}' does not exist.")
        else:
            print(err.errno)
        return err


def df_to_mysql_upon_connection(connection, df:pd.DataFrame, database:str, w2table:str):
    """
    将一个DataFrame写入MySQL数据库;1)表名,列名,全部加上了反引号;2)如果出现缺失值Nan,则转换成SQL的NULL；3)如果主键出现重复值,则跳过该行数据;
    connection: 连接到MySQL数据库的connection;connection = mysql.connector.connect(**config)
    df: 待写入的DataFrame数据;
    database: 写入的数据库名;
    w2table: 写入的表名;
    """
    cursor = connection.cursor()
    rows_sum = 0
    for row  in df.itertuples():
        field_name = ', '.join([f'`{c}`' for c in row._fields])   # 列名需加反引号
        field_value = []
        for c in row:
            if isinstance(c,str):
                field_value.append(f'"{c}"')  # 当字段值为字符串时,加上引号;
            elif pd.isna(c):
                field_value.append('NULL') # 缺失值Nan时, 改成SQL可以识别的NULL,表示空值
            else:
                field_value.append(f'{c}')
        field_value = ', '.join(field_value)
        insert_query = f"""
                                INSERT INTO {database}.`{w2table}`  
                                ({field_name}) 
                                VALUES ({field_value}); 
                                """
        # 表名 需要加反引号 # 中文列名需加反引号# 字符串需加引号,非字符串保持原值 # pandas的Nan改成SQL的NULL
        try:
            cursor.execute(insert_query)
            # 提交事务
            connection.commit()
            rows_sum += 1
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_DUP_ENTRY:
                print(err.msg)
                continue
            else:
                print(err.msg)
                print(insert_query)
                print(f"total {rows_sum} rows have been written into MySQL: {database}.{w2table}")
                # 关闭连接
                cursor.close()
            break

    print(f"total {rows_sum} rows have been written into MySQL {database}.{w2table}")
    # 关闭连接
    cursor.close()

def df_to_mysql(host='localhost', user:str='root', password=None, database:str=None, df:pd.DataFrame=None, w2table:str=None):
    """
    A:新建一个MySQL的连接, 当连接错误,或者Database不存在,退出;
    B: 将一个DataFrame写入MySQL数据库;1)表名,列名,全部加上了反引号;2)如果出现缺失值Nan,则转换成SQL的NULL；3)如果主键出现重复值,则跳过该行数据;
    database: 写入的数据库名;
    w2table: 写入的表名;
    """
    connection = connect_mysql(host=host, user=user, password=password, database=database)
    if isinstance(connection, mysql.connector.Error):
        print(connection)
        return connection
    df_to_mysql_upon_connection(connection, df, database, w2table)
    # 关闭连接
    connection.close()


if __name__ == '__main__':
    # 读取Excel文件
    # excel_path = r"l:\丁翊弘\高考\浙江省2021年普通高校招生普通类第一段平行投档分数线表.xls"
    excel_path = r"E:\Working Documents\装修\丁翊弘学习\高考\浙江省2018年普通高校招生普通类第一段平行投档分数线表.xls"
    df = pd.read_excel(excel_path, ) # xls文件可以不需要engine(即engine=None,然而,xlsx文件需要engine='openpyxl')
    # df = df.fillna() # 缺失值空缺

    # MySQL数据库配置
    config_path = "e:/Python_WorkSpace/config/mysql.ini"
    user, pw = config_read(config_path,section='MySQL', option1='user', option2='password')
    database = 'gaokao_stage1_score'
    config = {
        'host': 'localhost',
        'user': user,
        'password': pw,
        'database': database
    }
    # 创建MySQL表（如果需要）
    # create_table_query = f"CREATE TABLE {df.index.name} ({', '.join([f'{col} VARCHAR(255)' for col in columns])});"
    # cursor.execute(create_table_query)

    # 将数据写入MySQL表
    w2table = '2018'

    # DataFrame写入MySQL数据库的某个Table
    df_to_mysql(df=df, w2table=w2table, **config)
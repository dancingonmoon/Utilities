import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import configparser
import openpyxl
import os
from typing import List, Union

def config_read(
    config_path, section="DingTalkAPP_chatGLM", option1="Client_ID", option2=None
):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值;
    增加了config_path是否存在的检验;
    """
    # 检验config_path是否存在
    if os.path.exists(config_path) is False:
        print(f"config_path: {config_path} doesn't exist, pls double check the config_path")
        exit(111) # 结束进程,退出代码为111, (111为自定义的退出代码,可以根据需要修改)
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value


def connect_mysql(
    host="localhost", user: str = "root", password=None, database: str = None
):
    """
    连接MySQL数据库, 当错误,则返回错误值,退出程序;
    return: connection
    """
    try:
        connection = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cursor = connection.cursor()
        if database is not None:
            cursor.execute(
                f"USE {database}"
            )  # 通过USE database 是否错误,来判断是否已经存在指定的database;
            cursor.close()
        return connection
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"Database '{database}' does not exist.")
        else:
            print(err.errno)
        return err


def df_to_mysql_upon_connection(
    connection, df: pd.DataFrame, database: str, w2table: str
):
    """
    将一个DataFrame写入MySQL数据库;1)表名,列名,全部加上了反引号;2)如果出现缺失值Nan,则转换成SQL的NULL；3)如果主键出现重复值,则跳过该行数据;
    connection: 连接到MySQL数据库的connection;connection = mysql.connector.connect(**config)
    df: 待写入的DataFrame数据;
    database: 写入的数据库名;
    w2table: 写入的表名;
    """
    cursor = connection.cursor()
    rows_sum = 0
    columns = df.columns
    columns_sum = len(columns)
    for row in df.itertuples():
        field_name = ", ".join([f"`{c}`" for c in columns])  # 列名需加反引号
        field_value = []
        for i in range(1, columns_sum + 1 ): # row[0]是Index, 主键, 不需要写入 # 标记,此处处理Index方式,会出现不匹配,再查;
            c = row[i]
            if isinstance(c, str):
                field_value.append(f'"{c}"')  # 当字段值为字符串时,加上引号;
            elif pd.isna(c):
                field_value.append("NULL")  # 缺失值Nan时, 改成SQL可以识别的NULL,表示空值
            else:
                field_value.append(f"{c}")
        field_value = ", ".join(field_value)
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
                print(
                    f"total {rows_sum} rows have been written into MySQL: {database}.{w2table}"
                )
                # 关闭连接
                cursor.close()
            break

    print(f"total {rows_sum} rows have been written into MySQL {database}.{w2table}")
    # 关闭连接
    cursor.close()


def df2mysql(
    host="localhost",
    user: str = "root",
    password=None,
    database: str = None,
    df: pd.DataFrame = None,
    w2table: str = None,
):
    """
    A:新建一个MySQL的连接, 当连接错误,或者Database不存在,退出;
    B: 将一个DataFrame写入MySQL数据库;1)表名,列名,全部加上了反引号;2)如果出现缺失值Nan,则转换成SQL的NULL；3)如果主键出现重复值,则跳过该行数据;
    database: 写入的数据库名;
    w2table: 写入的表名;
    """
    connection = connect_mysql(
        host=host, user=user, password=password, database=database
    )
    if isinstance(connection, mysql.connector.Error):
        print(connection)
        return connection
    df_to_mysql_upon_connection(connection, df, database, w2table)
    # 关闭连接
    connection.close()


def mysql2df(
    host="localhost",
    user: str = "root",
    password=None,
    database: str = None,
    table: str = None,
):
    """
    从Mysql中读出指定database,指定的table,将table转化成DataFrame
    :param host:
    :param user:
    :param password:
    :return:
    """
    connection = connect_mysql(
        host=host, user=user, password=password, database=database
    )
    if isinstance(connection, mysql.connector.Error):
        print(connection)
        return connection

    # 获得table全部列名,定义DataFrame:
    describe_query = f"""
                    DESCRIBE {database}.`{table}`; 
                    """
    cursor = connection.cursor()
    try:
        cursor.execute(describe_query)
        # SELECT无需提交事务,否则commit后,cursor内容清楚
        # connection.commit()
    except mysql.connector.Error as err:
        print(err.msg)
        print(describe_query)

    # 读出每行:
    columns = [c[0] for c in cursor]
    # 定义空DataFrame:
    df = pd.DataFrame(columns=columns)

    # mysql: SELECT * FROM database.`table`
    select_query = f"""
                    SELECT * FROM {database}.`{table}`; 
                    """
    try:
        cursor.execute(select_query)
        # SELECT无需提交事务,否则commit后,cursor内容清楚
        # connection.commit()
    except mysql.connector.Error as err:
        print(err.msg)
        print(select_query)

    # 读出每行,写入DataFrame:
    row_sum = 0
    for row in cursor:
        df.loc[len(df)] = row
        row_sum += 1

    print(f"total {row_sum} read from {database}.{table}")
    # 关闭连接
    cursor.close()

    return df

def addColumn_mysqlTable(connection:mysql.connector, database: str, table: str, columns:Union[str,List[str]], datatype: str, varchar_n:int=None, null:bool=True, unsigned=False, ):
    """
    在MySQL的某个table中,添加一个column
    """
    cursor = connection.cursor()

    NULL = "NULL" if null else "NOT NULL"
    UNSIGNED = "UNSIGNED" if unsigned else ""

    sql_statement1 = f"ALTER TABLE `{database}`.`{table}`"
    sql_statement2 = ""
    if isinstance(columns, list):
        for column in columns:
            if datatype == 'str' or datatype=='string' or datatype == 'varchar':
                sql_statement2 = f"ADD COLUMN `{column}` VARCHAR({varchar_n}) {NULL};"
            elif datatype == 'int':
                sql_statement2 = f"ADD COLUMN `{column}` INT {UNSIGNED} {NULL};"

            sql_statement = ' '.join([sql_statement1, sql_statement2])
            try:
                cursor.execute(sql_statement)
                connection.commit()
            except mysql.connector.Error as err:
                print(err.msg)
                print(f"{sql_statement1}\n{sql_statement2}")

    elif isinstance(columns, str):
        if datatype == 'str' or datatype == 'string' or datatype == 'varchar':
            sql_statement2 = f"ADD COLUMN `{columns}` VARCHAR({varchar_n}) {NULL};"
        elif datatype == 'int':
            sql_statement2 = f"ADD COLUMN `{columns}` INT {UNSIGNED} {NULL};"

        sql_statement = ' '.join([sql_statement1, sql_statement2])
        try:
            cursor.execute(sql_statement)
            connection.commit()
        except mysql.connector.Error as err:
            print(err.msg)
            print(f"{sql_statement1}\n{sql_statement2}")

    else:
        print("columns must be str or list[str]")


    # 关闭连接
    cursor.close()



if __name__ == "__main__":

    # 读取Excel文件
    df_shortname = "2023"
    # excel_path_base = "E:/Working Documents/装修/丁翊弘学习/高考/浙江省{}年普通高校招生普通类第一段平行投档分数线表.{}"
    excel_path_base = "L:/丁翊弘/高考/浙江省{}年普通高校招生普通类第一段平行投档分数线表.{}"
    df_excelPath = excel_path_base.format(df_shortname, 'xlsx')
    sheet_name = "combine2023"
    df = pd.read_excel( df_excelPath, sheet_name=sheet_name )
    # df = df.fillna() # 缺失值空缺

    # 列名:
    print(df.columns)

    # MySQL数据库配置
    config_path = "l:/Python_WorkSpace/config/mysql.ini"
    user, pw = config_read(
        config_path, section="MySQL", option1="user", option2="password"
    )
    database = "gaokao_stage1_score"
    config = {"host": "localhost", "user": user, "password": pw, "database": database}
    # 获取mysql.connector:
    connector = connect_mysql(**config)
    # 创建MySQL表（如果需要）
    # create_table_query = f"CREATE TABLE {df.index.name} ({', '.join([f'{col} VARCHAR(255)' for col in df.columns])});"
    # cursor.execute(create_table_query)

    # 批量添加MySQL表中的字段:
    # columns = df.columns
    # varchar_n = 40
    # for column in columns:
    #     if "学校名称" in column or "_merge" in column:
    #         datatype="str"
    #         varchar_n = 50
    #     elif "专业名称" in column:
    #         datatype = "str"
    #         varchar_n = 128
    #     else :
    #         datatype='int'
    #         unsigned=True
    #
    #     addColumn_mysqlTable(connector,database=database, table=sheet_name,columns=column,datatype=datatype,varchar_n=varchar_n, null=True,unsigned=True)

    # 有字符串长度超出database.table.字段的定义要求,需要修改字段的varchar()长度
    # 将数据写入MySQL表
    w2table = "combine2023"

    # DataFrame写入MySQL数据库的某个Table
    df2mysql(df=df, w2table=w2table, **config)

    # # 读出指定database.table, 写入DataFrame:
    # read_table = "2023"
    # df_r = mysql2df(table=read_table, **config)

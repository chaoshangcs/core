"""
create database
"""

import os 
import sys 
import sqlite3
import base64 
import json
import config 
import time 
from config import lock 
from utils import lockit, param_equal 
import csv 
import numpy as np 
from collections import namedtuple, OrderedDict, Counter
from database_table_v2 import basic_tables, tables

class AffinityDatabase:
    """
    A simple warpper for sqlite3
    """

    def __init__(self):
        self.db_path = config.db_path
        self.tables = tables

        if not os.path.exists(os.path.dirname(self.db_path)):
            os.makedirs(os.path.dirname(self.db_path))
        
        if not os.path.exists(self.db_path):
            self.backup_and_reset_db()
        else:
            self.connect_db()

    def connect_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.connect = True
        print "connect to %s" % self.db_path

    def backup_db(self):

        backup_db_path = self.db_path.replace('.', '_'+time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime()) +'.')

        if os.path.exists(self.db_path):
            cmd = 'cp %s %s' % (self.db_path , backup_db_path)
            os.system(cmd)
            print "backup database %s" % backup_db_path

    def backup_and_reset_db(self):
        """
        backup current database and create a new one
        :return: 
        """
        if os.path.exists(self.db_path):
            backup_db_path = self.db_path.replace('.', '_' + time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime()) + '.')
            os.rename(self.db_path, backup_db_path)


        self.connect_db()
        self.init_table()

    def get_new_table_sn(self):
        stmt = 'select table_sn from db_info;'
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()
        values = map(lambda x:x[0], values)
        if len(values) == 0:
            sn = 1
        elif not len(values) == max(values):
            rest =  list(set(range(1, max(values))) - set(values))
            sn = rest[0]
        else:
            sn = max(values) + 1
    
        return sn

    def create_table(self, table_type, table_param):
        
        table_sn = self.get_new_table_sn()
        table_name = '{}_{}'.format(table_type, table_sn)
        tab = self.tables[table_type]

        encoded_param = base64.b64encode(json.dumps(table_param))
        create_time = time.strftime("%Y-%m-%d", time.gmtime())

        datum = [table_name, table_type, table_sn, create_time, encoded_param]
        data = [datum]
        self.insert('db_info',data)

        stmt = 'create table ' + table_name + ' ('        
        for key in tab.columns.keys():
            stmt += key + ' ' + tab.columns[key]
            if key in tab.primary_key:
                stmt += ' not null ,'
            else:
                stmt += ' ,'
        stmt += 'primary key(' + ','.join(tab.primary_key) + '));'
        self.conn.execute(stmt)

        if 'depend' in table_param.keys():
            depend_tables = table_param['depend']
            for tab_sn in depend_tables:
                self.insert('dependence',[[tab_sn, table_sn]])

        self.conn.commit()

        return table_sn

    def get_table_name_by_sn(self, sn):
        sn = int(sn)
        cursor = self.conn.cursor()
        stmt = 'select name from db_info where table_sn=%d;' % sn 
        cursor.execute(stmt)
        value = cursor.fetchone()
        if value is None:
            raise Exception("No table with sn number %d" %sn)
        else:
            return value[0]

    def get_table(self, sn, with_param=True):
        sn = int(sn)
        cursor = self.conn.cursor()
        stmt = 'select name, parameter from db_info where table_sn=%d;' % sn
        cursor.execute(stmt)
        value = cursor.fetchone()

        if value is None:
            raise Exception("No table with sn number {}".format(sn))
        else:
            table_name, coded_param = value 
            
            if with_param:
                param = json.loads(base64.b64decode(coded_param))
                return table_name, param
            else:
                return table_name 

    def get_folder(self, sn):
        sn = int(sn)
        table_name, table_param = self.get_table(sn)
        
        if not 'folder' in table_param.keys():
            raise Exception("table {} doesn't have corresponding folder".format(table_name))
        else:
            return table_param['folder']

    def delete_table(self, sn):
        sn = int(sn)
        # if exists get the ble_name and table_taparam
        table_name, table_param = self.get_table(sn)

        # delete all table depend on it
        cursor = self.conn.cursor()
        stmt = 'select dest from dependence where source=%d;' % sn
        cursor.execute(stmt)
        values = cursor.fetchall()
        if values:
            for val in values:
                self.delete_table(val)

        # delete from dependence
        stmt = 'delete from dependence where source=%d' % sn
        cursor.execute(stmt)

        # delete from db_info
        stmt = 'delete from db_info where table_sn=%d;' % sn
        cursor.execute(stmt)

        # drop table
        stmt = 'drop table %s' % table_name
        cursor.execute(stmt)

        self.conn.commit()
        # if have data with this table remove relative data
        if 'folder' in table_param.keys():
            folder_name = '{}_{}'.format(sn, table_param['folder'])
            del_folder_name = 'del_' + folder_name
            folder_dir = os.path.join(config.data_dir, folder_name)
            del_folder_dir = os.path.join(config.data_dir, del_folder_name)
            if os.path.exists(folder_dir):
                os.system('mv {} {} '.format(folder_dir, del_folder_dir))
                os.system('rm -r {}'.format(del_folder_dir))

        
    @lockit
    def insert(self, table_sn, values, head=None):
        self.insert_or_replace(table_sn, values, head)

    def insert_or_replace(self, table_sn, values, head=None):
        
        if table_sn in basic_tables.keys():
            table_name = table_sn
        else:
            table_sn = int(table_sn)       
            table_name = self.get_table(table_sn, with_param=False)

        db_value = lambda x:'"%s"' % x if type(x).__name__ in ['str','unicode'] else str(x)
        db_values = [ map(db_value, value) for value in values ]
        sql_values = [ '(' + ','.join(value) + ')' for value in db_values ]


        stmt = 'replace into ' + table_name + ' '
        if head is not None:
            stmt += '(' + ','.join(head) + ')'
        stmt += ' values '
        stmt += ','.join(sql_values)
        stmt += ';'

        #print stmt
        self.conn.execute(stmt)
        self.conn.commit()

    def primary_key_for(self, sn):
        sn = int(sn)
        stmt = 'select type from db_info where table_sn=%d;' % sn
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        value = cursor.fetchone()

        if value is None:
            raise Exception("No table with sn number %d" %sn)
        else:
            table_type = value[0]
            return tables[table_type].primary_key

    def get_all_success(self, sn):
        sn = int(sn)
        table_name = self.get_table(sn, with_param=False)
        primary_key = self.primary_key_for(sn)
        stmt = 'select ' + ','.join(primary_key) + ' from ' + table_name
        stmt += ' where state=1;'
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()
        return values

    def get_all_failed(self, sn):
        sn = int(sn)
        table_name = self.get_table(sn, with_param=False)
        primary_key =  self.primary_key_for(sn)
        stmt = 'select ' + ','.join(primary_key) + ' from ' + table_name
        stmt += ' where state=0;'
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()
        return values

    def get_all_sns(self):
        
        stmt = 'select table_sn from db_info;'
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()
        values = map(lambda x:x[0], values)
        return values

    def get_sns_by_type(self, table_type):
        
        stmt = 'select table_sn from db_info '
        stmt += ' where type="%s";' % table_type
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()
        values = map(lambda x:x[0], values)
        return values


    def get_success_data(self, sn, dataframe=False):
        sn = int(sn)
        table_name, table_param = self.get_table(sn, with_param=True)
        stmt = 'select * from ' + table_name 
        stmt += ' where state=1'
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()
        columns = list(map(lambda x:x[0], cursor.description))
        
        if dataframe:
            try:
                import pandas as pd 
            except:
                raise Exception("Cannot import pandas")
            df = pd.DataFrame(values, columns=columns)
            return (table_name, table_param, df)
        else:
            return (table_name, table_param, columns,  values)

    def get_failed_reason(self, sn ):
        sn = int(sn)
        table_name = self.get_table(sn, with_param=False)
        stmt = 'select comment from ' + table_name
        stmt += ' where state=0;'
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()
        reason_c = Counter(values)
        reason_n = np.asarray(reason_c.items())
        df = pd.DataFrame(reason_n, columns=['reason','count'])
        return (table_name, df)
        
        
    def get_param(self, sn):
        sn = int(sn)
        table_name, param = self.get_table(sn)
        

    def init_table(self):
        print 'init'
        for tab in basic_tables.values():
            stmt = 'create table '+ tab.type + ' ('
            for key in tab.columns.keys():
                stmt += key + ' ' + tab.columns[key]
                if key in tab.primary_key:
                    stmt += ' not null ,'
                else:
                    stmt += ' ,'
            stmt += 'primary key(' + ','.join(tab.primary_key) + '));'
            print "create ",tab.type
            print stmt
            self.conn.execute(stmt) 
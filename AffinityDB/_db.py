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
from collections import namedtuple, OrderedDict
from database_table import basic_tables, tables


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

    def next_table_sn(self):
        stmt = 'select count(*) from sqlite_master where type="table";'
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        table_num = cursor.fetchone()[0]
        sn = table_num - len(basic_tables.keys()) + 1
        return sn

    def create_table(self, table_name, table_type, parameter, dependence):
        encoded_param = base64.b64encode(json.dumps(parameter))
        tab = self.tables[table_type]

        stmt = 'create table ' + table_name + ' ('        
        for key in tab.columns.keys():
            stmt += key + ' ' + tab.columns[key]
            if key in tab.primary_key:
                stmt += ' not null ,'
            else:
                stmt += ' ,'
        stmt += 'primary key(' + ','.join(tab.primary_key) + '));'
        self.conn.execute(stmt)


        create_time = time.strftime("%Y-%m-%d", time.gmtime())

        values =[table_name, table_type, create_time, encoded_param]
        self.insert('db_info',[values])

        
        for depend_table in dependence:
            self.insert('dependence',[[depend_table, table_name]])
        
        self.conn.commit()

    def depend_source_for(self, table_name):
        
        stmt = 'select source from dependence where ' + 'dest="%s"; ' % table_name 

        cursor = self.conn.cursor()
        cursor.execute(stmt)

        source = cursor.fetchall()
        if len(source):
            source = map(lambda x:x[0], source)
        return list(source)

    def depend_dest_for(self, table_name):
        
        stmt = 'select dest from dependence where' + 'source="%s";' % table_name 

        cursor = self.conn.cursor()
        cursor.execute(stmt)

        dest = cursor.fetchall()
        if len(dest):
            dest = map(lambda x:x[0], dest)
        return list(dest)

    def table_type(self, table_name):
        stmt = 'select type from db_info where name="%s";' % table_name
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        value = cursor.fetchone()
        if value is None:
            raise Exception("table %s doesn't exists" % table_name)
        return value[0]

    def get_depend_table(self, table_name, table_type):
        
        depend_tables = self.depend_source_for(table_name)
        for tab in depend_tables:
            if self.table_type(tab) == table_type:
                return tab 
        raise Exception("Cannot find depend table of type {} for table {}" %(table_type, table_name))



    def get_table(self, table_type, dependence, parameter):

        if not table_type in self.tables.keys():
            raise Exception("Cannot create table of type {}\n".format(table_type),
                        + "available table type {}\n".format(list(self.tables.keys())))
        

        stmt = 'select name, parameter from db_info where type="%s"' % table_type
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()

        dest_table = None
        if len(values) > 0:
            for name, param in values:
                decoded_param = base64.b64decode(param)
                parsed_param = json.loads(decoded_param)
                print param_equal(parameter, parsed_param)

                if param_equal(parameter,parsed_param):
                    dest_table = name
                    break

        if dest_table is not None:
            depend_list = self.depend_source_for(dest_table)
            print depend_list 
            print dependence
            if not sorted(depend_list) == sorted(dependence):
                dest_table = None
        
        if dest_table is None:
            table_sn = self.next_table_sn()
            table_name = '{}_{}'.format(table_type,table_sn)
            self.create_table(table_name, table_type, parameter, dependence)
            dest_table = table_name
        
        return dest_table

    def insert(self, table_name, values, head=None):
        self.insert_or_replace(table_name, values, head)
            
    def insert_or_replace(self, table_name, values, head=None):
        
        db_value = lambda x:'"%s"' % x if type(x).__name__ in ['str','unicode'] else str(x)
        db_values = [ map(db_value, value) for value in values ]
        sql_values = [ '(' + ','.join(value) + ')' for value in db_values ]

        stmt = 'replace into ' + table_name + ' '
        if head is not None:
            stmt += '(' + ','.join(head) + ')'
        stmt += ' values '
        stmt += ','.join(sql_values)
        stmt += ';'

        print stmt
        self.conn.execute(stmt)
        self.conn.commit()

    def primary_key_for(self, table_name):
        
        stmt = 'select type from db_info where name="%s";' % table_name
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        value = cursor.fetchone()[0]

        return tables[value].primary_key

    def get_all_success(self, table_name):

        primary_key = self.primary_key_for(table_name)
        stmt = 'select ' + ','.join(primary_key) + ' from ' + table_name
        stmt += ' where state=1;'
        cursor = self.conn.cursor()
        cursor.execute(stmt)
        values = cursor.fetchall()
        return values

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
            self.conn.execute(stmt)
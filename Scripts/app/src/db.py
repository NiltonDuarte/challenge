#db.py
import mysql.connector

dbConn = None

def getDB():
  config = {
    'user': 'root',
    'password': 'root',
    'host': 'mysqldb',
    'port': '3306',
    'database': 'challenge'
  }
  global dbConn
  if dbConn == None:
    dbConn = mysql.connector.connect(**config, buffered = True)
  return dbConn

def closeDB(e=None):
  if dbConn is not None:
    dbConn.close()

#this is nowhere a safe method. do not ever, never copy it.
def getData(tableName, columns=['*']):
  print("db get data columns", columns, flush=True)
  if columns == None:
    return
  conn = getDB()
  cursor = conn.cursor(dictionary=True)
  query = 'SELECT {} FROM {}'.format(','.join(columns), tableName)
  print(query)
  cursor.execute(query)
  results = cursor.fetchall()
  cursor.close()
  return results

#this is nowhere a safe method. do not ever, never copy it.
def insertQueryDB(query):
  print("query db: ", query, flush=True)
  if query == None:
    return
  if query == '':
    return
  try:
    conn = getDB()
    cursor = conn.cursor(dictionary=True)
    result = cursor.execute(query)
    conn.commit()
    print ("Record inserted")
  except mysql.connector.Error as error :
    connection.rollback() #rollback if any exception occured
    print("Failed inserting record. Eroor: {}".format(error))
  finally:
    cursor.close()
  return result

  #this is nowhere a safe method. do not ever, never copy it.
def selectQueryDB(query):
  #print("query db: ", query, flush=True)
  if query == None:
    return
  if query == '':
    return
  try:
    conn = getDB()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    #print ("Record selected")
    results = cursor.fetchall()
  except mysql.connector.Error as error :
    print("Failed query. Error: {}".format(error))
    return
  finally:
    cursor.close()
  return results
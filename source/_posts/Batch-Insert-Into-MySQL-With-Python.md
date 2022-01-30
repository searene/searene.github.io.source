title: Batch Insert Into MySQL With Python
date: 2015-12-07 19:50:29
categories: Coding
tags: [mysql, python, mysqldb]
thumbnail: https://upload.wikimedia.org/wikipedia/en/thumb/6/62/MySQL.svg/1280px-MySQL.svg.png
---

In Python, you can use `MySQLdb`'s `executemany` to insert multiple records into MySQL at once. First, let's install `MySQLdb`. The command used to install it depends on your OS:

1. **easy_install mysql-python** (mix os)
2. **pip install mysql-python** (mix os/ python 2)
3. **pip install mysqlclient** (mix os/ python 3)
4. **apt-get install python-mysqldb** (Linux Ubuntu, ...)
5. **cd /usr/ports/databases/py-MySQLdb && make install clean** (FreeBSD)
6. **yum install MySQL-python** (Linux Fedora, CentOS ...)

(Source: [Stackoverflow](https://stackoverflow.com/a/5873259/1031769))

Then use `executemany` to insert multiple records at once.

```python
import MySQLdb
db=MySQLdb.connect(user="searene",passwd="123",db="test")
c=db.cursor()
c.executemany(
      """INSERT INTO breakfast (name, spam, eggs, sausage, price)
      VALUES (%s, %s, %s, %s, %s)""",
      [
      ("Spam and Sausage Lover's Plate", 5, 1, 8, 7.95 ),
      ("Not So Much Spam Plate", 3, 2, 0, 3.95 ),
      ("Don't Wany ANY SPAM! Plate", 0, 4, 3, 5.95 )
      ] )
db.commit()

# close db if you don't need to execute other SQLs.
db.close()
```

(Source: [MySQLdb documentation](http://mysql-python.sourceforge.net/MySQLdb.html))
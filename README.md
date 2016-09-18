# Galvanize Reference #

This is designed to be a catch-all reference tool for my time in Galvanize's Data Science Immersive program.  Others might find it of use as well.

---

## Python ##

#### Data Types ####

1. `string:` immutable
2. `tuple:` immutable
3. `int:` immutable
4. `float:` immutable
1. `list:` mutable
6. `dict:` mutable, a series of hashable key/value pairs.
5. `set:` mutable

string = string['abc']; str[-1] # note indexing is cardinal, not oridinal like R
tuple = (5, 3, 8); tuple[3]
  tuple2 = (tuple, 2, 14) # nested tuple
list = [5, 5, 7, 1]; list[2]
  list2 = [list, 23, list] # nested list
dict = {‘key’: 'value', 'first': 1};
  dict.keys(); dict.values();
  dict['newkey'] = 'value'
‘abc’[-1] # last item
‘abc’[0:2] # slicing a sub-string

#### Subsetting ####

#### Built-in Functions ####

#### List Comprehension ####

#### Lambda Functions ####

#### Classes ####

@property # decorator
Magic methods: http://www.rafekettler.com/magicmethods.html

#### Testing and Debugging ####
1. Use test_[file to test].py
2. Use `from unittest import TestCase`

add this in line you're examining: import pdb; pdb.set_trace()
nosetests

#### A Note on Style ####

Classes are capitalized LikeThis (camel case) and functions like_this (snake case).
https://www.python.org/dev/peps/pep-0008/

Use this block to run code if the .py doc is called directly but not if imported as a module:
  if __name__ == ‘__main__’:
    [Code to run if called directly here]

---

### Python Packages - pandas ###

Data munging, python equivalent of dplyr.  

Objects:
1. `series:`

---

### Python Packages - numpy ###

Linear algebra package

---

### Python Packages - scipy ###

---

### Python Packages - itertools ###

Combinatoric generators

### Python Packages - matbplotlib ###

import matplotlib.pyplot as plt

---

### Python Packages - sklearn/statsmodels ###

---

## SQL ##

Access SQL through `sqlite3 [.sql file]` at the command line.
Also accessible through postgres.


There are a few types of joins:

1. `INNER JOIN:` joins based on rows that appear in both tables.  This is the default for saying simply JOIN
** SELECT * FROM TableA INNER JOIN TableB ON TableA.name = TableB.name
** SELECT c.id, v.created at FROM customers as c, visits as v WEHRE c.id = v.customer_id # joins w/o specifying it's a join
2. `LEFT OUTER JOIN:` joins based on all rows on the left table even if there are no values on the right table
3. `RIGHT OUTER JOIN:` joins based on all rows on the left table even if there are no values on the right table
4. `FULL OUTER JOIN:` joins all rows from both tables even if there are some in either the left or right tables that don't match

Resources:
1. http://sqlzoo.net/wiki/SELECT_basics
2. An illustrated explanation of joins: https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/

---

## Git ##

---

## Command Line ##

* `ls`: list files in current directory
* `cd directory`: change directories to directory
* `cd ..`: navigate up one directory
* `mkdir new-dir`: create a directory called new-dir
* `rm some-file`: remove some-file
* `man some-cmd`: pull up the manual for some-cmd
* `pwd`: find the path of the current directory
* `mv path/to/file new/path/to/file`: move a file or directory (also used for renaming)
* `find . -name blah`: find files in the current directory (and children) that have blah in their name
* To jump to beginning of line: __CTRL__ + __a__
* To jump to end of line: __CTRL__ + __e__
* To cycle through previous commands: __UP ARROW__ / __DOWN ARROW__
* `python -i oop.py`: opens python and imports oop.py
* `which python`: Tells you the path the's executed with the command python

Ctr-Z puts current process to the background.  
fg brings the suspended process to the foreground

---

## Linear Algebra ##

---

## Note on Style and Other Tools ##


jupyter
iPython
Markdown: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
Atom: command / hashes out text

Sniffer which wraps nose tests so every time you save a file it will automatically run the tests - pip install sniffer nose-timer

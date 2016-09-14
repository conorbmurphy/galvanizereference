# Galvanize Reference #

This is designed to be a catch-all reference tool for my time in Galvanize's Data Science Immersive program.  Others might find it of use as well.

---

## Python ##

#### Data Types ####

2. `string:` immutable
2. `tuple:` immutable
3. `int:` immutable
4. `float:` immutable
1. `list:` mutable
6. `dict:` mutable
5. `set:` mutable

Tuple = (5,)
List = [5]
list[0] # Python indexes at 0, R indexes at 1
dict = {‘key’: 'value', 'first': 1}; dict.keys(); dict['newkey'] = 'value'
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

### Python Packages - numpy ###

---

### Python Packages - scipy ###

---

### Python Packages - numpy ###

---

### Python Packages - itertools ###

Combinatoric generators

---

## SQL ##

Access SQL through `sqlite3 [.sql file]` at the command line.


There are a few types of joins:

1. `inner join:` joins based on rows that appear in both tables
** SELECT * FROM TableA INNER JOIN TableB ON TableA.name = TableB.name
2. `left outer join:` joins based on all rows on the left table even if there are no values on the right table
3. `full outer join:` joins all rows from both tables even if there are some in either the left or right tables that don't match

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

---

## Linear Algebra ##

---

## Note on Style and Other Tools ##


jupyter
iPython
Markdown: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
Atom: command / hashes out text

Sniffer which wraps nose tests so every time you save a file it will automatically run the tests - pip install sniffer nose-timer

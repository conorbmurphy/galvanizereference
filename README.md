# Galvanize Reference #

This is designed to be a catch-all reference tool for my time in Galvanize's Data Science Immersive program.  Others might find it of use as well.

---

## Python ##

There are many different programming paradigms such as declarative, functional, and procedural.  Object-oriented programming (OOP) languages (and Python in particular) allows for much of the functionality of these other paradigms.  One of the values of OOP is encapsulation, which makes working in teams easy.  Objects are a combination of state and behavior.  The benefits of OOP include:

* Readability
* Complexity management (scoping)
* Testability
* Isolated modifications
* DRY code (don't repeat yourself)
* Confidence

OOP has classes and objects.  A class is a combination of state and behaviors.  Python has a logic where everybody can access anything, making it difficult to obfuscate code.  This makes Python particularly adept for the open source community.

#### Data Types ####

* `string:` immutable
* `tuple:` immutable
* `int:` immutable
* `float:` immutable
* `list:` mutable, uses append
* `dict:` mutable, uses append, a series of hashable key/value pairs.
* `set:` mutable, uses append, also uses hashing.  Sets are like dict's without values (similar to a mathematical set).

string = string['abc']; str[-1] # note indexing is cardinal, not oridinal like R
tuple = (5, 3, 8); tuple[3]
  tuple2 = (tuple, 2, 14) # nested tuple
list = [5, 5, 7, 1]; list[2]
  list2 = [list, 23, list] # nested list
dict = {‘key’: 'value', 'first': 1};
  dict['first'];
  dict.keys(); dict.values();
  dict['newkey'] = 'value'

Hashing allows us to retrieve information much more quickly because we can go directly to a value, similar to a card in a card catalogue.  

#### Importing Data ####

* `open`:
* `read`:

#### Built-in Functions ####

* `abs()`:
* `all()`:
* `any()`:
* `basestring()`:
* `bin()`:
* `bool()`:
* `bytearray()`:
* `callable()`:
* `chr()`:
* `classmethod()`:
* `cmp()`:
* `compile()`:
* `complex()`:
* `delattr()`:
* `dict()`:
* `dir()`:
* `divmod()`:
* `enumerate()`:
* `eval()`:
* `execfile()`:
* `file()`:
* `filter()`:
* `float()`:
* `format()`:
* `frozenset()`:
* `getattr()`:
* `globals()`:
* `hasattr()`:
* `hash()`:
* `help()`:
* `hex()`:
* `id()`:
* `input()`:
* `int()`:
* `isinstance()`:
* `issubclass()`:
* `iter()`:
* `len()`:
* `list()`:
* `locals()`:
* `long()`:  
* `map()`:  
* `max()`:  
* `memoryview()`:
* `min()`:
* `next()`:
* `object()`:
* `oct()`:
* `open()`:
* `ord()`:
* `pow()`:
* `print()`:
* `property()`:
* `range()`:
* `raw_input()`:
* `reduce()`:
* `reload()`:
* `repr()`:
* `reversed()`:
* `round()`:
* `set()`:
* `setattr()`:
* `slice()`:
* `sorted()`:
* `staticmethod()`:
* `str()`:
* `sum()`:
* `super()`:
* `tuple()`:
* `type()`:
* `unichr()`:
* `unicode()`:
* `vars()`:
* `xrange()`:
* `zip()`:
* `__import__()`:

#### Classes ####

To make a class:

class MapPoint(object): # This is extending from the object class (inheratance)
     def __init__(self):
          self.x = None
          self.y = None
     def lon(self): # This is a method, or a function w/in a class
          return self.x

Magic methods allow you to define things like printing and comparing values:

def __cmp__(self, other):
     if self._fullness < other._fullness:
          return -1 # This is a convention
     elif self._fullness > other._fullness:
          return 1
     elif self._fullness == other._fullness:
          return 0
     else:
          raise ValueError(“Couldn’t compare {} to {}”.format(self, other))

A decorator is a way to say that you're going to access some attribute of a class as though it's not a function.  This is not a function; it's encapsulated.
@property
def full(self):
     return self._fullness == self._capacity

@property
def empty(self):
     return self._fullness == 0

Basic vocabulary regarding classes:

* `encapsulation` is the idea that you don’t need to worry about the specifics of how a given thing is working.  
* `composition` is the idea that things can be related (fruit to backpacks, for instance).  
* `Inheritance` is where children take the quality of their parents
* `Polymorphism` is where you customize a given behavior for a child

Reference: http://www.rafekettler.com/magicmethods.html

#### List Comprehension ####

#### Lambda Functions ####

Lambda functions are for defining functions without a name:
     lambda x, y : x + y
     map(lambda x: x**2)


#### Testing and Debugging ####
* add this in line you're examining: import pdb; pdb.set_trace()
* Use test_[file to test].py when labeling test files
* Use `from unittest import TestCase`


nosetests

from unittest import TestCase

class OOPTests(TestCase):

     def test_backpack(self):
          pack = Backpack()
          x = 1
          pack.throw_in(x)
          self.assertIn(1, pack._items)
You need assertions in unittest in order to test things.

#### A Note on Style ####

Classes are capitalized LikeThis (camel case) and functions like_this (snake case).
https://www.python.org/dev/peps/pep-0008/

DRY: don’t repeat yourself.  WET: we enjoy typing.

pep8 help # at the terminal, this will help you evaluate your code for pep8

Use this block to run code if the .py doc is called directly but not if imported as a module:
  if __name__ == ‘__main__’:
    [Code to run if called directly here]

There’s a package called flake8 which combines pep8 (the style guide) with flake (with code introspection, eg flagging calling a variable you didn’t define).  You should be able to install this for Atom

#### Common Errors ####

Here are some common errors to avoid:

* `AttributeError`: Thrown when you call an attribute an object doesn't have
* `ZeroError`: Thrown when dividing by a zero

---

### iPython ###



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

Git is the version control software.  Github builds a UI around git.  A repo is a selection of files (usually code, not data—`data on github is not a preferred practice`).  You can fork a repo, so it becomes your copy of the repo.  Then you can clone that repo you’re making a copy of it on your local file system.  This puts it on your computer.  You now have three copies: the main repo, your repo, and your clone on your computer.  You should commit the smallest changes to the code.

* git add [file name] # add new files; can also use '.' instead of spcific file name
* git commit -m “message” # you have to add a message for a git commit  If you forget the message, it will open vim so type “:q” to get out of it.  This adds a commit to the code on the local file system.  You can go back if need be.
* git commit -am “message” # commits all new files with the same message
* git push # This pushes all commits to your repo on the cloud.  These won’t make it back to the original repo
* git clone [url] # clones repo to your machine
* git status # check the commit status
* git log # shows the log status of

Adding is building to a commit.  Ideally a commit would be a whole new feature/version.  `It’s only with commands that have the word ‘force’ that you risk losing data.`  When you make a pull request, it means you have a new version of a repo and you want these to be a part of the original repo.

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
Markdown: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
Atom: command / hashes out text

Sniffer which wraps nose tests so every time you save a file it will automatically run the tests - pip install sniffer nose-timer

---

## Interview Questions Topics ##

Runtime analysis is a popular interview topic: do you understand the concequences of what you’ve written?  Use generators and dicts when possible.
You need to know the basics of OOP in interviews.  A common interview question is how to design a game: start with designing classes.

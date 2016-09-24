# Galvanize Reference #

This is designed to be a catch-all reference tool for my time in Galvanize's Data Science Immersive program.  Others might find it of use as well.

The main responsibilities of a data scientist are:

1. ideation (experimental design)
2. importing (SQL/postgres/psycopg2)
 * defining the ideal dataset
 * understanding your data compares to the ideal
4. exploratory analysis (python/pandas)
 * summaries
 * missing data
 * exploratory plots
 * exploratory analyses
4. data munging (pandas)
5. feature engineering
6. modeling
7. presentation

In practice, data scientists spend a good majority of their time in SQL.

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

* `string`: immutable
  * string = string['abc']; str[-1] # note indexing is cardinal, not oridinal like R
* `tuple`: immutable
  * tuple = (5, 3, 8); tuple[3]
  * tuple2 = (tuple, 2, 14) # nested tuple
* `int`: immutable
* `float`: immutable
* `list`: mutable, uses append
  * list = [5, 5, 7, 1]; list[2]
  * list2 = [list, 23, list] # nested list
* `dict`: mutable, uses append, a series of hashable key/value pairs.
  * dict = {‘key’: 'value', 'first': 1};
  * dict['first'];
  * dict.keys(); dict.values();
  * dict['newkey'] = 'value'
* `set`: mutable, uses append, also uses hashing.  Sets are like dict's without values (similar to a mathematical set).




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

Pandas is a library for data manipulation and analysis, the python equivalent of R's dplyr.  Pandas objects are based on numpy arrays so **vectorized options (e.g. apply) over iterative ones offer large performance increases.**

Objects:

* `series`: `prices = pd.Series([1, 2, 3, 4, 5], index = ['apple', 'pear', 'banana', 'mango', 'jackfruit'])``
 * `prices.iloc[1:3]`: works on position, returns rows 1 stopping 1 before 3
 * `prices.loc['pear']`: works on index, returns row 'pear'
 * `prices.loc['pear':]`: Returns pear on
 * `prices.ix[1:4]`: Similar but returns index and value
 * `prices['inventory' > 20]`: subsets based on a boolean string
* `dataframe`: `inventory = pd.DataFrame({'price': [1, 2, 3, 4, 5], 'inventory': prices.index})`
 * `inventory.T`: Transpose of inventory
 * `inventory.price`: Returns a series of that column
 * `inventory.drop(['pear'], axis = 0)`: deletes row 'pear'
 * `inventory.drop(['price'], axis = 1, inplace = True)`: deletes column 'price' and applies to current df

Common functions:

* `prices.mean()`
* `prices.std()`
* `prices.median()`
* `prices.describe()`
* `pd.concat(frames)`: Concats a list of objects
* `pd.merge(self, right, on = 'key')`: joins df's.  Can specify left_on, right_on if column names are different.  You can also specify how for inner or outer.

**Split-apply-combine** is a strategy for working on groups of data where you split the data based upon a given characteristic, apply a function, and combine it into a new object.

      `inventory = pd.DataFrame({'price': [1, 2, 3, 4, 5], 'inventory': prices.index})`
      `inventory['abovemean'] = inventory.price > 2.5`

* `grouped = inventory.groupby('abovemean')`: creates groupby object
* `grouped.aggregate(np.sum)`: aggregates the sum of those above the mean versus below the mean
* `grouped.transform(lambda x: x - x.mean())`: tranforms groups by lambda equation
* `grouped.filter(lambda x: len(x)>2)`: filters by lambda function
* `grouped.apply(sum)`: applies sum to each column by `abovemean`

---

### Python Packages - numpy ###

Linear algebra package

---

### Python Packages - scipy ###

loc = mean
---

### Python Packages - itertools ###

Combinatoric generators

### Python Packages - matbplotlib ###

`import matplotlib.pyplot as plt`

Matplotlib is the defacto choice for plotting in python.  There's also Plotly, Bokeh, Seaborne, Pandas, and ggplot (port of R package).  Seaborne and Pandas were both built on matplotlib.

There are three levels it can be accesed on:

1. plt - minimal, fast interface
2. OO interface w/ pyplot - fine-grained control over figure, axes, etc
3. pure OO interface - embed plots in GUI applications (will probably never use)

      plt.figure
      x_data = np.arange(0, 4, .011)
      y_data = np.sin(x_data)
      plt.plot(x_data, y_data)
      plt.show()

The objects involved are the figure and axes.  We can call individual axes, but normally we deal with them together.  The figure defines the area on which we can draw.  The axes is how we plot specific data.

add lines
multiple plots

      fig, ax_list = plt.subplots(4, 2)
      for ax, flips in zip(ax_list.flatten(), value):
        x_value = [data changed by value]
        y_value = [data changed by value]
        ax.plot(x_value, y_value)


---

### Python Packages - sklearn/statsmodels ###

---

## SQL ##

SQL (Structured Query Language) is a special-purpose programming language designed for managing data held in relational database management systems (RDBMS).  In terms of market share, the following database engines are most popular: Oracle, MySQL, Microsoft SQL Server, and PostgreSQL (or simply Postgres, the open source counterpart to the other, more popular proprietary products).  While Postgres is the database system we will be working with, psql is the command-line program which can be used to enter SQL queries directly or execute them from a file.

Steps to setting up a local database using psql (you can also access SQL using sqlite3 [filename.sql]):

1. Open Postgres
2. Type `psql` at the command line
3. Type `CREATE DATABASE [name];`
4. Quit psql (`\q`)
5. Navigate to where the .sql file is stored
6. Type `psql [name] < [filename.sql]`
7. Type `psql readychef`

Basic commands:

* `\d`: returns the name of all tables in the database
* `\d [table]`: returns table schema
* `;`: excecutes a query
* `\help` or `\?`: help
* `\q`: quit

**To understand SQL, you must understand SELECT statements.**  All SQL queries have three main ingredients:

1. `SELECT`: What data do you want?
2. `FROM`: Where do you want that data from?
3. `WHERE`: Under what conditions?

The order of evaluation of a SQL SELECT statement is as follows:

1. `FROM + JOIN`: first the product of all tables is formed
2. `WHERE`: the where clause filters rows that do not meet the search condition
3. `GROUP BY + (COUNT, SUM, etc)`: the rows are grouped using the columns in the group by clause and the aggregation functions are applied on the grouping
4. `HAVING`: like the WHERE clause, but can be applied after aggregation
5. `SELECT`: the targeted list of columns are evaluated and returned
6. `DISTINCT`: duplicate rows are eliminated
7. `ORDER BY`: the resulting rows are sorted

Here are some common commands on SELECT statements:

* `*`
* `COUNT`
* `MAX or MIN`
* `DISTINCT`
* `SUM`

Other:

* `GROUP BY`: subsets by this value
* `ORDER BY [value] DESC`: orders result
* `HAVING`:
* `CASE WHEN gender = 'F' THEN 'female' ELSE 'male' END AS gender_r`: this is SQL's if/else construction.  This is good for dealing with null values

Joins are used to query across multiple tables using foreign keys.  **Every join has two segments: the tables to join and the columns to match.**  There are three types of joins that should be imagined as a venn diagram:

1. `INNER JOIN:` joins based on rows that appear in both tables.  This is the default for saying simply JOIN and would bee the center portion of the venn diagram.
** SELECT * FROM TableA INNER JOIN TableB ON TableA.name = TableB.name
** SELECT c.id, v.created at FROM customers as c, visits as v WEHRE c.id = v.customer_id # joins w/o specifying it's a join
2. `LEFT OUTER JOIN:` joins based on all rows on the left table even if there are no values on the right table.  A right join is possible too, but is only the inverse of a left join.  This would bee the left two sections of a venn diagram.
3. `FULL OUTER JOIN:` joins all rows from both tables even if there are some in either the left or right tables that don't match.  This would be all three sections of a venn diagram.

Resources:
* http://sqlzoo.net/wiki/SELECT_basics
* An illustrated explanation of joins: https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/
* Databases by market share: http://db-engines.com/en/ranking
* SQL practice: https://pgexercises.com/

### SQL using Pandas and Psycopg2 ###

You can make connections to Postgres databases using psycopg2 including creating cursors for SQL queries, commit SQL actions, and close the cursor and connection.  **Commits are all or nothing: if they fail, no changes will be made to the database.**  You can set `autocommit = True` to automatticaly commit after each query.  A cursor points at the resulting output of a query and can only read each observation once.  If you want to see a previously read observation, you must rerun the query.

**Beware of SQL injection where you add code.**  Use `%s` where possible, which will be examined by psychopg2 to make sure you're not injecting anything.  You can use ORM's like ruby on rails or the python equivalent jengo to take care of SQL injection too.


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

## Probability ##

Probability theory is the study of uncertainty.  There are two general camps: frequentist and Bayesian.

### Set Operations ###

A set is a range of all possible outcomes or events, also called the sampling space and can be discrete or continuous.  It is useful to think of set operations in the context of venn diagram.  Union is all of a venn diagram of A and B, intersection is the center portion, difference is the A portion and complement is the B portion.

* `Union`: $ A \cup B = \{x: x \in A \vee x\in B\}$ - The union of sets A and B is x such that x is in A or x is in B.
* `Intersection`: $ A \cap B = \{x: x \in A \wedge x\in B\}$ - The intersection is x such that x is in A and in B.
* `Difference`: $ A \setminus B = \{x: x \in A \wedge x \notin B\}$ - The difference is x such that x is in A and not in B.
* `Complement`: $ A^C = \{x: x\notin A\} $ - Complement is x such that x is in A and in B.
* `Null (empty) set`: $ \emptyset $

DeMorgan's Law converts and's to or's.  The tetris-looking symbol is for 'not.'

* $ \neg (A \vee B) \iff \neg A \wedge \neg B $
* $ \neg (A \wedge B) \iff \neg A \vee \neg B $

### Key Definitions ###

|                    | Population  | Sample   |
|--------------------|:-----------:|---------:|
| size               | N           | n        |
| mean               | mu          | xbar     |
| variance           | sigma**2    | s**2     |
| standard deviation | sigma       | s        |
| proportion         | $\pi$  π     | p ("little pea") |


* `S`: sample space, or a range of all possible outcomes (discrete or continuous)
* `s`:
* `X`:
* `x`:

* `i.i.d.`: independent, identically distributed (refers to when draws from X are not dependent on previous draws and fall into the same distribution)
* `\alpha`: threshold for rejecting a hypothesis
* `\beta`:
* `lambda`:

Hat means it refers to the sample, not the population

Maximum Likelihood Estimation (MLE) chooses the parameter(s) that maximize the likelihood of observing our given sample.

|                     | H<sub>0</sub> is true    | H<sub>0</sub> is false  |
| ------------------- |:-------------:| -----:|
| Fail to reject H<sub>0</sub>   | correctly accept | Type II error/beta |
| Reject H<sub>0</sub>           | Type I error/alpha      |   correctly reject* |

* This is 1-beta or pi **this is the domain of power**

### Frequentist Statistics ###

In frequentist statistics, there are four standard methods of estimation and sampling.

#### Method of Moments (MOM) ####

MOM has three main steps:

1. Assume the underlying distribution (e.g. Poisson, Gamma, Exponential)
2. Compute the relevant sample moments (e.g. mean, variance)
3. Plug those sample moments into the PMF/PDF of the assumed distribution

There are four main moments, each raised to a different power:

1. `Mean/Expected value`: the central tendency of a distribution or random variable
2. `Variance`: the expectation of the squared deviation of a random variable from its mean
3. `Skewness`: a measure of asymmetry of a probability distribution about its mean.  Since it’s to the 3rd power, we care about whether it’s positive or negative.  
4. `Kurtosis`: a measure of the "tailedness" of the probability distribution

Example: your visitor log shows the following number of visits for each of the last seven days: [6, 4, 7, 4, 9, 3, 5].  What's the probability of have ing zero visits tomorrow?

      lambda = np.mean([6, 4, 7, 4, 9, 3, 5])
      scs.poisson(0, lambda)

#### Maximum Likelihood Estimation (MLE) ####

MLE has three main steps:

1. Assume the underlying distribution (e.g. Poisson, Gamma, Exponential)
2. Define the likelihood function
3. Choose the parameter set that maximizes the likelihood function

FILL THIS IN W/ THE LECTURE RECORDING!!

#### Maximum a Posteriori (MAP) ####

This is a Bayesian method that I mention here because it is the opposite of MLE in that it looks at your parameters given that you have a certain dataset.  MAP is proportionate to MLE with information on what you thought going into the analysis.  More on this under the Bayesian section.

#### Kernel Density Estimation (KDE) ####

Using **nonparametric** techniques allows us to model data that does not follow a known distribution.  KDE is a nonparametric technique that allows you to estimate the PDF of a random variable, making a histogram by summing kernel functions using curves instead of boxes.  In plotting with this method, there is a bias verses variance trade-off so choosing the best representation of the data is relatively subjective.  

#### Common distributions? ####

### Bayesian Statistics ###

$$
P(B|A) = \frac{P(A|B)P(B)}{P(A)}
$$

### Combinatorics ###

Combinatorics is the mathematics of ordering and choosing sets.  

Factorial
Combinations
Permutations

---

## Statistics ##


### Power ###

Power visualized: http://rpsychologist.com/d3/NHST/

---

## Note on Style and Other Tools ##


jupyter notebook or Apache Zeppelin
Markdown: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
Atom: command / hashes out text
For not having to specify integer division: from __future__ import division

Sniffer which wraps nose tests so every time you save a file it will automatically run the tests - pip install sniffer nose-timer

Visualization:
* Tableau
* D3.js - based in javascript
* Shiny

---

## Career Pointers ##

Common Interview Questions:

* Narrative: why do you want to be a data scientist?
* Runtime analysis:
  * do you understand the concequences of what you’ve written?  Use generators and dicts when possible.
  * Indexes can speed up queries (btree is common, allowing you to subdivide values.)
  * Vectorized options are more effective than iterative ones
* You need to know the basics of OOP in interviews.  A common interview question is how to design a game: start with designing classes.
* SQL will be addressed on all interviews
  * What is the difference between WHERE and HAVING?  (HAVING is like WHERE but can be applied after an aggregation)
  * Types of joins
* Confounding factors in experimental design
* Basic probability

O'Reilly (Including salary averages): https://www.oreilly.com

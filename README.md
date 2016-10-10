# Galvanize Reference #

## Introduction ##

This is designed to be a catch-all reference tool for my notes while in Galvanize's Data Science Immersive program.  My goal is to have a living  document that I can update with tips, tricks, and additional resources as I progress in my data science-ing.  Others might find it of use as well.

The main responsibilities of a data scientist are:

1. `Ideation` (experimental design)
2. `Importing` (SQL/postgres/psycopg2)
 * defining the ideal dataset
 * understanding your data compares to the ideal
4. `Exploratory data analysis (EDA)` (python/pandas)
4. `Data munging` (pandas)
5. `Feature engineering`
6. `Modeling` (sklearn)
7. `Presentation`

While these represent the core competencies of a data scientist, the method for implementing them is best served by the Cross-Industry Standard Process for Data Mining (CRISP-DM), pictured below.

![Image of CRISP-DM](https://github.com/conorbmurphy/galvanizereference/blob/master/images/CRISP.png)

This system helps refocus our process on business understanding and business needs.  Always ask what your ideal data set is before moving into the data understanding stage, then approach the data you do have in that light.  Always, always, always focus on business solutions.  

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

### Base Data Types ###

Python has a few base datatypes whose different characteristics can be leveraged for faster, drier code.  Data types based on hashing, such as dictionaries, allow us to retrieve information much more quickly because we can go directly to a value, similar to a card in a card catalogue.  

* `string`: immutable
  * string = string['abc']; str[-1] # note indexing is cardinal, not ordinal like R
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

### Built-in Functions ###

The following are the base functions Python offers, some more useful than others.  Knowing these with some fluency makes dealing with more complex programs easier.

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

Note the difference between functions like `range` and `xrange` above.  `range` will create a list at the point of instantiation and save it to memory.  `xrange`, by contrast, will generate a new value each time it's called upon.  **Generators** like this (`zip` versus `izip` from itertools is another common example) are especially powerful in long for loops.


### Classes ###

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
          __

A decorator is a way to say that you're going to access some attribute of a class as though it's not a function.  This is not a function; it's encapsulated.

      @property
      def full(self):
           return self._fullness == self._capacity

      @property
      def empty(self):
           return self._fullness == 0
      __

Basic vocabulary regarding classes:

* `encapsulation` is the idea that you don’t need to worry about the specifics of how a given thing is working.  
* `composition` is the idea that things can be related (fruit to backpacks, for instance).  
* `Inheritance` is where children take the quality of their parents
* `Polymorphism` is where you customize a given behavior for a child

Reference: http://www.rafekettler.com/magicmethods.html

#### Loops and List Comprehension ####

#### Lambda Functions ####

Lambda functions are for defining functions without a name:
     lambda x, y : x + y
     map(lambda x: x**2)


#### Testing and Debugging ####
* add this in line you're examining: import pdb; pdb.set_trace()
** Resource: http://frid.github.io/blog/2014/06/05/python-ipdb-cheatsheet/
* Use test_[file to test].py when labeling test files
* Use `from unittest import TestCase`


nosetests

`from unittest import TestCase`

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

Working with datetime objects, you can do df.dt. and then hit tab to see the different options.  Similarly, df.str can give us all of our string functions.  

`crosstab` is similar to table in R

---

### Python Packages - numpy ###

Linear algebra package
In calculating `np.std()`, be sure to specify `ddof = 1` when refering to the sample.

`np.concatenate((foo, bar), axis = 1)` # adds a column
`np.hstack((foo, bar))` # adds a column
`np.vstack((foo, bar))` # adds a row
`foo.min(axis = 0)` # takes column mins

---

### Python Packages - scipy ###

loc = mean
scale = standard deviation
---

### Python Packages - itertools ###

Combinatoric generators

### Python Packages for Plotting - matbplotlib and seaborn ###

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

A useful plot from seaborn is heatmap and violin plot (which is the kde mirrored):

      agg = cars.groupby(['origin'], cars['year'])
      ax = sns.heatmap(agg.unstack(level = 'year'), annot = True) # Be sure to annotate so you know what your values are

      fit, axes = plt.subplots(3, 2)
      for ax, var in zip(axes.ravel(), num_vars):
        sns.violinplot(y = var, data = cars, ax = ax)

---

### Python Packages - statsmodels ###

Statsmodels is the de facto library for performing regression tasks in Python.  
`import statsmodels.api as sm`
Note that logistic regression in statsmodels will fail to converge if data perfectly separable.  Logistic regression in sklearn normalizes (punishes betas) so it will converge.


### Python Packages - sklearn ###

Sklearn does not allow for categorical variables.  Everything must be encoded as a float.

Splitting test/training data:

      from sklearn.model_selection import train_test_split
      from sklearn.cross_validation import train_test_split # Older version
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


`from sklearn.linear_model import LinearRegression`
`from sklearn.linear_model import LogisticRegression`
`from sklearn.neighbors import KNeighborsClassifier`
`from sklearn.tree import DecisionTreeRegressor`

`from sklearn.ensemble import RandomForestClassifier`
* `max_features`: for classification start with sqrt(p) and for regression p/3
* `min_sample_leaf`: start with None and try others
* `n_jobs`: -1 will make it run on the max # of proocessors

`from sklearn.ensemble import GradientBoostingRegressor`
* You can used `staged_predict` to access the boosted steps, which is especially useful in plotting error rate over time

`from sklearn.svm import SVC` # sklearn uses SVC's even though it's a SVM.
* By default, SVC will use radial basis, which will enlarge your feature space with higher-order functions


A few other fun tools:

`from sklearn.model_selection import GridSearchCV`
Searches over designated parameters to tune a model
* GridSearch can parallelize jobs so set `n_jobs = -1`

`from sklearn.pipeline import Pipeline`
Note that pipeline is helpful for keeping track of changes

`from sklearn.preprocessing import LabelEncoder`
This tool will transform classes into numerical values


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

While data scientists are mostly accessing data, it's also useful to know how to create tables.

          CREATE TABLE table_name
          (
            column_name data_type(size),
            column_name data_type(size),
            column_name data_type(size)
            );

Data types include varchar, integer, decimal, date, etc.  The size specifies the maximum length of the column of the table.

Resources:
* http://sqlzoo.net/wiki/SELECT_basics
* An illustrated explanation of joins: https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/
* Databases by market share: http://db-engines.com/en/ranking
* SQL practice: https://pgexercises.com/
* SQL data types: http://www.w3schools.com/sql/sql_datatypes.asp

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

Here's a workflow:

1. One person creates a repository (and adds others as collaborators)
2. Everyone clones the repository
3. User A makes and commits a small change.  Like, really small.
4. User A pushes change.
5. User B makes and commits a small change.
6. User B tries to push, but gets an error.
7. User B pull User A's changes.
8. User B resolves any possible merge conflicts (this is why you keep commits small)
9. User B pushes.
10. Repeat

Centralized Git Workflow: https://www.atlassian.com/git/tutorials/comparing-workflows/

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
* `Ctr-Z`: puts current process to the background.  
* `fg`: brings the suspended process to the foreground

You can also access your bash profile with `atom ~/.bash_profile`

---

## Exploratory Data Analysis (EDA) ##

EDA is the first cut analysis where you evaluate the following points:

* What are the feature names and types?
* Are there missing values?
* Are the data types correct?
* Are there outliers?
* Which features are continuous and which are categorical?
* What is the distribution of the features?
* What is the distrubiton of the target?
* How do the variables relate to one another?

Some common functions to do this are `pd.head()`, `.describe()`, `.info()`, and `pd.crosstab()`.  If you see 'object' in the info result where you should have numbers, it is likely that you have a string hidden in the column somewhere.  When removing NA's, be sure that you don't drop any data that might be included in your final analysis.

There are a few options for dealing with NA's:

* Drop that data
* Fill those values with:
 * Column averages
 * 0 or other neutral number
 * Fill forwards or backwards (especially for time series)
 * Impute the values with a prediction (e.g. mean, mode)
* Ignore them and hope your model can handle them

One trick for imputation is to add another column to your model that's your variable imputed with a third column that's binary for whether you did impute it.  Using this method, your model can unlearn the imputation if you shouldn't have done it.

Exploratory plots such as scattermatrix.

---

## Linear Algebra ##

Linear algebra is about being able to solve equations in a more efficient manner.  A **scalar** (denoted by a lower-case, unbolded letter) has only magnitude (denoted as ||a||).  A **vector** (denoted by a lower-case, bold letter) is a one-dimensional matrix that has both magnitude and a heading.  **Distance** is the total area covered while **displacement** is the direct distance between your start and your end point.  A **matrix** is a m row by n column brick of numbers.  

You can initialize matrices and vectors using numpy as follows:

        mat = np.array([[4, -5], [-2, 3]])
        vect = np.array([-13, 9])
        column_vect = np.array([[13], [9]])
        np.ones((2, 4)) # Creates a 2 x 4 matrix of ones
        np.zeros((3, 2))
        mat.shape() # Returns the dimensions
        vect.reshape(2, 1) # Transposes the column vector vect
        mat' # Aliases the transpose (switches columns and rows) of mat
        np.transpose(mat) # Copies the transpose
        mat[1, 0] # returns -2
        mat[1] # returns row
        mat[:,1] # returns column
        np.concatenate((mat, vect)) # adds vect to mat by adding a row (use axis = 1 to add as a new column)
        mat + 1 # scalar operation (element-wise addition); can do w/ same-size matrices too

Matrix multiplication can only happen when *the number of columns of the first matrix equals the number of rows in the second*.  The inner product, or **dot product**, for vectors is the summation of the corresponding entities of the two sequences of numbers (returning a single, scalar value).  This can be accomplished with `np.dot(A, B)`.  The **outer product** of a 4-dimensional column vector and a 4-dimensional row vector is a 4x4 matrix where each value is the product of the corresponding column/vector value.

**Matrix-matrix multiplication** is a series of vector-vector products and is not communicative (meaning A*B != B*A).  A 2x3 matrix times 3x2 matrix gives a 2x2 result where 1,1 of the result is the dot product of the first row of the first matrix and the first column of the second matrix:

        A = [1, 2]
            [3, 4]
        B = [9, 7]
            [5, 8]
        AB = [1*9+2*5, 1*7+2*8]  =  [19, 23]
            [3*9+4*5, 3*7+4*8]     [47, 53]

An **identity matrix** is a square matrix with 1's along the diagonal and 0's everywhere else.  If you multiply any matrix by an identity matrix of the same size, you get the same matrix back.  There is no matrix division.  The **inverse** of a matrix is an alternative to division where you do 1 over the value of the given location.  A matrix multiplied by its inverse gives you an identity matrix.  A **transpose** is where the rows are exchanged for columns.

**Axis-wise** operations aggregate over the whole matrix.  For example, `A.mean()` returns the mean for the whole matrix.  `A.mean(axis = 0)` returns a mean for every column.  **Rank** is defined as the number of linearly dependent rows or columns in a matrix, such as a column that's the summation of two others or a multiple of another.  A **feature matrix** is a matrix where each column represents a variable and each row a datapoint.  By convention, we use X to be our feature matrix and y to be our dependent variable.  

An **orthogonal matrix** is important for statement of preference surveys. **Eigenvectors** and **eigenvalues** are good for page rank analysis.  The **stochastic matrix** is central to the Markov process, or a square matrix specifying the probabilities of going from one state to another such that every column of the matrix sums to 1.


---

## Probability ##

**Probability** is the measure of the likelihood that an event will occur written as the number of successes over the number of trials.  **Odds** are the number of successes over the number of failures.  Probability takes a value between 0 and 1 and odds usually take the form successes:failures.  

### Set Operations and Notation ###

A set is a range of all possible outcomes or events, also called the sampling space and can be discrete or continuous.  It is useful to think of set operations in the context of venn diagram.  Union is all of a venn diagram of A and B, intersection is the center portion, difference is the A portion and complement is the B portion.

* `Union`: A ∪ B = {x: x ∈ A ∨ x ∈ B} - The union of sets A and B is x such that x is in A or x is in B.
* `Intersection`: A ∩ B = {x: x ∈ A ∧ x ∈ B } - The intersection is x such that x is in A and in B.
* `Difference`: A ∖ B = {x: x ∈ A ∧ x ∉ B } - The difference is x such that x is in A and not in B.
* `Complement`: A <sup>C</sup> = {x: x ∉ A} - The complement is x such that x is in A and in B.
* `Null (empty) set`: ∅

DeMorgan's Law converts and's to or's.  The tetris-looking symbol is for 'not.'

* ¬(A ∨ B) ⟺ ¬A ∧ ¬B
* ¬(A ∧ B) ⟺ ¬A ∨ ¬B

Events are independent (A ⊥ B) if P(A ∩ B) = P(A)P(B) or (equivalently) P(A|B)=P(A).  Remember to think of a Venn Diagram in conceptualizing this.  This is conditional probaiblity.  

### Combinatorics ###

Combinatorics is the mathematics of ordering and choosing sets.  There are three basic approaches:

1. `Factorial`: Take the factorial of n to determine the total possible ordering of the items given that all the items will be used.
2. `Combinations`: The number of ways to choose k things given n options and that the order doesn't matter.
3. `Permutations`: The number of ways to choose k things given n options and that the order does matter.

Combinations: `n! / ((n-k)! * k!)`
Permutations: `n! / (n-k)!`

### Bayes Theorum ###

Bayes theorum states that P(B|A) = P(A|B)P(B) / P(A).  The denominator here is a normalizing function computed by all the possible ways that A could happen.  Let's take the following example:

The probability of a positive test result from a drug test given that one has doped is .99.  The probability of a positive test result given that they haven't doped is .05.  The probability of having doped is .005.  What is the probability of having doped given a positive test result?

      P(+|doped) - .99
      P(+|clean) - .05
      P(doped) - .005

      P(doped|+) = P(+|doped) * P(doped) / P(+)
       = P(+|doped) * P(doped) / P(doped) * P(+|uses) + P(clean) * P(+|clean)
       = (.99 * .005) / (.005 * .99 + (1-.005) * .05 )
       = .09

It's helpful to write this out a decision tree.  The denominator is the sum of all the possible ways you can get A, which means it's the sum of each final branch of the tree.

The **base rate fallacy** is the effect of a small population that has a disease on your ability to accurately predict it.  For rare diseases, multiple tests must be done in order to accurately evaluate if a person has the disease due to this fallacy.  

### Random Variables ###

A **random variable** is a function that maps events in our sample space to some numerical quantity.  There are three general types of these functions in applied probability:

1. `Probability mass function (PMF)`: Used for discrete random variables, the PMF return the probability of receiving that specific value.  Technically, the PMF encompasses both discrete and continuous variables.
2. `Probability density function (PDF)`: Used exclusively for continuous random variables, the PDF returns the probability that a value is in a given range
3. `Cumulative distribution function (CDF)`: tells us the probability that any draw will be less than a specific quantity (it's complement is the survivor function).  The CDF always stays constant or increases as x increases since it refers to the likelihood of a value less than it.  

We can compute the **covariance** of two different variables using the following: `Cov[X,Y] = E[(x−E[X]) (y−E[Y])]`.  This is related to the **correlation** which is the covariance over the multiplication of their standard deviations: `Cov(X,Y) / σ(X)σ(Y)`.  Correlation puts covariance on a -1 to 1 scale, allowing you to see proportion.  These look uniquely at linear relationships.

**Marginal distributions** take a possibly not independent multivariate distribution and considers only a single dimension.  By looking at the marginal distribution, you are able to **marginalize out** variables that have little covariance.  We always need to be thinking about the histograms of the two variables we're comparing as well as their intersect.

The **conditional distribution** is the joint distribution divided by the marginal distribution evaluated at a given point.  The conditional distribution says that we know a height, what's the distribution of weight given that height?  In data science, this is *the* thing we want to know.

The case of **Anscombe's quartet** shows us how statistics can often show us how poorly these statistics account for variance.  Correlation captures direction, not non-linearity, steep slopes, etc. This is why want to know the conditional distribution, not just summary stats.

**Pearson correlation** evaluates linear relationships between two continuous variables.  The **Spearman correlation** evaluates the monotonic relationship between two continuous or ordinal variables without assuming the linearity of the variables.

## Statistics ##

Statistics is the study of the collection, analysis, interpretation, presentation, and organization of data.  There are two general camps: frequentist and Bayesian.  Bayesians are allowed to impose prior knowledge onto their analysis.  The difference between these two camps largely boils down to what is fixed versus what is not.  Frequentists think that data are a repeatable random sample where the underlying parameters remain constant.  Bayesians, by contrast, believe that the data, not the underlying parameters, are fixed.  There is an argument that the two camps are largely similar if no prior is imposed on the analysis.  Bayseian statistics require a lot of computationally intense programming.

* Frequentist:
** Point estimates and standard errors
** Deduction from P(data|H0), by setting α in advance
** P-value determines acceptance of H0 or H1
* Bayesian:
** Start with a prior π(θ) and calculate the posterior π(θ|data)
** Broad descriptions

### Key Definitions ###

|                    | Population  | Sample   |
|--------------------|:-----------:|---------:|
| size               | N           | n        |
| mean               | μ          | xbar     |
| variance           | σ<sup>2</sup>    | s<sup>2</sup>     |
| standard deviation | σ       | s        |
| proportion         | π      | p ("little pea") |

Capital letters refer to random variables; lowercase refers to a specific realization.  Variables with a hat often refer to a predicted value.  `X` refers to all possible things that can happen in the population; `x` refers to draws from X.  

Other vocabulary:

* `S`: sample space, or a range of all possible outcomes (discrete or continuous)
* `i.i.d.`: independent, identically distributed (refers to when draws from X are not dependent on previous draws and fall into the same distribution)
* `α`: threshold for rejecting a hypothesis
* `β`:
* `λ`:

Maximum Likelihood Estimation (MLE) chooses the parameter(s) that maximize the likelihood of observing our given sample.

### Common distributions ###

Rules for choosing a good distribution:

* Is data discrete or continuous?
* Is data symmetric?
* What limits are there on possible values for the data?
* How likely are extreme values?

* Discrete:
 * Bernoulli: Model one instance of a success or failure trial (p)
 * Binomial: Number of successes out of a number of trials (n), each with probability of success (p)
 * Poisson: Model the number of events occurring in a fixed interval and events occur at an average rate (lambda) independently of the last event
 * Geometric: Sequence of Bernoulli trials until first success (p)
* Continuous:
 * Uniform: Any of the values in the interval of a to b are equally likely
 * Gaussian: Commonly occurring distribution shaped like a bell curve, often comes up because of the Central Limit Theorem
 * Exponential: Model time between Poisson events where events occur continuously and independently

### Frequentist Statistics ###

In frequentist statistics, there are four standard methods of estimation and sampling, to be explored below.  Central to frequentist statistics is the **Central Limit Theorum (CLT)** which states that the sample mean converges on the true mean as the sample size increases.  Variance also decreases as the sample size increases.

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

Variance is calculated as the squared deviation of the mean: `var(x) = E[(x - μ)<sup>2</sup>]`.  σ<sup>2</sup> and s<sup>2</sup> are different in that s<sup>2</sup> is multiplied by 1/(n-1) because n-1 is considered to be the degrees of freedom and a sample tends to understate a true population variance.  Because you have a smaller sample, you expect the true variance to be larger.  The number of **degrees of freedom** is the number of values in the final calculation of a statistic that are free to vary.  The sample variance has N-1 degrees of freedom, since it is computed from N random scores minus the only 1 parameter estimated as intermediate step, which is the sample mean.

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

http://glowingpython.blogspot.com/2012/08/kernel-density-estimation-with-scipy.html

#### Confidence Intervals ####

Assuming normality, you would use the following for your 95% confidence interval:

        xbar +- 1.96*(s / sqrt(n))

Z tests can be conducted for a smaller sample size (n < 30) while T tests are generally reserved for lager sample sizes.

**Bootstraping** estimates the sampling distribution of an estimator by sampling with replacement from the original sample.  Bootstrapping is often used to estimate the standard errors and confidence intervals of an unknown parameter, but can also be used for your beta coefficients.  We bootstrap when the theoretical distribution of the statistical parameter is complicated or unknown (like wanting a confidence interval on a median or correlation), when n is too small, and we we favor accuracy over computational costs.  It comes with almost no assumptions.

1. Start with your dataset of size n
2. Sample from your dataset with replacement to create a bootstrap sample of size n
3. Repeat step 2 a number of times (50 is good, though you often see 1k-10k)
4. Each bootstrap sample can then be used as a separate dataset for estimation and model fitting (often using percentile instead of standard error)

### Hypothesis Testing ###

1. State the null (H<sub>0</sub>) hypothesis and the alternative (H<sub>1</sub>)
2. Choose the level of significance (alpha)
3. Choose an appropriate statistical test and find the test statistic
4. Compute the p-value and either reject or fail to reject the H<sub>0</sub>

The following maps out type I and II errors.

|                     | H<sub>0</sub> is true    | H<sub>0</sub> is false  |
| ------------------- |:-------------:| -----:|
| Fail to reject H<sub>0</sub>   | correctly accept | Type II error/beta |
| Reject H<sub>0</sub>           | Type I error/alpha      |   correctly reject* |

* This is 1-beta or pi **this is the domain of power**

The court of law worries about type I error while in medicine we worry about type II error.  Tech generally worries about Type I error (especially in A/B testing) since we don't want a worse product.  The **power** of a test is the probability of rejecting the null hypothesis given that it is false.  If you want a smaller type II error, you're going to get it at the expense of a larger type I error.  Power is the complement of beta.

Use a **T test** when sigma is unknown and n < 30.  If you're not sure, just use a T test.  Scipy assumes that you're talking about a population.  You must set ddof = 1 for a sample.  A **Z test** is used for estimating a proportion.  **Welch's T Test** can be used when the variance is not equal (if unsure, set equal_var = False since it will only have a nominal effect on the result).

The **Bonferroni Correction** reduces the alpha value we use based upon the test that we're correcting.  We divide alpha by the number of tests.  It is a conservative estimate.

**Chi-squared tests** estimate whether two random variables are independent and estimate how closely an observed distribution matches an expected distribution (known as a goodness-of-fit test).  `chisquare` assumes a goodness-of-fit test while `chi1_contingency` assumes a contingency table.

**Experimental design** must address confounding factors that might also account for the variance.  You want to minimize confounding factors but there is such thing as controlling for too many confounding factors.  You can get to the point that you can’t have any data because you’ve over controlled, for instance women in SF in tech in start-ups post series b funding etc.

Power visualized: http://rpsychologist.com/d3/NHST/

### Bayesian Statistics ###

The first step is always to specify a probability model for unknown parameter values that includes some prior knowledge about the parameters if available.  We then update this knowledge about the unknown parameters by conditioning this probability model on observed data.  You then evaluate the fit of the model to the data.  If you don’t specify a prior then you will likely have a very similar response than the frequentists.

When we look at the posterior distribution, the denominator is just a normalizing constant.  The posterior is the new belief through the data that we’ve been given.  Priors come from published research, a researcher’s intuition, an expert option and non-informative prior.

#### Bayesian A/B Testing ####

In frequentist A/B testing, you can only reject or fail to reject.  You can't amass evidence for another hypothesis.  In Bayesian A/B testing, you can use a uniform distribution for an uninformative prior.  Depending on the study you're doing, an uninformative prior (which gives equal probability to all possible values) can be effectively the same as a frquentist approach.  If you use a bad prior, it will take longer to converge on the true value.  You can get a distribtuion from your test and then perform an element-wise comparison and take the mean to see where they overlap.  To test a 5% improvement, do element-wise plus .05.

The **multi-armed bandit** is the question of which option you take given prior knowledge.  There are two operative terms.  **Exploitation** leverages your current knowledge in order to get the highest expected reward at that time.  **Exploration** is testing other options to determine how good each one is.  Multi-armed bandit is now a big part of reinforcement learning (a branch of AI) more than it is part of stats.  You can use this for dynamic A/B testing, budget allocation amongst competing projects, clinical trials, adaptive routing for networks minimizing delays, and reinforcement learning.

**Regret** is teh difference between the maximal reward mean and the reward at time t.  You can never know what our actual regret is.  We don't know the true mean of a click-through rate, for instance.  Regret can be seen as how often you choose the suboptimal bandit (a cost function to minimize).

There are four main multi-armed bandit algorithms:

1. `Epsilon-Greedy`: Epsilon is the percent of time that we explore, frequently set at 10%.  Think of epsilon as how often you try a new restaurant.  Normally, you eat at your favorite spot but you want to choose a new one sometimes.  You try a new place, don’t like it, but it has low regret.
2. `UCB1`: Part of a set of algorithms optimized by upper confidence.  The UCB1 greedily chooses the bandit with the highest expected payout but with a clever factor that automatically balances exploration and exploitation by calculating exploration logarithmically.  This is a zero-regret strategy.
3. `Softmax`: creates a probability distribution over all the bandits in proportion to how good we think each lever is.  It’s a multi-dimensional sigmoid (a “squashing” function).  This is a probability matching algorithm.  Aneeling is where you vary tau as you go on, similar to lowering the temperature slowly when blowing glass or steel refining.
4. `Bayesian bandit`: softmax has one distribution governing the process, here we have distributions that sum to the number of values in our distribution.

---

## Modeling ##

### Linear Regression Introduction and Univariate ###

Linear regression is essentially fitting lines to data, originally coming from trying to predict child height to parent height.  In univariate linear regression, we are investigating the following equation:

yˆ = βˆ0 + βˆ1x + ε

Where ε is an error term that's iid and normally distributed with a mean of 0.

**Ordinary Least Squares** is the most common method for estimating the unknown parameters in a linear regression with the goal of minimizing the sum of the squares of the differences between observed and predicted values.  The difference between the ith observed response and the ith response value that is predicted by our linear model (or ei = yi −yˆi) is called the **residual**.  The **residual sum of squares (RSS)** is a measure of over fit denoted by RSS = e21 + e22 + ··· + e2n or RSS = (y1 −βˆ0−βˆ1x1)2 + (y2 −βˆ0−βˆ1x2) +...+ (yn−βˆ0−βˆ1xn)2.

Just like with estimated values, we can look at the standard error of our regression line to compute the range that the true population regression line likely falls within.  The **residual standard error (RSE)** is given by the formula RSE = sqrt(RSS/(n − 2)).  Using this SE, we can calculate the confidence interval, or, in other words, the bounds within which the true value likely falls.  We can use SE's to perform hypothesis tests on the coefficients.  Most commonly, we test the null hypothesis that there is no relationship between X and Y versus the alternative that there is a relationship (or H<sub>0</sub>: B<sub>1</sub> = 0 verus H<sub>1</sub>: B<sub>1</sub> != 0).

**Mean squared error** takes the average of the distance between our expected and real values.  It is a good error metric but is not comparable across different datasets.  

Roughly speaking, we interpret the **p-value** as follows: a small p-value indicates that it is unlikely to observe such a substantial association between the predictor and the response due to chance, in the absence of any real association between the predictor and the response. Hence, if we see a small p-value, then we can infer that there is an association between the predictor and the response. We reject the null hypothesis—that is, we declare a relationship to exist between X and Y —if the p-value is small enough.

The **F-statistic** allows us to compare models to see if we can reduce it.  The F-test can also be used generally to see if the model is useful beyond just predicting the mean.  

linear relations versus polynomial terms

**Assessing linear model fit** is normally done through two related terms: the RSE and the R**2 statistic.  The RSE estimates the standard deviation of ε, our error term.  It is roughly the average amount that our response will deviate from our regression line.  It is computed as sqrt((1/(n-2)) * RSS).  RSE is a measure of lack of model fit.

Since RSE is in the units of Y, it is not always clear what constitutes a good value.  **R**2** takes the form of a proportion of variance explained by the model between 0 and 1.  The R**2 formula is 1 - (RSS/TSS) where TSS is the sum of (yi − y¯)**2 or the **total sum of squares.** In simple regression, it's possible to show that the squared correlation between X and Y and the R**2 value are identical, however this doesn't extend to multivariate regression.



### Multivariate Regression ###

Multivariate regression takes the following form:

        Y = β0 + β1X1 + β2X2 + ··· + βpXp + ε

While univariate regression uses ordinary least squares to predict the coefficents, multivariate regression uses multiple least squares streamlined with matrix algebra.  To do this in python, use the following:

        est = sm.OLS(y, X) # This comes from statsmodels
        est = est.fit()
        est.summary()

This printout tells you the dependent variable (y), the type of model, and the method.  The F statistic will tell us if any of our coefficients are equal to 0.  AIC and BIC should be minimized.  We want to make sure the t statistic is larger than 2.  We want to then exclude the largest p values.  **Backwards stepwise** moves backwards through the model by removing the variables with the largest p values.  Skew and kurtosis are also important in reading a linear model because if they're drastic, it violates our t test since a t test assumes normality.  

The assumptions of a linear model are as follows:

1. `Linearity`
2. `Constant variance (homoscedasticity)`: this can often be rectified with a log
3. `Independence of errors`
4. `Normality of errors`
5. `Lack of multicollinearity`

**Residual plots** allow us to visualize whether there's a pattern not accounted for by our model, testing the assumptions of our model.

A **leverage point** is an observation with an unusual X value.  We calculate leverage using the **hat matrix** and the diagonals of which respond to its leverage.  **Studentized residuals** are the most common way of quantifying outliers.

**Variance Inflation Factors (VIF)** allows us to compute multicollinearity by the ratio of the variance of β^j when fitting the full model divided by the variance of β^j if fit on its own.  The smallest value for VIF is 1, which indicates a complete absence of multicollinearity.  The rule of thumb is that a VIF over 10 is problematic.  *Multicollinearity only affects the standard errors of your estimates, not your predictive power.*

**QQ plots** allow you to test normality by dividing your normal curve into n + 1 sections, giving you a visual for normality.  **Omitted variable bias** is when a variable you leave out of the model inflates the value of other variables which should be omitted had the original variable been included.

**Categorical variables** take a non-numeric value such as gender.  When using a categorical variable, you use a constant of all ones and then other variables (such as removing one ethnicity as the constant and adding two new variables for two other ethnicities).  To vary the slop, you can add an **interaction term** such as income multiplied by whether they're a student.  An **interaction effect** would be, for instance, when radio advertisements combined with tv has a more pronounced effect than separate.  This can be dealt with by multplying the two.  

Here are some potential transformations:

| Model   | Transformation(s) | Regression equation  | Predicted value (y^) |
| ---------- |:-----------------:| ----------------------:| --------------------:|
| Standard linear | none | y = β0 + β1x | y^ = β0 + β1x |
| Exponential | Dependent variable = log(y) | log(y) = β0 + β1x | y^ = 10**(β0 + β1x) |
| Quadradic | Dependent variable = sqrt(y) |sqrt(y) = β0 + β1x | y^ = (β0 + β1x)**2 |
| Reciprocal | Dependent variable = 1/y|1/y = β0 + β1x | y^ = 1 / (β0 + β1x) |
| Logarithmic | Independent variable = log(x) |y = β0 + β1log(x) | y^ = β0 + β1log(x) |
| Power | Dep and Ind variables = log(y) and log(x) |log(y) = β0 + β1log(x) | y^ = 10**(β0 + β1log(x)) |

![Transformations] (https://github.com/conorbmurphy/galvanizereference/blob/master/images/transformations.png)

Reference: http://emp.byui.edu/brownd/stats-intro/dscrptv/graphs/qq-plot_egs.htm


### Logistic Regression ###

Linear regression is good choice when your dependent variable is continuous and your independent variable(s) are either continuous or categorical.  *Logistic regression, a subset of linear regression, is used when your dependent variable is categorical.*  Examples of when you use logistic regression include customer churn, species extinction, patient outcome, prospective buyer purchase, eye color, and spam email.  Since linear regression is not bounded to our discrete outcome, we need something that takes a continuous input and produces a 0 or 1, has intuitive transition, and interpretable coefficients.  The **logit function** asymptotically approaches 0 and 1, coming from the sigmoid family.  It is solved via maximum likelihood.

If your results say "optimization not terminated successfully," do not use them as it's likely an issue with multicollinearity.  We look at the difference between LL-Null and Log-Likelihood, which tell us how far we could go to 0 (a perfect model) and how far we went (Log-Likelihood).  Doing `likelihood-ratio-test()` will return the p-value of one model being better than the other if one model is the same as the other with subtracted parameters.

*The coefficients for logistic regression are in log odds.*  In other words, we would interpret them as:

        exp(β0 + β1X1 + β2X2 + ··· + βpXp)
        ----------------------------------
        1 + exp(β0 + β1X1 + β2X2 + ··· + βpXp)

Similar to hypothesis testing, a confusion matrix gives you your true and false positive and negatives:

|                    | Predicted positive  | Predicted negative   |
|--------------------|:-----------:|---------:|
| Actually positive               | True positive           | False negative        |
| Actually negative               | False positive          | True negative     |

When you fill in this matrix, you can calculate accuracy (true poitives and negatives over n), misclassification (accuracy's compliment), true positive (actual yes over predicted yes), true negative, specificity, precision, etc.

The **Receiver Operator Characteristic (ROC) curve** is the sigmoid function with our different quadrants for true and false negatives and positives.   After you run your regression model, you’re going to get a score, which is a probability.  You’ll run a bunch of models with their respective confusion matrix.  You then plot them to see which model is better.  You might have to decide if you’re more worried about false positives or negatives.  You make a ROC curve by putting your sensitivity on the y axis and your false positive rate on the x axis.



### Cross-validation ###

Cross-validation is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent dataset.  The biggest mistake you can make in terms of validating a model is to keep all of your data together.  Do not evaluate yourself on your training set.  The **train/test split** divides your data into separate sets for you to train and evaluate your model.  When talking about splits, convention is to flip our dataset on its side to have variables as rows.  You can:

* Split the data evenly into two equal-size groups
* Split the data into n groups (each data point in its own group)
* `Leave one out Cross-validation`: Split the data into n groups of k points and train a completely new model on all the data minus one datapoint, evaluating it on that accuracy.  Ideally the size of k is 1.  Then you take the mean on a bunch of reporting models.  At the end, you retrain the model on the whole dataset.
* `10-fold cross-validation`: Same as above but using the convention of groups of 10

When training hyperparameters, set aside another subset of the data for testing at the end.  

Time series cross-validation example: http://robjhyndman.com/hyndsight/tscvexample/

### Bias/Variance Tradeoff ###

The error of any model is the function of three things: the **variance** (the amount that the function would change if trained on a different dataset) plus **bias** (the error due to simplified approximation) plus the irreducible error (epsilon).  Your bias goes up as your variance goes down and vice versa.  **Underfitting** is when the model does not fully capture the signal in X, being insufficiently flexible.  **Overfitting** is when the model erroneously interprets noise as signal, being too flexible.

In linear regression, we minimize the RSS by using the right betas.  You want to avoid large course corrections because that would mean high variance.  **Ridge (or L2) regression** and **lasso (L1) regression** avoid high variance by penalizing high betas.  In ridge, you add a hyperparameter that sums up the squared betas.  As the betas increase, the sum of the beta squared's increase as well.  Since you multiply your summation of squared betas by lambda, the bigger the lambda the more this is penalized. This evens out teh regression line.  Since high betas are penalized, we have to standardize the predictors.

Lasso regression is almost identical to ridge except you take the magnitude (or absolute value).  When you plot lasso, it looks like a V while ridge looks like a parabola.  Lasso still penalizes you when you're close to 0 so it's helpful for feature selection and more sparse models.  *Most consider lasso is better than ridge but it depends on the situation.*

### Gradient Ascent ###

Minimization algorithms are optimizations techniques.  In linear regression, we minimized the sum of squared errors; we minimized with respect to our betas.  In logistic regression, we maximized the likelihood function.  **Gradient ascent** is built on the idea that if we want to find the maximum point on a function, the best way to move is in the direction of the gradient. Gradient descent is the same function except our formula subtracts as to minimize a function.

It works on a small subset of problems but is used everywhere because we don't have many functions for optimization.  Here are some characteristics:

* It is like a derivative but with many dimensions
* It works on "smooth functions" with a gradient
* Must be some kind of convex function that has a minimum

If the above criteria are met, it will get asymptotically close to the global optimum.  The biggest concern is getting stuck in a local minimum where the derivative is zero.  In a broad sense, it works by taking a point on a curve and varying it slightly to see if its value has increased or decreased.  You then use that information to find the direction of the minimum.  **Alpha** is the size of the step you take towards that minimum.

**Stochastic gradient descent (SGD)** saves computation by taking one observation in the data and calculates the gradient based upon that.  The path is more chaotic however it will converge as long as there's no local minimum.  **Minibatch SDG** takes a subset of your data and computes the gradient based on that.  Relative to SGD, it takes a more direct path to the optimum while keeping us from having to calculate our minimum using the whole dataset.

In terms of increasing time (not necessarily number of iterations), SGD is often fastest ahead of minibatch and then standard gradient descent.  SGD often gets an accurate enough response and is good for big data as it converges faster on average and can work online (with updating new data and sunsetting old observations by pushing itself away from the optimum to check for change).  

The **Newton-Raphson Method** chooses our learning rate (alpha) in GD.  When the derivative is changing quickly, it takes a larger step.  When we're close to the minimum, it takes a smaller step by looking at the tangent's intersection with the x axis.

https://www.wolframalpha.com/

### Machine Learning (ML) ###

**Machine learning (ML)** is a "Field of study that gives computers the ability to learn without being explicitly programmed."  There are three broad categories of ML depending on the feedback available to the system:

1. `Supervised learning`: The algorithm is given example inputs and outputs with the goal of mapping the two together.  k-Nearest Neighbors and decision trees are non-parametric, supervised learning algorithms.  
2. `Unsupervised learning`: No labels are given to the algorithm, leaving it to find structure within the input itself.  Discovering hidden patterns or feature engineering is often the goal of this approach
3. `Reinforcement learning`: A computer interacts with a dynamic environment in which it must perform a goal (like driving a car) without being explicitly told if it has come close to its goal

The best model to choose often depends on computational costs at training time versus at prediction time in addition to sample size.  There are a number of possible outputs of ML:

* `Classification`: results are classified into one or more classes, such as spam filters.  For this, we have logistic regression, decision trees, random forest, KNN, SVM's, and boosting
* `Regression`: provides continuous outputs.
* `Clustering`: returns a set of groups where, unlike classification, the groups are not known beforehand
* `Density estimations`: finds the distribution of inputs in some space
* `Dimensionality reduction`: simplifies inputs by mapping them into a lower-dimensional space

An **ensemble** leverages the idea that more predictors can make a better model than any one predictor independently, combining many predictors with average or weighted averages.  There are a few types of ensembles:

1. `Committees`: this is the domain of random forest where regressions have an unweighted average and classification uses a majority
2. `Weighted averages`: these give more weight to better predictors (the domain of boosting)
3. `Predictor of predictors`: Treats predictors as features in a different model.  We can do a linear regression as a feature and random forest as another (part of the MLXtend package)



### k-Nearest Neighbors (KNN) ###

KNN is a highly accurate ML algorithm that is insensitive to outliers and makes no assumptions about the data.  You can also do online updates easily (you just store another data point), use as many classes as you want, and learn a complex function with no demands on relationships between variables (like linearity).  The downside is that it is computationally very expensive because it's **IO bound** (you have to read every data point to use it) and noise can affect results.  Categorical variables makes feature interpretation tricky.  It works with numeric and nominal values.  

KNN is a classifier algorithm that works by comparing a new piece of data to every piece of data in a training set.  We then look at the top k most similar pieces of data and take a majority vote, the winner of which will get its label assigned to the new data.  Your error rate is the number of misclassifications over the number of total classifications.

The method:

1. Collect: any method
2. Prepare: numeric values are needed for a distance calculation; structured data is best.  We need to feature scale in order to balance our distances.
3. Analyze: any method
4. Train: notable about KNN is that there is no training step.  You can change k and your distance metric, but otherwise there's not much to adapt
5. Test: calculate the error rate

You prediction is the majority (in the case of classification) or the average (in the case of regression) of the k nearest points.  Defining k is a challenge.  When k is 1, you get localized neighborhoods around what could be outliers (high variance) as well as 100% accuracy if you evaluate yourself on your training data.  If k is n then you're only guessing the most common class (high bias).  A good place to start is `k = sqrt(n)`.

Distance metrics can include any of the following:

* `Euclidean`: a straight line (and a safe default)
* `Manhattan`: straight lines from the base point to the destination using the number of stops at other points along the way.  Manhattan distance is based on how far a taxi would go following blocks in a city
* `Cosine`: uses the angle from the origin to the new point.  For instance, you can plot different points and the magnitude (distance from the origin) won't matter, only the distance from the origin. *Cosine distance is key in NLP.*
* `Custom`: you can always design your own distance metric as well

The **curse of dimensionality** is that as dimensionality increases, the performance of KNN commonly decreases.  This starts to affect you when you have p (number of dimensions) as 5 or greater.  The nearest neighbors are no longer nearby neighbors.  Adding useful features (that are truly associated with the response) is generally helpful but noise features increase dimensionality without an upside.  The more dimensions you have (p), the smaller the amount of space you have in your non-outlier region.  *This is the kryptonite of KNN*.  You'll often have to reduce dimensions to use it, especially with NLP.  **Locality-sensitive hashing** is a way to reduce the time to lookup each datapoint.

KNN can be used for classification, regression (neighbors averaged to give continuous value), imputation (replace missing data), and anomaly detection (if the nearest neighbor is far away, it might be an outlier).


### Decision Trees ###

Decision trees use **information theory** (the science of splitting data) to classify data into different sets, subsetting those sets further as needed.  One benefit of decision trees over KNN is that it's *incredibly interpretable*.  They are computationally inexpensive, feature interaction is already built in, and they can handle mixed data (discrete, continuous, and categorical).  The downside is that they are prone to overfitting.  They are also terrible at extrapolation (e.g. if I make $10 for 1 hour work and $20 for 2 hours work, they'll predict $20 for 10 hours of work).  Like KNN they work with numeric and nominal values however numeric values have to be translated into a nominal one.  They are computationally challenging at the training phase and inexpensive at the prediction phase as well as able to deal with irrelevant features and NA's (the opposite of KNN).

Trees consist of **nodes**.  One type of node is the common element throughout the tree known as the **root** (at the top, so it's botanically inaccurate).  The **stump** is the first node.  The **leaves** are a type of node at the end of the tree.  

**Information gain** is the difference between the pre-split and and post-split entropy.  The **entropy** of a set is a measure of the amount of disorder.  We want to create splits that minimize the entropy in each side of split.  **Cross-entropy** is a measure of node purity using the log and **Gini index** does the same with a slightly different formula (both are effectively the same).  You  can do regression decision trees by using RSS against the mean value of each leaf instead of cross-entropy or Gini.  You can also use variance before and after.

The method:

1. Collect: any method
2. Prepare: any continuous values need to be quantized into nominal ones
3. Analyze: any method, but trees should be examined after they're built
4. Train: construct a tree by calculating the information gain for every possible split and select the split that has the highest information gain.  Splits for categorical features are `variable = value` or `!= variable`  and for continuous variable `variable <= threshold`
5. Test: calculate the error rate

With *continuous variables*, decision trees divide the space and use the mean from that given space.  For continuous variables, there are three general split methods for decision tree regression:

1. Minimize RSS after the split
2. Reduce the weighted variance after each split (weighted by the number of each side of split over the total number of values going to that split)
3. Reduce weighted std after the split


Decision trees are high variance since they are highly dependent on the training data.  We can ease up on the variance by **pruning**, which is necessary whenever you make trees. **Prepruning** is when you prune as you build the tree.  You can do this with leaf size (stopping when there's few data points at a node), depth (stop when a tree gets too deep), class mix (stop when some percent of data points are the same class), and error reduction (stop when the information gains are small).  For **postpruning**, you can build a full tree, then cut off some leaves (meaning merge them) using a formula similar to ridge regression.  You will likely extend your trees with bagging, random forests, or boosting.  

Algorithms for splitting data include ID3, C4.5, and CART.

### Bagging and Random Forests ###

**Bagging** is Bootstrap AGGregatING where you take a series of bootstrapped samples from your data following this method:

1. Draw a random bootstrap sample of size n
2. Grow a decision tree using the same split criteria at each node by maximizing the information gain (no pruning)
3. Repeat steps 1 & 2 k times
4. Aggregate the prediction by each tree to assign the class label by majority vote (for classification) or average (for regression)

Bagging starts with high variance and averages it away.  Boosting does the opposite, starting with high bias and moving towards higher variance.

**Random forest** is the same as bagging but with one new step: in step 2 it randomly selects d features without resampling (further decorrelating the trees).  We can do this process in parallel to save computation.  Start with 100-500 trees and plot the number of trees versus your error.  You can then do many more trees at the end of your analysis.  Generally you don't need interaction variables though there may be times when you need them.

Random forest offers **feature importance**, or the relative importance of the variables in your data.  This makes your forest more interpretable and gives you free feature selection.  Note that you are only interested in rank, not magnitude, and that multicollinearity will inflate the value of certain variables.  There are two ways of calculating this, called the first and second way:

1. Start with an array that is the same number of features in your model and then calculate the information gain and points split with each variable.
2. Calculate OOB error for a given tree by evaluating predictions on the observations that were not used in building the base learner.  After, take your features and give them a random value between the min and max and calculate how much worse it makes your model.

One downside of random forest is that it sacrifices the interpretability of individual trees.  *Explain or predict, don't do both*.  Some models have to be explained to stakeholders.  Others just need high predictive accuracy.  Try to separate these two things whenever possible.

**Out of Bag (OOB) Error** pertains to bootstrap samples.  Since each bootstrap is different and you're already calling it, you can use your OOB samples for cross-validation using `oob_score = True`.  *You will rarely cross-validate a random forest because OOB error effectively acts as the cross-validation* meaning that you only need a test/train split, not a validation set.

For categorical data, strings need to be converted to numeric.  If possible, convert to a continuous variable (e.g. S, M, L into a weight and height).

Galit Shmueli's paper "To Explain or to Predict?""

### Boosting ###

**Boosting** is generally considered to be the best out of box supervised learning algorithm for classification and regression.  In practice, you often get similar results from bagging and random forests while boosting generally gives you better performance.  Boosting does well if you have more features than sample size, and is especially helpful for website data where we're collecting more and more information.

While it's most natural to think of boosting in the context of trees, it applies to other 'weak learners' including linear regression.  It does not apply to strong learners like SVM's.  While random forest and bagging creates a number of trees and looks for consensus amongst them, boosting trains a single tree.  That is, random forest takes place in parallel while boosting must be done in series since each tree relies on the last (making it slower to train).  Boosting does not involve bootstrap sampling: each tree is fit on the error of the previous model.

The first predicting function does practically nothing where the error is basically everything.  Here's the key terminology:

* `B`: your number of trees, likely in the hundreds or thousands.  Since boosting can overfit (though it's rare), pick this with cross-validation
* `D`: your depth control on your trees (also known as the interaction depth since it controls feature interaction order/degree).  Often D is 1-3 where 1 is known as stumps.
* `λ`: your learning rate.  It is normally a small number like .1, .01, or .001.  Use CV to pick it.  λ is in tension with B because learning more slowly means needing more trees
* `r`: your error rate.  Note that you're fitting to r instead of y.


For every B, you fit a tree with d splits to r.  You update that with a version of the tree shrunken with λ and update the residuals.  The boosted model is the sum of all your functions multiplied by λ.  At each step, you upweight the residuals you got wrong.  We control depth and learning rate to avoid a 'hard fit' where we overfit the data.  This is the opposite of random forest where we make complex models and then trim them back.

There are many different types of boosting with the best option depending on computation/speed, generality, flexibility and nomenclature:

* `AdaBoost`: 'adaptive boosting' that upweights the points you're getting wrong to focus more on them.  It will weigh each individual weak learner based on its performance.  This is a special case of gradient boosting, which was discovered after AdaBoost.  AdaBoost allows us to be more nuanced in how we shrink a function as it's not just a constant lambda.  You have two sets of weights: d, our weight for an underrepresented class (updated as it goes), and alpha, which is the weight of our weak learners (constant).  *Anything better than random is valuable.*  Alpha flips at .5, meaning that if a given learner is wrong more than it's right we predict the opposite outcome.
* `Gradient Boosting`: improves the handling of loss functions for robustness and speed.  It needs differentiable loss functions
** `XGBoost`: a derivation on gradient boosting invented, interestingly enough, by a student in China to win a Kaggle competition

To *compare our models*, lets imagine you want a copy of a painting.  Bagging would be the equivalent of asking a bunch of painters to observe the painting from the same location and then go paint it from memory.  Random forests would be the same except these painters could stand anywhere around it.  In both of these cases, you would average the results of all the painters.  Boosting, by contrast, would be the equivalent of asking one single painter to sit by the painting, waiting for each stroke to dry, and then painting the 'error' between what they've already painted and the true painting.

A particularly helpful visualization: https://github.com/zipfian/DSI_Lectures/blob/master/boosting/matt_drury/Boosting-Presentation-Galvanize.ipynb

### Maximal Margin Classifier, Support Vector Classifiers, Support Vector Machines ###

Support Vector Machines (SVM) are typically thought of as a classification algorithm however they can be used for regressions as well.  SVM's used to be as well researched as neural networks are today and are considered to be one of the best "out of the box" classifiers.  They are useful when you have a sparcity of solutions achievable by the l1 penalty and when you know that kernels and margins can be an effective way to approach your data.

In p-dimensional space, a **hyperplane** is a flat affine subspace of dimension p-1.  With a dat set of an n x p matrix, we have n points in p-dimensional space.  Points of different classes, if completely separable, are able to be separated by an infinite number of hyperplanes.  The **maximal margin hyperplane** is the separating hyperplane that is farthest from the training observations.  The **margin** is the orthogonal distance between points between two classes (when the inner product of two vectors is 0, it's orthoginal).  The **maximal margin classifier** then is when we classify a test observation based on which side of the maximal margin hyperplane it lies.

This margin relies on just three points, those closest to the hyperplane.  These points are called the **support vectors**.

While a maximal margin classifier relies on a perfect division of classes, a **support vector classifier** (also known as a **soft margin classifier**) is a generalization to the non-separable case.  Using the term **C** we control the number and severity of violations to the margin.  In practice, we evaluate this tuning parameter at a variety of values using cross-validation.  C is how we control the bias/variance trade-off for this model, as it increases we become more tolerant of violations, giving us more bias and less variance.

Finally, **support vector machines (SVM)** allow us to address the problem of possibly non-linear boundaries by enlarging the feature space using quadratic, cubic, and higher-order polynomial functions as our predictors.  **Kernels** are a computationally efficient way to enlarge your feature space as they rely only on the inner products of the observations, not the observations themselves.  There are a number of different kernels that can be used.

The slack variable allows us to relax our constraint in a given case.  Class imbalance is an issue for SVM (when you have a lot of classes of different positive and negative values).  When you penalize the betas with respect to the size of the beta, then you should scale them.  SVM's encode yi as (-1, 1) to denote which side of our hyperplane it's on, a change from logistic regression where it's encoded as (0, 1).

If there are more than two classes of data, there are two options to approach the problem:

1. `One versus the rest`: train k models for your k classes and choose the model that predicts the highest probability for your specific class
2. `One versus one`: choose the best model based on ties

Theree are many *differences between logisic regression and SVM's*.  A logistic regression only asymptotically approaches 0 or 1.  Perfectly separable data is a problem for logistic regression where it won't converge.  SVM's give you a class without a probability; logistic regression assigns a probability for a given class.

MIT lecture on SVM's: https://www.youtube.com/watch?v=_PwhiWxHK8o

---

## Helpful Visualizations ##

Feature importance: Plot your features in order of importance (tells you importance but not if they correlate positively or negatively)
Partial dependency: This makes predictions having froze a given feature and incrementing it up.  FOr instance, you can plot two features against the 'partial dependence', which is your outcome.  
ROC
Residual plots
QQ Norm

---

## Note on Style and Other Tools ##

iPython offers features like tab completion and auto-reload over the main python install.  You can also type `%debug` to debut code.
Jupyter/iPython notebook or Apache Zeppelin
Markdown: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
Atom: command / hashes out text
Anacoda
Homebrew
AWS: https://gist.github.com/iamatypeofwalrus/5183133

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
* Basic combinatorics
* Linear regression basics, especially that LR is more complex than y = MX + B
* Interpreting coefficients for linear and logistic regression
* How do you know if you overfit a model and how do you adjust for it?

O'Reilly (Including salary averages): https://www.oreilly.com

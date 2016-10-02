# Galvanize Reference #

This is designed to be a catch-all reference tool for my time in Galvanize's Data Science Immersive program.  Others might find it of use as well.

The main responsibilities of a data scientist are:

1. `Ideation` (experimental design)
2. `Importing` (SQL/postgres/psycopg2)
 * defining the ideal dataset
 * understanding your data compares to the ideal
4. `Exploratory data analysis (EDA)` (python/pandas)
4. `Data munging` (pandas)
5. `Feature engineering`
6. `Modeling`
7. `Presentation`

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

Working with datetime objects, you can do df.dt. and then hit tab to see the different options.  Similarly, df.str can give us all of our string functions.  

`crosstab` is similar to table in R

---

### Python Packages - numpy ###

Linear algebra package
In calculating `np.std()`, be sure to specify `ddof = 1` when refering to the sample.

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

### Python Packages - sklearn/statsmodels ###

Statsmodels is the de facto library for performing regression tasks in Python.  
`import statsmodels.api as sm`


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

There are a few options for dealing with NA's, including deleting them from your dataset and imputing their values using prediction, mean, mode, etc.

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

Probability is the measure of the likelihood that an event will occur.

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

Boostraping estimates the sampling distribution of an estimator by sampling with replacement from the original sample.  Bootstrapping is often used to estimate the standard errors and confidence intervals of an unknown parameter.  We bootstrap when the theoretical distribution of the statistical parameter is complicated or unknown (like wanting a confidence interval on a median or correlation), when n is too small, and we we favor accuracy over computational costs.  It comes with almost no assumptions.

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

**QQ plots** allow you to test normality by dividing your normal curve into n + 1 sections, giving you a visual for normality.

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


Reference: http://emp.byui.edu/brownd/stats-intro/dscrptv/graphs/qq-plot_egs.htm


### Logistic Regression

**Gradient ascent** is built on the idea that if we want to find the maximum point on a function, the best way to move is in the direction of the gradient. Gradient descent is the same function except our formula subtracts as to minimize a function.

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
* Basic combinatorics
* Linear regression basics, especially that LR is more complex than y = MX + B

O'Reilly (Including salary averages): https://www.oreilly.com


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("/Users/mrpurtas/Downloads/en.openfoodfacts.org.products.tsv", delimiter='\t')

# ############## ############ Exercise 1 ################## ####################
#
df.head(5)

df.shape[0]

df.shape[1]

print(df.columns)

df.columns[104]

df.iloc[:, 104].dtype

How is the dataset indexed?????

df.loc[18, "product_name"]

# ############## ############  Ex2 - Getting and Knowing your Data   ################## ####################
"""

Ex2 - Getting and Knowing your Data
Check out Chipotle Exercises Video Tutorial to watch a data scientist go through the exercises

This time we are going to pull data directly from the internet. Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Step 1. Import the necessary libraries
import pandas as pd
import numpy as np
Step 2. Import the dataset from this address.
Step 3. Assign it to a variable called chipo.
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    
chipo = pd.read_csv(url, sep = '\t')
Step 4. See the first 10 entries
chipo.head(10)
order_id	quantity	item_name	choice_description	item_price
0	1	1	Chips and Fresh Tomato Salsa	NaN	$2.39
1	1	1	Izze	[Clementine]	$3.39
2	1	1	Nantucket Nectar	[Apple]	$3.39
3	1	1	Chips and Tomatillo-Green Chili Salsa	NaN	$2.39
4	2	2	Chicken Bowl	[Tomatillo-Red Chili Salsa (Hot), [Black Beans...	$16.98
5	3	1	Chicken Bowl	[Fresh Tomato Salsa (Mild), [Rice, Cheese, Sou...	$10.98
6	3	1	Side of Chips	NaN	$1.69
7	4	1	Steak Burrito	[Tomatillo Red Chili Salsa, [Fajita Vegetables...	$11.75
8	4	1	Steak Soft Tacos	[Tomatillo Green Chili Salsa, [Pinto Beans, Ch...	$9.25
9	5	1	Steak Burrito	[Fresh Tomato Salsa, [Rice, Black Beans, Pinto...	$9.25
Step 5. What is the number of observations in the dataset?
# Solution 1

chipo.shape[0]  # entries <= 4622 observations
4622
# Solution 2

chipo.info() # entries <= 4622 observations
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4622 entries, 0 to 4621
Data columns (total 5 columns):
order_id              4622 non-null int64
quantity              4622 non-null int64
item_name             4622 non-null object
choice_description    3376 non-null object
item_price            4622 non-null object
dtypes: int64(2), object(3)
memory usage: 180.6+ KB
Step 6. What is the number of columns in the dataset?
chipo.shape[1]
5
Step 7. Print the name of all the columns.
chipo.columns
Index([u'order_id', u'quantity', u'item_name', u'choice_description',
       u'item_price'],
      dtype='object')
Step 8. How is the dataset indexed?
chipo.index
RangeIndex(start=0, stop=4622, step=1)
Step 9. Which was the most-ordered item?
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
order_id	quantity
item_name		
Chicken Bowl	713926	761
Step 10. For the most-ordered item, how many items were ordered?
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
order_id	quantity
item_name		
Chicken Bowl	713926	761
Step 11. What was the most ordered item in the choice_description column?
c = chipo.groupby('choice_description').sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
# Diet Coke 159
order_id	quantity
choice_description		
[Diet Coke]	123455	159
Step 12. How many items were orderd in total?
total_items_orders = chipo.quantity.sum()
total_items_orders
4972
Step 13. Turn the item price into a float
Step 13.a. Check the item price type
chipo.item_price.dtype
dtype('O')
Step 13.b. Create a lambda function and change the type of item price
dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollarizer)
Step 13.c. Check the item price type
chipo.item_price.dtype
dtype('float64')
Step 14. How much was the revenue for the period in the dataset?
revenue = (chipo['quantity']* chipo['item_price']).sum()

print('Revenue was: $' + str(np.round(revenue,2)))
Revenue was: $39237.02
Step 15. How many orders were made in the period?
orders = chipo.order_id.value_counts().count()
orders
1834
Step 16. What is the average revenue amount per order?
# Solution 1

chipo['revenue'] = chipo['quantity'] * chipo['item_price']
order_grouped = chipo.groupby(by=['order_id']).sum()
order_grouped.mean()['revenue']
21.394231188658654
# Solution 2

chipo.groupby(by=['order_id']).sum().mean()['revenue']
21.394231188658654
Step 17. How many different items are sold?
chipo.item_name.value_counts().count()
50
"""

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"

chipo = pd.read_csv(url, delimiter='\t')

chipo.head(10)

chipo.shape[0]

chipo.shape[1]

print(chipo.columns)

chipo.index

chipo["item_name"].value_counts().idxmax()

chipo[chipo["item_name"] == "Chicken Bowl"]["quantity"].sum()

chipo["choice_description"].value_counts().idxmax()

chipo["quantity"].sum()

chipo["item_price"].dtype
chipo["item_price"] = chipo["item_price"].str.replace("$", " ").str.strip()
chipo["item_price"].astype("float")
chipo["item_price"] = chipo["item_price"].apply(lambda x: float(x.replace("$", "").strip()))

chipo["item_price"].sum()

chipo["quantity"].sum()

chipo["item_price"].mean()

chipo["item_name"].nunique()

######################   Fictional Army - Filtering and Sorting   #########################################
"""

Ex2 - Getting and Knowing your Data
Check out Chipotle Exercises Video Tutorial to watch a data scientist go through the exercises

This time we are going to pull data directly from the internet. Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Step 1. Import the necessary libraries
import pandas as pd
import numpy as np
Step 2. Import the dataset from this address.
Step 3. Assign it to a variable called chipo.
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    
chipo = pd.read_csv(url, sep = '\t')
Step 4. See the first 10 entries
chipo.head(10)
order_id	quantity	item_name	choice_description	item_price
0	1	1	Chips and Fresh Tomato Salsa	NaN	$2.39
1	1	1	Izze	[Clementine]	$3.39
2	1	1	Nantucket Nectar	[Apple]	$3.39
3	1	1	Chips and Tomatillo-Green Chili Salsa	NaN	$2.39
4	2	2	Chicken Bowl	[Tomatillo-Red Chili Salsa (Hot), [Black Beans...	$16.98
5	3	1	Chicken Bowl	[Fresh Tomato Salsa (Mild), [Rice, Cheese, Sou...	$10.98
6	3	1	Side of Chips	NaN	$1.69
7	4	1	Steak Burrito	[Tomatillo Red Chili Salsa, [Fajita Vegetables...	$11.75
8	4	1	Steak Soft Tacos	[Tomatillo Green Chili Salsa, [Pinto Beans, Ch...	$9.25
9	5	1	Steak Burrito	[Fresh Tomato Salsa, [Rice, Black Beans, Pinto...	$9.25
Step 5. What is the number of observations in the dataset?
# Solution 1

chipo.shape[0]  # entries <= 4622 observations
4622
# Solution 2

chipo.info() # entries <= 4622 observations
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4622 entries, 0 to 4621
Data columns (total 5 columns):
order_id              4622 non-null int64
quantity              4622 non-null int64
item_name             4622 non-null object
choice_description    3376 non-null object
item_price            4622 non-null object
dtypes: int64(2), object(3)
memory usage: 180.6+ KB
Step 6. What is the number of columns in the dataset?
chipo.shape[1]
5
Step 7. Print the name of all the columns.
chipo.columns
Index([u'order_id', u'quantity', u'item_name', u'choice_description',
       u'item_price'],
      dtype='object')
Step 8. How is the dataset indexed?
chipo.index
RangeIndex(start=0, stop=4622, step=1)
Step 9. Which was the most-ordered item?
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
order_id	quantity
item_name		
Chicken Bowl	713926	761
Step 10. For the most-ordered item, how many items were ordered?
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
order_id	quantity
item_name		
Chicken Bowl	713926	761
Step 11. What was the most ordered item in the choice_description column?
c = chipo.groupby('choice_description').sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
# Diet Coke 159
order_id	quantity
choice_description		
[Diet Coke]	123455	159
Step 12. How many items were orderd in total?
total_items_orders = chipo.quantity.sum()
total_items_orders
4972
Step 13. Turn the item price into a float
Step 13.a. Check the item price type
chipo.item_price.dtype
dtype('O')
Step 13.b. Create a lambda function and change the type of item price
dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollarizer)
Step 13.c. Check the item price type
chipo.item_price.dtype
dtype('float64')
Step 14. How much was the revenue for the period in the dataset?
revenue = (chipo['quantity']* chipo['item_price']).sum()

print('Revenue was: $' + str(np.round(revenue,2)))
Revenue was: $39237.02
Step 15. How many orders were made in the period?
orders = chipo.order_id.value_counts().count()
orders
1834
Step 16. What is the average revenue amount per order?
# Solution 1

chipo['revenue'] = chipo['quantity'] * chipo['item_price']
order_grouped = chipo.groupby(by=['order_id']).sum()
order_grouped.mean()['revenue']
21.394231188658654
# Solution 2

chipo.groupby(by=['order_id']).sum().mean()['revenue']
21.394231188658654
Step 17. How many different items are sold?
chipo.item_name.value_counts().count()
50
"""

raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35], 'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9],
'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973,
1005, 1099, 1523],
'veterans': [1, 5, 62, 26, 73, 37, 949, 48, 48, 435, 63, 345], 'readiness': [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 3],
'armored': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
'deserters': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3], 'origin': ['Arizona', 'California', 'Texas', 'Florida',
'Maine', 'Iowa', 'Alaska', 'Washington', 'Oregon', 'Wyoming', 'Louisana', 'Georgia']}

army = pd.DataFrame(raw_data)

army.set_index("origin", inplace=True)

print(army["veterans"])

print(army[["veterans", "deaths"]])

print(army.columns)

army.loc[['Maine', 'Alaska'],['deaths', 'size', 'deserters']]

army.iloc[2:7, 2:6]

army.iloc[3: , :

army.iloc[:, 2:7]

army[(army["deaths"] > 500) | (army["deaths"] < 50)]

army[army["regiment"] != "Dragoons"]

army.loc[["Texas", "Arizona"]]
################################################################################################################
################################################################################################################
""""################################################################################################################
loc[] ve loc[[]] farklı amaçlara hizmet eden DataFrame indeksleme yöntemleridir:

loc[] tek bir köşeli parantez içinde kullanılır ve sadece bir satır veya bir sütunu seçmek için kullanılır. Örneğin:

army.loc["Arizona", "deaths"]

df.loc['Satır1']  # Tek bir satırı seçer
df.loc[:, 'Sütun1']  # Tek bir sütunu seçer

loc[[]] çift köşeli parantez içinde kullanılır ve birden fazla satır veya sütunu seçmek için kullanılır. Örneğin:
df.loc[['Satır1', 'Satır2']]  # Birden fazla satırı seçer
df.loc[:, ['Sütun1', 'Sütun2']]  # Birden fazla sütunu seçer
################################################################################################################
################################################################################################################
################################################################################################################

"""
######################################      EX_GROUPBY     ####################################################


df =pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv")

df.groupby("continent")["beer_servings"].mean()

df.groupby("continent")["wine_servings"].describe()

df.groupby("continent").mean()

df.groupby("continent").median()

df.groupby("continent").spirit_servings.agg(["mean", "max", "min"])

######################################      REGİMENT     ####################################################
"""
Introduction:
Special thanks to: http://chrisalbon.com/ for sharing the dataset and materials.

Step 1. Import the necessary libraries
import pandas as pd
Step 2. Create the DataFrame with the following values:
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
Step 3. Assign it to a variable called regiment.
Don't forget to name each column
regiment = pd.DataFrame(raw_data, columns = raw_data.keys())
regiment
regiment	company	name	preTestScore	postTestScore
0	Nighthawks	1st	Miller	4	25
1	Nighthawks	1st	Jacobson	24	94
2	Nighthawks	2nd	Ali	31	57
3	Nighthawks	2nd	Milner	2	62
4	Dragoons	1st	Cooze	3	70
5	Dragoons	1st	Jacon	4	25
6	Dragoons	2nd	Ryaner	24	94
7	Dragoons	2nd	Sone	31	57
8	Scouts	1st	Sloan	2	62
9	Scouts	1st	Piger	3	70
10	Scouts	2nd	Riani	2	62
11	Scouts	2nd	Ali	3	70
Step 4. What is the mean preTestScore from the regiment Nighthawks?
regiment[regiment['regiment'] == 'Nighthawks'].groupby('regiment').mean()
preTestScore	postTestScore
regiment		
Dragoons	15.50	61.5
Nighthawks	15.25	59.5
Scouts	2.50	66.0
Step 5. Present general statistics by company
regiment.groupby('company').describe()
postTestScore	preTestScore
company			
1st	count	6.000000	6.000000
mean	57.666667	6.666667
std	27.485754	8.524475
min	25.000000	2.000000
25%	34.250000	3.000000
50%	66.000000	3.500000
75%	70.000000	4.000000
max	94.000000	24.000000
2nd	count	6.000000	6.000000
mean	67.000000	15.500000
std	14.057027	14.652645
min	57.000000	2.000000
25%	58.250000	2.250000
50%	62.000000	13.500000
75%	68.000000	29.250000
max	94.000000	31.000000
Step 6. What is the mean of each company's preTestScore?
regiment.groupby('company').preTestScore.mean()
company
1st     6.666667
2nd    15.500000
Name: preTestScore, dtype: float64
Step 7. Present the mean preTestScores grouped by regiment and company
regiment.groupby(['regiment', 'company']).preTestScore.mean()
regiment    company
Dragoons    1st         3.5
            2nd        27.5
Nighthawks  1st        14.0
            2nd        16.5
Scouts      1st         2.5
            2nd         2.5
Name: preTestScore, dtype: float64
Step 8. Present the mean preTestScores grouped by regiment and company without heirarchical indexing
regiment.groupby(['regiment', 'company']).preTestScore.mean().unstack()
company	1st	2nd
regiment		
Dragoons	3.5	27.5
Nighthawks	14.0	16.5
Scouts	2.5	2.5
Step 9. Group the entire dataframe by regiment and company
regiment.groupby(['regiment', 'company']).mean()
preTestScore	postTestScore
regiment	company		
Dragoons	1st	3.5	47.5
2nd	27.5	75.5
Nighthawks	1st	14.0	59.5
2nd	16.5	59.5
Scouts	1st	2.5	66.0
2nd	2.5	66.0
Step 10. What is the number of observations in each regiment and company
regiment.groupby(['company', 'regiment']).size()
company  regiment  
1st      Dragoons      2
         Nighthawks    2
         Scouts        2
2nd      Dragoons      2
         Nighthawks    2
         Scouts        2
dtype: int64
Step 11. Iterate over a group and print the name and the whole data from the regiment
# Group the dataframe by regiment, and for each regiment,
for name, group in regiment.groupby('regiment'):
    # print the name of the regiment
    print(name)
    # print the data of that regiment
    print(group)
Dragoons
   regiment company    name  preTestScore  postTestScore
4  Dragoons     1st   Cooze             3             70
5  Dragoons     1st   Jacon             4             25
6  Dragoons     2nd  Ryaner            24             94
7  Dragoons     2nd    Sone            31             57
Nighthawks
     regiment company      name  preTestScore  postTestScore
0  Nighthawks     1st    Miller             4             25
1  Nighthawks     1st  Jacobson            24             94
2  Nighthawks     2nd       Ali            31             57
3  Nighthawks     2nd    Milner             2             62
Scouts
   regiment company   name  preTestScore  postTestScore
8    Scouts     1st  Sloan             2             62
9    Scouts     1st  Piger             3             70
10   Scouts     2nd  Riani             2             62
11   Scouts     2nd    Ali             3             70
"""
import pandas as pd
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'],
'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3], 'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}

df = pd.DataFrame(raw_data, columns= raw_data.keys())

df

df[df["regiment"] == "Nighthawks"].groupby("regiment").mean()

df.groupby("company").describe()

df.groupby("company").preTestScore.agg("mean")

df.groupby(["company", "regiment"]).preTestScore.agg("mean")

df.groupby(["company", "regiment"]).preTestScore.agg("mean").unstack()

df.head()

df.groupby(["regiment", "country"]).mean()

df.groupby(['company', 'regiment']).size()

df.groupby("regiment").mean()

for name, group in df.groupby("regiment"):
    print(name)
    print(group)

#####################  United States-Crime Rates-1960–2014 #################
"""
United States - Crime Rates - 1960 - 2014
Check out Crime Rates Exercises Video Tutorial to watch a data scientist go through the exercises

Introduction:
This time you will create a data

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Step 1. Import the necessary libraries
import numpy as np
import pandas as pd
Step 2. Import the dataset from this address.
Step 3. Assign it to a variable called crime.
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv"
crime = pd.read_csv(url)
crime.head()
Year	Population	Total	Violent	Property	Murder	Forcible_Rape	Robbery	Aggravated_assault	Burglary	Larceny_Theft	Vehicle_Theft
0	1960	179323175	3384200	288460	3095700	9110	17190	107840	154320	912100	1855400	328200
1	1961	182992000	3488000	289390	3198600	8740	17220	106670	156760	949600	1913000	336000
2	1962	185771000	3752200	301510	3450700	8530	17550	110860	164570	994300	2089600	366800
3	1963	188483000	4109500	316970	3792500	8640	17650	116470	174210	1086400	2297800	408300
4	1964	191141000	4564600	364220	4200400	9360	21420	130390	203050	1213200	2514400	472800
Step 4. What is the type of the columns?
crime.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 55 entries, 0 to 54
Data columns (total 12 columns):
Year                  55 non-null int64
Population            55 non-null int64
Total                 55 non-null int64
Violent               55 non-null int64
Property              55 non-null int64
Murder                55 non-null int64
Forcible_Rape         55 non-null int64
Robbery               55 non-null int64
Aggravated_assault    55 non-null int64
Burglary              55 non-null int64
Larceny_Theft         55 non-null int64
Vehicle_Theft         55 non-null int64
dtypes: int64(12)
memory usage: 5.2 KB
Have you noticed that the type of Year is int64. But pandas has a different type to work with Time Series. Let's see it now.
Step 5. Convert the type of the column Year to datetime64
# pd.to_datetime(crime)
crime.Year = pd.to_datetime(crime.Year, format='%Y')
crime.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 55 entries, 0 to 54
Data columns (total 12 columns):
Year                  55 non-null datetime64[ns]
Population            55 non-null int64
Total                 55 non-null int64
Violent               55 non-null int64
Property              55 non-null int64
Murder                55 non-null int64
Forcible_Rape         55 non-null int64
Robbery               55 non-null int64
Aggravated_assault    55 non-null int64
Burglary              55 non-null int64
Larceny_Theft         55 non-null int64
Vehicle_Theft         55 non-null int64
dtypes: datetime64[ns](1), int64(11)
memory usage: 5.2 KB
Step 6. Set the Year column as the index of the dataframe
crime = crime.set_index('Year', drop = True)
crime.head()
Population	Total	Violent	Property	Murder	Forcible_Rape	Robbery	Aggravated_assault	Burglary	Larceny_Theft	Vehicle_Theft
Year											
1960-01-01	179323175	3384200	288460	3095700	9110	17190	107840	154320	912100	1855400	328200
1961-01-01	182992000	3488000	289390	3198600	8740	17220	106670	156760	949600	1913000	336000
1962-01-01	185771000	3752200	301510	3450700	8530	17550	110860	164570	994300	2089600	366800
1963-01-01	188483000	4109500	316970	3792500	8640	17650	116470	174210	1086400	2297800	408300
1964-01-01	191141000	4564600	364220	4200400	9360	21420	130390	203050	1213200	2514400	472800
Step 7. Delete the Total column
del crime['Total']
crime.head()
Population	Violent	Property	Murder	Forcible_Rape	Robbery	Aggravated_assault	Burglary	Larceny_Theft	Vehicle_Theft
Year										
1960-01-01	179323175	288460	3095700	9110	17190	107840	154320	912100	1855400	328200
1961-01-01	182992000	289390	3198600	8740	17220	106670	156760	949600	1913000	336000
1962-01-01	185771000	301510	3450700	8530	17550	110860	164570	994300	2089600	366800
1963-01-01	188483000	316970	3792500	8640	17650	116470	174210	1086400	2297800	408300
1964-01-01	191141000	364220	4200400	9360	21420	130390	203050	1213200	2514400	472800
Step 8. Group the year by decades and sum the values
Pay attention to the Population column number, summing this column is a mistake
# To learn more about .resample (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html)
# To learn more about Offset Aliases (http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)

# Uses resample to sum each decade
crimes = crime.resample('10AS').sum()

# Uses resample to get the max value only for the "Population" column
population = crime['Population'].resample('10AS').max()

# Updating the "Population" column
crimes['Population'] = population

crimes
Population	Violent	Property	Murder	Forcible_Rape	Robbery	Aggravated_assault	Burglary	Larceny_Theft	Vehicle_Theft
1960	201385000	4134930	45160900	106180	236720	1633510	2158520	13321100	26547700	5292100
1970	220099000	9607930	91383800	192230	554570	4159020	4702120	28486000	53157800	9739900
1980	248239000	14074328	117048900	206439	865639	5383109	7619130	33073494	72040253	11935411
1990	272690813	17527048	119053499	211664	998827	5748930	10568963	26750015	77679366	14624418
2000	307006550	13968056	100944369	163068	922499	4230366	8652124	21565176	67970291	11412834
2010	318857056	6072017	44095950	72867	421059	1749809	3764142	10125170	30401698	3569080
Step 9. What is the most dangerous decade to live in the US?
# apparently the 90s was a pretty dangerous time in the US
crime.idxmax(0)
Population            2010
Violent               1990
Property              1990
Murder                1990
Forcible_Rape         1990
Robbery               1990
Aggravated_assault    1990
Burglary              1980
Larceny_Theft         1990
Vehicle_Theft         1990
dtype: int64
"""

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv")

df.info()

df["Year"] = pd.to_datetime(df["Year"], format="%Y")
"""
Sütun türleri arasında "Year" sütununun "int64" olduğunu göreceksiniz. Ancak, zaman serileriyle çalışmak için Pandas'ın farklı bir veri türü olan "datetime64" kullanmak istiyoruz.

Adım 5: "Year" sütununun veri türünü "datetime64" olarak dönüştürelim:

python
Copy code
crime['Year'] = pd.to_datetime(crime['Year'], format='%Y')
Bu kod, "Year" sütununun veri türünü "datetime64" olarak değiştirir. format='%Y' kullanarak, yıl bilgisi olduğunu belirtiyoruz. Dönüşüm tamamlandığında, "Year" sütunu artık bir tarih/saat sütunu olarak işlenecektir.
"""

df.set_index("Year", drop=True)


del df["Year"]
df.head()

df.resample("10Y").sum()
population = df['Population'].resample('10AS').max()

df["Population"] = population
df.idxmax(0)

#####################################  MPG CARS ##################################
"""

MPG Cars
Check out Cars Exercises Video Tutorial to watch a data scientist go through the exercises

Introduction:
The following exercise utilizes data from UC Irvine Machine Learning Repository

Step 1. Import the necessary libraries
import pandas as pd
import numpy as np
Step 2. Import the first dataset cars1 and cars2.
Step 3. Assign each to a to a variable called cars1 and cars2
cars1 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv")
cars2 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv")

print(cars1.head())
print(cars2.head())
    mpg  cylinders  displacement horsepower  weight  acceleration  model  \
0  18.0          8           307        130    3504          12.0     70   
1  15.0          8           350        165    3693          11.5     70   
2  18.0          8           318        150    3436          11.0     70   
3  16.0          8           304        150    3433          12.0     70   
4  17.0          8           302        140    3449          10.5     70   

   origin                        car  Unnamed: 9  Unnamed: 10  Unnamed: 11  \
0       1  chevrolet chevelle malibu         NaN          NaN          NaN   
1       1          buick skylark 320         NaN          NaN          NaN   
2       1         plymouth satellite         NaN          NaN          NaN   
3       1              amc rebel sst         NaN          NaN          NaN   
4       1                ford torino         NaN          NaN          NaN   

   Unnamed: 12  Unnamed: 13  
0          NaN          NaN  
1          NaN          NaN  
2          NaN          NaN  
3          NaN          NaN  
4          NaN          NaN  
    mpg  cylinders  displacement horsepower  weight  acceleration  model  \
0  33.0          4            91         53    1795          17.4     76   
1  20.0          6           225        100    3651          17.7     76   
2  18.0          6           250         78    3574          21.0     76   
3  18.5          6           250        110    3645          16.2     76   
4  17.5          6           258         95    3193          17.8     76   

   origin                 car  
0       3         honda civic  
1       1      dodge aspen se  
2       1   ford granada ghia  
3       1  pontiac ventura sj  
4       1       amc pacer d/l  
Step 4. Oops, it seems our first dataset has some unnamed blank columns, fix cars1
cars1 = cars1.loc[:, "mpg":"car"]
cars1.head()
mpg	cylinders	displacement	horsepower	weight	acceleration	model	origin	car
0	18.0	8	307	130	3504	12.0	70	1	chevrolet chevelle malibu
1	15.0	8	350	165	3693	11.5	70	1	buick skylark 320
2	18.0	8	318	150	3436	11.0	70	1	plymouth satellite
3	16.0	8	304	150	3433	12.0	70	1	amc rebel sst
4	17.0	8	302	140	3449	10.5	70	1	ford torino
Step 5. What is the number of observations in each dataset?
print(cars1.shape)
print(cars2.shape)
(198, 9)
(200, 9)
Step 6. Join cars1 and cars2 into a single DataFrame called cars
cars = cars1.append(cars2)
cars
mpg	cylinders	displacement	horsepower	weight	acceleration	model	origin	car
0	18.0	8	307	130	3504	12.0	70	1	chevrolet chevelle malibu
1	15.0	8	350	165	3693	11.5	70	1	buick skylark 320
2	18.0	8	318	150	3436	11.0	70	1	plymouth satellite
3	16.0	8	304	150	3433	12.0	70	1	amc rebel sst
4	17.0	8	302	140	3449	10.5	70	1	ford torino
5	15.0	8	429	198	4341	10.0	70	1	ford galaxie 500
6	14.0	8	454	220	4354	9.0	70	1	chevrolet impala
7	14.0	8	440	215	4312	8.5	70	1	plymouth fury iii
8	14.0	8	455	225	4425	10.0	70	1	pontiac catalina
9	15.0	8	390	190	3850	8.5	70	1	amc ambassador dpl
10	15.0	8	383	170	3563	10.0	70	1	dodge challenger se
11	14.0	8	340	160	3609	8.0	70	1	plymouth 'cuda 340
12	15.0	8	400	150	3761	9.5	70	1	chevrolet monte carlo
13	14.0	8	455	225	3086	10.0	70	1	buick estate wagon (sw)
14	24.0	4	113	95	2372	15.0	70	3	toyota corona mark ii
15	22.0	6	198	95	2833	15.5	70	1	plymouth duster
16	18.0	6	199	97	2774	15.5	70	1	amc hornet
17	21.0	6	200	85	2587	16.0	70	1	ford maverick
18	27.0	4	97	88	2130	14.5	70	3	datsun pl510
19	26.0	4	97	46	1835	20.5	70	2	volkswagen 1131 deluxe sedan
20	25.0	4	110	87	2672	17.5	70	2	peugeot 504
21	24.0	4	107	90	2430	14.5	70	2	audi 100 ls
22	25.0	4	104	95	2375	17.5	70	2	saab 99e
23	26.0	4	121	113	2234	12.5	70	2	bmw 2002
24	21.0	6	199	90	2648	15.0	70	1	amc gremlin
25	10.0	8	360	215	4615	14.0	70	1	ford f250
26	10.0	8	307	200	4376	15.0	70	1	chevy c20
27	11.0	8	318	210	4382	13.5	70	1	dodge d200
28	9.0	8	304	193	4732	18.5	70	1	hi 1200d
29	27.0	4	97	88	2130	14.5	71	3	datsun pl510
...	...	...	...	...	...	...	...	...	...
170	27.0	4	112	88	2640	18.6	82	1	chevrolet cavalier wagon
171	34.0	4	112	88	2395	18.0	82	1	chevrolet cavalier 2-door
172	31.0	4	112	85	2575	16.2	82	1	pontiac j2000 se hatchback
173	29.0	4	135	84	2525	16.0	82	1	dodge aries se
174	27.0	4	151	90	2735	18.0	82	1	pontiac phoenix
175	24.0	4	140	92	2865	16.4	82	1	ford fairmont futura
176	23.0	4	151	?	3035	20.5	82	1	amc concord dl
177	36.0	4	105	74	1980	15.3	82	2	volkswagen rabbit l
178	37.0	4	91	68	2025	18.2	82	3	mazda glc custom l
179	31.0	4	91	68	1970	17.6	82	3	mazda glc custom
180	38.0	4	105	63	2125	14.7	82	1	plymouth horizon miser
181	36.0	4	98	70	2125	17.3	82	1	mercury lynx l
182	36.0	4	120	88	2160	14.5	82	3	nissan stanza xe
183	36.0	4	107	75	2205	14.5	82	3	honda accord
184	34.0	4	108	70	2245	16.9	82	3	toyota corolla
185	38.0	4	91	67	1965	15.0	82	3	honda civic
186	32.0	4	91	67	1965	15.7	82	3	honda civic (auto)
187	38.0	4	91	67	1995	16.2	82	3	datsun 310 gx
188	25.0	6	181	110	2945	16.4	82	1	buick century limited
189	38.0	6	262	85	3015	17.0	82	1	oldsmobile cutlass ciera (diesel)
190	26.0	4	156	92	2585	14.5	82	1	chrysler lebaron medallion
191	22.0	6	232	112	2835	14.7	82	1	ford granada l
192	32.0	4	144	96	2665	13.9	82	3	toyota celica gt
193	36.0	4	135	84	2370	13.0	82	1	dodge charger 2.2
194	27.0	4	151	90	2950	17.3	82	1	chevrolet camaro
195	27.0	4	140	86	2790	15.6	82	1	ford mustang gl
196	44.0	4	97	52	2130	24.6	82	2	vw pickup
197	32.0	4	135	84	2295	11.6	82	1	dodge rampage
198	28.0	4	120	79	2625	18.6	82	1	ford ranger
199	31.0	4	119	82	2720	19.4	82	1	chevy s-10
398 rows × 9 columns

Step 7. Oops, there is a column missing, called owners. Create a random number Series from 15,000 to 73,000.
nr_owners = np.random.randint(15000, high=73001, size=398, dtype='l')
nr_owners
array([29487, 25680, 65268, 31827, 69215, 72602, 52693, 58440, 16183,
       45014, 32318, 72942, 62163, 35951, 57625, 59355, 36533, 67048,
       58159, 69743, 25146, 22755, 44966, 46792, 56553, 65013, 55908,
       69563, 22030, 59561, 15593, 52998, 54795, 16169, 24809, 35580,
       46590, 38792, 43099, 37166, 21390, 56496, 68606, 21110, 56334,
       45477, 51961, 27625, 51176, 30796, 61809, 65450, 67375, 23342,
       27499, 50585, 57302, 56191, 60281, 32865, 58605, 66374, 15315,
       31791, 28670, 38796, 69214, 41055, 32353, 31574, 65799, 42998,
       72785, 18415, 31977, 29812, 65439, 21161, 60871, 67151, 22179,
       32821, 55392, 34586, 67937, 31646, 66397, 35258, 63815, 71291,
       51130, 27684, 49648, 52691, 50681, 68185, 32635, 51553, 28970,
       19112, 26035, 67666, 55471, 51477, 62055, 53003, 41265, 18565,
       48851, 48673, 45832, 67891, 57638, 29240, 41236, 16950, 31449,
       50528, 22397, 15876, 26414, 16736, 23896, 46104, 17583, 65951,
       38538, 31443, 19299, 46095, 31239, 19290, 38051, 68575, 61755,
       22560, 34460, 35395, 34608, 56906, 44895, 48429, 20900, 49770,
       50513, 59402, 26893, 37233, 19036, 20523, 18765, 46333, 42831,
       53698, 25218, 63106, 16928, 34901, 43674, 65453, 54428, 68502,
       19043, 20325, 45039, 29466, 49672, 67972, 30547, 22522, 69354,
       40489, 72887, 15724, 51442, 65182, 64555, 42138, 72988, 20861,
       67898, 20768, 36415, 47480, 16820, 48739, 62610, 43473, 23002,
       43488, 62581, 37724, 63019, 44912, 35595, 59188, 51814, 65283,
       53479, 27660, 38237, 22957, 47870, 15533, 41944, 51830, 56676,
       57481, 48529, 72220, 66675, 50099, 30585, 25436, 49195, 26050,
       24899, 37213, 25870, 67447, 23808, 71275, 67572, 18545, 43553,
       54858, 23077, 33705, 31282, 26298, 23742, 36110, 51491, 18019,
       60655, 27453, 35563, 63627, 35315, 56717, 59281, 55634, 18415,
       59570, 47320, 20110, 18425, 19352, 18032, 31816, 28573, 66030,
       54723, 21592, 37160, 59518, 35629, 47619, 52359, 34566, 64932,
       24072, 39445, 31203, 63975, 62041, 70175, 51029, 32058, 19428,
       65553, 50799, 48190, 68061, 68201, 53389, 15901, 44585, 54723,
       30446, 63716, 57488, 67134, 22033, 53694, 40002, 24854, 59747,
       59827, 53378, 53196, 68686, 20784, 28181, 33044, 41694, 39857,
       57296, 69021, 17359, 29794, 22515, 55877, 22806, 50027, 56787,
       50844, 17420, 65259, 19141, 40204, 19530, 30116, 34973, 15641,
       53492, 59574, 59082, 64400, 70163, 43058, 69696, 67996, 26158,
       32936, 45461, 47390, 32368, 15400, 40895, 16572, 31776, 62121,
       56704, 39335, 27716, 52565, 50831, 45049, 25173, 25018, 18606,
       71177, 66288, 46754, 68175, 35829, 24959, 54792, 19059, 29092,
       58736, 62938, 44733, 17884, 33905, 33965, 24641, 52257, 28178,
       29515, 37703, 56036, 51556, 23590, 61888, 70224, 53730, 41328,
       16501, 30360, 54106, 29101, 35631, 56173, 30424, 46887, 23657,
       17723, 71709, 45270, 30380, 27779, 33774, 36379, 47127, 63625,
       16750, 65740, 53802, 40995, 37487, 42791, 21825, 69344, 63210,
       15982, 20259])
Step 8. Add the column owners to cars
cars['owners'] = nr_owners
cars.tail()
mpg	cylinders	displacement	horsepower	weight	acceleration	model	origin	car	owners
195	27.0	4	140	86	2790	15.6	82	1	ford mustang gl	21825
196	44.0	4	97	52	2130	24.6	82	2	vw pickup	69344
197	32.0	4	135	84	2295	11.6	82	1	dodge rampage	63210
198	28.0	4	120	79	2625	18.6	82	1	ford ranger	15982
199	31.0	4	119	82	2720	19.4	82	1	chevy s-10	20259"""

import pandas as pd
import numpy as np

df1 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv")

df2 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv")

df1 = df1.loc[:, ~df1.columns.str.contains("Unnamed")]
df1.head()

df1.shape[0]
df2.shape[0]


df3 = pd.concat([df1,df2])
df3.info()

nr_owners = np.random.randint(15000, high=73001, size=398, dtype="l")
"""
dtype='l' ifadesi NumPy kütüphanesinde kullanılan bir parametredir ve verilerin hangi türde saklanacağını belirtir. dtype parametresi, veri türünü (data type) belirlemek için kullanılır. 'l' burada bir veri türünü temsil eder.

'l' (küçük "L") veri türü, "long integer" yani uzun tam sayı veri türünü temsil eder. Bu, büyük tam sayıları saklamak için kullanılır ve genellikle Python'daki int veri türünden daha büyük değerlerle çalışmak için kullanılır.

Örneğinizde np.random.randint işlevi ile rastgele uzun tam sayılar üretiliyor ve bu tam sayılar 'l' veri türünde saklanıyor. Bu, büyük aralıklardaki tam sayıları saklamak için kullanışlıdır."""

df3.head()

df3["owners"] = nr_owners
df3.head()


#####################################  Fictitious Names  ##################################
"""

Fictitious Names
Check out Fictitious Names Exercises Video Tutorial to watch a data scientist go through the exercises

Introduction:
This time you will create a data again

Special thanks to Chris Albon for sharing the dataset and materials. All the credits to this exercise belongs to him.

In order to understand about it go to here.

Step 1. Import the necessary libraries
import pandas as pd
Step 2. Create the 3 DataFrames based on the following raw data
raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
Step 3. Assign each to a variable called data1, data2, data3
data1 = pd.DataFrame(raw_data_1, columns = ['subject_id', 'first_name', 'last_name'])
data2 = pd.DataFrame(raw_data_2, columns = ['subject_id', 'first_name', 'last_name'])
data3 = pd.DataFrame(raw_data_3, columns = ['subject_id','test_id'])

data3
subject_id	test_id
0	1	51
1	2	15
2	3	15
3	4	61
4	5	16
5	7	14
6	8	15
7	9	1
8	10	61
9	11	16
Step 4. Join the two dataframes along rows and assign all_data
all_data = pd.concat([data1, data2])
all_data
subject_id	first_name	last_name
0	1	Alex	Anderson
1	2	Amy	Ackerman
2	3	Allen	Ali
3	4	Alice	Aoni
4	5	Ayoung	Atiches
0	4	Billy	Bonder
1	5	Brian	Black
2	6	Bran	Balwner
3	7	Bryce	Brice
4	8	Betty	Btisan
Step 5. Join the two dataframes along columns and assing to all_data_col
all_data_col = pd.concat([data1, data2], axis = 1)
all_data_col
subject_id	first_name	last_name	subject_id	first_name	last_name
0	1	Alex	Anderson	4	Billy	Bonder
1	2	Amy	Ackerman	5	Brian	Black
2	3	Allen	Ali	6	Bran	Balwner
3	4	Alice	Aoni	7	Bryce	Brice
4	5	Ayoung	Atiches	8	Betty	Btisan
Step 6. Print data3
data3
subject_id	test_id
0	1	51
1	2	15
2	3	15
3	4	61
4	5	16
5	7	14
6	8	15
7	9	1
8	10	61
9	11	16
Step 7. Merge all_data and data3 along the subject_id value
pd.merge(all_data, data3, on='subject_id')
subject_id	first_name	last_name	test_id
0	1	Alex	Anderson	51
1	2	Amy	Ackerman	15
2	3	Allen	Ali	15
3	4	Alice	Aoni	61
4	4	Billy	Bonder	61
5	5	Ayoung	Atiches	16
6	5	Brian	Black	16
7	7	Bryce	Brice	14
8	8	Betty	Btisan	15
Step 8. Merge only the data that has the same 'subject_id' on both data1 and data2
pd.merge(data1, data2, on='subject_id', how='inner')
subject_id	first_name_x	last_name_x	first_name_y	last_name_y
0	4	Alice	Aoni	Billy	Bonder
1	5	Ayoung	Atiches	Brian	Black
Step 9. Merge all values in data1 and data2, with matching records from both sides where available.
pd.merge(data1, data2, on='subject_id', how='outer')
subject_id	first_name_x	last_name_x	first_name_y	last_name_y
0	1	Alex	Anderson	NaN	NaN
1	2	Amy	Ackerman	NaN	NaN
2	3	Allen	Ali	NaN	NaN
3	4	Alice	Aoni	Billy	Bonder
4	5	Ayoung	Atiches	Brian	Black
5	6	NaN	NaN	Bran	Balwner
6	7	NaN	NaN	Bryce	Brice
7	8	NaN	NaN	Betty	Btisan"""
import pandas as pd

raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}

data1 = pd.DataFrame(raw_data_1, columns = ['subject_id', 'first_name', 'last_name'])
data2 = pd.DataFrame(raw_data_2, columns = ['subject_id', 'first_name', 'last_name'])
data3 = pd.DataFrame(raw_data_3, columns = ['subject_id','test_id'])


all_data = pd.concat([data1, data2])

all_data_col = pd.concat([data1, data2], axis=1)
all_data_col

data3

 pd.merge(all_data, data3, on="subject_id")

pd.merge(data1, data2, on="subject_id", how="outer")

pd.merge(data1, data2, on='subject_id', how='inner')


#####################################  TİPS ##################################
"""

Tips
Check out Tips Visualization Exercises Video Tutorial to watch a data scientist go through the exercises

Introduction:
This exercise was created based on the tutorial and documentation from Seaborn
The dataset being used is tips from Seaborn.

Step 1. Import the necessary libraries:
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# print the graphs in the notebook
% matplotlib inline

# set seaborn style to white
sns.set_style("white")
Step 2. Import the dataset from this address.
Step 3. Assign it to a variable called tips
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Tips/tips.csv'
tips = pd.read_csv(url)

tips.head()
Unnamed: 0	total_bill	tip	sex	smoker	day	time	size
0	0	16.99	1.01	Female	No	Sun	Dinner	2
1	1	10.34	1.66	Male	No	Sun	Dinner	3
2	2	21.01	3.50	Male	No	Sun	Dinner	3
3	3	23.68	3.31	Male	No	Sun	Dinner	2
4	4	24.59	3.61	Female	No	Sun	Dinner	4
Step 4. Delete the Unnamed 0 column
del tips['Unnamed: 0']

tips.head()
total_bill	tip	sex	smoker	day	time	size
0	16.99	1.01	Female	No	Sun	Dinner	2
1	10.34	1.66	Male	No	Sun	Dinner	3
2	21.01	3.50	Male	No	Sun	Dinner	3
3	23.68	3.31	Male	No	Sun	Dinner	2
4	24.59	3.61	Female	No	Sun	Dinner	4
Step 5. Plot the total_bill column histogram
# create histogram
ttbill = sns.distplot(tips.total_bill);

# set lables and titles
ttbill.set(xlabel = 'Value', ylabel = 'Frequency', title = "Total Bill")

# take out the right and upper borders
sns.despine()

Step 6. Create a scatter plot presenting the relationship between total_bill and tip
sns.jointplot(x ="total_bill", y ="tip", data = tips)
<seaborn.axisgrid.JointGrid at 0x1197d84d0>

Step 7. Create one image with the relationship of total_bill, tip and size.
Hint: It is just one function.
sns.pairplot(tips)
<seaborn.axisgrid.PairGrid at 0x11844c090>

Step 8. Present the relationship between days and total_bill value
sns.stripplot(x = "day", y = "total_bill", data = tips, jitter = True);

Step 9. Create a scatter plot with the day as the y-axis and tip as the x-axis, differ the dots by sex
sns.stripplot(x = "tip", y = "day", hue = "sex", data = tips, jitter = True);

Step 10. Create a box plot presenting the total_bill per day differetiation the time (Dinner or Lunch)
sns.boxplot(x = "day", y = "total_bill", hue = "time", data = tips);

Step 11. Create two histograms of the tip value based for Dinner and Lunch. They must be side by side.
# better seaborn style
sns.set(style = "ticks")

# creates FacetGrid
g = sns.FacetGrid(tips, col = "time")
g.map(plt.hist, "tip");

Step 12. Create two scatterplots graphs, one for Male and another for Female, presenting the total_bill value and tip relationship, differing by smoker or no smoker
They must be side by side.
g = sns.FacetGrid(tips, col = "sex", hue = "smoker")
g.map(plt.scatter, "total_bill", "tip", alpha =.7)

g.add_legend();

BONUS: Create your own question and answer it using a graph.
 """

import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sns.set_style("white")

df = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Tips/tips.csv")


del df["Unnamed: 0"]
df.head()

ttbill = sns.displot(df.total_bill)

ttbill.set(xlabel= "Value", ylabel = "Frequency", title= "Total_Bill")
sns.despine()

plt.show()


sns.jointplot(x = "total_bill", y = "tip", data = df)
plt.show()


df.head()

sns.pairplot(df)
plt.show()

sns.barplot(x="day", y="total_bill", data=df)
plt.show()

sns.stripplot(x="tip", y="day", hue="sex", data=df, jitter=True)
plt.show()

sns.boxplot(y="total_bill", x="day", hue="time", data=df)
plt.show()

plt.subplot(1,2,1)
plt.plot(df["tip"], df["time"] == "Dinner")
plt.show()
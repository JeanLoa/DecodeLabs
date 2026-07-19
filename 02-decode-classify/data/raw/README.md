# Raw data

`iris.csv` is generated from `sklearn.datasets.load_iris` and contains the 150
canonical Iris observations: 50 setosa, 50 versicolor and 50 virginica.

Its model contract is:

- `sepal_length_cm`
- `sepal_width_cm`
- `petal_length_cm`
- `petal_width_cm`
- `species`

The four measurements are expressed in centimeters. The target uses readable
species names instead of scikit-learn's numeric class identifiers.

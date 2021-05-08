{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4a4a7a83",
   "metadata": {},
   "source": [
    "## The Sparks Foundation: Task 2- Unsupervised Learning (K Means Clustering) - Iris Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aa761b35",
   "metadata": {},
   "source": [
    "## K- Means Clustering Assignment - 2 \n",
    "                  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a0fee5b",
   "metadata": {},
   "source": [
    "### Author : Syed Mukarram"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea1ebc39",
   "metadata": {},
   "source": [
    "###  1) Importing the libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "49fa02e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing the libraries\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "%matplotlib inline\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8f6cde2",
   "metadata": {},
   "source": [
    "### 2) Importing the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7e4b4bd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "Iris =pd.read_csv('Iris.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "dec4c442",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <th>PetalWidthCm</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>146</td>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>147</td>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>148</td>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>149</td>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>150</td>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  \\\n",
       "0      1            5.1           3.5            1.4           0.2   \n",
       "1      2            4.9           3.0            1.4           0.2   \n",
       "2      3            4.7           3.2            1.3           0.2   \n",
       "3      4            4.6           3.1            1.5           0.2   \n",
       "4      5            5.0           3.6            1.4           0.2   \n",
       "..   ...            ...           ...            ...           ...   \n",
       "145  146            6.7           3.0            5.2           2.3   \n",
       "146  147            6.3           2.5            5.0           1.9   \n",
       "147  148            6.5           3.0            5.2           2.0   \n",
       "148  149            6.2           3.4            5.4           2.3   \n",
       "149  150            5.9           3.0            5.1           1.8   \n",
       "\n",
       "            Species  \n",
       "0       Iris-setosa  \n",
       "1       Iris-setosa  \n",
       "2       Iris-setosa  \n",
       "3       Iris-setosa  \n",
       "4       Iris-setosa  \n",
       "..              ...  \n",
       "145  Iris-virginica  \n",
       "146  Iris-virginica  \n",
       "147  Iris-virginica  \n",
       "148  Iris-virginica  \n",
       "149  Iris-virginica  \n",
       "\n",
       "[150 rows x 6 columns]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Iris"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "17f56ffb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 150 entries, 0 to 149\n",
      "Data columns (total 6 columns):\n",
      " #   Column         Non-Null Count  Dtype  \n",
      "---  ------         --------------  -----  \n",
      " 0   Id             150 non-null    int64  \n",
      " 1   SepalLengthCm  150 non-null    float64\n",
      " 2   SepalWidthCm   150 non-null    float64\n",
      " 3   PetalLengthCm  150 non-null    float64\n",
      " 4   PetalWidthCm   150 non-null    float64\n",
      " 5   Species        150 non-null    object \n",
      "dtypes: float64(4), int64(1), object(1)\n",
      "memory usage: 7.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "1bd44e30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <th>PetalWidthCm</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>75.500000</td>\n",
       "      <td>5.843333</td>\n",
       "      <td>3.054000</td>\n",
       "      <td>3.758667</td>\n",
       "      <td>1.198667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>43.445368</td>\n",
       "      <td>0.828066</td>\n",
       "      <td>0.433594</td>\n",
       "      <td>1.764420</td>\n",
       "      <td>0.763161</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.300000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>38.250000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>2.800000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>0.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>75.500000</td>\n",
       "      <td>5.800000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.350000</td>\n",
       "      <td>1.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>112.750000</td>\n",
       "      <td>6.400000</td>\n",
       "      <td>3.300000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>1.800000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>150.000000</td>\n",
       "      <td>7.900000</td>\n",
       "      <td>4.400000</td>\n",
       "      <td>6.900000</td>\n",
       "      <td>2.500000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm\n",
       "count  150.000000     150.000000    150.000000     150.000000    150.000000\n",
       "mean    75.500000       5.843333      3.054000       3.758667      1.198667\n",
       "std     43.445368       0.828066      0.433594       1.764420      0.763161\n",
       "min      1.000000       4.300000      2.000000       1.000000      0.100000\n",
       "25%     38.250000       5.100000      2.800000       1.600000      0.300000\n",
       "50%     75.500000       5.800000      3.000000       4.350000      1.300000\n",
       "75%    112.750000       6.400000      3.300000       5.100000      1.800000\n",
       "max    150.000000       7.900000      4.400000       6.900000      2.500000"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "e20101f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Dividig this into Independent and dependent features\n",
    "x=Iris.iloc[:, [1,4]].values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "32601000",
   "metadata": {},
   "source": [
    "### 3) Using the elbow method to find the optmimal number of clusters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "325a83bb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.cluster import KMeans\n",
    "wcss=[]\n",
    "for i in range(1, 11):\n",
    "    kmeans = KMeans(n_clusters = i, init ='k-means++', random_state =42)\n",
    "    kmeans.fit(x)\n",
    "    wcss.append(kmeans.inertia_)\n",
    "plt.plot(range(1, 11), wcss)\n",
    "plt.title('The elbow Method')\n",
    "plt.xlabel('Number of clusters')\n",
    "plt.ylabel('WCSS')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c06e5b59",
   "metadata": {},
   "source": [
    "### 4) Training the KMeans model on the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "92f8862a",
   "metadata": {},
   "outputs": [],
   "source": [
    "kmeans = KMeans(n_clusters = 3, init ='k-means++', random_state = 42)\n",
    "y_kmeans=kmeans.fit_predict(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "5ccf1a1f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 1 1 1 1 0 2 0 2 0 2 2 1 0 2 1 2 2 2 2 0 2 2 2 2 2 2 2 2\n",
      " 2 0 0 0 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 1 2 2 2 2 1 2 0 2 0 2 0 0 2 0 0 0 0\n",
      " 0 0 2 2 0 0 0 0 2 0 2 0 2 0 0 2 2 0 0 0 0 0 2 2 0 0 0 2 0 0 0 2 0 0 0 0 0\n",
      " 0 2]\n"
     ]
    }
   ],
   "source": [
    "print(y_kmeans)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "83828fbd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x[y_kmeans ==0, 0], x[y_kmeans==0, 1], s=100, c='red', label ='Iris-setosa')\n",
    "plt.scatter(x[y_kmeans ==1, 0], x[y_kmeans==1, 1], s=100, c='blue', label ='Iris-versicolour')\n",
    "plt.scatter(x[y_kmeans ==2, 0], x[y_kmeans==2, 1], s=100, c='green', label ='Iris-virginica')\n",
    "plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='pink', label ='Centroids')\n",
    "plt.title('Cluster of Iris Data')\n",
    "plt.xlabel('Sepal Length', fontsize =18)\n",
    "plt.ylabel('Sepal Width', fontsize =18)\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4b9ee27",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

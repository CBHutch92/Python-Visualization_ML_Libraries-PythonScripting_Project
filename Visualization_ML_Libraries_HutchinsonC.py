# Assignment 5
# Casey Hutchinson
# ITEC 3100
# Due by 11/3/2024
# I did not include the function Scikit_Tutorial() - which contains the code from the Sckikit Tutorial - in my main function

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np

# Part A
def age_weight_plot():
    print("Hello there! Today, I'm going to ask you for 10 heights and weights. Then, We're going to plot them on a graph you help me design.")

    # gets 10 ages as input from the user and stores it a list labeled "ages" (Step 1)
    ages = []
    for i in range(10):
        while True:
            try:
                user_ages = int(input(f"Please enter the age in years for person number {i + 1} (No decimals): "))
                if user_ages > 0:
                    ages.append(user_ages)
                    break
                else:
                    ValueError
                    print("Invalid input. Please enter a valid number value with no digits.")
            except ValueError:
                print("Invalid input. Please enter a valid number value with no digits.")
    print("\n")

    # gets 10 weights as input from the user and stores it a list labeled "weights" (Step 1)
    weights = []
    for i in range(10):
        while True:
            try:
                user_weights = int(input(f"Now, please enter the weight in pounds for person number {i + 1} (No decimals): "))
                if user_weights > 0:
                    weights.append(user_weights)
                    break
                else:
                    ValueError
                    print("Invalid input. Please enter a valid number value with no digits.")
            except ValueError:
                print("Invalid input. Please enter a valid number value with no digits.")
    print("\n")
    print("Now that we have our height and weight data, we can plot them onto our graph.")
    plt.plot(ages, weights)
    plt.show()
    print("\n")

    # this show's the graph as a default before we get the users choices. I know this part wasn't required but I felt like it worked with the flow of the program.
    print("That was our current graph, but let's see if we can make it look a little nicer.")
    print("\n")
    print("\n")

    # asks the user for input to select a marker style from a list of options for the updated graph (Step 3)
    while True:
        print("Use the list below to choose what type of marker you'd like to use to plot each point along the graph.")
        print("'.' to use a point marker")
        print("',' to use a pixel marker")
        print("'o' to use a circle marker")
        print("'v' to use a triangle down marker")
        print("'^' to use a triangle up marker")
        print("'<' to use a triangle left marker")
        print("'>' to use a triangle right marker")
        print("'1' to use a tri down marker")
        print("'2' to use a tri up marker")
        print("'3' to use a tri left marker")
        print("'4' to use a tri right marker")
        print("'s' to use a square marker")
        print("'p' to use a pentagon marker")
        print("'*' to use a star marker")
        print("'h' to use the hexagon1 marker")
        print("'H' to use the hexagon2 marker")
        print("'+' to use a plus marker")
        print("'x' to use an x marker")
        print("'D' to use a diamond marker")
        print("'d' to use a thin diamond marker")
        print("'|' to use a vline marker")
        print("'_' to use an hline marker")
        print("Type in the character in the quotation marks that corresponds with your marker choice below.")
        marker_choice = input("Which type of marker would you like to use for our graph? ")
        print("\n")
        marker_options = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        if marker_choice in marker_options:
            user_marker = marker_choice
            break
        else:
            print("Invalid input. Please look at the list and enter an appropriate option.")
            print("\n")
    print("\n")

    # asks the user for input to select a line style from a list of options for the updated graph (Step 4)
    print("Now let's select the style of the line for our graph.")
    while True:
        print("Use the list below to choose what type of line style you'd like to use for our graph.")
        print("'-' to use a solid line")
        print("'--' to use a dashed line")
        print("'-.' to use a dash-dot line")
        print("':' to use a dotted line")
        print("Type in the character in the quotation marks that corresponds with your line style choice below.")
        line_choice = input("Which line style would you like to use for our graph? ")
        print("\n")
        line_options = ['-', '--', '-.', ':']
        if line_choice in line_options:
            user_line = line_choice
            break
        else:
            print("Invalid input. Please look at the list and enter an appropriate option.")
            print("\n")
    print("\n")

    # asks the user for input to change the color of the graph from a list of options for the updated graph (Step 5)
    print("Lastly, let's decide what the color of the graph will be.")
    while True:
        print("Use the list below to choose what color you'd like the graph to be.")
        print("'b' for blue")
        print("'g' for green")
        print("'r' for red")
        print("'c' for cyan")
        print("'m' for magenta")
        print("'y' for yellow")
        print("'k' for black")
        print("'w' for white")
        print("Type in the letter in the quotation marks that corresponds with your color choice below.")
        color_choice = input("Which color would you like to use for our graph? ")
        print("\n")
        color_options = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        if color_choice in color_options:
            user_color = color_choice
            break
        else:
            print("Invalid input. Please look at the list and enter an appropriate option.")
            print("\n")
    print("\n")
    print("Now that you've helped me with some choices, lets take a look at our new graph.")

    # I labeled the data in a dataframe to call them like this when plotting cause I wasn't sure if you wanted this kind of label or the labels below (Step 6)
    data = {"Age": ages,
            "Weight": weights}
    age_weight_df = pd.DataFrame(data)
    plt.plot('Age', 'Weight', data=age_weight_df, marker=f'{user_marker}', linestyle=f'{user_line}', color=f'{user_color}')                 # plots age versus weight on a graph using matplotlib (Step 2)
    plt.xlabel('Ages')                                                      
    plt.ylabel('Weights')
    plt.title('Age/Weight graph (info from user)')                                                                                          # x and y labels were included as well as a graph title (Step 6)
    plt.show()
    print("\n")
    print("Thanks for hepling me build a graph!")



# Part B
def seaborn_tips():

    # loaded tips.csv into dataset called tips_data (Step 1)
    tips_data = pd.read_csv('tips.csv')
    while True:
        print("\n")
        print("\n")
        print("Now let's look at a different set of data.")
        print("I've uploaded a dataset that contains 244 rows of data with the following 7 categories: total bill, tip, sex, smoker(y/n), day, time, and size.")
        print("You can see a snippet below:")
        print(tips_data.head())

        # asks the user to select which categories they'd like to use for the graph (Step 2)
        print("Now that we know what categories we have, I'd like you to help me plot out the data. We can use any of the available categories, themes, and scale for the plot that you'd like.")
        print("First tell me what categories you'd like to use on the x-axis and y-axis by choosing from the options below:")
        print("\n")
        print("total_bill")
        print("tip")
        print("sex")
        print("smoker")
        print("day")
        print("time")
        print("size")
        print("Type in the category name as it appears above to make your axis choices.")
        x_choice = input("Which category would you like to use as your x-axis? ")
        axis_options = ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
        if x_choice in axis_options:
            x_axis = x_choice
            y_choice = input("Which category would you like to use as your y-axis? ")
            if y_choice == x_choice:
                print("Invalid input. Your x-axis and y-axis can not be the same category. Please try again.")
            elif y_choice in axis_options:
                y_axis = y_choice
                break
            else:
                print("Invalid input. Please look at the list and enter an appropriate option.")
        else:
            print("Invalid input. Please look at the list and enter an appropriate option.")
            print("\n")
    print("\n")

    # asks the user to select which style theme they'd like to use (Step 3)
    print("Now that you've selected the categories to use for our plot, lets select a theme.")
    while True:
        print("The available theme options are listed below:")
        print("\n")
        print("darkgrid")
        print("whitegrid")
        print("dark")
        print("white")
        print("ticks")
        print("Type in the theme as it appears above to make your choice.")
        theme_options = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
        theme_choice = input("Which theme would you like to use for the plot? ")
        if theme_choice in theme_options:
            theme = theme_choice
            break
        else:
            print("Invalid input. Please look at the list and enter an appropriate option.")
            print("\n")
    print("\n")

    # asks the user to select the scale they'd like to use (Step 4)
    print("Lastly, lets select the scale for our graph.")
    while True:
        print("The available color options are listed below.")
        print("\n")
        print("paper")
        print("notebook")
        print("talk")
        print("poster")
        print("Type in the scale as it appears above to make your choice.")
        scale_options = ['paper', 'notebook', 'talk', 'poster']
        scale_choice = input("Which scale would you like to use for our graph? ")
        if scale_choice in scale_options:
            scale = scale_choice
            break
        else:
            print("Invalid input. Please look at the list and enter an appropriate option.")
    print("\n")

    # use Seaborn to plot the results based on the user selections of columns, theme, and scale and labels the data appropriately (Step 2 and 5)
    sns.set_style(f'{theme}')
    sns.set_context(f'{scale}')
    sns.relplot(x=f'{x_axis}', y=f'{y_axis}', data=tips_data)
    plt.title(f"{x_axis}/{y_axis} graph based on data from tips.csv")
    plt.show()
    print("\n")
    print("Thanks for hepling me build a graph!")

def main():
    age_weight_plot()
    seaborn_tips()

if __name__ == "__main__":
    main()





# This is the code from following the Scikit LEarn Tutorial
def Scikit_Tutorial():
    # Modelling Process
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    # Part 1
    print("Dataset Loading")
    iris = load_iris()
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    print("Feature name:", feature_names)
    print("Target names:", target_names)
    print("\nFirst 10 rows of X:\n", x[:10])
    print("\n")

    # Part 2
    print("Splitting the dataset")
    iris = load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print("\n")

    # Part 3
    print("Train the Model")
    iris = load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=1)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    classifier_knn = KNeighborsClassifier(n_neighbors = 3)
    classifier_knn.fit(x_train, y_train)
    y_pred = classifier_knn.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Providing sample data - the model will make a prediction out of that data
    sample = [[5, 5, 3, 2], [2, 4, 3, 5]]
    preds = classifier_knn.predict(sample)
    pred_species = [iris.target_names[p] for p in preds] 
    print("Predictions:", pred_species)
    print("\n")

    # Part 4
    print("Preprocessing the Data")
    Input_data = np.array(
        [[2.1, -1.9, 5.5],
        [-1.5, 2.4, 3.5],
        [0.5, -7.9, 5.6],
        [5.9, 2.3, -5.8]]
    )
    data_binarized = preprocessing.Binarizer(threshold=0.5).transform(Input_data)
    print("\nBinarized data:\n", data_binarized)
    print("\n")

    # Part 5
    print("Mean Removal")
    Input_data = np.array(
        [[2.1, -1.9, 5.5],
        [-1.5, 2.4, 3.5],
        [0.5, -7.9, 5.6],
        [5.9, 2.3, -5.8]]
    )
    # This displays the mean and standard deviation of the data
    print("Mean =", Input_data.mean(axis=0))
    print("Standard deviation =", Input_data.std(axis=0))
    # This removes the mean and the standard deviation from the data
    data_scaled = preprocessing.scale(Input_data)
    print("Mean removed =", data_scaled.mean(axis=0))
    print("Standard deviation removed =", data_scaled.std(axis=0))
    print("\n")

    # Part 6
    print("Scaling")
    Input_data = np.array(
        [[2.1, -1.9, 5.5],
        [-1.5, 2.4, 3.5],
        [0.5, -7.9, 5.6],
        [5.9, 2.3, -5.8]]
    )
    data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
    data_scaled_minmax = data_scaler_minmax.fit_transform(Input_data)
    print("\nMin max scaled data:\n", data_scaled_minmax)
    print("\n")

    # Part 7
    print("Normalisation")
    # L1 Normalisation
    print("L1 Normalisation")
    Input_data = np.array(
        [[2.1, -1.9, 5.5],
        [-1.5, 2.4, 3.5],
        [0.5, -7.9, 5.6],
        [5.9, 2.3, -5.8]]
    )
    data_normalized_l1 = preprocessing.normalize(Input_data, norm='l1')
    print("\nL1 Normalized data:\n", data_normalized_l1)
    print("\n")
    print("L2 Normalisation")
    Input_data = np.array(
        [[2.1, -1.9, 5.5],
        [-1.5, 2.4, 3.5],
        [0.5, -7.9, 5.6],
        [5.9, 2.3, -5.8]]
    )
    data_normalized_l2 = preprocessing.normalize(Input_data, norm='l2')
    print("\nL1 Normalized data:\n", data_normalized_l2)
    print("\n")

    

    # Data Representation
    # Part 1
    print("Data as a table")
    iris = sns.load_dataset('iris')
    print(iris.head())
    print("\n")

    # Part 2
    print("Data as Target array (view graphs):")
    iris = sns.load_dataset('iris')
    sns.set()
    sns.pairplot(iris, hue='species', height=3);
    plt.show()
    print("\n")



    # Estimator API
    # Part 1
    print("Complete working/executable example 1:")
    iris = sns.load_dataset('iris')
    X_iris = iris.drop('species', axis = 1)
    X_iris.shape
    y_iris = iris['species']
    y_iris.shape

    rng = np.random.RandomState(35)
    x = 10*rng.rand(40)
    y = 2*x-1+rng.randn(40)
    plt.scatter(x,y);
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True)
    model
    X = x[:, np.newaxis]
    X.shape

    model.fit(X, y)
    model.coef_
    model.intercept_

    xfit = np.linspace(-1, 11)
    Xfit = xfit[:, np.newaxis]
    yfit = model.predict(Xfit)
    plt.scatter(x, y)
    plt.plot(xfit, yfit);
    plt.show()
    print("\n")

    # Part 2
    print("Complete working/executable example 2:")
    iris2 = sns.load_dataset('iris')
    X_iris = iris2.drop('species', axis = 1)
    X_iris.shape
    y_iris = iris2['species']
    y_iris.shape
    rng = np.random.RandomState(35)
    x = 10*rng.rand(40)
    y = 2*x-1+rng.randn(40)
    plt.scatter(x,y);
    from sklearn.decomposition import PCA

    model = PCA(n_components=2)
    model
    model.fit(X_iris)
    X_2D = model.transform(X_iris)
    iris2['PCA1'] = X_2D[:, 0]
    iris2['PCA2'] = X_2D[:, 1]
    sns.lmplot(x="PCA1", y="PCA2", hue='species', data=iris2, fit_reg=False);
    plt.show()
    print("\n")



    # Conventions
    # Part 1
    print("Type casting")
    from sklearn import random_projection
    range = np.random.RandomState(0)
    x = range.rand(10, 2000)
    x = np.array(x, dtype = 'float32')
    x.dtype
    Transformer_data = random_projection.GaussianRandomProjection()
    x_new = Transformer_data.fit_transform(x)
    print(x_new.dtype)
    print("\n")

    # Part 2
    print("Refitting & Updating Parameters")
    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    iris = load_iris()
    x, y = iris.data, iris.target
    clf = SVC(kernel='linear')
    clf.fit(x, y)
    print(clf.predict(x[:5]))
    print("\n")

    # Part 3
    print("Multiclass & Multilabel fitting")
    print("Example 1 - one dimensional array of multiclass labels")
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import LabelBinarizer
    x = [[1, 2], [3, 4], [4, 5], [5, 2], [1, 1]]
    y = [0, 0, 1, 1, 2]
    classif = OneVsRestClassifier(estimator = SVC(gamma = 'scale', random_state = 0))
    print(classif.fit(x, y).predict(x))
    print("\n")

    print("Example 2 - two dimensional array")
    x = [[1, 2], [3, 4], [4, 5], [5, 2], [1, 1]]
    y = LabelBinarizer().fit_transform(y)
    print(classif.fit(x, y).predict(x))
    print("\n")

    print("Example 3 - multiple labels")
    from sklearn.preprocessing import MultiLabelBinarizer
    y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
    y = MultiLabelBinarizer().fit_transform(y)
    print(classif.fit(x, y).predict(x))
    print("\n")



    # Extended Linear Modeling
    # Part 1
    print("Implementation Example")
    from sklearn.preprocessing import PolynomialFeatures
    y = np.arange(8).reshape(4, 2)
    poly = PolynomialFeatures(degree=2)
    print(poly.fit_transform(y))
    print("\n")

    # Part 2
    print("Streamlining using Pipeline tools")
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
    x = np.arange(5)
    y = 3 - 2 * x + x ** 2 - x ** 3
    Stream_model = model.fit(x[:, np.newaxis], y)
    print(Stream_model.named_steps['linear'].coef_)
    print("\n")



    # Stochastic Gradient Descent
    # Part 1
    print("SGD Classifier")
    # SGD Classifier linear model
    from sklearn import linear_model
    x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    SGDClf = linear_model.SGDClassifier(max_iter = 1000, tol=1e-3, penalty = "elasticnet")
    print(SGDClf.fit(x, y))
    # Now that the model is fitted, it can predict new values
    print(SGDClf.predict([[2.,2.]]))
    # get the weight vector
    print(SGDClf.coef_)
    # value of the intercept
    print(SGDClf.intercept_)
    # signed distance to the hyperplane using SGDClassifier.decision_function
    print(SGDClf.decision_function([[2., 2.]]))
    print("\n")

    # Part 2
    print("SGD Regressor")
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y = rng.randn(n_samples)
    x = rng.randn(n_samples, n_features)
    SGDReg =linear_model.SGDRegressor(
    max_iter = 1000,penalty = "elasticnet",loss = 'huber',tol = 1e-3, average = True)
    print(SGDReg.fit(x, y))
    # now that the model is fitted, we can get the weight vector
    print(SGDReg.coef_)
    # value of the intercept
    print(SGDReg.intercept_)
    # number of weight updates during training phase
    print(SGDReg.t_)



    # Support Vector Machines
    # Part 1
    print("SVC")
    x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    from sklearn.svm import SVC
    SVCClf = SVC(kernel = 'linear', gamma = 'scale', shrinking = False,)
    print(SVCClf.fit(x, y))
    # now that model is fitted, we can get weight vector
    print(SVCClf.coef_)
    # using model to get other values:
    print(SVCClf.predict([[-0.5,-0.8]]))
    print(SVCClf.n_support_)
    print(SVCClf.support_vectors_)
    print(SVCClf.support_)
    print(SVCClf.intercept_)
    print(SVCClf.fit_status_)
    print("\n")

    # Part 2
    print("NuSVC")
    x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    from sklearn.svm import NuSVC
    NuSVCClf = NuSVC(kernel = 'linear', gamma = 'scale', shrinking = False,)
    print(NuSVCClf.fit(x, y))
    print("\n")

    # Part 3
    print("LinearSVC")
    from sklearn.svm import LinearSVC
    from sklearn.datasets import make_classification
    x, y = make_classification(n_features = 4, random_state = 0)
    LSVCClf = LinearSVC(dual = False, random_state = 0, penalty = 'l1', tol = 1e-5)
    LSVCClf.fit(x, y)
    # once fitted, the model can predict new values
    print(LSVCClf.predict([[0,0,0,0]]))
    # weight vector
    print(LSVCClf.coef_)
    # value of intercept
    print(LSVCClf.intercept_)
    print("\n")

    # Part 4
    print("SVR")
    from sklearn import svm
    x = [[1, 1], [2, 2]]
    y = [1, 2]
    SVRReg = svm.SVR(kernel = 'linear', gamma = 'auto')
    print(SVRReg.fit(x, y))
    # now that model is fitted, we can get weight vector
    print(SVRReg.coef_)
    # using model to get other values:
    print(SVRReg.predict([[1,1]]))
    print("\n")

    # Part 5
    print("NuSVR")
    from sklearn.svm import NuSVR
    n_samples, n_features = 20, 15
    np.random.seed(0)
    y = np.random.randn(n_samples)
    x = np.random.randn(n_samples, n_features)
    NuSVRReg = NuSVR(kernel = 'linear', gamma = 'auto',C = 1.0, nu = 0.1)
    print(NuSVRReg.fit(x, y))
    # now that model is fitted, we can get weight vector
    print(NuSVRReg.coef_)
    print("\n")

    # Part 6
    print("LinearSVR")
    from sklearn.svm import LinearSVR
    from sklearn.datasets import make_regression
    x, y = make_regression(n_features = 4, random_state = 0)
    LSVRReg = LinearSVR(dual = False, random_state = 0,
    loss = 'squared_epsilon_insensitive', tol = 1e-5)
    print(LSVRReg.fit(x, y))
    # once fitted, the model can predict new values
    print(LSVRReg.predict([[0,0,0,0]]))
    # weight vector
    print(LSVRReg.coef_)
    # value of intercept
    print(LSVRReg.intercept_)
    print("\n")



    # Anomoly detection
    # Part 1
    print("Fitting an elliptic envelop")
    import numpy as np
    from sklearn.covariance import EllipticEnvelope
    true_cov = np.array([[.5, .6],[.6, .4]])
    X = np.random.RandomState(0).multivariate_normal(mean = [0, 0], cov=true_cov,size=500)
    cov = EllipticEnvelope(random_state = 0).fit(X)
    print(cov.predict([[0, 0],[2, 2]]))
    print("\n")

    # Part 2
    print("Isolation Forest")
    from sklearn.ensemble import IsolationForest
    x = np.array([[-1, -2], [-3, -3], [-3, -4], [0, 0], [-50, 60]])
    OUTDClf = IsolationForest(n_estimators = 10)
    print(OUTDClf.fit(x))
    print("\n")

    # Part 3
    print("Local Outlier Factor")
    from sklearn.neighbors import NearestNeighbors
    samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    LOFneigh = NearestNeighbors(n_neighbors = 1, algorithm = "ball_tree",p=1)
    print(LOFneigh.fit(samples))
    # now we can ask from this constructed classifier what is the closest point to [0.5, 1, 1.5]:
    print(LOFneigh.kneighbors([[.5, 1., 1.5]]))
    print("\n")

    # Part 4
    print("One-Class SVM")
    from sklearn.svm import OneClassSVM
    x = [[0], [0.89], [0.90], [0.91], [1]]
    OSVMclf = OneClassSVM(gamma = 'scale').fit(x)
    # now we can get the score_sample for input data
    print(OSVMclf.score_samples(x))
    print("\n")



    # KNN Learning
    # Part 1
    print("Unsupervised KNN Learning")
    from sklearn.neighbors import NearestNeighbors
    Input_data = np.array([[-1, 1], [-2, 2], [-3, 3], [1, 2], [2, 3], [3, 4],[4, 5]])
    nrst_neigh = NearestNeighbors(n_neighbors = 3, algorithm = 'ball_tree')
    nrst_neigh.fit(Input_data)
    distances, indices = nrst_neigh.kneighbors(Input_data)
    print(indices)
    print("\n")
    print(distances)
    print("\n")
    # show a connection between neighboring points by producing a sparse graph
    print(nrst_neigh.kneighbors_graph(Input_data).toarray())
    print("\n")

    # Part 2
    print("Supervised KNN Learning")
    from sklearn.datasets import load_iris
    iris = load_iris()
    x = iris.data[:, :4]
    y = iris.target
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    from sklearn.neighbors import KNeighborsRegressor
    knnr = KNeighborsRegressor(n_neighbors = 8)
    print(knnr.fit(X_train, y_train))
    # find the Mean Squared Error
    print("The MSE is:",format(np.power(y-knnr.predict(x),4).mean()))
    print("\n")
    # use it to predict values
    x = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    from sklearn.neighbors import KNeighborsRegressor
    knnr = KNeighborsRegressor(n_neighbors = 3)
    knnr.fit(x, y)
    print(knnr.predict([[2.5]]))
    print("\n")

    # Part 3
    print("RadiusNeighborsRegressor")
    from sklearn.datasets import load_iris
    iris = load_iris()
    x = iris.data[:, :4]
    y = iris.target
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    from sklearn.neighbors import RadiusNeighborsRegressor
    knnr_r = RadiusNeighborsRegressor(radius=1)
    print(knnr_r.fit(X_train, y_train))
    # find the Mean Squared Error
    print("The MSE is:",format(np.power(y-knnr_r.predict(x),4).mean()))
    print("\n")
    # use it to predict other values
    x = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    from sklearn.neighbors import RadiusNeighborsRegressor
    knnr_r = RadiusNeighborsRegressor(radius=1)
    knnr_r.fit(x, y)
    print(knnr_r.predict([[2.5]]))
    print("\n")



    # Classification with Naive Bayes
    print("Building Naive Bayes Classifier")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']
    print(label_names)
    print(labels[0])
    print(feature_names[0])
    print(features[0])
    train, test, train_labels, test_labels = train_test_split(features,labels,test_size = 0.40, random_state = 42)
    from sklearn.naive_bayes import GaussianNB
    GNBclf = GaussianNB()
    model = GNBclf.fit(train, train_labels)
    preds = GNBclf.predict(test)
    print(preds)



    # Decision Trees
    # Part 1
    print("Classification with Decision Trees")
    from sklearn import tree
    from sklearn.model_selection import train_test_split
    X=[[165,19],[175,32],[136,35],[174,65],[141,28],[176,15],[131,32],[166,6],[128,32],[179,10],[136,34],[186,2],[126,25],[176,28],[112,38],[169,9],[171,36],[116,25],[196,25], [196,38], [126,40], [197,20], [150,25], [140,32],[136,35]]
    Y=['Man','Woman','Woman','Man','Woman','Man','Woman','Man','Woman','Man','Woman','Man','Woman','Woman','Woman','Man','Woman','Woman','Man', 'Woman', 'Woman', 'Man', 'Man', 'Woman', 'Woman']
    data_feature_names = ['height','length of hair']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
    DTclf = tree.DecisionTreeClassifier()
    DTclf = DTclf.fit(X,Y)
    prediction = DTclf.predict([[135,29]])
    print(prediction)
    # predicting the probability of each class
    prediction = DTclf.predict_proba([[135,29]])
    print(prediction)
    print("\n")

    # Part 2
    print("Regression with Decision Trees")
    x = [[1, 1], [5, 5]]
    y = [0.1, 1.5]
    DTclf = tree.DecisionTreeRegressor()
    DTreg = DTclf.fit(x, y)
    print(DTreg.predict([[4, 5]]))



    # Randomized Decision Trees
    # Part 1
    print("Classification with Random Forest")
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_blobs
    from sklearn.ensemble import RandomForestClassifier
    x, y = make_blobs(n_samples = 10000, n_features = 10, centers = 100,random_state = 0)
    RFclf = RandomForestClassifier(n_estimators = 10,max_depth = None,min_samples_split = 2, random_state = 0)
    scores = cross_val_score(RFclf, x, y, cv = 5)
    print(scores.mean())
    print("\n")

    # Part 2
    print("Building a Random Forest Classifier")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    dataset = pd.read_csv(path, names = headernames)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    RFclf = RandomForestClassifier(n_estimators = 50)
    RFclf.fit(X_train, y_train)
    y_pred = RFclf.predict(X_test)
    result = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, y_pred)
    print("Classification Report:",)
    print (result1)
    result2 = accuracy_score(y_test,y_pred)
    print("Accuracy:",result2)
    print("\n")

    # Part 3
    print("Regression with Random Forest")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    x, y = make_regression(n_features = 10, n_informative = 2,random_state = 0, shuffle = False)
    RFregr = RandomForestRegressor(max_depth = 10,random_state = 0,n_estimators = 100)
    print(RFregr.fit(x, y))
    # once fitted, we can predict from the regression model:
    print(RFregr.predict([[0, 2, 3, 0, 1, 1, 1, 1, 2, 2]]))
    print("\n")

    # Part 4
    print("Regression with Extre-Tree Method")
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.datasets import make_regression
    x, y = make_regression(n_features = 10, n_informative = 2,random_state = 0, shuffle = False)
    ETregr = ExtraTreesRegressor(max_depth = 10,random_state = 0,n_estimators = 100)
    print(ETregr.fit(x, y))
    # once fitted, we can predict from the regression model:
    print(ETregr.predict([[0, 2, 3, 0, 1, 1, 1, 1, 2, 2]]))



    # Boosting Methods
    # Part 1
    print("Classification with AdaBoost")
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples = 1000, n_features = 10,n_informative = 2, n_redundant = 0,random_state = 0, shuffle = False)
    ADBclf = AdaBoostClassifier(n_estimators = 100, random_state = 0)
    ADBclf.fit(x, y)
    # once fitted we can predict for new values:
    print(ADBclf.predict([[0, 2, 3, 0, 1, 1, 1, 1, 2, 2]]))
    # we can also check the score:
    print(ADBclf.score(x, y))
    print("\n")

    # Part 2
    print("Gradient Tree Boosting")
    from sklearn.datasets import make_hastie_10_2
    from sklearn.ensemble import GradientBoostingClassifier
    x, y = make_hastie_10_2(random_state = 0)
    X_train, X_test = x[:5000], x[5000:]
    y_train, y_test = y[:5000], y[5000:]
    GDBclf = GradientBoostingClassifier(n_estimators = 50, learning_rate = 1.0,max_depth = 1, random_state = 0).fit(X_train, y_train)
    print(GDBclf.score(X_test, y_test))
    print("\n")

    # Part 3
    print("Regression with Gradient Tree Boost")
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import GradientBoostingRegressor
    x, y = make_friedman1(n_samples = 2000, random_state = 0, noise = 1.0)
    X_train, X_test = x[:1000], x[1000:]
    y_train, y_test = y[:1000], y[1000:]
    GDBreg = GradientBoostingRegressor(n_estimators = 80, learning_rate=0.1, max_depth = 1, random_state = 0, loss = 'squared_error').fit(X_train, y_train)
    # once fitted, we can find the Mean Squared Error
    mse = mean_squared_error(y_test, GDBreg.predict(X_test))
    print(mse)



    # Clustering Methods
    # Part 1
    print("K-Means Clustering on Scikit-learn Digit dataset")
    import seaborn as sns; sns.set()
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    digits = load_digits()
    print(digits.data.shape)
    # performing K-Means clustering:
    kmeans = KMeans(n_clusters = 10, random_state = 0)
    clusters = kmeans.fit_predict(digits.data)
    print(kmeans.cluster_centers_.shape)
    print("\n")



    # Clustering Performance Evaluation
    # Part 1
    print("Adjusted Rand Index")
    from sklearn.metrics.cluster import adjusted_rand_score
    labels_true = [0, 0, 1, 1, 1, 1]
    labels_pred = [0, 0, 2, 2, 3, 3]
    print(adjusted_rand_score(labels_true, labels_pred))
    print("\n")

    # Part 2
    print("Mutual Information Based Score")
    from sklearn.metrics.cluster import normalized_mutual_info_score
    labels_true = [0, 0, 1, 1, 1, 1]
    labels_pred = [0, 0, 2, 2, 3, 3]
    print(normalized_mutual_info_score (labels_true, labels_pred))
    print("\n")

    # Part 3
    print("Adjusted Mutual Information (AMI)")
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    labels_true = [0, 0, 1, 1, 1, 1]
    labels_pred = [0, 0, 2, 2, 3, 3]
    print(adjusted_mutual_info_score (labels_true, labels_pred))
    print("\n")

    # Part 4
    print("Fowlkes-Mallow Score")
    from sklearn.metrics.cluster import fowlkes_mallows_score
    labels_true = [0, 0, 1, 1, 1, 1]
    labels_pred = [0, 0, 2, 2, 3, 3]
    print(fowlkes_mallows_score (labels_true, labels_pred))
    print("\n")

    # Part 5
    print("Silhouette Coefficient")
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import pairwise_distances
    from sklearn import datasets
    from sklearn.cluster import KMeans
    dataset = datasets.load_iris()
    x = dataset.data
    y = dataset.target
    kmeans_model = KMeans(n_clusters = 3, random_state = 1).fit(x)
    labels = kmeans_model.labels_
    print(silhouette_score(x, labels, metric = 'euclidean'))
    print("\n")

    # Part 6
    print("Contingency Matrix")
    from sklearn.metrics.cluster import contingency_matrix
    x = ["a", "a", "a", "b", "b", "b"]
    y = [1, 1, 2, 0, 1, 2]
    print(contingency_matrix(x, y))
    print("\n")



    # Dimensionality Reduction using PCA
    # Part 1
    print("Incremental PCA")
    from sklearn.datasets import load_digits
    from sklearn.decomposition import IncrementalPCA
    x, _ = load_digits(return_X_y = True)
    transformer = IncrementalPCA(n_components = 10, batch_size = 100)
    transformer.partial_fit(x[:100, :])
    X_transformed = transformer.fit_transform(x)
    print(X_transformed.shape)
    print("\n")

    # Part 2
    print("Kernel PCA")
    from sklearn.datasets import load_digits
    from sklearn.decomposition import KernelPCA
    x, _ = load_digits(return_X_y = True)
    transformer = KernelPCA(n_components = 10, kernel = 'sigmoid')
    X_transformed = transformer.fit_transform(x)
    print(X_transformed.shape)
######## Import libraries ########
import streamlit as st

######## Page, config, style ########

st.set_page_config(page_title="Part B - Data Science Product")

######## Main ########

st.markdown("""

# Part B - Data Science Product

## *Question / How can we predict house prices?*
- What sort of Machine Learning problem is this?
- What features would you use as input to solve this problem? Why those?
- What algorithm would you use to solve this problem? Why this algorithm?
- How would you set up training and evaluation?
- How would you assess the performance of your solution?
- How would you assess the quality of your source code?
- How would you ship your trained ML model to the customer?
- Two week after shipping your product your customer calls you and complains about low accuracy of your product. How would you react?

## *Concept / Data and Model*

**Problem / Definition.** The task of predicting house prices is a regression problem because the target *y* is a continious
variable and can be handled with supervised learning models that use existing *X / y* for training.

**Features / Preprocessing.** The most promissing features *X* for a prediction model to start with are the MedInc
(median income in each block group) as well as the features Latitude and Longitude (geo-information).
This are on the one hand common factors that influence housing prices but also our EDA supports this hypothesis.

Concerning the preprocessing, there are no duplicates and no missing values (no imputation of missing values necessary)
though we need a strategy for the outliers of various features. Therefore, in a first attempt, we could make use of
a Robust Scaler (MedInc - ev. log(feature) if the data is still skewed) or alternatively get rid of them
(in the case of AveRooms, AveBedrms, Population and AveOccup, since in these cases one could interpret them as the
prevalence of a different house typology (ex. hotels). We could also try to balance (undersample) MedHouseVal since the
prices above 500k are seemingly summarized and therefore not well distributed.

Further we could try some feature engineering for Latitude and Longitude (instead of using a Standard Scaler) - since lat/lon
are cyclical features; try to either cluster them via Kmeans (or SOM, see below) and get the distance from the points to
the centers as features or extract X, Y, and Z in order to be able to normalize them (x = cos(lat) * cos(lon) //
y = cos(lat) * sin(lon) // z = sin(lat)).

**Algorithms / Model.** I would propose to start with a simple linear regression model as a baseline but move on to models
such as *Ridge (L2 regularization), Lasso (L1 regularization, ElasticNet (L1 and L2 regularization)* with k-fold cross
validation in order to avoid overfitting due to the fact that the dataset is quite small. I would also try to train them
on all the features in order to also check a potential multicolinarity of the features (ex. rooms).

In order to increase the complexity of the model I would also propose to try other approaches such as *Decission Trees
and Ensemble Methods (Random Forrest, GradientBoosting, Xgboost...all made to find non-linear relationships)* with
various hyperparameters in a RandomizedSearchCV and/or GridSearchCV with different models (again check feature
importances and/or permutation -> in general less features allows for more interpretability, is faster to train
(speed, computational costs) and easier to implement and maintain in production). Since there are not too many
features (as well as samples) the computational costs will in our case be easily manageable.

As another alternative approach I would propose to use *SOM (Self-organizing-maps)* for clustering (unsupervised)
of the features (dealing with outliers) as well as the prediction part (supervised) of the model in order to map
the data to 2d from the higher dimensional spaces. The algorithm thereby preserves the topology of the data, provides
neighborhoods that one can read (codebook, u-matrix, activation of lattice), allows us to learn about the
distribution/weight/importance/ of the features, and last but not least further allows us to predict bmu
(best matching units) for future data samples.

**Train / Evaluation.** Make a train/validation/test split (70/15/15) of X (features) and y (target) - in order to avoid
data leackage (never use the test-set for evaluation). At best use k-fold cross validation and pipelines with fit - transform.

**Performance / Metrics.** As a performance metrics I would use MSE/RMSE (comparability of different models) and R2
(goodness-of-fit, interpretability). Further check the learning curves of each model and its hyperparameters (see that
they converge at a high score) - all in order to find a proper bias / variance tradeoff alongside further hypterparameter tuning.

**Code / Qualities.** In order to check the quality of my code I would revise it (make functions, modules -> finally
create a package - setup.py) following best practices such as extensive commenting, using descriptive variable names,
proper coding style, write a README.md (description), check requirements.txt (package dependencies), make .env/.envrc/direnv
(environment variables, behaviours, resources, credentials - especially if we want to put the model on the cloud), use
Makefiles (command line) and also by testing the code (make test files) as well as the package on another machine.

**Product / Shipment.** The model could be shipped as a webapplication made with Streamlit (or flask etc...) where the
customer can input new data and get a prediction from the pre-trained model. But all this depends on the customer, if
he wants to have a standalone application (frondend, terminal,...?), or use the model embedded in another application
(CAD/BIM software, Rhino 3D (Grasshopper), Blender 3D...?). The model could also be put on the cloud so that it can be used via an
API, as well as being optimized, maintained and regularly updated, such as with new datapoints from real-time data
streams such as real estate platforms - this would especially make sense in our case of house price predictions.

**Lifecycle / Maintainance.** We could ask the customer for the data he wants to get the prediction for to get more
insight into the problem/issue. Maybe the trained model has been overfitted. Maybe the new data contains outliers.
Maybe we can get more samples to train on, or updated (real-time) data since the housing market is rapidly evolving.
Further we could revise the data quality, model quality, model bias, and model explainability - get more features (open data),
do further feature engineering (distance to city centers, sea, public transport, schools, and other urban infrastructure)
and then retrain the model.

""")

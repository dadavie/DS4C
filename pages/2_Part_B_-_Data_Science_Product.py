######## Import libraries ########
import streamlit as st

######## Page, config, style ########

st.set_page_config(page_title="Part B - Data Science Product")

######## Main ########

st.markdown("""

# Part B - Data Science Product

## *Question / How can we predict house prices?*
- What sort of **Machine Learning** problem is this?
- What **features** would you use as input to solve this problem? Why those?
- What **algorithm** would you use to solve this problem? Why this algorithm?
- How would you set up **training and evaluation**?
- How would you assess the **performance** of your solution?
- How would you assess the **quality** of your source code?
- How would you **ship your trained ML model** to the customer?
- Two week after shipping your product your customer calls you and **complains about low accuracy** of your product. How would you react?

## *Concept / Data and Model*

**Problem / Definition.** The task of predicting house prices is a regression problem as the target *y* is a continuous
variable. It can be addressed using supervised learning models with existing *X / y* data for training.

**Features / Preprocessing.** Promising features *X* for the prediction model are the MedInc
(median income in each block group) as well as the features Latitude and Longitude (geo-information).
These are common factors that influence house prices but also the EDA supports this hypothesis.

Concerning the preprocessing of the data; there are no duplicates and no missing values (no imputation necessary).
However, we need a strategy for the outliers of certain features. Therefore, in a first attempt, we could make use of
a Robust Scaler for MedInc (ev. log(feature) if the data is still skewed) and get rid of features such as AveRooms,
AveBedrms, Population and AveOccup, since they could be interpreted as not primarily important due to the obvious
prevalence of other typologies (ex. hotels) in the dataset. We could also try to balance (undersample) the target MedHouseVal since the
prices above 500k are seemingly cumulated and therefore not well distributed.

Feature engineering may be necessary for Latitude and Longitude (instead of using a scaler) - since lat/lon
are cyclical features; try to either cluster them via Kmeans (or SOM, see below) and get the distances from the points to
the centers as features or extract X, Y, and Z in order to be able to normalize them (x = cos(lat) * cos(lon) //
y = cos(lat) * sin(lon) // z = sin(lat)).

**Algorithms / Models.** A simple linear regression model is proposed as a baseline. Models such as *Ridge (L2 regularization),
Lasso (L1 regularization, ElasticNet (L1 and L2 regularization)* with k-fold cross validation might prevent us from overfitting
due to the fact that the dataset is quite small. I would alternatively also propose to train the models on all the features
in order to find out more about a potential multicollienarity of certain features (ex. rooms).

In order to find out more about non-linear relationships we should increase the complexity of the applied ML models.
Therefore I would propose to try also other algorithms and attempts such as *Decission Trees and Ensemble Methods
(Random Forrest, GradientBoosting, XGboost...)* with various hyperparameters in a RandomizedSearchCV and/or GridSearchCV
(again check feature importances and/or permutation -> in general less features allows for more interpretability, is
faster to train (speed, computational costs) and easier to implement and maintain in production). However,
since there are not too many features (as well as samples) in our dataset, the computational part will be easily manageable.

As another alternative approach I would suggest using *SOM (Self-organizing-maps)* for clustering (unsupervised)
of the features (dealing with outliers) and the prediction part (supervised). SOM's allow us to map
the data from higher dimensional spaces to 2D. The algorithm thereby preserves the topology of the data, provides
neighborhoods that one can read (codebook, u-matrix, activation of lattice), enable us to learn about the
distribution/weight/importance of the features and last but not least further predict bmu
(best matching units) of future data samples.

**Train / Evaluation.** Make a train/validation/test split (70/15/15) of X (features) and y (target) - in order to avoid
data leakage (never use the test-set for evaluation). At best use k-fold cross validation and pipelines with fit - transform.
Besides the performance metrics also check fit and score time in order to learn more about the computational performance of each model.

**Performance / Metrics.** As a performance metrics I would propose MSE/RMSE (comparability of different models) and R2
(goodness-of-fit, interpretability). Further check the learning curves of each model and its hyperparameters (see that
they converge at a high score) in order to find a proper bias / variance tradeoff alongside further hyperparameter tuning.

**Code / Qualities.** In order to check the quality of my code I would revise it (make functions, modules -> finally
create a package - setup.py) following best practices such as extensive commenting, using descriptive variable names,
proper coding style, write a README.md (description), check requirements.txt (package dependencies), make .env/.envrc/direnv
(environment variables, behaviors, resources, credentials - especially if we want to put the model on the cloud), use
Makefiles (command line) and also by testing the code (make test files) as well as the package on other machines/operating systems.

**Product / Shipment.** The model could be shipped as a web application made with Streamlit (or flask etc...) where the
customer can input new data and get a prediction from the pre-trained model. However all this depends on the needs of the customer.
Does she/he want to have a standalone application (frontend, terminal,...?), or use the model embedded in another application
(CAD/BIM/3D software). The model could also run on the cloud so that it can be used via an
API. Thereby it can be easily updated, regularly maintained and continuously optimized.

**Lifecycle / Maintenance.** We should ask the customer to provide us with the data he wants to get the prediction for in order
to get a deeper insight into the problem. Maybe the trained model has been overfitted. Maybe the new data contains outliers. Since
the housing market is evolving rapidly, we should try to get more samples to train the model on, such as real-time data from online
real estate platforms. We could further propose to get more features (open data) and do further feature engineering (distance to city centers,
sea, public transport, schools, and other urban infrastructures). Finally we should revise the data and model quality as well as
the bias/variance tradeoff and thereupon retrain the model.

""")

# Load the installed packages
using DataFrames
using CSV
using Plots
using Lathe
using GLM
using Statistics
using StatsPlots
using MLBase
using RDatasets

cars = dataset("datasets", "cars")
df = DataFrame(cars)
first(df,5)

println(size(df))
describe(df)

# Check column names
names(df)

T = rand(10:30,50)
S = randn(50)
df[:Time] = T
df[:Slope] = S
first(df,5)
#Outlier Analysis using Box Plot
# Box Plot
p1  = boxplot(df.Speed, title = "Cars Speed", ylabel = "Speed", legend = false)
p2  = boxplot(df.Dist, title = "Cars Distance", ylabel = "Distance", legend = false)

plot(p1,p2, layout=(1,2))

# Outlier removal
first_percentile = percentile(df.Speed, 25)
iqr_value = iqr(df.Speed)
df = df[df.Speed .>  (first_percentile - 1.5*iqr_value),:];

first_percentile = percentile(df.Dist, 25)
iqr_value = iqr(df.Dist)
df = df[df.Dist .>  (first_percentile - 1.5*iqr_value),:];

first(df,10)

# Density Plot
p3  = density(df.Speed , title = "Density Plot - Speed", ylabel = "Speed", xlabel = "Speed", legend = false)
p4  = density(df.Dist , title = "Density Plot - Dist", ylabel = "Distance", xlabel = "Distance", legend = false)
plot(p3,p4,layout=(1,2))
# Correlation Analysis using Scatter Plot
# Correlation Analysis
println("Correlation of Speed with Distance is ", round(cor(df.Speed,df.Dist),digits=3))

# Scatter plot
train_plot = scatter(df.Speed,df.Dist, title = "Scatter Plot Speed vs Distance", ylabel = "Distance", xlabel = "Speed",legend = false)

# Train-Test Split
using Random
shuffled  =  shuffle(collect(1:50))
rows = shuffled[1:40]

train = df[rows,:]
test = df[setdiff(1:end, rows),:]

# Model Building
# Built the linear regression model using “GLM” package. It’s very similar to the “GLM” package in R.
names(train)
fm = @formula(Dist ~ Speed)
linearRegressor = lm(fm, train)

# Model Dignostics: P-value, T-statistic, R-square

# P-value: A variable in linear regression model is said to be statistically significant only if the p-value is less than a pre-determined statistical significance level, which usually is 0.05.
# T-statistic: A larger t-value indicates that it is less likely that the coefficient is not equal to zero purely by chance. So, the higher the t-value, the better.
# R-square: R-Square is a statistical measure which tells us the proportion of variation in the dependent (y) variable that is explained by different features (independent variables) in this model.

# R Square value of the model
round(r2(linearRegressor),digits=3)

# # Prediction
ypredicted_test = predict(linearRegressor, test)
ypredicted_train = predict(linearRegressor, train)

# Model Prediction and Performance

# Mean Aboslute Error
# Mean Absolute Percentage Error
# Root Mean Square Error
# Mean Squared Error
# Error Distribution

# Test Performance DataFrame (compute squared error)
performance_testdf = DataFrame(y_actual = test[!,:Dist], y_predicted = ypredicted_test)
performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error

# Train Performance DataFrame (compute squared error)
performance_traindf = DataFrame(y_actual = train[!,:Dist], y_predicted = ypredicted_train)
performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error

# Defining the Performance functions
# MAPE function defination
function mape(performance_df)
    mape = mean(abs.(performance_df.error./performance_df.y_actual))
    return mape
end

# RMSE function defination
function rmse(performance_df)
    rmse = sqrt(mean(performance_df.error.*performance_df.error))
    return rmse
end

# Test Error
println("Mean Absolute test error: ", round(mean(abs.(performance_testdf.error)),digits=3), "\n")
println("Mean Aboslute Percentage test error: ",round(mape(performance_testdf),digits=3), "\n")
println("Root mean square test error: ",round(rmse(performance_testdf),digits=3), "\n")
println("Mean square test error: ",round(mean(performance_testdf.error_sq),digits=3), "\n")

# Train Error
println("Mean Absolute test error: ", round(mean(abs.(performance_traindf.error)),digits=3), "\n")
println("Mean Aboslute Percentage test error: ",round(mape(performance_traindf),digits=3), "\n")
println("Root mean square test error: ",round(rmse(performance_traindf),digits=3), "\n")
println("Mean square test error: ",round(mean(performance_traindf.error_sq),digits=3), "\n")

# Error Distribution
# Histogram of error to see if it's normally distributed  on test dataset
histogram(performance_testdf.error, bins = 5, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)

# Histogram of error to see if it's normally distributed  on train dataset
histogram(performance_traindf.error, bins = 5, title = "Training Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)

# Scatter plot of actual vs predicted values on test dataset
test_plot = scatter(performance_testdf[!,:y_actual],performance_testdf[!,:y_predicted], title = "Predicted value vs Actual value on Test Data", ylabel = "Predicted value", xlabel = "Actual value", legend = false)
# Scatter plot of actual vs predicted values on train dataset
train_plot = scatter(performance_traindf[!,:y_actual],performance_traindf[!,:y_predicted], title = "Predicted value vs Actual value on Train Data", ylabel = "Predicted value", xlabel = "Actual value",legend = false)

# Cross Validation

# A good model should perform equally well on new data (X vars) as it did on the training data.
# If the performance on new data (test data and any future data) deteriorates, it is an indication that the model may be overfit.
# That is, it is too complex that it explains the training data, but not general enough to perform as well on test.

# Cross Validation function defination
function cross_validation(train,k, fm = @formula(Dist ~ Speed))
    a = collect(Kfold(size(train)[1], k))
    for i in 1:k
        row = a[i]
        temp_train = train[row,:]
        temp_test = train[setdiff(1:end, row),:]
        linearRegressor = lm(fm, temp_train)
        performance_testdf = DataFrame(y_actual = temp_test[!,:Dist], y_predicted = predict(linearRegressor, temp_test))
        performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]

        println("Mean error for set $i is ", round(mean(abs.(performance_testdf.error)),digits=3))
    end
end

cross_validation(train,10)

# Multiple Linear Regression

fm = @formula(Dist ~ Speed + Time + Slope)

linearRegressor = lm(fm, train)

round(r2(linearRegressor), digits=3)

# Model Prediction and Performance

# Prediction
ypredicted_test = predict(linearRegressor, test)
ypredicted_train = predict(linearRegressor, train)

# Test Performance DataFrame
performance_testdf = DataFrame(y_actual = test[!,:Dist], y_predicted = ypredicted_test)
performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error

# Train Performance DataFrame
performance_traindf = DataFrame(y_actual = train[!,:Dist], y_predicted = ypredicted_train)
performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error

# Test Error
println("Mean Absolute test error: ",round(mean(abs.(performance_testdf.error)),digits=3), "\n")
println("Mean Aboslute Percentage test error: ",round(mape(performance_testdf),digits=3), "\n")
println("Root mean square test error: ",round(rmse(performance_testdf),digits=3), "\n")
println("Mean square test error: ",round(mean(performance_testdf.error_sq),digits=3), "\n")


# Influencial Points using Cook’s Distance
# Cook’s Distance
# Cook’s distance is a measure computed with respect to a given regression model and therefore is
# impacted only by the X variables included in the model.
# It computes the influence exerted by each data point (row) on the predicted outcome.

# Cook's Distance function definition
function cook_distance(train,n_coeff = 2, fm = @formula(Dist ~ Speed))
    linearRegressor = lm(fm, train)

    # MSE calculation
    performance_traindf = DataFrame(y_actual = train[!,:Dist], y_predicted = predict(linearRegressor, train))
    performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
    performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error
    mse = mean(performance_traindf.error_sq)

    cooks_distance = []
    for i in 1:size(train)[1]
        total_rows = collect(1:size(train)[1])
        training_rows = deleteat!(total_rows, total_rows .== i)
        temp_train = train[training_rows,:]
        linearRegressor = lm(fm, temp_train)

        temp_performance_traindf = performance_traindf[training_rows,:]

        # Predicted Value
        ypredicted_train = predict(linearRegressor, temp_train)

        # Test Performance DataFrame
        performance_df = DataFrame(yi = temp_performance_traindf.y_predicted, yj = ypredicted_train)
        performance_df.y_diff = performance_df[!,:yi] - performance_df[!,:yj]
        performance_df.y_diff_sq = performance_df.y_diff.*performance_df.y_diff
        cooks_d = sum(performance_df.y_diff_sq)/(n_coeff*mse)
        push!(cooks_distance,cooks_d)
    end
    cooks_distance_df = DataFrame(index = collect(1:size(train)[1]), cooks_d = cooks_distance )
    return cooks_distance_df
end

# Calculate Cook's Distance
cooks_distance_df = cook_distance(train, 9, fm);
mean_cooks_distance = mean(cooks_distance_df.cooks_d)
influencial_points = cooks_distance_df[cooks_distance_df.cooks_d .> 4*mean_cooks_distance,:]
println(mean_cooks_distance)

high_cooksd_points = cooks_distance_df[cooks_distance_df.cooks_d .> 0.02,:]
influencial_points_df = train[high_cooksd_points.index,[:Dist, :Speed, :Time, :Slope]]
# Influence measures
# In general use, those observations that have a cook’s distance greater than 4 times the mean
# may be classified as influential. This is not a hard boundary.

plot = scatter(cooks_distance_df[!,:index], cooks_distance_df[!,:cooks_d], title = "Cook's Distance Analysis", ylabel = "Cook's Distance", xlabel = "Index",  label = ["Cook's Distance"], ylim = (0,0.03))
hline!([mean_cooks_distance], lw = 3,  label = ["Mean Cook's Distance"])

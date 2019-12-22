library(forecast)
library(lubridate)
library(tseries)
library(fpp)
library(ggplot2)

#Reading the base data#
base_data<-read.csv('D:/Freelancer_questions/ARIMAX/1134769745_daily_with_workdays_27.csv')
colnames(base_data)<-c("Date","Temperature","Humidity","Windspeed","Count","All_workdays")
base_data<-base_data[order(as.Date(base_data$Date, format="%d/%m/%Y")),]
base_data$All_workdays<-as.factor(base_data$All_workdays)
#ARIMA programming starts
## 75% of the sample size
smp_size <- floor(0.95 * nrow(base_data))


## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(base_data)), size = smp_size)

train <- base_data[train_ind, ]
test <- base_data[-train_ind, ]

# Create matrix of numeric predictors
xreg <- cbind(as_workday=model.matrix(~train$All_workdays), 
              Temp=train$Temperature,
              Humidity=train$Humidity,
              Windspeed=train$Windspeed
)

# Remove intercept
xreg <- xreg[,-1]

# Rename columns
colnames(xreg) <- c("All_workdays","Temp","Humidity","Windspeed")

#creating the same for the test data

xreg1 <- cbind(as_workday=model.matrix(~test$All_workdays), 
               Temp=test$Temperature,
               Humidity=test$Humidity,
               Windspeed=test$Windspeed
)

# Remove intercept
xreg1 <- xreg1[,-1]

# Rename columns
colnames(xreg1) <- c("All_workdays","Temp","Humidity","Windspeed")

# Variable to be modelled
Count <- ts(train$Count, start=c(2016,7,7),frequency=365)


# Find ARIMAX model
modArima <- auto.arima(Count, xreg=xreg)
modArima
Forecasted_values<-forecast(modArima,nrow(test),xreg=xreg1)
Final_forecasted_values<-Forecasted_values$mean

forecast(modArima,nrow(test),xreg=xreg1) -> fc
autoplot(Count, series="Data") + 
autolayer(fc, series="Forecast") + 
autolayer(fitted(fc), series="Fitted")

library(ggplot2)
autoplot(Forecasted_values)

#finding the MSE for the ARIMAX forecasted values on the test data
mean((test$Count - Final_forecasted_values)^2)

#with seasonality removal

decompose_data = decompose(Count, "additive")
adjust_count = Count - decompose_data$seasonal
plot(adjust_count)

# Find ARIMAX model of deseasoned data
modArima_desea <- auto.arima(adjust_count, xreg=xreg)
modArima_desea
Forecasted_values_des<-forecast(modArima_desea,nrow(test),xreg=xreg1)
Final_forecasted_values_des<-Forecasted_values_des$mean

library(ggplot2)
forecast(modArima_desea,nrow(test),xreg=xreg1) -> fc1
autoplot(Count, series="Data") + 
autolayer(fc1, series="Forecast") + 
autolayer(fitted(fc1), series="Fitted")

#Mean square 
mean((test$Count - Final_forecasted_values_des)^2)

#Dynamic regression of the data

library(dplyr)


x<-train[order(as.Date(train$Date, format="%d/%m/%Y")),]

train_aug <- x %>%
  mutate(count_lag1 = lag(Count, n = 1, order_by = Date),
         count_lag2 = lag(Count, n = 2, order_by = Date),
         temp_lag1 = lag(Temperature, n = 1, order_by = Date),
         temp_lag2 = lag(Temperature, n = 2, order_by = Date),
         Hum_lag1 = lag(Humidity, n = 1, order_by = Date),
         Hum_lag2 = lag(Humidity, n = 2, order_by = Date),
         Wind_lag1 = lag(Windspeed, n = 1, order_by = Date),
         Wind_lag2 = lag(Windspeed, n = 2, order_by = Date)
  )

x1<-test[order(as.Date(test$Date, format="%d/%m/%Y")),]

test_aug <- x1 %>%
  mutate(count_lag1 = lag(Count, n = 1, order_by = Date),
         count_lag2 = lag(Count, n = 2, order_by = Date),
         temp_lag1 = lag(Temperature, n = 1, order_by = Date),
         temp_lag2 = lag(Temperature, n = 2, order_by = Date),
         Hum_lag1 = lag(Humidity, n = 1, order_by = Date),
         Hum_lag2 = lag(Humidity, n = 2, order_by = Date),
         Wind_lag1 = lag(Windspeed, n = 1, order_by = Date),
         Wind_lag2 = lag(Windspeed, n = 2, order_by = Date)
  )

#OLS regression using the Dynamic lagged variables
my_lm <- lm(Count ~ count_lag1 + count_lag2 + temp_lag1 + temp_lag2 +Hum_lag1+Hum_lag2+Wind_lag1+Wind_lag2,data = train_aug[3:nrow(train_aug), ])
summary(my_lm)

# Significant variables

My_lm_final <-lm(Count ~ count_lag1 + count_lag2 + temp_lag1 + temp_lag2 , data = train_aug[3:nrow(train_aug), ])
summary(My_lm_final)

predicted_Dynm<-predict(My_lm_final,newdata = test_aug)

test_aug$Predicted<-predicted_Dynm
test_aug_1<-na.omit(test_aug)
# Mean Square error for the dynamic regression#
mean((test_aug_1$Count - test_aug_1$Predicted)^2)

plot(test_aug_1$Count,test_aug_1$Predicted,
     xlab="predicted",ylab="actual")
abline(a=0,b=1)

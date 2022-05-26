# 
# library(arrow)
# library(dplyr)
# 
# data <- read_parquet("~/Desktop/Faks/magisterij/1.letnik/GEN-I/data/trds.parquet")
# 
# x <- data$trd_price
# y <- data$trd_delivery_time_start
# 
# timeseries <- ts.union(x, y)

my_data <- read.csv("/Users/janzaloznik/Desktop/Faks/magisterij/1.letnik/GEN-I/machine-learning-price-prediction-for-hourly-blocks/filename.csv",header=TRUE, dec = ',') #decimal seperator
my_data2 <- read.csv("/Users/janzaloznik/Desktop/Faks/magisterij/1.letnik/GEN-I/machine-learning-price-prediction-for-hourly-blocks/All_data.csv",header=TRUE, dec = ',') #decimal seperator

x <- as.numeric(my_data$trd_price)


require(graphics)
x <- ts(x, start=c(2022,768),frequency = 8760)
print(x)



par(mai=c(0.5,0.5,0.4,0.2),
    mgp=c(1.5,0.3,0),
    cex.main=1)

plot(x,
     main="data",
     ylab="Value", lwd = 1)

d <- diff(x)


plot(d,
     main="",
     ylab=expression(paste(nabla,x[t]," [pp]")))


abline(h=mean(d),col="blue")
abline(h=c(mean(d)-sd(d),mean(d)+sd(d)),col="blue",lty="dotted")




library(TSA)

d.p <- periodogram(d)
#we can see some seasonality

order(d.p$spec,decreasing=TRUE) #these are heights

d.p$freq[c(100,101,201)]

1/d.p$freq[c(100,101,201)]


spectrum(d,
         kernel=kernel(coef='modified.daniell', m=2),
         log="no",
         demean=TRUE,
         detrend=FALSE,
         taper=0)

spectrum(d,
         spans=7, # L = 2m+1 for Modified Daniell kernel (default); "rounds up" if even
         log="no",
         demean=TRUE,
         detrend=FALSE,
         taper=0)

d.sp <- spectrum(d,
                 kernel=kernel(coef='daniell', m=c(3,3)),
                 log="no",
                 demean=TRUE,
                 detrend=FALSE,
                 taper=0,
                 lwd = 1.5)


order(d.sp$spec,decreasing = TRUE) #Gonna take the first values

d.sp$freq[c(100,101,99)]

1/d.p$freq[c(100,101,99)]

t <- time(d)

mod.H <- lm(d ~ I(sin(2*pi*t*730)) + I(cos(2*pi*t*730)) + I(sin(2*pi*t*737.3)) + I(cos(2*pi*t*737.3)))
summary(mod.H2)

d.fit <- mod.H$fitted.values
d.fit <- ts(d.fit,start=start(d),frequency=frequency(d))


plot(d,
     main="",
     ylab="")

points(d.fit,col="red",type="o",pch=16,cex=0.5)

r <- residuals(mod.H)


library(forecast)

plot(r)

# ACF and PACF

Acf(r) 

Pacf(r) 

Box.test(r,lag=5,type="Box-Pierce")

# Ljung-Box test

Box.test(r,lag=,type="Ljung-Box")

library(tseries)

adf.test(r,alternative="stationary")



### arma 

my.arma <- stats::arima(r, order=c(1,0,1), include.mean = FALSE, method ='ML')

AIC(my.arma)

ic.values <- matrix(NA,nrow=5,ncol=5,
                    dimnames=list(paste("AR=",0:4,sep=""),
                                  paste("MA=",0:4,sep="")))


for(i in 1:5){
  for(j in 1:5){
    my.arma <- stats::arima(r, order=c(i-1,0,j-1), include.mean = FALSE, method ='ML')
    ic.values[i,j] <- AIC(my.arma)
    remove(my.arma)
  }
}

which.min(ic.values) #or
ic.values == min(ic.values)

# AR(2) is optimal

my.arma <- stats::arima(r, order=c(2,0,2), include.mean = FALSE, method ='ML')

confint(my.arma, level = 0.95)


x.res <- residuals(my.arma)


plot(x.res,
     main="",
     ylab="ARIMA residuals")

abline(h=mean(x.res),col="blue")
abline(h=c(mean(x.res)-sd(x.res),mean(x.res)+sd(x.res)),col="blue",lty="dotted")

# ACF and PACF

Acf(x.res) # autocorrelation function

Pacf(x.res) # partial autocorrelation function


#tests

Box.test(x.res,lag=5,type='Box-Pierce')
#it looks like white noise

# Ljung-Box test

Box.test(x.res,lag=5,type='Ljung-Box')

# Normal dist?
qnorm(x.res)



# ARMA  predictions
A <- predict(my.arma,n.ahead=30)$pred

#Harmonic regression prediction
B <- predict(mod.H,newdata=data.frame(t=seq(2021 + 1949/8760, 2021 + 1979/8760, 1/8760))) #prediction of seasonality

#now we have to sum these two back together (because we took a difference in beginning)

A+B

# Undiferencing
#Predict d_T,..,d_(T+1) = X_(T+1) - X_T, so
# X_T = X_(T+1) - d_T

C <- x[1182] + cumsum(A+B)

C <- ts(C,start=c(2022,1949),frequency = 8760)
X <- ts.union(x,C)

plot(X,plot.type="single",
     main="",
     ylab="Price",
     col=c("black","red"), lwd = 1)

legend("topright",
       c("Observed","Predicted"),
       col=c("black","blue"),
       lty="solid",
       bty="n", lwd = 2)


#################################################################################
#################################################################################
#################################################################################
x2 <- as.numeric(my_data2$trd_price)


require(graphics)
x2 <- ts(x2, start=c(2021,7488),frequency = 8760)




plot(x2,
     main="data",
     ylab="Value", lwd = 1)

d2 <- diff(x2)


plot(d2,
     main="",
     ylab="")


abline(h=mean(d2),col="blue")
abline(h=c(mean(d2)-sd(d2),mean(d2)+sd(d2)),col="blue",lty="dotted")




library(TSA)

d.p2 <- periodogram(d2)
#we can see some seasonality

order(d.p$spec,decreasing=TRUE) #these are heights

d.p2$freq[c(362,181,360)]

1/d.p2$freq[c(362,181,360)]


spectrum(d2,
         kernel=kernel(coef='modified.daniell', m=2),
         log="no",
         demean=TRUE,
         detrend=FALSE,
         taper=0)

spectrum(d2,
         spans=7, # L = 2m+1 for Modified Daniell kernel (default); "rounds up" if even
         log="no",
         demean=TRUE,
         detrend=FALSE,
         taper=0)

d.sp2 <- spectrum(d2,
                 kernel=kernel(coef='daniell', m=c(3,3)),
                 log="no",
                 demean=TRUE,
                 detrend=FALSE,
                 taper=0,
                 lwd = 1.5)


order(d.sp2$spec,decreasing = TRUE) #Gonna take the first values

d.sp2$freq[c(362,361,363)]

1/d.sp2$freq[c(362,361,363)]

t2 <- time(d2)

mod.H2 <- lm(d2 ~ I(sin(2*pi*t2*734)) + I(cos(2*pi*t2*734)) + I(sin(2*pi*t2*732)) + I(cos(2*pi*t2*732)))
summary(mod.H2)

d.fit2 <- mod.H2$fitted.values
d.fit2 <- ts(d.fit2,start=start(d2),frequency=frequency(d2))


plot(d2,
     main="",
     ylab="")

points(d.fit2,col="red",type="o",pch=16,cex=0.5)

r2 <- residuals(mod.H2)


library(forecast)

plot(r2)

# ACF and PACF

Acf(r2) 

Pacf(r2) 

Box.test(r2,lag=5,type="Box-Pierce")

# Ljung-Box test

Box.test(r2,lag=,type="Ljung-Box")

library(tseries)

adf.test(r2,alternative="stationary")



### arma 

my.arma2 <- stats::arima(r2, order=c(1,0,1), include.mean = FALSE, method ='ML')

AIC(my.arma2)

ic.values2 <- matrix(NA,nrow=5,ncol=5,
                    dimnames=list(paste("AR=",0:4,sep=""),
                                  paste("MA=",0:4,sep="")))


for(i in 1:5){
  for(j in 1:5){
    my.arma2 <- stats::arima(r, order=c(i-1,0,j-1), include.mean = FALSE, method ='ML')
    ic.values2[i,j] <- AIC(my.arma2)
    remove(my.arma2)
  }
}

which.min(ic.values2) #or
ic.values2 == min(ic.values2)

# AR(1), MA(1) is optimal

my.arma2 <- stats::arima(r2, order=c(1,0,1), include.mean = FALSE, method ='ML')

confint(my.arma2, level = 0.95)


x2.res <- residuals(my.arma2)


plot(x2.res,
     main="",
     ylab="ARIMA residuals")

abline(h=mean(x2.res),col="blue")
abline(h=c(mean(x2.res)-sd(x2.res),mean(x2.res)+sd(x2.res)),col="blue",lty="dotted")

# ACF and PACF

Acf(x2.res) # autocorrelation function

Pacf(x2.res) # partial autocorrelation function


#tests

Box.test(x2.res,lag=5,type='Box-Pierce')
#it looks like white noise

# Ljung-Box test

Box.test(x2.res,lag=5,type='Ljung-Box')

# Normal dist?
# qnorm(x.res)



# ARMA  predictions
A2 <- predict(my.arma2, n.ahead = 24)$pred

#Harmonic regression prediction
B2 <- predict(mod.H2,newdata=data.frame(t2=seq(2022 + 2928/8760,2022 + 2951/8760, 1/8760))) #prediction of seasonality

#now we have to sum these two back together (because we took a difference in beginning)

A2 + B2

# Undiferencing
#Predict d_T,..,d_(T+1) = X_(T+1) - X_T, so
# X_T = X_(T+1) - d_T

C2 <- x2[4129] + cumsum(A2 + B2)

C2 <- ts(C2,start=c(2022,2860),frequency = 8760)
X2 <- ts.union(x2,C2)

plot(X2,plot.type="single",
     main="date",
     ylab="Price",
     col=c("black","blue"), lwd = 1)

legend("topright",
       c("Observed","Predicted"),
       col=c("black","blue"),
       lty="solid",
       bty="n", lwd = 2)



################################################################################################
################################################################################################
# 16.5.2022


#I used harmonic regression prediction (that looks like this graph)
plot(d,
     main="Differencing",
     ylab="diff() values")

points(d.fit,col="red",type="o",pch=16,cex=0.5)

#plot first intra_day data

plot(X,plot.type="single",
     main="",
     ylab="Price",
     col=c("black","red"), lwd = 1)

legend("topright",
       c("Observed","Predicted"),
       col=c("black","red"),
       lty="solid",
       bty="n", lwd = 2)










#All data



## the same with All data and harmonic prediciton

plot(d2,
     main="Differencing",
     ylab="diff() values")

points(d.fit2,col="red",type="o",pch=16,cex=0.5)




plot(X2,plot.type="single",
     main="date",
     ylab="Price",
     col=c("black","blue"), lwd = 1)

legend("topright",
       c("Observed","Predicted"),
       col=c("black","blue"),
       lty="solid",
       bty="n", lwd = 2)





# Reading the data

mData <- read.csv("Data.csv") # Lines 1-8 read the data of the original paper ("Data.csv"), we do not need these lines
attach(mData)

fnComplete <- function(x,l=301) { 
	return(c(x,rep(x[length(x)],l-length(x)))) # function used 
}

# Building the MCS

vZ <- 300+4.9*(1:301) # vZ is the vector DGP of the benchmark data considered in the original paper
		      # ("v" vector, "m" matrix, "l" list, "i" integer, "d" double, "fn" function)

mDist <- NULL # The following loop is used to compute the loss fn used in the original paper (lines 15-27)
for (i in 1:16) {
	mDist2 <- matrix(0,nrow=200,ncol=2)
	mDist2[,2] <- i
	mDataCOP <- mData[mData$COP==i,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	for (j in 1:200) {
		vX <- fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2)
		mDist2[j,1] <- sum(abs(vX-vZ)^2)/(301*1000*1000)
	}
	mDist	<- rbind(mDist,mDist2)
	mDataCOP <- NULL
}
mDist <- as.data.frame(mDist) # Dataframe with 2 cols, col 1 values of the loss function col 2 associated CoP
names(mDist) <- c("Dist","COP")

## We do not need the previous lines, as the loss function is computed elsewhere
## The loss fn must be computed for each MC runs of each CoP and then we must take the average across MC runs
## We will end up with "iCoP" distances, where iCoP is the number of CoP

library(data.table) 
mD <- setDT(mDist)[,list(mean=mean(Dist),var=var(Dist)),by=c("COP")] # convert lists and data.frames to data.table by reference to save memory

## Lines 41-61 compute the statistical test

fnTest <- function(vMean,vVar,iN,dA=1) {
	iM <- length(vMean)
	if (iM>2) {
		mA <- cbind(rep(1,iM-1),diag(-rep(1,iM-1)))
		mVar <- diag(vVar[-1])
		mVar2 <- mVar+vVar[1]
		dTest <- drop(iN*t(mA%*%vMean)%*%solve(mVar2)%*%(mA%*%vMean)) # Compute the test
	} else {
		dTest <- iN*(vMean[1]-vMean[2])^2/(vVar[1]+vVar[2])
	}
	if (iM>1) {
		dQuan <- qchisq(1-dA,iM-1) # Compute quantile
		dPVa <- 1-pchisq(dTest,iM-1) # Compute p-value (asymp. the test is distr as chi-square)
		} else {
		dQuan <- 0
		dPVa <- 1
	}
	dResult <- 1*(dTest>dQuan) # Takes value 1 when the test is greater than quantile
	if (is.na(dResult)) dResult <- 0
	return(list(test=dTest,q=dQuan,p=dPVa,result=dResult))
}

## Lines 63-68 compute the elimination rule

fnElim	<- function(vMean,vVar,vIndex) {
	iM <-	which.max(vMean) # elimination rule ($\arg\max(\widehat{i})$)
	return(list(mean=vMean[-iM],var=vVar[-iM],indices=vIndex[-iM],elim=vIndex[iM]))
}

## Lines 72-96 MCS algorithm

fnMCS <- function(vMean,vVar,vIndex,dA,verbose=1) {
	iI <- length(vIndex)
	iStop <- fnTest(vMean,vVar,dA)$result
	mPValue <- matrix(0,ncol=2,nrow=16)
	for (i in 1:iI) {
		lTest <- fnTest(vMean,vVar,iN,dA)
		iStop <- lTest$result
		if (iStop!=1) break # Stopping rule
		lElim <- fnElim(vMean,vVar,vIndex) # elimination rule
		vMean <- lElim$mean # vector of eliminated means
		vVar <- lElim$var # vector of eliminated variances
		vIndex <- lElim$indices # vector of eliminated indices
	 	if (verbose==1) cat("Eliminated COP: ",lElim$elim,"; ","p-value: ", lTest$p,"\n", sep="") # concatenate and print
	 	if (verbose==1) cat("Remaining COPs:",vIndex,"\n") # concatenate and print
		mPValue[i,] <- c(lElim$elim,lTest$p)
	}
	if (i==iI) {
		mPValue[iI,1] <- setdiff(vIndex,mPValue[1:15,1]) # compute the (nonsymmetric) set difference of subsets of a probability space.
		mPValue[iI,2] <- 1
	} else {
		mPValue <- mPValue[1:(i-1),]
		mPValue[,2] <- cummax(mPValue[,2])
	}
	return(mPValue)
}

vIndex <- 1:16 # Number of CoPs
vMean <- mD$mean # Vector of means of the loss function across MC runs (see line 37)
vVar <- mD$var # Vector of variances of the loss function across MC runs (see line 37)
dA <- 1
iN <- 200 # Number of MC runs for each CoP

mPValue <- fnMCS(vMean,vVar,vIndex,dA,verbose=0) # Compute p-values associated to the different CoPs

## Plot CoPs greater than 0.05 (we do not need these lines)

pdf("MCS.pdf")
dYLim <- 0.07
plot(NULL, xlim=c(1,16), ylim=c(0,dYLim), ylab="MCS p-values", xlab="Configurations of parameters",xaxt="n", yaxt="n")
axis(1,at=1:16,labels=mPValue[,1])
axis(2,at=(0:7)*0.01,labels=(0:7)*0.01)
for (i in 1:16) {
	segments(x0=i,x1=i,y0=0,y1=mPValue[i,2],lwd=5)
}
abline(h=0.05,col="grey")
abline(h=0.01,col="grey",lty=2)
dev.off()

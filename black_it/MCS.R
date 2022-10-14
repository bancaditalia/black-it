# Reading the data

mData <- read.csv("Data.csv")
attach(mData)

fnComplete <- function(x,l=301) {
	return(c(x,rep(x[length(x)],l-length(x))))
}

# Building the MCS

vZ <- 300+4.9*(1:301)

mDist <- NULL
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
mDist <- as.data.frame(mDist)
names(mDist) <- c("Dist","COP")

library(data.table) 
mD <- setDT(mDist)[,list(mean=mean(Dist),var=var(Dist)),by=c("COP")]

fnTest <- function(vMean,vVar,iN,dA=1) {
	iM <- length(vMean)
	if (iM>2) {
		mA <- cbind(rep(1,iM-1),diag(-rep(1,iM-1)))
		mVar <- diag(vVar[-1])
		mVar2 <- mVar+vVar[1]
		dTest <- drop(iN*t(mA%*%vMean)%*%solve(mVar2)%*%(mA%*%vMean))
	} else {
		dTest <- iN*(vMean[1]-vMean[2])^2/(vVar[1]+vVar[2])
	}
	if (iM>1) {
		dQuan <- qchisq(1-dA,iM-1)
		dPVa <- 1-pchisq(dTest,iM-1)
		} else {
		dQuan <- 0
		dPVa <- 1
	}
	dResult <- 1*(dTest>dQuan)
	if (is.na(dResult)) dResult <- 0
	return(list(test=dTest,q=dQuan,p=dPVa,result=dResult))
}

fnElim	<- function(vMean,vVar,vIndex) {
	iM <-	which.max(vMean)
	return(list(mean=vMean[-iM],var=vVar[-iM],indices=vIndex[-iM],elim=vIndex[iM]))
}

fnMCS <- function(vMean,vVar,vIndex,dA,verbose=1) {
	iI <- length(vIndex)
	iStop <- fnTest(vMean,vVar,dA)$result
	mPValue <- matrix(0,ncol=2,nrow=16)
	for (i in 1:iI) {
		lTest <- fnTest(vMean,vVar,iN,dA)
		iStop <- lTest$result
		if (iStop!=1) break
		lElim <- fnElim(vMean,vVar,vIndex)
		vMean <- lElim$mean
		vVar <- lElim$var
		vIndex <- lElim$indices
	 	if (verbose==1) cat("Eliminated COP: ",lElim$elim,"; ","p-value: ", lTest$p,"\n", sep="")
	 	if (verbose==1) cat("Remaining COPs:",vIndex,"\n")
		mPValue[i,] <- c(lElim$elim,lTest$p)
	}
	if (i==iI) {
		mPValue[iI,1] <- setdiff(vIndex,mPValue[1:15,1])
		mPValue[iI,2] <- 1
	} else {
		mPValue <- mPValue[1:(i-1),]
		mPValue[,2] <- cummax(mPValue[,2])
	}
	return(mPValue)
}

vIndex <- 1:16
vMean <- mD$mean
vVar <- mD$var
dA <- 1
iN <- 200

mPValue <- fnMCS(vMean,vVar,vIndex,dA,verbose=0)

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

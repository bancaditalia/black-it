# Reading the data

mData <- read.csv("Data.csv")
attach(mData)

fnComplete <- function(x,l=301) {
	return(c(x,rep(x[length(x)],l-length(x))))
}

# Estimating the rate functions

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
}
mDist <- as.data.frame(mDist)
names(mDist) <- c("Dist","COP")

library(data.table) 
mD <- setDT(mDist)[,list(mean=mean(Dist),var=var(Dist)),by=c("COP")]

fnLmbdStar <- function(y,vDist) {
	u <- seq(-200,+400,by=0.0005)
	vLmbd <- log(apply(exp(outer(vDist,u)),2,mean))
	vC <- diff(vLmbd)/diff(u)
	vC <- c(-Inf,vC,+Inf)
	vIndex <- findInterval(x=y,vec=cummax(vC))
	vU <- u[vIndex]
	if (length(y)>1) {
		return(y*vU-log(apply(exp(outer(vDist,vU)),2,mean)))
	} else {
		return(y*vU-log(mean(exp(vDist*vU))))
	}
}

iMesh <- 10000
mY <- matrix(0,ncol=iMesh, nrow=16)
mF <- matrix(0,ncol=iMesh, nrow=16)
for (j in 1:16) {
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	y <- seq(min(vDist),max(vDist),length.out=iMesh)
	mY[j,] <- y
	mF[j,] <- fnLmbdStar(y,vDist)
}

dMin <- min(mDist[,1])
dMax <- max(mDist[,1])

pdf("RateFunctions.pdf")
layout(matrix(1:16,nrow=4,ncol=4))
par(mar=c(0,0,0,0),omi=c(0.5,0.5,0.5,0.5))
for (j in 1:3) {
	plot(mY[j,],mF[j,],xlab="",ylab="",ylim=c(0,10),xlim=c(dMin,dMax),type="l",xaxt="n",yaxt="n")
	axis(2,at=(0:4)*2,labels=(0:4)*2)
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	abline(v=min(vDist),lty=2)
	abline(v=max(vDist),lty=2)
	legend(-0.15,8,j,box.col="white",bg="white",cex=2,xjust=0,yjust=0.5,bty="n")
}
for (j in 4) {
	plot(mY[j,],mF[j,],xlab="",ylab="",ylim=c(0,10),xlim=c(dMin,dMax),type="l",xaxt="n",yaxt="n")
	axis(1,c(0,0.5,1.0),c(0.0,0.5,1.0))
	axis(2,at=(0:4)*2,labels=(0:4)*2)
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	abline(v=min(vDist),lty=2)
	abline(v=max(vDist),lty=2)
	legend(-0.15,8,j,box.col="white",bg="white",cex=2,xjust=0,yjust=0.5,bty="n")
}
for (j in 5:7) {
	plot(mY[j,],mF[j,],xlab="",ylab="",ylim=c(0,10),xlim=c(dMin,dMax),type="l",xaxt="n",yaxt="n")
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	abline(v=min(vDist),lty=2)
	abline(v=max(vDist),lty=2)
	legend(-0.15,8,j,box.col="white",bg="white",cex=2,xjust=0,yjust=0.5,bty="n")
}
for (j in 8) {
	plot(mY[j,],mF[j,],xlab="",ylab="",ylim=c(0,10),xlim=c(dMin,dMax),type="l",xaxt="n",yaxt="n")
	axis(1,c(0,0.5,1.0),c(0.0,0.5,1.0))
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	abline(v=min(vDist),lty=2)
	abline(v=max(vDist),lty=2)
	legend(-0.15,8,j,box.col="white",bg="white",cex=2,xjust=0,yjust=0.5,bty="n")
}
for (j in 9:11) {
	plot(mY[j,],mF[j,],xlab="",ylab="",ylim=c(0,10),xlim=c(dMin,dMax),type="l",xaxt="n",yaxt="n")
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	abline(v=min(vDist),lty=2)
	abline(v=max(vDist),lty=2)
	legend(-0.15,8,j,box.col="white",bg="white",cex=2,xjust=0,yjust=0.5,bty="n")
}
for (j in 12) {
	plot(mY[j,],mF[j,],xlab="",ylab="",ylim=c(0,10),xlim=c(dMin,dMax),type="l",xaxt="n",yaxt="n")
	axis(1,c(0,0.5,1.0),c(0.0,0.5,1.0))
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	abline(v=min(vDist),lty=2)
	abline(v=max(vDist),lty=2)
	legend(-0.15,8,j,box.col="white",bg="white",cex=2,xjust=0,yjust=0.5,bty="n")
}
for (j in 13:15) {
	plot(mY[j,],mF[j,],xlab="",ylab="",ylim=c(0,10),xlim=c(dMin,dMax),type="l",xaxt="n",yaxt="n")
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	abline(v=min(vDist),lty=2)
	abline(v=max(vDist),lty=2)
	legend(-0.15,8,j,box.col="white",bg="white",cex=2,xjust=0,yjust=0.5,bty="n")
}
for (j in 16) {
	plot(mY[j,],mF[j,],xlab="",ylab="",ylim=c(0,10),xlim=c(dMin,dMax),type="l",xaxt="n",yaxt="n")
	axis(1,c(0,0.5,1.0),c(0.0,0.5,1.0))
	vDist <- c(t(mDist[as.vector(mDist[,2]==j),1]))
	abline(v=min(vDist),lty=2)
	abline(v=max(vDist),lty=2)
	legend(-0.15,8,j,box.col="white",bg="white",cex=2,xjust=0,yjust=0.5,bty="n")
}
dev.off()

fnLS <- function(y,j) {
	iIndex <- findInterval(x=y,vec=mY[j,])
	if (y<=max(mY[j,]) & y>=min(mY[j,])) {
		return( mF[j,iIndex]+(mF[j,iIndex+1]-mF[j,iIndex])*(y-mY[j,iIndex])/(mY[j,iIndex+1]-mY[j,iIndex]) )	
	} else {
		return(10000)
	}
}

fnLSTot <- function(vY,j) {
	dF <- fnLS(vY[1],j)
	k <- 2
	for (i in setdiff(1:16,j)) {
		dF <- dF+fnLS(vY[1]+vY[k]^2,i)
		k <- k+1
	}
	return(dF)
}

mRF <- matrix(0,ncol=3,nrow=15)
k <- 1
for (h in setdiff(1:16,which.min(mD$mean))) {
	iJ <- h
	vY0 <- rep(0,16)
	vY0[1] <- min(mY[iJ,])+.Machine$double.eps^.25
	for (i in 2:16) {
		vY0[i] <- .Machine$double.eps^.25
	}
	lEst <- optim(vY0,fnLSTot,method="Nelder-Mead",control=list(maxit=10000,abstol=.Machine$double.eps),j=iJ)
	lEst <- optim(lEst$par,fnLSTot,method="BFGS",control=list(maxit=10000,abstol=.Machine$double.eps),j=iJ)
	lEst
	mRF[k,] <- c(h,lEst$value,lEst$convergence)
	k <- k+1
}
mRF <- mRF[order(mRF[,2],decreasing = T),]
dfRF <- as.data.frame(mRF)
names(dfRF) <- c("COP","minimum","convergence")


# Reading the data

mData <- read.csv("Data.csv")
attach(mData)

fnComplete <- function(x,l=301) {
	return(c(x,rep(x[length(x)],l-length(x))))
}

# Plotting the data

vZ <- 300+4.9*(1:301)

pdf("Patterns.pdf")
layout(matrix(1:16,nrow=4,ncol=4))
par(mar=c(0,0,0,0),omi=c(0.5,0.5,0.5,0.5))
for (j in 1:3) {
	mDataCOP <- mData[mData$COP==j,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	plot(1:301,fnComplete(mDataCOP[mDataCOP$run==dRun0,]$pr.solved2),xlab="",ylab="",ylim=c(0,2000),xlim=c(0,301),type="l",xaxt="n",yaxt="n")
	axis(2,at=(1:4)*500,labels=(1:4)*500)
	text(40,1800,labels=j,cex=2)
	for (j in 2:200) {lines(1:301,fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2))}
	lines(1:301,vZ,lwd=4,col="grey",lty=1)
}
for (j in 4) {
	mDataCOP <- mData[mData$COP==j,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	plot(1:301,fnComplete(mDataCOP[mDataCOP$run==dRun0,]$pr.solved2),xlab="",ylab="",ylim=c(0,2000),xlim=c(0,301),type="l",yaxt="n")
	axis(2,at=(1:4)*500,labels=(1:4)*500)
	text(40,1800,labels=j,cex=2)
	for (j in 2:200) {lines(1:301,fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2))}
	lines(1:301,vZ,lwd=4,col="grey",lty=1)
}
for (j in 5:7) {
	mDataCOP <- mData[mData$COP==j,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	plot(1:301,fnComplete(mDataCOP[mDataCOP$run==dRun0,]$pr.solved2),xlab="",ylab="",ylim=c(0,2000),xlim=c(0,301),type="l",xaxt="n",yaxt="n")
	text(40,1800,labels=j,cex=2)
	for (j in 2:200) {lines(1:301,fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2))}
	lines(1:301,vZ,lwd=4,col="grey",lty=1)
}
for (j in 8) {
	mDataCOP <- mData[mData$COP==j,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	plot(1:301,fnComplete(mDataCOP[mDataCOP$run==dRun0,]$pr.solved2),xlab="",ylab="",ylim=c(0,2000),xlim=c(0,301),type="l",yaxt="n")
	text(40,1800,labels=j,cex=2)
	for (j in 2:200) {lines(1:301,fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2))}
	lines(1:301,vZ,lwd=4,col="grey",lty=1)
}
for (j in 9:11) {
	mDataCOP <- mData[mData$COP==j,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	plot(1:301,fnComplete(mDataCOP[mDataCOP$run==dRun0,]$pr.solved2),xlab="",ylab="",ylim=c(0,2000),xlim=c(0,301),type="l",xaxt="n",yaxt="n")
	text(40,1800,labels=j,cex=2)
	for (j in 2:200) {lines(1:301,fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2))}
	lines(1:301,vZ,lwd=4,col="grey",lty=1)
}
for (j in 12) {
	mDataCOP <- mData[mData$COP==j,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	plot(1:301,fnComplete(mDataCOP[mDataCOP$run==dRun0,]$pr.solved2),xlab="",ylab="",ylim=c(0,2000),xlim=c(0,301),type="l",yaxt="n")
	text(40,1800,labels=j,cex=2)
	for (j in 2:200) {lines(1:301,fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2))}
	lines(1:301,vZ,lwd=4,col="grey",lty=1)
}
for (j in 13:15) {
	mDataCOP <- mData[mData$COP==j,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	plot(1:301,fnComplete(mDataCOP[mDataCOP$run==dRun0,]$pr.solved2),xlab="",ylab="",ylim=c(0,2000),xlim=c(0,301),type="l",xaxt="n",yaxt="n")
	text(40,1800,labels=j,cex=2)
	for (j in 2:200) {lines(1:301,fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2))}
	lines(1:301,vZ,lwd=4,col="grey",lty=1)
}
for (j in 16) {
	mDataCOP <- mData[mData$COP==j,]
	vRun <- unique(mDataCOP$run)
	dRun0 <- min(vRun)
	plot(1:301,fnComplete(mDataCOP[mDataCOP$run==dRun0,]$pr.solved2),xlab="",ylab="",ylim=c(0,2000),xlim=c(0,301),type="l",yaxt="n")
	text(40,1800,labels=j,cex=2)
	for (j in 2:200) {lines(1:301,fnComplete(mDataCOP[mDataCOP$run==(dRun0-1+j),]$pr.solved2))}
	lines(1:301,vZ,lwd=4,col="grey",lty=1)
}
dev.off()

import matplotlib.pyplot as plt
import statistics as st



fo = open("out3","r+")

x= range(1,1001,1)
y=[]
summ=0
for line in fo:
    y.append(int(line))
    summ += int(line) 
    
    
plt.hist(y, bins=20)    
plt.xlabel('Number of steps')
plt.ylabel('Frequency')
plt.title('Primary agent taking random steps (Initial Distance > 3)') 
plt.savefig("hist3.png") 
plt.clf()

print ( " Max of array ", max(y))
print ( " Mean of array ", st.mean(y))
print ( " Median of array ", st.median(y)) 
print ( " Standard Variance of array ", st.stdev(y))
fo.close()






fo = open("out6","r+")


x= range(1,1001,1)
y=[]
summ=0
for line in fo:
    y.append(int(line))
    summ += int(line) 
    
    
plt.hist(y, bins=20)    
plt.xlabel('Number of steps')
plt.ylabel('Frequency')
plt.title('Primary agent taking random steps (Initial Distance > 6)') 
plt.savefig("hist6.png") 

print ( " Max of array ", max(y)) 
print ( " Mean of array ", st.mean(y))
print ( " Median of array ", st.median(y)) 
print ( " Standard Variance of array ", st.stdev(y))
fo.close()


compile: PCA.cpp
	g++ -o pca *.cpp --std=c++11

clean:
	rm pca
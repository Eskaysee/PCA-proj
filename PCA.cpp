#include <iostream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include <fstream>

namespace CLNSIH001{
    using namespace std;
    using namespace Eigen;
    struct PC{
        double component;   //Principle Component value
        Vector2d eigenvec;  //Principle Component eigenvector
        double eigenval;    //Principle Component eigenvalue
    };
    struct PCA
    {
        //The two vectors (each being a variable) form the inputMatrix, each comprising of 64 data-elements
        std::vector<double> Jan = {51.2, 51.2, 39.5, 45.1, 29.9, 24.8, 32, 35.6, 54.6, 67.2, 42.4, 29, 22.9, 23.8, 27.9, 19.4, 31.3, 33.3, 52.9, 21.5, 33.4, 29.2, 25.5, 14.2, 8.5, 12.2, 47.1, 27.8, 31.3, 20.5, 22.6, 31.9, 20.6, 32.7, 35.2, 21.5, 23.7, 32.2, 42.1, 40.5, 8.2, 31.1, 26.9, 28.4, 36.8, 38.1, 32.3, 28.1, 28.4, 45.4, 14.2, 40.5, 38.3, 44.8, 43.6, 52.1, 28, 16.8, 40.5, 37.5, 25.4, 34.5, 19.4, 26.6};
        std::vector<double> July = {81.6, 91.2, 81.4, 75.2, 73, 72.7, 75.8, 78.7, 81, 82.3, 78, 74.5, 71.9, 75.1, 75, 75.1, 80.7, 76.9, 81.9, 68, 76.6, 73.3, 73.3, 63.8, 65.6, 71.9, 81.7, 78.8, 78.6, 69.3, 77.2, 69.3, 69.7, 75.1, 78.7, 72, 70.1, 76.6, 78.5, 77.5, 70.8, 75.6, 71.4, 73.6, 81.5, 67.1, 76.8, 71.9, 72.1, 81.2, 73.3, 79.6, 79.6, 84.8, 82.3, 83.3, 76.7, 69.8, 78.3, 77.9, 69.7, 75, 69.9, 69.1};
        //Covariance Matrix
        Matrix<double, 2, 2> covMatrix;
        //Principle Component
        PC pc1, pc2;
        //Total Variance
        double totalVar = 0.0;
        PCA(){
            createMatrix();
            eigens();
            totalVar = TSV();
        }

        //Calculates the eigenvectors and eigenvalues
        void eigens(){
            SelfAdjointEigenSolver<Matrix2d> EigenSolver(covMatrix);
            Vector2d v = EigenSolver.eigenvalues();
            if (v(0) > v(1)){
                pc1.eigenvec = EigenSolver.eigenvectors().col(0);
                pc1.eigenval = v(0);
                pc2.eigenvec = EigenSolver.eigenvectors().col(1);
                pc2.eigenval = v(1);
            }
            else {
                pc1.eigenvec = EigenSolver.eigenvectors().col(1);
                pc1.eigenval = v(1);
                pc2.eigenvec = EigenSolver.eigenvectors().col(0);
                pc2.eigenval = v(0);
            }
            findPC(pc1.eigenvec, pc2.eigenvec);
        }

        //works out the Principle Component Value
        void findPC(Vector2d e1, Vector2d e2){
            RowVector2d X;
            X(0) = e1.mean();
            X(1) = e2.mean();
            pc1.component = X.dot(e1);
            pc2.component = X.dot(e2);
        }
        
        //finds the average of the datapoints for a variable 
        double avg(vector<double> v){
            double sumV=0.0;
            for (int i = 0; i < 64; ++i)
            {
                sumV += v.at(i);
            }
            return sumV/64;
        }

        //creates covariance matrix
        void createMatrix(){
            covMatrix(0,0) = covariance(Jan, Jan);
            covMatrix(0,1) = covariance(Jan, July);
            covMatrix(1,0) = covariance(July, Jan);
            covMatrix(1,1) = covariance(July, July);
        }

        //Total Sample Variance
        double TSV(){
            return (double)(covariance(Jan, Jan) + covariance(July, July));
        }

        //finds each element in the covariance matrix
        double covariance(vector<double> v1, vector<double> v2){
            double sigma = 0.0;
            double avgV1=0.0, avgV2=0.0;
            avgV1 = avg(v1); avgV2 = avg(v2);
            for (int a=0; a<64; a++){
                sigma += (v1.at(a) - avgV1)*(v2.at(a) - avgV2);
            }
            return (double)(sigma/63);
        }

        friend ostream & operator<<(std::ostream& os, const PCA& PCA);
    };
    ostream & operator<<(std::ostream& os, const PCA& PCA){
        os << "Where the principle component value is the linear combination of the eigenvector elements..." << '\n';
        os << "The first principle component is " << PCA.pc1.component << " and the second principle component is " << PCA.pc2.component << ".\n";
        os << "The principal component algorithm answers the following questions.\n";
        os << "\n------------------------------------------------------------------------------------------------------------------------------------------------------\n";
        
        os << "\n1. What are the Eigenvalues for the principal components 1 and 2?\n";
        os << "Principle Component 1's Eigenvalue is: " << PCA.pc1.eigenval << "\nPrinciple Component 2's Eigenvalue is: " << PCA.pc2.eigenval << '\n';
        os << '\n';
        os << "2. What are the Eigenvectors for the principal components 1 and 2 (showing July and January component values for each)?\n";
        os << "Principle Component 1's Eigenvector is:\n" << PCA.pc1.eigenvec << '\n' << "where January is " << PCA.pc1.eigenvec(0) << " and July is " << PCA.pc1.eigenvec(1) << '\n';
        os << "Principle Component 2's Eigenvector is:\n" << PCA.pc2.eigenvec << '\n' << "where Janaury is " << PCA.pc2.eigenvec(0) << " and July is " << PCA.pc2.eigenvec(1) << '\n';
        os << '\n';
        os << "3. Compute the values for the covariance matrix.\n";
        os << PCA.covMatrix << '\n';
        os << '\n';
        os << "4. What is the total variance?\n";
        os << "Total Sample Variance (TSV) is " << PCA.totalVar << '\n' << "where TSV is the sum for the variance of January and July" << '\n';
        os << '\n';
        os << "5. What proportion (as a percentage) of total variance do principal components 1 and 2 ”explain”?\n";
        os << "Where the principle component value is the linear combination the eigenvector elements..." << '\n';
        os << "(PC1's eigenVal/TSV)x100 = (" << PCA.pc1.eigenval << "/" << PCA.totalVar << ")x100 = " << (PCA.pc1.eigenval/PCA.totalVar)*100 << "%\n";
        os << "(PC2's eigenVal/TSV)x100 = (" << PCA.pc2.eigenval << "/" << PCA.totalVar << ")x100 = " << (PCA.pc2.eigenval/PCA.totalVar)*100 << "%\n";
        os << '\n';
        return os;
    }
}

using namespace CLNSIH001;
int main(int argc, char * argv[]){
    CLNSIH001::PCA pca;
    ofstream ans("Answers.txt", ios::out);
    ans << pca;
    ans.close();
    cout << "The answer to the questions have been written to Answer.txt" << endl;
    return 0;
}
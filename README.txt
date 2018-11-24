Instructions to run the program
Download the project from the following github repository:
https://github.com/MelannieTorres-academico/AI_Perceptron
To run the java program enter the java folder:  cd fork-join-perceptron
Compile the java programs type the following command:  javac *.java
To run the example type: java MainForkJoinPerceptron < ./tests/breast_cancer.txt
To run your own dataset type: java MainForkJoinPerceptron < path_to_your_dataset.txt
Go back to the main folder: cd ..
Go into the C++ folder: cd c++
To compile the C++ files type:  g++ sequential.cpp 
To run the program type: ./a.out < ./tests/breast_cancer.txt
 To run the program with your own dataset type: ./a.out < path_to_your_dataset.txt
Go back to the main folder: cd ..
 Go into the python folder: cd python
 Run the python program: python3 ann.py < ./tests/breast_cancer.txt
 To run with your own dataset: python3 ann.py < path_to_your_dataset

Take into account the following considerations when creating your own dataset:
The file should be a txt
The values should be separated by commas
The dataset must be all numeric
The dataset must be linearly separable
The output must be binary (0 or 1)
The first number indicates the number of attributes
The second number indicates the number of records used to train
The third number indicates the number of tests to run
The following lines should be the training records and the tests to run

The breast cancer dataset was taken from:

Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of
Information and Computer Science. Retrieved from
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/b
reast-cancer-wisconsin.data


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>
#include <cstdlib>
#include "test.h"   
#include "KNN.h"

using namespace std;

bool load_iris(const string& path, vector<vector<double>>& X, vector<string>& y){
    ifstream file(path);
    if(!file.is_open()) return false;

    string line;
    getline(file, line);                      
    while(getline(file, line)){
        if(!line.empty() && line.back() == '\r') line.pop_back();
        if(line.empty()) continue;

        stringstream ss(line);
        string cell;
        getline(ss, cell, ',');                 

        vector<double> row;
        for(int i=0; i<4; i++){
            getline(ss, cell, ',');
            row.push_back(stod(cell));
        }
        getline(ss, cell, ',');                   
        X.push_back(row);
        y.push_back(cell);
    }
    return true;
}

int read_int(const string& prompt, int lo, int hi){
    int v;
    while(true){
        cout << prompt;
        if(cin >> v && v >= lo && v <= hi) return v;
        if(cin.eof()){ cerr << "\nNo input.\n"; exit(1); }
        cout << "Please enter a number between " << lo << " and " << hi << ".\n";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }
}

int main(int argc, char* argv[]){
    string path = (argc > 1) ? argv[1] : "Iris.csv";

    vector<vector<double>> X;
    vector<string> y;
    if(!load_iris(path, X, y)){
        cerr << "Could not open " << path << endl;
        return 1;
    }
    int n = X.size();
    cout << "Loaded " << n << " samples.\n";

    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    mt19937 rng(42);                              
    shuffle(idx.begin(), idx.end(), rng);

    int n_train = static_cast<int>(0.8 * n);     
    vector<vector<double>> train_X, test_X;
    vector<string> train_y, test_y;
    for(int i=0; i<n; i++){
        if(i < n_train){ train_X.push_back(X[idx[i]]); train_y.push_back(y[idx[i]]); }
        else           { test_X.push_back(X[idx[i]]);  test_y.push_back(y[idx[i]]);  }
    }
    int n_test = test_X.size();
    cout << "Train: " << n_train << "  Test: " << n_test << "\n\n";

    TrainDataSet train_set(train_X, train_y);
    TestDataSet  test_set(test_X, test_y, train_set);

    vector<vector<double>> train_std, test_std;
    for(int i=0; i<n_train; i++){
        vector<double> f = train_X[i];
        string l = train_y[i];
        train_std.push_back(Datapoint(f, l, train_set).stand_features);
    }
    for(int i=0; i<n_test; i++){
        vector<double> f = test_X[i];
        string l = test_y[i];
        test_std.push_back(Datapoint(f, l, test_set).stand_features);
    }

    int k = read_int("Enter K (1-" + to_string(n_train) + "): ", 1, n_train);


    KNN knn(k);
    knn.fit(train_std, train_y);
    int correct = 0;
    map<string, map<string,int>> confusion;      
    for(int i=0; i<n_test; i++){
        string pred = knn.predict(test_std[i]);
        confusion[test_y[i]][pred]++;
        if(pred == test_y[i]) correct++;
    }

    cout << "\nK = " << k << "\n";
    cout << "Accuracy: " << correct << "/" << n_test << " = "
         << 100.0 * correct / n_test << "%\n\n";
    cout << "Confusion matrix (rows = actual, cols = predicted):\n";
    for(auto& [actual, row] : confusion){
        cout << "  " << actual << ": ";
        for(auto& [pred, cnt] : row) cout << pred << "=" << cnt << "  ";
        cout << "\n";
    }
    return 0;
}

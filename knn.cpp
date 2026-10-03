#include "KNN.h"
#include <cmath>
#include <algorithm>
#include <map>
using namespace std;
KNN::KNN(int k){
    this->k=k;
}
double KNN::euclideanDistance(const vector<double>& a,const vector<double>& b) const{
    double sum=0.0;
    for(int i=0;i<a.size();i++){
        double difference=a[i]-b[i];
        sum+=difference*difference;
    }
    return sqrt(sum);
}
void KNN::fit(const vector<vector<double>>& x,
              const vector<string>& y) {

    x_train = x;
    y_train = y;
}
string KNN::predict(const vector<double>& x) const{
    vector<pair<double,string>> distances;
    for(int i=0;i<x_train.size();i++){
        double distance=euclideanDistance(x,x_train[i]);
        distances.push_back({distance,y_train[i]});
    }
    sort(distances.begin(),distances.end());
    map<string,int> votes;
    for(int i=0;i<k;i++){
        votes[distances[i].second]++;
    }
    string prediction;
    int maxVotes =0;
    for(const auto& vote : votes){
        if(vote.second > maxVotes){
            maxVotes=vote.second;
            prediction=vote.first;
        }
    }
    return prediction;
}
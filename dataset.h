#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;


class DataSet{
    protected:
        vector<vector<double>> samples_arrays;
        vector<string> labels;
        vector<double> means;
        vector<double> variance;
        vector<vector<double>> features_arrays;
        int no_samples;
        int no_features;

        void cal_Stats(){
            for(int i=0; i<no_features; i++){
                double temp_sum=0;
                double temp_var_sum = 0;

                for(double ele: features_arrays[i]){
                    temp_sum += ele;
                }
                means[i] = temp_sum/no_samples;

                for(double ele: features_arrays[i]){
                    temp_var_sum += (ele - means[i]) * (ele - means[i]);

                }
                variance[i] = temp_var_sum/no_samples;

            }           
        }
        
        void extract_to_features(){

            for(int i=0; i<no_features; i++){
                for(int j=0; j<no_samples; j++){
                features_arrays[i][j] = samples_arrays[j][i];
                }
            }
        }


        void init_structure(){
            no_samples = samples_arrays.size();
            no_features = samples_arrays[0].size();
            means.resize(no_features);
            variance.resize(no_features);
            labels.resize(no_samples);
            
            features_arrays.resize(no_features);
            for(int i=0; i<no_features; i++){
                features_arrays[i].resize(no_samples);
            }
            extract_to_features();
        }

        void compute_stats(){
            cal_Stats();
        }

    public:
        DataSet(vector<vector<double>> &s, vector<string> &l) {
            samples_arrays = s;
            labels = l;
            init_structure();
        }




        const vector<vector<double>>& get_features() const {
            return features_arrays;
        }

        const vector<vector<double>>& get_samples() const {
            return samples_arrays;
        }

        vector<double> get_mean() const {
            return means;
        }

        vector<double> get_variance() const{
            return variance;
        }
        
        vector<string> get_labels() const{
            return labels;
        }

        int get_no_samples() const{
            return no_samples;
        }

        int get_no_features() const{
            return no_features;
        }

};

class TrainDataSet : public DataSet{
    public:

        TrainDataSet(vector<vector<double>> &train_samples, vector<string> &l) : DataSet(train_samples, l){
            compute_stats();
        }
};


class TestDataSet : public DataSet{
    private:

    public:
        TestDataSet(vector<vector<double>> &test_samples, vector<string> &l, TrainDataSet& td1) : DataSet(test_samples, l) {
            this->means = td1.get_mean();
            this->variance = td1.get_variance();
        }

};

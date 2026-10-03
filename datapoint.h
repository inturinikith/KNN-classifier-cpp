#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "dataset.h"

using namespace std;

class Distance;

class Datapoint{
private:

    vector<double> features;
    string label;

    DataSet& d;

public:

    vector<double> stand_features;

    vector<double> standardise(vector<double>& sample_array){

        vector<double> temp(d.get_no_features());

        for(int i=0; i<d.get_no_features(); i++){

            temp[i] = (sample_array[i] - d.get_mean()[i])/sqrt(d.get_variance()[i]);

        }

        return temp;

    }

    Datapoint(vector<double>& sample_array, string& sample_label, DataSet& d1) : d(d1){

        features = sample_array;

        label = sample_label;

        stand_features = standardise(sample_array);

    }

    string get_label() const{

        return label;

    }

    void show(){

        cout << "Features: [";

        for(double ele: features){

            cout << ele << " ";

        }

        cout << "]" << endl;

        cout << "Standard Features: [";

        for(double ele: stand_features){

            cout << ele << " ";

        }

        cout << "]" << endl;

    }

    double distance_to(const Datapoint& d1, const Distance& metric);

};

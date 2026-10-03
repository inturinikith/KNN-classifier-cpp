#pragma once
#include <cmath>
#include "datapoint.h"

class Distance{
public:

    virtual double calculate(const Datapoint& dt1, const Datapoint& dt2) const = 0;

};

inline double Datapoint::distance_to(const Datapoint& d1, const Distance& metric){

    return metric.calculate(*this, d1);

}

class Euclidean : public Distance{
public:

    double calculate(const Datapoint& dt1, const Datapoint& dt2) const override {

        double sq_dis = 0;

        for(int i=0; i<dt1.stand_features.size(); i++){

            sq_dis += (dt1.stand_features[i] - dt2.stand_features[i]) *
                      (dt1.stand_features[i] - dt2.stand_features[i]);

        }

        return sqrt(sq_dis);

    }

};

class Manhattan : public Distance{
public:

    double calculate(const Datapoint& dt1, const Datapoint& dt2) const override {

        double dis = 0;

        for(int i=0; i<dt1.stand_features.size(); i++){

            dis += abs(dt1.stand_features[i] - dt2.stand_features[i]);

        }

        return dis;

    }

};

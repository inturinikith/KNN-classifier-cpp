#pragma once

#include <vector>
#include <string>
#include "datapoint.h"
#include "dataset.h"
#include "distance.h"

using namespace std;

class KNN {
private:
    int k;
    vector<Datapoint> train_points;
    const Distance* metric;

public:
    KNN(int k, const Distance& metric);

    void fit(TrainDataSet& train_set);

    string predict(Datapoint& test_point) const;
};

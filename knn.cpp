#include "knn.h"

#include <algorithm>
#include <map>

using namespace std;

KNN::KNN(int k, const Distance& metric) {
    this->k = k;
    this->metric = &metric;
}

void KNN::fit(TrainDataSet& train_set) {

    train_points.clear();

    for (int i = 0; i < train_set.get_no_samples(); i++) {

        vector<double> features = train_set.get_samples()[i];
        string label = train_set.get_labels()[i];

        train_points.emplace_back(features, label, train_set);
    }
}

string KNN::predict(Datapoint& test_point) const {

    vector<pair<double, string>> distances;

    // Calculate distance from test point
    // to every training point
    for (const Datapoint& point : train_points) {

        double distance =
            test_point.distance_to(point, *metric);

        distances.push_back({distance, point.get_label()});
    }

    // Sort from smallest distance to largest
    sort(distances.begin(), distances.end());

    // Majority voting
    map<string, int> votes;

    for (int i = 0; i < k; i++) {
        votes[distances[i].second]++;
    }

    // Find class with maximum votes
    string prediction;
    int max_votes = 0;

    for (const auto& vote : votes) {

        if (vote.second > max_votes) {
            max_votes = vote.second;
            prediction = vote.first;
        }
    }

    return prediction;
}

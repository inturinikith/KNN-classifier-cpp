#include "knn.h"

#include <algorithm>
#include <map>
#include <limits>
#include <iostream>

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

    for (const Datapoint& point : train_points) {

        double distance =
            test_point.distance_to(point, *metric);

        distances.push_back({distance, point.get_label()});
    }

    sort(distances.begin(), distances.end());

    map<string, int> votes;
    map<string, double> total_distances;

    int neighbors = min(k, static_cast<int>(distances.size()));

    for (int i = 0; i < neighbors; i++) {

        string label = distances[i].second;
        double distance = distances[i].first;

        votes[label]++;
        total_distances[label] += distance;
    }

    string prediction;
    int max_votes = -1;
    double min_distance = numeric_limits<double>::infinity();

    for (const auto& vote : votes) {

        string label = vote.first;
        int count = vote.second;

        if (count > max_votes) {
            max_votes = count;
            prediction = label;
        }
    }

    vector<string> tied_classes;

    for (const auto& vote : votes) {
        if (vote.second == max_votes) {
            tied_classes.push_back(vote.first);
        }
    }

    if (tied_classes.size() > 1) {

        cout << "\nTie breaker happened!\n";

        for (const string& label : tied_classes) {
            cout << label
                 << " -> "
                 << votes[label]
                 << " votes, total distance = "
                 << total_distances[label]
                 << "\n";
        }

        for (const string& label : tied_classes) {

            if (total_distances[label] < min_distance) {
                min_distance = total_distances[label];
                prediction = label;
            }
        }

        cout << "Tie breaker winner: "
             << prediction
             << "\n";
    }

    return prediction;
}

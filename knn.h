#include <vector>
#include <string>
using namespace std;
class KNN{
private:
    int k;
    vector<vector<double>> x_train;
    vector<string> y_train;
    double euclideanDistance(const vector<double>& a,const vector<double>& b) const;
public:
    KNN(int k);
    void fit(const vector<vector<double>>& x,
             const vector<string>& y);
     string predict(const vector<double>& x) const;
};
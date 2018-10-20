#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <map>
#include <any>
#include <algorithm>
#include <random>

using namespace std;

typedef pair<long long, double> value_t;

random_device rd;
mt19937 g(rd());

struct kNNClassifier {
    struct grt {
        bool operator() (const value_t& x, const value_t& y) const {return x.second < y.second;}
    };
    struct ls {
        bool operator() (const value_t& x, const value_t& y) const {return x.second > y.second;}
    };
    typedef priority_queue<value_t, vector<value_t>, grt> que_t_grt;
    typedef priority_queue<value_t, vector<value_t>, ls> que_t_ls;
    
    enum metrics {MANH, EUCL};
    metrics metric;
    enum kernels {TRIANGULAR, UNIFORM};
    kernels kernel;
    double win_size;
    size_t f_amount, el_amount, k_neighbours;
    bool use_k_neighbours;
    vector<vector<long long>> data;
    vector<long long> classes;
    vector<size_t> ignore;
    vector<size_t> indices;
    
    kNNClassifier() : metric(MANH), kernel(UNIFORM), win_size(-1) {}
    
    kNNClassifier(map<string, any> f_dict) {
        kNNClassifier();
        set_params(f_dict);
    }
    
    void set_params(map<string, any> const& f_dict) {
        for (auto const& [key, val] : f_dict) {
            if (key == "k_neighbours") {
                k_neighbours = any_cast<size_t>(val);
            }
            if (key == "win_size") {
                win_size = any_cast<double>(val);
            }
            if (key == "kernel") {
                string k_name = any_cast<string>(val);
                if (k_name == "triangular") {
                    kernel = TRIANGULAR;
                }
                if (k_name == "uniform") {
                    kernel = UNIFORM;
                }
            }
            if (key == "metric") {
                string m_name = any_cast<string>(val);
                if (m_name == "manhatttan") {
                    metric = MANH;
                }
                if (m_name == "euclidean") {
                    metric = EUCL;
                }
            }
        }
        
        use_k_neighbours = false;
        if (k_neighbours >= 1 && win_size <= 0) {
            use_k_neighbours = true;
        }
    }
    
    void fit(vector<vector<long long>> const& X, vector<long long> const& y) {
        assert(X.size() == y.size());
        
        data = X;
        f_amount = X[0].size();
        el_amount = X.size();
        classes = y;
        ignore.assign(X.size(), 0);
        indices.resize(X.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
    }
    
    void fit(vector<vector<long long>> const& X, vector<long long> const& y, vector<size_t> const& ign, vector<size_t> const& ind) {
        assert(X.size() == y.size());
        assert(X.size() == ign.size());

        data = X;
        f_amount = X[0].size();
        el_amount = X.size();
        classes = y;
        ignore = ign;
        indices = ind;
    }
    
    vector<vector<value_t>> predict(vector<vector<long long>> const& X) {
        return predict(X, vector<bool>(X.size(), true));
    }
    
    vector<vector<value_t>> predict(vector<vector<long long>> const& X, vector<bool> const& to_val) {
        assert(X.size() == to_val.size());
        
        que_t_grt que;
        vector<double> wts;
        vector<vector <value_t>> prediction(X.size());
        for (size_t el = 0; el < X.size(); ++el) {
            if (!to_val[el]) {
                continue;
            }
            
            wts.assign(el_amount, 0);
            if (use_k_neighbours) {
                win_size = dist(X[el], data[find_k1_neighbour(X[el])]);
            }
            for (size_t i = 0; i < data.size(); ++i) {
                if (ignore[i] != 0) {
                    continue;
                }
                wts[i] += weight(dist(X[el], data[i]));
            }
            que = que_t_grt();
            for (size_t i = 0; i < el_amount; ++i) {
                que.push(value_t(indices[i] + 1, wts[i]));
            }
            
            for (size_t i = 0; que.top().second != 0 && !que.empty(); ++i) {
                prediction[el].push_back(que.top());
                que.pop();
            }
        }
        
        return prediction;
    }
    
private:
    
    double dist(vector<long long> const& a, vector<long long> const& b) {
        assert(a.size() == b.size());
        
        double res = 0;
        switch (metric) {
            case MANH:
                for (size_t i = 0; i < a.size(); ++i) {
                    res += abs(a[i] - b[i]);
                }
                return res;
            case EUCL:
                for (size_t i = 0; i < a.size(); ++i) {
                    res += (a[i] - b[i]) * (a[i] - b[i]);
                }
                return sqrt(res);
        }
    }
    
    double weight(double dist) {
        assert(win_size > 0);
        
        dist /= win_size;
        switch (kernel) {
            case UNIFORM:
                if (dist <= 1 && dist >= -1) {
                    return 0.5;
                }
                return 0;
            case TRIANGULAR:if (dist <= 1 && dist >= -1) {
                return 1 - abs(dist);
            }
                return 0;
        }
    }
    
    size_t find_k1_neighbour(vector<long long> const& el) {
        que_t_ls que = que_t_ls();
        for (size_t i = 0; i < data.size(); ++i) {
            if (ignore[i] != 0) {
                continue;
            }
            que.push(value_t(i, dist(el, data[i])));
        }
        value_t last;
        for (size_t i = 0; i < k_neighbours && !que.empty(); ++i) {
            last = que.top();
            que.pop();
        }
        if (!que.empty()) {
            return que.top().first;
        }
        return last.first;
    }
};

template <typename T>
struct GridSearchCV {
    T classifier;
    size_t cv, cl_amount;
    vector<vector<long long>> data;
    vector<long long> classes;
    vector<map<string, any>> combined_params;
    map<string, any> best_params;
    
    GridSearchCV(map<string, vector<any>> const& p_grid) : GridSearchCV(p_grid, 5) {}
    
    GridSearchCV(map<string, vector<any>> const& p_grid, size_t cv) : cv(cv) {
        combine_params(p_grid);
    }
    
    void fit(vector<vector<long long>> X, vector<long long> y) {
        assert(X.size() == y.size());
        size_t tmp_cv = cv;
        if (X.size() < cv) {
            cv = X.size();
        }
        
        data = X;
        classes = y;
        cl_amount = 0;
        for (auto const& el : classes) {
            if (el > cl_amount) {
                cl_amount = el;
            }
        }
        
        for (size_t i = 0; i < X.size(); ++i) {
            X[i].push_back(y[i]);
            X[i].push_back(i);
        }
        
        shuffle(X.begin(), X.end(), g);
        
        vector<size_t> inds(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            inds[i] = X[i].back();
            X[i].pop_back();
            y[i] = X[i].back();
            X[i].pop_back();
        }
        
        // partition once
        size_t cur = 0;
        vector<size_t> indices(X.size(), 0);
//        uncomment if test set needed
//        for (size_t i = 0; i < X.size(); ++i) {
//            if (cur == 0) {
//                indices[i] = cv + 1;
//            }
//            cur = (cur + 1) % (cv);
//        }
        cur = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            if (indices[i] == 0) {
                indices[i] = cur + 1;
                cur = (cur + 1) % (cv);
            }
        }
        
        vector<size_t> ignore(X.size());
        vector<bool> to_val(X.size());
        vector<vector<value_t>> res;
        vector<long long> eval_res;
        double score, b_score = 0;
        size_t b_ind = 0;
        for (size_t el = 0; el < combined_params.size(); ++el) {
            classifier.set_params(combined_params[el]);
            score = 0;
            for (cur = 0; cur < cv; ++cur) {
                for (size_t i = 0; i < indices.size(); ++i) {
                    ignore[i] = 0;
                    to_val[i] = false;
                    if (indices[i] == cur + 1) {
                        ignore[i] = 1;
                        to_val[i] = true;
                    }
                    if (indices[i] == cv + 1) {
                        ignore[i] = 2;
                    }
                }
                classifier.fit(X, y, ignore, inds);
                res = classifier.predict(X, to_val);
                eval_res = eval_prediction(res);
                score += micro_f_score(eval_res, to_val);
            }
            score /= cv;
            if (score > b_score) {
                b_score = score;
                b_ind = el;
            }
        }
        
        best_params = combined_params[b_ind];
        
        cv = tmp_cv;
    }
    
    vector<vector<value_t>> predict(vector<vector<long long>> const& X) {
        classifier.set_params(best_params);
        classifier.fit(data, classes);
        return classifier.predict(X);
    }
    
private:
    
    void combine_params(map<string, vector<any>> const& param_grid) {
        queue<map<string, any>> que = queue<map<string, any>>();
        size_t it_amount;
        map<string, any> el, tmp;
        
        que.push(map<string, any>());
        for (auto const& [key, val] : param_grid) {
            it_amount = que.size();
            for (size_t i = 0; i < it_amount; ++i) {
                el = que.front();
                que.pop();
                for (any obj : val) {
                    tmp = el;
                    tmp[key] = obj;
                    que.push(tmp);
                }
            }
        }

        while (!que.empty()) {
            combined_params.push_back(que.front());
            que.pop();
        }
    }
    
    vector<long long> eval_prediction(vector<vector<value_t>> const& pr) {
        vector<long long> ans(pr.size());
        
        map<size_t, double> wts;
        vector<double> max_w;
        vector<long long> ans_i;
        for (size_t i = 0; i < pr.size(); ++i) {
            wts.clear();
            max_w.clear();
            ans_i.clear();
            for (size_t j = 0; j < pr[i].size(); ++j) {
                if (wts.find(pr[i][j].first) != wts.end()) {
                    wts[pr[i][j].first] += pr[i][j].second;
                } else {
                    wts[pr[i][j].first] = pr[i][j].second;
                }
            }
            
            for (auto const& [key, val] : wts) {
                if (max_w.size() > 0) {
                    if (val > max_w[0]) {
                        max_w.clear();
                        max_w.push_back(val);
                        ans_i.clear();
                        ans_i.push_back(classes[key]);
                    } else if (val == max_w[0]) {
                        max_w.push_back(val);
                        ans_i.push_back(classes[key]);
                    }
                } else {
                    max_w.push_back(val);
                    ans_i.push_back(classes[key]);
                }
            }
            
            if (ans_i.size() > 0) {
                ans[i] = ans_i[g() % ans_i.size()];
            } else {
                ans[i] = g() % cl_amount + 1;
            }
        }
        
        return ans;
    }
    
    double micro_f_score(vector<long long> const& predicted, vector<bool> const& to_val) {
        vector<long long> tp(cl_amount, 0), fp(cl_amount, 0), fn(cl_amount, 0), sum(cl_amount, 0);
        size_t whole_sum = 0;
        double wgh_precision = 0, wgh_recall = 0;
        
        for (size_t i = 0; i < predicted.size(); ++i) {
            if (to_val[i]) {
                if (classes[i] == predicted[i]) {
                    // tp
                    ++tp[classes[i]];
                }
                if (classes[i] != predicted[i]) {
                    // fp [classes[i]]
                    ++fp[classes[i]];
                    // fn [predicted[i]]
                    ++fn[predicted[i]];
                }
            }
        }
        
        for (size_t i = 0; i < cl_amount; ++i) {
            sum[i] = tp[i] + fp[i];
            whole_sum += sum[i];
        }
        
        for (size_t i = 0; i < cl_amount; ++i) {
            wgh_precision += sum[i] * (sum[i] == 0 ? 0 : (tp[i] / sum[i]));
            wgh_recall = sum[i] * (tp[i] + fn[i] == 0 ? 0 : (tp[i] / (tp[i] + fn[i])));
        }
        
//        wgh_precision /= (whole_sum == 0 ? 1 : whole_sum);
//        wgh_recall /= (whole_sum == 0 ? 1 : whole_sum);
        
//        double ans = wgh_precision + wgh_recall == 0 ? 0 : 2 * wgh_precision * wgh_recall / (wgh_precision + wgh_recall);
        
        return 1.0;
    }
};

int main() {
    vector<vector<long long>> X, test;
    X.resize(4);
    for (size_t i = 0; i < X.size(); ++i) {
        X[i].resize(2);
    }
    X[0][0] = 2;
    X[0][1] = 2;
    X[1][0] = 6;
    X[1][1] = 2;
    X[2][0] = 4;
    X[2][1] = 1;
    X[3][0] = 4;
    X[3][1] = 3;
    test.resize(2);
    for (size_t i = 0; i < test.size(); ++i) {
        test[i].resize(2);
    }
    test[0][0] = 1;
    test[0][1] = 2;
    test[1][0] = 5;
    test[1][1] = 2;
    vector<long long> y;
    y.resize(4);
    y[0] = 1;
    y[1] = 1;
    y[2] = 2;
    y[3] = 2;
    
    map<string, vector<any>> param_grid = map<string, vector<any>> {
        {"k_neighbours", vector<any> {any(static_cast<size_t> (0)), any(static_cast<size_t> (2))}},
        {"win_size", vector<any> {any(static_cast<double> (4.0)), any(static_cast<double> (7.0))}},
        {"kernel", vector<any> {any(static_cast<string> ("triangular"))}},
        {"metric", vector<any> {any(static_cast<string> ("euclidean"))}},
    };
    
    map<string, any> f_dict = map<string, any> {
        {"k_neighbours", any(static_cast<size_t> (0))},
        {"win_size", any(static_cast<double> (4.0))},
        {"kernel", any(static_cast<string> ("triangular"))},
        {"metric", any(static_cast<string> ("euclidean"))},
    };
    
    GridSearchCV<kNNClassifier> cv = GridSearchCV<kNNClassifier> (param_grid);
    
    cv.fit(X, y);
    
    cv.predict(test);
    
    cv.fit(X, y);
    vector<vector<value_t>> res = cv.predict(test);
     for (size_t i = 0; i < res.size(); ++i) {
         cout << res[i].size() << " ";
         for (size_t j = 0; j < res[i].size(); ++j) {
             cout << res[i][j].first << " " << res[i][j].second << " ";
         }
         cout << endl;
     }
     cout << endl;
}

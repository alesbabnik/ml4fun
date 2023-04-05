#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_FEATURES 3 // Number of features in the data
#define NUM_POINTS 6 // Number of points in the training data

struct Point {
    float features[NUM_FEATURES];
    char label;
};

struct knn {
    int k;
    int num_points;
    struct Point* train_data;
};

float euclidean_distance(float* p, float* q) {
    float distance = 0.0;
    for (int i = 0; i < NUM_FEATURES; i++) {
        distance += pow(p[i] - q[i], 2);
    }
    return sqrt(distance);
}

struct knn* knn_create(int k) {
    struct knn* model = malloc(sizeof(struct knn));
    model->k = k;
    model->num_points = NUM_POINTS;
    model->train_data = malloc(NUM_POINTS * sizeof(struct Point));
    return model;
}

void knn_fit(struct knn* model, struct Point* train_data) {
    for (int i = 0; i < NUM_POINTS; i++) {
        model->train_data[i] = train_data[i];
    }
}

char knn_predict(struct knn* model, float* new_point) {
    float* distances = (float*) malloc(model->num_points * sizeof(float));

    for (int i = 0; i < model->num_points; i++) {
        distances[i] = euclidean_distance(model->train_data[i].features, new_point);
    }

    // Sort the distances and select the top k nearest neighbors
    for (int i = 0; i < model->num_points - 1; i++) {
        for (int j = i + 1; j < model->num_points; j++) {
            if (distances[j] < distances[i]) {
                float temp_dist = distances[i];
                distances[i] = distances[j];
                distances[j] = temp_dist;
                struct Point temp_point = model->train_data[i];
                model->train_data[i] = model->train_data[j];
                model->train_data[j] = temp_point;
            }
        }
    }
    free(distances);

    // Count the votes for each label among the k nearest neighbors
    char* votes = (char*) malloc(model->k * sizeof(char));
    for (int i = 0; i < model->k; i++) {
        votes[i] = model->train_data[i].label;
    }
    char result;
    int max_votes = 0;
    for (int i = 0; i < model->k; i++) {
        int count = 0;
        for (int j = 0; j < model->k; j++) {
            if (votes[j] == votes[i]) {
                count++;
            }
        }
        if (count > max_votes) {
            max_votes = count;
            result = votes[i];
        }
    }
    free(votes);
    return result;
}

int main() {
    struct Point train_data[NUM_POINTS] = {
        {{1.0, 2.0, 4.0, 3.0, 4.0}, '!'},
        {{4.0, 3.0, 4.0, 4.0, 2.0}, '!'},
        {{3.0, 1.0, 4.0, 3.0, 6.0}, '!'},
        {{6.0, 5.0, 8.0, 4.0, 3.0}, '?'},
        {{7.0, 7.0, 7.0, 6.0, 5.0}, '?'},
        {{8.0, 6.0, 7.0, 7.0, 4.0}, '?'}};
    
    struct knn* model = knn_create(3);
    knn_fit(model, train_data);

    float new_point[NUM_FEATURES] = {2.0, 3.0, 4.0, 5.0, 6.0};
    char result = knn_predict(model, new_point);
    printf("The new point belongs to class %c", result);

    return 0;
}

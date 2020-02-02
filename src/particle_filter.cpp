/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>
#include <cassert>
#include <unordered_map>

#include "helper_functions.h"

using namespace std;

const double EPSILON =  0.000001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    num_particles = 64;

    random_device rd;
    default_random_engine gen(rd());
    normal_distribution<double> N_x(x, std[0]);
    normal_distribution<double> N_y(y, std[1]);
    normal_distribution<double> N_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = N_x(gen);
        p.y = N_y(gen);
        p.theta = N_theta(gen);
        p.weight = 1.;
        particles.push_back(p);
        weights.push_back(1.);
        cout << "## init << " << p.x << " " << p.y << " " << p.theta << endl;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    random_device rd;
    default_random_engine gen(rd());
    for (auto &particle : particles) {
        double x = particle.x;
        double y = particle.y;
        double theta = particle.theta;

        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double delta_yaw = yaw_rate * delta_t;
        if (fabs(yaw_rate) < EPSILON) {
            double delta_velocity = velocity * delta_t;
            x += delta_velocity * cos_theta;
            y += delta_velocity * sin_theta;
        } else {
            double velocity_rate = velocity / yaw_rate;
            x += velocity_rate * (sin(theta + delta_yaw) - sin_theta);
            y += velocity_rate * (cos_theta - cos(theta + delta_yaw));
        }
        theta += delta_yaw;

        normal_distribution<double> N_x(x, std_pos[0]);
        normal_distribution<double> N_y(y, std_pos[1]);
        normal_distribution<double> N_theta(theta, std_pos[2]);

        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
    }

}

unordered_map<int, LandmarkObs> ParticleFilter::dataAssociation(const vector<LandmarkObs> &predicted,
                                                                vector<LandmarkObs> &observations) {
    unordered_map<int, LandmarkObs> predictions_per_id{};

    for (auto &observation: observations) {
        double min_dist = numeric_limits<double>::max();
        LandmarkObs closest{};
        for (auto &prediction: predicted) {
            double act_dist = dist(observation.x, observation.y, prediction.x, prediction.y);
            if (act_dist < min_dist) {
                min_dist = act_dist;
                closest = prediction;
            }
        }
        predictions_per_id[closest.id] = closest;
        observation.id = closest.id;
    }
    return predictions_per_id;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations, const Map &map_landmarks) {
    double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    double std_norm_x = 2 * pow(std_landmark[0], 2);
    double std_norm_y = 2 * pow(std_landmark[1], 2);

    weights.clear();

    for (auto &particle : particles) {
        double cos_theta = cos(particle.theta);
        double sin_theta = sin(particle.theta);

        // Observations in map coordinates (from particle's POV)
        vector<LandmarkObs> trans_observations;
        for (auto &obs: observations) {
            LandmarkObs trans_obs{};
            trans_obs.x = particle.x + obs.x * cos_theta - obs.y * sin_theta;
            trans_obs.y = particle.y + obs.x * sin_theta + obs.y * cos_theta;
            trans_observations.push_back(trans_obs);
        }

        // cycle through map_landmarks (real landmark positions) and filter out landmarks > sensor distance
        vector<LandmarkObs> predicted_landmarks;
        for (auto &map_landmark : map_landmarks.landmark_list) {
            double distance = dist(particle.x, particle.y, map_landmark.x_f, map_landmark.y_f);

            if (distance <= sensor_range) {
                LandmarkObs prediction{};
                prediction.id = map_landmark.id_i;
                prediction.x = map_landmark.x_f;
                prediction.y = map_landmark.y_f;
                predicted_landmarks.push_back(prediction);
            }
        }

        // use dataAssociation(predicted, observations) to match the closest real landmark to EACH observed landmark
        // run data association on observed landmarks and landmarks within sensor range
        unordered_map<int, LandmarkObs> predictions_per_id = dataAssociation(predicted_landmarks, trans_observations);

        double weight = 1.0;
        for (LandmarkObs &obs : trans_observations) {
            LandmarkObs predicted_landmark = predictions_per_id[obs.id];

            double delta_x = pow(obs.x - predicted_landmark.x, 2);
            double delta_y = pow(obs.y - predicted_landmark.y, 2);
            double exponent = (delta_x / std_norm_x) + (delta_y / std_norm_y);

            weight *= gauss_norm * exp(-exponent);
        }
        weights.push_back(weight);
        particle.weight = weight;
    }
}

void ParticleFilter::resample() {
    /**
     *   See
     *   https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     *   and
     *   https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution/discrete_distribution
     */
    random_device rd;
    default_random_engine gen(rd());
    discrete_distribution<int> distribution(weights.begin(), weights.end());

    vector<Particle> updated_particles = vector<Particle>(particles.size());
    vector<double> updated_weights = vector<double>(weights.size());
    for (size_t i = 0; i < particles.size(); i++) {
        int rand_idx = distribution(gen);
        updated_particles[i] = particles[rand_idx];
        updated_weights[i] = particles[rand_idx].weight;
    }

    particles = updated_particles;
    weights = updated_weights;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(const Particle &best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(const Particle &best, const string &coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
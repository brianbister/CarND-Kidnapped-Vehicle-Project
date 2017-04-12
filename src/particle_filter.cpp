/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <numeric>
#include <random>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).
  num_particles = 100;

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    particles.push_back(particle);
    weights.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  std::default_random_engine gen;

  for (Particle &particle : particles) {
    particle.x +=
        (velocity / yaw_rate) *
        (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
    particle.y +=
        (velocity / yaw_rate) *
        (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
    particle.theta += yaw_rate * delta_t;

    std::normal_distribution<double> dist_x(particle.x, std_pos[0]);
    std::normal_distribution<double> dist_y(particle.y, std_pos[1]);
    std::normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.
  for (LandmarkObs &observation : observations) {

    double closest_distance = DBL_MAX;

    for (const LandmarkObs &predict : predicted) {

      double dx = predict.x - observation.x;
      double dy = predict.y - observation.y;
      double distance = sqrt(dx * dx + dy * dy);

      if (distance < closest_distance) {

        closest_distance = distance;
        observation.id = predict.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems.
  //   Keep in mind that this transformation requires both rotation AND
  //   translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation
  //   3.33. Note that you'll need to switch the minus sign in that equation to
  //   a plus to account
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  weights.clear();

  // Transform the landmark list into LandmarkObs objects.
  std::vector<LandmarkObs> predicted;
  for (const auto &landmark : map_landmarks.landmark_list) {
    LandmarkObs predicted_landmark;
    predicted_landmark.id = landmark.id_i;
    predicted_landmark.x = landmark.x_f;
    predicted_landmark.y = landmark.y_f;
    predicted.push_back(predicted_landmark);
  }

  for (Particle &particle : particles) {

    // Transform our observations to the global space.
    std::vector<LandmarkObs> transformed_observations;
    for (const LandmarkObs &observation : observations) {
      LandmarkObs transformed_observation;
      transformed_observation.id = observation.id;
      transformed_observation.x = (observation.x * cos(particle.theta)) -
                                  (observation.y * sin(particle.theta)) +
                                  particle.x;
      transformed_observation.y = (observation.x * sin(particle.theta)) +
                                  (observation.y * cos(particle.theta)) +
                                  particle.y;
      transformed_observations.push_back(transformed_observation);
    }

    dataAssociation(predicted, transformed_observations);

    double weight = 1;
    for (LandmarkObs observation : transformed_observations) {
      // The landmarks are 1-indexed so we need to subtract 1 here.
      double dx = observation.x - predicted[observation.id - 1].x;
      double dy = observation.y - predicted[observation.id - 1].y;
      weight *= exp(-0.5 * ((dx * dx) / (std_x * std_x) +
                            (dy * dy) / (std_y * std_y))) /
                sqrt(2 * M_PI * std_x * std_y);
    }

    particle.weight = weight;
    weights.push_back(weight);
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::default_random_engine gen;
  std::discrete_distribution<> weight_distribution(weights.begin(),
                                                   weights.end());

  std::vector<Particle> resampled_particles;
  for (int i = 0; i < num_particles; ++i) {
    resampled_particles.push_back(particles[weight_distribution(gen)]);
  }
  particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);
  for (int i = 0; i < num_particles; ++i) {
    dataFile << particles[i].x << " " << particles[i].y << " "
             << particles[i].theta << "\n";
  }
  dataFile.close();
}

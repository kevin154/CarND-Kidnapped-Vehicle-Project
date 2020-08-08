/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * 
 * Updated: August 8th 2020 
 * Author: Kevin Smyth 
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    
  // Set the number of particles
  num_particles = 256;  
  
  // Engine for generating random variables
  default_random_engine gen;
   
  // Create normal distributions for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Sample from the normal distributions
  double sample_x = dist_x(gen);
  double sample_y = dist_y(gen);
  double sample_theta = dist_theta(gen);
  
  // Set particle vector
  particles = vector<Particle>(num_particles);
  
  // Set weights vector
  weights = std::vector<double>(num_particles, 1.0);
  
  for (int i=0; i < num_particles; ++i){    
      particles[i].id = i;
      particles[i].x = sample_x;
      particles[i].y = sample_y;
      particles[i].theta = sample_theta;
      particles[i].weight = 1.0;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
   
   default_random_engine gen;
   double x_old, y_old, theta_old, x_new, y_new, theta_new;
   
   // Calculate new particle positions
   for (int i=0; i < num_particles; ++i){
       // Shorthand variables to make equations easier to read
       x_old = particles[i].x;
       y_old = particles[i].y;
       theta_old = particles[i].theta;
     
       // Case where yaw rate is close to 0
       if (fabs(yaw_rate) < 0.00001){
           x_new = x_old + velocity * delta_t * cos(theta_old);
           y_new = y_old + velocity * delta_t * sin(theta_old);
           theta_new = theta_old;
       }
       // Case where yaw rate is not close to 0
       else {
           x_new = x_old + (velocity / yaw_rate) * (sin(theta_old + yaw_rate * delta_t) - sin(theta_old));
           y_new = y_old + (velocity / yaw_rate) * (cos(theta_old) - cos(theta_old + yaw_rate * delta_t));
           theta_new = theta_old + yaw_rate * delta_t;       
       }
       // Distributions of new particles centred around new positions
       normal_distribution<double> dist_x(x_new, std_pos[0]);
       normal_distribution<double> dist_y(y_new, std_pos[1]);
       normal_distribution<double> dist_theta(theta_new, std_pos[2]);
     
       // Update new positions with distribution noise
       particles[i].x = dist_x(gen);
       particles[i].y = dist_y(gen);
       particles[i].theta = dist_theta(gen);
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  
  // Variables to store the difference at each step and minimum difference found  
  double tmpDiff, minDiff;
  
  for (auto &observation : observations) {
    
    // Initialise differences to largest possible value
    minDiff = tmpDiff = std::numeric_limits<double>::infinity();
    
    for (auto const &prediction : predicted){
        // Calculate Euclidean distance between prediction and observation
        tmpDiff = dist(prediction.x, prediction.y, observation.x, observation.y);
         
        // If new minimum found update running minimum and ID
        if (tmpDiff < minDiff) {       
            minDiff = tmpDiff;
            observation.id = prediction.id;
        }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  
  // Vector of transformed observations
  vector<LandmarkObs> trans_observations(observations.size());
  
  // Vector of landmarks that fall within sensor range
  vector<LandmarkObs> landmarks_in_range;
  
  LandmarkObs observation, range_landmark;
  double theta;
 
  for (int i=0; i < num_particles; ++i){  
      // Variable to keep track of current particle in loop
      Particle &particle = particles[i];
      // Heading of current particle
      theta = particle.theta;
      
      // Transform each observation to map co-ordinates
      for (unsigned int j=0; j < observations.size(); ++j){
          observation = observations[j];
          trans_observations[j].x = particle.x + observation.x * cos(theta) - observation.y * sin(theta);
          trans_observations[j].y = particle.y + observation.x * sin(theta) + observation.y * cos(theta); 
      }

      // Clear landmarks in range
      landmarks_in_range.clear();
    
      // Populate the list of landmarks within sensor range
      for (auto const &landmark : map_landmarks.landmark_list){
          
          if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range){
            range_landmark.id = landmark.id_i;
            // Promote float values in map struct to double values in landmark struct 
            range_landmark.x = static_cast<double>(landmark.x_f);
            range_landmark.y = static_cast<double>(landmark.y_f);
            landmarks_in_range.push_back(range_landmark);
          } 
      }
      
      // Associate each transposed observation with a landmark 
      dataAssociation(landmarks_in_range, trans_observations);
      
      double landmark_x, landmark_y, observation_prob; 
      
      // Reset weight to 1
      particles[i].weight = 1.0;  
    
      // Calculate particle weight given transformed observations and 
      for (auto const &tobs : trans_observations){
          // Decrement landmark indices since they are 1-based
          landmark_x = static_cast<double>(map_landmarks.landmark_list[tobs.id - 1].x_f);
          landmark_y = static_cast<double>(map_landmarks.landmark_list[tobs.id - 1].y_f);
          observation_prob = multiv_gauss(std_landmark[0], std_landmark[1], tobs.x, tobs.y, landmark_x, landmark_y);
          particles[i].weight *= observation_prob;
      }
      // Update weights vector with current particle weight
      weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {

  vector<Particle> resampled_particles(num_particles);
  
  default_random_engine gen;
  // Generate indices with frequency dictated by weights
  std::discrete_distribution<int> distribution(weights.begin(), weights.end()); 
  
  // Generate new particle set by resampling from current set given current particle weights
  for (int i=0; i < num_particles; ++i){
      resampled_particles[i] = particles[distribution(gen)];
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
